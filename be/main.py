import torch
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from PIL import Image
import io
from torchvision import transforms
from sklearn.preprocessing import minmax_scale

# Import kiến trúc
from models import UserTower, TextEncoder, ImageEncoder, ReviewTabularEncoder, GatingFusionLayer, FusionGate, D_LATENT, device
from utils_data import prepare_user_batch, calculate_weighted_rating

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 1. LOAD DATA & MODELS
# ==========================================
print("🔧 Đang khởi động Backend Multimodal...")

# Load Database
if not os.path.exists("weights/items_db.pt"): raise Exception("Chưa chạy setup.py!")
item_db = torch.load("weights/items_db.pt", map_location=device, weights_only=False)
ITEM_VECTORS = item_db["vectors"].to(device)
ITEM_METADATA = item_db["metadata"]
STATS = item_db.get("stats", {"C": 3.0, "m": 10.0}) 
# Thêm weights_only=False
USER_DB = torch.load("weights/users_db.pt", weights_only=False) if os.path.exists("weights/users_db.pt") else {}
NUM_ITEMS = ITEM_VECTORS.shape[0]

# --- KHỞI TẠO MODELS ---
# 1. Encoders cho Query
text_encoder = TextEncoder().to(device)
img_encoder = ImageEncoder().to(device)       # <--- MỚI: Dùng để encode ảnh user upload
query_fusion_gate = FusionGate(D_LATENT).to(device) # <--- MỚI: Dùng để trộn Text + Ảnh query

# 2. Models cho User Tower (Reranking)
tab_encoder = ReviewTabularEncoder().to(device)
user_fusion_layer = GatingFusionLayer(D_LATENT, D_LATENT, 32).to(device)
dummy_items = torch.zeros(NUM_ITEMS, D_LATENT).to(device)
user_tower = UserTower(dummy_items, text_encoder, tab_encoder, user_fusion_layer).to(device)

# --- LOAD TRỌNG SỐ ---
def load_w(model, path):
    if os.path.exists(path): 
        try: model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=False)
        except: pass

load_w(text_encoder, "weights/text_encoder.pth")
load_w(img_encoder, "weights/img_encoder.pth")      # <--- Load trọng số ảnh
load_w(query_fusion_gate, "weights/fusion_gate.pth") # <--- Load trọng số Fusion
load_w(tab_encoder, "weights/tab_encoder.pth")

if os.path.exists("weights/user_tower.pth"):
    state = torch.load("weights/user_tower.pth", map_location=device)
    # Resize nếu lệch size
    if state['item_embedding_lookup.weight'].shape[0] != NUM_ITEMS:
        user_tower.item_embedding_lookup = torch.nn.Embedding(state['item_embedding_lookup.weight'].shape[0], D_LATENT).to(device)
    user_tower.load_state_dict(state, strict=False)
    user_tower.item_embedding_lookup = torch.nn.Embedding.from_pretrained(ITEM_VECTORS, freeze=True)

# Chuyển sang chế độ Eval
text_encoder.eval()
img_encoder.eval()
query_fusion_gate.eval()
user_tower.eval()

# --- HÀM XỬ LÝ ẢNH ---
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = img_transform(image).unsqueeze(0).to(device) # [1, 3, 224, 224]
        return tensor
    except: return None

# ==========================================
# 2. API SEARCH ĐA PHƯƠNG THỨC
# ==========================================

@app.post("/search")
async def search(
    query_text: str = Form(""),           # Nhận Text từ Form Data
    user_id: str = Form(...),             # Nhận UserID bắt buộc
    image: UploadFile = File(None),       # Nhận File ảnh (Optional)
    history_items: str = Form(None)       # Nhận lịch sử dạng chuỗi JSON (Optional)
):
    # Parse history nếu có (Vì gửi qua Form-data nên nó là string)
    # Ở đây mình làm đơn giản: giả định client gửi user_id để tra cứu DB thôi
    
    K_RETRIEVAL = 100
    K_RERANK = 20
    ALPHA = 0.5 
    BETA = 0.5 

    # --- BƯỚC 1: XỬ LÝ QUERY (TEXT + IMAGE) ---
    with torch.no_grad():
        # 1. Encode Text
        if query_text.strip():
            q_text_emb = text_encoder([query_text]) 
            has_text = True
        else:
            q_text_emb = torch.zeros(1, D_LATENT).to(device)
            has_text = False
            
        # 2. Encode Image (Nếu có upload)
        if image:
            content = await image.read()
            img_tensor = process_image(content)
            if img_tensor is not None:
                q_img_emb = img_encoder(img_tensor)
                has_image = True
            else:
                q_img_emb = torch.zeros(1, D_LATENT).to(device)
                has_image = False
        else:
            q_img_emb = torch.zeros(1, D_LATENT).to(device)
            has_image = False
            
        # 3. Fusion Gate
        # Mask logic: [Text, Image, Tabular]
        # Tabular query luôn là False vì user không nhập thông số kỹ thuật
        mask = torch.tensor([[has_text, has_image, False]], dtype=torch.bool).to(device)
        
        # Nếu người dùng không nhập gì cả -> Lỗi hoặc trả về ngẫu nhiên
        if not has_text and not has_image:
             return {"error": "Vui lòng nhập text hoặc tải ảnh lên"}

        q_tab_dummy = torch.zeros_like(q_text_emb)
        
        # Trộn vector!
        q_fused, _ = query_fusion_gate(q_text_emb, q_img_emb, q_tab_dummy, mask=mask)
        
        # 4. Retrieval (Tìm kiếm thô)
        retrieval_scores = torch.matmul(q_fused, ITEM_VECTORS.t()).squeeze(0)
        top_scores, top_indices = torch.topk(retrieval_scores, k=min(K_RETRIEVAL, NUM_ITEMS))

    candidate_indices = top_indices.cpu().numpy()
    candidate_scores = top_scores.cpu().numpy()

    # --- BƯỚC 2: PERSONALIZATION (RERANK) ---
    # Kiểm tra lịch sử User
    h_items = []
    if user_id in USER_DB:
        h_items = USER_DB[user_id].get('indices', [])
        h_reviews = USER_DB[user_id].get('reviews', [])
        h_ratings = USER_DB[user_id].get('ratings', [])
    
    rerank_scores = []
    is_cold_start = True

    if len(h_items) >= 2:
        is_cold_start = False
        # Tạo batch User Tower
        dummy_votes = [0] * len(h_items)
        dummy_verified = [1] * len(h_items)
        batch = prepare_user_batch(h_items, h_reviews, h_ratings, dummy_votes, dummy_verified)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            
        with torch.no_grad():
            user_vec = user_tower(batch)
            candidate_vectors = ITEM_VECTORS[top_indices]
            rerank_scores = torch.matmul(user_vec, candidate_vectors.t()).squeeze(0).cpu().numpy()
    else:
        # Cold Start: Dùng Weighted Rating
        stats = []
        for idx in candidate_indices:
            meta = ITEM_METADATA[idx]
            v = float(meta.get('rating_number', 0) or 0)
            R = float(meta.get('average_rating', 0) or 0)
            stats.append(calculate_weighted_rating(v, R, STATS['m'], STATS['C']))
        rerank_scores = minmax_scale(stats) if len(stats) > 0 else np.zeros(len(stats))

    # --- BƯỚC 3: TỔNG HỢP ---
    norm_retrieval = minmax_scale(candidate_scores) if len(candidate_scores) > 0 else candidate_scores
    final_scores = (ALPHA * norm_retrieval) + (BETA * rerank_scores)
    
    sorted_local_idx = np.argsort(final_scores)[::-1][:K_RERANK]
    
    results = []
    for i in sorted_local_idx:
        g_idx = candidate_indices[i]
        meta = ITEM_METADATA[g_idx]
        
        img_url = "https://placehold.co/300"
        if meta.get("images"): 
            img_url = meta["images"][0].get("large") or meta["images"][0].get("hi_res")

        results.append({
            "id": int(g_idx),
            "title": meta.get("title", "Sản phẩm"),
            "image": img_url,
            "score": float(final_scores[i]),
            "match_type": "Image + Text" if (has_image and has_text) else ("Image Only" if has_image else "Text Only")
        })

    return {"results": results}