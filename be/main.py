import torch
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from torchvision import transforms
from sklearn.preprocessing import minmax_scale

# Import kiến trúc (đảm bảo file models.py đã có đủ class)
from models import TextEncoder, ImageEncoder, FusionGate, D_LATENT, device

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==============================================================================
# 1. LOAD DATA & EMBEDDINGS (Thay thế cho việc load df và chạy inference lại)
# ==============================================================================
print("🚀 Đang tải dữ liệu vào RAM...")

def load_pt(path):
    # Thêm weights_only=False để tránh lỗi PyTorch mới
    full_path = os.path.join("weights", path)
    if os.path.exists(full_path):
        return torch.load(full_path, map_location=device, weights_only=False)
    return None

# --- A. Load Tensors (Embeddings đã tính sẵn) ---
ITEM_EMBS_GPU = load_pt("item_embeddings.pt") # [N_Items, 256]
USER_EMBS_GPU = load_pt("user_embeddings.pt") # [N_Users, 256]

if ITEM_EMBS_GPU is not None:
    ITEM_EMBS_GPU = ITEM_EMBS_GPU.to(device)
if USER_EMBS_GPU is not None:
    USER_EMBS_GPU = USER_EMBS_GPU.to(device)

# --- B. Load Metadata & Maps (Thay thế Dataframe) ---
USER_MAP  = load_pt("user_map.pt")        # Dict: {'User123': 0, 'User456': 1...}
USER_DB   = load_pt("users_db.pt") or {}  # Dict: Lưu lịch sử user để check Cold Start
ITEM_META = load_pt("item_metadata.pt")   # List: Thay thế cho 'df' để lấy title, rating
STATS     = load_pt("stats.pt") or {'C': 3.0, 'm': 10.0} # C và m

NUM_ITEMS = ITEM_EMBS_GPU.shape[0] if ITEM_EMBS_GPU is not None else 0
print(f"✅ Đã load: {NUM_ITEMS} items, {len(USER_MAP) if USER_MAP else 0} users.")

# ==============================================================================
# 2. LOAD MODEL ENCODERS (Để xử lý Query Text/Image mới)
# ==============================================================================
text_encoder = TextEncoder().to(device)
img_encoder = ImageEncoder().to(device)
fusion_gate = FusionGate(D_LATENT).to(device)

def load_weights(model, filename):
    path = os.path.join("weights", filename)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=False)
            print(f"   ✅ Loaded {filename}")
        except Exception as e:
            print(f"   ⚠️ Lỗi load {filename}: {e}")

load_weights(text_encoder, "text_encoder.pth")
load_weights(img_encoder, "img_encoder.pth")
load_weights(fusion_gate, "fusion_gate.pth")

text_encoder.eval()
img_encoder.eval()
fusion_gate.eval()

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
# Hàm xử lý ảnh
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img_transform(image).unsqueeze(0).to(device)
    except: return None

# Hàm tính Weighted Rating (Giống hệt notebook)
def calculate_weighted_rating(v, R, m, C):
    # v: vote count, R: average rating
    return (v / (v + m) * R) + (m / (v + m) * C)

# ==============================================================================
# 4. API SEARCH (LOGIC GIỮ NGUYÊN CẤU TRÚC CỦA BẠN)
# ==============================================================================
@app.post("/search")
async def run_full_query_api(
    query_text: str = Form(""),
    user_id: str = Form(...),
    image: UploadFile = File(None),
    # Các tham số Hyperparams
    k_retrieval: int = Form(100),
    k_rerank: int = Form(20),
    alpha: float = Form(0.5), # ALPHA_INTENT
    beta: float = Form(0.5)   # BETA_PERSONALIZATION
):
    print("\n" + "="*50)
    print(f"Bắt đầu query: '{query_text}' - User: {user_id}")
    
    is_new_user = False 
    user_rerank_emb = None

    # --- 0. Kiểm tra User (Logic giống hệt) ---
    # Kiểm tra trong User Map (có tồn tại trong file embeddings không)
    if USER_MAP is None or user_id not in USER_MAP:
        is_new_user = True
    else:
        user_index = USER_MAP[user_id]
        
        # Kiểm tra lịch sử (để xem có đủ 2 item không)
        # Thay vì user_df.iloc, ta tra cứu dictionary USER_DB
        user_history = USER_DB.get(user_id, {}).get('indices', [])
        
        if len(user_history) < 2:
            is_new_user = True # Vẫn là "Cold Start"
        else:
            is_new_user = False
            # Lấy embedding từ Tensor đã load vào RAM
            user_rerank_emb = USER_EMBS_GPU[user_index].unsqueeze(0) # [1, 256]

    if is_new_user:
        print(f" -> Chế độ 'Cold Start' (Gợi ý Phổ biến).")
    else:
        print(f" -> Cá nhân hóa cho User: {user_id}")

    # --- BẮT ĐẦU XỬ LÝ ---
    with torch.no_grad():
        # 1. RETRIEVAL (Mã hóa Query)
        # Encode Text
        if query_text.strip():
            q_text_emb = text_encoder([query_text])
            has_text = True
        else:
            q_text_emb = torch.zeros(1, D_LATENT).to(device)
            has_text = False
            
        # Encode Image (Nếu có)
        q_img_emb = torch.zeros(1, D_LATENT).to(device)
        has_image = False
        if image:
            content = await image.read()
            img_tensor = process_image(content)
            if img_tensor is not None:
                q_img_emb = img_encoder(img_tensor)
                has_image = True
        
        # Tạo Mask và Dummy Tabular
        q_tab_emb = torch.zeros_like(q_text_emb).to(device)
        # Mask: [Text, Image, Tabular]
        query_mask = torch.tensor([[has_text, has_image, False]], dtype=torch.bool).to(device)
        
        # Chạy Fusion Gate
        q_fused_emb, _ = fusion_gate(
            q_text_emb, 
            q_img_emb, 
            q_tab_emb, 
            mask=query_mask
        )
        
        # Dot Product với toàn bộ Item Embeddings
        retrieval_scores_all = torch.matmul(q_fused_emb, ITEM_EMBS_GPU.T) # [1, N_Items]
        
        # Lấy Top K Retrieval
        k_retrieval_real = min(k_retrieval, NUM_ITEMS)
        top_k_retrieval_scores, top_k_indices = torch.topk(
            retrieval_scores_all.squeeze(0), 
            k=k_retrieval_real
        ) 
        print(f"Đã tìm thấy {k_retrieval_real} ứng viên.")

        # 2. RERANKING
        # Lấy indices về CPU để truy xuất metadata
        candidate_indices_cpu = top_k_indices.cpu().numpy()
        
        if is_new_user:
            # Logic Popularity (Weighted Rating)
            pop_scores = []
            for idx in candidate_indices_cpu:
                # Lấy thông tin từ ITEM_META thay vì df.iloc
                item_info = ITEM_META[idx]
                # Ép kiểu về float để tránh lỗi
                v = float(item_info.get('rating_number', 0) or 0)
                R = float(item_info.get('average_rating', 0) or 0)
                wr = calculate_weighted_rating(v, R, STATS['m'], STATS['C'])
                pop_scores.append(wr)
            
            # MinMax Scale
            if len(pop_scores) > 1:
                # Chuyển sang numpy 2D để scale rồi chuyển về tensor
                pop_scores_np = np.array(pop_scores).reshape(-1, 1)
                scaled_scores = minmax_scale(pop_scores_np).flatten()
                rerank_scores = torch.tensor(scaled_scores, device=device, dtype=torch.float)
            else:
                rerank_scores = torch.tensor([1.0] * len(pop_scores), device=device, dtype=torch.float)
        else:
            # Logic Personalization (User Embedding)
            candidate_item_embs = ITEM_EMBS_GPU[top_k_indices]
            rerank_scores = torch.matmul(user_rerank_emb, candidate_item_embs.T).squeeze(0)
            rerank_scores = rerank_scores.squeeze(0) 
        
        # 3. FINAL COMBINED SCORE
        # Lưu ý: top_k_retrieval_scores cần chuẩn hóa nếu biên độ khác rerank_scores
        # Nhưng để giữ nguyên cấu trúc của bạn, ta cộng trực tiếp:
        final_combined_score = (
            (alpha * top_k_retrieval_scores) + 
            (beta * rerank_scores)
        )
        
        # Lấy Top K Final
        k_rerank_real = min(k_rerank, len(final_combined_score))
        final_scores, final_relative_indices = torch.topk(
            final_combined_score, 
            k=k_rerank_real
        )
        
        # Map ngược lại ra Index gốc của sản phẩm
        final_item_indices = top_k_indices[final_relative_indices]
        
        print("--- Hoàn tất Reranking ---")

    # 4. FORMAT KẾT QUẢ TRẢ VỀ
    results = []
    final_indices_cpu = final_item_indices.cpu().numpy()
    final_scores_cpu = final_scores.cpu().numpy()
    
    for i, idx in enumerate(final_indices_cpu):
        item = ITEM_META[idx]
        
        # Xử lý ảnh để hiển thị đẹp
        img_url = "https://placehold.co/300"
        if item.get("images"):
            imgs = item["images"]
            if isinstance(imgs, list) and len(imgs) > 0:
                first = imgs[0]
                if isinstance(first, dict):
                    img_url = first.get("large") or first.get("hi_res") or first.get("thumb")
                elif isinstance(first, str):
                    img_url = first

        results.append({
            "id": int(idx),
            "title": item.get("title", "No Title"),
            "image": img_url,
            "score": float(final_scores_cpu[i]),
            "average_rating": item.get("average_rating"),
            "type": "Popular" if is_new_user else "Personalized"
        })

    return {"results": results}
@app.get("/history/{user_id}")
async def get_user_history(user_id: str):
    # Kiểm tra xem User có trong DB không
    if user_id not in USER_DB:
        return {"history": []}
    
    # Lấy danh sách ID sản phẩm đã mua
    item_indices = USER_DB[user_id].get('indices', [])
    
    history_items = []
    for idx in item_indices:
        # Map từ ID sang thông tin chi tiết (ITEM_METADATA)
        if idx < len(ITEM_META):
            meta = ITEM_META[idx]
            
            # Xử lý link ảnh
            img_url = "https://placehold.co/100"
            if meta.get("images"):
                imgs = meta["images"]
                if isinstance(imgs, list) and len(imgs) > 0:
                    first = imgs[0]
                    if isinstance(first, dict):
                        img_url = first.get("thumb") or first.get("large")
                    elif isinstance(first, str):
                        img_url = first
            
            history_items.append({
                "id": int(idx),
                "title": meta.get("title", "No Title"),
                "image": img_url
            })
            
    # Trả về danh sách (đảo ngược để hiện cái mới nhất lên đầu)
    return {"history": history_items[::-1]}