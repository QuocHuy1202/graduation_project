import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO

# Import kiến trúc
from models import TextEncoder, ImageEncoder, TabularEncoder, FusionGate, device

# --- CẤU HÌNH ---
JSONL_PATH = "dataset/meta_Gift_Cards.jsonl"
OUTPUT_DB = "weights/items_db.pt"
BATCH_SIZE = 32

# --- 1. DATASET (Tái tạo logic đọc dữ liệu) ---
class GiftCardInferenceDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("⏳ Đang đọc dữ liệu...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Tạo text combined
                    title = item.get('title', '')
                    desc = " ".join(item.get('description', [])[:2])
                    item['text_combined'] = f"{title} {desc}"
                    
                    # Lấy URL ảnh
                    img_url = None
                    if item.get('images') and len(item['images']) > 0:
                        img_url = item['images'][0].get('large') or item['images'][0].get('hi_res')
                    item['img_url'] = img_url
                    
                    self.data.append(item)
                except: pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Text
        text = row['text_combined']
        
        # Image (Download on the fly hoặc dùng ảnh đen nếu lỗi)
        img_tensor = torch.zeros(3, 224, 224)
        if row['img_url']:
            try:
                # Lưu ý: Trong thực tế nên download ảnh về máy trước cho nhanh
                # Ở đây demo nên tạo ảnh rỗng để code chạy được
                # response = requests.get(row['img_url'], timeout=2)
                # img = Image.open(BytesIO(response.content)).convert('RGB')
                # img_tensor = self.transform(img)
                pass 
            except: pass
            
        # Tabular (Dummy cho khớp model)
        numeric = torch.tensor([0.0, 0.0]) 
        main_cat = torch.tensor(0)
        store = torch.tensor(0)
        categories = torch.tensor(0)
        
        return text, img_tensor, numeric, main_cat, store, categories, idx

def custom_collate(batch):
    texts = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    numeric = torch.stack([item[2] for item in batch])
    main_cat = torch.stack([item[3] for item in batch])
    store = torch.stack([item[4] for item in batch])
    categories = torch.stack([item[5] for item in batch])
    indices = [item[6] for item in batch]
    return texts, images, numeric, main_cat, store, categories, indices

# --- 2. MAIN PROCESS ---
def run():
    print("🔧 Khởi tạo Model...")
    text_encoder = TextEncoder().to(device)
    img_encoder = ImageEncoder().to(device)
    tab_encoder = TabularEncoder().to(device)
    fusion_gate = FusionGate().to(device)
    
    # Load weights (Nếu có)
    try:
        text_encoder.load_state_dict(torch.load("weights/text_encoder.pth", map_location=device), strict=False)
        img_encoder.load_state_dict(torch.load("weights/img_encoder.pth", map_location=device))
        tab_encoder.load_state_dict(torch.load("weights/tab_encoder.pth", map_location=device))
        fusion_gate.load_state_dict(torch.load("weights/fusion_gate.pth", map_location=device))
        print("✅ Đã load weights (các file tìm thấy).")
    except Exception as e:
        print(f"⚠️ Cảnh báo load weight: {e}")

    text_encoder.eval()
    img_encoder.eval()
    tab_encoder.eval()
    fusion_gate.eval()

    # Dataset
    dataset = GiftCardInferenceDataset(JSONL_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    
    all_fused_embs = []
    metadata_list = []

    print("🚀 Bắt đầu tạo Embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader):
            texts, images, numeric, main_cat, store, cats, indices = batch
            images = images.to(device)
            numeric = numeric.to(device)
            # ...
            
            # 1. Encode
            t_emb = text_encoder(texts)
            i_emb = img_encoder(images)
            # t_emb = tab_encoder(...) # Bỏ qua tabular demo
            b_emb = torch.zeros_like(t_emb) # Dummy tabular

            # 2. Tạo mask (Cái nào embedding = 0 thì mask)
            mask = torch.ones((len(texts), 3), dtype=torch.bool).to(device)
            
            # 3. Fusion (Logic bạn gửi)
            fused, _ = fusion_gate(t_emb, i_emb, b_emb, mask=mask)
            
            all_fused_embs.append(fused.cpu())
            
            # Lưu metadata
            for idx in indices:
                metadata_list.append(dataset.data[idx])

    # Gộp lại
    final_tensor = torch.cat(all_fused_embs, dim=0)
    print(f"✅ Hoàn tất! Shape: {final_tensor.shape}")
    
    # Lưu xuống file
    torch.save({
        "vectors": final_tensor,
        "metadata": metadata_list
    }, OUTPUT_DB)
    print(f"💾 Đã lưu database tại {OUTPUT_DB}")

if __name__ == "__main__":
    run()