import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.ml_manager import ml, device 

router = APIRouter()

class PurchaseRequest(BaseModel):
    user_id: str  
    item_idx: int 
    rating: float = 5.0  # Mặc định hành vi Mua Hàng tương đương đánh giá 5 sao

@router.post("/buy")
async def process_purchase(data: PurchaseRequest):
    # 1. Lấy vị trí (index) của User trong Tensor thông qua USER_MAP
    if data.user_id not in ml.USER_MAP:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại trong hệ thống bộ nhớ")
    
    user_idx = ml.USER_MAP[data.user_id]

    # 2. Kiểm tra tính hợp lệ của Sản phẩm
    if data.item_idx < 0 or data.item_idx >= ml.NUM_ITEMS:
        raise HTTPException(status_code=400, detail="ID Sản phẩm vượt quá giới hạn")

    # 3. TÍNH TOÁN VÀ CẬP NHẬT TRỰC TIẾP EMBEDDING (RAM/VRAM)
    with torch.no_grad():
        user_emb = ml.USER_EMBS_GPU[user_idx]
        item_emb = ml.ITEM_EMBS_GPU[data.item_idx]

        sim_before = F.cosine_similarity(user_emb.unsqueeze(0), item_emb.unsqueeze(0)).item()
        print(f"\n--- LOG KIỂM TRA AI ---")
        print(f"1. Độ tương đồng TRƯỚC khi mua: {sim_before:.4f}")

        ALPHA = 0.85 
        LEARNING_RATE = 1.0 - ALPHA 

        new_user_emb = (ALPHA * user_emb) + (LEARNING_RATE * item_emb)
        new_user_emb = F.normalize(new_user_emb, p=2, dim=-1)

        ml.USER_EMBS_GPU[user_idx] = new_user_emb

        sim_after = F.cosine_similarity(new_user_emb.unsqueeze(0), item_emb.unsqueeze(0)).item()
        print(f"2. Độ tương đồng SAU khi mua:   {sim_after:.4f}")
        print(f"👉 Mức độ thay đổi: {sim_after - sim_before:+.4f}")
        print(f"-----------------------\n")

    # ==========================================
    # 4. CẬP NHẬT LỊCH SỬ (USER_DB) CHO GEMINI RAG
    # ==========================================
    # Nếu user chưa từng có lịch sử trong DB, khởi tạo dict rỗng
    if data.user_id not in ml.USER_DB:
        ml.USER_DB[data.user_id] = {'indices': [], 'ratings': []}
        
    # Thêm sản phẩm vừa mua và rating vào cuối mảng lịch sử
    ml.USER_DB[data.user_id]['indices'].append(data.item_idx)
    ml.USER_DB[data.user_id]['ratings'].append(data.rating)
    
    print(f"📚 Đã cập nhật Lịch sử RAG: User {data.user_id} -> Item {data.item_idx} ({data.rating} sao)")

    return {
        "status": "success",
        "action": "Online_Update_Positive",
        "message": f"Sở thích và lịch sử của User {data.user_id} đã được cập nhật với sản phẩm {data.item_idx}."
    }