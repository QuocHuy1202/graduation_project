import torch
import numpy as np
from fastapi import UploadFile
from sklearn.preprocessing import minmax_scale
from models import D_LATENT, device
from app.utils.image_utils import process_image
from app.utils.math_utils import calculate_weighted_rating

async def perform_search(
    query_text: str, user_id: str, image: UploadFile, 
    k_retrieval: int, k_rerank: int, alpha: float, beta: float, ml
):
    """Thực thi luồng Retrieval và Reranking"""
    is_new_user = False 
    user_rerank_emb = None

    # --- 1. Kiểm tra User ---
    if ml.USER_MAP is None or user_id not in ml.USER_MAP:
        is_new_user = True
    else:
        user_index = ml.USER_MAP[user_id]
        user_history = ml.USER_DB.get(user_id, {}).get('indices', [])
        
        if len(user_history) < 2:
            is_new_user = True
        else:
            user_rerank_emb = ml.USER_EMBS_GPU[user_index].unsqueeze(0)

    # --- 2. Bắt đầu AI Pipeline ---
    with torch.no_grad():
        # Encode Text
        if query_text.strip():
            q_text_emb = ml.text_encoder([query_text])
            has_text = True
        else:
            q_text_emb = torch.zeros(1, D_LATENT).to(device)
            has_text = False
            
        # Encode Image
        q_img_emb = torch.zeros(1, D_LATENT).to(device)
        has_image = False
        if image:
            content = await image.read()
            img_tensor = process_image(content)
            if img_tensor is not None:
                q_img_emb = ml.img_encoder(img_tensor)
                has_image = True
        
        # Fusion
        q_tab_emb = torch.zeros_like(q_text_emb).to(device)
        query_mask = torch.tensor([[has_text, has_image, False]], dtype=torch.bool).to(device)
        q_fused_emb, _ = ml.fusion_gate(q_text_emb, q_img_emb, q_tab_emb, mask=query_mask)
        
        # Retrieval
        retrieval_scores_all = torch.matmul(q_fused_emb, ml.ITEM_EMBS_GPU.T)
        k_retrieval_real = min(k_retrieval, ml.NUM_ITEMS)
        top_k_retrieval_scores, top_k_indices = torch.topk(
            retrieval_scores_all.squeeze(0), k=k_retrieval_real
        ) 
        
        # Reranking
        candidate_indices_cpu = top_k_indices.cpu().numpy()
        
        if is_new_user:
            pop_scores = []
            for idx in candidate_indices_cpu:
                item_info = ml.ITEM_META[idx]
                v = float(item_info.get('rating_number', 0) or 0)
                R = float(item_info.get('average_rating', 0) or 0)
                wr = calculate_weighted_rating(v, R, ml.STATS['m'], ml.STATS['C'])
                pop_scores.append(wr)
            
            if len(pop_scores) > 1:
                pop_scores_np = np.array(pop_scores).reshape(-1, 1)
                scaled_scores = minmax_scale(pop_scores_np).flatten()
                rerank_scores = torch.tensor(scaled_scores, device=device, dtype=torch.float)
            else:
                rerank_scores = torch.tensor([1.0] * len(pop_scores), device=device, dtype=torch.float)
        else:
            candidate_item_embs = ml.ITEM_EMBS_GPU[top_k_indices]
            rerank_scores = torch.matmul(user_rerank_emb, candidate_item_embs.T).squeeze(0)
            rerank_scores = rerank_scores.squeeze(0) 
        
        # Final Combine
        final_combined_score = (alpha * top_k_retrieval_scores) + (beta * rerank_scores)
        k_rerank_real = min(k_rerank, len(final_combined_score))
        final_scores, final_relative_indices = torch.topk(final_combined_score, k=k_rerank_real)
        final_item_indices = top_k_indices[final_relative_indices]
        
    # --- 3. Format Kết quả ---
    results = []
    final_indices_cpu = final_item_indices.cpu().numpy()
    final_scores_cpu = final_scores.cpu().numpy()
    
    for i, idx in enumerate(final_indices_cpu):
        item = ml.ITEM_META[idx]
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
        
    return results, final_indices_cpu