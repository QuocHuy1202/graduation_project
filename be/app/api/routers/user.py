from fastapi import APIRouter
from app.core.ml_manager import ml

router = APIRouter()

@router.get("/history/{user_id}")
async def get_user_history(user_id: str):
    # Dùng biến ml thay vì biến toàn cục
    if user_id not in ml.USER_DB:
        return {"history": []}
    
    item_indices = ml.USER_DB[user_id].get('indices', [])
    history_items = []
    
    for idx in item_indices:
        if idx < len(ml.ITEM_META):
            meta = ml.ITEM_META[idx]
            
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
            
    return {"history": history_items[::-1]}