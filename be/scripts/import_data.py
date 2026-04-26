import asyncio
import json
import os
import sys

# Đảm bảo Python nhận diện được module 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select
from app.core.database import AsyncSessionLocal
from app.models import User, Item, Review, Category

# CHÚ Ý: Sửa đường dẫn này nếu file của bạn nằm ở chỗ khác
METADATA_FILE = "item_metadata.jsonl"
REVIEW_FILE = "user_review.jsonl"

def safe_float(val):
    try: return float(val) if val is not None else None
    except: return None

def safe_int(val):
    try: return int(val) if val is not None else None
    except: return None

async def import_data():
    async with AsyncSessionLocal() as db:
        print("🚀 BẮT ĐẦU QUÁ TRÌNH IMPORT DỮ LIỆU AMAZON...")

        # ==========================================
        # 0. QUÉT VÀ IMPORT CATEGORIES (DANH MỤC)
        # ==========================================
        print("📁 0. Đang quét và xây dựng cây Danh mục (Categories)...")
        categories_set = set()
        
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                cats = data.get('categories', [])
                if isinstance(cats, list):
                    # Amazon thường trả về list các list [ ["Cat1", "Cat2"] ]
                    for cat_path in cats:
                        if isinstance(cat_path, list):
                            for cat_name in cat_path:
                                if cat_name: categories_set.add(cat_name.strip())
                        elif isinstance(cat_path, str):
                            categories_set.add(cat_path.strip())

        if categories_set:
            cat_data = [{"name": c} for c in categories_set]
            stmt_cats = insert(Category).values(cat_data).on_conflict_do_nothing(index_elements=['name'])
            await db.execute(stmt_cats)
            await db.commit()
            print(f"✅ Đã lưu {len(categories_set)} Danh mục (Categories) vào DB.")
        
        # ==========================================
        # 1. IMPORT ITEM METADATA
        # ==========================================
        print("📦 1. Đang đọc và lưu Item Metadata...")
        items_data = []
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                parent_asin = data.get('parent_asin') or data.get('asin')
                if not parent_asin: continue

                items_data.append({
                    "parent_asin": parent_asin,
                    "title": data.get('title'),
                    "main_category": data.get('main_category'),
                    "price": safe_float(data.get('price')),
                    "average_rating": safe_float(data.get('average_rating')),
                    "rating_number": safe_int(data.get('rating_number')),
                    "store": data.get('store'),
                    
                    "features": data.get('features', []),
                    "description": data.get('description', []),
                    "images": data.get('images', []),
                    "videos": data.get('videos', []),
                    "categories": data.get('categories', []), 
                    "details": data.get('details', {}),
                    "bought_together": data.get('bought_together', [])
                })

        stmt_items = insert(Item).values(items_data).on_conflict_do_nothing(index_elements=['parent_asin'])
        await db.execute(stmt_items)
        await db.commit()
        print(f"✅ Đã lưu {len(items_data)} Items vào DB.")

        # ==========================================
        # 2. IMPORT USERS (Trích từ file Review)
        # ==========================================
        print("👤 2. Đang quét danh sách Users từ Reviews...")
        users_dict = {} 
        reviews_raw = []
        
        with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                review = json.loads(line)
                u_id = review.get('user_id')
                
                if u_id and u_id not in users_dict:
                    users_dict[u_id] = {
                        "original_user_id": u_id,
                        "password_hash": None 
                    }
                reviews_raw.append(review) 

        stmt_users = insert(User).values(list(users_dict.values())).on_conflict_do_nothing(index_elements=['original_user_id'])
        await db.execute(stmt_users)
        await db.commit()
        print(f"✅ Đã lọc và lưu {len(users_dict)} Users vào DB.")

        # ==========================================
        # 3. MAP KHÓA NGOẠI VÀ IMPORT REVIEWS
        # ==========================================
        print("🔄 3. Đang Map Khóa Ngoại cho Reviews...")
        
        db_items = await db.execute(select(Item.parent_asin, Item.id))
        item_map = {row.parent_asin: row.id for row in db_items}

        db_users = await db.execute(select(User.original_user_id, User.id))
        user_map = {row.original_user_id: row.id for row in db_users}

        reviews_data = []
        for r in reviews_raw:
            item_asin = r.get('parent_asin')
            u_id = r.get('user_id')

            if item_asin in item_map and u_id in user_map:
                reviews_data.append({
                    "item_id": item_map[item_asin], 
                    "user_id": user_map[u_id],      
                    "rating": safe_float(r.get('rating')),
                    "title": r.get('title', ''),
                    "text": r.get('text', ''),
                    "images": r.get('images', []), 
                    "review_timestamp": safe_int(r.get('timestamp')),
                    "verified_purchase": r.get('verified_purchase', False),
                    "helpful_vote": safe_int(r.get('helpful_vote')) or 0
                })

        print(f"⭐ Đang lưu {len(reviews_data)} Reviews hợp lệ vào DB (chia nhỏ để tránh tràn RAM)...")
        chunk_size = 10000 
        for i in range(0, len(reviews_data), chunk_size):
            chunk = reviews_data[i:i+chunk_size]
            await db.execute(insert(Review).values(chunk))
        
        await db.commit()
        print("🎉 QUÁ TRÌNH IMPORT HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    asyncio.run(import_data())