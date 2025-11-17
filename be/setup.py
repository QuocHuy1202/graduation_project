import torch
import json
import pandas as pd
from models import TextEncoder, FusionGate, device

# FILE PATHS
JSONL_PATH = "dataset/meta_Gift_Cards.jsonl"
OUTPUT_ITEMS_DB = "weights/items_db.pt"
OUTPUT_USER_DB = "weights/users_db.pt" # Tạo giả lập vì bạn chưa gửi file reviews

def setup():
    print("⏳ Đang khởi tạo Item Database...")
    
    # 1. Load Models (Để tạo vector item nếu chưa có)
    # Ở đây mình giả định bạn ĐÃ chạy generate_embeddings.py ở bước trước 
    # và có items_db.pt rồi. Nếu chưa, hãy chạy generate_embeddings.py trước.
    # Tuy nhiên, mình cần CẬP NHẬT thêm thông tin Rating vào DB đó.
    
    try:
        db = torch.load(OUTPUT_ITEMS_DB, map_location=device)
        item_vectors = db["vectors"]
        metadata = db["metadata"]
        print("   -> Đã load vector items cũ.")
    except:
        print("❌ Lỗi: Hãy chạy file generate_embeddings.py (ở câu trả lời trước) để tạo vector item trước!")
        return

    # 2. Tính toán thông số toàn cục cho Weighted Rating (Cold Start)
    # Trích xuất rating và rating_number từ metadata
    df_meta = pd.DataFrame(metadata)
    
    # Xử lý dữ liệu thiếu
    df_meta['average_rating'] = pd.to_numeric(df_meta['average_rating'], errors='coerce').fillna(0)
    df_meta['rating_number'] = pd.to_numeric(df_meta['rating_number'], errors='coerce').fillna(0)
    
    C = df_meta['average_rating'].mean()
    m = df_meta['rating_number'].quantile(0.90)
    print(f"   -> Thống kê Cold Start: C={C:.2f}, m={m:.2f}")

    # Lưu lại DB mới bao gồm cả C và m
    torch.save({
        "vectors": item_vectors,
        "metadata": metadata,
        "stats": {"C": C, "m": m}
    }, OUTPUT_ITEMS_DB)
    print(f"✅ Đã cập nhật Item DB với thông số Cold Start.")

    # 3. Tạo User DB (Giả lập từ file code notebook mẫu)
    # Trong thực tế, bạn cần file reviews.jsonl để tạo cái này.
    # Vì bạn không gửi, mình sẽ tạo một DB rỗng để demo logic "User mới".
    # Nếu bạn có file reviews, hãy code đoạn đọc file đó ở đây.
    user_db = {} 
    # Ví dụ cấu trúc: user_db['USER_ID'] = {'indices': [1,2], 'ratings': [5,4]...}
    
    torch.save(user_db, OUTPUT_USER_DB)
    print(f"✅ Đã tạo User DB (Demo rỗng) tại {OUTPUT_USER_DB}")

if __name__ == "__main__":
    setup()