import pandas as pd
import sys
import os

# --- CẤU HÌNH ĐƯỜNG DẪN IMPORT ---
# Giúp Python tìm thấy folder 'llm' nằm ngang hàng với folder 'be'
# Cấu trúc:
# root/
#   be/
#       rag_wrapper.py
#       llm/llm_utils.py
current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
llm_dir = os.path.join(current_dir, 'llm')
sys.path.append(llm_dir)
# --- IMPORT TỪ LLM_UTILS ---
try:
    from llm_utils import generate_rag_recommendation, configure_gemini
    print("✅ Đã import thành công llm_utils")
except ImportError as e:
    print(f"⚠️ Lỗi import llm_utils: {e}")
    print(f"👉 Python đang tìm ở: {sys.path}")
    
    # Hàm dummy để không crash nếu thiếu file
    def generate_rag_recommendation(*args, **kwargs): 
        return {"error": "Module llm_utils not found"}
    def configure_gemini(): return False

# Cấu hình Gemini ngay khi load file
configure_gemini()

# --- HÀM CHÍNH MÀ MAIN.PY ĐANG TÌM KIẾM ---
def run_rag_pipeline(query_text, reranked_indices, item_meta, user_db, user_id):
    """
    Wrapper chuyển đổi dữ liệu từ Backend (Dict/List) -> DataFrame 
    để gọi hàm RAG bên llm_utils.
    """
    print(f"🔄 RAG Wrapper: Đang xử lý cho User {user_id}...")

    # 1. Tạo User DataFrame giả lập
    # Cấu trúc: user_df có cột ['user_id', 'item_indices_seq', 'ratings_seq']
    # Backend lưu history dạng list, ta cần chuyển thành string list "[1, 2]" để khớp logic cũ
    u_indices = user_db.get(user_id, {}).get('indices', [])
    u_ratings = user_db.get(user_id, {}).get('ratings', [])
    
    user_data = {
        'user_id': [user_id],
        'item_indices_seq': [str(u_indices)], # Convert list -> string representation
        'ratings_seq': [str(u_ratings)]
    }
    user_df = pd.DataFrame(user_data)
    # 2. Tạo Main DataFrame (Item) giả lập
    # Convert list of dicts -> DataFrame
    # Chỉ convert những item cần thiết hoặc toàn bộ (nếu list nhỏ)
    print("   -> Converting Item Metadata...")
    main_df = pd.DataFrame(item_meta)
    
    # Map tên cột cho khớp với llm_utils (nếu cần)
    # llm_utils dùng: 'title', 'description_text', 'features_text', 'price', 'average_rating'
    
    # Xử lý cột description (Backend đang là List -> nối thành String)
    if 'description' in main_df.columns:
        main_df['description_text'] = main_df['description'].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
    
    # Xử lý cột features
    if 'features' in main_df.columns:
        main_df['features_text'] = main_df['features'].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )

    # 3. Gọi hàm RAG gốc
    print("   -> Calling Gemini API...")
    try:
        result = generate_rag_recommendation(
            original_query=query_text,
            reranked_indices=reranked_indices, # List int
            main_df=main_df,
            user_df=user_df,
            user_id_for_rerank=user_id,
            top_m=5 
        )
        return result
    except Exception as e:
        print(f"❌ RAG Runtime Error: {e}")
        return {"recommendations": [], "general_advice": "Lỗi khi gọi AI."}