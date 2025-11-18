import google.generativeai as genai
import os
import pandas as pd
import warnings
import ast # Để parse (phân tích) các chuỗi list từ CSV
import json
from dotenv import load_dotenv

# --- CẤU HÌNH LOAD .ENV  ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
be_dir = os.path.dirname(current_file_dir)
env_path = os.path.join(be_dir, '.env')
load_dotenv(dotenv_path=env_path)
def configure_gemini():
    """
    Cấu hình API Gemini bằng cách đọc key từ biến môi trường.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    print(api_key)
    if not api_key:
        print("ERROR: Environment variable 'GEMINI_API_KEY' not found.")
        print("Please set 'GEMINI_API_KEY' before running.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        print("Gemini API key configured successfully.")
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

def format_product_context(reranked_indices, main_df, top_m=3):
    """
    Trích xuất metadata (theo pipeline.pdf) từ top_m item 
    và định dạng chúng thành một context string.
    (Đọc từ main_df, tức là 'item.csv')
    (Đã xử lý NaN)
    """
    context_str = "Here are the top products we found related to the query:\n\n"
    
    if hasattr(reranked_indices, 'cpu'):
        top_m_indices = reranked_indices.cpu().numpy()[:top_m]
    else:
        top_m_indices = reranked_indices[:top_m]
    
    for i, item_index in enumerate(top_m_indices):
        try: # Khôi phục try-except để xử lý lỗi an toàn
            item = main_df.iloc[item_index]
            
            # --- XỬ LÝ NaN ---
            title_val = item.get('title')
            title = 'N/A' if pd.isna(title_val) else str(title_val)

            desc_val = item.get('description_text')
            description = '' if pd.isna(desc_val) else str(desc_val)
            
            feat_val = item.get('features_text')
            features = '' if pd.isna(feat_val) else str(feat_val)

            price_val = item.get('price')
            price = 'Not available' if pd.isna(price_val) else str(price_val)

            rating_val = item.get('average_rating')
            rating = 'Not available' if pd.isna(rating_val) else str(rating_val)
            # --- KẾT THÚC XỬ LÝ NaN ---
            
            if len(description) > 150:
                description = description[:150] + "..."
            if len(features) > 150:
                features = features[:150] + "..."

            context_str += f"--- RECOMMENDED PRODUCT {i+1} (Index: {item_index}) ---\n"
            context_str += f"Product Name: {title}\n"
            
            if description:
                context_str += f"Description: {description}\n"
            if features:
                context_str += f"Features: {features}\n"
                
            context_str += f"Price: ${price}\n"
            context_str += f"Rating: {rating} / 5 stars\n"
            context_str += "\n"
            
        except Exception as e:
            print(f"Error processing item index {item_index}: {e}")
            
    return context_str

# --- (Hàm format_user_history không thay đổi) ---
def format_user_history(user_id, user_df, item_df, max_history=5):
    """
    Trích xuất lịch sử mua hàng của user (với ratings)
    từ user_df.csv và item.csv.
    """
    try:
        user_row = user_df[user_df['user_id'] == user_id]
        if user_row.empty:
            return "No purchase history found for this user.\n"
        
        user_row = user_row.iloc[0]
        
        try:
            item_indices = ast.literal_eval(user_row['item_indices_seq'])
            ratings = ast.literal_eval(user_row['ratings_seq'])
        except (ValueError, SyntaxError):
            print(f"Error parsing history for user {user_id}. Is it a valid list string?")
            return "Could not parse user history.\n"
            
        if not item_indices:
            return "User has an empty purchase history.\n"

        history_str = "Here is the user's recent purchase history (most recent first):\n\n"
        
        recent_indices = reversed(item_indices[-max_history:])
        recent_ratings = reversed(ratings[-max_history:])
        
        for item_idx, rating in zip(recent_indices, recent_ratings):
            try:
                item_title = item_df.iloc[item_idx].get('title', 'Unknown Item')
                history_str += f"- (Rated: {rating}/5) {item_title}\n"
            except IndexError:
                history_str += f"- (Rated: {rating}/5) Item at invalid index {item_idx}\n"
            
        return history_str + "\n"
        
    except Exception as e:
        print(f"Error formatting user history: {e}")
        return "Error loading user history.\n"


# --- HÀM GỌI API GEMINI (SỬ DỤNG JSON MODE) ---
def call_gemini_api(system_prompt, user_prompt, response_schema):
    """
    Hàm trợ giúp để gọi API Gemini ở chế độ JSON và trả về một dict.
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=system_prompt,
            # Cấu hình để yêu cầu output là JSON
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        response = model.generate_content(user_prompt)
        
        # Parse chuỗi JSON từ response.text thành một Python dict
        return json.loads(response.text) 
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Trả về một đối tượng lỗi tuân thủ schema (nhưng rỗng hoặc báo lỗi)
        return {
            "recommendations": [],
            "general_advice": f"An error occurred while calling the AI: {e}"
        }

# --- CẤU TRÚC JSON MỚI (CHO PHÉP NHIỀU ITEM) ---
RECOMMENDATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "recommendations": {
            "type": "ARRAY",
            "description": "A list of 1 to 3 recommended products. Can contain only 1 item if it's the clear best choice.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "selected_item_index": {
                        "type": "NUMBER",
                        "description": "The numerical index of the recommended product (e.g., 273)."
                    },
                    "selected_item_title": {
                        "type": "STRING",
                        "description": "The full 'Product Name' of the item."
                    },
                    "recommendation_reason": {
                        "type": "STRING",
                        "description": "A brief explanation of why this specific product fits the user's query and history."
                    }
                },
                "required": ["selected_item_index", "selected_item_title", "recommendation_reason"]
            }
        },
        "general_advice": {
            "type": "STRING",
            "description": "A concluding sentence giving general advice based on the user's history (e.g., 'Since you like X, you might enjoy these options.')."
        }
    },
    "required": ["recommendations", "general_advice"]
}


# --- HÀM CHÍNH RAG (ĐÃ CẬP NHẬT) ---
def generate_rag_recommendation(
    original_query, 
    reranked_indices, 
    main_df,      # DataFrame từ item.csv
    user_df,      # DataFrame từ user_df.csv
    user_id_for_rerank,
    top_m=5 # Tăng top_m lên để LLM có nhiều lựa chọn hơn
):
    """
    Thực hiện bước RAG + LLM cuối cùng, yêu cầu output dạng JSON (cho phép nhiều item).
    """
    print("--- Starting RAG Pipeline (Multi-Item Mode) ---")
    
    # --- 1. Lấy Context Sản Phẩm (Từ reranking) ---
    print(f"Extracting metadata for Top {top_m} products...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        product_context = format_product_context(reranked_indices, main_df, top_m)

    # --- 2. Lấy Context Người Dùng (Từ Lịch sử) ---
    print(f"Extracting purchase history for User {user_id_for_rerank}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        user_history_context = format_user_history(user_id_for_rerank, user_df, main_df)

    # --- 3. Chuẩn bị Prompt ---
    user_query_str = f"User's query: '{original_query}'"
    
    # System prompt MỚI, linh hoạt hơn về số lượng item
    system_prompt = (
        "You are a helpful AI shopping assistant. Your task is to analyze the user's query, "
        "their purchase history, and a list of recommended products. "
        "Select the best product(s) from the 'RECOMMENDED PRODUCT' list. "
        "You can recommend just ONE product if it's clearly the best match. "
        "If there are multiple good options (e.g., different styles, price points), you can recommend up to 3 products. "
        "Do NOT force multiple recommendations if they aren't relevant. "
        "You MUST respond ONLY with a valid JSON object adhering to the provided schema."
    )
    
    user_prompt = (
        f"--- USER ID ---\n{user_id_for_rerank}\n\n"
        f"--- USER QUERY ---\n{user_query_str}\n\n"
        f"--- USER PURCHASE HISTORY ---\n{user_history_context}"
        f"--- RECOMMENDED PRODUCTS FOR QUERY ---\n{product_context}"
        "Analyze all information. Which product(s) fit best? "
        "Provide your top recommendations (1-3 items) in the specified JSON format."
    )
    
    print("Prompt created. Calling Gemini API (JSON Mode)...")
    print(user_history_context)
    # --- 4. Gọi LLM (với Schema mới) ---
    final_recommendation_dict = call_gemini_api(
        system_prompt, 
        user_prompt, 
        RECOMMENDATION_SCHEMA 
    )
    
    print("--- RAG Complete ---")
    return final_recommendation_dict