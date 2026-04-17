import os
import json
import ast
import warnings
import pandas as pd
import google.generativeai as genai
from app.core.config import settings

# =====================================================================
# 1. CẤU HÌNH GEMINI
# =====================================================================
def configure_gemini():
    """Cấu hình API Gemini từ settings chung của hệ thống."""
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("ERROR: 'GEMINI_API_KEY' not found in settings/.env.")
        return False
    try:
        genai.configure(api_key=api_key)
        print("Gemini API key configured successfully.")
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

# Chạy cấu hình ngay khi file này được import
configure_gemini()

# =====================================================================
# 2. XỬ LÝ CONTEXT (SẢN PHẨM & LỊCH SỬ)
# =====================================================================
def format_product_context(reranked_indices, main_df, top_m=3):
    """Trích xuất metadata từ DataFrame và định dạng thành context string."""
    context_str = "Here are the top products we found related to the query:\n\n"
    
    if hasattr(reranked_indices, 'cpu'):
        top_m_indices = reranked_indices.cpu().numpy()[:top_m]
    else:
        top_m_indices = reranked_indices[:top_m]
    
    for i, item_index in enumerate(top_m_indices):
        try: 
            item = main_df.iloc[item_index]
            
            # Xử lý NaN
            title = str(item.get('title', 'N/A')) if pd.notna(item.get('title')) else 'N/A'
            description = str(item.get('description_text', '')) if pd.notna(item.get('description_text')) else ''
            features = str(item.get('features_text', '')) if pd.notna(item.get('features_text')) else ''
            price = str(item.get('price', 'Not available')) if pd.notna(item.get('price')) else 'Not available'
            rating = str(item.get('average_rating', 'Not available')) if pd.notna(item.get('average_rating')) else 'Not available'
            
            if len(description) > 150: description = description[:150] + "..."
            if len(features) > 150: features = features[:150] + "..."

            context_str += f"--- RECOMMENDED PRODUCT {i+1} (Index: {item_index}) ---\n"
            context_str += f"Product Name: {title}\n"
            if description: context_str += f"Description: {description}\n"
            if features: context_str += f"Features: {features}\n"
            context_str += f"Price: ${price}\n"
            context_str += f"Rating: {rating} / 5 stars\n\n"
            
        except Exception as e:
            print(f"Error processing item index {item_index}: {e}")
            
    return context_str

def format_user_history(user_id, user_df, item_df, max_history=5):
    """Trích xuất lịch sử mua hàng của user."""
    try:
        user_row = user_df[user_df['user_id'] == user_id]
        if user_row.empty:
            return "No purchase history found for this user.\n"
        
        user_row = user_row.iloc[0]
        
        try:
            item_indices = ast.literal_eval(user_row['item_indices_seq'])
            ratings = ast.literal_eval(user_row['ratings_seq'])
        except (ValueError, SyntaxError):
            return "Could not parse user history.\n"
            
        if not item_indices:
            return "User has an empty purchase history.\n"

        history_str = "Here is the user's recent purchase history (most recent first):\n\n"
        recent_indices = list(reversed(item_indices[-max_history:]))
        recent_ratings = list(reversed(ratings[-max_history:]))
        
        for item_idx, rating in zip(recent_indices, recent_ratings):
            try:
                item_title = item_df.iloc[item_idx].get('title', 'Unknown Item')
                history_str += f"- (Rated: {rating}/5) {item_title}\n"
            except IndexError:
                pass
                
        return history_str + "\n"
        
    except Exception as e:
        print(f"Error formatting user history: {e}")
        return "Error loading user history.\n"

# =====================================================================
# 3. SCHEMA VÀ LLM CALL
# =====================================================================
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

def call_gemini_api(system_prompt, user_prompt, response_schema):
    """Gọi API Gemini ở chế độ JSON."""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        response = model.generate_content(user_prompt)
        return json.loads(response.text) 
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            "recommendations": [],
            "general_advice": f"An error occurred while calling the AI: {e}"
        }

def generate_rag_recommendation(original_query, reranked_indices, main_df, user_df, user_id_for_rerank, top_m=5):
    """Gom dữ liệu và yêu cầu LLM phân tích."""
    print("--- Starting RAG Pipeline (Multi-Item Mode) ---")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        product_context = format_product_context(reranked_indices, main_df, top_m)
        user_history_context = format_user_history(user_id_for_rerank, user_df, main_df)

    user_query_str = f"User's query: '{original_query}'"
    
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
    final_recommendation_dict = call_gemini_api(system_prompt, user_prompt, RECOMMENDATION_SCHEMA)
    print("--- RAG Complete ---")
    return final_recommendation_dict

# =====================================================================
# 4. HÀM WRAPPER ĐỂ BACKEND (SEARCH API) GỌI
# =====================================================================
def run_rag_pipeline(query_text, reranked_indices, item_meta, user_db, user_id):
    """
    Wrapper chuyển đổi dữ liệu từ Backend (Dict/List) -> DataFrame 
    để gọi hàm phân tích RAG.
    """
    print(f"🔄 RAG Wrapper: Đang xử lý cho User {user_id}...")

    # 1. Tạo User DataFrame giả lập
    u_indices = user_db.get(user_id, {}).get('indices', [])
    u_ratings = user_db.get(user_id, {}).get('ratings', [])
    
    user_data = {
        'user_id': [user_id],
        'item_indices_seq': [str(u_indices)],
        'ratings_seq': [str(u_ratings)]
    }
    user_df = pd.DataFrame(user_data)
    
    # 2. Tạo Main DataFrame (Item) giả lập
    print("   -> Converting Item Metadata...")
    main_df = pd.DataFrame(item_meta)
    
    if 'description' in main_df.columns:
        main_df['description_text'] = main_df['description'].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
    
    if 'features' in main_df.columns:
        main_df['features_text'] = main_df['features'].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )

    # 3. Gọi hàm RAG gốc
    print("   -> Calling Gemini API...")
    try:
        result = generate_rag_recommendation(
            original_query=query_text,
            reranked_indices=reranked_indices,
            main_df=main_df,
            user_df=user_df,
            user_id_for_rerank=user_id,
            top_m=5 
        )
        return result
    except Exception as e:
        print(f"❌ RAG Runtime Error: {e}")
        return {"recommendations": [], "general_advice": "Lỗi khi gọi AI."}