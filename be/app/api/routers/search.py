from fastapi import APIRouter, File, UploadFile, Form
from app.core.ml_manager import ml
from app.services.search_service import perform_search
from app.services.rag_wrapper import run_rag_pipeline

router = APIRouter()

@router.post("/search")
async def run_full_query_api(
    query_text: str = Form(""),
    user_id: str = Form(...),
    image: UploadFile = File(None),
    k_retrieval: int = Form(100),
    k_rerank: int = Form(20),
    alpha: float = Form(0.7),
    beta: float = Form(0.3),
    use_rag: bool = Form(False)
):
    print(f"\nBắt đầu query: '{query_text}' - User: {user_id}")
    
    # Hàm perform_search sẽ chứa toàn bộ khối lệnh "with torch.no_grad():" của bạn
    # Mình gói nó lại để API nhìn siêu gọn gàng.
    results, final_indices = await perform_search(
        query_text, user_id, image, k_retrieval, k_rerank, alpha, beta, ml
    )
    
    rag_response = None
    if use_rag:
        print(" Đang gọi LLM để phân tích RAG...")
        # Lưu ý: Hàm này có thể tốn vài giây chờ API của LLM phản hồi
        rag_response = run_rag_pipeline(
            query_text=query_text,
            reranked_indices=final_indices, # Truyền danh sách ID sản phẩm Top K
            item_meta=ml.ITEM_META,         # Truyền kho dữ liệu để LLM đọc thông tin
            user_db=ml.USER_DB,             # Truyền lịch sử user để LLM hiểu bối cảnh
            user_id=user_id
        )

    return {
        "results": results,
        "rag_analysis": rag_response
    }