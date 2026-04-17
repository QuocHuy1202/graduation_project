from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.ml_manager import ml
from app.api.routers import search, user

# 1. Khởi tạo App
app = FastAPI(title="Multimodal Recommendation System API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# 2. Sự kiện Startup (Chạy ngay khi bật Server)
@app.on_event("startup")
async def startup_event():
    # Tải toàn bộ Model 1 lần duy nhất
    ml.load_all_models()

# 3. Gắn các API Routers vào App chính
app.include_router(search.router, tags=["Search"])
app.include_router(user.router, tags=["User"])

# Chạy server bằng lệnh: uvicorn app.main:app --reload