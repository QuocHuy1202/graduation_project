import os
from pydantic_settings import BaseSettings,  SettingsConfigDict

class Settings(BaseSettings):
    HF_TOKEN: str = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
    REPO_ID: str = "huyakaz/my-multimodal-recsys"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    CROSS_ENCODER_REPO: str = "huyakaz/my-crossencoder-v1"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

# Set biến môi trường cho thư viện Hugging Face
os.environ["HUGGINGFACE_HUB_TOKEN"] = settings.HF_TOKEN