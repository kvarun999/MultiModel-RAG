from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # Enforced Environment Variables
    api_port: int = 8000
    environment: str = "development"
    chroma_persist_dir: str
    clip_model_name: str
    vlm_provider: str
    vlm_api_key: str
    document_ingestion_dir: str
    reranker_model_name: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # Good practice: ignores extra vars in .env we don't care about
    )

@lru_cache()
def get_settings():
    return Settings()