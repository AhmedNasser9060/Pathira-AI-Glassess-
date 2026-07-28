import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

# Get the directory where this file is located
CORE_DIR = Path(__file__).resolve().parent
# Get the backend directory (one level up from core)
BACKEND_DIR = CORE_DIR.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(BACKEND_DIR, ".env"),
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )

    PROJECT_NAME: str = "Pathira"
    API_V1_STR: str = ""
    
    # Security (MUST be provided in .env)
    JWT_SECRET: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database (MUST be provided in .env)
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    SQL_ECHO: bool = False

    # ML / Vision
    ML_WEIGHTS_DIR: str = "backend/ml_weights"
    OCR_TESSERACT_CMD: Optional[str] = None
    OCR_MAX_IMAGE_BYTES: int = 5 * 1024 * 1024
    FACE_MATCH_THRESHOLD: float = 0.6

    # OCR.space cloud API (preferred when OCR_SPACE_API_KEY is set; falls
    # back to local Tesseract on hard failure).
    OCR_SPACE_API_KEY: Optional[str] = None
    OCR_SPACE_URL: str = "https://api.ocr.space/parse/image"
    OCR_SPACE_ENGINE: int = 2  # Engine 2: better for special chars + confusing backgrounds
    OCR_SPACE_TIMEOUT: int = 30  # seconds

    # Voice commands. STT runs locally via faster-whisper, intent parsing
    # via a locally-hosted Ollama Mistral instance. Both are free and
    # work without internet egress.
    WHISPER_MODEL: str = "tiny.en"  # ~40 MB, fast on CPU; bump to "base.en" for accuracy
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"  # change to whatever `ollama list` shows
    OLLAMA_TIMEOUT: int = 30
    VOICE_MAX_AUDIO_BYTES: int = 5 * 1024 * 1024  # 5 MB cap on recordings

    @property
    def get_database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

settings = Settings()
