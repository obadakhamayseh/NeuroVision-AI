

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    API_TITLE: str = "NeuroVision AI REST API"
    API_DESCRIPTION: str = (
        "REST API backend exposing deep learning model for Brain Tumor Detection from MRI. "
        "Intended for research, educational, and decision-support purposes only."
    )
    API_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
    ]

    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "uploads")
    
    MAX_UPLOAD_SIZE: int = 15 * 1024 * 1024

    MODEL_CHECKPOINT_DIR: str = str(
        Path(__file__).resolve().parent.parent.parent.parent / "ml" / "artifacts" / "checkpoints"
    )
    MODEL_VERSION: str = "1.0.0"

    @property
    def resolved_upload_dir(self) -> Path:
        
        path = Path(self.UPLOAD_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()
