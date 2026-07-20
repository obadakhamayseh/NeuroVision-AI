

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from backend.app.config.settings import settings

class UploadValidator:

    @staticmethod
    def validate_extension(filename: str) -> None:
        
        suffix = Path(filename).suffix.lower()
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        if suffix not in allowed:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Allowed formats: {sorted(allowed)}"
            )

    @staticmethod
    def validate_size(size: int) -> None:
        
        if size > settings.MAX_UPLOAD_SIZE:
            max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
            raise ValueError(
                f"File size exceeds maximum limit of {max_mb:.1f} MB."
            )

    @staticmethod
    def validate_image_data(file_bytes: bytes) -> None:
        
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Invalid image file. Image data is corrupted or unreadable.") from exc

        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
        except Exception as exc:
            raise ValueError("Corrupt image data found during pixel decoding.") from exc
