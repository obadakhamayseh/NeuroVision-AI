

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("brain_tumor_inference")

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
)

def load_image_pil(image_path: str | os.PathLike) -> Image.Image:
    
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: '{path}'")
    return Image.open(path)

def load_image_rgb(image_path: str | os.PathLike) -> Image.Image:
    
    with load_image_pil(image_path) as img:
        rgb_img: Image.Image = img.convert("RGB")
    
    return rgb_img.copy()

def get_image_metadata(
    image_path: str | os.PathLike,
) -> Dict[str, Any]:
    
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: '{path}'")

    size_kb = path.stat().st_size / 1024.0

    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format or "unknown"
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Could not read image metadata for '%s': %s", path, exc)
        width, height, mode, fmt = 0, 0, "unknown", "unknown"

    return {
        "path": str(path),
        "filename": path.name,
        "size_kb": round(size_kb, 2),
        "width": width,
        "height": height,
        "mode": mode,
        "format": fmt,
        "extension": path.suffix.lower(),
    }

def is_valid_image_path(
    image_path: str | os.PathLike,
    allowed_extensions: Optional[frozenset[str]] = None,
) -> bool:
    
    exts = allowed_extensions or _SUPPORTED_EXTENSIONS
    path = Path(image_path)
    return path.exists() and path.is_file() and path.suffix.lower() in exts
