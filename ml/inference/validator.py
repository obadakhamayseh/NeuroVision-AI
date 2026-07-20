

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple, Sequence

from PIL import Image, UnidentifiedImageError

from inference.exceptions import (
    CorruptedImageError,
    ImageNotFoundError,
    UnsupportedFormatError,
)

logger = logging.getLogger("brain_tumor_inference")

class ImageValidator:

    _DEFAULT_EXTENSIONS: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    )

    def __init__(
        self,
        allowed_extensions: Sequence[str] | None = None,
        expected_channels: int = 3,
        strict_dimensions: bool = False,
    ) -> None:
        self._allowed: frozenset[str] = frozenset(
            ext.lower()
            for ext in (allowed_extensions or self._DEFAULT_EXTENSIONS)
        )
        self._expected_channels = expected_channels
        self._strict_dimensions = strict_dimensions

    def validate(self, image_path: str | os.PathLike) -> Path:
        
        path = Path(image_path).resolve()
        self._check_exists(path)
        self._check_extension(path)
        self._check_openable(path)
        return path

    def _check_exists(self, path: Path) -> None:
        
        if not path.exists():
            raise ImageNotFoundError(
                f"Image not found: '{path}'"
            )
        if not path.is_file():
            raise ImageNotFoundError(
                f"Path exists but is not a file: '{path}'"
            )

    def _check_extension(self, path: Path) -> None:
        
        ext = path.suffix.lower()
        if ext not in self._allowed:
            raise UnsupportedFormatError(
                f"Unsupported file extension '{ext}'. "
                f"Allowed formats: {sorted(self._allowed)}"
            )

    def _check_openable(self, path: Path) -> None:
        
        try:
            with Image.open(path) as img:

                try:
                    img.verify()
                except Exception as exc:
                    raise CorruptedImageError(
                        f"Image verification failed (corrupt / truncated): '{path}'",
                        cause=exc,
                    ) from exc

        except (UnidentifiedImageError, OSError) as exc:
            raise CorruptedImageError(
                f"Image cannot be decoded: '{path}'",
                cause=exc,
            ) from exc

        try:
            with Image.open(path) as img:
                img.load()   
                self._check_channels(img, path)
        except CorruptedImageError:
            raise
        except Exception as exc:
            raise CorruptedImageError(
                f"Image failed pixel decode: '{path}'",
                cause=exc,
            ) from exc

    def _check_channels(self, img: Image.Image, path: Path) -> None:
        
        mode = img.mode
        
        mode_channels = {
            "1": 1, "L": 1, "P": 1,
            "RGB": 3, "RGBA": 4, "CMYK": 4,
            "YCbCr": 3, "LAB": 3, "HSV": 3,
            "I": 1, "F": 1, "LA": 2, "PA": 2,
            "RGBa": 4, "La": 2, "PA": 2,
        }
        channels = mode_channels.get(mode, -1)

        if channels != self._expected_channels:
            msg = (
                f"Image mode is '{mode}' ({channels} ch); "
                f"preprocessing will convert to RGB automatically. "
                f"Path: '{path}'"
            )
            if self._strict_dimensions:
                from inference.exceptions import ImageValidationError
                raise ImageValidationError(msg)
            logger.debug(msg)
