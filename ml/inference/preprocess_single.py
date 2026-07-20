

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from inference.exceptions import PreprocessingError

logger = logging.getLogger("brain_tumor_inference")

_RESIZE_BEFORE_CROP: int = 540

class SingleImagePreprocessor:

    def __init__(
        self,
        image_size: Tuple[int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        device: torch.device,
    ) -> None:
        self._device = device
        self._image_size = image_size

        self._transform = T.Compose([
            
            T.Resize(
                _RESIZE_BEFORE_CROP,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            
            T.CenterCrop(size=image_size),
            
            T.ToTensor(),
            
            T.Normalize(
                mean=list(mean),
                std=list(std),
            ),
        ])

        logger.debug(
            "SingleImagePreprocessor ready | size=%s | mean=%s | std=%s | device=%s",
            image_size, mean, std, device,
        )

    def __call__(
        self, image_path: str | os.PathLike
    ) -> torch.Tensor:
        
        return self.preprocess(image_path)

    def preprocess(
        self, image_path: str | os.PathLike
    ) -> torch.Tensor:
        
        path = Path(image_path)
        try:

            image: Image.Image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to open image for preprocessing: '{path}'",
                cause=exc,
            ) from exc

        try:
            
            tensor: torch.Tensor = self._transform(image)
        except Exception as exc:
            raise PreprocessingError(
                f"Transform pipeline failed on image: '{path}'",
                cause=exc,
            ) from exc

        try:

            batched: torch.Tensor = tensor.unsqueeze(0).to(
                self._device, non_blocking=True
            )
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to move tensor to device '{self._device}': {exc}",
                cause=exc,
            ) from exc

        logger.debug(
            "Preprocessed '%s' | shape=%s | dtype=%s | device=%s",
            path.name, batched.shape, batched.dtype, batched.device,
        )
        return batched
