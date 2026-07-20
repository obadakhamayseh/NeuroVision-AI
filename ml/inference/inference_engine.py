

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from inference.exceptions import DeviceError, InferenceRuntimeError
from inference.model_loader import ModelLoader
from inference.postprocess import PostProcessor
from inference.preprocess_single import SingleImagePreprocessor
from inference.response import PredictionResponse
from inference.validator import ImageValidator

logger = logging.getLogger("brain_tumor_inference")

class InferenceEngine:

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._model, self._device = ModelLoader.get_instance(cfg)

        self._validator = ImageValidator(
            allowed_extensions=list(cfg.VALID_EXTENSIONS),
            expected_channels=cfg.NUM_CHANNELS,
        )

        self._preprocessor = SingleImagePreprocessor(
            image_size=cfg.IMAGE_SIZE,
            mean=cfg.IMAGENET_MEAN,
            std=cfg.IMAGENET_STD,
            device=self._device,
        )

        top_k: int = getattr(cfg, "INFERENCE_TOP_K", 4)
        self._postprocessor = PostProcessor(
            class_names=list(cfg.CLASS_NAMES),
            top_k=top_k,
            model_version=cfg.DATASET_VERSION,
            device=self._device,
        )

        self._use_cuda_timing = self._device.type == "cuda"

        logger.info(
            "InferenceEngine ready | device=%s | image_size=%s | top_k=%d",
            self._device, cfg.IMAGE_SIZE, top_k,
        )

    def predict(
        self,
        image_path: str | os.PathLike,
    ) -> PredictionResponse:
        
        path = Path(image_path).resolve()
        logger.debug("Inference requested | image='%s'", path.name)

        self._validator.validate(path)

        tensor: torch.Tensor = self._preprocessor.preprocess(path)

        logits, elapsed_ms = self._forward(tensor)

        response: PredictionResponse = self._postprocessor.process(
            logits=logits,
            inference_time_ms=elapsed_ms,
            image_path=path,
        )

        return response

    def warmup(self, n_iterations: int = 3) -> None:
        
        logger.info("Running %d warmup iterations ...", n_iterations)
        h, w = self._cfg.IMAGE_SIZE
        dummy = torch.zeros(
            1, 3, h, w, dtype=torch.float32, device=self._device
        )
        with torch.inference_mode():
            for _ in range(n_iterations):
                _ = self._model(dummy)
        if self._use_cuda_timing:
            torch.cuda.synchronize()
        logger.info("Warmup complete.")

    @property
    def model(self) -> nn.Module:
        
        return self._model

    @property
    def device(self) -> torch.device:
        
        return self._device

    @property
    def preprocessor(self) -> SingleImagePreprocessor:
        
        return self._preprocessor

    def _forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        
        try:
            if self._use_cuda_timing:
                return self._forward_cuda(tensor)
            return self._forward_cpu(tensor)

        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise DeviceError(
                "CUDA out-of-memory during inference. "
                "Try reducing image size or switching to CPU.",
                cause=exc,
            ) from exc
        except (DeviceError, InferenceRuntimeError):
            raise
        except Exception as exc:
            raise InferenceRuntimeError(
                f"Forward pass failed: {exc}", cause=exc
            ) from exc

    def _forward_cuda(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            start_event.record()
            logits: torch.Tensor = self._model(tensor)
            end_event.record()

        torch.cuda.synchronize()
        elapsed_ms: float = start_event.elapsed_time(end_event)
        return logits, elapsed_ms

    def _forward_cpu(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        
        with torch.inference_mode():
            t0 = time.perf_counter()
            logits: torch.Tensor = self._model(tensor)
            elapsed_ms: float = (time.perf_counter() - t0) * 1000.0
        return logits, elapsed_ms
