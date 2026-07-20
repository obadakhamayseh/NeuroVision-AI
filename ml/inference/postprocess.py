

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import torch

from inference.exceptions import InferenceRuntimeError
from inference.probabilities import (
    build_probability_map,
    compute_probabilities,
    get_top_k,
    get_top_prediction,
)
from inference.response import PredictionResponse

logger = logging.getLogger("brain_tumor_inference")

class PostProcessor:

    _MODEL_ARCH: str = "EfficientNet-B0"

    def __init__(
        self,
        class_names: List[str],
        top_k: int,
        model_version: str,
        device: torch.device,
    ) -> None:
        self._class_names = class_names
        self._top_k = top_k
        self._model_version = model_version
        self._device_str = str(device)

    def process(
        self,
        logits: torch.Tensor,
        inference_time_ms: float,
        image_path: Optional[str | os.PathLike] = None,
    ) -> PredictionResponse:
        
        try:
            
            probs: torch.Tensor = compute_probabilities(logits)

            label, class_index, confidence = get_top_prediction(
                probs, self._class_names
            )

            prob_map = build_probability_map(probs, self._class_names)

            top_k_list = get_top_k(probs, self._class_names, k=self._top_k)

        except Exception as exc:
            raise InferenceRuntimeError(
                f"Post-processing failed: {exc}", cause=exc
            ) from exc

        path_str = str(image_path) if image_path else ""

        response = PredictionResponse(
            prediction=label,
            class_index=class_index,
            confidence=confidence,
            probabilities=prob_map,
            top_k=top_k_list,
            inference_time_ms=inference_time_ms,
            device=self._device_str,
            model=self._MODEL_ARCH,
            model_version=self._model_version,
            image_path=path_str,
        )

        logger.info(
            "Prediction: %-14s | Confidence: %6.2f%% | Time: %.1f ms | Device: %s",
            response.prediction,
            response.confidence,
            response.inference_time_ms,
            response.device,
        )

        return response
