

from __future__ import annotations

import logging
import threading
from typing import Optional

from inference.inference_engine import InferenceEngine
from inference.response import PredictionResponse

logger = logging.getLogger("brain_tumor_inference")

class Predictor:

    _instance: Optional["Predictor"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, cfg=None) -> None:
        if cfg is None:
            from config import Config
            cfg = Config()
        self._engine = InferenceEngine(cfg)

    @classmethod
    def get_instance(cls, cfg=None) -> "Predictor":
        
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
                logger.info("Predictor Singleton created.")

        return cls._instance

    def predict(self, image_path: str) -> PredictionResponse:
        
        return self._engine.predict(image_path)

    def warmup(self, n_iterations: int = 3) -> None:
        
        self._engine.warmup(n_iterations)

    @property
    def device(self) -> str:
        
        return str(self._engine.device)

    @property
    def engine(self) -> InferenceEngine:
        
        return self._engine
