

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ml.config import Config
from ml.inference import Predictor
from backend.app.config.settings import settings

if TYPE_CHECKING:
    from ml.inference.response import PredictionResponse

logger = logging.getLogger("brain_tumor_api")

class PredictionService:

    def __init__(self) -> None:
        
        self.cfg = Config()

        if settings.DEBUG:
            logger.info("Initializing prediction service with model checkpoint path: %s", settings.MODEL_CHECKPOINT_DIR)
            
        self.predictor = Predictor.get_instance(self.cfg)

    def predict_image(self, file_path: Path) -> PredictionResponse:
        
        logger.info("Running model prediction for image: %s", file_path.name)
        
        response = self.predictor.predict(str(file_path))
        return response
