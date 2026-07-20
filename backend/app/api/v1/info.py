

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from backend.app.config.settings import settings
from backend.app.schemas.response import APIResponse, ModelInfoResponse
from backend.app.dependencies import get_prediction_service
from backend.app.services.prediction_service import PredictionService

router = APIRouter()

@router.get(
    "",
    response_model=APIResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model Information",
    description="Retrieve deep learning classification architecture specs, version metadata, and target classes."
)
async def get_model_info(
    pred_service: PredictionService = Depends(get_prediction_service)
) -> APIResponse:
    
    info = ModelInfoResponse(
        architecture="EfficientNet-B0",
        version=settings.MODEL_VERSION,
        framework="PyTorch",
        device=str(pred_service.predictor.device),
        classes=pred_service.predictor.predictor._class_names
    )
    
    return APIResponse(
        success=True,
        message="Model information metadata retrieved successfully.",
        data=info.dict()
    )
