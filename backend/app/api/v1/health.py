

from __future__ import annotations

import time
from fastapi import APIRouter, Depends, status

from backend.app.config.settings import settings
from backend.app.schemas.response import APIResponse, HealthResponse
from backend.app.dependencies import get_prediction_service
from backend.app.services.prediction_service import PredictionService

router = APIRouter()

START_TIME = time.perf_counter()

@router.get(
    "",
    response_model=APIResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Service Health",
    description="Check the system status, uptime, device type, and model initialization state."
)
async def check_health(
    pred_service: PredictionService = Depends(get_prediction_service)
) -> APIResponse:
    
    uptime = time.perf_counter() - START_TIME
    
    health_info = HealthResponse(
        status="healthy",
        version=settings.API_VERSION,
        uptime_seconds=round(uptime, 2),
        device=str(pred_service.predictor.device),
        model_loaded=pred_service.predictor.engine.model is not None
    )
    
    return APIResponse(
        success=True,
        message="System is healthy and fully operational.",
        data=health_info.dict()
    )
