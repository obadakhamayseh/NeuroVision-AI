

from __future__ import annotations

from typing import Generator
from backend.app.services.prediction_service import PredictionService
from backend.app.services.gradcam_service import GradCAMService
from backend.app.services.report_service import ReportService

_prediction_service = None
_gradcam_service = None
_report_service = None

def get_prediction_service() -> PredictionService:
    
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service

def get_gradcam_service() -> GradCAMService:
    
    global _gradcam_service
    if _gradcam_service is None:
        _gradcam_service = GradCAMService()
    return _gradcam_service

def get_report_service() -> ReportService:
    
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service
