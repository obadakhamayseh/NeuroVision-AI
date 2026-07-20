

from __future__ import annotations

import logging
import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, status

from backend.app.config.settings import settings
from backend.app.schemas.response import APIResponse, PredictResponse
from backend.app.dependencies import get_prediction_service, get_gradcam_service, get_report_service
from backend.app.services.prediction_service import PredictionService
from backend.app.services.gradcam_service import GradCAMService
from backend.app.services.report_service import ReportService
from backend.app.utils.file_utils import sanitize_upload_filename
from backend.app.utils.validators import UploadValidator
from backend.app.exceptions import APIException

router = APIRouter()
logger = logging.getLogger("brain_tumor_api")

@router.post(
    "",
    response_model=APIResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload Brain MRI Scan for Classification",
    description="Accepts an uploaded image file, validates it, runs inference, creates Grad-CAM overlays, and compiles reports."
)
async def predict_mri(
    file: UploadFile = File(..., description="The brain MRI scan image (supported: JPG, PNG, BMP, TIFF)."),
    pred_service: PredictionService = Depends(get_prediction_service),
    gc_service: GradCAMService = Depends(get_gradcam_service),
    rep_service: ReportService = Depends(get_report_service)
) -> APIResponse:
    
    filename = file.filename or "mri_scan.jpg"

    try:
        UploadValidator.validate_extension(filename)
    except ValueError as exc:
        raise APIException(status.HTTP_400_BAD_REQUEST, str(exc))

    file_bytes = await file.read()

    try:
        UploadValidator.validate_size(len(file_bytes))
    except ValueError as exc:
        raise APIException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc))

    try:
        UploadValidator.validate_image_data(file_bytes)
    except ValueError as exc:
        raise APIException(status.HTTP_400_BAD_REQUEST, str(exc))

    safe_name = sanitize_upload_filename(filename)
    upload_path = settings.resolved_upload_dir / safe_name
    try:
        with open(upload_path, "wb") as f:
            f.write(file_bytes)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise APIException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to save upload image to disk."
        )

    try:
        inference_res = pred_service.predict_image(upload_path)
    except Exception as exc:
        logger.error("Inference Engine failed: %s", exc)
        raise APIException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Prediction failed: {exc}"
        )

    try:

        tensor = pred_service.predictor.engine.preprocessor.preprocess(upload_path)

        results_root = Path("ml/artifacts/results").resolve()
        results_root.mkdir(parents=True, exist_ok=True)
        
        heatmap_path, overlay_path = gc_service.generate_heatmap_and_overlay(
            model=pred_service.predictor.engine.model,
            preprocessed_tensor=tensor,
            original_image_path=upload_path,
            target_class_idx=inference_res.class_index,
            output_dir=results_root / "temp"
        )
    except Exception as exc:
        logger.error("Grad-CAM generation failed: %s", exc)
        raise APIException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Explainability overlay generation failed: {exc}"
        )

    try:
        reports = rep_service.generate_report(
            prediction=inference_res.prediction,
            confidence=inference_res.confidence,
            probabilities=inference_res.probabilities,
            original_image_path=upload_path,
            heatmap_path=heatmap_path,
            overlay_path=overlay_path,
            inference_time_ms=inference_res.inference_time_ms,
            device=inference_res.device,
            model_version=inference_res.model_version
        )
    except Exception as exc:
        logger.error("Clinical Report Generation failed: %s", exc)
        raise APIException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Clinical report generation failed: {exc}"
        )

    try:
        if upload_path.exists():
            upload_path.unlink()
        if heatmap_path.exists():
            heatmap_path.unlink()
        if overlay_path.exists():
            overlay_path.unlink()
    except Exception as exc:
        logger.warning("Could not clean up temporary working files: %s", exc)

    from ml.report.metadata import ConfidenceLevel
    qualitative_confidence = ConfidenceLevel.get_level(inference_res.confidence)

    results_root = Path("ml/artifacts/results").resolve()

    def to_url(abs_path: str) -> str:
        
        try:
            rel = Path(abs_path).resolve().relative_to(results_root)
            return f"results/{rel.as_posix()}"
        except ValueError:
            return abs_path  

    predict_data = PredictResponse(
        prediction=inference_res.prediction,
        confidence=round(inference_res.confidence, 2),
        confidence_level=qualitative_confidence,
        probabilities={k.title() if k.lower() != "notumor" else "No Tumor": float(v) for k, v in inference_res.probabilities.items()},
        gradcam_image=to_url(reports["overlay_image"]),
        heatmap_image=to_url(reports["heatmap_image"]),
        overlay_image=to_url(reports["overlay_image"]),
        report_pdf=to_url(reports["report_pdf"]),
        report_html=to_url(reports["report_html"]),
        report_json=to_url(reports["report_json"]),
        report_id=reports["report_id"],
        model="EfficientNet-B0",
        model_version=inference_res.model_version,
        device=inference_res.device.upper(),
        inference_time_ms=round(inference_res.inference_time_ms, 2),
        timestamp=datetime.datetime.now().isoformat()
    )

    return APIResponse(
        success=True,
        message="MRI scan analyzed and reports generated successfully.",
        data=predict_data.dict()
    )
