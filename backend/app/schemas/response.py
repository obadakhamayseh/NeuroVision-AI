

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class APIResponse(BaseModel):
    
    success: bool = Field(..., description="Indicates if the request was successful.")
    message: str = Field(..., description="Response summary message.")
    data: Optional[Any] = Field(None, description="The primary payload data.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Any additional metadata.")

class HealthResponse(BaseModel):
    
    status: str = Field(..., description="API service status (e.g. 'healthy').")
    version: str = Field(..., description="API software version.")
    uptime_seconds: float = Field(..., description="Server uptime in seconds.")
    device: str = Field(..., description="Model computation device (e.g. 'cuda', 'cpu').")
    model_loaded: bool = Field(..., description="Status of the deep learning model loading.")

class ModelInfoResponse(BaseModel):
    
    architecture: str = Field(..., description="Model architecture name.")
    version: str = Field(..., description="Model checkpoint version.")
    framework: str = Field(..., description="Framework used (e.g. PyTorch).")
    device: str = Field(..., description="Hardware device running inference.")
    classes: List[str] = Field(..., description="Supported classification tumor categories.")

class PredictResponse(BaseModel):
    
    prediction: str = Field(..., description="Predicted medical class label (e.g., 'Glioma', 'No Tumor').")
    confidence: float = Field(..., description="Model confidence as a percentage (0-100).")
    confidence_level: str = Field(..., description="Qualitative confidence interpretation (e.g. 'Very High Confidence').")
    probabilities: Dict[str, float] = Field(..., description="Probability dictionary for all classes.")
    gradcam_image: Optional[str] = Field(None, description="Relative URL or Base64 encoding of the Grad-CAM image.")
    heatmap_image: Optional[str] = Field(None, description="Relative URL or Base64 encoding of the raw Heatmap image.")
    overlay_image: Optional[str] = Field(None, description="Relative URL or Base64 encoding of the Overlay image.")
    report_pdf: Optional[str] = Field(None, description="Relative URL or Base64 path of the generated PDF report.")
    report_html: Optional[str] = Field(None, description="Relative URL or Base64 path of the generated HTML report.")
    report_json: Optional[str] = Field(None, description="Relative URL or Base64 path of the JSON report.")
    report_id: str = Field(..., description="Unique generated Report Identifier.")
    model: str = Field(..., description="Model architecture name.")
    model_version: str = Field(..., description="Version of the model dataset checkpoint.")
    device: str = Field(..., description="Device used for computing prediction.")
    inference_time_ms: float = Field(..., description="Logits computation time in milliseconds.")
    timestamp: str = Field(..., description="ISO 8601 formatting of analysis time.")
