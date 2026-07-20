

from __future__ import annotations

import datetime
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

class ConfidenceLevel:
    
    VERY_HIGH = "Very High Confidence"
    HIGH = "High Confidence"
    MODERATE = "Moderate Confidence"
    LOW = "Low Confidence"
    VERY_LOW = "Very Low Confidence"

    @classmethod
    def get_level(cls, score: float) -> str:
        
        if score >= 95.0:
            return cls.VERY_HIGH
        elif score >= 85.0:
            return cls.HIGH
        elif score >= 70.0:
            return cls.MODERATE
        elif score >= 50.0:
            return cls.LOW
        else:
            return cls.VERY_LOW

@dataclass(frozen=True)
class ImageMetadata:
    
    filename: str
    resolution: Tuple[int, int]  
    format: str                  
    file_size_kb: float
    channels: int = 3

@dataclass(frozen=True)
class ModelMetadata:
    
    architecture: str = "EfficientNet-B0"
    framework: str = "PyTorch"
    inference_device: str = "CPU"
    inference_time_ms: float = 0.0
    model_version: str = "1.0.0"

@dataclass(frozen=True)
class ReportMetadata:
    
    report_id: str
    generation_date: datetime.datetime = field(
        default_factory=datetime.datetime.now
    )
    language: str = "en"
    organization: str = "NeuroVision AI Lab"

    @classmethod
    def generate_unique(cls, language: str = "en", org: str = "NeuroVision AI Lab") -> ReportMetadata:
        
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        random_suffix = uuid.uuid4().hex[:4].upper()
        report_id = f"REPORT-{timestamp}-{random_suffix}"
        return cls(
            report_id=report_id,
            generation_date=now,
            language=language,
            organization=org
        )

@dataclass
class ReportConfig:
    
    output_dir: str = "artifacts/reports"
    results_dir: str = "artifacts/results"
    theme_color: str = "medical-blue"  
    language: str = "en"
    include_original: bool = True
    include_heatmap: bool = True
    include_overlay: bool = True
    pdf_quality: str = "high"  
    image_format: str = "png"  
    html_theme: str = "clean-minimal"

    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "output_dir": self.output_dir,
            "results_dir": self.results_dir,
            "theme_color": self.theme_color,
            "language": self.language,
            "include_original": self.include_original,
            "include_heatmap": self.include_heatmap,
            "include_overlay": self.include_overlay,
            "pdf_quality": self.pdf_quality,
            "image_format": self.image_format,
            "html_theme": self.html_theme,
        }
