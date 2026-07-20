

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass(frozen=True)
class TopKPrediction:

    rank: int
    class_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "rank": self.rank,
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
        }

@dataclass(frozen=True)
class PredictionResponse:

    prediction: str
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    top_k: List[TopKPrediction]
    inference_time_ms: float
    device: str
    model: str
    model_version: str
    image_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "prediction": self.prediction,
            "class_index": self.class_index,
            "confidence": round(self.confidence, 4),
            "probabilities": {
                k: round(v, 4) for k, v in self.probabilities.items()
            },
            "top_k": [t.to_dict() for t in self.top_k],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "device": self.device,
            "model": self.model,
            "model_version": self.model_version,
            "image_path": self.image_path,
        }

    def __str__(self) -> str:
        sep = "=" * 54
        lines = [
            sep,
            f"  Prediction       : {self.prediction}",
            f"  Confidence       : {self.confidence:.2f} %",
            f"  Device           : {self.device.upper()}",
            f"  Inference Time   : {self.inference_time_ms:.1f} ms",
            f"  Model            : {self.model}",
            sep,
            "  Class Probabilities:",
        ]
        for class_name, prob in self.probabilities.items():
            bar_len = int(prob / 2)          
            bar = "#" * bar_len + "-" * (50 - bar_len)
            lines.append(f"    {class_name:<14} {prob:6.2f} %  |{bar}|")
        lines.append(sep)
        return "\n".join(lines)
