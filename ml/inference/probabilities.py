

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from inference.response import TopKPrediction

logger = logging.getLogger("brain_tumor_inference")

_DISPLAY_LABELS: Tuple[str, ...] = (
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
)

def compute_probabilities(logits: torch.Tensor) -> torch.Tensor:
    
    return F.softmax(logits.float().squeeze(0), dim=0)

def get_top_prediction(
    probs: torch.Tensor,
    class_names: List[str],
) -> Tuple[str, int, float]:
    
    class_index: int = probs.argmax(dim=0).item()  
    confidence: float = probs[class_index].item() * 100.0
    label = _to_display_label(class_names[class_index])
    return label, class_index, confidence

def build_probability_map(
    probs: torch.Tensor,
    class_names: List[str],
) -> Dict[str, float]:
    
    prob_list: List[float] = probs.cpu().tolist()
    return {
        _to_display_label(name): round(p * 100.0, 4)
        for name, p in zip(class_names, prob_list)
    }

def get_top_k(
    probs: torch.Tensor,
    class_names: List[str],
    k: int = 4,
) -> List[TopKPrediction]:
    
    k = min(k, len(class_names))
    top_values, top_indices = torch.topk(probs, k=k)

    results: List[TopKPrediction] = []
    for rank, (idx, val) in enumerate(
        zip(top_indices.tolist(), top_values.tolist()), start=1
    ):
        results.append(
            TopKPrediction(
                rank=rank,
                class_name=_to_display_label(class_names[idx]),
                confidence=round(val * 100.0, 4),
            )
        )
    return results

def _to_display_label(class_name: str) -> str:
    
    _MAP: Dict[str, str] = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "no tumor": "No Tumor",
        "pituitary": "Pituitary",
    }
    return _MAP.get(class_name.lower(), class_name.title())
