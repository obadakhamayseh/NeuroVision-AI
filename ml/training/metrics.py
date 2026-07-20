

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger("brain_tumor_pipeline")

class MetricsCalculator:

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

    def compute(
        self,
        y_true: List[int],
        y_pred: List[int],
        loss: float,
    ) -> Dict:
        
        y_true_arr = np.array(y_true, dtype=np.int32)
        y_pred_arr = np.array(y_pred, dtype=np.int32)

        labels = list(range(self.num_classes))
        zero_div = 0.0  

        accuracy: float = float(accuracy_score(y_true_arr, y_pred_arr))

        macro_precision: float = float(
            precision_score(
                y_true_arr, y_pred_arr,
                average="macro", labels=labels, zero_division=zero_div,
            )
        )
        macro_recall: float = float(
            recall_score(
                y_true_arr, y_pred_arr,
                average="macro", labels=labels, zero_division=zero_div,
            )
        )
        macro_f1: float = float(
            f1_score(
                y_true_arr, y_pred_arr,
                average="macro", labels=labels, zero_division=zero_div,
            )
        )

        weighted_precision: float = float(
            precision_score(
                y_true_arr, y_pred_arr,
                average="weighted", labels=labels, zero_division=zero_div,
            )
        )
        weighted_recall: float = float(
            recall_score(
                y_true_arr, y_pred_arr,
                average="weighted", labels=labels, zero_division=zero_div,
            )
        )
        weighted_f1: float = float(
            f1_score(
                y_true_arr, y_pred_arr,
                average="weighted", labels=labels, zero_division=zero_div,
            )
        )

        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

        return {
            "loss": round(loss, 6),
            "accuracy": round(accuracy, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_precision": round(weighted_precision, 6),
            "weighted_recall": round(weighted_recall, 6),
            "weighted_f1": round(weighted_f1, 6),
            "confusion_matrix": cm,
        }
