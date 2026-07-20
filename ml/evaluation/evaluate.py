

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from device import get_device

logger = logging.getLogger("brain_tumor_pipeline")

class Evaluator:

    def __init__(
        self,
        model: nn.Module,
        cfg,
        test_loader: DataLoader,
        class_names: List[str],
    ) -> None:
        self.cfg         = cfg
        self.test_loader = test_loader
        self.class_names = class_names
        self.device      = get_device(cfg.DEVICE)
        self.model       = model.to(self.device)
        self._use_amp    = cfg.USE_AMP and self.device.type == "cuda"
        self._out_dir    = cfg.ARTIFACTS_REPORTS_DIR
        self._fig_dir    = cfg.ARTIFACTS_FIGURES_DIR

    def evaluate(self) -> Dict:
        
        logger.info("=" * 60)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 60)

        y_true, y_pred, avg_loss = self._run_inference()
        labels = list(range(len(self.class_names)))

        report_str = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )
        logger.info("Classification Report:\n%s", report_str)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        accuracy        = float(accuracy_score(y_true, y_pred))
        macro_f1        = float(f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0))
        weighted_f1     = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
        macro_prec      = float(precision_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0))
        weighted_prec   = float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
        macro_rec       = float(recall_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0))
        weighted_rec    = float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

        results = {
            "generated_at":          datetime.now().isoformat(),
            "test_loss":             round(avg_loss, 6),
            "accuracy":              round(accuracy, 6),
            "macro_precision":       round(macro_prec, 6),
            "macro_recall":          round(macro_rec, 6),
            "macro_f1":              round(macro_f1, 6),
            "weighted_precision":    round(weighted_prec, 6),
            "weighted_recall":       round(weighted_rec, 6),
            "weighted_f1":           round(weighted_f1, 6),
            "classification_report": report_str,
            "confusion_matrix":      cm.tolist(),
            "class_names":           self.class_names,
        }

        logger.info(
            "Test | loss=%.4f | acc=%.4f | macro_f1=%.4f | weighted_f1=%.4f",
            avg_loss, accuracy, macro_f1, weighted_f1,
        )

        self._save_report(results)
        self._plot_confusion_matrix(cm)
        return results

    def _run_inference(self) -> Tuple[List[int], List[int], float]:
        
        self.model.eval()
        all_preds:  List[int] = []
        all_labels: List[int] = []
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(self.test_loader, desc="Test Evaluation", unit="batch")
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self._use_amp,
                ):
                    logits = self.model(images)
                    loss   = criterion(logits, labels)
                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = running_loss / max(len(self.test_loader), 1)
        return all_labels, all_preds, avg_loss

    def _save_report(self, results: Dict) -> None:
        
        os.makedirs(self._out_dir, exist_ok=True)
        path = os.path.join(self._out_dir, "eval_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Evaluation report saved: %s", path)

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        
        os.makedirs(self._fig_dir, exist_ok=True)
        n = cm.shape[0]
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(n),
            yticks=np.arange(n),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            xlabel="Predicted Label",
            ylabel="True Label",
            title="Test Set Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)

        thresh = cm_norm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        out = os.path.join(self._fig_dir, "test_confusion_matrix.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Test confusion matrix saved: %s", out)
