

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("brain_tumor_pipeline")

class TrainingHistory:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.epochs: List[Dict[str, Any]] = []
        self._json_path: str = cfg.HISTORY_JSON
        self._csv_path:  str = cfg.HISTORY_CSV
        os.makedirs(os.path.dirname(self._json_path), exist_ok=True)

    def update(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        epoch_time: float,
    ) -> None:
        
        record: Dict[str, Any] = {
            "epoch":               epoch + 1,
            "learning_rate":       lr,
            "train_loss":          train_metrics.get("loss"),
            "val_loss":            val_metrics.get("loss"),
            "train_accuracy":      train_metrics.get("accuracy"),
            "val_accuracy":        val_metrics.get("accuracy"),
            "train_macro_f1":      train_metrics.get("macro_f1"),
            "val_macro_f1":        val_metrics.get("macro_f1"),
            "train_weighted_f1":   train_metrics.get("weighted_f1"),
            "val_weighted_f1":     val_metrics.get("weighted_f1"),
            "train_macro_prec":    train_metrics.get("macro_precision"),
            "val_macro_prec":      val_metrics.get("macro_precision"),
            "train_macro_recall":  train_metrics.get("macro_recall"),
            "val_macro_recall":    val_metrics.get("macro_recall"),
            "epoch_time_s":        round(epoch_time, 2),
            "timestamp":           datetime.now().isoformat(),
        }
        self.epochs.append(record)

    def save(self) -> None:
        
        self._save_json()
        self._save_csv()

    def _save_json(self) -> None:
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"epochs": self.epochs, "total_epochs": len(self.epochs)},
                f,
                indent=2,
            )

    def _save_csv(self) -> None:
        if not self.epochs:
            return
        fieldnames = list(self.epochs[0].keys())
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epochs)

    def to_dict(self) -> Dict:
        
        return {"epochs": self.epochs, "total_epochs": len(self.epochs)}

    def plot(
        self,
        save_dir: Optional[str] = None,
        confusion_matrix: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        
        if not self.epochs:
            logger.warning("No epochs recorded; skipping training plots.")
            return

        out_dir = save_dir or self.cfg.ARTIFACTS_FIGURES_DIR
        os.makedirs(out_dir, exist_ok=True)

        epochs_x = [r["epoch"] for r in self.epochs]
        train_loss = [r["train_loss"] for r in self.epochs]
        val_loss   = [r["val_loss"]   for r in self.epochs]
        train_acc  = [r["train_accuracy"] for r in self.epochs]
        val_acc    = [r["val_accuracy"]   for r in self.epochs]
        lrs        = [r["learning_rate"]  for r in self.epochs]

        _STYLE = {
            "train": {"color": "#2196F3", "label": "Train"},
            "val":   {"color": "#F44336", "label": "Validation"},
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs_x, train_loss, color=_STYLE["train"]["color"],
                label=_STYLE["train"]["label"], linewidth=2, marker="o", markersize=3)
        ax.plot(epochs_x, val_loss,   color=_STYLE["val"]["color"],
                label=_STYLE["val"]["label"],   linewidth=2, marker="s", markersize=3)
        ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(out_dir, "training_loss_curve.png")
        fig.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curve saved: %s", loss_path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs_x, [a * 100 for a in train_acc],
                color=_STYLE["train"]["color"], label=_STYLE["train"]["label"],
                linewidth=2, marker="o", markersize=3)
        ax.plot(epochs_x, [a * 100 for a in val_acc],
                color=_STYLE["val"]["color"], label=_STYLE["val"]["label"],
                linewidth=2, marker="s", markersize=3)
        ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_path = os.path.join(out_dir, "training_accuracy_curve.png")
        fig.savefig(acc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Accuracy curve saved: %s", acc_path)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs_x, lrs, color="#4CAF50", linewidth=2, marker="o", markersize=3)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        lr_path = os.path.join(out_dir, "training_lr_curve.png")
        fig.savefig(lr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("LR curve saved: %s", lr_path)

        if confusion_matrix is not None:
            self._plot_confusion_matrix(
                confusion_matrix, class_names, out_dir
            )

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]],
        out_dir: str,
    ) -> None:
        
        n = cm.shape[0]
        names = class_names or [str(i) for i in range(n)]

        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(n), yticks=np.arange(n),
            xticklabels=names, yticklabels=names,
            xlabel="Predicted Label", ylabel="True Label",
            title="Confusion Matrix (row-normalised)",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)

        thresh = cm_norm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        cm_path = os.path.join(out_dir, "training_confusion_matrix.png")
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved: %s", cm_path)
