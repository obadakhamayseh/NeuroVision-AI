

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("brain_tumor_pipeline")

_LAST_CKPT  = "last_model.pth"
_BEST_CKPT  = "best_model.pth"

class CheckpointManager:

    def __init__(self, cfg) -> None:
        self.ckpt_dir: str = cfg.CHECKPOINTS_DIR
        os.makedirs(self.ckpt_dir, exist_ok=True)
        logger.info("Checkpoint directory: %s", self.ckpt_dir)

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        val_metrics: Dict,
        history: Any,
        best_val_loss: float,
        is_best: bool = False,
    ) -> None:
        
        state: Dict[str, Any] = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict() if hasattr(scheduler, "state_dict")
                else {}
            ),
            "best_val_loss":   best_val_loss,
            "val_metrics":     {
                k: v for k, v in val_metrics.items()
                if not hasattr(v, "__len__") or k == "confusion_matrix"
            },
            "history":         history.to_dict() if hasattr(history, "to_dict") else {},
        }

        last_path = os.path.join(self.ckpt_dir, _LAST_CKPT)
        torch.save(state, last_path)
        logger.debug("Checkpoint saved: %s (epoch %d)", last_path, epoch + 1)

        if is_best:
            best_path = os.path.join(self.ckpt_dir, _BEST_CKPT)
            torch.save(state, best_path)
            logger.info(
                "Best model saved: val_loss=%.6f  epoch=%d",
                best_val_loss, epoch + 1,
            )

    def load_latest(self) -> Optional[Dict[str, Any]]:
        
        path = os.path.join(self.ckpt_dir, _LAST_CKPT)
        if not os.path.isfile(path):
            logger.info("No checkpoint found at %s; starting fresh.", path)
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(
            "Checkpoint loaded: %s  (epoch %d completed)",
            path, state.get("epoch", -1) + 1,
        )
        return state

    def load_best(self) -> Optional[Dict[str, Any]]:
        
        path = os.path.join(self.ckpt_dir, _BEST_CKPT)
        if not os.path.isfile(path):
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Best checkpoint loaded from: %s", path)
        return state

    def checkpoint_exists(self) -> bool:
        
        return os.path.isfile(os.path.join(self.ckpt_dir, _LAST_CKPT))
