

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("brain_tumor_pipeline")

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)   
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        
        num_classes = logits.size(1)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        p_t = torch.exp(-ce_loss)

        focal_weight = (1.0 - p_t) ** self.gamma

        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class LossFactory:

    @staticmethod
    def build(
        cfg,
        class_weights: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        
        loss_name: str = getattr(cfg, "LOSS_NAME", "cross_entropy").lower()
        w_info = "inverse-freq" if class_weights is not None else "uniform"

        if loss_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=cfg.LABEL_SMOOTHING,
                reduction="mean",
            )
            logger.info(
                "Loss: CrossEntropyLoss | label_smoothing=%.3f | "
                "class_weights=%s",
                cfg.LABEL_SMOOTHING,
                w_info,
            )
            return criterion

        if loss_name == "focal":
            gamma = getattr(cfg, "FOCAL_LOSS_GAMMA", 2.0)
            criterion = FocalLoss(
                gamma=gamma,
                weight=class_weights,
                reduction="mean",
                label_smoothing=0.0,   
            )
            logger.info(
                "Loss: FocalLoss | gamma=%.2f | class_weights=%s",
                gamma,
                w_info,
            )
            return criterion

        raise ValueError(
            f"Unsupported loss function: '{loss_name}'. "
            "Supported: 'cross_entropy', 'focal'."
        )
