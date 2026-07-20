

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

logger = logging.getLogger("brain_tumor_pipeline")

class BrainTumorClassifier(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features: int = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=cfg.DROPOUT_RATE, inplace=True),
            nn.Linear(in_features=in_features, out_features=cfg.NUM_CLASSES),
        )

        logger.info(
            "BrainTumorClassifier ready | backbone=EfficientNet-B0 "
            "| in_features=%d | num_classes=%d | dropout=%.2f",
            in_features,
            cfg.NUM_CLASSES,
            cfg.DROPOUT_RATE,
        )
        self.log_parameter_summary()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        n_frozen = sum(
            1
            for n, p in self.backbone.named_parameters()
            if not p.requires_grad
        )
        logger.info("Backbone frozen: %d parameter tensors frozen.", n_frozen)
        self.log_parameter_summary()

    def unfreeze_backbone(self) -> None:
        
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Full backbone unfrozen for fine-tuning.")
        self.log_parameter_summary()

    def partial_freeze(self, unfreeze_from: str) -> None:
        
        freeze = True
        for name, param in self.backbone.named_parameters():
            if unfreeze_from in name:
                freeze = False
            param.requires_grad = not freeze

        n_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        logger.info(
            "Partial freeze applied from \'%s\': %d trainable parameters.",
            unfreeze_from,
            n_trainable,
        )
        self.log_parameter_summary()

    def count_parameters(self) -> Tuple[int, int]:
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return total, trainable

    def log_parameter_summary(self) -> None:
        
        total, trainable = self.count_parameters()
        frozen = total - trainable
        logger.info(
            "Parameters | total: %s | trainable: %s | frozen: %s",
            f"{total:,}",
            f"{trainable:,}",
            f"{frozen:,}",
        )
