

import logging
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("brain_tumor_pipeline")

class OptimizerFactory:

    @staticmethod
    def build(cfg, model: nn.Module) -> optim.Optimizer:

        decay_params, no_decay_params = OptimizerFactory._split_params(model)
        param_groups = [
            {"params": decay_params,    "weight_decay": cfg.WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        name = cfg.OPTIMIZER_NAME.lower()

        if name == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=cfg.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True,
            )
        elif name == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=cfg.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif name == "sgd":
            optimizer = optim.SGD(
                param_groups,
                lr=cfg.LEARNING_RATE,
                momentum=cfg.OPTIMIZER_SGD_MOMENTUM,
                nesterov=cfg.OPTIMIZER_SGD_NESTEROV,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: '{name}\'. "
                "Supported: \'adam\', \'adamw\', \'sgd\'."
            )

        n_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            "Optimizer: %s | lr=%.2e | weight_decay=%.2e | "
            "trainable_params=%s",
            optimizer.__class__.__name__,
            cfg.LEARNING_RATE,
            cfg.WEIGHT_DECAY,
            f"{n_trainable:,}",
        )
        return optimizer

    @staticmethod
    def _split_params(model: nn.Module):
        
        no_decay_names = {"bias", "norm"}
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay_names):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay
