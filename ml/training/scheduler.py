

import logging
from typing import Tuple

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

logger = logging.getLogger("brain_tumor_pipeline")

_Scheduler = object  

class SchedulerFactory:

    @staticmethod
    def build(
        cfg,
        optimizer: optim.Optimizer,
    ) -> Tuple[_Scheduler, bool]:
        
        name = cfg.SCHEDULER_NAME.lower()

        if name == "reduce_lr_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.SCHEDULER_REDUCE_LR_FACTOR,
                patience=cfg.SCHEDULER_REDUCE_LR_PATIENCE,
                min_lr=cfg.SCHEDULER_REDUCE_LR_MIN_LR,
            )
            logger.info(
                "Scheduler: ReduceLROnPlateau | factor=%.2f | "
                "patience=%d | min_lr=%.2e",
                cfg.SCHEDULER_REDUCE_LR_FACTOR,
                cfg.SCHEDULER_REDUCE_LR_PATIENCE,
                cfg.SCHEDULER_REDUCE_LR_MIN_LR,
            )
            return scheduler, True

        if name == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.SCHEDULER_COSINE_T_MAX,
                eta_min=cfg.SCHEDULER_REDUCE_LR_MIN_LR,
            )
            logger.info(
                "Scheduler: CosineAnnealingLR | T_max=%d | eta_min=%.2e",
                cfg.SCHEDULER_COSINE_T_MAX,
                cfg.SCHEDULER_REDUCE_LR_MIN_LR,
            )
            return scheduler, False

        if name == "step_lr":
            scheduler = StepLR(
                optimizer,
                step_size=cfg.SCHEDULER_STEP_LR_STEP_SIZE,
                gamma=cfg.SCHEDULER_STEP_LR_GAMMA,
            )
            logger.info(
                "Scheduler: StepLR | step_size=%d | gamma=%.2f",
                cfg.SCHEDULER_STEP_LR_STEP_SIZE,
                cfg.SCHEDULER_STEP_LR_GAMMA,
            )
            return scheduler, False

        raise ValueError(
            f"Unsupported scheduler: '{name}\'. "
            "Supported: \'reduce_lr_on_plateau\', "
            "\'cosine_annealing\', \'step_lr\'."
        )
