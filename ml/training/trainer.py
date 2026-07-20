

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from device import get_device
from seed import set_seed
from training.callbacks import EarlyStopping
from training.checkpoint import CheckpointManager
from training.history import TrainingHistory
from training.losses import LossFactory
from training.metrics import MetricsCalculator
from training.optimizer import OptimizerFactory
from training.scheduler import SchedulerFactory

logger = logging.getLogger("brain_tumor_pipeline")

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        cfg,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.class_names  = class_names

        self.device = get_device(cfg.DEVICE)

        self.model = model.to(self.device)

        self._use_amp: bool = cfg.USE_AMP and self.device.type == "cuda"
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        self.criterion = LossFactory.build(cfg, class_weights=class_weights)
        self.criterion  = self.criterion.to(self.device)
        self.optimizer  = OptimizerFactory.build(cfg, self.model)
        self.scheduler, self._sched_needs_metric = SchedulerFactory.build(
            cfg, self.optimizer
        )

        self.metrics_calc = MetricsCalculator(
            num_classes=cfg.NUM_CLASSES,
            class_names=class_names,
        )

        self.history       = TrainingHistory(cfg)
        self.ckpt_mgr      = CheckpointManager(cfg)
        self.early_stopping = EarlyStopping(
            patience=cfg.EARLY_STOPPING_PATIENCE,
            monitor=cfg.EARLY_STOPPING_MONITOR,
            mode="min" if "loss" in cfg.EARLY_STOPPING_MONITOR else "max",
            min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
        )

        self.start_epoch:   int   = 0
        self.best_val_loss: float = float("inf")

        if cfg.RESUME_TRAINING and self.ckpt_mgr.checkpoint_exists():
            self._resume_from_checkpoint()

        set_seed(cfg.RANDOM_SEED, deterministic=True)

        logger.info(
            "Trainer ready | device=%s | AMP=%s | acc_steps=%d",
            self.device, self._use_amp, cfg.GRADIENT_ACCUMULATION_STEPS,
        )

    def train(self) -> TrainingHistory:
        
        logger.info(
            "Training started | epochs=%d | train_batches=%d | val_batches=%d",
            self.cfg.EPOCHS,
            len(self.train_loader),
            len(self.val_loader),
        )

        last_val_cm = None

        for epoch in range(self.start_epoch, self.cfg.EPOCHS):
            epoch_start = time.time()

            train_metrics = self._train_epoch(epoch)

            val_metrics, val_cm = self._val_epoch(epoch)
            last_val_cm = val_cm

            current_lr = self._step_scheduler(val_metrics["loss"])

            epoch_time = time.time() - epoch_start

            self.history.update(
                epoch, train_metrics, val_metrics, current_lr, epoch_time
            )
            self.history.save()

            self._log_epoch_summary(
                epoch, train_metrics, val_metrics, current_lr, epoch_time
            )

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            self.ckpt_mgr.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                val_metrics=val_metrics,
                history=self.history,
                best_val_loss=self.best_val_loss,
                is_best=is_best,
            )

            if self.early_stopping(val_metrics):
                logger.info(
                    "Early stopping triggered after epoch %d.", epoch + 1
                )
                break

        logger.info(
            "Training complete | best_val_loss=%.6f | epochs_run=%d",
            self.best_val_loss,
            len(self.history.epochs),
        )

        self.history.plot(
            confusion_matrix=last_val_cm,
            class_names=self.class_names,
        )

        return self.history

    def _train_epoch(self, epoch: int) -> Dict:
        
        self.model.train()
        running_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1:03d}/{self.cfg.EPOCHS} [Train]",
            leave=False,
            unit="batch",
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self._use_amp,
            ):
                logits = self.model(images)
                logits = logits.float()
                logits = torch.clamp(logits, min=-75.0, max=75.0)
                loss   = self.criterion(logits, labels)

            scaled_loss = loss / self.cfg.GRADIENT_ACCUMULATION_STEPS
            self._scaler.scale(scaled_loss).backward()

            if step % self.cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.GRADIENT_CLIP_MAX_NORM,
                )
                self._scaler.step(self.optimizer)
                self._scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            preds = logits.detach().argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix(
                loss=f"{running_loss / step:.4f}",
                acc=f"{sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels):.3f}",
            )

        remainder = len(self.train_loader) % self.cfg.GRADIENT_ACCUMULATION_STEPS
        if remainder != 0:
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.GRADIENT_CLIP_MAX_NORM
            )
            self._scaler.step(self.optimizer)
            self._scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss = running_loss / len(self.train_loader)
        return self.metrics_calc.compute(all_labels, all_preds, avg_loss)

    def _val_epoch(self, epoch: int) -> Tuple[Dict, object]:
        
        self.model.eval()
        running_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1:03d}/{self.cfg.EPOCHS} [Val  ]",
            leave=False,
            unit="batch",
        )

        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self._use_amp,
                ):
                    logits = self.model(images)
                    logits = logits.float()
                    logits = torch.clamp(logits, min=-75.0, max=75.0)
                    loss   = self.criterion(logits, labels)

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.4f}")

        avg_loss = running_loss / len(self.val_loader)
        metrics  = self.metrics_calc.compute(all_labels, all_preds, avg_loss)
        cm       = metrics.pop("confusion_matrix", None)
        return metrics, cm

    def _step_scheduler(self, val_loss: float) -> float:
        
        if self._sched_needs_metric:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    def _log_epoch_summary(
        self,
        epoch: int,
        train_m: Dict,
        val_m: Dict,
        lr: float,
        t: float,
    ) -> None:
        logger.info(
            "Epoch %03d/%d | "
            "loss: %.4f/%.4f | "
            "acc: %.3f/%.3f | "
            "f1: %.3f/%.3f | "
            "lr: %.2e | "
            "time: %.1fs",
            epoch + 1, self.cfg.EPOCHS,
            train_m["loss"],          val_m["loss"],
            train_m["accuracy"],      val_m["accuracy"],
            train_m.get("macro_f1", 0), val_m.get("macro_f1", 0),
            lr, t,
        )

    def _resume_from_checkpoint(self) -> None:
        
        state = self.ckpt_mgr.load_latest()
        if state is None:
            return
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            self.scheduler.load_state_dict(state["scheduler_state"])
        self.start_epoch    = state["epoch"] + 1
        self.best_val_loss  = state.get("best_val_loss", float("inf"))
        history_data        = state.get("history", {})
        self.history.epochs = history_data.get("epochs", [])
        logger.info(
            "Resumed from epoch %d | best_val_loss=%.6f",
            self.start_epoch, self.best_val_loss,
        )
