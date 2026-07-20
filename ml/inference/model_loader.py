

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from inference.exceptions import DeviceError, ModelLoadError

logger = logging.getLogger("brain_tumor_inference")

class ModelLoader:

    _model: nn.Module | None = None
    _device: torch.device | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls, cfg) -> Tuple[nn.Module, torch.device]:
        
        if cls._model is not None:
            return cls._model, cls._device  

        with cls._lock:

            if cls._model is None:
                cls._model, cls._device = cls._load(cfg)

        return cls._model, cls._device  

    @classmethod
    def _load(cls, cfg) -> Tuple[nn.Module, torch.device]:
        
        device = cls._resolve_device(cfg)
        model = cls._build_architecture(cfg, device)
        cls._load_weights(model, cfg, device)
        model.eval()
        logger.info(
            "Model ready | arch=EfficientNet-B0 | device=%s | mode=eval",
            device,
        )
        return model, device

    @staticmethod
    def _resolve_device(cfg) -> torch.device:
        
        try:
            pref: str = getattr(cfg, "DEVICE", "auto")
            if pref == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device(pref)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise DeviceError(
                        f"Requested device='{pref}' but CUDA is not available."
                    )

            if device.type == "cuda":
                idx = device.index if device.index is not None else 0
                props = torch.cuda.get_device_properties(idx)
                logger.info(
                    "Inference device: %s | GPU: %s | VRAM: %.1f GB | CUDA: %s",
                    device, props.name,
                    props.total_memory / 1024 ** 3,
                    torch.version.cuda,
                )
            else:
                logger.info("Inference device: %s", device)

            return device

        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(
                f"Failed to resolve device: {exc}", cause=exc
            ) from exc

    @staticmethod
    def _build_architecture(cfg, device: torch.device) -> nn.Module:
        
        try:
            import sys, os

            project_root = Path(__file__).resolve().parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from models import BrainTumorClassifier
            model = BrainTumorClassifier(cfg).to(device)
            return model

        except Exception as exc:
            raise ModelLoadError(
                f"Failed to build model architecture: {exc}", cause=exc
            ) from exc

    @staticmethod
    def _load_weights(
        model: nn.Module, cfg, device: torch.device
    ) -> None:
        
        ckpt_path = Path(cfg.CHECKPOINTS_DIR) / "best_model.pth"

        if not ckpt_path.exists():
            raise ModelLoadError(
                f"Checkpoint not found: '{ckpt_path}'. "
                "Run training first or verify CHECKPOINTS_DIR in config.py."
            )

        logger.info("Loading checkpoint: '%s'", ckpt_path)

        try:
            checkpoint = torch.load(
                ckpt_path,
                map_location=device,
                weights_only=True,   
            )
        except Exception as exc:
            
            try:
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=device,
                )
            except Exception as exc2:
                raise ModelLoadError(
                    f"Failed to deserialise checkpoint '{ckpt_path}': {exc2}",
                    cause=exc2,
                ) from exc2

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            epoch = checkpoint.get("epoch", "?")
            val_loss = checkpoint.get("val_loss", float("nan"))
            logger.info(
                "Checkpoint metadata | epoch=%s | val_loss=%.6f", epoch, val_loss
            )
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            
            state_dict = checkpoint

        try:
            missing, unexpected = model.load_state_dict(
                state_dict, strict=True
            )
            if missing:
                logger.warning("Missing keys in checkpoint: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys in checkpoint: %s", unexpected)
        except RuntimeError as exc:
            raise ModelLoadError(
                f"State dict incompatible with model architecture: {exc}",
                cause=exc,
            ) from exc

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Weights loaded successfully | parameters=%s", f"{n_params:,}"
        )

    @classmethod
    def reset(cls) -> None:
        
        with cls._lock:
            if cls._model is not None:
                del cls._model
                if cls._device is not None and cls._device.type == "cuda":
                    torch.cuda.empty_cache()
            cls._model = None
            cls._device = None
        logger.info("ModelLoader cache cleared.")
