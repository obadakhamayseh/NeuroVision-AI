

import logging
from typing import Optional

import torch

logger = logging.getLogger("brain_tumor_pipeline")

def get_device(device_pref: str = "auto") -> torch.device:
    
    if device_pref == "auto":
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
        device = torch.device(device_pref)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "device_pref='cuda' requested but CUDA is not available."
            )

    _log_device_info(device)
    return device

def _log_device_info(device: torch.device) -> None:
    
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        logger.info(
            "Device: %s | GPU: %s | VRAM: %.1f GB | CUDA: %s",
            device,
            props.name,
            props.total_memory / 1024 ** 3,
            torch.version.cuda,
        )
    elif device.type == "mps":
        logger.info("Device: Apple Silicon MPS")
    else:
        logger.info("Device: CPU  (no GPU acceleration)")
