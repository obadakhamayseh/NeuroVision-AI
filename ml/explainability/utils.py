

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger("brain_tumor_xai")

HeatmapArray = np.ndarray     

RGBArray = np.ndarray

def find_target_layer(model: nn.Module) -> nn.Module:

    backbone = _get_backbone(model)

    features = _get_features_module(backbone)

    target = _last_conv_block(features)

    layer_name = _module_name_in_parent(features, target)
    logger.debug(
        "Auto-detected Grad-CAM target layer: features.%s  (%s)",
        layer_name,
        type(target).__name__,
    )
    return target

def find_target_layer_by_name(model: nn.Module, layer_path: str) -> nn.Module:
    
    parts = layer_path.split(".")
    current: nn.Module = model
    for i, part in enumerate(parts):
        if not hasattr(current, part):
            resolved = ".".join(parts[:i])
            raise ValueError(
                f"Cannot resolve layer path '{layer_path}': "
                f"attribute '{part}' not found on "
                f"'{type(current).__name__}' (resolved so far: '{resolved}')."
            )
        current = getattr(current, part)
        if not isinstance(current, nn.Module):
            raise ValueError(
                f"Layer path '{layer_path}' resolves to a "
                f"non-Module object at '{part}' ({type(current).__name__})."
            )
    return current

def to_numpy_f32(tensor: torch.Tensor) -> np.ndarray:
    
    return tensor.detach().cpu().float().numpy()

def validate_batch_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    
    if tensor.dim() < 2:
        raise ValueError(
            f"Expected '{name}' to have at least 2 dimensions "
            f"(batch + spatial/channel), got shape {tuple(tensor.shape)}."
        )

def squeeze_batch(tensor: torch.Tensor) -> torch.Tensor:
    
    if tensor.shape[0] != 1:
        raise ValueError(
            f"squeeze_batch expects batch_size=1, got {tensor.shape[0]}."
        )
    return tensor.squeeze(0)

def pil_to_numpy_rgb(image: Image.Image) -> RGBArray:
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image, dtype=np.uint8)

def numpy_rgb_to_pil(array: RGBArray) -> Image.Image:
    
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected shape (H, W, 3), got {array.shape}."
        )
    arr = array.astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def tensor_to_numpy_rgb(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> RGBArray:
    
    t = tensor.detach().cpu().float()
    if t.dim() == 4:
        t = t.squeeze(0)          
    if t.dim() != 3 or t.shape[0] != 3:
        raise ValueError(
            f"Expected tensor of shape (3, H, W) or (1, 3, H, W), "
            f"got {tuple(tensor.shape)}."
        )
    
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)
    t = t * std_t + mean_t
    t = t.clamp(0.0, 1.0)
    
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr

def normalise_min_max(
    array: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    
    arr = array.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    span = max(float(hi - lo), eps)
    return (arr - lo) / span

def normalise_percentile(
    array: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    eps: float = 1e-8,
) -> np.ndarray:
    
    arr = array.astype(np.float32)
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    arr = arr.clip(lo, hi)
    return normalise_min_max(arr, eps=eps)

def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_output_path(
    output_dir: Union[str, os.PathLike],
    stem: str,
    suffix: str,
) -> Path:
    
    return ensure_dir(output_dir) / f"{stem}{suffix}"

def image_stem(image_path: Union[str, os.PathLike]) -> str:
    
    return Path(image_path).stem

@contextmanager
def timer_ms(
    device: Optional[torch.device] = None,
) -> Generator[List[float], None, None]:
    
    result: List[float] = [0.0]

    use_cuda = (
        device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    )

    if use_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        yield result
        end_evt.record()
        torch.cuda.synchronize()
        result[0] = start_evt.elapsed_time(end_evt)
    else:
        t0 = time.perf_counter()
        yield result
        result[0] = (time.perf_counter() - t0) * 1000.0

def setup_xai_logger(
    log_path: Union[str, os.PathLike],
    level: int = logging.DEBUG,
) -> logging.Logger:
    
    log_path = Path(log_path)
    ensure_dir(log_path.parent)

    xai_logger = logging.getLogger("brain_tumor_xai")
    xai_logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path_str = str(log_path)
    already_has_file = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == file_path_str
        for h in xai_logger.handlers
    )
    if not already_has_file:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        xai_logger.addHandler(fh)

    already_has_stream = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in xai_logger.handlers
    )
    if not already_has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        xai_logger.addHandler(sh)

    return xai_logger

def _get_backbone(model: nn.Module) -> nn.Module:
    
    if hasattr(model, "backbone"):
        return model.backbone          
    return model

def _get_features_module(backbone: nn.Module) -> nn.Module:
    
    if not hasattr(backbone, "features"):
        raise ValueError(
            f"Model '{type(backbone).__name__}' has no 'features' attribute. "
            "Automatic layer detection requires a torchvision-style model. "
            "Use find_target_layer_by_name() with an explicit path instead."
        )
    return backbone.features  

def _last_conv_block(features: nn.Module) -> nn.Module:
    
    children = list(features.named_children())
    for _name, block in reversed(children):
        has_conv = any(
            isinstance(m, nn.Conv2d) for m in block.modules()
        )
        if has_conv:
            return block
    raise ValueError(
        f"No convolutional block found in {type(features).__name__}. "
        "The model may use a non-standard feature extractor. "
        "Use find_target_layer_by_name() with an explicit path instead."
    )

def _module_name_in_parent(parent: nn.Module, target: nn.Module) -> str:
    
    for name, mod in parent.named_children():
        if mod is target:
            return name
    return "?"
