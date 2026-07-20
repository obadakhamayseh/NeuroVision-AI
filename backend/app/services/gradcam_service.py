

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from backend.app.config.settings import settings
from ml.explainability.utils import find_target_layer, normalise_min_max

logger = logging.getLogger("brain_tumor_api")

class GradCAMService:

    def __init__(self) -> None:
        self.gradients = []
        self.activations = []

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients.append(grad)

    def _save_activation(self, act: torch.Tensor) -> None:
        self.activations.append(act)

    def generate_heatmap_and_overlay(
        self,
        model: torch.nn.Module,
        preprocessed_tensor: torch.Tensor,
        original_image_path: Path,
        target_class_idx: int,
        output_dir: Path
    ) -> Tuple[Path, Path]:
        
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = original_image_path.stem

        self.gradients.clear()
        self.activations.clear()

        target_layer = find_target_layer(model)

        def forward_hook(module, input, output):
            self._save_activation(output)

        def backward_hook(module, grad_input, grad_output):
            self._save_gradient(grad_output[0])

        f_hook = target_layer.register_forward_hook(forward_hook)
        b_hook = target_layer.register_backward_hook(backward_hook)

        model.zero_grad()
        
        with torch.set_grad_enabled(True):
            logits = model(preprocessed_tensor)
            loss = logits[0, target_class_idx]
            loss.backward()

        f_hook.remove()
        b_hook.remove()

        if not self.gradients or not self.activations:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients/activations.")

        grads = self.gradients[0].cpu().data.numpy()[0]  
        acts = self.activations[0].cpu().data.numpy()[0]  

        weights = np.mean(grads, axis=(1, 2))  

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        cam = np.maximum(cam, 0)

        orig_img = Image.open(original_image_path).convert("RGB")
        width, height = orig_img.size

        cam = cv2.resize(cam, (width, height))
        cam = normalise_min_max(cam)

        heatmap = np.uint8(255 * cam)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        orig_cv = np.array(orig_img)[:, :, ::-1].copy()

        alpha = 0.4
        overlay = cv2.addWeighted(heatmap_colored, alpha, orig_cv, 1.0 - alpha, 0)

        heatmap_path = output_dir / f"{file_stem}_heatmap.png"
        overlay_path = output_dir / f"{file_stem}_overlay.png"

        cv2.imwrite(str(heatmap_path), heatmap_colored)
        cv2.imwrite(str(overlay_path), overlay)

        logger.info("Saved Grad-CAM outputs for %s to %s", file_stem, output_dir)
        return heatmap_path, overlay_path
