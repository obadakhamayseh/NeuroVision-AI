

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import Config
from utils import ensure_dir, get_logger

def _tensor_to_displayable(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:

    img = tensor.clone().detach().cpu().float()

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    img = img * std_t + mean_t

    img = img.clamp(0.0, 1.0)

    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img_np

def _safe_show_or_save(
    fig: plt.Figure,
    save_path: Optional[str],
    title: str,
    logger,
) -> None:
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Figure saved → %s", save_path)
    if matplotlib.is_interactive():
        plt.show()
    plt.close(fig)

def show_class_samples(
    valid_paths: Dict[str, List[str]],
    idx_to_class: Dict[int, str],
    class_to_idx: Dict[str, int],
    cfg: Config,
    n_per_class: int = 4,
    save_path: Optional[str] = None,
) -> None:
    
    logger = get_logger()
    classes = sorted(valid_paths.keys())
    n_classes = len(classes)

    if n_classes == 0:
        logger.warning("No classes found; cannot show class samples.")
        return

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 3, n_classes * 3),
    )
    fig.suptitle("Random Samples per Class (Original Images)", fontsize=16)

    if n_classes == 1:
        axes = [axes]

    rng = np.random.default_rng(cfg.RANDOM_SEED)

    for row_idx, class_name in enumerate(classes):
        paths = valid_paths[class_name]
        chosen = rng.choice(paths, size=min(n_per_class, len(paths)), replace=False)

        for col_idx in range(n_per_class):
            ax = axes[row_idx][col_idx] if n_per_class > 1 else axes[row_idx]
            ax.axis("off")

            if col_idx < len(chosen):
                try:
                    img = Image.open(chosen[col_idx]).convert("RGB")
                    ax.imshow(np.array(img))
                    if col_idx == 0:
                        ax.set_ylabel(class_name, fontsize=10, rotation=90)
                except Exception as exc:  
                    logger.warning("Could not display %s: %s", chosen[col_idx], exc)

    plt.tight_layout()
    path = save_path or os.path.join(cfg.LOG_DIR, "class_samples.png")
    _safe_show_or_save(fig, path, "Class Samples", logger)

def show_preprocessing(
    image_path: str,
    train_transform,
    val_transform,
    cfg: Config,
    save_path: Optional[str] = None,
) -> None:
    
    logger = get_logger()

    try:
        original = Image.open(image_path).convert("RGB")
    except Exception as exc:  
        logger.error("Cannot open '%s': %s", image_path, exc)
        return

    val_tensor = val_transform(original)
    train_tensor = train_transform(original)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Preprocessing Comparison\n{os.path.basename(image_path)}", fontsize=14
    )

    axes[0].imshow(np.array(original))
    axes[0].set_title(f"Original\n{original.size[0]}×{original.size[1]} px")
    axes[0].axis("off")

    axes[1].imshow(
        _tensor_to_displayable(val_tensor, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
    )
    axes[1].set_title(
        f"Val / Test Transform\n{cfg.IMAGE_SIZE[0]}×{cfg.IMAGE_SIZE[1]} px"
    )
    axes[1].axis("off")

    axes[2].imshow(
        _tensor_to_displayable(train_tensor, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
    )
    axes[2].set_title(
        f"Train Transform (augmented)\n{cfg.IMAGE_SIZE[0]}×{cfg.IMAGE_SIZE[1]} px"
    )
    axes[2].axis("off")

    plt.tight_layout()
    path = save_path or os.path.join(cfg.LOG_DIR, "preprocessing_comparison.png")
    _safe_show_or_save(fig, path, "Preprocessing", logger)

def show_augmentations(
    image_path: str,
    train_transform,
    cfg: Config,
    n_augmentations: int = 8,
    save_path: Optional[str] = None,
) -> None:
    
    logger = get_logger()
    try:
        original = Image.open(image_path).convert("RGB")
    except Exception as exc:  
        logger.error("Cannot open '%s': %s", image_path, exc)
        return

    n_cols = 4
    n_rows = (n_augmentations + n_cols - 1) // n_cols  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(
        f"Augmented Versions (Training Transforms)\n{os.path.basename(image_path)}",
        fontsize=14,
    )

    axes_flat = axes.flatten() if n_rows > 1 else [axes] * n_augmentations

    for i in range(n_augmentations):
        ax = axes_flat[i]
        ax.axis("off")
        aug_tensor = train_transform(original)
        ax.imshow(
            _tensor_to_displayable(aug_tensor, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
        )
        ax.set_title(f"Aug #{i + 1}", fontsize=9)

    for j in range(n_augmentations, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    path = save_path or os.path.join(cfg.LOG_DIR, "augmentation_samples.png")
    _safe_show_or_save(fig, path, "Augmentations", logger)

def show_class_distribution(
    class_counts: Dict[str, int],
    cfg: Config,
    save_path: Optional[str] = None,
) -> None:
    
    logger = get_logger()
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))  
    bars = ax.bar(classes, counts, color=colors, edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, counts):
        pct = 100 * count / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Class Distribution", fontsize=14)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.2)
    plt.tight_layout()

    path = save_path or os.path.join(cfg.LOG_DIR, "class_distribution.png")
    _safe_show_or_save(fig, path, "Class Distribution", logger)

def show_batch(
    dataloader,
    idx_to_class: Dict[int, str],
    cfg: Config,
    max_images: int = 16,
    save_path: Optional[str] = None,
) -> None:
    
    logger = get_logger()

    try:
        images, labels = next(iter(dataloader))
    except StopIteration:
        logger.warning("DataLoader is empty; nothing to visualise.")
        return

    n = min(max_images, images.size(0))
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle("DataLoader Batch Preview", fontsize=14)
    axes_flat = axes.flatten() if n_rows > 1 else [axes] * n

    for i in range(n):
        ax = axes_flat[i]
        ax.axis("off")
        img_np = _tensor_to_displayable(images[i], cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
        ax.imshow(img_np)
        class_name = idx_to_class.get(labels[i].item(), str(labels[i].item()))
        ax.set_title(class_name, fontsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    path = save_path or os.path.join(cfg.LOG_DIR, "batch_preview.png")
    _safe_show_or_save(fig, path, "Batch Preview", logger)
