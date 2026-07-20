

import os
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from config import Config
from utils import compute_file_hash, get_logger, validate_image

def deduplicate_valid_paths(
    valid_paths: Dict[str, List[str]],
    algorithm: str = "md5",
) -> Tuple[Dict[str, List[str]], int]:
    
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("PRE-SPLIT DEDUPLICATION  (algorithm: %s)", algorithm.upper())
    logger.info("=" * 60)

    deduped: Dict[str, List[str]] = {}
    total_removed = 0

    for class_name, paths in valid_paths.items():
        seen_hashes: Dict[str, str] = {}   
        kept: List[str] = []
        removed: List[str] = []

        for path in sorted(paths):          
            digest = compute_file_hash(path, algorithm)
            if digest is None:
                kept.append(path)           
                continue
            if digest not in seen_hashes:
                seen_hashes[digest] = path
                kept.append(path)
            else:
                removed.append(path)
                logger.warning(
                    "Duplicate removed: '%s' is identical to '%s'",
                    path, seen_hashes[digest],
                )

        deduped[class_name] = kept
        total_removed += len(removed)
        logger.info(
            "  %-20s  kept=%4d  removed=%4d",
            class_name, len(kept), len(removed),
        )

    logger.info(
        "Deduplication complete: %d duplicate file(s) removed.", total_removed
    )
    return deduped, total_removed

class BrainTumorDataset(Dataset):

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        split: str = "unknown",
    ) -> None:
        super().__init__()
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.split = split
        self.logger = get_logger()

        self.logger.info(
            "BrainTumorDataset [%s] created: %d samples, %d classes.",
            self.split,
            len(self.samples),
            len(self.class_to_idx),
        )

    def __len__(self) -> int:
        
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path)

            image = image.convert("RGB")

        except Exception as exc:  
            self.logger.error(
                "Failed to load image at index %d ('%s'): %s",
                idx, img_path, exc,
            )
            raise RuntimeError(
                f"Cannot read image '{img_path}': {exc}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        if isinstance(image, torch.Tensor) and image.dtype != torch.float32:
            image = image.float()

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        
        n_classes = len(self.class_to_idx)
        counts = [0] * n_classes
        for _, label in self.samples:
            counts[label] += 1

        total = len(self.samples)
        weights = [
            total / (n_classes * count) if count > 0 else 0.0
            for count in counts
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        
        class_weights = self.get_class_weights()
        return torch.tensor(
            [class_weights[label].item() for _, label in self.samples],
            dtype=torch.float32,
        )

    def get_class_distribution(self) -> Dict[str, int]:
        
        dist: Dict[str, int] = defaultdict(int)
        for _, label in self.samples:
            dist[self.idx_to_class[label]] += 1
        return dict(dist)

    def __repr__(self) -> str:
        return (
            f"BrainTumorDataset("
            f"split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"n_classes={len(self.class_to_idx)}, "
            f"transform={'yes' if self.transform else 'no'})"
        )

class DatasetSplitter:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger()

    def split(
        self,
        valid_paths: Dict[str, List[str]],
        class_to_idx: Dict[str, int],
    ) -> Tuple[
        List[Tuple[str, int]],
        List[Tuple[str, int]],
        List[Tuple[str, int]],
    ]:
        
        self.logger.info("=" * 60)
        self.logger.info("STRATIFIED DATASET SPLITTING")
        self.logger.info(
            "Ratios → train=%.2f  val=%.2f  test=%.2f",
            self.cfg.TRAIN_RATIO,
            self.cfg.VAL_RATIO,
            self.cfg.TEST_RATIO,
        )
        self.logger.info("=" * 60)

        rng = random.Random(self.cfg.RANDOM_SEED)

        valid_paths, n_removed = deduplicate_valid_paths(
            valid_paths, self.cfg.DUPLICATE_HASH_ALGO
        )
        if n_removed:
            self.logger.warning(
                "%d duplicate file(s) removed before splitting.", n_removed
            )

        train_samples: List[Tuple[str, int]] = []
        val_samples: List[Tuple[str, int]] = []
        test_samples: List[Tuple[str, int]] = []

        for class_name in sorted(valid_paths.keys()):
            paths = list(valid_paths[class_name])  
            label = class_to_idx[class_name]

            rng.shuffle(paths)

            n = len(paths)
            n_train = int(n * self.cfg.TRAIN_RATIO)
            n_val = int(n * self.cfg.VAL_RATIO)
            
            n_test = n - n_train - n_val

            cls_train = [(p, label) for p in paths[:n_train]]
            cls_val = [(p, label) for p in paths[n_train : n_train + n_val]]
            cls_test = [(p, label) for p in paths[n_train + n_val :]]

            train_samples.extend(cls_train)
            val_samples.extend(cls_val)
            test_samples.extend(cls_test)

            self.logger.info(
                "  %-20s → train=%4d  val=%4d  test=%4d  (total=%d)",
                class_name, len(cls_train), len(cls_val), len(cls_test), n,
            )

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        rng.shuffle(test_samples)

        self.logger.info("-" * 60)
        self.logger.info(
            "  TOTALS → train=%d  val=%d  test=%d",
            len(train_samples), len(val_samples), len(test_samples),
        )
        self.logger.info("-" * 60)

        return train_samples, val_samples, test_samples

    def split_with_prebuilt_test(
        self,
        train_valid_paths: Dict[str, List[str]],
        test_valid_paths: Dict[str, List[str]],
        class_to_idx: Dict[str, int],
    ) -> Tuple[
        List[Tuple[str, int]],
        List[Tuple[str, int]],
        List[Tuple[str, int]],
    ]:
        
        self.logger.info(
            "Using pre-built test set from Testing directory."
        )
        
        orig_train = self.cfg.TRAIN_RATIO
        orig_val = self.cfg.VAL_RATIO
        total_tv = orig_train + orig_val
        train_ratio = orig_train / total_tv if total_tv > 0 else 0.8
        val_ratio = orig_val / total_tv if total_tv > 0 else 0.2

        train_valid_paths, n_removed = deduplicate_valid_paths(
            train_valid_paths, self.cfg.DUPLICATE_HASH_ALGO
        )
        if n_removed:
            self.logger.warning(
                "%d Training duplicate(s) removed before train/val split.", n_removed
            )

        rng = random.Random(self.cfg.RANDOM_SEED)
        train_samples: List[Tuple[str, int]] = []
        val_samples: List[Tuple[str, int]] = []

        for class_name in sorted(train_valid_paths.keys()):
            paths = list(train_valid_paths[class_name])
            label = class_to_idx[class_name]
            rng.shuffle(paths)

            n = len(paths)
            n_train = int(n * train_ratio)

            cls_train = [(p, label) for p in paths[:n_train]]
            cls_val = [(p, label) for p in paths[n_train:]]
            train_samples.extend(cls_train)
            val_samples.extend(cls_val)

            self.logger.info(
                "  Train class %-20s → train=%4d  val=%4d",
                class_name, len(cls_train), len(cls_val),
            )

        test_samples: List[Tuple[str, int]] = []
        for class_name in sorted(test_valid_paths.keys()):
            label = class_to_idx.get(class_name, -1)
            for p in test_valid_paths[class_name]:
                test_samples.append((p, label))

        self.logger.info(
            "Pre-built test set: %d samples.", len(test_samples)
        )

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)

        return train_samples, val_samples, test_samples
