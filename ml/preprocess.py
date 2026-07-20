

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from config import Config
from utils import (
    ensure_dir,
    gather_image_paths,
    get_image_metadata,
    get_logger,
    log_skipped_image,
    compute_stats,
    validate_image,
)

class DatasetScanner:

    def __init__(self, cfg: Config) -> None:
        
        self.cfg = cfg
        self.logger = get_logger()
        ensure_dir(cfg.LOG_DIR)

        self.valid_paths: Dict[str, List[str]] = {}
        self.skipped: List[Tuple[str, str]] = []
        self.class_to_idx: Dict[str, int] = {}

    def scan(
        self, root: Optional[str] = None
    ) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        
        root = root or self.cfg.TRAINING_DIR
        self.logger.info("=" * 60)
        self.logger.info("DATASET SCANNING STARTED")
        self.logger.info("Root: %s", root)
        self.logger.info("=" * 60)

        raw_paths = gather_image_paths(root, self.cfg.VALID_EXTENSIONS)
        self.valid_paths = {}
        self.skipped = []

        for class_name, paths in raw_paths.items():
            valid_for_class: List[str] = []

            for img_path in paths:
                is_ok, reason = validate_image(img_path)
                if is_ok:
                    valid_for_class.append(img_path)
                else:
                    self.skipped.append((img_path, reason))
                    log_skipped_image(self.cfg.SKIPPED_LOG, img_path, reason)
                    self.logger.warning(
                        "Skipped '%s': %s", img_path, reason
                    )

            self.valid_paths[class_name] = valid_for_class

        self.class_to_idx = {
            name: idx for idx, name in enumerate(sorted(self.valid_paths))
        }

        self._log_scan_summary()
        return self.valid_paths, self.class_to_idx

    def _log_scan_summary(self) -> None:
        
        self.logger.info("-" * 60)
        self.logger.info("SCAN SUMMARY")
        self.logger.info("-" * 60)
        total_valid = 0
        for class_name in sorted(self.valid_paths):
            n = len(self.valid_paths[class_name])
            idx = self.class_to_idx[class_name]
            self.logger.info(
                "  [%d] %-20s → %5d valid images", idx, class_name, n
            )
            total_valid += n

        self.logger.info("  %s", "-" * 40)
        self.logger.info("  TOTAL VALID           → %5d", total_valid)
        self.logger.info("  TOTAL SKIPPED         → %5d", len(self.skipped))
        self.logger.info(
            "  Skipped log           → %s", self.cfg.SKIPPED_LOG
        )
        self.logger.info("-" * 60)

class DatasetAnalyser:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger()

    def analyse(
        self, valid_paths: Dict[str, List[str]]
    ) -> Dict:
        
        self.logger.info("=" * 60)
        self.logger.info("DATASET ANALYSIS STARTED")
        self.logger.info("=" * 60)

        widths: List[float] = []
        heights: List[float] = []
        aspect_ratios: List[float] = []
        channels: List[int] = []
        extensions: List[str] = []
        resolutions: List[Tuple[int, int]] = []
        class_counts: Dict[str, int] = {}
        failed_metadata = 0

        for class_name, paths in valid_paths.items():
            class_counts[class_name] = len(paths)
            for img_path in paths:
                meta = get_image_metadata(img_path)
                if meta is None:
                    failed_metadata += 1
                    continue
                widths.append(meta["width"])
                heights.append(meta["height"])
                aspect_ratios.append(meta["aspect_ratio"])
                channels.append(meta["channels"])
                extensions.append(meta["extension"])
                resolutions.append((meta["width"], meta["height"]))

        width_stats = compute_stats(widths)
        height_stats = compute_stats(heights)
        ar_stats = compute_stats(aspect_ratios)

        channel_dist = dict(Counter(channels))
        ext_dist = dict(Counter(extensions))
        res_dist = Counter(resolutions).most_common(10)  

        counts = list(class_counts.values())
        imbalance_ratio = (
            round(max(counts) / min(counts), 4) if min(counts) > 0 else float("inf")
        )
        is_imbalanced = imbalance_ratio > 1.5  

        results = {
            "class_counts": class_counts,
            "total_images": sum(counts),
            "width_stats": width_stats,
            "height_stats": height_stats,
            "aspect_ratio_stats": ar_stats,
            "channel_distribution": channel_dist,
            "extension_distribution": ext_dist,
            "top_resolutions": res_dist,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": is_imbalanced,
            "failed_metadata": failed_metadata,
        }

        self._log_results(results)
        self._save_report(results)
        return results

    def _log_results(self, r: Dict) -> None:
        
        self.logger.info("-" * 60)
        self.logger.info("CLASS DISTRIBUTION")
        for cls, cnt in sorted(r["class_counts"].items()):
            pct = 100 * cnt / r["total_images"] if r["total_images"] else 0
            self.logger.info("  %-20s : %5d  (%5.1f%%)", cls, cnt, pct)
        self.logger.info("  TOTAL               : %5d", r["total_images"])

        self.logger.info("-" * 60)
        self.logger.info("IMAGE DIMENSIONS  (pixels)")
        self._log_stat_block("Width", r["width_stats"])
        self._log_stat_block("Height", r["height_stats"])
        self._log_stat_block("Aspect Ratio", r["aspect_ratio_stats"])

        self.logger.info("-" * 60)
        self.logger.info("CHANNEL DISTRIBUTION : %s", r["channel_distribution"])
        self.logger.info("EXTENSION DISTRIBUTION: %s", r["extension_distribution"])
        self.logger.info("TOP 10 RESOLUTIONS   :")
        for res, cnt in r["top_resolutions"]:
            self.logger.info("  %s × %s : %d images", res[0], res[1], cnt)

        self.logger.info("-" * 60)
        self.logger.info(
            "IMBALANCE RATIO : %.4f  (%s)",
            r["imbalance_ratio"],
            "⚠️  IMBALANCED" if r["is_imbalanced"] else "✅  BALANCED",
        )
        if r["is_imbalanced"]:
            self.logger.warning(
                "Dataset is imbalanced (ratio %.2f > 1.5). "
                "Consider using WeightedRandomSampler or focal loss "
                "during training.",
                r["imbalance_ratio"],
            )
        self.logger.info("-" * 60)

    def _log_stat_block(self, label: str, stats: Dict) -> None:
        if not stats:
            return
        self.logger.info(
            "  %-14s  min=%6.1f  max=%6.1f  mean=%6.1f  "
            "std=%5.1f  p25=%6.1f  p75=%6.1f",
            label,
            stats["min"], stats["max"], stats["mean"],
            stats["std"], stats["p25"], stats["p75"],
        )

    def _save_report(self, r: Dict) -> None:
        
        ensure_dir(self.cfg.LOG_DIR)
        lines = [
            "=" * 70,
            "  BRAIN TUMOR MRI DATASET ANALYSIS REPORT",
            "=" * 70,
            "",
            "CLASS DISTRIBUTION",
            "-" * 40,
        ]
        for cls, cnt in sorted(r["class_counts"].items()):
            pct = 100 * cnt / r["total_images"] if r["total_images"] else 0
            lines.append(f"  {cls:<20} : {cnt:>5}  ({pct:5.1f}%)")
        lines += [
            f"  {'TOTAL':<20} : {r['total_images']:>5}",
            "",
            "IMAGE DIMENSION STATISTICS (pixels)",
            "-" * 40,
        ]
        for label, stats in [
            ("Width", r["width_stats"]),
            ("Height", r["height_stats"]),
            ("Aspect Ratio", r["aspect_ratio_stats"]),
        ]:
            if stats:
                lines.append(
                    f"  {label:<14}  min={stats['min']:.1f}  "
                    f"max={stats['max']:.1f}  mean={stats['mean']:.1f}  "
                    f"std={stats['std']:.1f}"
                )
        lines += [
            "",
            "CHANNEL DISTRIBUTION",
            "-" * 40,
            f"  {r['channel_distribution']}",
            "",
            "FILE EXTENSION DISTRIBUTION",
            "-" * 40,
            f"  {r['extension_distribution']}",
            "",
            "TOP 10 RESOLUTIONS",
            "-" * 40,
        ]
        for res, cnt in r["top_resolutions"]:
            lines.append(f"  {res[0]}×{res[1]:>4}  :  {cnt} images")
        lines += [
            "",
            "IMBALANCE",
            "-" * 40,
            f"  Ratio : {r['imbalance_ratio']:.4f}",
            f"  Status: {'IMBALANCED ⚠️' if r['is_imbalanced'] else 'BALANCED ✅'}",
            "",
            "=" * 70,
        ]
        with open(self.cfg.ANALYSIS_REPORT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        self.logger.info("Analysis report saved → %s", self.cfg.ANALYSIS_REPORT)
        self._save_report_json(r)

    def _save_report_json(self, r: Dict) -> None:
        
        ensure_dir(self.cfg.LOG_DIR)

        top_res_serialisable = [
            {"width": res[0], "height": res[1], "count": cnt}
            for res, cnt in r["top_resolutions"]
        ]

        channel_dist_str = {str(k): v for k, v in r["channel_distribution"].items()}

        json_payload = {
            "report_generated_at": datetime.now().isoformat(),
            "class_counts": r["class_counts"],
            "total_images": r["total_images"],
            "width_stats": r["width_stats"],
            "height_stats": r["height_stats"],
            "aspect_ratio_stats": r["aspect_ratio_stats"],
            "channel_distribution": channel_dist_str,
            "extension_distribution": r["extension_distribution"],
            "top_resolutions": top_res_serialisable,
            "imbalance_ratio": r["imbalance_ratio"],
            "is_imbalanced": r["is_imbalanced"],
            "failed_metadata": r["failed_metadata"],
        }

        with open(self.cfg.ANALYSIS_REPORT_JSON, "w", encoding="utf-8") as fh:
            json.dump(json_payload, fh, indent=2)
        self.logger.info(
            "JSON report saved → %s", self.cfg.ANALYSIS_REPORT_JSON
        )

class TransformBuilder:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger()

    def build_train(self) -> T.Compose:
        
        transforms: List = []

        transforms.append(
            T.Resize(
                self.cfg.IMAGE_SIZE,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,   
            )
        )
        self.logger.debug("Transform: Resize → %s", self.cfg.IMAGE_SIZE)

        transforms.append(T.Lambda(lambda img: img.convert("RGB")))
        self.logger.debug("Transform: Convert → RGB")

        if self.cfg.AUG_HORIZONTAL_FLIP:

            transforms.append(
                T.RandomHorizontalFlip(p=self.cfg.AUG_HORIZONTAL_FLIP_P)
            )
            self.logger.debug(
                "Transform: RandomHorizontalFlip (p=%.2f)",
                self.cfg.AUG_HORIZONTAL_FLIP_P,
            )

        if self.cfg.AUG_ROTATION:

            transforms.append(
                T.RandomRotation(
                    degrees=self.cfg.AUG_ROTATION_DEGREES,
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0,
                )
            )
            self.logger.debug(
                "Transform: RandomRotation (±%.1f°)",
                self.cfg.AUG_ROTATION_DEGREES,
            )

        if self.cfg.AUG_AFFINE:

            transforms.append(
                T.RandomAffine(
                    degrees=0,  
                    translate=self.cfg.AUG_AFFINE_TRANSLATE,
                    shear=self.cfg.AUG_AFFINE_SHEAR,
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0,
                )
            )
            self.logger.debug(
                "Transform: RandomAffine (translate=%s, shear=%.1f°)",
                self.cfg.AUG_AFFINE_TRANSLATE,
                self.cfg.AUG_AFFINE_SHEAR,
            )

        if self.cfg.AUG_RANDOM_CROP:

            transforms.append(
                T.Resize(
                    self.cfg.RESIZE_BEFORE_CROP,
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
            transforms.append(
                T.RandomCrop(size=self.cfg.IMAGE_SIZE)
            )
            self.logger.debug(
                "Transform: Resize(%d) → RandomCrop(%s)  [replaces RandomResizedCrop]",
                self.cfg.RESIZE_BEFORE_CROP,
                self.cfg.IMAGE_SIZE,
            )

        if self.cfg.AUG_BRIGHTNESS_CONTRAST:

            transforms.append(
                T.ColorJitter(
                    brightness=self.cfg.AUG_BRIGHTNESS_FACTOR,
                    contrast=self.cfg.AUG_CONTRAST_FACTOR,
                    saturation=0,   
                    hue=0,          
                )
            )
            self.logger.debug(
                "Transform: ColorJitter brightness=%.2f, contrast=%.2f",
                self.cfg.AUG_BRIGHTNESS_FACTOR,
                self.cfg.AUG_CONTRAST_FACTOR,
            )

        transforms.append(T.ToTensor())
        self.logger.debug("Transform: ToTensor → [0, 1] float32")

        transforms.append(
            T.Normalize(
                mean=list(self.cfg.IMAGENET_MEAN),
                std=list(self.cfg.IMAGENET_STD),
            )
        )
        self.logger.debug(
            "Transform: Normalize mean=%s, std=%s",
            self.cfg.IMAGENET_MEAN,
            self.cfg.IMAGENET_STD,
        )

        if self.cfg.AUG_GAUSSIAN_NOISE:

            transforms.append(
                _GaussianNoiseTensor(
                    mean=self.cfg.AUG_GAUSSIAN_NOISE_MEAN,
                    std=self.cfg.AUG_GAUSSIAN_NOISE_STD,
                )
            )
            self.logger.debug(
                "Transform: GaussianNoise (mean=%.3f, std=%.3f)",
                self.cfg.AUG_GAUSSIAN_NOISE_MEAN,
                self.cfg.AUG_GAUSSIAN_NOISE_STD,
            )

        pipeline = T.Compose(transforms)
        self.logger.info(
            "Training transform pipeline built (%d steps).", len(transforms)
        )
        return pipeline

    def build_val_test(self) -> T.Compose:
        
        pipeline = T.Compose([
            
            T.Resize(
                self.cfg.IMAGE_SIZE,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            
            T.Lambda(lambda img: img.convert("RGB")),
            
            T.ToTensor(),
            
            T.Normalize(
                mean=list(self.cfg.IMAGENET_MEAN),
                std=list(self.cfg.IMAGENET_STD),
            ),
        ])
        self.logger.info("Validation/Test transform pipeline built (4 steps).")
        return pipeline

class _GaussianNoiseTensor(torch.nn.Module):

    def __init__(self, mean: float = 0.0, std: float = 0.02) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        
        if self.std == 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
