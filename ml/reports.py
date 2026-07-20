

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import Config
from utils import ensure_dir, get_logger

class ReportGenerator:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger()

    def generate_duplicate_report(
        self,
        duplicates: Dict[str, List[str]],
        moved_files: Dict[str, List[str]],
        total_images: int,
    ) -> Dict[str, Any]:
        
        n_groups = len(duplicates)
        
        n_extra = sum(len(v) - 1 for v in duplicates.values())
        n_moved = sum(len(v) for v in moved_files.values())
        pct = round(100.0 * n_extra / total_images, 2) if total_images else 0.0

        groups = [
            {
                "hash": h,
                "canonical_file": paths[0],
                "duplicate_copies": paths[1:],
                "count": len(paths),
            }
            for h, paths in duplicates.items()
        ]

        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "duplicate_groups": n_groups,
            "duplicate_files": n_extra,
            "moved_files": n_moved,
            "duplicate_percentage": pct,
            "handling_policy": {
                "check": self.cfg.CHECK_DUPLICATES,
                "move": self.cfg.MOVE_DUPLICATES,
                "remove": self.cfg.REMOVE_DUPLICATES,
            },
            "moved_by_class": moved_files,
            "groups": groups,
        }

        self._write_json(report, self.cfg.DUPLICATE_REPORT_JSON)
        return report

    def generate_leakage_report(
        self,
        cross_folder_result: Dict[str, Any],
        split_leakage_result: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        
        cross_leaked = cross_folder_result.get("total_leaked", 0)
        split_leaked = sum(len(v) for v in split_leakage_result.values())
        any_leaked = cross_leaked > 0 or split_leaked > 0

        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "leakage_found": any_leaked,
            "status": "WARNING" if any_leaked else "SAFE",
            "cross_folder_check": {
                "description": (
                    "Images shared between raw Training/ and Testing/ folders"
                ),
                "leakage_found": cross_leaked > 0,
                "leaked_images": cross_leaked,
                "leaked_files": cross_folder_result.get("leaked_files", []),
            },
            "split_check": {
                "description": (
                    "Images shared between train/val/test splits after splitting"
                ),
                "leakage_found": split_leaked > 0,
                "train_val_overlap": len(
                    split_leakage_result.get("train_val", [])
                ),
                "train_test_overlap": len(
                    split_leakage_result.get("train_test", [])
                ),
                "val_test_overlap": len(
                    split_leakage_result.get("val_test", [])
                ),
            },
        }

        self._write_json(report, self.cfg.LEAKAGE_REPORT_JSON)

        if any_leaked:
            self.logger.warning("=" * 52)
            self.logger.warning("  LEAKAGE STATUS : WARNING")
            self.logger.warning(
                "  Cross-folder: %d   Split: %d", cross_leaked, split_leaked
            )
            self.logger.warning("=" * 52)
        else:
            self.logger.info("=" * 52)
            self.logger.info("  LEAKAGE STATUS : SAFE")
            self.logger.info("  No leakage detected between Training/Testing.")
            self.logger.info("=" * 52)

        return report

    def generate_dataset_summary(
        self,
        analysis_results: Dict[str, Any],
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Any,
        n_skipped: int,
        n_corrupted: int,
        n_duplicates: int,
        n_removed: int,
        class_to_idx: Dict[str, int],
    ) -> Dict[str, Any]:
        
        summary: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "dataset_version": self.cfg.DATASET_VERSION,
            "total_images": (
                len(train_dataset) + len(val_dataset) + len(test_dataset)
            ),
            "train_images": len(train_dataset),
            "validation_images": len(val_dataset),
            "test_images": len(test_dataset),
            "class_counts": analysis_results.get("class_counts", {}),
            "skipped_images": n_skipped,
            "corrupted_images": n_corrupted,
            "duplicate_images": n_duplicates,
            "removed_duplicates": n_removed,
            "image_size": list(self.cfg.IMAGE_SIZE),
            "channels": self.cfg.NUM_CHANNELS,
            "classes": sorted(class_to_idx.keys()),
            "class_to_index": class_to_idx,
            "augmentation_strategy": (
                f"Resize({self.cfg.RESIZE_BEFORE_CROP})"
                f" -> RandomCrop({self.cfg.IMAGE_SIZE})"
            ),
            "normalization_mean": list(self.cfg.IMAGENET_MEAN),
            "normalization_std": list(self.cfg.IMAGENET_STD),
            "is_balanced": not analysis_results.get("is_imbalanced", False),
            "imbalance_ratio": analysis_results.get("imbalance_ratio", 1.0),
        }

        self._write_json(summary, self.cfg.DATASET_SUMMARY_JSON)
        return summary

    def generate_health_report(
        self,
        analysis_results: Dict[str, Any],
        n_duplicates: int,
        n_corrupted: int,
        n_moved: int,
        leakage_found: bool,
        tensor_validation_passed: bool,
    ) -> str:
        
        balanced = not analysis_results.get("is_imbalanced", False)
        healthy = (
            n_corrupted == 0
            and not leakage_found
            and tensor_validation_passed
        )
        status = "Healthy" if healthy else "Unhealthy"
        leakage_status = "WARNING" if leakage_found else "None"
        dup_handling = (
            "Moved" if self.cfg.MOVE_DUPLICATES and n_moved > 0
            else ("Detected Only" if n_duplicates > 0 else "None")
        )
        ready = "YES" if healthy else "NO"

        lines = [
            "=" * 52,
            "  DATASET HEALTH REPORT",
            f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 52,
            "",
            f"  Dataset Status         : {status}",
            f"  Balanced               : {'Yes' if balanced else 'No'}",
            f"  Corrupted Images       : {n_corrupted if n_corrupted else 'None'}",
            f"  Duplicate Images       : {n_duplicates}",
            f"  Duplicate Handling     : {dup_handling}",
            f"  Train/Test Leakage     : {leakage_status}",
            f"  Tensor Validation      : {'PASS' if tensor_validation_passed else 'FAIL'}",
            f"  Ready For Training     : {ready}",
            "",
            "=" * 52,
        ]

        content = "\n".join(lines)
        ensure_dir(os.path.dirname(self.cfg.HEALTH_REPORT_TXT))
        with open(self.cfg.HEALTH_REPORT_TXT, "w", encoding="utf-8") as fh:
            fh.write(content)
        self.logger.info(
            "Health report saved -> %s", self.cfg.HEALTH_REPORT_TXT
        )
        return content

    def print_pipeline_status(self, checks: Dict[str, bool]) -> None:
        
        labels = [
            ("dataset_scanned",     "Dataset Scanned"),
            ("dataset_validated",   "Dataset Validated"),
            ("dataset_analysed",    "Dataset Analysed"),
            ("dataset_balanced",    "Dataset Balanced"),
            ("duplicate_detection", "Duplicate Detection Completed"),
            ("leakage_check",       "Data Leakage Check Completed"),
            ("reports_generated",   "Reports Generated"),
            ("figures_generated",   "Figures Generated"),
            ("dataloaders_ready",   "DataLoaders Ready"),
            ("tensor_validation",   "Tensor Validation Passed"),
            ("ready_for_training",  "Ready For Training"),
        ]
        print("\n" + "=" * 52)
        print("  PIPELINE STATUS")
        print("=" * 52)
        for key, label in labels:
            mark = "[OK]  " if checks.get(key, False) else "[FAIL]"
            print(f"  {mark}  {label}")
        print("=" * 52 + "\n")

    def _write_json(self, data: Dict[str, Any], path: str) -> None:
        
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        self.logger.info("Report saved -> %s", path)
