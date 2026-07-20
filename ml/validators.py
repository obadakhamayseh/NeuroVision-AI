

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from config import Config
from utils import get_logger

class TensorValidator:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self._results: List[Dict] = []

    def validate_loaders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> bool:
        
        self._results.clear()
        splits = [
            ("train", train_loader),
            ("val",   val_loader),
            ("test",  test_loader),
        ]
        all_passed = all(
            self._validate_single_loader(loader, name)
            for name, loader in splits
        )
        self._display_results()
        return all_passed

    def _validate_single_loader(
        self, loader: DataLoader, split_name: str
    ) -> bool:
        
        try:
            images, labels = next(iter(loader))
        except StopIteration:
            self.logger.error(
                "DataLoader '%s' is empty - cannot validate.", split_name
            )
            self._results.append(
                {"split": split_name, "status": "FAIL", "checks": {}}
            )
            return False
        except Exception as exc:
            self.logger.error(
                "DataLoader '%s' raised an exception: %s", split_name, exc
            )
            self._results.append(
                {"split": split_name, "status": "FAIL", "checks": {}}
            )
            return False

        exp_c = self.cfg.NUM_CHANNELS
        exp_h, exp_w = self.cfg.IMAGE_SIZE
        act_shape = list(images.shape)  

        checks: Dict[str, Dict] = {}

        checks["shape"] = {
            "expected": f"[B, {exp_c}, {exp_h}, {exp_w}]",
            "actual": str(act_shape),
            "pass": (
                len(act_shape) == 4
                and act_shape[1] == exp_c
                and act_shape[2] == exp_h
                and act_shape[3] == exp_w
            ),
        }

        checks["dtype"] = {
            "expected": "float32",
            "actual": str(images.dtype).replace("torch.", ""),
            "pass": images.dtype == torch.float32,
        }

        checks["channels"] = {
            "expected": str(exp_c),
            "actual": str(act_shape[1] if len(act_shape) >= 2 else "?"),
            "pass": len(act_shape) >= 2 and act_shape[1] == exp_c,
        }

        checks["image_size"] = {
            "expected": f"{exp_h}x{exp_w}",
            "actual": (
                f"{act_shape[2]}x{act_shape[3]}"
                if len(act_shape) == 4 else "?"
            ),
            "pass": (
                len(act_shape) == 4
                and act_shape[2] == exp_h
                and act_shape[3] == exp_w
            ),
        }

        img_min = images.min().item()
        img_max = images.max().item()
        checks["normalization"] = {
            "expected": "min in [-5, 0]  max in [0, 5]",
            "actual": f"min={img_min:.3f}  max={img_max:.3f}",
            "pass": -5.0 <= img_min <= 0.0 and 0.0 <= img_max <= 5.0,
        }

        overall = all(c["pass"] for c in checks.values())
        self._results.append(
            {
                "split": split_name,
                "status": "PASS" if overall else "FAIL",
                "checks": checks,
            }
        )
        return overall

    def _display_results(self) -> None:
        
        self.logger.info("=" * 70)
        self.logger.info("TENSOR VALIDATION RESULTS")
        self.logger.info("=" * 70)
        for r in self._results:
            self.logger.info(
                "  Split: %-6s  Status: %s",
                r["split"].upper(), r["status"],
            )
            for name, detail in r.get("checks", {}).items():
                mark = "PASS" if detail["pass"] else "FAIL"
                self.logger.info(
                    "    %-18s [%s]  expected=%-30s  actual=%s",
                    name, mark, detail["expected"], detail["actual"],
                )
        self.logger.info("=" * 70)
