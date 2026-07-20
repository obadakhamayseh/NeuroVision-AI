

import os
import shutil
from pathlib import Path
from typing import Dict, List

from config import Config
from utils import ensure_dir, get_logger

class ArtifactsManager:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger()

    def setup_directories(self) -> None:
        
        dirs = [
            self.cfg.ARTIFACTS_DIR,
            self.cfg.ARTIFACTS_REPORTS_DIR,
            self.cfg.ARTIFACTS_FIGURES_DIR,
            self.cfg.ARTIFACTS_DUPLICATES_DIR,
            self.cfg.ARTIFACTS_LOGS_DIR,
            self.cfg.ARTIFACTS_METADATA_DIR,
        ]
        for d in dirs:
            ensure_dir(d)
        self.logger.info(
            "Artifact directory tree ready: %s", self.cfg.ARTIFACTS_DIR
        )

    def move_duplicate_files(
        self,
        duplicates: Dict[str, List[str]],
        valid_paths: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        
        if not self.cfg.MOVE_DUPLICATES:
            self.logger.info(
                "MOVE_DUPLICATES=False - duplicate files kept in place."
            )
            return {}

        if not duplicates:
            self.logger.info("No duplicate files to move.")
            return {}

        path_to_class: Dict[str, str] = {
            p: cls
            for cls, paths in valid_paths.items()
            for p in paths
        }

        moved: Dict[str, List[str]] = {}
        total_moved = 0

        for hash_val, paths in duplicates.items():
            
            for dup_path in paths[1:]:
                if not os.path.isfile(dup_path):
                    continue  

                class_name = path_to_class.get(dup_path, "unknown")
                dest_dir = os.path.join(
                    self.cfg.ARTIFACTS_DUPLICATES_DIR, class_name
                )
                ensure_dir(dest_dir)

                dest_path = os.path.join(dest_dir, os.path.basename(dup_path))

                if os.path.exists(dest_path):
                    stem = Path(dest_path).stem
                    ext = Path(dest_path).suffix
                    dest_path = os.path.join(
                        dest_dir, f"{stem}_dup{hash_val[:8]}{ext}"
                    )

                try:
                    shutil.move(dup_path, dest_path)
                    moved.setdefault(class_name, []).append(dest_path)
                    total_moved += 1
                    self.logger.debug(
                        "Moved duplicate: %s -> %s", dup_path, dest_path
                    )
                except OSError as exc:
                    self.logger.error(
                        "Failed to move duplicate '%s': %s", dup_path, exc
                    )

        self.logger.info(
            "Moved %d duplicate file(s) to artifacts/duplicates/", total_moved
        )
        return moved

    def collect_figures(self) -> List[str]:
        
        figure_extensions = {".png", ".jpg", ".jpeg"}
        copied: List[str] = []

        if not os.path.isdir(self.cfg.LOG_DIR):
            self.logger.warning(
                "Log directory not found; skipping figure collection."
            )
            return copied

        for fname in sorted(os.listdir(self.cfg.LOG_DIR)):
            if Path(fname).suffix.lower() in figure_extensions:
                src = os.path.join(self.cfg.LOG_DIR, fname)
                dst = os.path.join(self.cfg.ARTIFACTS_FIGURES_DIR, fname)
                try:
                    shutil.copy2(src, dst)
                    copied.append(dst)
                    self.logger.debug("Copied figure: %s", fname)
                except OSError as exc:
                    self.logger.warning(
                        "Could not copy figure '%s': %s", fname, exc
                    )

        self.logger.info(
            "Copied %d figure(s) to artifacts/figures/", len(copied)
        )
        return copied

    def collect_logs(self) -> None:
        
        if not os.path.isdir(self.cfg.LOG_DIR):
            return

        n_copied = 0
        for fname in os.listdir(self.cfg.LOG_DIR):
            if Path(fname).suffix.lower() == ".log":
                src = os.path.join(self.cfg.LOG_DIR, fname)
                dst = os.path.join(self.cfg.ARTIFACTS_LOGS_DIR, fname)
                try:
                    shutil.copy2(src, dst)
                    n_copied += 1
                    self.logger.debug("Copied log: %s", fname)
                except OSError as exc:
                    self.logger.warning(
                        "Could not copy log '%s': %s", fname, exc
                    )

        self.logger.info(
            "Copied %d log file(s) to artifacts/logs/", n_copied
        )
