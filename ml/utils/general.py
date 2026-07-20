

import io
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

def setup_logging(
    log_dir: str,
    log_level: str = "INFO",
    log_filename: str = "pipeline.log",
) -> logging.Logger:

    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_filename)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger("brain_tumor_pipeline")
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    try:
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        
        utf8_stdout = sys.stdout

    stream_handler = logging.StreamHandler(utf8_stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logger initialised → %s", log_path)
    return logger

def get_logger(name: str = "brain_tumor_pipeline") -> logging.Logger:
    
    return logging.getLogger(name)

def ensure_dir(path: str) -> None:
    
    os.makedirs(path, exist_ok=True)

def gather_image_paths(
    root: str,
    valid_extensions: List[str],
) -> Dict[str, List[str]]:
    
    logger = get_logger()
    class_paths: Dict[str, List[str]] = {}

    if not os.path.isdir(root):
        logger.error("Dataset root does not exist: %s", root)
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for class_dir in sorted(os.scandir(root), key=lambda e: e.name):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        image_list: List[str] = []

        for dirpath, _, filenames in os.walk(class_dir.path):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in valid_extensions:
                    image_list.append(os.path.join(dirpath, fname))

        class_paths[class_name] = image_list
        logger.info(
            "Class '%s': found %d image(s).", class_name, len(image_list)
        )

    return class_paths

def validate_image(image_path: str) -> Tuple[bool, str]:

    if not os.path.exists(image_path):
        return False, "File does not exist"

    if os.path.getsize(image_path) == 0:
        return False, "File is empty (0 bytes)"

    try:
        
        img = Image.open(image_path)

        img.verify()  

        img = Image.open(image_path)

        img.load()

        w, h = img.size
        if w == 0 or h == 0:
            return False, f"Image has zero dimension ({w}×{h})"

    except UnidentifiedImageError:
        return False, "Unrecognised / unsupported image format"
    except OSError as exc:
        
        return False, f"OSError while reading image: {exc}"
    except Exception as exc:  
        return False, f"Unexpected validation error: {exc}"

    return True, ""

def log_skipped_image(log_path: str, image_path: str, reason: str) -> None:
    
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{image_path}  |  {reason}\n")

def get_image_metadata(image_path: str) -> Optional[Dict]:
    
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            mode = img.mode

            mode_to_channels = {
                "1": 1, "L": 1, "P": 1,
                "RGB": 3, "RGBA": 4, "CMYK": 4,
                "YCbCr": 3, "LAB": 3, "HSV": 3,
                "I": 1, "F": 1,
            }
            channels = mode_to_channels.get(mode, -1)
            aspect = round(w / h, 4) if h > 0 else 0.0
            size_kb = os.path.getsize(image_path) / 1024.0
            ext = Path(image_path).suffix.lower()

        return {
            "width": w,
            "height": h,
            "channels": channels,
            "mode": mode,
            "extension": ext,
            "aspect_ratio": aspect,
            "file_size_kb": round(size_kb, 2),
        }
    except Exception:  
        return None

def compute_stats(values: List[float]) -> Dict[str, float]:
    
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }

def set_global_seed(seed: int) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  

    get_logger().info("Global random seed set to %d.", seed)

def compute_file_hash(file_path: str, algorithm: str = "md5") -> Optional[str]:
    
    import hashlib
    h = hashlib.new(algorithm)
    try:
        
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None

def detect_duplicates(
    valid_paths: Dict[str, List[str]],
    algorithm: str = "md5",
) -> Dict[str, List[str]]:
    
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("DUPLICATE DETECTION STARTED  (algorithm: %s)", algorithm.upper())
    logger.info("=" * 60)

    all_paths: List[str] = []
    for paths in valid_paths.values():
        all_paths.extend(paths)

    total = len(all_paths)
    logger.info("Hashing %d images …", total)

    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    failed = 0

    for i, img_path in enumerate(all_paths, start=1):
        if i % 500 == 0 or i == total:
            logger.debug("  Hashed %d / %d …", i, total)
        digest = compute_file_hash(img_path, algorithm)
        if digest is None:
            failed += 1
            logger.warning("Could not hash '%s'; skipping.", img_path)
            continue
        hash_to_paths[digest].append(img_path)

    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    n_dup_groups = len(duplicates)
    n_dup_files = sum(len(v) for v in duplicates.values())

    if n_dup_groups == 0:
        logger.info("✅  No duplicate images detected.")
    else:
        logger.warning(
            "⚠️  Found %d duplicate group(s) involving %d files:",
            n_dup_groups, n_dup_files,
        )
        for h, paths in duplicates.items():
            logger.warning("  [%s…]  %d copies:", h[:12], len(paths))
            for p in paths:
                logger.warning("      %s", p)

    if failed:
        logger.warning("%d file(s) could not be hashed.", failed)

    logger.info(
        "Duplicate scan complete: %d unique images, %d duplicate groups.",
        len(hash_to_paths) - n_dup_groups, n_dup_groups,
    )
    return duplicates

def detect_data_leakage(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    algorithm: str = "md5",
    block_on_detect: bool = True,
) -> Dict[str, List[str]]:
    
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("DATA LEAKAGE DETECTION STARTED  (algorithm: %s)", algorithm.upper())
    logger.info("=" * 60)

    def _hash_split(samples: List[Tuple[str, int]], name: str) -> Dict[str, str]:
        
        result: Dict[str, str] = {}
        for path, _ in samples:
            digest = compute_file_hash(path, algorithm)
            if digest is not None:
                result[digest] = path
        logger.info("  %s split: %d images hashed.", name, len(result))
        return result

    train_hashes = _hash_split(train_samples, "Train")
    val_hashes   = _hash_split(val_samples,   "Val  ")
    test_hashes  = _hash_split(test_samples,  "Test ")

    train_set = set(train_hashes)
    val_set   = set(val_hashes)
    test_set  = set(test_hashes)

    tv_overlap = train_set & val_set     
    tt_overlap = train_set & test_set    
    vt_overlap = val_set   & test_set    

    leakage: Dict[str, List[str]] = {
        "train_val":  [train_hashes[h] for h in tv_overlap],
        "train_test": [train_hashes[h] for h in tt_overlap],
        "val_test":   [val_hashes[h]   for h in vt_overlap],
    }

    total_leaked = sum(len(v) for v in leakage.values())

    if total_leaked == 0:
        logger.info(
            "✅  No data leakage detected. "
            "Train / Val / Test splits are completely disjoint."
        )
    else:
        for pair, paths in leakage.items():
            if paths:
                logger.error(
                    "❌  LEAKAGE DETECTED in [%s]: %d image(s) shared:",
                    pair, len(paths),
                )
                for p in paths:
                    logger.error("      %s", p)

        message = (
            f"DATA LEAKAGE DETECTED: {total_leaked} image(s) appear in "
            f"multiple splits.  "
            f"train∩val={len(leakage['train_val'])}, "
            f"train∩test={len(leakage['train_test'])}, "
            f"val∩test={len(leakage['val_test'])}. "
            f"Fix the split logic before training."
        )
        if block_on_detect:
            raise RuntimeError(message)
        else:
            logger.warning(message)

    logger.info("-" * 60)
    return leakage

class StageTimer:

    def __init__(self) -> None:
        import time as _time
        self._time = _time
        self._start_times: Dict[str, float] = {}
        self._elapsed: Dict[str, float] = {}   

    def start(self, stage: str) -> None:
        
        self._start_times[stage] = self._time.perf_counter()

    def stop(self, stage: str) -> float:
        
        t0 = self._start_times.get(stage)
        if t0 is None:
            get_logger().warning("StageTimer: stop('%s') called without start.", stage)
            return 0.0
        elapsed = self._time.perf_counter() - t0
        self._elapsed[stage] = round(elapsed, 3)
        get_logger().debug("Stage '%s' finished in %.3f s", stage, elapsed)
        return elapsed

    def display_table(self) -> None:
        
        if not self._elapsed:
            return
        total = sum(self._elapsed.values())
        col_w = 34
        print("\n" + "=" * 52)
        print("  PIPELINE TIMING")
        print("=" * 52)
        print(f"  {'Stage':<{col_w}} {'Time (s)':>8}  {'%':>5}")
        print("  " + "-" * 48)
        for stage, t in self._elapsed.items():
            pct = 100.0 * t / total if total > 0 else 0.0
            print(f"  {stage:<{col_w}} {t:>8.3f}  {pct:>4.1f}%")
        print("  " + "-" * 48)
        print(f"  {'TOTAL':<{col_w}} {total:>8.3f}  100.0%")
        print("=" * 52 + "\n")

def detect_cross_folder_leakage(
    training_paths: Dict[str, List[str]],
    testing_paths: Dict[str, List[str]],
    algorithm: str = "md5",
) -> Dict:
    
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("CROSS-FOLDER LEAKAGE DETECTION  (Training vs Testing)")
    logger.info("=" * 60)

    train_all: List[str] = [p for paths in training_paths.values() for p in paths]
    test_all: List[str] = [p for paths in testing_paths.values() for p in paths]

    if not train_all or not test_all:
        logger.warning("One or both folder path lists are empty; skipping check.")
        return {"total_leaked": 0, "leaked_files": [], "leakage_found": False}

    logger.info("Hashing %d Training images …", len(train_all))
    train_hashes: Dict[str, str] = {}   
    for p in train_all:
        h = compute_file_hash(p, algorithm)
        if h:
            train_hashes[h] = p

    logger.info("Hashing %d Testing images …", len(test_all))
    test_hashes: Dict[str, str] = {}
    for p in test_all:
        h = compute_file_hash(p, algorithm)
        if h:
            test_hashes[h] = p

    overlap = set(train_hashes) & set(test_hashes)
    leaked_files = [
        {"training": train_hashes[h], "testing": test_hashes[h]}
        for h in overlap
    ]

    if not overlap:
        logger.info(
            "SAFE: No images are shared between Training/ and Testing/ folders."
        )
    else:
        logger.warning(
            "WARNING: %d image(s) found in BOTH Training/ and Testing/ folders:",
            len(overlap),
        )
        for item in leaked_files:
            logger.warning("  Train: %s", item["training"])
            logger.warning("  Test : %s", item["testing"])

    logger.info("-" * 60)
    return {
        "total_leaked": len(overlap),
        "leaked_files": leaked_files,
        "leakage_found": len(overlap) > 0,
    }
