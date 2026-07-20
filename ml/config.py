

import os
from dataclasses import dataclass, field
from typing import Tuple, List

_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))

@dataclass
class Config:

    DATASET_ROOT: str = os.path.join(
        _PROJECT_ROOT,
        "Brain Tumor Detection from MRI",
    )

    TRAINING_DIR: str = os.path.join(DATASET_ROOT, "Training")
    TESTING_DIR: str = os.path.join(DATASET_ROOT, "Testing")

    LOG_DIR: str = os.path.join(_PROJECT_ROOT, "ml", "logs")
    SKIPPED_LOG: str = os.path.join(LOG_DIR, "skipped_images.log")
    ANALYSIS_REPORT: str = os.path.join(LOG_DIR, "dataset_analysis_report.txt")

    ANALYSIS_REPORT_JSON: str = os.path.join(LOG_DIR, "dataset_report.json")

    DUPLICATE_HASH_ALGO: str = "md5"

    LEAKAGE_BLOCK_ON_DETECT: bool = True

    ARTIFACTS_DIR: str = os.path.join(_PROJECT_ROOT, "ml", "artifacts")
    ARTIFACTS_REPORTS_DIR: str = os.path.join(ARTIFACTS_DIR, "reports")
    ARTIFACTS_FIGURES_DIR: str = os.path.join(ARTIFACTS_DIR, "figures")
    ARTIFACTS_DUPLICATES_DIR: str = os.path.join(ARTIFACTS_DIR, "duplicates")
    ARTIFACTS_LOGS_DIR: str = os.path.join(ARTIFACTS_DIR, "logs")
    ARTIFACTS_METADATA_DIR: str = os.path.join(ARTIFACTS_DIR, "metadata")

    CHECK_DUPLICATES: bool = True

    MOVE_DUPLICATES: bool = True

    REMOVE_DUPLICATES: bool = False

    DUPLICATE_REPORT_JSON: str = os.path.join(
        ARTIFACTS_REPORTS_DIR, "duplicate_report.json"
    )
    LEAKAGE_REPORT_JSON: str = os.path.join(
        ARTIFACTS_REPORTS_DIR, "data_leakage_report.json"
    )
    DATASET_SUMMARY_JSON: str = os.path.join(
        ARTIFACTS_REPORTS_DIR, "dataset_summary.json"
    )
    HEALTH_REPORT_TXT: str = os.path.join(
        ARTIFACTS_REPORTS_DIR, "dataset_health_report.txt"
    )

    DATASET_VERSION: str = "1.0.0"

    CACHE_DIR: str = os.path.join(_PROJECT_ROOT, "ml", "cache")

    IMAGE_SIZE: Tuple[int, int] = (512, 512)

    NUM_CHANNELS: int = 3

    VALID_EXTENSIONS: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    )

    TRAIN_RATIO: float = 0.70   
    VAL_RATIO: float = 0.20     
    TEST_RATIO: float = 0.10    

    STRATIFIED_SPLIT: bool = True

    BATCH_SIZE: int = 16

    NUM_WORKERS: int = 0

    PIN_MEMORY: bool = True

    SHUFFLE_TRAIN: bool = True
    SHUFFLE_VAL: bool = False
    SHUFFLE_TEST: bool = False

    DROP_LAST_TRAIN: bool = True
    DROP_LAST_VAL: bool = False
    DROP_LAST_TEST: bool = False

    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    AUG_HORIZONTAL_FLIP: bool = True
    AUG_HORIZONTAL_FLIP_P: float = 0.5   

    AUG_VERTICAL_FLIP: bool = False

    AUG_ROTATION: bool = True
    AUG_ROTATION_DEGREES: float = 15.0

    AUG_AFFINE: bool = True
    AUG_AFFINE_TRANSLATE: Tuple[float, float] = (0.05, 0.05)
    AUG_AFFINE_SHEAR: float = 10.0

    AUG_RANDOM_CROP: bool = True
    RESIZE_BEFORE_CROP: int = 540   

    AUG_GAUSSIAN_BLUR: bool = False

    AUG_COLOR_JITTER: bool = False

    AUG_BRIGHTNESS_CONTRAST: bool = True
    AUG_BRIGHTNESS_FACTOR: float = 0.10   
    AUG_CONTRAST_FACTOR: float = 0.10     

    AUG_GAUSSIAN_NOISE: bool = True
    AUG_GAUSSIAN_NOISE_MEAN: float = 0.0
    AUG_GAUSSIAN_NOISE_STD: float = 0.02  

    RANDOM_SEED: int = 42

    LOG_LEVEL: str = "INFO"

    CLASS_NAMES: List[str] = field(
        default_factory=lambda: ["glioma", "meningioma", "notumor", "pituitary"]
    )

    NUM_CLASSES: int = 4

    DROPOUT_RATE: float = 0.5

    FREEZE_BACKBONE: bool = False

    FINE_TUNE_FROM_LAYER: str = ""

    EPOCHS: int = 30
    GRADIENT_CLIP_MAX_NORM: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 1

    DEVICE: str = "auto"

    USE_AMP: bool = True

    OPTIMIZER_NAME: str = "adamw"
    LEARNING_RATE: float = 3e-5
    WEIGHT_DECAY: float = 1e-4

    OPTIMIZER_SGD_MOMENTUM: float = 0.9
    OPTIMIZER_SGD_NESTEROV: bool = True

    SCHEDULER_NAME: str = "reduce_lr_on_plateau"

    SCHEDULER_REDUCE_LR_PATIENCE: int = 5
    SCHEDULER_REDUCE_LR_FACTOR: float = 0.5
    SCHEDULER_REDUCE_LR_MIN_LR: float = 1e-7

    SCHEDULER_COSINE_T_MAX: int = 30

    SCHEDULER_STEP_LR_STEP_SIZE: int = 10
    SCHEDULER_STEP_LR_GAMMA: float = 0.1

    LOSS_NAME: str = "focal"

    LABEL_SMOOTHING: float = 0.0

    USE_CLASS_WEIGHTS: bool = True

    FOCAL_LOSS_GAMMA: float = 2.0
    
    FOCAL_LOSS_ALPHA: float = 1.0

    EARLY_STOPPING_PATIENCE: int = 8
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    EARLY_STOPPING_MONITOR: str = "loss"

    CHECKPOINTS_DIR: str = os.path.join(ARTIFACTS_DIR, "checkpoints")
    HISTORY_JSON: str = os.path.join(ARTIFACTS_REPORTS_DIR, "history.json")
    HISTORY_CSV:  str = os.path.join(ARTIFACTS_REPORTS_DIR, "history.csv")

    RESUME_TRAINING: bool = False

    INFERENCE_TOP_K: int = 4

    INFERENCE_LOG: str = os.path.join(ARTIFACTS_LOGS_DIR, "inference.log")

    def __post_init__(self) -> None:

        total = self.TRAIN_RATIO + self.VAL_RATIO + self.TEST_RATIO
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.3f}. "
                f"Check TRAIN_RATIO={self.TRAIN_RATIO}, "
                f"VAL_RATIO={self.VAL_RATIO}, TEST_RATIO={self.TEST_RATIO}."
            )
        if self.IMAGE_SIZE[0] <= 0 or self.IMAGE_SIZE[1] <= 0:
            raise ValueError("IMAGE_SIZE dimensions must be positive integers.")
        if not (0 < self.AUG_HORIZONTAL_FLIP_P <= 1):
            raise ValueError("AUG_HORIZONTAL_FLIP_P must be in (0, 1].")

    def display(self) -> None:
        
        print("\n" + "=" * 60)
        print("  ACTIVE CONFIGURATION")
        print("=" * 60)
        for key, val in self.__dict__.items():
            print(f"  {key:<35} = {val}")
        print("=" * 60 + "\n")
