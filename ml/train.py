

import io
import logging
import os
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from config import Config
from device import get_device
from seed import set_seed

from preprocess import DatasetAnalyser, DatasetScanner
from dataset import BrainTumorDataset, DatasetSplitter
from dataloader import DataLoaderFactory
from utils import (
    detect_data_leakage,
    detect_duplicates,
    setup_logging,
)

from models import BrainTumorClassifier
from training.trainer import Trainer
from evaluation.evaluate import Evaluator
from training.checkpoint import CheckpointManager

logger = logging.getLogger("brain_tumor_pipeline")

def setup_training_logger(cfg: Config) -> logging.Logger:
    
    os.makedirs(cfg.ARTIFACTS_LOGS_DIR, exist_ok=True)
    log_path = os.path.join(cfg.ARTIFACTS_LOGS_DIR, "training.log")

    logger = logging.getLogger("brain_tumor_pipeline")

    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == log_path
        for h in logger.handlers
    ):
        return logger

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.info("Training log initialised: %s", log_path)
    return logger

def build_dataloaders(cfg: Config):
    
    from artifacts import ArtifactsManager
    artifacts_mgr = ArtifactsManager(cfg)
    artifacts_mgr.setup_directories()
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)

    train_scanner = DatasetScanner(cfg)
    train_valid_paths, class_to_idx = train_scanner.scan(cfg.TRAINING_DIR)

    test_valid_paths = {}
    if os.path.isdir(cfg.TESTING_DIR):
        test_scanner = DatasetScanner(cfg)
        test_valid_paths, _ = test_scanner.scan(cfg.TESTING_DIR)

    analyser = DatasetAnalyser(cfg)
    analysis_results = analyser.analyse(train_valid_paths)

    import torchvision.transforms as T

    train_transform = T.Compose([

        T.Resize(
            540,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.Lambda(lambda img: img.convert("RGB")),
        T.CenterCrop(size=cfg.IMAGE_SIZE),   
        T.RandomHorizontalFlip(p=cfg.AUG_HORIZONTAL_FLIP_P),
        T.ToTensor(),
        T.Normalize(
            mean=list(cfg.IMAGENET_MEAN),
            std=list(cfg.IMAGENET_STD),
        ),
    ])
    logger.info(
        "Train transforms: Resize(540) → CenterCrop(%s) → HFlip → Normalize",
        cfg.IMAGE_SIZE,
    )

    val_transform = T.Compose([
        T.Resize(
            540,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.Lambda(lambda img: img.convert("RGB")),
        T.CenterCrop(size=cfg.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(
            mean=list(cfg.IMAGENET_MEAN),
            std=list(cfg.IMAGENET_STD),
        ),
    ])

    if cfg.CHECK_DUPLICATES:
        detect_duplicates(train_valid_paths, algorithm=cfg.DUPLICATE_HASH_ALGO)

    splitter = DatasetSplitter(cfg)
    if test_valid_paths:
        train_samples, val_samples, test_samples = splitter.split_with_prebuilt_test(
            train_valid_paths=train_valid_paths,
            test_valid_paths=test_valid_paths,
            class_to_idx=class_to_idx,
        )
    else:
        train_samples, val_samples, test_samples = splitter.split(
            train_valid_paths, class_to_idx
        )

    detect_data_leakage(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        algorithm=cfg.DUPLICATE_HASH_ALGO,
        block_on_detect=cfg.LEAKAGE_BLOCK_ON_DETECT,
    )

    import torch as _torch
    train_labels = [label for _, label in train_samples]
    num_classes = cfg.NUM_CLASSES
    class_counts = [train_labels.count(c) for c in range(num_classes)]
    total = sum(class_counts)
    class_weights = _torch.tensor(
        [total / (num_classes * max(cnt, 1)) for cnt in class_counts],
        dtype=_torch.float32,
    )
    logger.info(
        "Class weights (inverse-freq): %s",
        dict(zip(sorted(class_to_idx, key=class_to_idx.get), class_weights.tolist())),
    )

    train_dataset = BrainTumorDataset(train_samples, class_to_idx, train_transform, "train")
    val_dataset   = BrainTumorDataset(val_samples,   class_to_idx, val_transform,   "val")
    test_dataset  = BrainTumorDataset(test_samples,  class_to_idx, val_transform,   "test")

    factory = DataLoaderFactory(cfg)
    bundle  = factory.build(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        use_weighted_sampler=analysis_results.get("is_imbalanced", False),
    )

    return bundle, class_to_idx, analysis_results, class_weights

def main() -> None:

    cfg = Config()
    setup_logging(log_dir=cfg.LOG_DIR, log_level=cfg.LOG_LEVEL)
    logger = setup_training_logger(cfg)

    logger.info("=" * 70)
    logger.info("BRAIN TUMOR MRI  -  TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info("Dataset version : %s", cfg.DATASET_VERSION)
    logger.info("Model           : EfficientNet-B0 (ImageNet pretrained)")
    logger.info("Classes         : %s", cfg.CLASS_NAMES)
    logger.info("Epochs          : %d", cfg.EPOCHS)
    logger.info("Batch size      : %d", cfg.BATCH_SIZE)
    logger.info("Optimizer       : %s  lr=%.2e", cfg.OPTIMIZER_NAME, cfg.LEARNING_RATE)
    logger.info("Scheduler       : %s", cfg.SCHEDULER_NAME)
    logger.info("AMP             : %s", cfg.USE_AMP)
    logger.info("Resume          : %s", cfg.RESUME_TRAINING)

    set_seed(cfg.RANDOM_SEED, deterministic=True)

    logger.info("Building DataLoaders ...")
    bundle, class_to_idx, analysis_results, class_weights = build_dataloaders(cfg)
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    logger.info(
        "DataLoaders ready | train=%d  val=%d  test=%d",
        len(bundle.train_loader.dataset),
        len(bundle.val_loader.dataset),
        len(bundle.test_loader.dataset),
    )

    model = BrainTumorClassifier(cfg)

    if cfg.FREEZE_BACKBONE:
        model.freeze_backbone()
    elif cfg.FINE_TUNE_FROM_LAYER:
        model.partial_freeze(cfg.FINE_TUNE_FROM_LAYER)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        class_names=class_names,
        class_weights=class_weights if cfg.USE_CLASS_WEIGHTS else None,
    )
    history = trainer.train()

    logger.info("Loading best model for test-set evaluation ...")
    ckpt_mgr = CheckpointManager(cfg)
    best_state = ckpt_mgr.load_best()

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    else:
        logger.warning(
            "No best checkpoint found; evaluating with last model state."
        )

    evaluator = Evaluator(
        model=model,
        cfg=cfg,
        test_loader=bundle.test_loader,
        class_names=class_names,
    )
    results = evaluator.evaluate()

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best val loss  : %.6f", trainer.best_val_loss)
    logger.info("  Test accuracy  : %.4f%%", results["accuracy"] * 100)
    logger.info("  Test macro F1  : %.4f", results["macro_f1"])
    logger.info("  Checkpoints    : %s", cfg.CHECKPOINTS_DIR)
    logger.info("  Artifacts      : %s", cfg.ARTIFACTS_DIR)
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
