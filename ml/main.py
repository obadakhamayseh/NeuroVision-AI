

import io
import os
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from config import Config
from dataset import BrainTumorDataset, DatasetSplitter
from dataloader import DataLoaderFactory
from preprocess import DatasetAnalyser, DatasetScanner, TransformBuilder
from artifacts import ArtifactsManager
from reports import ReportGenerator
from validators import TensorValidator
from utils import (
    StageTimer,
    detect_cross_folder_leakage,
    detect_data_leakage,
    detect_duplicates,
    get_logger,
    set_global_seed,
    setup_logging,
)
from visualize import (
    show_augmentations,
    show_batch,
    show_class_distribution,
    show_class_samples,
    show_preprocessing,
)

def main() -> None:

    cfg = Config()

    logger = setup_logging(
        log_dir=cfg.LOG_DIR,
        log_level=cfg.LOG_LEVEL,
        log_filename="pipeline.log",
    )
    logger.info("Brain Tumor MRI Preprocessing Pipeline started.")
    cfg.display()

    set_global_seed(cfg.RANDOM_SEED)

    artifacts_mgr = ArtifactsManager(cfg)
    artifacts_mgr.setup_directories()

    report_gen = ReportGenerator(cfg)
    validator  = TensorValidator(cfg)
    timer      = StageTimer()

    status: dict = {k: False for k in [
        "dataset_scanned", "dataset_validated", "dataset_analysed",
        "dataset_balanced", "duplicate_detection", "leakage_check",
        "reports_generated", "figures_generated", "dataloaders_ready",
        "tensor_validation", "ready_for_training",
    ]}

    timer.start("Dataset Scan (Training)")
    train_scanner = DatasetScanner(cfg)

    if not os.path.isdir(cfg.TRAINING_DIR):
        logger.error(
            "Training directory not found: %s\n"
            "Please update TRAINING_DIR in config.py.",
            cfg.TRAINING_DIR,
        )
        sys.exit(1)

    train_valid_paths, class_to_idx = train_scanner.scan(cfg.TRAINING_DIR)
    timer.stop("Dataset Scan (Training)")

    timer.start("Dataset Scan (Testing)")
    test_scanner = DatasetScanner(cfg)

    if not os.path.isdir(cfg.TESTING_DIR):
        logger.warning(
            "Testing directory not found: %s. "
            "Will fall back to 3-way split of Training data.",
            cfg.TESTING_DIR,
        )
        test_valid_paths: dict = {}
    else:
        test_valid_paths, _ = test_scanner.scan(cfg.TESTING_DIR)
    timer.stop("Dataset Scan (Testing)")

    status["dataset_scanned"]   = True
    status["dataset_validated"] = True

    timer.start("Cross-folder Leakage Detection")
    cross_folder_result: dict = {"total_leaked": 0, "leaked_files": [], "leakage_found": False}
    if test_valid_paths:
        cross_folder_result = detect_cross_folder_leakage(
            training_paths=train_valid_paths,
            testing_paths=test_valid_paths,
            algorithm=cfg.DUPLICATE_HASH_ALGO,
        )
    timer.stop("Cross-folder Leakage Detection")

    timer.start("Dataset Analysis")
    analyser = DatasetAnalyser(cfg)
    analysis_results = analyser.analyse(train_valid_paths)
    timer.stop("Dataset Analysis")

    status["dataset_analysed"] = True
    status["dataset_balanced"] = not analysis_results.get("is_imbalanced", False)

    builder = TransformBuilder(cfg)
    train_transform = builder.build_train()
    val_transform   = builder.build_val_test()

    timer.start("Duplicate Detection")
    train_duplicates: dict = {}
    test_duplicates: dict  = {}
    n_dup_groups = 0

    if cfg.CHECK_DUPLICATES:
        train_duplicates = detect_duplicates(
            valid_paths=train_valid_paths,
            algorithm=cfg.DUPLICATE_HASH_ALGO,
        )
        n_dup_groups = len(train_duplicates)

        if test_valid_paths:
            test_duplicates = detect_duplicates(
                valid_paths=test_valid_paths,
                algorithm=cfg.DUPLICATE_HASH_ALGO,
            )
            n_dup_groups += len(test_duplicates)
    timer.stop("Duplicate Detection")

    status["duplicate_detection"] = True

    n_dup_files = (
        sum(len(v) - 1 for v in train_duplicates.values())
        + sum(len(v) - 1 for v in test_duplicates.values())
    )

    timer.start("Dataset Split")
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
    timer.stop("Dataset Split")

    timer.start("Split Leakage Detection")
    leakage_results = detect_data_leakage(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        algorithm=cfg.DUPLICATE_HASH_ALGO,
        block_on_detect=cfg.LEAKAGE_BLOCK_ON_DETECT,
    )
    total_split_leaked = sum(len(v) for v in leakage_results.values())
    timer.stop("Split Leakage Detection")

    any_leakage_found = (
        cross_folder_result.get("leakage_found", False)
        or total_split_leaked > 0
    )
    status["leakage_check"] = True

    moved_files: dict = {}
    n_moved = 0
    if cfg.CHECK_DUPLICATES and cfg.MOVE_DUPLICATES:
        
        moved_files = artifacts_mgr.move_duplicate_files(
            duplicates=train_duplicates,
            valid_paths=train_valid_paths,
        )
        n_moved = sum(len(v) for v in moved_files.values())

    total_train_images = sum(len(v) for v in train_valid_paths.values())
    dup_report = report_gen.generate_duplicate_report(
        duplicates=train_duplicates,
        moved_files=moved_files,
        total_images=total_train_images,
    )

    leakage_report = report_gen.generate_leakage_report(
        cross_folder_result=cross_folder_result,
        split_leakage_result=leakage_results,
    )

    timer.start("Dataset Creation")
    train_dataset = BrainTumorDataset(
        samples=train_samples,
        class_to_idx=class_to_idx,
        transform=train_transform,
        split="train",
    )
    val_dataset = BrainTumorDataset(
        samples=val_samples,
        class_to_idx=class_to_idx,
        transform=val_transform,
        split="val",
    )
    test_dataset = BrainTumorDataset(
        samples=test_samples,
        class_to_idx=class_to_idx,
        transform=val_transform,
        split="test",
    )
    timer.stop("Dataset Creation")

    logger.info(
        "Datasets created -> train=%d  val=%d  test=%d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    timer.start("DataLoader Creation")
    factory = DataLoaderFactory(cfg)
    bundle = factory.build(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        use_weighted_sampler=analysis_results.get("is_imbalanced", False),
    )
    timer.stop("DataLoader Creation")

    status["dataloaders_ready"] = True

    timer.start("Tensor Validation")
    tensor_ok = validator.validate_loaders(
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
    )
    timer.stop("Tensor Validation")

    status["tensor_validation"] = tensor_ok

    timer.start("Visualisations")
    logger.info("Generating visualisations ...")

    show_class_samples(
        valid_paths=train_valid_paths,
        idx_to_class={v: k for k, v in class_to_idx.items()},
        class_to_idx=class_to_idx,
        cfg=cfg,
        n_per_class=4,
    )

    sample_img_path = None
    for paths in train_valid_paths.values():
        if paths:
            sample_img_path = paths[0]
            break

    if sample_img_path:
        show_preprocessing(
            image_path=sample_img_path,
            train_transform=train_transform,
            val_transform=val_transform,
            cfg=cfg,
        )
        show_augmentations(
            image_path=sample_img_path,
            train_transform=train_transform,
            cfg=cfg,
            n_augmentations=8,
        )

    show_class_distribution(
        class_counts=analysis_results["class_counts"],
        cfg=cfg,
    )

    show_batch(
        dataloader=bundle.train_loader,
        idx_to_class={v: k for k, v in class_to_idx.items()},
        cfg=cfg,
        max_images=16,
    )
    timer.stop("Visualisations")

    status["figures_generated"] = True

    artifacts_mgr.collect_figures()
    artifacts_mgr.collect_logs()

    n_skipped   = len(train_scanner.skipped) + len(test_scanner.skipped)
    n_corrupted = 0  

    summary = report_gen.generate_dataset_summary(
        analysis_results=analysis_results,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        n_skipped=n_skipped,
        n_corrupted=n_corrupted,
        n_duplicates=n_dup_files,
        n_removed=n_moved,
        class_to_idx=class_to_idx,
    )

    health_content = report_gen.generate_health_report(
        analysis_results=analysis_results,
        n_duplicates=n_dup_files,
        n_corrupted=n_corrupted,
        n_moved=n_moved,
        leakage_found=any_leakage_found,
        tensor_validation_passed=tensor_ok,
    )

    status["reports_generated"] = True
    status["ready_for_training"] = (
        status["tensor_validation"]
        and not any_leakage_found
        and n_corrupted == 0
    )

    timer.display_table()

    report_gen.print_pipeline_status(status)

    dup_status  = "[!] DUPLICATES FOUND" if n_dup_groups else "[OK] None"
    leak_status = "[FAIL] LEAKAGE DETECTED" if any_leakage_found else "[OK] Splits are clean"
    cross_status = "WARNING" if cross_folder_result.get("leakage_found") else "SAFE"

    print("=" * 70)
    print("  BRAIN TUMOR MRI PREPROCESSING PIPELINE - COMPLETE")
    print("=" * 70)
    print(f"  Classes detected     : {sorted(class_to_idx.keys())}")
    print(f"  Class -> label map   : {class_to_idx}")
    print(f"  Train samples        : {len(train_dataset)}")
    print(f"  Val   samples        : {len(val_dataset)}")
    print(f"  Test  samples        : {len(test_dataset)}")
    print(f"  Total samples        : {len(train_dataset)+len(val_dataset)+len(test_dataset)}")
    print(f"  Skipped images       : {n_skipped}")
    print(f"  Duplicate groups     : {n_dup_groups}  {dup_status}")
    print(f"  Duplicate files      : {n_dup_files}")
    print(f"  Moved to artifacts   : {n_moved}")
    print(f"  Cross-folder leakage : {cross_status}")
    print(f"  Split leakage        : {total_split_leaked} images  {leak_status}")
    print(f"    train & val        : {len(leakage_results.get('train_val', []))}")
    print(f"    train & test       : {len(leakage_results.get('train_test', []))}")
    print(f"    val   & test       : {len(leakage_results.get('val_test', []))}")
    print(f"  Imbalanced dataset   : {analysis_results.get('is_imbalanced', False)}")
    print(f"  Imbalance ratio      : {analysis_results.get('imbalance_ratio', 'N/A')}")
    print(f"  Tensor validation    : {'PASS' if tensor_ok else 'FAIL'}")
    print(f"  Batch size           : {cfg.BATCH_SIZE}")
    print(f"  Train batches        : {len(bundle.train_loader)}")
    print(f"  Val   batches        : {len(bundle.val_loader)}")
    print(f"  Test  batches        : {len(bundle.test_loader)}")
    print(f"  Image size           : {cfg.IMAGE_SIZE}")
    print(f"  Augment strategy     : Resize({cfg.RESIZE_BEFORE_CROP}) -> RandomCrop({cfg.IMAGE_SIZE})")
    print("  ---")
    print(f"  Logs dir             : {cfg.LOG_DIR}")
    print(f"  Artifacts dir        : {cfg.ARTIFACTS_DIR}")
    print(f"  Analysis report      : {cfg.ANALYSIS_REPORT}")
    print(f"  JSON report          : {cfg.ANALYSIS_REPORT_JSON}")
    print(f"  Duplicate report     : {cfg.DUPLICATE_REPORT_JSON}")
    print(f"  Leakage report       : {cfg.LEAKAGE_REPORT_JSON}")
    print(f"  Dataset summary      : {cfg.DATASET_SUMMARY_JSON}")
    print(f"  Health report        : {cfg.HEALTH_REPORT_TXT}")
    print("=" * 70)
    ready_str = "[OK] Pipeline ready." if status["ready_for_training"] else "[WARN] Check reports."
    print(f"\n  {ready_str}")
    print("  [>]  Next step: implement the EfficientNet-B0 model module.\n")

if __name__ == "__main__":
    main()
