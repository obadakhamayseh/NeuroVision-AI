

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config
from inference import (
    Predictor,
    PredictionResponse,
    InferenceError,
    ImageValidationError,
    ModelLoadError,
    InferenceRuntimeError,
    DeviceError,
)
from utils.image_utils import get_image_metadata

def _setup_logging(verbose: bool) -> None:
    
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(os.path.join(_PROJECT_ROOT, "artifacts", "logs"), exist_ok=True)
    log_path = os.path.join(_PROJECT_ROOT, "artifacts", "logs", "inference.log")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger("brain_tumor_inference").addHandler(file_handler)
    logging.getLogger("brain_tumor_inference").setLevel(logging.DEBUG)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "Brain Tumor MRI Classification Inference Engine.\n"
            "Classifies a single MRI image into one of four categories:\n"
            "  Glioma | Meningioma | Pituitary | No Tumor"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        metavar="PATH",
        help="Path to the MRI image to classify.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=4,
        metavar="K",
        help="Number of top predictions to display (default: 4).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON to stdout instead of the formatted display.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip GPU warmup passes.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging to stderr.",
    )
    return parser.parse_args()

_SEP = "=" * 54

def _print_header(image_path: str) -> None:
    meta = get_image_metadata(image_path)
    print(f"\n{_SEP}")
    print(f"  Brain Tumor MRI  -  Inference Engine")
    print(_SEP)
    print(f"  Image   : {meta['filename']}")
    print(f"  Size    : {meta['width']} × {meta['height']} px  ({meta['size_kb']:.1f} KB)")
    print(f"  Format  : {meta['format']} ({meta['mode']})")
    print(_SEP)

def _print_result(result: PredictionResponse) -> None:
    
    print(f"\n  Prediction       : {result.prediction}")
    print(f"  Confidence       : {result.confidence:.2f} %")
    print(f"  Device           : {result.device.upper()}")
    print(f"  Inference Time   : {result.inference_time_ms:.1f} ms")
    print(f"  Model            : {result.model} (v{result.model_version})")
    print(_SEP)
    print("  Class Probabilities:")
    print()
    for class_name, prob in result.probabilities.items():
        bar_len = int(prob / 2)         
        bar = "#" * bar_len + "-" * (50 - bar_len)
        marker = " <" if class_name == result.prediction else ""
        print(f"    {class_name:<14} {prob:6.2f} %  |{bar}|{marker}")
    print()
    if len(result.top_k) > 1:
        print("  Top Predictions:")
        for entry in result.top_k:
            print(
                f"    {entry.rank}.  {entry.class_name:<14}  {entry.confidence:.2f} %"
            )
    print(_SEP)

def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    image_path = os.path.abspath(args.image)

    if not args.json:
        try:
            _print_header(image_path)
        except FileNotFoundError:
            print(f"\n[ERROR] Image not found: '{image_path}'", file=sys.stderr)
            return 1

    try:
        cfg = Config()
        
        cfg.__class__.__dict__   
        predictor = Predictor(cfg)
    except ModelLoadError as exc:
        print(f"\n[ERROR] Model loading failed: {exc}", file=sys.stderr)
        return 2
    except DeviceError as exc:
        print(f"\n[ERROR] Device error: {exc}", file=sys.stderr)
        return 3

    if not args.no_warmup:
        if not args.json:
            print("  Warming up inference engine ...", end="", flush=True)
        predictor.warmup(n_iterations=2)
        if not args.json:
            print(" done.\n")

    try:
        result: PredictionResponse = predictor.predict(image_path)
    except ImageValidationError as exc:
        print(f"\n[ERROR] Image validation failed: {exc}", file=sys.stderr)
        return 1
    except InferenceRuntimeError as exc:
        print(f"\n[ERROR] Inference failed: {exc}", file=sys.stderr)
        return 3
    except DeviceError as exc:
        print(f"\n[ERROR] Device error during inference: {exc}", file=sys.stderr)
        return 3
    except InferenceError as exc:
        print(f"\n[ERROR] Inference engine error: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_result(result)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nAborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        print("\n[UNEXPECTED ERROR]", file=sys.stderr)
        traceback.print_exc()
        sys.exit(4)
