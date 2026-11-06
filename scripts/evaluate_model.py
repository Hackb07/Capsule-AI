"""Evaluate the trained capsule YOLO model and record original metrics.

Produces a JSON file with the exact numbers reported in the research paper:
per-class precision/recall/mAP, overall mAP50 / mAP50-95, model size,
parameter count, FLOPs, and CPU inference latency.

Run from the project root::

    .venv\\Scripts\\python.exe scripts/evaluate_model.py

Environment:
    YOLO_CONFIG_DIR should point inside the project (see edge/README.md)
    so Ultralytics does not write outside the workspace.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO  # noqa: E402


def build_eval_yaml() -> Path:
    """Write a data yaml with absolute local paths (the shipped data.yaml
    points at the original training machine)."""
    out = BASE_DIR / "outputs" / "metrics" / "eval_data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {(BASE_DIR / 'data').as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "nc: 8",
        "names:",
        "  - Normal-Z-Line",
        "  - Polyps",
        "  - dyed-lifted-polyps",
        "  - dyed-resection-margins",
        "  - esophagitis",
        "  - normal-cecum",
        "  - normal-pylorus",
        "  - ulcerative-colities",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def count_images(split: str) -> int:
    d = BASE_DIR / "data" / split / "images"
    return sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def benchmark_latency(model: YOLO, images: list[Path], runs_per_image: int = 5) -> dict:
    """Measure per-image inference latency on CPU (milliseconds)."""
    lat: list[float] = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Warmup once per image.
        model.predict(img, device="cpu", verbose=False)
        for _ in range(runs_per_image):
            start = time.perf_counter()
            model.predict(img, device="cpu", verbose=False)
            lat.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(lat, dtype=np.float64)
    return {
        "n_samples": int(arr.size),
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def main() -> None:
    model_path = BASE_DIR / "models" / "best.pt"
    if not model_path.exists():
        raise SystemExit(f"Trained weights not found: {model_path}")

    data_yaml = build_eval_yaml()
    model = YOLO(str(model_path))

    # --- Accuracy evaluation on the held-out test split ---
    metrics = model.val(data=str(data_yaml), split="test", device="cpu", verbose=False)
    box = metrics.box
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    per_class: list[dict] = []
    nc = len(names)
    for i in range(nc):
        per_class.append(
            {
                "class_id": i,
                "class_name": str(names.get(i, f"class_{i}")),
                "precision": round(float(box.p[i]) if len(box.p) > i else 0.0, 4),
                "recall": round(float(box.r[i]) if len(box.r) > i else 0.0, 4),
                "ap50": round(float(box.ap50[i]) if hasattr(box, "ap50") and len(box.ap50) > i else 0.0, 4),
                "ap": round(float(box.maps[i]) if len(box.maps) > i else 0.0, 4),
            }
        )

    results = metrics.results_dict
    overall = {
        "precision": round(float(results.get("metrics/precision(B)", box.mp)), 4),
        "recall": round(float(results.get("metrics/recall(B)", box.mr)), 4),
        "mAP50": round(float(results.get("metrics/mAP50(B)", box.map50)), 4),
        "mAP50_95": round(float(results.get("metrics/mAP50-95(B)", box.map)), 4),
        "fitness": round(float(results.get("fitness", 0.0)), 4),
    }

    # --- Model cost ---
    info = model.info(verbose=False)
    # model.info() returns (params, ...)? Handle common shapes defensively.
    params_m, gflops = None, None
    try:
        import re

        txt = str(info)
        m = re.search(r"([\d.]+)\s*M\s*parameters", txt)
        if m:
            params_m = float(m.group(1))
        m = re.search(r"([\d.]+)\s*GFLOPs", txt)
        if m:
            gflops = float(m.group(1))
    except Exception:
        pass
    if params_m is None:
        try:
            params_m = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
        except Exception:
            params_m = 0.0

    # --- Latency benchmark on a fixed subset of test images ---
    test_images = sorted((BASE_DIR / "data" / "test" / "images").glob("*.jpg"))[:10]
    latency = benchmark_latency(model, test_images)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_path.name,
        "model_bytes": model_path.stat().st_size,
        "model_params_M": params_m,
        "model_GFLOPs": gflops,
        "device": "cpu",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "dataset": {
            "train_images": count_images("train"),
            "valid_images": count_images("valid"),
            "test_images": count_images("test"),
            "nc": 8,
            "names": [str(names.get(i)) for i in range(nc)],
        },
        "overall": overall,
        "per_class": per_class,
        "latency_cpu_ms": latency,
    }

    out = BASE_DIR / "outputs" / "metrics" / "baseline_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
