"""Benchmark PyTorch vs ONNX edge backends on CPU.

Compares, on the same held-out test images and the same protocol:
    torch-fp32-640   baseline (models/best.pt)
    onnx-fp32-640    exported ONNX, 640 px
    onnx-int8-640    dynamic-quantized ONNX, 640 px
    onnx-fp32-416    exported ONNX, 416 px (low-latency option)

For each backend records validation accuracy (P/R/mAP50/mAP50-95 on the
test split), per-image latency statistics, model file size and prediction
agreement with the baseline. Results go to outputs/metrics/edge_metrics.json
and are the numbers reported in the research paper.

Run from the project root (after edge/export_onnx.py)::

    .venv\\Scripts\\python.exe edge/benchmark_edge.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO  # noqa: E402

BACKENDS = [
    ("torch-fp32-640", "models/best.pt", 640),
    ("onnx-fp32-640", "models/capsule_yolo11n_640.onnx", 640),
    ("onnx-int8-640", "models/capsule_yolo11n_640_int8.onnx", 640),
    ("onnx-fp32-416", "models/capsule_yolo11n_416.onnx", 416),
]


def validate(name: str, weights: Path, imgsz: int, data_yaml: Path) -> dict:
    """Full test-split validation through the Ultralytics interface."""
    try:
        model = YOLO(str(weights))
        m = model.val(data=str(data_yaml), split="test", imgsz=imgsz,
                      device="cpu", verbose=False)
        r = m.results_dict
        return {
            "ok": True,
            "precision": round(float(r.get("metrics/precision(B)", m.box.mp)), 4),
            "recall": round(float(r.get("metrics/recall(B)", m.box.mr)), 4),
            "mAP50": round(float(r.get("metrics/mAP50(B)", m.box.map50)), 4),
            "mAP50_95": round(float(r.get("metrics/mAP50-95(B)", m.box.map)), 4),
        }
    except Exception as exc:  # pragma: no cover - reported, not fatal
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def latency_stats(model: YOLO, images: list[Path], imgsz: int,
                  runs: int = 5) -> dict:
    lat: list[float] = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)  # warmup
        for _ in range(runs):
            start = time.perf_counter()
            model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
            lat.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(lat, dtype=np.float64)
    return {
        "n_samples": int(arr.size),
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
    }


def boxes_of(model: YOLO, img: np.ndarray, imgsz: int, conf: float = 0.25):
    res = model.predict(img, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    out = []
    for b in res.boxes:
        out.append((int(b.cls.item()), float(b.conf.item()),
                    [float(v) for v in b.xyxy[0].tolist()]))
    return out


def iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def agreement(base: list, cand: list) -> float:
    """F1-style box agreement: same class + IoU>=0.5 greedy matching."""
    if not base and not cand:
        return 1.0
    if not base or not cand:
        return 0.0
    used = set()
    match = 0
    for cb, cf, bx in base:
        best, bj = 0.0, -1
        for j, (cc, _, cx) in enumerate(cand):
            if j in used or cc != cb:
                continue
            v = iou(bx, cx)
            if v > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj)
            match += 1
    return round(2 * match / (len(base) + len(cand)), 4)


def main() -> None:
    data_yaml = BASE_DIR / "outputs" / "metrics" / "eval_data.yaml"
    if not data_yaml.exists():
        raise SystemExit("Run scripts/evaluate_model.py first (creates eval_data.yaml).")
    for _, rel, _ in BACKENDS:
        if not (BASE_DIR / rel).exists():
            raise SystemExit(f"Missing {rel}: run edge/export_onnx.py first.")

    images = sorted((BASE_DIR / "data" / "test" / "images").glob("*.jpg"))
    bench_images = images[:10]

    baseline_boxes: dict[str, list] = {}
    rows: dict[str, dict] = {}
    for name, rel, imgsz in BACKENDS:
        weights = BASE_DIR / rel
        print(f"[{name}] validating...", flush=True)
        acc = validate(name, weights, imgsz, data_yaml)
        print(f"[{name}] accuracy: {acc}", flush=True)
        model = YOLO(str(weights))
        print(f"[{name}] benchmarking latency...", flush=True)
        lat = latency_stats(model, bench_images, imgsz)
        print(f"[{name}] latency: {lat}", flush=True)

        boxes: dict[str, list] = {}
        for img_path in images:
            img = cv2.imread(str(img_path))
            boxes[img_path.name] = boxes_of(model, img, imgsz)
        if name == "torch-fp32-640":
            baseline_boxes = boxes
            agr = 1.0
        else:
            vals = [agreement(baseline_boxes[k], boxes[k]) for k in baseline_boxes]
            agr = round(float(np.mean(vals)), 4) if vals else 0.0

        rows[name] = {
            "weights": rel,
            "imgsz": imgsz,
            "file_bytes": weights.stat().st_size,
            "accuracy": acc,
            "latency_cpu_ms": lat,
            "agreement_with_baseline": agr,
        }

    base_ms = rows["torch-fp32-640"]["latency_cpu_ms"]["mean_ms"]
    for name, row in rows.items():
        row["speedup_vs_baseline"] = round(base_ms / row["latency_cpu_ms"]["mean_ms"], 3)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "cpu, warmup 1 + 5 timed runs x 10 test images; "
                    "val on 40-image test split; agreement at conf=0.25",
        "backends": rows,
    }
    out = BASE_DIR / "outputs" / "metrics" / "edge_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
