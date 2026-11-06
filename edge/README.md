# Edge deployment: capsule YOLO on a Raspberry Pi 5-class device

This directory holds the hardware-efficient conversion of the trained
capsule-endoscopy detector (`models/best.pt`, YOLO11n, 8 classes).

## What was converted, and why

| Backend | Weights | mAP50 | mAP50-95 | Mean latency (CPU) | File |
|---|---|---|---|---|---|
| PyTorch FP32 @640 (baseline) | `models/best.pt` | 0.5163 | 0.3349 | 93.26 ms | 5.48 MB |
| ONNX FP32 @640 | `models/capsule_yolo11n_640.onnx` | 0.5168 | 0.3215 | 53.09 ms | 10.57 MB |
| ONNX INT8 @640 (dynamic quant) | `models/capsule_yolo11n_640_int8.onnx` | 0.4655 | 0.3070 | 779.68 ms | 3.00 MB |
| ONNX FP32 @416 (recommended) | `models/capsule_yolo11n_416.onnx` | 0.4973 | 0.3188 | 29.73 ms | 10.47 MB |

Measured on an AMD Ryzen 5 7235HS CPU, 40-image held-out test split,
warmup + 5 timed runs x 10 images (see `outputs/metrics/edge_metrics.json`).
The same ONNX graphs run unmodified on a Raspberry Pi 5 (ARM Cortex-A76)
via ONNX Runtime; absolute milliseconds differ per CPU, but the relative
ranking and the elimination of the torch dependency transfer directly.

Findings:

- **ONNX FP32 @640: 1.76x faster** than PyTorch with identical mAP50
  (0.5168 vs 0.5163) and no torch dependency (~800 MB saved on device).
- **ONNX FP32 @416: 3.14x faster** (29.73 ms vs 93.26 ms) for a mAP50 cost
  of only 0.019 — the recommended edge operating point.
- **Naive INT8 dynamic quantization is a negative result here**: it shrinks
  the file to 3.00 MB but slows CPU inference ~8x (779.68 ms) on this
  convolution-heavy detector while losing 0.05 mAP50. Kept in the table for
  honesty; not recommended.

End-to-end edge pipeline (`edge_infer.py`, ONNX @640, conf 0.40, 40 images):
preprocess 5.95 ms + inference 39.86 ms + postprocess 0.83 ms =
**46.65 ms mean per image** on the reference CPU.

## Files

- `export_onnx.py` — PyTorch -> ONNX export (640 + 416) + INT8 variant.
- `edge_infer.py` — dependency-light inference app (onnxruntime + numpy +
  opencv only). Runs a single image or directory, writes annotated images,
  `detections.json` and per-image timings.
- `benchmark_edge.py` — reproduces the table above into
  `outputs/metrics/edge_metrics.json`.
- `requirements-edge.txt` — minimal install set for the Pi.

## Reproduce on a workstation

```powershell
$env:YOLO_CONFIG_DIR="E:\Capsule-AI\.ultralytics"
$env:MPLCONFIGDIR="E:\Capsule-AI\.mpl-cache"
.\.venv\Scripts\python.exe edge/export_onnx.py
.\.venv\Scripts\python.exe edge/benchmark_edge.py
.\.venv\Scripts\python.exe edge/edge_infer.py --model models/capsule_yolo11n_416.onnx --source data/test/images --conf 0.40 --save outputs/edge
```

## Deploy on a Raspberry Pi 5 (8 GB, 64-bit Raspberry Pi OS)

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-opencv
pip install --break-system-packages -r edge/requirements-edge.txt  # or inside a venv
scp models/capsule_yolo11n_416.onnx pi@<pi-ip>:~/capsule/
scp edge/edge_infer.py pi@<pi-ip>:~/capsule/
python3 edge_infer.py --model capsule_yolo11n_416.onnx --source <image-or-dir> --save out
```

Optional: run `benchmark_edge.py` on the Pi for device-native milliseconds
(it needs the torch stack only for the PyTorch column; the ONNX columns work
with the edge requirements plus `ultralytics`).

## Notes

- The `.onnx` files are generated artifacts; the three shipped ones are kept
  in `models/` so the Pi needs no export step.
- Input sizes are fixed at export (640 or 416); the app letterboxes inputs
  with the same padding convention as training.
