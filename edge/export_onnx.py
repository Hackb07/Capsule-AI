"""Export the trained capsule YOLO model to hardware-friendly ONNX graphs.

Why ONNX: the PyTorch checkpoint needs the full torch stack (~800 MB) and
runs an unoptimized graph. The exported ONNX graph runs under ONNX Runtime
with CPU graph optimizations and no torch dependency, which is what makes
deployment on a small board (Raspberry Pi 5, 8 GB) practical.

Artifacts (written to ``models/``):
    capsule_yolo11n_640.onnx       FP32 ONNX, 640 px input (accuracy parity)
    capsule_yolo11n_640_int8.onnx  INT8 dynamic-quantized copy of the above
    capsule_yolo11n_416.onnx       FP32 ONNX, 416 px input (low-latency option)

Run from the project root::

    .venv\\Scripts\\python.exe edge/export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO  # noqa: E402


def export(weights: Path, out_dir: Path, imgsz: int) -> Path:
    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=imgsz, opset=17, verbose=False)
    src = Path(exported)
    dst = out_dir / f"capsule_yolo11n_{imgsz}.onnx"
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    print(f"exported {dst.name}: {dst.stat().st_size / 1e6:.2f} MB (imgsz={imgsz})")
    return dst


def quantize_dynamic(src: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst = src.with_name(src.stem + "_int8.onnx")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    print(f"quantized {dst.name}: {dst.stat().st_size / 1e6:.2f} MB")
    return dst


def main() -> None:
    weights = BASE_DIR / "models" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Trained weights not found: {weights}")
    out_dir = BASE_DIR / "models"
    onnx640 = export(weights, out_dir, 640)
    quantize_dynamic(onnx640)
    export(weights, out_dir, 416)


if __name__ == "__main__":
    main()
