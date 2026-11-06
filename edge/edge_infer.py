"""Edge inference app for the capsule YOLO model (Raspberry Pi 5 target).

Runs the exported ONNX graph with ONNX Runtime on CPU only — no torch, no
Ultralytics dependency. Designed for a Raspberry Pi 5 (ARM Cortex-A76 CPU)
but runs on any machine with ``onnxruntime``, ``numpy`` and ``opencv``::

    python edge/edge_infer.py --model models/capsule_yolo11n_640.onnx \\
        --source data/test/images --conf 0.40 --save outputs/edge

Outputs annotated images, a ``detections.json`` file and per-image timings.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc


# Class order of the trained capsule model (see data/data.yaml).
CLASS_NAMES = [
    "Normal-Z-Line",
    "Polyps",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "ulcerative-colities",
]


@dataclass
class EdgeDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2] in original-image pixels


def letterbox(img: np.ndarray, size: int):
    """Resize with unchanged aspect ratio + gray padding (YOLO style).

    Always pads to the full ``size`` square: the exported ONNX graph has a
    static input shape, so stride-multiple (``auto``) padding is not enough
    for non-square inputs.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = size - nh
    pad_w = size - nw
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS. boxes: (N,4) xyxy. Returns kept indices."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thr]
    return keep


class EdgeDetector:
    """ONNX Runtime wrapper with YOLOv8/v11 style decoding."""

    def __init__(self, model_path: Path, conf: float = 0.40, iou: float = 0.45,
                 threads: int | None = None) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # Static export gives [1, 3, S, S]; fall back to 640 otherwise.
        self.imgsz = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 640
        self.conf = float(conf)
        self.iou = float(iou)

    def predict(self, img_bgr: np.ndarray) -> tuple[list[EdgeDetection], dict]:
        """Run full pipeline on one BGR image. Returns detections + timings."""
        t = {}
        t0 = time.perf_counter()
        padded, scale, (pad_x, pad_y) = letterbox(img_bgr, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None].copy()
        t["pre_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: inp})[0]
        t["infer_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        dets = self._decode(out, scale, pad_x, pad_y, img_bgr.shape[:2])
        t["post_ms"] = (time.perf_counter() - t2) * 1000.0
        t["total_ms"] = t["pre_ms"] + t["infer_ms"] + t["post_ms"]
        return dets, {k: round(v, 2) for k, v in t.items()}

    def _decode(self, out: np.ndarray, scale: float, pad_x: int, pad_y: int,
                orig_hw: tuple) -> list[EdgeDetection]:
        pred = np.asarray(out)
        if pred.ndim == 3:
            pred = pred[0]  # (4+nc, N) or (N, 4+nc)
        if pred.shape[0] > pred.shape[1]:
            pred = pred.T  # normalize to (4+nc, N)
        nc = pred.shape[0] - 4
        boxes_xywh = pred[:4].T  # (N, 4)
        scores = pred[4:].T  # (N, nc)
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        keep_conf = confidences >= self.conf
        if not np.any(keep_conf):
            return []
        boxes_xywh = boxes_xywh[keep_conf]
        confidences = confidences[keep_conf]
        class_ids = class_ids[keep_conf]

        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2 - pad_x
        y1 = cy - h / 2 - pad_y
        x2 = cx + w / 2 - pad_x
        y2 = cy + h / 2 - pad_y
        boxes = np.stack([x1, y1, x2, y2], axis=1) / scale
        H, W = orig_hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

        dets: list[EdgeDetection] = []
        for c in np.unique(class_ids):
            idx = np.where(class_ids == c)[0]
            for j in nms(boxes[idx], confidences[idx], self.iou):
                k = idx[j]
                cid = int(c)
                dets.append(
                    EdgeDetection(
                        class_id=cid,
                        class_name=CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}",
                        confidence=round(float(confidences[k]), 4),
                        bbox=[round(float(v), 1) for v in boxes[k].tolist()],
                    )
                )
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets


def draw(img: np.ndarray, dets: list[EdgeDetection]) -> np.ndarray:
    colors = [(0, 165, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0),
              (255, 0, 255), (255, 255, 0), (0, 255, 255), (200, 200, 200)]
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        color = colors[d.class_id % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Capsule YOLO edge inference (ONNX Runtime CPU)")
    ap.add_argument("--model", required=True, help="Path to .onnx model")
    ap.add_argument("--source", required=True, help="Image file or directory")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=None, help="ORT intra-op threads")
    ap.add_argument("--save", default=None, help="Output directory for annotated images + JSON")
    args = ap.parse_args()

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not files:
        raise SystemExit(f"No images found at {src}")

    detector = EdgeDetector(Path(args.model), conf=args.conf, iou=args.iou, threads=args.threads)
    out_dir = Path(args.save) if args.save else None
    if out_dir:
        (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    all_timings: list[float] = []
    records: dict[str, object] = {}
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"skip unreadable: {f.name}")
            continue
        dets, timing = detector.predict(img)
        all_timings.append(timing["total_ms"])
        print(f"{f.name}: {len(dets)} detections, "
              f"pre={timing['pre_ms']} infer={timing['infer_ms']} "
              f"post={timing['post_ms']} total={timing['total_ms']} ms")
        records[f.name] = {
            "detections": [asdict(d) for d in dets],
            "timing_ms": timing,
        }
        if out_dir:
            cv2.imwrite(str(out_dir / "annotated" / f.name), draw(img, dets))

    if out_dir:
        (out_dir / "detections.json").write_text(json.dumps(records, indent=2))
        print(f"saved -> {out_dir}")
    if all_timings:
        arr = np.array(all_timings)
        print(f"mean total: {arr.mean():.1f} ms | median: {np.median(arr):.1f} ms "
              f"| p90: {np.percentile(arr, 90):.1f} ms over {arr.size} images")


if __name__ == "__main__":
    main()
