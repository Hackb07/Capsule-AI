"""Unit + integration tests for the edge inference app (no torch needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from edge.edge_infer import EdgeDetector, letterbox, nms  # noqa: E402


def test_nms_keeps_highest_and_drops_overlap():
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 250, 250]],
                     dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.45)
    assert keep[0] == 0  # highest score kept
    assert 1 not in keep  # heavy overlap with box 0 dropped
    assert 2 in keep  # distant box kept


def test_nms_empty():
    keep = nms(np.zeros((0, 4)), np.zeros((0,)), 0.45)
    assert keep == []


def test_letterbox_output_shape_and_scale():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, (px, py) = letterbox(img, 640)
    assert padded.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert px == 0 and py == 80  # (640-480)/2 vertical padding


def test_letterbox_small_image_upscales():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, scale, _ = letterbox(img, 416)
    assert padded.shape == (416, 416, 3)
    assert scale == pytest.approx(416 / 200)


def test_edge_detector_real_model_single_image():
    weights = BASE_DIR / "models" / "capsule_yolo11n_640.onnx"
    if not weights.exists():
        pytest.skip("export first: python edge/export_onnx.py")
    imgs = sorted((BASE_DIR / "data" / "test" / "images").glob("*.jpg"))
    if not imgs:
        pytest.skip("test images not present")
    det = EdgeDetector(weights, conf=0.40)
    img = cv2.imread(str(imgs[0]))
    dets, timing = det.predict(img)
    assert set(timing) == {"pre_ms", "infer_ms", "post_ms", "total_ms"}
    assert timing["total_ms"] > 0
    for d in dets:
        assert 0 <= d.class_id < 8
        assert 0.40 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= img.shape[1]
        assert 0 <= y1 < y2 <= img.shape[0]


def test_edge_detector_non_square_input():
    """Static-shape ONNX graph must accept non-square images via padding."""
    weights = BASE_DIR / "models" / "capsule_yolo11n_640.onnx"
    if not weights.exists():
        pytest.skip("export first: python edge/export_onnx.py")
    det = EdgeDetector(weights, conf=0.40)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets, timing = det.predict(img)
    assert timing["total_ms"] > 0
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 480
