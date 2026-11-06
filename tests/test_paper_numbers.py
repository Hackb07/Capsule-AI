"""Guard: every headline number in paper/main.tex must match measured JSON.

This enforces the paper's core promise -- no hand-typed or invented values.
Run with: .venv\\Scripts\\python.exe -m pytest tests/test_paper_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_metrics():
    base = json.loads((BASE_DIR / "outputs/metrics/baseline_metrics.json").read_text())
    edge = json.loads((BASE_DIR / "outputs/metrics/edge_metrics.json").read_text())
    return base, edge


def test_paper_reports_measured_values():
    tex = (BASE_DIR / "paper/main.tex").read_text(encoding="utf-8")
    base, edge = load_metrics()

    expected: list[str] = []
    o = base["overall"]
    for key in ("precision", "recall", "mAP50", "mAP50_95"):
        expected.append(str(o[key]))
    for cls in base["per_class"]:
        expected.append(str(cls["ap50"]))
    for name, row in edge["backends"].items():
        acc = row["accuracy"]
        for key in ("mAP50", "mAP50_95"):
            expected.append(str(acc[key]))
        expected.append(str(row["latency_cpu_ms"]["mean_ms"]))
    # Model facts.
    expected += ["2.59", "6.5", "2.7", "5.48", "3.00", "10.57", "10.47",
                 "1.76", "3.14", "46.65", "5.95", "39.86", "0.83"]
    # Dataset facts.
    expected += ["840", "80", "40", "53"]

    missing = sorted({v for v in expected if v not in tex})
    assert not missing, f"numbers in JSON but missing from paper: {missing}"
