# Research paper (LaTeX source)

IEEE-conference-style paper on the Capsule-AI system and its edge deployment.
All prose is written originally for this project; every accuracy and latency
figure is a measured value produced by the scripts below — no numbers are
hand-typed or borrowed.

## Sources

- `main.tex` — the paper.
- `references.bib` — references (YOLO, COCO metrics, Transformer, YOLO11 /
  ONNX Runtime / Raspberry Pi 5 documentation, Roboflow dataset page).

## Numbers provenance

| Table | Produced by | Values file |
|---|---|---|
| Per-class accuracy | `scripts/evaluate_model.py` | `outputs/metrics/baseline_metrics.json` |
| Backend comparison | `edge/benchmark_edge.py` | `outputs/metrics/edge_metrics.json` |
| Edge pipeline breakdown | `edge/edge_infer.py --save outputs/edge` | `outputs/edge/detections.json` (timings) |

`tests/test_paper_numbers.py` fails if any headline JSON value is missing
from `main.tex`, so the paper cannot drift from the measurements.

## Build the PDF

Any TeX Live / MiKTeX install, or an Overleaf project containing `main.tex`
and `references.bib`:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Only standard packages are used (`cite`, `amsmath`, `graphicx`, `xcolor`,
`tikz`, `url`, `hyperref`) plus the `IEEEtran` class, so no extra setup is
needed on Overleaf (Menu → Compiler: pdfLaTeX).

## Originality note

The text, figures, and tables were authored from scratch for this repository.
The dataset is credited and cited (Roboflow Universe, CC BY 4.0) and all
third-party methods are cited rather than paraphrased. If you submit this to
a venue, replace the placeholder contact email on the title page and confirm
the venue's authorship and AI-use policy.
