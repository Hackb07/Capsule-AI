"""Structural integrity checks for paper/main.tex (no TeX install needed).

Verifies balanced braces/environments, label/ref consistency, citation keys,
and table column counts. A full PDF build still needs TeX Live / Overleaf
(see paper/README.md).
"""

from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEX = (BASE_DIR / "paper/main.tex").read_text(encoding="utf-8")
BIB = (BASE_DIR / "paper/references.bib").read_text(encoding="utf-8")


def strip_comments(tex: str) -> str:
    return "\n".join(
        re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines()
    )


def test_braces_balanced():
    body = strip_comments(TEX)
    # Ignore escaped braces.
    body = body.replace("\\{", "").replace("\\}", "")
    assert body.count("{") == body.count("}"), (
        f"unbalanced braces: {body.count('{')} vs {body.count('}')}"
    )


def test_environments_balanced():
    body = strip_comments(TEX)
    begins = re.findall(r"\\begin\{(\w+)\*?\}", body)
    ends = re.findall(r"\\end\{(\w+)\*?\}", body)
    assert sorted(begins) == sorted(ends), (
        f"begin/end mismatch: {sorted(begins)} vs {sorted(ends)}"
    )
    assert "document" in begins


def test_labels_and_refs():
    body = strip_comments(TEX)
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:eq|fig|tab|sec)?ref\{([^}]+)\}", body))
    # Also catch plain \ref (already covered) -- be explicit:
    refs |= set(re.findall(r"\\ref\{([^}]+)\}", body))
    assert labels, "no labels found"
    missing = refs - labels
    assert not missing, f"refs without labels: {missing}"
    assert len(labels) == len(re.findall(r"\\label\{([^}]+)\}", body)), \
        "duplicate labels"


def test_citations_resolve():
    body = strip_comments(TEX)
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        cited.update(c.strip() for c in group.split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", BIB))
    assert cited, "no citations found"
    assert cited - defined == set(), f"citations missing from bib: {cited - defined}"
    assert defined - cited == set(), f"bib entries never cited: {defined - cited}"


def test_table_columns_consistent():
    body = strip_comments(TEX)
    for m in re.finditer(r"\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}",
                         body, re.DOTALL):
        spec, content = m.group(1), m.group(2)
        ncols = len(re.sub(r"[^lcr]", "", spec))
        assert ncols > 1
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("\\hline") or "&" not in line:
                continue
            row = line.split("\\\\")[0]
            assert row.count("&") == ncols - 1, (
                f"row has {row.count('&') + 1} cells, spec needs {ncols}: {row[:80]}"
            )


def test_required_sections_present():
    for sec in ("Introduction", "Related Work", "System Overview",
                "Edge Acceleration", "Experiments", "Discussion", "Conclusion",
                "abstract", "bibliography"):
        assert sec in TEX, f"missing: {sec}"
