"""Scoped equation evidence, conservative HTML alignment and human corrections."""

from __future__ import annotations

import hashlib
import json
import re

from paper_agent.arxiv_source import load_source


def correction_key(page: int, raw: str, layout: dict) -> str:
    return hashlib.sha256(
        json.dumps([page, raw, layout.get("bbox")], sort_keys=True).encode()
    ).hexdigest()[:24]


def load_correction(workspace, page: int, raw: str, layout: dict) -> dict:
    path = workspace.path(f"math/corrections/{correction_key(page, raw, layout)}.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_correction(workspace, page: int, raw: str, layout: dict, latex: str) -> dict:
    from paper_agent.math_agent import _safe_display_latex

    clean = _safe_display_latex(latex)
    if not clean or len(clean) > 6000:
        raise ValueError("Enter valid, balanced LaTeX up to 6000 characters.")
    result = {
        "latex": clean,
        "page": page,
        "raw": raw,
        "bbox": layout.get("bbox"),
        "source": "user_correction",
        "scope": "this_paper_and_region",
    }
    workspace.write_json(f"math/corrections/{correction_key(page, raw, layout)}.json", result)
    return result


def align_source_equation(parsed, workspace, raw: str, geometry: str) -> dict | None:
    from paper_agent.math_agent import _latex_similarity

    source = load_source(parsed, workspace)
    if source.get("status") != "ready":
        return None
    label = re.search(r"\((\d+)\)\s*[,.;]?\s*$", raw)
    scored = []
    for item in source.get("items", []):
        if item.get("kind") != "equation":
            continue
        similarity = _latex_similarity(geometry, item.get("latex", "")) if geometry else 0
        anchor_match = bool(
            label and re.search(r"(?:^|[._])E" + label[1] + r"$", item.get("anchor", ""), re.I)
        )
        if similarity >= 0.86 or (anchor_match and similarity >= 0.68):
            scored.append((similarity + (0.15 if anchor_match else 0), item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored or (len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08):
        return None
    return dict(
        scored[0][1],
        alignment="label_and_notation" if label else "notation",
        version=source.get("version"),
    )


def tutor_context(parsed, workspace, page: int, raw: str, latex: str) -> dict:
    # Reading order determines proximity; proximity never asserts a dependency.
    from paper_agent.evidence_index import build_evidence_records

    equations = [r for r in build_evidence_records(parsed, workspace) if r["kind"] == "equation"]
    nearby = sorted(equations, key=lambda r: abs((r.get("page") or page) - page))
    neighbors = [r for r in nearby if r["text"].strip() != raw.strip()][:4]
    return {
        "neighbors": [
            {
                "id": r["id"],
                "page": r["page"],
                "text": r["text"][:1000],
                "relationship": "nearby_equation_dependency_unverified",
            }
            for r in neighbors
        ],
        "notation_scope": {"page": page, "equation": latex},
        "guidance": "Use definitions in this equation's section. Nearby equations are candidates, not established derivation dependencies.",
    }
