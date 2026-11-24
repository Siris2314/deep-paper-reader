"""Run real PDF extraction, optional model explanation, and retrieval evaluations."""

from __future__ import annotations

import re
import time

import fitz

from paper_agent.math_regions import extract_math_regions
from paper_agent.parser import ParsedPaper, extract_pdf_text
from paper_agent.workspace import Workspace


def live_predictions(
    benchmark: dict, pdf_map: dict[str, str], workspace: Workspace, explain: bool = False
) -> list[dict]:
    """Gold LaTeX/symbols are deliberately inaccessible to the inference path.

    A case locator is its source page and printed equation label, or input.bbox.
    Unlocatable regions are reported as misses; no nearest-gold-text matching.
    """
    loaded = {}
    predictions = []
    for case in benchmark["cases"]:
        source = case["source"]
        path = pdf_map.get(source["paper_id"])
        if not path:
            continue
        started = time.monotonic()
        if path not in loaded:
            full, pages, metadata = extract_pdf_text(path)
            loaded[path] = ParsedPaper(metadata, full, pages, {}, [], [], [])
        parsed = loaded[path]
        from paper_agent.evidence_index import document_id

        paper_workspace = Workspace(workspace.root / "papers" / document_id(parsed))
        page_number = int(source["page"])
        with fitz.open(path) as doc:
            page = doc[page_number - 1]
            regions = extract_math_regions(
                page.get_text("dict", sort=True), page_number, page.rect.width, page.rect.height
            )
            bbox = (case.get("input") or {}).get("bbox")
            label = str(source.get("equation_label") or "")
            matches = [
                r
                for r in regions
                if r["kind"] == "display"
                and label.isdigit()
                and re.search(r"\(\s*" + label + r"\s*\)", r["raw"])
            ]
            if not matches and label.isdigit():
                labels = page.search_for(f"({label})")
                if len(labels) == 1:
                    y = (labels[0].y0 + labels[0].y1) / 2
                    aligned = [
                        r
                        for r in regions
                        if r["kind"] == "display"
                        and r["layout_evidence"]["bbox"][1] - 5
                        <= y
                        <= r["layout_evidence"]["bbox"][3] + 5
                    ]
                    matches = aligned
            if bbox:
                target = fitz.Rect(bbox)
                matches = sorted(
                    [r for r in regions if r["kind"] == "display"],
                    key=lambda r: (fitz.Rect(r["layout_evidence"]["bbox"]) & target).get_area(),
                    reverse=True,
                )
                matches = [
                    r
                    for r in matches
                    if (fitz.Rect(r["layout_evidence"]["bbox"]) & target).get_area() > 0
                ]
        region = matches[0] if len(matches) == 1 or (bbox and matches) else None
        prediction = {
            "id": case["id"],
            "status": "region_not_found",
            "latex": "",
            "symbols": [],
            "boundary": {},
            "latency_ms": 0,
            "vision_used": False,
        }
        if region:
            layout = region["layout_evidence"]
            prediction.update(
                status="extracted",
                latex=layout.get("geometry_latex", ""),
                boundary={"region_kind": region["kind"], "bbox": layout["bbox"]},
            )
            from paper_agent.math_grounding import build_math_context, extract_latex_symbols

            prediction["symbols"] = extract_latex_symbols(prediction["latex"])
            bundle = build_math_context(parsed, page_number, prediction["latex"], region["raw"])
            prediction["symbol_grounding"] = [vars(item) for item in bundle.symbols]
            if explain:
                from paper_agent.math_agent import explain_equation

                result = explain_equation(
                    parsed,
                    paper_workspace,
                    page_number,
                    region["id"],
                    region["raw"],
                    context=region.get("context", ""),
                    layout_evidence=layout,
                )
                prediction.update(
                    latex=result.display_latex,
                    role=result.role,
                    symbol_grounding=result.symbols,
                    symbols=extract_latex_symbols(result.display_latex),
                    vision_used=result.vision_attempted,
                    status=result.status,
                    explanation=result.plain_english,
                    evaluation=result.evaluation,
                )
        prediction["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
        predictions.append(prediction)
    workspace.write_json("evaluation/live_predictions.json", predictions)
    return predictions


def evaluate_retrieval_cases(cases: list[dict], retrieve) -> dict:
    rows = []
    for case in cases:
        started = time.monotonic()
        results = retrieve(case["question"], case.get("history", []))
        ids = [item["id"] for item in results]
        gold = set(case.get("evidence_ids", []))
        first = next((i + 1 for i, value in enumerate(ids) if value in gold), None)
        rows.append(
            {
                "id": case["id"],
                "recall": len(gold & set(ids)) / len(gold) if gold else None,
                "reciprocal_rank": 1 / first if first else 0,
                "correct_abstention": not ids if case.get("unanswerable") else None,
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
            }
        )
    return {"cases": rows}
