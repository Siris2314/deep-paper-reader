"""Redistributable, authored PDF regression corpus; not a scientific quality benchmark."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import fitz

from paper_agent.evidence_index import build_evidence_records, search_records
from paper_agent.parser import ParsedPaper, extract_pdf_text
from paper_agent.workspace import Workspace


def render_fixture(spec: dict, path: Path) -> Path:
    document = fitz.open()
    for blocks in spec["pages"]:
        page = document.new_page(width=612, height=792)
        columns = 2 if spec["layout"] == "columns" else 1
        heights = [55.0] * columns
        for index, text in enumerate(blocks):
            column = index % columns
            x = 45 + 275 * column
            width = 245 if columns == 2 else 520
            height = 82 if columns == 2 else 65
            result = page.insert_textbox(
                fitz.Rect(x, heights[column], x + width, heights[column] + height),
                text,
                fontsize=11,
                fontname="helv",
            )
            if result < 0:
                raise ValueError("Fixture text overflow")
            heights[column] += height + 12
    if spec["layout"] in {"scan", "mixed_scan"}:
        scanned = fitz.open()
        for original in document:
            page = scanned.new_page(width=612, height=792)
            page.insert_image(
                page.rect,
                stream=original.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png"),
            )
            if spec["layout"] == "mixed_scan":
                page.insert_text((45, 30), "Digitized research note", fontsize=9)
        document.close()
        document = scanned
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()
    return path


def run_regression(corpus: dict, destination: Path, limit: int = 8) -> dict:
    loaded = {}
    for spec in corpus["documents"]:
        workspace = Workspace(destination / spec["id"])
        pdf = render_fixture(spec, workspace.path("source.pdf"))
        statuses = {}

        def record_ocr(page, status, layout):
            statuses[str(page)] = status
            if layout:
                workspace.write_json(f"parsed/ocr_layout/{page:04d}.json", layout)

        full, pages, metadata = extract_pdf_text(pdf, record_ocr)
        workspace.write_json("parsed/ocr_status.json", statuses)
        paper = ParsedPaper(metadata, full, pages, {}, [], [], [])
        loaded[spec["id"]] = (build_evidence_records(paper, workspace), workspace, statuses)
    rows = []
    for case in corpus["cases"]:
        records, workspace, statuses = loaded[case["document"]]
        if case.get("requires_ocr") and "ocr" not in statuses.values():
            rows.append({"id": case["id"], "status": "ocr_unavailable", "passed": False})
            continue
        started = time.monotonic()
        hits, diagnostics = search_records(records, case["question"], limit, workspace)

        def normalize(value):
            return " ".join(value.split())

        quote = normalize(case.get("expected_quote", ""))
        matching = [h for h in hits if quote and quote in normalize(h["text"])]
        valid_sections = (
            all(h.get("section") != case.get("forbidden_section") for h in hits)
            if case.get("forbidden_section")
            else True
        )
        passed = (not hits if case.get("unanswerable") else bool(matching)) and valid_sections
        rows.append(
            {
                "id": case["id"],
                "status": "evaluated",
                "passed": passed,
                "quote_hit": bool(matching) if quote else None,
                "correct_abstention": not hits if case.get("unanswerable") else None,
                "source_ids": [h["id"] for h in hits],
                "retrieval": diagnostics,
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
            }
        )
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    return {
        "run_kind": "authored_pdf_regression",
        "scientific_generalization_evaluated": False,
        "case_count": len(rows),
        "coverage": len(evaluated) / len(rows),
        "passed": all(r["passed"] for r in rows),
        "metrics": {
            name: statistics.fmean(r[name] for r in evaluated if r.get(name) is not None)
            if any(r.get(name) is not None for r in evaluated)
            else None
            for name in ("quote_hit", "correct_abstention", "latency_ms")
        },
        "cases": rows,
    }
