"""Real PDF retrieval evaluation, with no answer text provided to retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from paper_agent.evidence_index import build_evidence_records, resolve_query, search_records
from paper_agent.parser import ParsedPaper, extract_pdf_text
from paper_agent.workspace import Workspace


def discover_pdfs(directory: Path) -> dict[str, str]:
    import re

    import fitz

    result = {}
    for path in directory.glob("*.pdf"):
        with fitz.open(path) as doc:
            stamp = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", doc[0].get_text(), re.I)
        if stamp:
            result["arXiv:" + stamp[1]] = str(path)
            if stamp[2]:
                result["arXiv:" + stamp[1] + stamp[2]] = str(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("llmops/retrieval_benchmark.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("uploaded_papers"))
    parser.add_argument("--output", type=Path, default=Path("llmops/reports/retrieval-live.json"))
    args = parser.parse_args()
    pdfs = discover_pdfs(args.pdf_dir)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    loaded, rows = {}, []
    for case in benchmark["cases"]:
        if case["paper_id"] not in pdfs:
            rows.append({"id": case["id"], "status": "pdf_missing"})
            continue
        if case["paper_id"] not in loaded:
            full, pages, metadata = extract_pdf_text(pdfs[case["paper_id"]])
            paper = ParsedPaper(metadata, full, pages, {}, [], [], [])
            workspace = Workspace(
                args.output.parent / "retrieval-eval" / case["paper_id"].replace(":", "-")
            )
            loaded[case["paper_id"]] = (build_evidence_records(paper, workspace), workspace)
        records, workspace = loaded[case["paper_id"]]
        start = time.monotonic()
        hits, diagnostics = search_records(
            records, resolve_query(case["question"], case.get("history")), 8, workspace
        )
        gold = set(case.get("expected_pages", []))
        first = next((i + 1 for i, hit in enumerate(hits) if hit.get("page") in gold), None)
        rows.append(
            {
                "id": case["id"],
                "status": "evaluated",
                "hit_at_8": bool(first) if gold else None,
                "reciprocal_rank": 1 / first if first else 0 if gold else None,
                "correct_abstention": not hits if case.get("unanswerable") else None,
                "pages": [h.get("page") for h in hits],
                "ids": [h["id"] for h in hits],
                "latency_ms": round((time.monotonic() - start) * 1000, 2),
                "retrieval": diagnostics,
            }
        )
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    report = {
        "run_kind": "live_pdf_retrieval",
        "cases": rows,
        "coverage": len(evaluated) / len(rows),
        "metrics": {
            name: statistics.fmean(r[name] for r in evaluated if r.get(name) is not None)
            if any(r.get(name) is not None for r in evaluated)
            else None
            for name in ("hit_at_8", "reciprocal_rank", "correct_abstention", "latency_ms")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": report["coverage"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
