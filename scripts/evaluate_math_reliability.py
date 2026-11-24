from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_agent.math_reliability_eval import (
    baseline_predictions,
    evaluate_math_reliability,
    load_math_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score equation transcription and explanation outputs."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("llmops/math_reliability_benchmark.json"),
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", help="Run one case ID for a model-backed integration check")
    parser.add_argument(
        "--live-pdf-map", type=Path, help="JSON mapping paper IDs to local PDF paths"
    )
    parser.add_argument(
        "--pdf-dir", type=Path, help="Discover versioned arXiv PDFs in this directory"
    )
    parser.add_argument(
        "--explain", action="store_true", help="Include actual model explanation and judge calls"
    )
    parser.add_argument("--workspace", type=Path, default=Path("llmops/reports/live-math"))
    args = parser.parse_args()
    benchmark = load_math_benchmark(args.benchmark)
    if args.case:
        benchmark = dict(benchmark, cases=[c for c in benchmark["cases"] if c["id"] == args.case])
        if not benchmark["cases"]:
            parser.error("Unknown case ID")
    predictions = (
        json.loads(args.predictions.read_text(encoding="utf-8"))
        if args.predictions
        else baseline_predictions(benchmark)
    )
    if args.live_pdf_map or args.pdf_dir:
        from evaluate_paper_retrieval import discover_pdfs

        from paper_agent.live_eval import live_predictions
        from paper_agent.workspace import Workspace

        pdf_map = (
            json.loads(args.live_pdf_map.read_text(encoding="utf-8"))
            if args.live_pdf_map
            else discover_pdfs(args.pdf_dir)
        )
        predictions = live_predictions(
            benchmark, pdf_map, Workspace(args.workspace), explain=args.explain
        )
    if isinstance(predictions, dict):
        predictions = predictions.get("predictions", [])
    report = evaluate_math_reliability(benchmark, predictions)
    report["run_kind"] = (
        "live_pipeline"
        if args.live_pdf_map or args.pdf_dir
        else "external_predictions"
        if args.predictions
        else "fixture_smoke_test"
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
