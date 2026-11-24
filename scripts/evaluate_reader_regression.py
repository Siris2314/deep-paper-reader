from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_agent.reader_regression import run_regression


def main():
    parser = argparse.ArgumentParser(description="Run authored cross-layout PDF regression cases")
    parser.add_argument("--corpus", type=Path, default=Path("llmops/reader_regression_corpus.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("llmops/reports/reader-regression.json")
    )
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = run_regression(
        json.loads(args.corpus.read_text(encoding="utf-8")), args.output.parent / "regression-pdfs"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    if args.require_pass and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
