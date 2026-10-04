from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_agent.term_relevance_eval import run_term_relevance_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic term-relevance benchmark.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=REPO_ROOT / "llmops" / "term_relevance_benchmark.json",
    )
    args = parser.parse_args()
    result = run_term_relevance_benchmark(args.benchmark.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
