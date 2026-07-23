from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

from paper_agent.math_agent import analyze_expression  # noqa: E402


if __name__ == "__main__":
    expression = " ".join(sys.argv[1:]).strip()
    if not expression:
        raise SystemExit("Usage: analyze_expression.py '<expression or LaTeX>'")
    print(json.dumps(asdict(analyze_expression(expression)), indent=2, ensure_ascii=False))
