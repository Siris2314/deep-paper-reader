from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

from paper_agent.parser import load_parsed_paper_from_workspace  # noqa: E402
from paper_agent.term_highlighter import save_significant_terms  # noqa: E402
from paper_agent.workspace import Workspace  # noqa: E402


if __name__ == "__main__":
    workspace = Workspace(sys.argv[1] if len(sys.argv) > 1 else "paper_report")
    parsed = load_parsed_paper_from_workspace(workspace)
    terms = save_significant_terms(parsed, workspace)
    print(json.dumps([term.__dict__ for term in terms], indent=2, ensure_ascii=False))
