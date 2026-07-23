from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / 'src'
sys.path.insert(0, str(ROOT))
from paper_agent.parser import load_parsed_paper_from_workspace  # noqa
from paper_agent.workspace import Workspace  # noqa
from paper_agent.concepts import save_concept_index  # noqa

if __name__ == '__main__':
    workspace = Workspace(sys.argv[1] if len(sys.argv) > 1 else 'paper_report')
    parsed = load_parsed_paper_from_workspace(workspace)
    concepts = save_concept_index(parsed, workspace)
    print(json.dumps(concepts, indent=2, ensure_ascii=False))
