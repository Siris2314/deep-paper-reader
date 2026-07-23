#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
  echo "Project environment not found. Run './scripts/setup.sh' first." >&2
  exit 1
fi

if [ ! -f "agent_memory/hardware_profile.json" ]; then
  .venv/bin/python -m paper_agent.cli hardware --apply
fi

.venv/bin/python ui/paper_reader_server.py
