#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEV=0
OBSERVABILITY=0
SKIP_HARDWARE=0
for arg in "$@"; do
  case "$arg" in
    --dev) DEV=1 ;;
    --observability) OBSERVABILITY=1 ;;
    --skip-hardware) SKIP_HARDWARE=1 ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: ./scripts/setup.sh [--dev] [--observability] [--skip-hardware]" >&2
      exit 2
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 'Python 3.11 or newer is required.')"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating .venv..."
  "$PYTHON" -m venv .venv
fi

VENV_PYTHON="$ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
if [[ "$DEV" -eq 1 && "$OBSERVABILITY" -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install -e ".[dev,observability]"
elif [[ "$DEV" -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install -e ".[dev]"
elif [[ "$OBSERVABILITY" -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install -e ".[observability]"
else
  "$VENV_PYTHON" -m pip install -e .
fi

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example."
fi

if [[ "$SKIP_HARDWARE" -eq 0 ]]; then
  "$VENV_PYTHON" -m paper_agent.cli hardware --apply
fi

echo
echo "Setup complete."
echo "1. Add optional API keys to .env."
echo "2. Install the models reported by the hardware check with 'ollama pull <model>'."
echo "3. Run ./scripts/run_paper_reader_ui.sh"
