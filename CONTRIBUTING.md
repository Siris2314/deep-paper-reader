# Contributing

Deep Paper Reader is an early-stage research tool. Keep changes narrow, evidence-aware, and testable.

## Development setup

```powershell
.\scripts\setup.ps1 -Dev
```

```bash
./scripts/setup.sh --dev
```

Activate `.venv` (or invoke its Python executable directly). Run before opening a pull request:

```bash
python -m ruff check src ui scripts tests
python -m ruff format --check src ui scripts tests
python -m pytest -q
python -m build
```

## Design rules

- Treat the uploaded paper as the primary source.
- Label paper evidence, web context, and model inference separately.
- Retrieve relevant sections and artifacts instead of prompting with the whole paper.
- Require citation-context evidence before claiming that a paper extends, adopts, or modifies prior work.
- Keep the main agent context lean; put repeatable specialist workflows in `skills/`.
- Preserve deterministic fallbacks when a model or external service fails.
- Store generated artifacts and user memory only in ignored workspace directories.

## Changing a skill or prompt

Document the workflow contract in the skill's `SKILL.md`. Add or update tests for routing, evidence boundaries, output cleanup, and failure behavior. Prompt-only changes still need a concrete evaluation case; avoid relying on a single favorable manual run.

## Changing the UI

`ui/paper_reader_server.py` is the supported UI. Preserve the original PDF render and coordinate overlays. Verify embedded JavaScript syntax and test at desktop and narrow widths. Do not add a second UI entry point without first removing or migrating the old one.

## Pull requests

Keep pull requests focused. Include:

- the user-visible problem
- the implementation and evidence contract affected
- tests or evaluation performed
- screenshots for UI changes
- new environment variables or external network calls

Do not include uploaded papers, generated reports, `.env`, API keys, local model files, or long-term user memory.
