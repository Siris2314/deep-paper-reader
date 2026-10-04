# Setup and Configuration

This guide covers a clean local installation. Deep Paper Reader is designed to run with Ollama; external APIs are optional and are called only by the workflows that use them.

## 1. Install prerequisites

Install:

- [Python 3.11+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [Ollama](https://ollama.com/download)

Start Ollama before running the hardware check. On Windows and macOS, the desktop application normally starts the service. On Linux:

```bash
ollama serve
```

## 2. Create the project environment

The setup scripts create `.venv`, install the package and runtime dependencies, copy `.env.example` to `.env`, and write a conservative local hardware profile.

### Windows

```powershell
.\scripts\setup.ps1
```

PowerShell activation is not required. To activate the environment for interactive development:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS and Linux

```bash
chmod +x scripts/setup.sh scripts/run_paper_reader_ui.sh
./scripts/setup.sh
```

To include test, lint, and build tools, pass `-Dev` on Windows or `--dev` on macOS/Linux. Add `-Observability` or `--observability` to install the optional Langfuse SDK.

## 3. Install local models

Run:

```bash
.venv/bin/python -m paper_agent.cli hardware
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m paper_agent.cli hardware
```

The report lists missing model pulls. A balanced setup uses:

```bash
ollama pull qwen3.5:4b
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
```

Automatic profiles are conservative:

| Accelerator memory | Profile | Context | Loaded models |
|---|---|---:|---:|
| CPU only | `cpu` | 4K | 1 |
| Under 12 GB | `low-vram` | 4K | 1 |
| 12-23 GB | `balanced` | 8K | 2 |
| 24 GB or more | `high-vram` | 32K | 2 |

Override detection from an activated `.venv` with:

```bash
python -m paper_agent.cli hardware --profile low-vram --apply
```

On Windows, `python -m paper_agent.cli hardware --apply --apply-server` can also set a reversible Ollama service profile. Fully quit and restart Ollama afterward. Restore previous values with:

```powershell
.\scripts\configure_ollama_windows.ps1 -Action Reset
```

## 4. Configure optional services

Edit `.env` and add only the services you intend to use.

### Tavily

Create a key at [app.tavily.com](https://app.tavily.com/home):

```dotenv
TAVILY_API_KEY=tvly-...
```

Tavily provides the general explanation in concept cards and web evidence in `Web` or `Explore` chat mode. Calls are made on demand, not during every parse.

### Semantic Scholar

Request access from the [Semantic Scholar API](https://www.semanticscholar.org/product/api):

```dotenv
SEMANTIC_SCHOLAR_API_KEY=
SEMANTIC_SCHOLAR_ALLOW_UNAUTHENTICATED=false
```

Local bibliography parsing works without this service. The API enriches metadata; it does not prove that one method inherits from another.

### OpenAI

Ollama is the default. To use the optional OpenAI backend, create an [OpenAI API key](https://platform.openai.com/api-keys), set `MODEL_PROVIDER=openai`, and choose a compatible chat model in `.env`.

### Langfuse

Install the observability extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[observability]"
```

Create a project in [Langfuse](https://cloud.langfuse.com), then add both project keys and the correct regional base URL:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
LANGFUSE_CAPTURE_CONTENT=false
```

Each upload, concept enrichment, equation walkthrough, and paper-chat turn becomes a trace. Math-judge dimensions, chat grounding, parser success, repair attempts, and user feedback are exported as scores. With content capture disabled, prompts, answers, and paper excerpts are represented by hashes and sizes instead of raw text. Set `LANGFUSE_CAPTURE_CONTENT=true` only when the paper and user data may be sent to the configured Langfuse project.

The older LangSmith environment switches remain accepted by LangChain, but do not enable both tracing backends for the same local run.

After collecting traces, run the release-quality gate:

```powershell
.\.venv\Scripts\python.exe -m paper_agent.cli llmops-gate
```

Thresholds live in [`llmops/gate.json`](../llmops/gate.json), and the generated diagnosis is written to `llmops/reports/latest.json`. The strict gate fails when a metric regresses or does not yet have the configured minimum sample count. During initial trace collection, inspect coverage without blocking by adding `--allow-insufficient-data`.

The `LLMOps quality gate` GitHub Actions workflow runs the same command manually. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` as repository secrets. Add `LANGFUSE_BASE_URL` and `LANGFUSE_TRACING_ENVIRONMENT` as repository variables. Prompt or model promotion remains a human-approved action after the gate passes.

## 5. Run

```powershell
.\scripts\run_paper_reader_ui.ps1
```

```bash
./scripts/run_paper_reader_ui.sh
```

Open [http://localhost:8503](http://localhost:8503). The application does not upload a paper to an external service unless an explicitly enabled workflow sends selected context to that service.

## 6. Verify the installation

```bash
python -m paper_agent.cli hardware
python -c "import paper_agent; print(paper_agent.__version__)"
```

For a development install:

```bash
python -m ruff check src ui scripts tests
python -m ruff format --check src ui scripts tests
python -m pytest -q
python -m build
```

## Environment reference

The complete annotated list lives in [`.env.example`](../.env.example). The most commonly changed settings are:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_NUM_CTX` | `8192` | Active context allocation sent with local requests |
| `PAPER_READER_MATH_MODEL` | `qwen3.5:4b` | Math explanation model |
| `PAPER_READER_MATH_REPAIR_ATTEMPTS` | `1` | Bounded revise-and-rejudge attempts after a weak math answer |
| `PAPER_READER_MATH_MIN_IMPROVEMENT` | `0.1` | Minimum judge-score gain required to keep a repair |
| `PAPER_READER_AGENT_MODEL` | `qwen2.5:7b` | Concept and chat model |
| `PAPER_READER_JUDGE_MODEL` | `qwen2.5:7b` | Independent math judge |
| `PAPER_CHAT_MAX_PARALLEL` | `1` | Maximum concurrent local specialists |
| `TAVILY_RESEARCH_TIMEOUT` | `90` | Tavily research timeout in seconds |
| `LANGFUSE_ENABLED` | `false` | Enable optional Langfuse tracing when the SDK and both keys are present |
| `LANGFUSE_CAPTURE_CONTENT` | `false` | Export full prompts and outputs instead of privacy-preserving summaries |

## Local data

These ignored directories are created as needed:

- `uploaded_papers/`: local upload copies
- `paper_reports/`: isolated parsed paper workspaces and chat threads
- `agent_memory/`: user-confirmed cross-paper memory, math-judge alignment, and hardware profiles

Delete a paper workspace when it is no longer needed. Never commit `.env`, generated workspaces, API keys, or copyrighted PDFs without permission.

## Troubleshooting

### A Conda environment and `.venv` are both shown

That is harmless. The launch scripts invoke `.venv` directly. Confirm with:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

### Ollama is unavailable

Check:

```bash
ollama list
curl http://localhost:11434/api/tags
```

Start or restart the Ollama application, then rerun `python -m paper_agent.cli hardware`.

### A model is very slow or runs out of memory

Apply a smaller profile and reduce parallelism:

```bash
python -m paper_agent.cli hardware --profile low-vram --apply
```

Keep `PAPER_CHAT_MAX_PARALLEL=1`. Use `scripts/benchmark_ollama.py --cold` to compare load and generation time.

The shared [agent harness](HARNESS.md) also limits concurrent local-model calls across
chat, math, and reports. Its budgets and diagnostics are described there; use
`python scripts/benchmark_harness.py` to compare agent framework prompt overhead offline.
See [data source recommendations](DATA_SOURCES.md) for structured-paper and OCR options.

The reader's **Resolve metadata and related work** action uses exact identifiers from PDF page 1.
Crossref and casual OpenAlex lookup work without credentials. Set `CROSSREF_MAILTO` for the
Crossref polite pool, `OPENALEX_API_KEY` for a larger OpenAlex budget, and
`SEMANTIC_SCHOLAR_API_KEY` for Semantic Scholar metadata and citation enrichment. External
records stay labeled as discovery context and are never treated as paper evidence.

### Tavily enrichment stops

Verify `TAVILY_API_KEY`, restart the reader after editing `.env`, and inspect the error shown in the concept popover. Tavily queries are intentionally capped below provider length limits.

### Equations display as raw text

The reader currently loads MathJax from jsDelivr. The PDF and deterministic analysis remain local, but formatted browser math requires network access to that asset.
