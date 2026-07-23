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

To include test, lint, and build tools, pass `-Dev` on Windows or `--dev` on macOS/Linux.

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

### LangSmith

Tracing is disabled by default. To enable it, create a key in [LangSmith](https://smith.langchain.com/settings):

```dotenv
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-paper-agent
```

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
| `PAPER_READER_AGENT_MODEL` | `qwen2.5:7b` | Concept and chat model |
| `PAPER_READER_JUDGE_MODEL` | `qwen2.5:7b` | Independent math judge |
| `PAPER_CHAT_MAX_PARALLEL` | `1` | Maximum concurrent local specialists |
| `TAVILY_RESEARCH_TIMEOUT` | `90` | Tavily research timeout in seconds |

## Local data

These ignored directories are created as needed:

- `uploaded_papers/`: local upload copies
- `paper_reports/`: isolated parsed paper workspaces and chat threads
- `agent_memory/`: user-confirmed cross-paper memory and hardware profiles

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

### Tavily enrichment stops

Verify `TAVILY_API_KEY`, restart the reader after editing `.env`, and inspect the error shown in the concept popover. Tavily queries are intentionally capped below provider length limits.

### Equations display as raw text

The reader currently loads MathJax from jsDelivr. The PDF and deterministic analysis remain local, but formatted browser math requires network access to that asset.
