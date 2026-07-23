# Architecture

Deep Paper Reader separates deterministic PDF handling, external research, and model inference so every explanation can expose its evidence and provenance.

```text
PDF upload
  -> PyMuPDF parser and original-page renderer
  -> per-paper workspace
  -> Papers with Code archive + paper-local term extraction + user memory
  -> coordinate overlays on the original pages
  -> click-triggered concept or math workflow
  -> stateful paper-chat coordinator with routed specialists
  -> deterministic checks and independent evaluation
```

## Reader

`ui/paper_reader_server.py` is the only supported web UI. It serves the upload flow, parsing progress, PDF page images, term overlays, concept popovers, math popovers, paper chat, and background job status. The server uses only Python's HTTP server and the project package; Streamlit is not part of the current application.

The launch scripts run `paper-agent hardware --apply` once when no saved hardware profile exists. Detection and project configuration are implemented in `hardware_setup.py`; generated profiles and `.env` backups stay under ignored `agent_memory/`.

## Paper workspace

Each PDF receives an isolated directory under `paper_reports/`:

```text
raw/          extracted text
parsed/       metadata, sections, equations, figures, and tables
memory/       paper-specific ranked terms
concepts/     cached concept cards
web/          Tavily research artifacts
math/         explanations and independent evaluations
chat/         durable chat threads, latest result, and paper-local feedback
research/     cached concept-specific citation lineage
logs/         run configuration and failures
```

The uploaded paper is always the primary source. Web facts and model inference remain separately labeled.

## Term pipeline

`pwc_vocabulary.py` downloads and compacts the final public Papers with Code methods snapshot once. `term_highlighter.py` intersects that vocabulary with the current paper, adds paper-introduced names/acronyms, applies user-confirmed long-term memory, and ranks terms using method, equation, figure, table, and frequency evidence.

The archive is static, so it is a vocabulary seed rather than a source of truth. Newly published terminology comes from paper-local extraction and manual corrections.

## Concept workflow

Opening a highlighted concept creates a paper-only card immediately. When Tavily is configured, the reader starts enrichment on click:

1. Tavily Research produces the general explanation and sources.
2. The paper agent explains why the concept matters in this specific paper.
3. Both outputs are cached independently from paper evidence.

## Math workflow

Equation regions are grouped from PDF geometry and, for display equations, cropped for the multimodal math model. The workflow separates four stages: faithful transcription, deterministic symbol extraction, position-sensitive symbol-definition retrieval, and a compact reasoning call. This prevents one slow model call from discarding usable perception and grounding work. When reasoning fails, the UI receives a grounded deterministic explanation instead of a generic parser-only card.

An independent judge scores correctness, grounding, symbol coverage, LaTeX fidelity, and usefulness. Judge output never silently rewrites the explanation.

The local runtime intentionally uses two primary model weights: `qwen3.5:4b` generates math explanations, while `qwen2.5:7b` handles concept relevance and judges math output. This keeps the judge independent from the math generator without forcing a third large model through limited VRAM.

## Context diagnostics

Each local agent call records an estimated token count for its visible system prompt, retrieved paper evidence, and structured-output schema, along with its reserved output budget. The reader compares this planned usage with the configured `OLLAMA_NUM_CTX` limit and the model-native capacity reported by Ollama's `/api/show` endpoint. Framework overhead and image tokens remain explicitly excluded from the estimate.

The same panel reads `/api/ps` to show currently resident models, their active context lengths, and reported VRAM allocation. `scripts/configure_ollama_windows.ps1` manages a reversible Ollama server profile; `scripts/benchmark_ollama.py` measures cold/warm load and token throughput.

Hardware profiles are selected from detected accelerator memory and system RAM. Project `.env` updates are automatic and narrowly scoped. System-wide Ollama service settings remain explicit: Windows can apply them through the reversible setup command, while macOS and Linux receive commands appropriate to their service manager.

## Paper chat workflow

`paper_chat.py` owns the typed LangGraph state and is shared by the web reader and CLI. Its normal path is:

```text
question + thread history
  -> deterministic skill-aware route
  -> ranked page and artifact evidence packet
  -> one specialist on Fast/Web, bounded specialists on Deep/Explore
  -> synthesis only when multiple specialists ran
  -> deterministic citation and paper-grounding verification
  -> one bounded citation repair when required
  -> persisted answer, citations, trace, and compacted thread history
```

Specialists load their detailed `SKILL.md` only when routed. The general, math, implementation, experiments, and concept workers receive isolated evidence packets and return structured claims, uncertainty, and evidence IDs. Tavily is read-only and lazy: it runs only for `Web`, `Explore`, or an explicitly external/current question.

This is intentionally a coordinator architecture rather than a peer-to-peer swarm. One workflow owns retrieval, final synthesis, provenance, budgets, and failure handling. `Explore` provides bounded multi-perspective analysis without allowing uncontrolled handoffs.

## Research lineage

`research_lineage.py` parses numbered bibliography entries, finds the exact page context for each citation, and classifies concept-specific relationships as `extends`, `modifies`, `adopts`, `compares`, `evaluates_on`, or `background`. Strong dependency labels require explicit language in the citing sentence. Optional Semantic Scholar Graph API data adds canonical metadata and citation intent, but a graph edge or citation count cannot establish architectural inheritance by itself.

Ordinary read-only questions do not interrupt for approval. User feedback becomes cross-paper memory only after an explicit correction and remember action. Per-paper chat history is compacted as it grows; confirmed long-term corrections are retrieved by relevance rather than injected wholesale.

## Memory and skills

`AGENTS.md` contains only stable project rules. User-confirmed missed terms live under `agent_memory/`, while paper-specific artifacts stay in each workspace. Detailed workflows are loaded from `skills/*/SKILL.md` only when their capability is needed.
