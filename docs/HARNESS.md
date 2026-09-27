# Paper agent harness

The runtime combines existing paper retrieval, routing, skills, cached math explanations,
independent judging, and bounded repairs with shared resource controls in
`src/paper_agent/harness.py`. It retains Deep Agents for multi-step reports. Math
generation/judging and concept relevance use direct structured calls where available,
or a minimal LangChain agent without filesystem, planning, or delegation tools.

This follows the task-specific middleware approach described in
[LangChain's harness article](https://www.langchain.com/blog/how-to-build-a-custom-agent-harness)
and its [custom middleware documentation](https://docs.langchain.com/oss/python/langchain/middleware/custom).
The supplied PDF of S Sankar's *Agent Harness - A no-BS guide* informed the budget,
context, and failure-visibility requirements. Its coding examples are illustrative;
they do not authorize tools or actions in a paper-reading workflow.

## What is enforced

- One thread-safe budget per workflow, shared by nested judges, repairs, and report
  subagents. Chat worker threads preserve the context; report middleware also captures
  the run explicitly. Independent requests receive independent budgets.
- Limits on model-call reservations, tool calls, estimated cumulative input tokens,
  and elapsed time before dispatch. Failures consume reservations. Budget exceptions
  propagate through fallback handlers, so exhausted runs are not cached as successful answers.
- A process-wide local-model semaphore, defaulting to one simultaneous Ollama request,
  including equation vision. Tool execution does not hold that semaphore, allowing
  delegated work to call a model without deadlocking its parent.
- A stop after more than three consecutive identical tool-name/argument pairs. Different
  calls reset this detector; the overall call budget also bounds alternating loops.
- Text tool results above 8,000 characters are stored in `scratch/harness/` with a
  truncated result pointing to the full artifact. Structured graph commands remain intact.
- Report section/file tools return slices of at most 6,000 characters with continuation
  offsets. Equation, figure, and table tools page up to 12 cards in valid JSON.
- Base Ollama generation is capped at 1,800 output tokens and models stay warm for 15
  minutes by default. Existing chat, math, judge, and concept output limits override this
  default where configured. Existing single-specialist synthesis skipping is preserved.

The runtime does not add generic retries on top of model fallbacks and semantic repairs.
The harness itself adds no provider. The reader's separate, explicit exact-version arXiv
HTML action is documented in [DATA_SOURCES.md](DATA_SOURCES.md).

## Defaults and configuration

| Profile | Model reservations | Tool calls | Estimated cumulative input tokens | Dispatch deadline |
| --- | ---: | ---: | ---: | ---: |
| CHAT | 8 | 24 | 60,000 | 300 seconds |
| MATH | 12 | 24 | 100,000 | 480 seconds |
| CONCEPT | 4 | 12 | 30,000 | 180 seconds |
| REPORT | 64 | 120 | 600,000 | 1,800 seconds |

Override with `PAPER_HARNESS_<PROFILE>_MODEL_CALLS`, `_TOOL_CALLS`, `_INPUT_TOKENS`,
or `_SECONDS`. Invalid environment values fall back to defaults; numeric values are
clamped. `PAPER_HARNESS_LOCAL_CONCURRENCY` controls the process semaphore (1-16;
restart after changing it). `PAPER_HARNESS_MAX_OUTPUT_TOKENS` controls the base Ollama
generation cap (128-8192). Use the existing provider/role model settings for model selection.

The shell and PowerShell launchers need no new options. These controls are active when
the application loads the updated code. `.env.example` lists the common settings.

## Diagnostics and limits

Each run writes `logs/harness/<run-id>.json` inside its paper workspace, even when tracing
is disabled or the workflow fails. It records the policy, status, stop reason, reservations,
estimated input size, model/tool durations, queue time, returned usage when available,
and exception types. It does not store prompt or answer text in the metrics file.
Full tool-result artifacts can contain paper content and follow workspace retention.
Metrics write failures are logged without replacing the original workflow result/error.

These are synchronous dispatch deadlines, not hard cancellation. An in-flight HTTP call
is governed by its transport timeout. Calls admitted before a deadline can finish after
it; queued requests cannot start after it. The semaphore covers one Python process, not
other servers or applications using Ollama. Async and streaming entry points are not
implemented by this harness; the application currently invokes synchronous workflows.

Token estimates are heuristic and cumulative. They are not a tokenizer, a context-window
guarantee, a dollar budget, or an exact billing record. Vision image tokens are not estimated.
Direct structured parsers may discard provider usage. Framework-internal operations such
as Deep Agents' summarization model calls and provider-internal retries are not separately
charged by these application invocation/middleware hooks. The report's graph recursion
limit is an additional bound, not a substitute for model/tool budgets.

This change does not add resumable checkpoints, cross-run model-response caching, or
automatic model-quality escalation. Existing graph state, artifact caches, correction
memory, and evaluation policies continue to provide those capabilities they already support.

## Reproducible validation

Run from an activated project environment:

```bash
python -m pytest -q tests/test_harness.py
python scripts/benchmark_harness.py
```

The benchmark uses fake model responses and counts the messages and tool schemas sent
by actual installed agent graphs. It makes no inference requests. On the development
environment (Deep Agents 0.6.11 / LangChain 1.3.11), an identical tiny task produced:

| Agent construction | Injected tools | Prompt and schema characters | Estimated input tokens |
| --- | ---: | ---: | ---: |
| Deep Agent with `tools=[]` | 8 | 26,594 | 6,649 |
| Focused agent with `tools=[]` | 0 | 178 | 45 |

This isolates framework overhead, not full paper prompts, answer quality, or wall-clock
latency. The direct structured math path already avoided that overhead. The reduction
benefits concept relevance and the agent-based math/judge fallback paths. Use
`scripts/benchmark_ollama.py` and the new run logs for real model timing; compare grounding
quality and failure rates alongside speed on the same papers and questions.
