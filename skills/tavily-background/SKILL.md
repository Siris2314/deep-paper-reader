---
name: tavily-background
description: Gather and synthesize external context for a research paper using Tavily Search, Tavily Research, or cached web artifacts, including concept definitions, official paper pages, code, project pages, related work, benchmark pages, and follow-up papers. Use when the user asks for web context, opens an enriched concept card, or the task needs external verification.
---

# Tavily Background

Use external sources only when paper-only context is insufficient or explicitly requested.

## Workflow

1. Use Tavily Research `mini` for a narrow concept definition; use `pro` only for genuinely multi-part research.
2. Poll non-streaming research tasks until completion and cache the synthesized content plus sources.
3. Fall back to Tavily Search with `include_answer=True` only when Research is unavailable, and label the fallback.
4. Search official sources first for paper metadata, code, and benchmark claims.
5. Use external context to explain background, not to overwrite paper evidence.
6. Cache term research under `web/tavily_term_research/` and concept cards under `concepts/cards/`.
7. Record source URLs and synthesis provenance.

## Gotchas

- Do not use web search for paper-only answers.
- If search fails, say so.
