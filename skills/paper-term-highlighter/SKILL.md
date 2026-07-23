---
name: paper-term-highlighter
description: Identify and highlight significant research-paper terms from parsed paper artifacts and user-confirmed long-term term memory, especially main-method concepts, ML terms, math terms, equations, figures, tables, benchmarks, and clickable concept-card candidates. Use when a UI or agent needs to annotate a paper, learn a missed highlight, expose skill routing, or decide which terms should open math/concept/result panels.
---

# Paper Term Highlighter

Identify significant paper terms for clickable highlighting and skill probing.

## Workflow

1. Load the cached Papers with Code methods snapshot and retain only terms found in the current paper.
2. Extract newly introduced method names and acronyms directly from the paper so post-snapshot work is covered.
3. Read user-confirmed terms from `agent_memory/term_highlights.json` on demand.
4. Prefer parsed artifacts over whole-paper prompting.
5. Score terms from the main method, abstract, equations, figures, tables, and experiment sections.
6. Give matching user-confirmed terms a decisive score boost and preserve their confirmed category.
7. Categorize each new term as `math`, `method`, `result`, or `concept`.
8. Attach downstream skills:
   - `math` -> `math-walkthrough`
   - `method` -> `implementation-reconstructor`
   - `result` -> `figure-table-analysis`
   - all categories -> `concept-card`
9. Save paper-specific output under `memory/significant_terms.json`, including vocabulary provenance.
10. Overlay coordinates on the original PDF rendering; do not reconstruct the document from extracted text.
11. Save explicit user corrections to long-term memory and refresh the current paper overlays immediately.

## Output

Write JSON records with:

- `term`
- `category`
- `score`
- `paper_occurrences`
- `method_occurrences`
- `evidence`
- `skills`
- `learned`
- `correction_count`
- `vocabulary_source`

## Code Offload

Use `scripts/dump_terms.py <workspace>` to print deterministic significant terms when working outside the UI.

## Gotchas

- Do not highlight every capitalized phrase.
- The Papers with Code source is a static July 2025 archive, so paper-local extraction and user memory remain required.
- Do not treat references, appendices, or boilerplate as main-method evidence unless the user asks.
- Label web-enriched term cards separately from paper evidence.
