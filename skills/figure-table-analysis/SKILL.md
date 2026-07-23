---
name: figure-table-analysis
description: Analyze research-paper figures, diagrams, plots, tables, ablations, metrics, benchmark evidence, captions, and surrounding text. Use when a user asks what a visual/table shows, what claims it supports, or how to read experimental evidence.
---

# Figure Table Analysis

Explain visuals and tables using parsed cards plus nearby paper text.

## Workflow

1. Identify the figure/table id, page, caption, and surrounding text.
2. For figures, explain blocks, axes, arrows, comparisons, and workflow stages.
3. For tables, reconstruct columns, rows, metrics, baselines, and deltas.
4. State what claim the visual supports.
5. State what the visual does not support.

## Output

- Visual/table summary
- Key numbers or components
- Supported claims
- Caveats

## Gotchas

- Avoid overreading plots when exact values are unavailable.
- Do not round or alter table values unless you say you are approximating.
