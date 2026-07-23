---
name: implementation-reconstructor
description: Reconstruct implementation details from a research paper, including algorithms, architecture, pseudocode, data structures, training/runtime phases, missing details, edge cases, and test plans. Use when a user asks how to implement, code, reproduce, or operationalize a paper.
---

# Implementation Reconstructor

Turn paper method evidence into an implementation plan without pretending missing details are stated.

## Workflow

1. Identify method components and runtime/training phases.
2. Separate explicit paper details from implementation inference.
3. Produce inputs, outputs, data structures, and control flow.
4. Write pseudocode in small functions.
5. Mark missing details.
6. Include a minimal implementation plan and testing strategy.

## Output Sections

- Explicit paper algorithm
- Inferred implementation design
- Pseudocode
- Required data structures
- Edge cases
- Missing details / cannot infer
- Unit tests or smoke tests

## Gotchas

- Call out missing hyperparameters, exact shapes, storage layout, scheduler, batching, evaluation code, and hardware assumptions.
- Label all practical engineering choices that are inferred beyond the paper.
