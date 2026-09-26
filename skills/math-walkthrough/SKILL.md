---
name: math-walkthrough
description: Parse, typeset, explain, and evaluate research-paper equations, notation, symbols, derivations, losses, objectives, tensor shapes, assumptions, and mathematical cores using deterministic SymPy analysis, faithful LaTeX reconstruction, a paper-grounded math agent, and an independent LLM judge. Use for math-heavy questions, equation cards, hover explanations, derivation checks, implementation views of formulas, or math-answer quality evaluation.
---

# Math Walkthrough

Explain paper math from exact paper locations and parsed equation cards.

## Workflow

1. Identify the section, page, logical equation region, and nearby paragraph.
2. Group overlapping PDF spans, subscripts, superscripts, and wrapped rows into one equation with multiple highlight rectangles.
3. Detect inline math-font spans in explanatory prose and retain sentences containing `where`, `denotes`, `represents`, definitions, dimensions, or assumptions.
4. Attempt deterministic SymPy parsing for LaTeX or safe plain expressions; retain parser errors when extraction is lossy.
5. Transcribe the equation independently from its explanation. For a multimodal model, send only a tight equation crop and request LaTeX, confidence, and legibility issues, not reasoning. Retain the crop bounds, dimensions, and content hash as provenance.
6. Extract symbols from the recovered LaTeX, then retrieve position-sensitive definition windows from the current page and the rest of the paper. Do not derive the symbol table from corrupted plain PDF text when faithful LaTeX is available.
7. Label each symbol meaning as paper-stated, inferred, or unresolved and retain the defining page text. An empty symbol table is a failed grounding stage.
8. Run a compact reasoning agent only after transcription and symbol grounding. Do not ask this call to repeat perception, LaTeX reconstruction, or symbol extraction.
9. Classify the equation's role: definition, score, objective, loss, update rule, constraint, metric, or theorem.
10. Give two to six ordered computational steps, intuition, an implementation view, tensor shapes, derivation notes, and a small faithful example when useful. Clearly separate derivation steps shown by the paper, standard algebraic consequences, and omitted steps; label unknown or inferred dimensions.
11. When the model stage fails, produce a deterministic grounded fallback from the recovered equation and symbol table instead of a generic parser-only message. Persist the model exceptions for debugging.
12. Cite one to three paper facts that support the interpretation and explain the equation's concrete role in the method. Reject generic roles such as `mathematical inference`.
13. Reconcile the parser, PDF-geometry, and crop transcriptions as fallible candidates. Run deterministic checks and an independent judge model across correctness, paper grounding, symbol coverage, LaTeX fidelity, crop provenance, transcription confidence, and usefulness.
14. If the answer does not pass and both generation and judging are available, convert low-scoring dimensions and judge issues into targeted repair instructions. Keep the recovered equation, paper evidence, and paper-sourced symbol meanings fixed.
15. Re-run the independent judge after each repair. Accept a replacement only when its verdict improves or its score rises by the configured minimum and every deterministic repair target is resolved.
16. Retrieve a bounded set of user-confirmed judging principles and similar correction cases. Treat this memory as rubric guidance, never as paper evidence and never as authority over deterministic checks.
17. Stop when the answer passes, the attempt budget is exhausted, output repeats, evaluation does not improve, required feedback remains unresolved, generation fails, or judging is unavailable. Never run an open-ended self-critique loop.
18. Cache explanations under `math/explanations/`, evaluations under `math/evaluations/`, and the iteration trace under `math/loops/`.

## Output Schema

Use equation cards:

- equation_id
- page_or_section
- raw_equation
- purpose
- variables
- plain_english
- steps
- intuition
- dimensional_analysis
- paper_evidence
- implementation_view
- derivation_notes
- toy_example
- transcription_confidence
- explanation_confidence
- assumptions_or_missing_details

## Code Offload

Run `scripts/dump_equations.py <workspace>` when deterministic equation-card extraction is useful.

Run `scripts/analyze_expression.py '<expression or LaTeX>'` to inspect deterministic SymPy parsing without invoking an agent.

## Gotchas

- Label anything not shown in the paper as inference.
- Do not pretend an equation proves more than it does.
- Do not reject an equation only because PDF extraction prevents a clean SymPy parse; let the agent explain the visible raw form and report the parser limitation.
- Do not let the judge silently rewrite the explanation; evaluation and generation must remain separate artifacts.
- Never let a repair alter authoritative LaTeX or overwrite paper-sourced evidence with model inference.
- Reject LaTeX containing control characters, PDF private-use glyphs, unbalanced braces, or unsafe commands before it reaches MathJax.
