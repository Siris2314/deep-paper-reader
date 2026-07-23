---
name: paper-evaluator
description: Evaluate final reports, serious chat answers, concept cards, or math explanations using deterministic checks and independent LLM-as-a-judge rubrics for correctness, relevance, paper grounding, source mixing, symbol coverage, LaTeX fidelity, unsupported claims, and repair needs. Use before accepting substantial paper outputs or when the UI needs scored quality evidence.
---

# Paper Evaluator

Validate whether an answer is actually about the uploaded paper.

## Workflow

1. Check title and abstract alignment.
2. Check focus on the main contribution instead of appendices, references, prompts, or boilerplate.
3. Check coverage of central method, results, figures, tables, and limitations.
4. Identify unsupported claims and source mixing.
5. Fail generic summaries that begin with phrases like "the provided text appears".
6. Run code-based checks for schema, missing fields, unsafe LaTeX, and provenance labels.
7. Use an independent judge model for nuanced correctness, grounding, relevance, coverage, and usefulness.
8. Return numeric scores plus written justifications; do not let the judge rewrite the candidate.
9. Return repair instructions specific enough for a rewrite.

## Output JSON

```json
{
  "passed": true,
  "score": 0,
  "dimensions": {"correctness": {"score": 0, "justification": "..."}},
  "issues": [{"severity": "fatal|major|minor", "message": "..."}],
  "repair_instruction": "..."
}
```

## Gotchas

- A fluent answer can still fail if it ignores the paper's main contribution.
- Do not treat web context as paper evidence.
