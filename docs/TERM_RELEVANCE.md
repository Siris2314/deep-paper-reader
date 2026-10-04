# Term relevance evaluation

The reader separates two kinds of feedback:

- **Remember highlight** labels a term relevant in the current paper and promotes it through the
  existing cross-paper term memory.
- **Hide as irrelevant here** removes a proposed term only from the current paper. It does not
  suppress the same phrase in unrelated papers.

Paper-local labels are stored in `memory/term_relevance_feedback.json` inside the paper workspace.
Each record keeps the label, category, extractor score, vocabulary source, page, timestamps, and
revision count. Re-labeling a term replaces its current label while retaining its history count.

## Offline regression gate

Run the checked-in benchmark without model calls or API keys:

```powershell
python scripts/evaluate_term_highlighter.py
```

The benchmark in `llmops/term_relevance_benchmark.json` contains labeled positive and negative
terms across introduced methods, preference optimization, mathematical objectives, sentence
fragments, and document-format noise. The command exits nonzero when labeled precision or recall
falls below the configured thresholds.

This is a selective labeled metric: an extracted term that is not labeled in the benchmark is not
automatically counted as wrong. Expand the cases with reviewed paper examples before using the
score as a broad estimate of production accuracy.
