# Deep Paper Agent Memory

These are stable project rules injected as lightweight memory. Keep this file short.

- Main context stays lean. Do not load detailed math, implementation, evaluator, concept, or Tavily workflows by default.
- Use skills when a task requires a consistent multi-step workflow, portable domain expertise, token reduction, hallucination reduction, or deterministic helper code.
- Treat the uploaded paper as the primary source. Separate paper evidence, web context, and inference.
- Prefer section/artifact retrieval over whole-paper prompting.
- Do not summarize appendices, system prompts, references, or boilerplate as the paper's main contribution unless the user asks about them.
- Serious answers and reports should be evaluated for paper alignment.
- Treat citation edges as references, not proof of method inheritance; require citation-context evidence for lineage claims.
