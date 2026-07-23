---
name: chat-answer
description: Interactive research-paper Q&A for a parsed paper workspace. Use when answering user questions against parsed paper artifacts, chat history, concept cards, figures, tables, equations, cached Tavily context, or when routing a question to specialized paper-analysis skills.
---

# Chat Answer

Answer paper questions from workspace artifacts, not from a whole-paper prompt.

## Workflow

1. Classify the user's intent: summary, method, math, implementation, concept, experiment, critique, or related work.
2. Read recent `chat/history.jsonl` turns when present, but do not let history override the current paper evidence.
3. Retrieve only relevant sections and parsed artifacts.
4. Use specialized skills only when the question needs their workflow.
5. Separate paper evidence, web context, and inference.
6. Write the new turn to `chat/history.jsonl`.
7. Evaluate serious answers for drift.

## Gotchas

- Treat `AGENTS.md` as stable project memory, not per-paper memory.
- Treat `chat/history.jsonl`, `concepts/`, `web/`, and `parsed/` as per-paper memory.
- Prefer paper mechanisms, numbers, and named components over generic explanations.
