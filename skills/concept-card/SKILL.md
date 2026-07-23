---
name: concept-card
description: Create compact, provenance-labeled concept cards for terms, acronyms, benchmarks, methods, prerequisites, model families, or UI concept popups. Use when a user asks what a term means, why it matters in a paper, or when the app needs a cached popup combining Tavily Research with paper-agent analysis.
---

# Concept Card

Create a popup-sized explanation for one concept.

## Workflow

1. Start with how the term is used in this paper using direct paper evidence.
2. Generate the general explanation with Tavily Research and cache its sources.
3. Run a paper-grounded concept relevance agent to explain why the term matters in this paper.
4. List prerequisites.
5. Record the provider and model for both synthesized fields.
6. Treat cached snippets alone as incomplete enrichment.
7. Keep the card short enough for a popup.

## Output JSON

```json
{
  "term": "...",
  "paper_specific_meaning": "...",
  "general_explanation": "...",
  "why_it_matters_here": "...",
  "prerequisites": ["..."],
  "paper_evidence": ["..."],
  "web_sources": [{"title": "...", "url": "..."}],
  "general_explanation_source": "tavily_research",
  "general_explanation_model": "mini",
  "why_it_matters_source": "paper_agent",
  "why_it_matters_model": "ollama:model-name"
}
```

## Gotchas

- Never invent URLs.
- Label background that comes from web context separately from paper evidence.
- Do not use a deterministic placeholder after Tavily sources have been fetched.
- Do not let the paper agent use web context for the paper-specific relevance field.
