---
name: research-lineage
description: Trace how an uploaded research paper builds on, modifies, adopts, compares with, or evaluates against earlier work using its bibliography and exact citation contexts, with optional Semantic Scholar graph enrichment. Use for prior-work questions, concept ancestry, novelty claims, method lineage, citation relationships, foundational papers, or requests asking what a paper extends.
---

# Research Lineage

Build conservative, evidence-visible links between the current paper and cited prior work.

## Workflow

1. Parse numbered bibliography entries from the uploaded paper.
2. Retrieve every in-paper context that cites each entry.
3. Filter relationships by the requested concept or method when one is supplied.
4. Classify each relationship as `extends`, `modifies`, `adopts`, `compares`, `evaluates_on`, or `background`.
5. Require explicit citation-context language for `extends`, `modifies`, or `adopts`. A graph edge alone proves only that a citation exists.
6. Prefer the paper's own citation sentence over abstracts, recommendation similarity, or citation counts.
7. Enrich titles, abstracts, authors, citation intent, and influential-citation metadata through Semantic Scholar only when configured.
8. Return the exact source page and citation context with every relationship.
9. Mark ambiguous relationships as `background` instead of guessing a stronger dependency.
10. Keep newer citing work separate from earlier referenced work; direction matters.

## Output

For every connected paper, provide:

- bibliography number and title
- relationship label and confidence
- one-sentence explanation
- exact current-paper citation context and page
- external URL and metadata when available
- uncertainty or conflicting evidence

## Guardrails

- Do not call a paper foundational because it has many citations.
- Do not claim that the current paper extends another merely because both mention the same concept.
- Do not let external summaries override the uploaded paper's stated relationship.
- Separate architectural inheritance, training-objective inheritance, benchmark lineage, and general background.
- Cache lineage by paper and concept so repeated clicks do not repeat graph requests.
