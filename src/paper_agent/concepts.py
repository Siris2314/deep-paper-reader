from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from paper_agent.parser import ParsedPaper
from paper_agent.workspace import Workspace


CONCEPT_CARD_VERSION = 3


@dataclass
class ConceptCard:
    term: str
    slug: str
    paper_specific_meaning: str
    general_explanation: str
    why_it_matters_here: str
    prerequisites: list[str]
    paper_evidence: list[str]
    web_sources: list[dict[str, str]]
    status: str = "cached"
    general_explanation_source: str = "deterministic"
    general_explanation_model: str = "none"
    why_it_matters_source: str = "deterministic"
    why_it_matters_model: str = "none"
    card_version: int = CONCEPT_CARD_VERSION


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    return "-".join(part for part in cleaned.split("-") if part)[:80] or "concept"


def sentence_windows(text: str, term: str, limit: int = 3) -> list[str]:
    # PDF extraction wraps prose at visual line boundaries. Rejoin those lines before
    # sentence splitting so evidence cards do not end halfway through a sentence.
    normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    normalized = re.sub(r"\s*\n\s*", " ", normalized)
    chunks = re.split(r"(?<=[.!?])\s+", normalized)
    hits = [_clean_evidence_sentence(c) for c in chunks if term.lower() in c.lower()]
    hits = [item for item in hits if 40 <= len(item) <= 600]
    dedup: list[str] = []
    for hit in hits:
        if hit not in dedup:
            dedup.append(hit)
        if len(dedup) >= limit:
            break
    return dedup


def _clean_evidence_sentence(text: str) -> str:
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    if ":" in clean:
        prose, tail = clean.split(":", 1)
        math_markers = len(re.findall(r"[=∈∑∏≤≥]|\b(?:Top-k|softmax|Score\w*)\b", tail, re.I))
        identifier_runs = len(re.findall(r"\b[A-Za-z]+(?:Comp|Score|Mem)[A-Za-z]*\b", tail))
        if len(prose) >= 45 and math_markers + identifier_runs >= 2:
            clean = prose.rstrip(" .") + "."
    return clean


def _introduced_variant(term: str, evidence: list[str]) -> str | None:
    escaped = re.escape(term)
    candidates: list[str] = []
    for sentence in evidence:
        for match in re.finditer(rf"\b(?:[A-Z][A-Za-z0-9-]+\s+){{1,3}}{escaped}\b", sentence):
            phrase = match.group(0).strip()
            if phrase.casefold() != term.casefold() and phrase not in candidates:
                candidates.append(phrase)
    introduced = [
        phrase
        for phrase in candidates
        if any(
            re.search(rf"\b(?:propose|present|introduce)\b[^.]*\b{re.escape(phrase)}\b", item, re.I)
            for item in evidence
        )
    ]
    return (introduced or candidates or [None])[0]


def extract_candidate_concepts(parsed: ParsedPaper, max_terms: int = 80) -> list[str]:
    # Keep concept cards and PDF highlights on one ranked vocabulary path.
    from paper_agent.term_highlighter import extract_significant_terms

    return [item.term for item in extract_significant_terms(parsed, max_terms=max_terms)]


def _load_web_sources_from_workspace(
    workspace: Workspace, term: str, max_sources: int = 5
) -> list[dict[str, str]]:
    sources_path = workspace.path("web/tavily_sources.json")
    if not sources_path.exists():
        return []
    try:
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    hits: list[dict[str, str]] = []
    for src in sources:
        hay = json.dumps(src, ensure_ascii=False).lower()
        if term.lower() in hay:
            hits.append(
                {
                    "title": str(src.get("title", ""))[:200],
                    "url": str(src.get("url", ""))[:500],
                    "snippet": str(src.get("content", src.get("snippet", "")))[:600],
                }
            )
        if len(hits) >= max_sources:
            break
    return hits


def create_deterministic_concept_card(
    parsed: ParsedPaper, workspace: Workspace, term: str
) -> ConceptCard:
    evidence = sentence_windows(parsed.full_text, term, limit=4)
    title = parsed.metadata.title_guess or "this paper"
    paper_specific = (
        f"In `{title}`, `{term}` appears in the paper context shown below. "
        "The exact role should be interpreted from the paper evidence, not generic background alone."
    )
    if evidence:
        paper_specific = evidence[0]
        variant = _introduced_variant(term, evidence)
        if variant:
            paper_specific = (
                f"The paper uses {term} as a broader mechanism family and introduces {variant} "
                "as its paper-specific variant."
            )

    general = (
        f"`{term}` is a background concept relevant to understanding the paper. "
        "Use Tavily research mode to enrich this card with external sources if the cached paper evidence is not enough."
    )
    why = (
        f"This matters because the paper uses or mentions `{term}` as part of its method, evaluation, related work, "
        "or technical framing. Click into the source snippets before treating it as central."
    )
    prereqs = []
    t = term.lower()
    if "attention" in t or "transformer" in t or "kv" in t:
        prereqs = ["transformers", "attention", "tokenization", "inference memory"]
    elif "grpo" in t or "reinforcement" in t or "reward" in t:
        prereqs = ["reinforcement learning", "rollouts", "reward models", "policy optimization"]
    elif "self-instruct" in t or "synthetic" in t or "agentic" in t:
        prereqs = ["instruction tuning", "synthetic data", "LLM evaluation", "agent loops"]
    elif "loss" in t or "contrastive" in t or "encoder" in t:
        prereqs = ["embeddings", "optimization", "supervised learning"]

    return ConceptCard(
        term=term,
        slug=slugify(term),
        paper_specific_meaning=paper_specific,
        general_explanation=general,
        why_it_matters_here=why,
        prerequisites=prereqs,
        paper_evidence=evidence,
        web_sources=_load_web_sources_from_workspace(workspace, term),
        status="paper_only",
    )


def save_concept_index(parsed: ParsedPaper, workspace: Workspace) -> list[str]:
    concepts = extract_candidate_concepts(parsed)
    workspace.write_json("concepts/concepts.json", concepts)
    workspace.write_json(
        "concepts/concept_index.json",
        [{"term": t, "slug": slugify(t)} for t in concepts],
    )
    return concepts


def get_or_create_concept_card(parsed: ParsedPaper, workspace: Workspace, term: str) -> ConceptCard:
    slug = slugify(term)
    rel = f"concepts/cards/{slug}.json"
    path = workspace.path(rel)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("card_version") or 0) != CONCEPT_CARD_VERSION:
                refreshed = create_deterministic_concept_card(parsed, workspace, term)
                for name in (
                    "general_explanation",
                    "why_it_matters_here",
                    "web_sources",
                    "status",
                    "general_explanation_source",
                    "general_explanation_model",
                    "why_it_matters_source",
                    "why_it_matters_model",
                ):
                    if name in payload:
                        setattr(refreshed, name, payload[name])
                workspace.write_json(rel, asdict(refreshed))
                return refreshed
            return ConceptCard(**payload)
        except Exception:
            pass
    card = create_deterministic_concept_card(parsed, workspace, term)
    workspace.write_json(rel, asdict(card))
    return card


def concept_card_to_markdown(card: ConceptCard) -> str:
    srcs = (
        "\n".join(f"- {s.get('title') or s.get('url')}: {s.get('url')}" for s in card.web_sources)
        or "- none cached"
    )
    evidence = (
        "\n".join(f"- {e}" for e in card.paper_evidence) or "- no direct sentence evidence found"
    )
    prereqs = ", ".join(card.prerequisites) if card.prerequisites else "not inferred"
    return f"""# {card.term}

**Paper-specific meaning:** {card.paper_specific_meaning}

**General explanation:** {card.general_explanation}

**Why it matters here:** {card.why_it_matters_here}

**Prerequisites:** {prereqs}

## Paper evidence
{evidence}

## Cached web sources
{srcs}
"""
