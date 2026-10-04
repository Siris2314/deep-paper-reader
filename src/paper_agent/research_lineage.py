from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from paper_agent.parser import ParsedPaper, clean_line
from paper_agent.scholarly_metadata import paper_identity, scholarly_json_get
from paper_agent.workspace import Workspace


LINEAGE_CACHE_VERSION = "v3-reference-boundary"


@dataclass
class BibliographyEntry:
    number: int
    title: str
    raw: str
    year: int | None = None
    url: str | None = None
    external_id: str | None = None


@dataclass
class CitationContext:
    reference_number: int
    page: int
    text: str


@dataclass
class LineageRelation:
    reference_number: int
    title: str
    year: int | None
    url: str | None
    relation: str
    confidence: str
    explanation: str
    contexts: list[CitationContext] = field(default_factory=list)
    external: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchLineage:
    term: str
    paper_title: str
    summary: str
    relations: list[LineageRelation]
    bibliography_count: int
    external_status: str
    generated_at: str


RELATION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "extends",
        ("built upon", "builds upon", "builds on", "extends", "extension of", "following the"),
    ),
    (
        "modifies",
        (
            "replace",
            "replaces",
            "upgrade",
            "departure",
            "unlike",
            "modify",
            "modified",
            "instead of",
        ),
    ),
    ("evaluates_on", ("benchmark", "dataset", "evaluation", "suite", "task")),
    ("compares", ("baseline", "compared", "compare", "against", "outperform")),
    (
        "adopts",
        (
            "we reuse",
            "we leverage",
            "we adopt",
            "reusing the",
            "based directly on",
            "our architecture uses",
        ),
    ),
]


def _references_text(parsed: ParsedPaper) -> str:
    for name, text in parsed.sections.items():
        if name in {"references", "bibliography"}:
            return text
    match = re.search(
        r"(?:^|\n)\s*(?:references|bibliography)\s*\n(.+)$", parsed.full_text, re.I | re.S
    )
    return match.group(1) if match else ""


def _entry_title(raw: str) -> str:
    no_url = re.sub(r"https?\s*:\s*//\S+", "", raw, flags=re.I)
    parts = [clean_line(item) for item in re.split(r"\.\s+", no_url) if clean_line(item)]
    candidates = [
        item.rstrip(".")
        for item in parts
        if not re.fullmatch(r"(?:19|20)\d{2}", item)
        and len(re.findall(r"[A-Za-z][A-Za-z0-9-]*", item)) >= 3
        and not re.match(r"^(?:arxiv|proceedings|technical report|doi)\b", item, re.I)
    ]
    if candidates:
        punctuated = [item for item in candidates if re.search(r"[:?]", item)]
        return (punctuated[-1] if punctuated else candidates[-1])[:300]
    return clean_line(no_url)[:300]


def parse_bibliography(parsed: ParsedPaper) -> list[BibliographyEntry]:
    text = _references_text(parsed)
    matches = list(re.finditer(r"(?m)^\s*\[(\d+)\]\s*", text))
    entries: list[BibliographyEntry] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw = clean_line(text[match.end() : end])
        if not raw:
            continue
        url_match = re.search(r"https?\s*:\s*//\S+", raw, re.I)
        url = re.sub(r"\s+", "", url_match.group(0)) if url_match else None
        if url_match and url and url.endswith("/"):
            wrapped_tail = re.match(r"([A-Za-z0-9_.~-]+)", raw[url_match.end() :].lstrip())
            if wrapped_tail:
                url += wrapped_tail.group(1)
        if url:
            url = url.rstrip(".,;)")
        years = re.findall(r"\b(?:19|20)\d{2}\b", raw)
        entries.append(
            BibliographyEntry(
                number=int(match.group(1)),
                title=_entry_title(raw),
                raw=raw,
                year=int(years[-1]) if years else None,
                url=url,
            )
        )
    return entries


def _citation_numbers(marker: str) -> set[int]:
    numbers: set[int] = set()
    for part in marker.split(","):
        clean = part.strip()
        range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", clean)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            numbers.update(range(start, min(end, start + 20) + 1))
        elif clean.isdigit():
            numbers.add(int(clean))
    return numbers


def citation_contexts(parsed: ParsedPaper) -> dict[int, list[CitationContext]]:
    contexts: dict[int, list[CitationContext]] = {}
    reference_pages = [
        page
        for page, text in parsed.page_text.items()
        if re.search(r"^\s*(?:references|bibliography)\s*$", text, re.I | re.M)
    ]
    reference_start_page = min(reference_pages) if reference_pages else None
    for page, text in parsed.page_text.items():
        if reference_start_page is not None and page > reference_start_page:
            continue
        reference_heading = re.search(r"^\s*(?:references|bibliography)\s*$", text, re.I | re.M)
        if reference_heading:
            text = text[: reference_heading.start()]
        if not text.strip():
            continue
        compact = clean_line(text)
        for match in re.finditer(r"\[([0-9,\s–-]+)\]", compact):
            start = max(0, match.start() - 360)
            end = min(len(compact), match.end() + 360)
            window = compact[start:end].strip()
            for number in _citation_numbers(match.group(1)):
                item = CitationContext(number, page, window)
                bucket = contexts.setdefault(number, [])
                if not any(existing.text == item.text for existing in bucket):
                    bucket.append(item)
    return contexts


def _relation(contexts: list[CitationContext]) -> tuple[str, str, str]:
    focused: list[str] = []
    for item in contexts:
        marker = re.search(rf"\[[^\]]*\b{item.reference_number}\b[^\]]*\]", item.text)
        if marker:
            before = item.text[: marker.start()]
            after = item.text[marker.end() :]
            left_breaks = [match.end() for match in re.finditer(r"[.!?]\s+", before)]
            right_break = re.search(r"[.!?](?:\s+|$)", after)
            start = left_breaks[-1] if left_breaks else max(0, marker.start() - 180)
            end = marker.end() + (right_break.end() if right_break else min(len(after), 180))
            focused.append(item.text[start:end])
        else:
            focused.append(item.text)
    combined = " ".join(focused).casefold()
    for relation, needles in RELATION_PATTERNS:
        matched = [needle for needle in needles if needle in combined]
        if matched:
            confidence = "high" if len(matched) >= 2 or relation == "extends" else "medium"
            label = {
                "extends": "The paper explicitly presents its method as building on this work.",
                "modifies": "The paper changes or replaces part of this earlier approach.",
                "adopts": "The paper reuses an architecture, component, or technique from this work.",
                "compares": "The paper uses this work as a comparison or baseline.",
                "evaluates_on": "The paper uses this work as an evaluation task or benchmark.",
            }[relation]
            return relation, confidence, label
    return (
        "background",
        "low",
        "The paper cites this work as background; a stronger dependency is not explicit in the extracted context.",
    )


def _term_score(term: str, entry: BibliographyEntry, contexts: list[CitationContext]) -> int:
    tokens = [
        item
        for item in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", term.casefold())
        if item not in {"the", "and", "with"}
    ]
    if not tokens:
        return 1
    haystack = f"{entry.title} {entry.raw} {' '.join(item.text for item in contexts)}".casefold()
    return sum(3 if token in entry.title.casefold() else 1 for token in tokens if token in haystack)


def _external_paper_id(parsed: ParsedPaper) -> str | None:
    identity = paper_identity(parsed)
    if identity.get("arxiv_id"):
        return f"ARXIV:{identity['arxiv_id']}"
    if identity.get("doi"):
        return f"DOI:{identity['doi']}"
    return None


def _semantic_scholar_headers() -> dict[str, str]:
    headers = {"User-Agent": "DeepPaperReader/0.1 research-lineage"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", os.getenv("S2_API_KEY", "")).strip()
    if key:
        headers["x-api-key"] = key
    return headers


def _semantic_scholar_json(url: str, timeout: float = 12.0) -> dict[str, Any]:
    del timeout
    return scholarly_json_get(url, _semantic_scholar_headers())


def semantic_scholar_references(
    parsed: ParsedPaper, limit: int = 100
) -> tuple[str, list[dict[str, Any]]]:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", os.getenv("S2_API_KEY", "")).strip()
    allow_shared = os.getenv("SEMANTIC_SCHOLAR_ALLOW_UNAUTHENTICATED", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    if not key and not allow_shared:
        return "not_configured", []
    external_id = _external_paper_id(parsed)
    if not external_id:
        return "paper_id_unresolved", []
    paper_id = urllib.parse.quote(external_id, safe=":")
    fields = (
        "contexts,intents,isInfluential,title,year,url,abstract,authors,citationCount,externalIds"
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields={urllib.parse.quote(fields)}&limit={max(1, min(limit, 500))}"
    try:
        payload = _semantic_scholar_json(url)
    except urllib.error.HTTPError as exc:
        return f"http_{exc.code}", []
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}", []
    return "ok", [item for item in payload.get("data", []) if isinstance(item, dict)]


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _merge_external(entry: BibliographyEntry, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    target = set(_normalize_title(entry.title).split())
    best: tuple[float, dict[str, Any]] | None = None
    for item in candidates:
        paper = item.get("citedPaper") if isinstance(item.get("citedPaper"), dict) else {}
        words = set(_normalize_title(str(paper.get("title") or "")).split())
        score = len(target & words) / max(1, len(target | words))
        if score >= 0.45 and (best is None or score > best[0]):
            best = (score, item)
    if not best:
        return {}
    item = best[1]
    paper = item.get("citedPaper") if isinstance(item.get("citedPaper"), dict) else {}
    return {
        "paperId": paper.get("paperId"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "url": paper.get("url"),
        "abstract": paper.get("abstract"),
        "citationCount": paper.get("citationCount"),
        "authors": paper.get("authors"),
        "contexts": item.get("contexts", []),
        "intents": item.get("intents", []),
        "isInfluential": item.get("isInfluential"),
    }


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:70]
    digest = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:8]
    return f"{clean or 'paper'}-{digest}"


def build_research_lineage(
    parsed: ParsedPaper,
    workspace: Workspace,
    term: str = "",
    *,
    refresh: bool = False,
) -> ResearchLineage:
    clean_term = clean_line(term)[:160]
    cache_rel = (
        f"research/lineage/{LINEAGE_CACHE_VERSION}-{_slug(clean_term or 'paper-overview')}.json"
    )
    cache = workspace.path(cache_rel)
    if cache.exists() and not refresh:
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            payload["relations"] = [
                LineageRelation(
                    **{
                        **item,
                        "contexts": [
                            CitationContext(**context) for context in item.get("contexts", [])
                        ],
                    }
                )
                for item in payload.get("relations", [])
            ]
            return ResearchLineage(**payload)
        except Exception:
            pass

    bibliography = parse_bibliography(parsed)
    context_map = citation_contexts(parsed)
    external_status, external_candidates = semantic_scholar_references(parsed)
    ranked: list[tuple[int, int, LineageRelation]] = []
    relation_priority = {
        "extends": 6,
        "modifies": 5,
        "adopts": 4,
        "compares": 3,
        "evaluates_on": 2,
        "background": 1,
    }
    for entry in bibliography:
        contexts = context_map.get(entry.number, [])
        score = _term_score(clean_term, entry, contexts)
        relation, confidence, explanation = _relation(contexts)
        if clean_term and score == 0 and relation not in {"extends", "modifies"}:
            continue
        external = _merge_external(entry, external_candidates)
        ranked.append(
            (
                score,
                relation_priority[relation],
                LineageRelation(
                    reference_number=entry.number,
                    title=str(external.get("title") or entry.title),
                    year=int(external.get("year") or entry.year)
                    if (external.get("year") or entry.year)
                    else None,
                    url=str(external.get("url") or entry.url or "") or None,
                    relation=relation,
                    confidence=confidence,
                    explanation=explanation,
                    contexts=contexts[:3],
                    external=external,
                ),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    relations = [item[2] for item in ranked[:8]]
    strong = [item for item in relations if item.relation in {"extends", "modifies", "adopts"}]
    if strong:
        names = ", ".join(item.title for item in strong[:3])
        summary = f"The paper has explicit architectural or methodological dependencies on {names}. Open each relation to inspect the paper's own citation context."
    elif relations:
        summary = "Related references were found, but the extracted text does not establish a strong build-upon relationship. They are labeled conservatively."
    else:
        summary = "No matching prior-work relationship was resolved from the paper's bibliography and citation contexts."
    result = ResearchLineage(
        term=clean_term,
        paper_title=parsed.metadata.title_guess or "Untitled paper",
        summary=summary,
        relations=relations,
        bibliography_count=len(bibliography),
        external_status=external_status,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    workspace.write_json(cache_rel, _lineage_dict(result))
    return result


def _lineage_dict(result: ResearchLineage) -> dict[str, Any]:
    return {
        "term": result.term,
        "paper_title": result.paper_title,
        "summary": result.summary,
        "relations": [
            {
                **item.__dict__,
                "contexts": [context.__dict__ for context in item.contexts],
            }
            for item in result.relations
        ],
        "bibliography_count": result.bibliography_count,
        "external_status": result.external_status,
        "generated_at": result.generated_at,
    }


def research_lineage_payload(result: ResearchLineage) -> dict[str, Any]:
    return _lineage_dict(result)
