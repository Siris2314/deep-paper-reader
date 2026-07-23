from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from paper_agent.parser import ParsedPaper
from paper_agent.pwc_vocabulary import VocabularyTerm, load_method_vocabulary, terms_present_in_text
from paper_agent.term_memory import load_learned_terms, term_memory_path
from paper_agent.workspace import Workspace


@dataclass(frozen=True)
class SignificantTerm:
    term: str
    category: str
    score: float
    paper_occurrences: int
    method_occurrences: int
    evidence: list[str]
    skills: list[str]
    learned: bool = False
    correction_count: int = 0
    vocabulary_source: str = "paper_local"


STOP_TERMS = {
    "abstract",
    "appendix",
    "conclusion",
    "figure",
    "introduction",
    "method",
    "paper",
    "page",
    "pages",
    "references",
    "section",
    "table",
}

# These signals classify discovered terms; they are not a candidate vocabulary.
MATH_SIGNALS = (
    "loss",
    "objective",
    "gradient",
    "divergence",
    "softmax",
    "logit",
    "probability",
    "cross-entropy",
    "cross entropy",
)
RESULT_SIGNALS = ("benchmark", "metric", "ablation", "baseline", "accuracy", "evaluation")
LOCAL_STOP_WORDS = STOP_TERMS | {
    "about",
    "after",
    "also",
    "before",
    "between",
    "from",
    "into",
    "paper",
    "that",
    "their",
    "these",
    "this",
    "using",
    "where",
    "which",
    "with",
}


def method_section_text(parsed: ParsedPaper) -> str:
    preferred = [
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "architecture",
        "proposed_method",
        "algorithm",
    ]
    parts: list[str] = []
    for key, text in parsed.sections.items():
        low = key.lower()
        if any(name in low for name in preferred):
            parts.append(text)
    if parts:
        return "\n\n".join(parts)
    fallback_keys = ["abstract", "introduction", "front_matter"]
    return "\n\n".join(parsed.sections.get(key, "") for key in fallback_keys)


def _category_for(term: str, *, from_pwc: bool = False) -> str:
    low = term.lower()
    if any(piece in low for piece in MATH_SIGNALS):
        return "math"
    if any(piece in low for piece in RESULT_SIGNALS):
        return "result"
    return "method" if from_pwc else "concept"


def _skills_for(category: str) -> list[str]:
    if category == "math":
        return ["paper-term-highlighter", "math-walkthrough", "concept-card"]
    if category == "result":
        return ["paper-term-highlighter", "figure-table-analysis", "concept-card"]
    if category == "method":
        return ["paper-term-highlighter", "implementation-reconstructor", "concept-card"]
    return ["paper-term-highlighter", "concept-card"]


def _sentence_evidence(text: str, term: str, limit: int = 3) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    hits = []
    for piece in pieces:
        clean = re.sub(r"\s+", " ", piece).strip()
        if term.lower() in clean.lower() and 35 <= len(clean) <= 520:
            hits.append(clean)
        if len(hits) >= limit:
            break
    return hits


def _candidate_terms(
    parsed: ParsedPaper,
    learned_terms: dict[str, dict[str, object]] | None = None,
    vocabulary: list[VocabularyTerm] | None = None,
) -> tuple[list[str], set[str]]:
    text = parsed.full_text
    candidates: list[str] = []
    if learned_terms:
        candidates.extend(
            str(item.get("term")) for item in learned_terms.values() if item.get("term")
        )

    archive_hits = terms_present_in_text(
        text, vocabulary if vocabulary is not None else load_method_vocabulary()
    )
    candidates.extend(item.term for item in archive_hits)
    archive_keys = {item.term.casefold() for item in archive_hits}

    # Extract paper-authored names and acronyms so newly published methods need not exist in the archive.
    introduced_rx = re.compile(
        r"\b(?:we\s+(?:introduce|propose|present)|called|referred\s+to\s+as)\s+"
        r"(?:an?\s+)?([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,5})",
        re.I,
    )
    for match in introduced_rx.finditer(text):
        phrase = re.split(
            r"\b(?:that|which|for|to|with|using)\b", match.group(1), maxsplit=1, flags=re.I
        )[0]
        phrase = re.sub(r"\s+", " ", phrase).strip(" -_:;,.()")
        if 3 <= len(phrase) <= 80:
            candidates.append(phrase)

    phrase_rx = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,})(?:[- ][A-Z][A-Za-z0-9]+){0,4}\b|\b[A-Z]{3,12}\b"
    )
    for match in phrase_rx.finditer(text):
        term = re.sub(r"\s+", " ", match.group(0)).strip(" -_:;,.()")
        if len(term) < 4 or len(term) > 72:
            continue
        low = term.lower()
        if low in LOCAL_STOP_WORDS or low.startswith("page "):
            continue
        is_acronym = bool(re.fullmatch(r"[A-Z][A-Z0-9-]{2,14}", term))
        is_camel_case_name = bool(re.search(r"[a-z][A-Z]", term))
        if (
            not is_acronym
            and not is_camel_case_name
            and (" " not in term or _count_term(text, term) < 2)
        ):
            continue
        candidates.append(term)

    seen: set[str] = set()
    out: list[str] = []
    for term in candidates:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out, archive_keys


def term_pattern(term: str) -> str:
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"(?:-\s*|\s+)")
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def _count_term(text: str, term: str) -> int:
    return len(re.findall(term_pattern(term), text, flags=re.I))


def extract_significant_terms(
    parsed: ParsedPaper,
    max_terms: int = 80,
    learned_terms: dict[str, dict[str, object]] | None = None,
    vocabulary: list[VocabularyTerm] | None = None,
) -> list[SignificantTerm]:
    learned_terms = learned_terms if learned_terms is not None else load_learned_terms()
    full = parsed.full_text
    method_text = method_section_text(parsed)
    equation_text = "\n".join(card.raw for card in parsed.equation_cards)
    visual_text = "\n".join(
        [
            *(card.caption for card in parsed.figure_cards),
            *(card.caption + "\n" + card.raw_text for card in parsed.table_cards),
        ]
    )
    terms: list[SignificantTerm] = []
    candidates, archive_keys = _candidate_terms(parsed, learned_terms, vocabulary)
    for term in candidates:
        paper_count = _count_term(full, term)
        if paper_count == 0:
            continue
        method_count = _count_term(method_text, term)
        equation_count = _count_term(equation_text, term)
        visual_count = _count_term(visual_text, term)
        learned_record = learned_terms.get(term.lower())
        from_pwc = term.casefold() in archive_keys
        category = (
            str(learned_record.get("category"))
            if learned_record
            else _category_for(term, from_pwc=from_pwc)
        )
        correction_count = int(learned_record.get("correction_count", 0)) if learned_record else 0
        base = paper_count + 2.5 * method_count + 2.0 * equation_count + 1.5 * visual_count
        if category == "math":
            base += 3
        elif category == "method":
            base += 2
        elif category == "result":
            base += 1.5
        if from_pwc:
            base += 5
        if learned_record:
            # A direct user correction outranks heuristic extraction on every future paper.
            base += 14 + min(correction_count, 10)
        if base < 2:
            continue
        evidence_source = method_text if method_count else full
        terms.append(
            SignificantTerm(
                term=term,
                category=category,
                score=round(base, 2),
                paper_occurrences=paper_count,
                method_occurrences=method_count,
                evidence=_sentence_evidence(evidence_source, term),
                skills=_skills_for(category),
                learned=bool(learned_record),
                correction_count=correction_count,
                vocabulary_source="user_memory"
                if learned_record
                else "papers_with_code_archive"
                if from_pwc
                else "paper_local",
            )
        )
    terms.sort(
        key=lambda item: (item.score, item.method_occurrences, item.paper_occurrences), reverse=True
    )
    return terms[:max_terms]


def save_significant_terms(
    parsed: ParsedPaper, workspace: Workspace, max_terms: int = 80
) -> list[SignificantTerm]:
    terms = extract_significant_terms(parsed, max_terms=max_terms)
    workspace.write_json("memory/significant_terms.json", [asdict(term) for term in terms])
    return terms


def load_significant_terms(
    parsed: ParsedPaper, workspace: Workspace, max_terms: int = 80
) -> list[SignificantTerm]:
    path = workspace.path("memory/significant_terms.json")
    global_memory = term_memory_path()
    if not path.exists() or (
        global_memory.exists() and global_memory.stat().st_mtime > path.stat().st_mtime
    ):
        return save_significant_terms(parsed, workspace, max_terms=max_terms)
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if any(isinstance(item, dict) and "vocabulary_source" not in item for item in payload):
            return save_significant_terms(parsed, workspace, max_terms=max_terms)
        return [SignificantTerm(**item) for item in payload[:max_terms] if isinstance(item, dict)]
    except Exception:
        return save_significant_terms(parsed, workspace, max_terms=max_terms)


def terms_by_category(terms: Iterable[SignificantTerm]) -> dict[str, list[SignificantTerm]]:
    grouped: dict[str, list[SignificantTerm]] = {}
    for term in terms:
        grouped.setdefault(term.category, []).append(term)
    return grouped
