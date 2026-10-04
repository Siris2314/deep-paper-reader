from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from paper_agent.parser import ParsedPaper
from paper_agent.pwc_vocabulary import VocabularyTerm, load_method_vocabulary, terms_present_in_text
from paper_agent.term_feedback import feedback_path, relevance_key, suppressed_relevance_keys
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
    extraction_version: int = 4


EXTRACTION_VERSION = 4


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

# Paper metadata and broad singleton words are poor automatic annotations. User memory can still
# promote a broad singleton such as "model", while metadata remains excluded.
METADATA_TERMS = {
    "arxiv",
    "bibliography",
    "copyright",
    "doi",
    "preprint",
    "submitted",
}
GENERIC_SINGLETONS = {
    "algorithm",
    "data",
    "learning",
    "method",
    "model",
    "models",
    "result",
    "system",
    "task",
    "training",
}
NON_CONCEPT_ACRONYMS = {
    "API",
    "CSV",
    "HTML",
    "HTTP",
    "HTTPS",
    "JSON",
    "PDF",
    "URL",
    "XML",
}

# These are grammatical heads used to find phrases authored by the paper, not a catalog of
# ML concepts. For example, they let the paper contribute "large unsupervised language models"
# as one span instead of highlighting only "model".
RESEARCH_PHRASE_HEADS = {
    "agent",
    "agents",
    "algorithm",
    "algorithms",
    "architecture",
    "architectures",
    "assistant",
    "assistants",
    "attention",
    "benchmark",
    "benchmarks",
    "cache",
    "classifier",
    "classifiers",
    "corpus",
    "dataset",
    "datasets",
    "decoder",
    "decoders",
    "distribution",
    "distributions",
    "embedding",
    "embeddings",
    "encoder",
    "encoders",
    "evaluation",
    "feedback",
    "framework",
    "frameworks",
    "inference",
    "learning",
    "loss",
    "losses",
    "metric",
    "metrics",
    "model",
    "models",
    "objective",
    "objectives",
    "optimization",
    "policy",
    "policies",
    "preference",
    "preferences",
    "representation",
    "representations",
    "retrieval",
    "reward",
    "sampling",
    "training",
}
# These signals classify discovered terms; they are not a candidate vocabulary.
MATH_SIGNALS = (
    "activation",
    "binary cross-entropy",
    "binary cross entropy",
    "loss",
    "objective",
    "gradient",
    "divergence",
    "relu",
    "sigmoid",
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
PHRASE_BOUNDARIES = LOCAL_STOP_WORDS | {
    "a",
    "all",
    "although",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "called",
    "corresponding",
    "could",
    "either",
    "do",
    "does",
    "during",
    "each",
    "every",
    "for",
    "has",
    "have",
    "its",
    "in",
    "is",
    "it",
    "many",
    "may",
    "most",
    "more",
    "new",
    "no",
    "of",
    "on",
    "one",
    "or",
    "other",
    "our",
    "several",
    "same",
    "some",
    "than",
    "the",
    "those",
    "three",
    "to",
    "two",
    "unknown",
    "use",
    "used",
    "uses",
    "we",
    "were",
    "will",
    "while",
    "would",
    "whose",
    "your",
}

# Phrase extraction walks left from a technical noun. These words usually turn a useful noun
# phrase into a sentence fragment ("throughout training", "via metric learning", and so on).
# Stopping at them retains the useful suffix without maintaining a domain-specific term list.
PHRASE_BOUNDARIES |= {
    "across",
    "both",
    "caused",
    "compare",
    "compared",
    "comparing",
    "convert",
    "create",
    "created",
    "creates",
    "enables",
    "evaluate",
    "evaluates",
    "extensive",
    "final",
    "fixed",
    "full",
    "generate",
    "generated",
    "generates",
    "higher",
    "improve",
    "improved",
    "improves",
    "incorporating",
    "increased",
    "introduce",
    "introduces",
    "main",
    "isolates",
    "learn",
    "overall",
    "optimize",
    "optimizes",
    "outperform",
    "outperformed",
    "outperforming",
    "outperforms",
    "provides",
    "quality",
    "questions",
    "reads",
    "reducing",
    "simply",
    "specifically",
    "through",
    "throughout",
    "trained",
    "transitioning",
    "via",
    "very",
    "avoid",
    "avoids",
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


def _research_text(parsed: ParsedPaper) -> str:
    sections: list[str] = []
    for name, section_text in parsed.sections.items():
        low_name = name.casefold()
        if "references" in low_name or "bibliography" in low_name:
            break
        if "appendix" not in low_name and "supplement" not in low_name:
            sections.append(section_text)
    text = "\n".join(sections).strip() or parsed.full_text
    clean_lines: list[str] = []
    metadata_line = re.compile(
        r"(?:\barxiv\s*:|\bdoi\s*:|https?://arxiv\.org|^\s*---\s*page\s+\d+\s*---\s*$)",
        re.I,
    )
    for line in text.splitlines():
        if metadata_line.search(line):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def _candidate_is_noise(term: str) -> bool:
    low = term.casefold().strip()
    if not low or low in PHRASE_BOUNDARIES or low in METADATA_TERMS:
        return True
    if re.search(r"\barxiv\b|\bdoi\b|\bcs\.[a-z]{2}\b", low):
        return True
    if re.fullmatch(r"(?:19|20)\d{2}", low) or re.fullmatch(r"\d+(?:\.\d+){1,3}v?\d*", low):
        return True
    tokens = re.findall(r"[a-z0-9]+", low)
    if tokens and (tokens[0] in PHRASE_BOUNDARIES or tokens[-1] in PHRASE_BOUNDARIES):
        return True
    if tokens and (len(tokens[0]) == 1 or len(tokens[-1]) == 1):
        return True
    return False


def _clean_candidate(term: str) -> str:
    term = re.sub(r"\s+", " ", term).strip(" -_:;,.()[]{}")
    term = re.sub(r"^(?:an?|the|our|their|this|these)\s+", "", term, flags=re.I)
    return term.strip(" -_:;,.()[]{}")


def _paper_phrase_candidates(text: str) -> list[str]:
    phrases: list[str] = []
    token_rx = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
    for sentence in re.split(r"(?<=[.!?;:])\s+|\n+", text):
        tokens = token_rx.findall(sentence)
        for index, token in enumerate(tokens):
            if token.casefold() not in RESEARCH_PHRASE_HEADS:
                continue
            left = index
            while left > 0 and index - left < 4:
                previous = tokens[left - 1].casefold()
                if previous in PHRASE_BOUNDARIES:
                    break
                left -= 1
            window = tokens[left : index + 1]
            for size in range(2, min(5, len(window)) + 1):
                phrase = _clean_candidate(" ".join(window[-size:]))
                if 5 <= len(phrase) <= 90:
                    phrases.append(phrase)
    return phrases


def _acronym_initials(words: list[str]) -> str:
    return "".join(
        piece[0]
        for word in words
        for piece in re.findall(r"[A-Za-z]+", word)
        if piece.casefold() not in {"and", "for", "from", "of", "the", "with"}
    ).upper()


def _acronym_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pattern = re.compile(
        r"\b([A-Za-z][A-Za-z-]*(?:\s+(?:from|of|for|with|and|[A-Za-z][A-Za-z-]*)){1,7})"
        r"\s*\(([A-Z][A-Z0-9-]{1,14})\)"
    )
    for match in pattern.finditer(text):
        acronym = match.group(2)
        words = match.group(1).split()
        # Parenthetical definitions often follow a full clause. Select the shortest suffix whose
        # initials match, e.g. "Binary Cross-Entropy" rather than "we minimize ... Binary...".
        long_form = ""
        for start in range(len(words) - 1, -1, -1):
            suffix = words[start:]
            if len(suffix) >= 2 and _acronym_initials(suffix) == acronym:
                long_form = _clean_candidate(" ".join(suffix))
                break
        if long_form:
            pairs.append((long_form, acronym))
    return pairs


def _candidate_terms(
    parsed: ParsedPaper,
    learned_terms: dict[str, dict[str, object]] | None = None,
    vocabulary: list[VocabularyTerm] | None = None,
) -> tuple[list[str], dict[str, str]]:
    text = _research_text(parsed)
    candidates: list[tuple[str, str]] = []
    if learned_terms:
        candidates.extend(
            (str(item.get("term")), "user_memory")
            for item in learned_terms.values()
            if item.get("term")
        )

    archive_hits = terms_present_in_text(
        text, vocabulary if vocabulary is not None else load_method_vocabulary()
    )
    candidates.extend((item.term, "papers_with_code_archive") for item in archive_hits)

    # Extract paper-authored names and acronyms so newly published methods need not exist in the archive.
    introduced_rx = re.compile(
        r"\b(?:we[ \t]+(?:call|introduce|propose|present)|called|referred[ \t]+to[ \t]+as)[ \t]+"
        r"(?:an?[ \t]+)?([A-Za-z][A-Za-z0-9-]*(?:[ \t]+[A-Za-z][A-Za-z0-9-]*){0,5})",
        re.I,
    )
    for match in introduced_rx.finditer(text):
        phrase = re.split(
            r"\b(?:that|which|for|to|with|using)\b", match.group(1), maxsplit=1, flags=re.I
        )[0]
        phrase = re.sub(r"\s+", " ", phrase).strip(" -_:;,.()")
        if 3 <= len(phrase) <= 80:
            candidates.append((phrase, "paper_introduced"))

    acronym_pairs = _acronym_pairs(text)
    defined_acronyms = {acronym.casefold() for _, acronym in acronym_pairs}
    for long_form, acronym in acronym_pairs:
        if acronym in NON_CONCEPT_ACRONYMS:
            continue
        # Two-letter abbreviations are especially collision-prone. Keep them only when their
        # expansion itself names a research concept (RL, KV, and similar technical definitions).
        if len(acronym) == 2 and not any(
            token.casefold() in RESEARCH_PHRASE_HEADS
            for token in re.findall(r"[A-Za-z]+", long_form)
        ):
            continue
        candidates.extend(((long_form, "paper_acronym"), (acronym, "paper_acronym")))
    candidates.extend((term, "paper_phrase") for term in _paper_phrase_candidates(text))

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
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", term)
        ends_in_research_head = bool(tokens and tokens[-1].casefold() in RESEARCH_PHRASE_HEADS)
        count = _count_term(text, term)
        if is_acronym and (
            term in NON_CONCEPT_ACRONYMS or (count < 2 and low not in defined_acronyms)
        ):
            continue
        if (
            not is_acronym
            and not is_camel_case_name
            and (" " not in term or not ends_in_research_head or (count < 2 and len(tokens) < 3))
        ):
            continue
        candidates.append((term, "paper_local"))

    seen: set[str] = set()
    out: list[str] = []
    sources: dict[str, str] = {}
    source_priority = {
        "paper_local": 0,
        "paper_phrase": 1,
        "paper_acronym": 2,
        "paper_introduced": 3,
        "papers_with_code_archive": 4,
        "user_memory": 5,
    }
    for raw_term, source in candidates:
        term = _clean_candidate(raw_term)
        key = term.casefold()
        if _candidate_is_noise(term):
            continue
        if source != "user_memory" and " " not in term and key in GENERIC_SINGLETONS:
            continue
        if (
            source == "papers_with_code_archive"
            and " " not in term
            and not re.fullmatch(r"[A-Z][A-Z0-9-]{2,14}", term)
            and not re.search(r"[a-z][A-Z]", term)
            and _count_term(text, term) < 2
        ):
            continue
        if key in seen:
            if source_priority[source] > source_priority[sources[key]]:
                sources[key] = source
            continue
        seen.add(key)
        out.append(term)
        sources[key] = source
    return out, sources


def term_pattern(term: str) -> str:
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"(?:-\s*|\s+)")
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def _count_term(text: str, term: str) -> int:
    return len(re.findall(term_pattern(term), text, flags=re.I))


def _prune_duplicate_phrase_spans(terms: list[SignificantTerm]) -> list[SignificantTerm]:
    # Spacing and hyphenation variants match the same text through term_pattern(). Keep one card
    # and one ranking entry for variants such as "CoT-Self-Instruct" / "CoT Self-Instruct".
    aliases: dict[str, SignificantTerm] = {}
    for item in terms:
        key = "".join(re.findall(r"[a-z0-9]+", item.term.casefold()))
        existing = aliases.get(key)
        if existing is None or (item.learned, item.score, len(item.term)) > (
            existing.learned,
            existing.score,
            len(existing.term),
        ):
            aliases[key] = item

    selected: list[SignificantTerm] = []
    phrase_terms = sorted(
        aliases.values(),
        key=lambda item: (item.learned, item.score, len(item.term.split())),
        reverse=True,
    )
    for item in phrase_terms:
        item_tokens = re.findall(r"[a-z0-9]+", item.term.casefold())
        redundant = False
        if item.vocabulary_source == "paper_phrase" and not item.learned:
            for existing in selected:
                if (
                    existing.vocabulary_source != "paper_phrase"
                    or abs(existing.paper_occurrences - item.paper_occurrences) > 1
                ):
                    continue
                existing_tokens = re.findall(r"[a-z0-9]+", existing.term.casefold())
                shorter, longer = sorted((item_tokens, existing_tokens), key=len)
                for start in range(len(longer) - len(shorter) + 1):
                    if longer[start : start + len(shorter)] == shorter:
                        redundant = True
                        break
                if redundant:
                    break
        if not redundant:
            selected.append(item)
    return selected


def extract_significant_terms(
    parsed: ParsedPaper,
    max_terms: int = 160,
    learned_terms: dict[str, dict[str, object]] | None = None,
    vocabulary: list[VocabularyTerm] | None = None,
) -> list[SignificantTerm]:
    learned_terms = learned_terms if learned_terms is not None else load_learned_terms()
    full = _research_text(parsed)
    method_text = method_section_text(parsed)
    focus_text = "\n".join(
        section_text
        for name, section_text in parsed.sections.items()
        if any(
            label in name.casefold() for label in ("abstract", "approach", "introduction", "method")
        )
    )
    equation_text = "\n".join(card.raw for card in parsed.equation_cards)
    visual_text = "\n".join(
        [
            *(card.caption for card in parsed.figure_cards),
            *(card.caption + "\n" + card.raw_text for card in parsed.table_cards),
        ]
    )
    terms: list[SignificantTerm] = []
    candidates, candidate_sources = _candidate_terms(parsed, learned_terms, vocabulary)
    for term in candidates:
        paper_count = _count_term(full, term)
        if paper_count == 0:
            continue
        method_count = _count_term(method_text, term)
        focus_count = _count_term(focus_text, term)
        equation_count = _count_term(equation_text, term)
        visual_count = _count_term(visual_text, term)
        learned_record = learned_terms.get(term.lower())
        source = candidate_sources[term.casefold()]
        from_method_source = source in {"papers_with_code_archive", "paper_introduced"}
        category = (
            str(learned_record.get("category"))
            if learned_record
            else _category_for(term, from_pwc=from_method_source)
        )
        correction_count = int(learned_record.get("correction_count", 0)) if learned_record else 0
        base = (
            min(paper_count, 16)
            + 2.5 * min(method_count, 8)
            + 1.5 * min(focus_count, 8)
            + 2.0 * min(equation_count, 5)
            + 1.5 * min(visual_count, 5)
        )
        if category == "math":
            base += 3
        elif category == "method":
            base += 2
        elif category == "result":
            base += 1.5
        if source == "papers_with_code_archive":
            base += 5
        if source == "paper_introduced":
            base += 4
        elif source == "paper_acronym":
            base += 3
        elif source == "paper_phrase":
            base += 2 + min(len(term.split()) - 1, 3) * 0.75
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
                vocabulary_source=source,
                extraction_version=EXTRACTION_VERSION,
            )
        )
    terms = _prune_duplicate_phrase_spans(terms)
    terms.sort(
        key=lambda item: (item.score, item.method_occurrences, item.paper_occurrences), reverse=True
    )
    return terms[:max_terms]


def save_significant_terms(
    parsed: ParsedPaper, workspace: Workspace, max_terms: int = 160
) -> list[SignificantTerm]:
    suppressed = suppressed_relevance_keys(workspace)
    terms = extract_significant_terms(parsed, max_terms=max_terms + len(suppressed))
    if suppressed:
        terms = [item for item in terms if relevance_key(item.term) not in suppressed]
    terms = terms[:max_terms]
    workspace.write_json("memory/significant_terms.json", [asdict(term) for term in terms])
    return terms


def load_significant_terms(
    parsed: ParsedPaper, workspace: Workspace, max_terms: int = 160
) -> list[SignificantTerm]:
    path = workspace.path("memory/significant_terms.json")
    global_memory = term_memory_path()
    local_feedback = feedback_path(workspace)
    if (
        not path.exists()
        or (global_memory.exists() and global_memory.stat().st_mtime > path.stat().st_mtime)
        or (local_feedback.exists() and local_feedback.stat().st_mtime > path.stat().st_mtime)
    ):
        return save_significant_terms(parsed, workspace, max_terms=max_terms)
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if any(
            isinstance(item, dict)
            and (
                "vocabulary_source" not in item
                or item.get("extraction_version") != EXTRACTION_VERSION
            )
            for item in payload
        ):
            return save_significant_terms(parsed, workspace, max_terms=max_terms)
        return [SignificantTerm(**item) for item in payload[:max_terms] if isinstance(item, dict)]
    except Exception:
        return save_significant_terms(parsed, workspace, max_terms=max_terms)


def terms_by_category(terms: Iterable[SignificantTerm]) -> dict[str, list[SignificantTerm]]:
    grouped: dict[str, list[SignificantTerm]] = {}
    for term in terms:
        grouped.setdefault(term.category, []).append(term)
    return grouped
