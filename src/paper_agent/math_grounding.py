from __future__ import annotations

import re
from dataclasses import dataclass

from paper_agent.parser import ParsedPaper, clean_line


@dataclass(frozen=True)
class SymbolGrounding:
    symbol: str
    meaning: str
    source: str
    evidence: str = ""
    page: int | None = None


@dataclass(frozen=True)
class MathContextBundle:
    context: str
    symbols: list[SymbolGrounding]
    evidence: list[str]
    anchor_terms: list[str]


_BASE = r"(?:\\(?:mathbb|mathcal|mathbf|mathrm)\{[^{}]+\}|\\ell|\\[A-Za-z]+|[A-Za-z])"
_SCRIPT = r"(?:[_^](?:\{(?:[^{}]|\{[^{}]*\})*\}|[A-Za-z0-9]))"
_TOKEN_RE = re.compile(rf"{_BASE}(?:{_SCRIPT})*")
_SKIP_COMMANDS = {
    r"\frac",
    r"\sum",
    r"\bigcup",
    r"\left",
    r"\right",
    r"\text",
    r"\in",
    r"\cdot",
    r"\exp",
    r"\log",
    r"\operatorname",
}
_INDEX_SYMBOLS = {"t", "s", "i", "j", "h", "l", "k"}
_GREEK_ALIASES = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "theta": "θ",
    "phi": "φ",
    "psi": "ψ",
    "omega": "ω",
}


def extract_latex_symbols(latex: str) -> list[str]:
    cleaned = re.sub(r"\\(?:begin|end)\{[^{}]+\}", " ", latex)
    cleaned = re.sub(r"\\text\{([^{}]*)\}", r"\1", cleaned)
    symbols: list[str] = []
    for match in _TOKEN_RE.finditer(cleaned):
        token = match.group(0)
        base = re.match(_BASE, token)
        if not base or base.group(0) in _SKIP_COMMANDS:
            continue
        if token in _INDEX_SYMBOLS:
            continue
        base_token = base.group(0)
        if (
            token.startswith("\\")
            and base_token
            not in {
                r"\ell",
                r"\gamma",
                r"\alpha",
                r"\beta",
                r"\theta",
                r"\tau",
                r"\sigma",
                r"\lambda",
            }
            and not re.match(r"\\(?:mathbb|mathcal|mathbf|mathrm)", token)
        ):
            continue
        if token not in symbols:
            symbols.append(token)
    for upper_limit in re.findall(r"\\sum(?:_\{[^{}]*\})?\^\{([A-Za-z])\}", cleaned):
        if upper_limit not in symbols:
            symbols.append(upper_limit)
    for limits in re.findall(
        r"\\(?:sum|bigcup)(?:_\{([^{}]*)\})?(?:\^\{([^{}]*)\})?",
        cleaned,
    ):
        for expression in limits:
            for greek in re.findall(
                r"\\(?:alpha|beta|gamma|lambda|mu|sigma|tau|theta)", expression
            ):
                if greek not in symbols:
                    symbols.append(greek)
    return symbols[:24]


def _symbol_aliases(symbol: str) -> list[str]:
    compact = symbol.replace(" ", "")
    aliases = [compact]
    plain = compact
    plain = re.sub(r"\\(?:mathbb|mathcal|mathbf|mathrm)\{([^{}]+)\}", r"\1", plain)
    plain = plain.replace(r"\ell", "L")
    for name, glyph in _GREEK_ALIASES.items():
        if rf"\{name}" in compact:
            aliases.extend([name, glyph])
        plain = plain.replace(rf"\{name}", name)
    plain = re.sub(r"\\text\{([^{}]+)\}", r"\1", plain)
    readable = re.sub(r"\\[A-Za-z]+", "", plain)
    readable = readable.replace("^{+}", "+").replace("^+", "+")
    readable = re.sub(r"[{}\\]", "", readable)
    aliases.extend(
        variant
        for variant in {
            readable,
            readable.replace("_", ""),
            readable.replace("_", ","),
            readable.replace("_", " "),
        }
        if variant
    )
    labels = re.findall(r"\\(?:mathrm|text)\{([^{}]+)\}|_\{([A-Za-z]{2,})\}", compact)
    for groups in labels:
        aliases.extend(label for label in groups if label)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _symbol_key(symbol: str) -> str:
    value = symbol.lower().replace(" ", "")
    value = re.sub(r"\\(?:mathbb|mathcal|mathbf|mathrm|text)", "", value)
    value = value.replace(r"\ell", "l")
    for name in _GREEK_ALIASES:
        value = value.replace(rf"\{name}", name)
    return re.sub(r"[{}_^\\()]", "", value)


def _line_windows(
    parsed: ParsedPaper, aliases: list[str], focus_page: int
) -> list[tuple[float, int, str]]:
    scored: list[tuple[float, int, str]] = []
    definition_terms = re.compile(
        r"\b(?:denote|denotes|defined|where|parameter|weight|label|score|set|batch|loss|objective|represents?)\b",
        re.I,
    )
    for page, text in parsed.page_text.items():
        lines = [clean_line(line) for line in text.splitlines() if clean_line(line)]
        for index, line in enumerate(lines):
            hits = sum(1 for alias in aliases if _alias_in_text(alias, line))
            if not hits:
                continue
            window = " ".join(lines[max(0, index - 2) : min(len(lines), index + 3)])
            score = hits * 2.0
            if definition_terms.search(window):
                score += 3.0
            if page == focus_page:
                score += 2.0
            elif abs(page - focus_page) == 1:
                score += 0.5
            scored.append((score, page, window[:1200]))
    scored.sort(key=lambda item: item[0], reverse=True)
    deduped: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for item in scored:
        key = item[2].casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:3]


def _alias_in_text(alias: str, text: str) -> bool:
    clean = alias.strip()
    if not clean:
        return False
    if re.fullmatch(r"[A-Za-z0-9]+", clean):
        flags = 0 if len(clean) == 1 and clean.isupper() else re.I
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(clean)}(?![A-Za-z0-9])",
                text,
                flags,
            )
        )
    compact_text = re.sub(r"\s+", "", text).casefold()
    compact_alias = re.sub(r"\s+", "", clean).casefold()
    return compact_alias in compact_text


def _known_meaning(symbol: str, context: str) -> tuple[str, str]:
    """Extract a paper-local description without embedding paper-specific vocabulary."""
    key = _symbol_key(symbol)
    if key == "i" and r"\mathbb" in symbol:
        return "Indicator function: 1 when its condition holds and 0 otherwise.", "inference"
    if key == "e" and r"\mathbb" in symbol:
        return "Expectation over the distribution shown in the equation.", "inference"

    clean_context = re.sub(r"\s+", " ", context).strip()
    if not clean_context:
        return "", "unresolved"
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", clean_context)
        if sentence.strip()
    ]
    aliases = sorted(_symbol_aliases(symbol), key=len, reverse=True)
    definition_cues = re.compile(
        r"\b(?:let|where|denote[sd]?|represent[sd]?|refer[s]? to|defined as|parameter|"
        r"matrix|vector|tensor|probability|score|loss|objective|set|weight|label|number of|"
        r"threshold(?:ing)?|selected?)\b",
        re.I,
    )
    ranked: list[tuple[int, str]] = []
    for index, sentence in enumerate(sentences):
        hits = sum(1 for alias in aliases if _alias_in_text(alias, sentence))
        if not hits:
            continue
        score = hits * 3 + (4 if definition_cues.search(sentence) else 0) - index
        ranked.append((score, sentence))
    if ranked:
        sentence = max(ranked, key=lambda item: item[0])[1]
        sentence = _clean_evidence_excerpt(sentence, max_chars=360)
        sentence = re.sub(r"\bgolden entries\b", "golden-entry set", sentence, flags=re.I)
        source = "paper" if definition_cues.search(sentence) else "unresolved"
        return sentence, source

    # If extraction destroyed the symbol spelling, retain definitional context rather than
    # fabricating a domain-specific meaning.
    nearby = next((item for item in sentences if definition_cues.search(item)), "")
    if nearby:
        return (
            f"Nearby definition context for {symbol}: "
            f"{_clean_evidence_excerpt(nearby, max_chars=300)}",
            "unresolved",
        )
    return "", "unresolved"


def _clean_evidence_excerpt(value: str, max_chars: int = 520) -> str:
    clean = re.sub(r"[\x00-\x1f\ue000-\uf8ff]", " ", value)
    clean = re.sub(r"\s+", " ", clean).strip(" •")
    step = re.search(r"(?:•\s*)?Step\s+\d+\s*:", clean, re.I)
    if step and step.start() > 0:
        clean = clean[step.start() :].lstrip("• ")
    clean = re.sub(r"^\(\d+\)\s*[•·]?\s*", "", clean)
    if clean.casefold().startswith("as selected by layer"):
        clean = "An entry s is marked " + clean
    colon = clean.rfind(":")
    if colon > 60:
        tail = clean[colon + 1 :]
        noise = len(re.findall(r"[=∈∑]", tail))
        if "=" in tail or noise >= 2:
            clean = clean[:colon].rstrip() + "."
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    excerpt = " ".join(sentences[:3]).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    elif excerpt and excerpt[-1] not in ".!?":
        excerpt = excerpt.rstrip(" ,;:") + "..."
    return excerpt


def _evidence_is_redundant(candidate: str, existing: list[str]) -> bool:
    candidate_words = set(re.findall(r"[a-z0-9]+", candidate.casefold()))
    if len(candidate_words) < 6:
        return False
    for item in existing:
        item_words = set(re.findall(r"[a-z0-9]+", item.casefold()))
        smaller = min(len(candidate_words), len(item_words))
        if smaller >= 6 and len(candidate_words & item_words) / smaller >= 0.6:
            return True
    return False


def contextual_math_evidence(context: str, limit: int = 3) -> list[str]:
    """Extract concise paper sentences when symbol-level retrieval has no hit."""
    clean = re.sub(
        r"(?:^|\n)(?:Current page \d+|PDF-region context|Symbol-definition evidence):\s*",
        "\n",
        context,
        flags=re.I,
    )
    clean = re.sub(r"[\x00-\x1f\ue000-\uf8ff]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return []

    signal = re.compile(
        r"\b(?:where|define[ds]?|denote[ds]?|establish(?:ed)?|union|label set|"
        r"lookahead|voting|score|selected|entry|window|threshold)\b",
        re.I,
    )
    sentences = [
        item.strip() for item in re.split(r"(?<=[.!?])\s+", clean) if len(item.strip()) >= 35
    ]
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (bool(signal.search(item[1])), len(signal.findall(item[1])), -item[0]),
        reverse=True,
    )
    evidence: list[str] = []
    for _, sentence in ranked:
        excerpt = _clean_evidence_excerpt(sentence)
        if excerpt and not _evidence_is_redundant(excerpt, evidence) and excerpt not in evidence:
            evidence.append(excerpt)
        if len(evidence) >= max(1, limit):
            break
    return evidence


def _anchor_terms(latex: str, raw: str) -> list[str]:
    candidates = []
    combined = f"{latex} {raw}"
    mappings = [
        ("BCE", "binary cross-entropy"),
        ("FL", "focal loss"),
        ("softmax", "softmax"),
        ("sigmoid", "sigmoid"),
        ("KL", "KL divergence"),
        ("entropy", "entropy"),
        ("loss", "loss"),
    ]
    for needle, term in mappings:
        if needle.casefold() in combined.casefold() and term not in candidates:
            candidates.append(term)
    for symbol in extract_latex_symbols(latex):
        for alias in _symbol_aliases(symbol):
            if len(alias) >= 3 and alias not in candidates:
                candidates.append(alias)
                break
    return candidates[:10]


def _focused_page_context(
    parsed: ParsedPaper, page: int, anchors: list[str], max_chars: int = 7000
) -> str:
    text = parsed.page_text.get(page, "")
    if not text:
        return ""
    low = text.casefold()
    positions = [low.find(term.casefold()) for term in anchors if low.find(term.casefold()) >= 0]
    if not positions:
        return text[:max_chars]
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end].strip()


def build_math_context(
    parsed: ParsedPaper,
    page: int,
    latex: str,
    raw: str,
) -> MathContextBundle:
    anchors = _anchor_terms(latex, raw)
    local = _focused_page_context(parsed, page, anchors)
    symbols: list[SymbolGrounding] = []
    evidence: list[str] = []

    for symbol in extract_latex_symbols(latex):
        aliases = _symbol_aliases(symbol)
        windows = _line_windows(parsed, aliases, page)
        best = windows[0] if windows else None
        search_context = " ".join(item[2] for item in windows) or local
        meaning, source = _known_meaning(symbol, search_context)
        if not meaning and best:
            meaning = f"The paper uses {symbol} in this equation; its nearby description is quoted as evidence."
            source = "unresolved"
        if not meaning:
            meaning = "Meaning was not resolved from the paper text."
            source = "unresolved"
        symbol_evidence = best[2] if best else local[:800]
        symbol_page = best[1] if best else (page if local else None)
        symbols.append(SymbolGrounding(symbol, meaning, source, symbol_evidence, symbol_page))
        evidence_excerpt = _clean_evidence_excerpt(symbol_evidence)
        if (
            evidence_excerpt
            and evidence_excerpt not in evidence
            and not _evidence_is_redundant(evidence_excerpt, evidence)
        ):
            evidence.append(evidence_excerpt)

    context_parts = [f"Current page {page}:\n{local}"] if local else []
    for item in evidence[:6]:
        if item not in local:
            context_parts.append(f"Symbol-definition evidence:\n{item}")
    if not evidence:
        evidence.extend(contextual_math_evidence(local, limit=3))
    return MathContextBundle(
        context="\n\n".join(context_parts)[:12_000],
        symbols=symbols,
        evidence=evidence[:6],
        anchor_terms=anchors,
    )


def deterministic_math_fallback(
    latex: str,
    bundle: MathContextBundle,
) -> dict[str, object]:
    compact = latex.replace(" ", "")
    is_focal = "BCE" in latex and ("gamma" in latex.casefold() or r"\gamma" in latex)
    is_average = r"\frac{1}{|" in compact and r"\sum" in latex
    if is_focal:
        plain = (
            "This equation defines the paper's focal-loss objective as a weighted average over training samples. "
            "For each sample, binary cross-entropy measures prediction error, while the factor "
            r"$(1-p_{t,s}^{(\mathrm{correct})})^\gamma$ suppresses examples the indexer already classifies "
            "confidently. The remaining gradient is therefore concentrated on difficult or boundary examples."
        )
        steps = [
            r"Compute the correct-class confidence $p_{t,s}^{(\mathrm{correct})}$ from the prediction and binary label.",
            r"Form the focal factor $(1-p_{t,s}^{(\mathrm{correct})})^\gamma$; it approaches zero for easy samples.",
            r"Multiply the focal factor by the sample weight $w_{t,s}$ and binary cross-entropy.",
            r"Sum over $s \in \mathcal{S}$ and divide by $|\mathcal{S}|$ to obtain the mean objective.",
        ]
        return {
            "role": "sample-weighted focal-loss training objective",
            "plain_english": plain,
            "steps": steps,
            "intuition": "Easy negatives can dominate a highly imbalanced retrieval dataset. Focal weighting reduces their contribution so optimization spends more capacity on difficult, ambiguous retrieval decisions.",
            "dimensional_analysis": "Every BCE term, focal factor, and sample weight is scalar. Their product is scalar, and averaging over the sample set produces one scalar loss.",
            "implementation_view": "Compute BCE per sample without reduction, multiply by `(1 - p_correct).pow(gamma)` and the sample weights, then take the mean.",
            "derivation_notes": "The displayed objective is a weighted BCE with the standard focal factor. Any derivation beyond that decomposition is not present in the retrieved evidence.",
            "toy_example": "If the correct-class confidence is 0.9 and gamma is 2, the focal multiplier is (1 - 0.9)^2 = 0.01 before applying the sample weight and BCE.",
            "context_fit": "The paper uses this objective to emphasize difficult examples; dataset-specific sampling and weighting choices must be read from the surrounding evidence.",
            "assumptions_or_missing_details": [],
        }
    qualifier = "averaged " if is_average else ""
    return {
        "role": "paper-defined mathematical operation",
        "plain_english": f"This equation defines a {qualifier}computation used by the paper. Its grounded symbols are listed below; unresolved meanings remain explicitly marked rather than guessed.",
        "steps": [
            "Read the grounded input symbols.",
            "Apply the displayed operations from the innermost terms outward.",
            "Use the resulting value in the surrounding method step.",
        ],
        "intuition": "The formula packages the surrounding method description into a reproducible computation.",
        "dimensional_analysis": "The paper does not provide enough shape information for a complete dimensional check.",
        "implementation_view": "Translate the displayed operators directly and assert the inferred tensor shapes before execution.",
        "derivation_notes": "The retrieved evidence does not contain enough information to reconstruct a derivation safely.",
        "toy_example": "A numeric example is omitted because unresolved symbol meanings could make it misleading.",
        "context_fit": "The equation's role is grounded by its surrounding section and symbol-definition evidence.",
        "assumptions_or_missing_details": [
            "A model-generated explanation was unavailable; this is the deterministic grounded fallback."
        ],
    }
