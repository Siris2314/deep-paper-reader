from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from paper_agent.parser import ParsedPaper
from paper_agent.workspace import Workspace

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "using",
    "use",
    "used",
    "via",
    "than",
    "then",
    "they",
    "such",
    "these",
    "those",
    "which",
    "while",
    "can",
    "could",
    "may",
    "might",
    "also",
    "not",
    "but",
    "were",
    "was",
    "been",
    "being",
    "paper",
    "model",
    "models",
    "method",
    "methods",
    "task",
    "tasks",
    "data",
    "results",
    "show",
    "shows",
}

GENERIC_BAD_PHRASES = [
    "the provided text appears",
    "set of system prompts",
    "various roles in a data generation pipeline",
    "here's a breakdown of the different roles",
    "this role involves",
]

APPENDIX_PROMPT_DRIFT_TERMS = [
    "legal extractor",
    "legal challenger",
    "legal judge",
    "scientific reasoning challenger",
    "system prompts",
    "role.",
    "output format",
    "write ./",
    "figure 7",
    "figure 8",
    "figure 9",
    "figure 10",
    "figure 11",
    "figure 12",
    "figure 13",
    "figure 14",
    "figure 15",
]


def _words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)]


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in STOPWORDS and not w.isdigit()]


def _sentence_clip(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0]
    return clipped + "..."


def _first_page_text(parsed: ParsedPaper) -> str:
    return parsed.page_text.get(1, "") or parsed.full_text[:5000]


def _abstractish_text(parsed: ParsedPaper) -> str:
    first = _first_page_text(parsed)
    # Most PDFs put abstract before Figure 1 / Introduction. This is only a heuristic.
    candidates = re.split(
        r"\bFigure\s+1\b|\b1\s+Introduction\b|\n1\s*\nIntroduction", first, flags=re.I
    )
    if candidates:
        return candidates[0]
    return first[:3500]


def _main_paper_context(parsed: ParsedPaper, max_chars: int = 32_000) -> str:
    preferred_keys = [
        "front_matter",
        "abstract",
        "introduction",
        "autodata",
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "experiments",
        "experimental_setup",
        "evaluation",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
    ]
    parts: list[str] = []
    seen = set()
    for key in preferred_keys:
        if key in parsed.sections and key not in seen:
            parts.append(f"\n\n## {key}\n{parsed.sections[key]}")
            seen.add(key)
    # Add early non-reference sections in order for papers with numbered custom headings.
    for key, text in parsed.sections.items():
        if key in seen or key in {"references", "bibliography"}:
            continue
        if len("\n".join(parts)) > max_chars:
            break
        # Skip appendix-like prompt dumps when building the repair context.
        if re.search(
            r"subagent_system_prompts|appendix|figure_7|legal_challenger|scientific_reasoning",
            key,
            re.I,
        ):
            continue
        parts.append(f"\n\n## {key}\n{text}")
    joined = "\n".join(parts).strip() or parsed.full_text[:max_chars]
    return joined[:max_chars]


def expected_keywords(parsed: ParsedPaper, limit: int = 24) -> list[str]:
    title = parsed.metadata.title_guess or ""
    abstractish = _abstractish_text(parsed)
    # Weight title heavily. This catches drift where the final report ignores the actual paper title/topic.
    weighted = (title + " ") * 5 + abstractish
    counts = Counter(_content_words(weighted))
    author_tokens = {w for name in parsed.metadata.authors_guess for w in _content_words(name)}
    metadata_noise = {
        "arxiv",
        "date",
        "june",
        "correspondence",
        "email",
        "university",
        "institute",
        "researchers",
        "independent",
        "fair",
        "meta",
        "joint",
        "first",
        "author",
    }
    keywords: list[str] = []
    for word in _content_words(title):
        # Never filter title tokens as author/noise tokens; title coverage is the strongest anti-drift signal.
        if len(word) >= 4 and word not in metadata_noise and word not in keywords:
            keywords.append(word)
    for word, _ in counts.most_common(limit * 5):
        if len(word) < 4 or word in author_tokens or word in metadata_noise:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords[:limit]


@dataclass
class EvaluationIssue:
    code: str
    severity: str  # fatal | major | minor
    message: str
    evidence: str = ""


@dataclass
class ReportEvaluation:
    passed: bool
    score: float
    status: str
    issues: list[EvaluationIssue] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["issues"] = [asdict(i) for i in self.issues]
        return out

    # Compatibility with code paths that serialize pydantic models.
    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def evaluate_report_text(
    parsed: ParsedPaper, final_text: str, claim_ledger_text: str | None = None
) -> ReportEvaluation:
    issues: list[EvaluationIssue] = []
    lower = final_text.lower()
    title = parsed.metadata.title_guess or ""
    keywords = expected_keywords(parsed)
    matched = [kw for kw in keywords if kw.lower() in lower]
    missing = [kw for kw in keywords if kw.lower() not in lower]
    keyword_recall = len(matched) / max(1, len(keywords))

    title_words = [w for w in _content_words(title) if len(w) >= 4]
    title_hits = [w for w in title_words if w in lower]
    title_recall = len(set(title_hits)) / max(1, len(set(title_words)))

    if len(final_text.strip()) < 1200:
        issues.append(
            EvaluationIssue(
                "too_short",
                "major",
                "Final report is too short to be a useful study guide.",
                f"chars={len(final_text)}",
            )
        )

    if any(phrase in lower for phrase in GENERIC_BAD_PHRASES):
        issues.append(
            EvaluationIssue(
                "generic_model_preamble",
                "major",
                "Report begins like a generic text-classification answer instead of a paper-specific study guide.",
                "; ".join(p for p in GENERIC_BAD_PHRASES if p in lower),
            )
        )

    if title and title_recall < 0.35:
        issues.append(
            EvaluationIssue(
                "title_mismatch",
                "fatal",
                "Report appears not to be about the parsed paper title.",
                f"title={title!r}; title_recall={title_recall:.2f}; title_hits={title_hits}",
            )
        )

    if keyword_recall < 0.35:
        issues.append(
            EvaluationIssue(
                "low_keyword_coverage",
                "fatal",
                "Report misses most high-signal terms from the title/abstract/front matter.",
                f"keyword_recall={keyword_recall:.2f}; missing={missing[:12]}",
            )
        )
    elif keyword_recall < 0.55:
        issues.append(
            EvaluationIssue(
                "thin_keyword_coverage",
                "major",
                "Report covers only part of the paper's high-signal vocabulary.",
                f"keyword_recall={keyword_recall:.2f}; missing={missing[:12]}",
            )
        )

    drift_hits = [term for term in APPENDIX_PROMPT_DRIFT_TERMS if term in lower]
    drift_ratio = len(drift_hits) / max(1, len(APPENDIX_PROMPT_DRIFT_TERMS))
    if drift_ratio >= 0.20 and title_recall < 0.65:
        issues.append(
            EvaluationIssue(
                "appendix_prompt_drift",
                "fatal",
                "Report seems to summarize appendix prompt text instead of the main paper.",
                f"drift_terms={drift_hits[:10]}",
            )
        )
    elif drift_ratio >= 0.20:
        issues.append(
            EvaluationIssue(
                "appendix_overfocus",
                "major",
                "Report over-focuses on appendix/system-prompt material. This may be valid only as a minor section.",
                f"drift_terms={drift_hits[:10]}",
            )
        )

    if parsed.figure_cards and "figure" not in lower:
        issues.append(
            EvaluationIssue(
                "missing_figures",
                "major",
                "Report ignores detected figures.",
                f"figures={len(parsed.figure_cards)}",
            )
        )
    if parsed.table_cards and "table" not in lower:
        issues.append(
            EvaluationIssue(
                "missing_tables",
                "major",
                "Report ignores detected tables/results tables.",
                f"tables={len(parsed.table_cards)}",
            )
        )

    claim_count = 0
    if claim_ledger_text is not None:
        try:
            claim_obj = json.loads(claim_ledger_text or "[]")
            claim_count = len(claim_obj) if isinstance(claim_obj, list) else 0
        except Exception:
            issues.append(
                EvaluationIssue(
                    "claim_ledger_invalid", "major", "Claim ledger is not valid JSON.", ""
                )
            )
        if claim_count == 0:
            issues.append(
                EvaluationIssue(
                    "claim_ledger_empty",
                    "minor",
                    "Claim ledger is empty; debugging support is weak.",
                    "",
                )
            )

    fatal = sum(1 for i in issues if i.severity == "fatal")
    major = sum(1 for i in issues if i.severity == "major")
    minor = sum(1 for i in issues if i.severity == "minor")
    score = 1.0 - (0.40 * fatal + 0.12 * major + 0.03 * minor)
    score = max(0.0, min(1.0, score))
    passed = fatal == 0 and major <= 2 and score >= 0.68
    status = "pass" if passed else "fail"

    return ReportEvaluation(
        passed=passed,
        score=score,
        status=status,
        issues=issues,
        expected_keywords=keywords,
        matched_keywords=matched,
        missing_keywords=missing,
        metrics={
            "chars": len(final_text),
            "title": title,
            "title_recall": round(title_recall, 3),
            "keyword_recall": round(keyword_recall, 3),
            "appendix_prompt_drift_ratio": round(drift_ratio, 3),
            "claim_count": claim_count,
            "figure_count": len(parsed.figure_cards),
            "table_count": len(parsed.table_cards),
        },
    )


def save_report_evaluation(
    workspace: Workspace, evaluation: ReportEvaluation, name: str = "final/report_evaluation.json"
) -> None:
    workspace.write_json(name, evaluation.to_dict())
    lines = [
        "# Report Evaluation",
        "",
        f"Status: **{evaluation.status}**",
        f"Score: **{evaluation.score:.2f}**",
        "",
    ]
    if evaluation.issues:
        lines.append("## Issues")
        for issue in evaluation.issues:
            lines.append(
                f"- **{issue.severity.upper()} / {issue.code}**: {issue.message} {issue.evidence}"
            )
    else:
        lines.append("No blocking issues detected.")
    lines.extend(
        [
            "",
            "## Keyword coverage",
            f"Matched: {', '.join(evaluation.matched_keywords) or 'none'}",
            "",
            f"Missing: {', '.join(evaluation.missing_keywords) or 'none'}",
        ]
    )
    workspace.write_text("final/report_evaluation.md", "\n".join(lines).strip() + "\n")


def build_repair_prompt(
    parsed: ParsedPaper, bad_report: str, evaluation: ReportEvaluation, workspace: Workspace
) -> str:
    figures = (
        "\n".join(f"- {f.id} page {f.page}: {f.caption}" for f in parsed.figure_cards[:12])
        or "No figures detected."
    )
    tables = (
        "\n".join(
            f"- {t.id} page {t.page}: {t.caption}\n  {t.raw_text[:700]}"
            for t in parsed.table_cards[:10]
        )
        or "No tables detected."
    )
    equations = (
        "\n".join(f"- {e.id} page {e.page}: {e.raw[:500]}" for e in parsed.equation_cards[:20])
        or "No equations detected."
    )
    context = _main_paper_context(parsed, max_chars=60_000)
    eval_json = json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False)

    return f"""
You are the EVALUATOR/REPAIR agent for a paper-study-guide system.

The previous report failed quality checks. Write a corrected `final_study_guide.md` for the actual paper.

Hard constraints:
- The report must be about this title: {parsed.metadata.title_guess}
- Use the main paper, not appendix prompt dumps, unless appendix prompts are discussed briefly as supporting implementation detail.
- Do not start with "The provided text appears".
- Include concrete paper-specific terms from the title/abstract.
- Include figures/tables that matter.
- Include limitations and what not to overclaim.
- Clearly label unsupported or uncertain claims.
- Return markdown only. Do not include code fences around the whole report.

Evaluation failure report:
{eval_json}

Detected figures:
{figures}

Detected tables:
{tables}

Detected equations:
{equations}

Main paper context:
{context}

Previous bad report, for reference only. Do not copy its mistakes:
{bad_report[:18_000]}
""".strip()


def deterministic_fallback_report(
    parsed: ParsedPaper, evaluation: ReportEvaluation | None = None
) -> str:
    title = parsed.metadata.title_guess or "Untitled paper"
    abstractish = _sentence_clip(_abstractish_text(parsed), 1600)
    intro = _sentence_clip(parsed.sections.get("introduction", ""), 1800)
    conclusion = _sentence_clip(
        parsed.sections.get("conclusion", parsed.sections.get("conclusions", "")), 1400
    )

    likely_method_keys = [
        k
        for k in parsed.sections.keys()
        if re.search(r"method|autodata|approach|implementation|self", k, re.I)
    ]
    method_text = "\n\n".join(
        f"### {k}\n{_sentence_clip(parsed.sections[k], 1600)}" for k in likely_method_keys[:4]
    )
    if not method_text:
        method_text = _sentence_clip(_main_paper_context(parsed), 2200)

    experiment_keys = [
        k
        for k in parsed.sections.keys()
        if re.search(r"experiment|result|evaluation|computer|legal|scientific|meta", k, re.I)
    ]
    experiment_text = "\n\n".join(
        f"### {k}\n{_sentence_clip(parsed.sections[k], 1300)}" for k in experiment_keys[:6]
    )

    figs = (
        "\n".join(f"- **{f.id}, page {f.page}:** {f.caption}" for f in parsed.figure_cards[:12])
        or "- No figure captions detected."
    )
    tabs = (
        "\n".join(
            f"- **{t.id}, page {t.page}:** {t.caption}\n  `{_sentence_clip(t.raw_text, 500)}`"
            for t in parsed.table_cards[:12]
        )
        or "- No table captions detected."
    )
    eqs = (
        "\n".join(
            f"- **{e.id}, page {e.page}:** `{_sentence_clip(e.raw, 400)}`"
            for e in parsed.equation_cards[:12]
        )
        or "- No equation candidates detected."
    )
    issues = ""
    if evaluation and evaluation.issues:
        issues = "\n".join(
            f"- {i.severity.upper()} `{i.code}`: {i.message}" for i in evaluation.issues
        )
    else:
        issues = "- Fallback report generated because model report was unavailable or unreliable."

    return (
        f"""# {title}

> This fallback report was generated by deterministic parser/evaluator logic because the model-generated report failed quality checks. Use the **Model Debug** and **Report Evaluation** tabs to inspect the failed output.

## TLDR
{abstractish}

## Problem and motivation
{intro or abstractish}

## Method / core idea
{method_text}

## Experiments and results
{experiment_text or "The parser did not isolate a clean experiments section. See tables below for extracted results artifacts."}

## Figures that matter
{figs}

## Tables that matter
{tabs}

## Equation candidates / math artifacts
{eqs}

## Limitations and uncertainty
{conclusion or "No clean conclusion section was detected. Check the parsed sections manually."}

## Evaluator notes
{issues}

## What to check next
- Open `parsed/sections.json` and confirm section splitting is sane.
- Open `logs/agent_messages.md` to see whether the model drifted into appendix text.
- Re-run with a stronger model or lower `max_chars` if the model keeps summarizing appendix/system-prompt material instead of the main paper.
""".strip()
        + "\n"
    )


def build_minimal_claim_ledger(parsed: ParsedPaper) -> list[dict[str, str]]:
    title = parsed.metadata.title_guess or "the paper"
    claims = [
        {
            "claim": f"The analyzed paper is titled '{title}'.",
            "source_type": "paper",
            "support": "supported",
            "evidence": "parsed/metadata.json and page 1 title extraction",
            "confidence": "high",
        }
    ]
    if parsed.figure_cards:
        claims.append(
            {
                "claim": f"The paper contains at least {len(parsed.figure_cards)} detected figure caption(s).",
                "source_type": "paper",
                "support": "supported",
                "evidence": "parsed/figures.json",
                "confidence": "medium",
            }
        )
    if parsed.table_cards:
        claims.append(
            {
                "claim": f"The paper contains at least {len(parsed.table_cards)} detected table caption(s).",
                "source_type": "paper",
                "support": "supported",
                "evidence": "parsed/tables.json",
                "confidence": "medium",
            }
        )
    return claims
