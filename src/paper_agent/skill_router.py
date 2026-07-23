from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SkillTrigger:
    name: str
    reason: str
    matched_terms: list[str]
    confidence: str = "medium"


SKILL_RULES: dict[str, dict[str, object]] = {
    "math-walkthrough": {
        "terms": [
            "math",
            "equation",
            "derive",
            "derivation",
            "notation",
            "loss",
            "objective",
            "formula",
            "softmax",
            "logits",
            "gradient",
            "kl",
            "cross entropy",
            "probability",
            "tensor",
            "shape",
        ],
        "reason": "The question asks for equations, notation, objectives, losses, or mathematical mechanics.",
    },
    "implementation-reconstructor": {
        "terms": [
            "implement",
            "implementation",
            "code",
            "pseudocode",
            "architecture",
            "pipeline",
            "system",
            "data structure",
            "training loop",
            "runtime",
            "reproduce",
        ],
        "reason": "The question asks how to turn the paper method into code or an executable system.",
    },
    "figure-table-analysis": {
        "terms": [
            "figure",
            "table",
            "plot",
            "diagram",
            "ablation",
            "metric",
            "benchmark",
            "baseline",
            "result",
            "experiment",
        ],
        "reason": "The question asks about visuals, tables, metrics, experiments, or benchmark evidence.",
    },
    "concept-card": {
        "terms": [
            "what is",
            "explain",
            "define",
            "concept",
            "term",
            "means",
            "background",
            "prerequisite",
        ],
        "reason": "The question asks for a term or concept explanation.",
    },
    "paper-term-highlighter": {
        "terms": [
            "highlight",
            "significant term",
            "important term",
            "ml term",
            "method term",
            "main method",
            "click",
            "annotate",
            "term panel",
        ],
        "reason": "The question asks to identify, highlight, or inspect significant paper terms.",
    },
    "tavily-background": {
        "terms": [
            "web",
            "search",
            "latest",
            "related work",
            "official",
            "github",
            "codebase",
            "project page",
            "external",
            "follow-up",
        ],
        "reason": "The question asks for external context or verification beyond the paper.",
    },
    "research-lineage": {
        "terms": [
            "prior work",
            "previous work",
            "builds on",
            "build on",
            "built on",
            "extends",
            "extended",
            "lineage",
            "origin",
            "foundational",
            "citation",
            "related paper",
            "came from",
        ],
        "reason": "The question asks how this paper or concept relates to cited earlier research.",
    },
    "paper-evaluator": {
        "terms": [
            "evaluate",
            "check",
            "verify",
            "grounded",
            "unsupported",
            "claim",
            "critique",
            "is this right",
        ],
        "reason": "The question asks for grounding, quality checks, or claim validation.",
    },
    "chat-answer": {
        "terms": [],
        "reason": "Default interactive paper Q&A route.",
    },
}


def route_question_to_skills(question: str) -> list[SkillTrigger]:
    q = question.lower()
    triggers: list[SkillTrigger] = []
    for skill, rule in SKILL_RULES.items():
        if skill == "chat-answer":
            continue
        terms = [str(term) for term in rule["terms"] if str(term).lower() in q]
        if terms:
            confidence = "high" if len(terms) >= 2 else "medium"
            triggers.append(
                SkillTrigger(
                    name=skill,
                    reason=str(rule["reason"]),
                    matched_terms=terms,
                    confidence=confidence,
                )
            )
    triggers.append(
        SkillTrigger(
            name="chat-answer",
            reason=str(SKILL_RULES["chat-answer"]["reason"]),
            matched_terms=[],
            confidence="high" if not triggers else "medium",
        )
    )
    return triggers


def extract_question_terms(question: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question)
    blocked = {"what", "how", "why", "does", "the", "this", "that", "paper", "about", "explain"}
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        key = term.lower()
        if key in blocked or key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms[:12]
