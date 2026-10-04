from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.term_feedback import load_term_feedback, relevance_key
from paper_agent.term_highlighter import SignificantTerm, extract_significant_terms
from paper_agent.workspace import Workspace


@dataclass(frozen=True)
class TermRelevanceMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    labeled_terms: int


def evaluate_term_labels(
    predictions: Iterable[str], relevant: Iterable[str], irrelevant: Iterable[str]
) -> TermRelevanceMetrics:
    predicted_keys = {relevance_key(item) for item in predictions if relevance_key(item)}
    relevant_keys = {relevance_key(item) for item in relevant if relevance_key(item)}
    irrelevant_keys = {relevance_key(item) for item in irrelevant if relevance_key(item)}
    overlap = relevant_keys & irrelevant_keys
    if overlap:
        raise ValueError("A relevance benchmark cannot label the same term both ways.")

    true_positives = len(predicted_keys & relevant_keys)
    false_negatives = len(relevant_keys - predicted_keys)
    false_positives = len(predicted_keys & irrelevant_keys)
    true_negatives = len(irrelevant_keys - predicted_keys)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TermRelevanceMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        labeled_terms=len(relevant_keys | irrelevant_keys),
    )


def evaluate_workspace_feedback(
    workspace: Workspace, terms: Iterable[SignificantTerm]
) -> dict[str, Any]:
    records = [
        item
        for item in load_term_feedback(workspace).get("labels", {}).values()
        if isinstance(item, dict)
    ]
    relevant = [str(item.get("term")) for item in records if item.get("label") == "relevant"]
    irrelevant = [str(item.get("term")) for item in records if item.get("label") == "irrelevant"]
    metrics = evaluate_term_labels((item.term for item in terms), relevant, irrelevant)
    return {
        **asdict(metrics),
        "relevant_labels": len(relevant),
        "irrelevant_labels": len(irrelevant),
    }


def run_term_relevance_benchmark(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Term relevance benchmark must contain at least one case.")

    all_predictions: list[str] = []
    all_relevant: list[str] = []
    all_irrelevant: list[str] = []
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Benchmark case {index + 1} is not an object.")
        text = str(case.get("text") or "").strip()
        name = str(case.get("name") or f"case-{index + 1}")
        if not text:
            raise ValueError(f"Benchmark case {name} has no text.")
        parsed = ParsedPaper(
            metadata=PaperMetadata(
                title_guess=str(case.get("title") or name),
                authors_guess=[],
                page_count=1,
                source_pdf=f"benchmark/{name}.pdf",
            ),
            full_text=text,
            page_text={1: text},
            sections={"abstract": text, "method": text},
            equation_cards=[],
            figure_cards=[],
            table_cards=[],
        )
        terms = extract_significant_terms(parsed, max_terms=80, learned_terms={}, vocabulary=[])
        predictions = [item.term for item in terms]
        relevant = [str(item) for item in case.get("relevant", [])]
        irrelevant = [str(item) for item in case.get("irrelevant", [])]
        metrics = evaluate_term_labels(predictions, relevant, irrelevant)
        results.append(
            {
                "name": name,
                "metrics": asdict(metrics),
                "predictions": predictions,
                "missed_relevant": [
                    item
                    for item in relevant
                    if relevance_key(item) not in {relevance_key(term) for term in predictions}
                ],
                "surfaced_irrelevant": [
                    item
                    for item in irrelevant
                    if relevance_key(item) in {relevance_key(term) for term in predictions}
                ],
            }
        )
        prefix = f"{name}::"
        all_predictions.extend(prefix + relevance_key(item) for item in predictions)
        all_relevant.extend(prefix + relevance_key(item) for item in relevant)
        all_irrelevant.extend(prefix + relevance_key(item) for item in irrelevant)

    aggregate = evaluate_term_labels(all_predictions, all_relevant, all_irrelevant)
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    minimum_precision = float(thresholds.get("minimum_precision", 0.9))
    minimum_recall = float(thresholds.get("minimum_recall", 0.85))
    return {
        "version": payload.get("version", 1),
        "benchmark": path.name,
        "aggregate": asdict(aggregate),
        "thresholds": {
            "minimum_precision": minimum_precision,
            "minimum_recall": minimum_recall,
        },
        "passed": aggregate.precision >= minimum_precision and aggregate.recall >= minimum_recall,
        "cases": results,
    }
