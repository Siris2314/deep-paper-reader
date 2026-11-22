from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from paper_agent.math_agent import _latex_similarity


def _iou(left, right):
    if not all(isinstance(box, (list, tuple)) and len(box) == 4 for box in (left, right)):
        return None
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = (
        max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        + max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        - intersection
    )
    return intersection / union if union else 0.0


def _key(value: object) -> str:
    # Preserve case, grouping and delimiters: W != w, x^{a+b} != x^a+b.
    text = str(value or "").strip().strip("$")
    text = text.replace(r"\left", "").replace(r"\right", "").replace(r"\dfrac", r"\frac")
    text = re.sub(r"\\(?:,|;|!|quad\b|qquad\b)", "", text)
    # Optional braces around one script token do not change its scope.
    text = re.sub(r"([_^])\{([A-Za-z0-9]|\\[A-Za-z]+)\}", r"\1\2", text)
    return re.sub(r"\s+", "", text)


def _symbol_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {_key(item) for item in value if _key(item)}


def load_math_benchmark(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or len(cases) < 25:
        raise ValueError("Math reliability benchmark must contain at least 25 labeled cases.")
    required = {"latex", "symbols", "nearby_definitions", "role", "boundary"}
    for case in cases:
        expected = case.get("expected") if isinstance(case, dict) else None
        if not isinstance(expected, dict) or required - set(expected):
            raise ValueError(f"Benchmark case {case.get('id', '<unknown>')} is incomplete.")
    annotations = path.with_name("math_region_annotations.json")
    if annotations.exists():
        regions = json.loads(annotations.read_text(encoding="utf-8")).get("regions", {})
        for case in cases:
            if case["id"] in regions:
                case["expected"]["boundary"]["bbox"] = regions[case["id"]]
    return payload


def baseline_predictions(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for case in benchmark["cases"]:
        inputs = case.get("input") or {}
        predictions.append(
            {
                "id": case["id"],
                "latex": inputs.get("geometry_latex") or "",
                "symbols": inputs.get("geometry_symbols") or [],
                "symbol_grounding": inputs.get("geometry_symbol_grounding") or [],
                "role": inputs.get("predicted_role") or "",
                "boundary": inputs.get("predicted_boundary") or {},
                "explanation_claims": [],
                "latency_ms": 0,
                "vision_used": False,
            }
        )
    return predictions


def evaluate_math_reliability(
    benchmark: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    indexed = {
        str(item.get("id")): item
        for item in predictions
        if isinstance(item, dict) and item.get("id")
    }
    rows: list[dict[str, Any]] = []
    unsupported = 0
    claim_count = 0
    latencies: list[float] = []
    vision_required = 0
    vision_correctly_routed = 0
    for case in benchmark["cases"]:
        predicted = indexed.get(str(case["id"]), {})
        expected = case["expected"]
        expected_symbols = _symbol_set(expected["symbols"])
        predicted_symbols = _symbol_set(predicted.get("symbols"))
        overlap = expected_symbols & predicted_symbols
        precision = len(overlap) / len(predicted_symbols) if predicted_symbols else 0.0
        recall = len(overlap) / len(expected_symbols) if expected_symbols else 1.0
        grounding_items = predicted.get("symbol_grounding")
        grounded_symbols = (
            {
                _key(item.get("symbol"))
                for item in grounding_items
                if isinstance(item, dict)
                and str(item.get("meaning") or "").strip()
                and str(item.get("source") or "") in {"paper", "inference"}
                and (
                    str(item.get("source")) != "paper"
                    or bool(str(item.get("evidence") or "").strip())
                )
            }
            if isinstance(grounding_items, list)
            else set()
        )
        grounding_recall = (
            len(expected_symbols & grounded_symbols) / len(expected_symbols)
            if expected_symbols
            else 1.0
        )
        predicted_boundary = predicted.get("boundary")
        boundary_exact = isinstance(predicted_boundary, dict) and all(
            predicted_boundary.get(name) == value for name, value in expected["boundary"].items()
        )
        if not isinstance(predicted_boundary, dict) or not all(
            k in predicted_boundary for k in expected["boundary"]
        ):
            boundary_exact = None
        boundary_iou = _iou(
            (predicted_boundary or {}).get("bbox"), expected["boundary"].get("bbox")
        )
        if expected["boundary"].get("bbox") and boundary_iou is None:
            boundary_iou = 0.0
        claims = predicted.get("explanation_claims")
        if isinstance(claims, list):
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                claim_count += 1
                source = str(claim.get("source") or "unsupported")
                evidence = str(claim.get("evidence") or "").strip()
                if source == "unsupported" or (source == "paper" and not evidence):
                    unsupported += 1
        raw_latency = predicted.get("latency_ms")
        if isinstance(raw_latency, (int, float)) and raw_latency >= 0:
            latencies.append(float(raw_latency))
        requires_vision = bool(case.get("requires_vision"))
        if requires_vision:
            vision_required += 1
            if bool(predicted.get("vision_used")):
                vision_correctly_routed += 1
        latex = str(predicted.get("latex") or "")
        expected_latex = str(expected["latex"])
        rows.append(
            {
                "id": case["id"],
                "exact_latex": _key(latex) == _key(expected_latex),
                "latex_similarity": round(_latex_similarity(latex, expected_latex), 4),
                "symbol_precision": round(precision, 4),
                "symbol_recall": round(recall, 4),
                "symbol_grounding_recall": round(grounding_recall, 4),
                "boundary_exact": boundary_exact,
                "boundary_iou": boundary_iou,
                "role_exact": (
                    str(predicted["role"]).strip().casefold()
                    == str(expected["role"]).strip().casefold()
                )
                if predicted.get("role")
                else None,
            }
        )
    count = len(rows)
    sorted_latency = sorted(latencies)
    p95_index = max(0, min(len(sorted_latency) - 1, round(0.95 * (len(sorted_latency) - 1))))
    return {
        "benchmark_version": benchmark.get("version"),
        "case_count": count,
        "coverage": round(
            sum(str(case["id"]) in indexed for case in benchmark["cases"]) / count, 4
        ),
        "measurement_counts": {
            "predictions": sum(str(c["id"]) in indexed for c in benchmark["cases"]),
            "claims": claim_count,
            "latencies": len(latencies),
        },
        "metrics": {
            "mean_boundary_iou": statistics.fmean(
                r["boundary_iou"] for r in rows if r["boundary_iou"] is not None
            )
            if any(r["boundary_iou"] is not None for r in rows)
            else None,
            "exact_latex_accuracy": round(sum(row["exact_latex"] for row in rows) / count, 4),
            "mean_latex_similarity": round(
                statistics.fmean(row["latex_similarity"] for row in rows), 4
            ),
            "mean_symbol_precision": round(
                statistics.fmean(row["symbol_precision"] for row in rows), 4
            ),
            "mean_symbol_recall": round(statistics.fmean(row["symbol_recall"] for row in rows), 4),
            "mean_symbol_grounding_recall": round(
                statistics.fmean(row["symbol_grounding_recall"] for row in rows), 4
            ),
            "boundary_accuracy": round(
                statistics.fmean(
                    row["boundary_exact"] for row in rows if row["boundary_exact"] is not None
                ),
                4,
            )
            if any(row["boundary_exact"] is not None for row in rows)
            else None,
            "role_accuracy": round(
                statistics.fmean(
                    row["role_exact"] for row in rows if row["role_exact"] is not None
                ),
                4,
            )
            if any(row["role_exact"] is not None for row in rows)
            else None,
            "unsupported_claim_rate": (
                round(unsupported / claim_count, 4) if claim_count else None
            ),
            "median_latency_ms": (round(statistics.median(latencies), 2) if latencies else None),
            "p95_latency_ms": (round(sorted_latency[p95_index], 2) if sorted_latency else None),
            "required_vision_routing_recall": (
                round(vision_correctly_routed / vision_required, 4) if vision_required else None
            ),
        },
        "cases": rows,
    }
