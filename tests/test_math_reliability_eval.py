from pathlib import Path

from paper_agent.math_reliability_eval import (
    baseline_predictions,
    evaluate_math_reliability,
    load_math_benchmark,
)


def test_checked_in_math_benchmark_is_complete():
    benchmark = load_math_benchmark(Path("llmops/math_reliability_benchmark.json"))

    assert len(benchmark["cases"]) == 25
    assert {case["source"]["paper_id"] for case in benchmark["cases"]} == {
        "arXiv:2305.18290",
        "arXiv:2606.09079v2",
    }
    assert all(case["expected"]["nearby_definitions"] for case in benchmark["cases"])


def test_math_reliability_report_scores_all_dimensions():
    benchmark = load_math_benchmark(Path("llmops/math_reliability_benchmark.json"))
    report = evaluate_math_reliability(benchmark, baseline_predictions(benchmark))

    assert report["case_count"] == 25
    assert report["coverage"] == 1.0
    assert set(report["metrics"]) >= {
        "exact_latex_accuracy",
        "mean_symbol_recall",
        "mean_symbol_grounding_recall",
        "boundary_accuracy",
        "role_accuracy",
        "unsupported_claim_rate",
        "median_latency_ms",
        "required_vision_routing_recall",
    }
