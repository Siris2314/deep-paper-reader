from pathlib import Path

from paper_agent.term_relevance_eval import evaluate_term_labels, run_term_relevance_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_labeled_term_metrics_count_known_positive_and_negative_labels():
    metrics = evaluate_term_labels(
        ["DPO", "reward model", "JSON"],
        ["DPO", "reward model", "human preferences"],
        ["JSON", "PDF"],
    )

    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 1
    assert metrics.precision == 0.6667
    assert metrics.recall == 0.6667


def test_checked_in_term_relevance_benchmark_passes():
    result = run_term_relevance_benchmark(REPO_ROOT / "llmops" / "term_relevance_benchmark.json")

    assert result["passed"] is True
    assert result["aggregate"]["precision"] >= result["thresholds"]["minimum_precision"]
    assert result["aggregate"]["recall"] >= result["thresholds"]["minimum_recall"]
