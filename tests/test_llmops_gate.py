from datetime import datetime, timezone

from paper_agent.llmops_gate import evaluate_score_gate


def gate_config():
    return {
        "environment": "test",
        "lookbackHours": 24,
        "metrics": {
            "math_overall": {"minimumMean": 4.0, "minimumSamples": 2},
            "math_passed": {"minimumMean": 0.75, "minimumSamples": 2},
        },
    }


def test_gate_passes_when_quality_and_sample_thresholds_pass():
    report = evaluate_score_gate(
        [
            {"name": "math_overall", "value": 4.0},
            {"name": "math_overall", "value": 4.5},
            {"name": "math_passed", "value": True},
            {"name": "math_passed", "value": True},
        ],
        gate_config(),
        now=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )

    assert report.passed is True
    assert report.status == "pass"
    assert report.metrics[0].mean == 4.25


def test_gate_fails_on_regression():
    report = evaluate_score_gate(
        [
            {"name": "math_overall", "value": 3.0},
            {"name": "math_overall", "value": 3.5},
            {"name": "math_passed", "value": 1},
            {"name": "math_passed", "value": 0},
        ],
        gate_config(),
    )

    assert report.passed is False
    assert report.status == "fail"
    assert any("math_overall" in item for item in report.failures)
    assert any("math_passed" in item for item in report.failures)


def test_gate_requires_enough_data_unless_explicitly_allowed():
    strict = evaluate_score_gate(
        [{"name": "math_overall", "value": 5.0}],
        gate_config(),
    )
    exploratory = evaluate_score_gate(
        [{"name": "math_overall", "value": 5.0}],
        gate_config(),
        allow_insufficient_data=True,
    )

    assert strict.passed is False
    assert strict.metrics[0].status == "insufficient"
    assert exploratory.passed is True
    assert exploratory.status == "pass_with_insufficient_data"


def test_gate_scopes_scores_to_configured_release():
    config = gate_config()
    config["release"] = "release-new"
    scores = [
        {"name": "math_overall", "value": 1.0, "release": "release-old"},
        {"name": "math_passed", "value": 0.0, "release": "release-old"},
        {"name": "math_overall", "value": 4.5, "release": "release-new"},
        {"name": "math_overall", "value": 4.5, "release": "release-new"},
        {"name": "math_passed", "value": 1.0, "release": "release-new"},
        {"name": "math_passed", "value": 1.0, "release": "release-new"},
    ]

    report = evaluate_score_gate(scores, config)

    assert report.passed is True
    assert report.release == "release-new"
    assert report.score_count == 4
    assert report.fetched_score_count == 6
