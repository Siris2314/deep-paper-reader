import json

import paper_agent.judge_memory as judge_memory
from paper_agent.judge_memory import (
    record_math_judge_feedback,
    retrieve_math_judge_memory,
)
from paper_agent.workspace import Workspace


def explanation(latex: str = r"M_{i,l} = \{s \mid P_{i,l,s} \leq p\}"):
    return {
        "equation_id": "math-5-display-2",
        "page": 5,
        "display_latex": latex,
        "role": "Top-p threshold set",
        "plain_english": "The equation retains entries under a cumulative probability threshold.",
        "context_fit": "This filters history before cross-layer voting.",
        "symbols": [{"symbol": "p", "meaning": "nucleus threshold", "source": "paper"}],
        "evaluation": {
            "verdict": "fail",
            "overall_score": 3.2,
            "summary": "The judge over-penalized valid notation.",
        },
    }


def test_feedback_builds_semantic_and_episodic_judge_memory(tmp_path, monkeypatch):
    memory_path = tmp_path / "agent_memory" / "math_judge_alignment.json"
    monkeypatch.setattr(judge_memory, "_memory_path", lambda: memory_path)
    workspace = Workspace(tmp_path / "workspace")
    principle = "Do not penalize valid MathJax commands merely because they are stylistic."

    record_math_judge_feedback(
        workspace,
        explanation=explanation(),
        rating="disagree",
        dimension="latex_fidelity",
        feedback=principle,
        remember=True,
    )
    record_math_judge_feedback(
        workspace,
        explanation=explanation(),
        rating="disagree",
        dimension="latex_fidelity",
        feedback=principle,
        remember=True,
    )

    payload = json.loads(memory_path.read_text(encoding="utf-8"))
    relevant = retrieve_math_judge_memory(explanation())

    assert len(payload["semantic"]) == 1
    assert payload["semantic"][0]["feedbackCount"] == 2
    assert len(payload["episodes"]) == 1
    assert relevant["principles"][0]["principle"] == principle
    assert relevant["episodes"][0]["equationKey"] == payload["episodes"][0]["equationKey"]
    assert workspace.path("math/judge_feedback.jsonl").exists()


def test_paper_only_feedback_does_not_update_long_term_memory(tmp_path, monkeypatch):
    memory_path = tmp_path / "agent_memory" / "math_judge_alignment.json"
    monkeypatch.setattr(judge_memory, "_memory_path", lambda: memory_path)

    record_math_judge_feedback(
        Workspace(tmp_path / "workspace"),
        explanation=explanation(),
        rating="disagree",
        feedback="This correction is specific to the current paper.",
        remember=False,
    )

    assert not memory_path.exists()
