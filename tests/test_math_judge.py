from paper_agent.math_judge import (
    MathJudgePayload,
    JudgeScore,
    deterministic_math_checks,
    evaluate_math_explanation,
)
from paper_agent.workspace import Workspace


def explanation_payload():
    return {
        "raw_equation": "cQ_t = h_t W_DQ",
        "paper_context": "The paper maps the current hidden state into a compressed query.",
        "display_latex": r"cQ_t = h_t W_{DQ}",
        "latex_source": "paper_math_agent",
        "parse": {"status": "unparsed"},
        "role": "update rule",
        "plain_english": "The hidden state is projected into a compressed query.",
        "steps": ["Read the hidden state.", "Apply the projection matrix."],
        "symbols": [
            {"symbol": "h_t", "meaning": "current hidden state", "source": "paper"},
            {"symbol": "W_DQ", "meaning": "down-projection matrix", "source": "paper"},
        ],
        "intuition": "The projection compresses the query before per-head expansion.",
        "dimensional_analysis": "The hidden-state width must match the first dimension of W_DQ.",
        "implementation_view": "compressed_query = hidden_state @ W_DQ",
        "paper_evidence": ["The paper maps the current hidden state into a compressed query."],
        "context_fit": "This creates the query used by the indexer.",
        "model": "ollama:qwen2.5:7b",
    }


def test_deterministic_judge_rejects_missing_latex():
    payload = explanation_payload()
    payload["display_latex"] = ""

    checks = deterministic_math_checks(payload)

    assert checks["latex_present"] is False


def test_independent_math_judge_scores_and_justifies(tmp_path):
    class FakeJudge:
        def invoke(self, payload):
            score = JudgeScore(
                score=5, justification="Supported by the equation and paper context."
            )
            return {
                "structured_response": MathJudgePayload(
                    correctness=score,
                    paper_grounding=score,
                    symbol_coverage=score,
                    latex_fidelity=score,
                    usefulness=score,
                    summary="The explanation is faithful and grounded.",
                    issues=[],
                )
            }

    result = evaluate_math_explanation(
        Workspace(tmp_path / "workspace"),
        explanation_payload(),
        chat_model=object(),
        agent_factory=lambda **kwargs: FakeJudge(),
    )

    assert result.verdict == "pass"
    assert result.overall_score == 5.0
    assert result.judge_model.startswith("ollama:")


def test_default_judge_uses_direct_json_schema_output(tmp_path):
    score = JudgeScore(score=4, justification="The explanation is supported and useful.")

    class StructuredJudge:
        def invoke(self, messages):
            assert messages[0]["role"] == "system"
            return MathJudgePayload(
                correctness=score,
                paper_grounding=score,
                symbol_coverage=score,
                latex_fidelity=score,
                usefulness=score,
                summary="A grounded explanation with faithful LaTeX.",
                issues=[],
            )

    class FakeChatModel:
        def with_structured_output(self, schema, method):
            assert schema is MathJudgePayload
            assert method == "json_schema"
            return StructuredJudge()

    result = evaluate_math_explanation(
        Workspace(tmp_path / "workspace"),
        explanation_payload(),
        chat_model=FakeChatModel(),
    )

    assert result.verdict == "pass"
    assert result.overall_score == 4.0


def test_deterministic_failure_overrides_generous_judge(tmp_path):
    class GenerousJudge:
        def invoke(self, payload):
            score = JudgeScore(score=5, justification="Looks good.")
            return {
                "structured_response": MathJudgePayload(
                    correctness=score,
                    paper_grounding=score,
                    symbol_coverage=score,
                    latex_fidelity=score,
                    usefulness=score,
                    summary="Generous assessment.",
                    issues=[],
                )
            }

    payload = explanation_payload()
    payload["display_latex"] = ""
    result = evaluate_math_explanation(
        Workspace(tmp_path / "workspace"),
        payload,
        chat_model=object(),
        agent_factory=lambda **kwargs: GenerousJudge(),
    )

    assert result.verdict == "fail"
    assert any("No display LaTeX" in issue for issue in result.issues)
