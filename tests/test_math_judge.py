from contextlib import contextmanager

import paper_agent.math_judge as math_judge
from paper_agent.math_judge import (
    MathJudgePayload,
    JudgeScore,
    _judge_messages,
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


def test_deterministic_judge_rejects_malformed_inline_math():
    payload = explanation_payload()
    payload["steps"] = [
        "Sort $ ext{Sorted}(P)$.",
        "Compare $x imes y$ before returning the selected set.",
    ]

    checks = deterministic_math_checks(payload)

    assert checks["inline_math_valid"] is False
    assert any("damaged LaTeX command" in issue for issue in checks["inline_math_issues"])


def test_deterministic_judge_accepts_supported_dots_notation():
    payload = explanation_payload()
    payload["steps"] = [r"Read the sequence $x_1, x_2, \dots, x_n$."]

    checks = deterministic_math_checks(payload)

    assert checks["inline_math_valid"] is True


def test_deterministic_judge_rejects_bare_latex_in_prose():
    payload = explanation_payload()
    payload["dimensional_analysis"] = (
        r"[\mathcal{M}_{i,l}: \text{Set of indices } s], "
        r"[P_{i,l,j}: \text{scalar probability in range }[0,1]]."
    )

    checks = deterministic_math_checks(payload)

    assert checks["inline_math_valid"] is False
    assert any("outside math delimiters" in issue for issue in checks["inline_math_issues"])


def test_textstyle_is_valid_when_delimited():
    payload = explanation_payload()
    payload["steps"] = [r"Compute $\textstyle \sum_i p_i$."]

    assert deterministic_math_checks(payload)["inline_math_valid"] is True


def test_standard_probability_and_supremum_commands_are_valid():
    payload = explanation_payload()
    payload["steps"] = [
        r"Compare $\pi_\theta(y \mid x)$ with $\pi_{\mathrm{ref}}(y \mid x)$.",
        r"Take $\sup_{q \in \mathcal{Q}} f(q)$ over the feasible family.",
    ]

    checks = deterministic_math_checks(payload)

    assert checks["inline_math_valid"] is True


def test_uncertain_transcription_is_a_deterministic_failure(tmp_path):
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
    payload["transcription_confidence"] = 0.4
    payload["transcription_warnings"] = ["Vision and PDF geometry disagree."]

    result = evaluate_math_explanation(
        Workspace(tmp_path / "workspace"),
        payload,
        chat_model=object(),
        agent_factory=lambda **kwargs: GenerousJudge(),
    )

    assert result.verdict == "fail"
    assert result.deterministic_checks["transcription_confident"] is False
    assert any("transcription" in issue.lower() for issue in result.issues)


def test_vision_transcription_requires_bounded_crop_provenance(tmp_path):
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
    payload["latex_source"] = "vision_math_agent"
    payload["transcription_confidence"] = 0.8
    payload["transcription_candidates"] = {"vision_math_agent": payload["display_latex"]}

    result = evaluate_math_explanation(
        Workspace(tmp_path / "workspace"),
        payload,
        chat_model=object(),
        agent_factory=lambda **kwargs: GenerousJudge(),
    )

    assert result.verdict == "fail"
    assert result.deterministic_checks["vision_crop_grounded"] is False
    assert any("bounded crop" in issue.lower() for issue in result.issues)


def test_malformed_crop_provenance_is_rejected_without_crashing():
    payload = explanation_payload()
    payload["latex_source"] = "vision_math_agent"
    payload["layout_evidence"] = {
        "crop_provenance": {
            "sha256": "abc123",
            "pixelWidth": "not-a-number",
            "pixelHeight": None,
        }
    }

    checks = deterministic_math_checks(payload)

    assert checks["vision_crop_grounded"] is False


def test_judge_prompt_includes_human_alignment_memory():
    messages = _judge_messages(
        explanation_payload(),
        deterministic_math_checks(explanation_payload()),
        {
            "principles": [
                {
                    "dimension": "latex_fidelity",
                    "principle": "Do not penalize valid MathJax style commands.",
                }
            ],
            "episodes": [],
        },
    )

    assert "Human-aligned judge working memory" in messages[1]["content"]
    assert "Do not penalize valid MathJax style commands" in messages[1]["content"]


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


def test_judge_trace_names_identify_repair_phase(tmp_path, monkeypatch):
    observed_names = []

    @contextmanager
    def fake_stage(name, **kwargs):
        observed_names.append(name)

        class Stage:
            def update(self, **values):
                pass

        yield Stage()

    def fake_invoke(runnable, payload, *, name, **kwargs):
        observed_names.append(name)
        return runnable.invoke(payload)

    score = JudgeScore(score=4, justification="The explanation is supported and useful.")

    class FakeJudge:
        def invoke(self, payload):
            return {
                "structured_response": MathJudgePayload(
                    correctness=score,
                    paper_grounding=score,
                    symbol_coverage=score,
                    latex_fidelity=score,
                    usefulness=score,
                    summary="A grounded explanation.",
                    issues=[],
                )
            }

    monkeypatch.setattr(math_judge, "observed_stage", fake_stage)
    monkeypatch.setattr(math_judge, "invoke_observed", fake_invoke)
    evaluate_math_explanation(
        Workspace(tmp_path / "workspace"),
        explanation_payload(),
        chat_model=object(),
        agent_factory=lambda **kwargs: FakeJudge(),
        phase="repair-1",
    )

    assert observed_names == ["validate-repair-1", "judge-after-repair-1"]


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


def test_malformed_inline_math_overrides_generous_judge(tmp_path):
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
    payload["plain_english"] = "The value is $x imes y$."
    result = evaluate_math_explanation(
        Workspace(tmp_path / "workspace"),
        payload,
        chat_model=object(),
        agent_factory=lambda **kwargs: GenerousJudge(),
    )

    assert result.verdict == "fail"
    assert result.deterministic_checks["inline_math_valid"] is False
    assert any("damaged LaTeX command" in issue for issue in result.issues)
