import json
from types import SimpleNamespace

from paper_agent.math_agent import (
    _clean_generated_list,
    _compact_repair_payload,
    _repair_generated_math_text,
    _required_feedback_resolved,
    _select_transcription,
    _specific_role,
    analyze_expression,
    explain_equation,
)
from paper_agent.math_judge import MathJudgeResult
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.workspace import Workspace


def math_payload(plain_english: str) -> dict[str, object]:
    return {
        "latex": "s = x^2 + y",
        "role": "definition",
        "plain_english": plain_english,
        "steps": ["Square x.", "Add y."],
        "symbols": [
            {"symbol": "x", "meaning": "input", "source": "paper"},
            {"symbol": "y", "meaning": "target", "source": "paper"},
        ],
        "intuition": "The score combines both terms.",
        "dimensional_analysis": "The squared input and target must be add-compatible.",
        "implementation_view": "s = x*x + y",
        "derivation_notes": "The equation is a direct definition; no additional derivation is shown.",
        "toy_example": "For x = 2 and y = 1, the score is 5.",
        "paper_evidence": ["The paper defines x as input and y as target."],
        "context_fit": "This equation defines the score used by the method.",
        "assumptions_or_missing_details": [],
    }


def judge_result(verdict: str, score: float, issue: str = "") -> MathJudgeResult:
    dimensions = {
        name: {
            "score": score,
            "justification": issue or f"{name} scored {score}.",
        }
        for name in (
            "correctness",
            "paper_grounding",
            "symbol_coverage",
            "latex_fidelity",
            "usefulness",
        )
    }
    return MathJudgeResult(
        verdict=verdict,
        overall_score=score,
        dimensions=dimensions,
        summary=issue or f"{verdict} at {score}.",
        issues=[issue] if issue else [],
        deterministic_checks={},
        judge_model="test-judge",
        evaluated_at="2026-01-01T00:00:00+00:00",
    )


def parsed_math_paper() -> ParsedPaper:
    text = "The score is defined below, where x is the input and y is the target.\ns = x^2 + y"
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Math Paper", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text=text,
        page_text={1: text},
        sections={"method": text},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


def test_sympy_parses_safe_plain_equation():
    parsed = analyze_expression("x^2 + y = 3")

    assert parsed.status == "parsed_plain"
    assert parsed.sympy_form == "Eq(x**2 + y, 3)"
    assert parsed.latex
    assert parsed.free_symbols == ["x", "y"]


def test_sympy_parses_latex_when_available():
    parsed = analyze_expression(r"\frac{x^2}{2}")

    assert parsed.status == "parsed_latex"
    assert parsed.free_symbols == ["x"]


def test_display_transcription_prefers_validated_vision_over_unverified_sympy():
    selection = _select_transcription(
        parsed_latex=r"p = \exp(a + \exp(b))",
        geometry_latex="",
        vision_latex=r"p = \frac{\exp(a)}{\exp(a) + \exp(b)}",
        raw="lossy flattened PDF equation",
        region_kind="display",
    )

    assert selection.latex == r"p = \frac{\exp(a)}{\exp(a) + \exp(b)}"
    assert selection.source == "vision_math_agent"
    assert selection.warnings
    assert selection.confidence < 0.6


def test_compact_repair_payload_omits_large_layout_and_duplicate_context():
    current = math_payload("A grounded explanation.")
    current.update(
        {
            "display_latex": "s = x^2 + y",
            "paper_context": "context " * 5000,
            "layout_evidence": {"spans": [{"text": "x"}] * 5000},
            "evaluation": {"very_large": "feedback " * 5000},
            "loop": {"iterations": ["large"] * 5000},
        }
    )
    evaluation = judge_result("review", 3.0, "Clarify the computation.")

    payload = _compact_repair_payload(
        current,
        evaluation,
        context="nearby paper context " * 1000,
        symbols=current["symbols"],
        evidence=current["paper_evidence"],
    )
    encoded = json.dumps(payload)

    assert "layout_evidence" not in encoded
    assert "very_large" not in encoded
    assert len(encoded) < 14_000


def test_generated_steps_drop_schema_instruction_leaks():
    steps = _clean_generated_list(
        [
            "1. Compute the query projection.",
            "2. Apply the activation.",
            "tensor_dims_explanation_text_2d_or_more_detailed_than_one_line",
            "implementation_view_intuition_with_paper_facts_citations_if_any",
            "symbol_meanings_table_2_to_6_rows_no_empty_cells",
        ],
        max_items=6,
    )

    assert steps == ["Compute the query projection.", "Apply the activation."]


def test_generated_math_repairs_json_escape_control_characters():
    corrupted = (
        "Sort $"
        + "\t"
        + "ext{Sorted}(P)$, compare $x "
        + "\t"
        + "imes y$ and $x "
        + "\n"
        + "eq y$, then return $"
        + "\b"
        + "oldsymbol{M}$."
    )

    repaired = _repair_generated_math_text(corrupted)

    assert r"$\text{Sorted}(P)$" in repaired
    assert r"$x \times y$" in repaired
    assert r"$x \neq y$" in repaired
    assert r"$\mathbf{M}$" in repaired
    assert not any(character in repaired for character in ("\b", "\t", "\n"))


def test_generated_math_collapses_overescaped_commands_and_repairs_limits():
    corrupted = (
        r"Use $V_{i,s} = \\sum\\nlimits_{l=1}^{L}"
        r" \\text{I}(s \\in \\mathcal{M}_{i,l})$."
    )

    repaired = _repair_generated_math_text(corrupted)

    assert repaired == (
        r"Use $V_{i,s} = \sum\limits_{l=1}^{L}"
        r" \text{I}(s \in \mathcal{M}_{i,l})$."
    )


def test_generated_math_wraps_bracketed_latex_descriptions():
    raw = (
        r"[\mathcal{M}_{i,l}: \text{Set of indices } s], "
        r"[P_{i,l,j}: \text{scalar probability in range }[0,1]], "
        r"[p: \text{scalar threshold}]."
    )

    repaired = _repair_generated_math_text(raw)

    assert repaired == (
        r"$\mathcal{M}_{i,l}: \text{Set of indices } s$, "
        r"$P_{i,l,j}: \text{scalar probability in range }[0,1]$, "
        r"$p: \text{scalar threshold}$."
    )


def test_generic_role_is_humanized_from_voting_equation():
    role = _specific_role(
        "mathematical_inference",
        r"V_{i,s} = \sum_{l=1}^{L} \mathbb{I}(s \in M_{i,l})",
    )

    assert role == "cross-layer voting score"


def test_math_agent_explanation_is_cached(tmp_path):
    class FakeAgent:
        def invoke(self, payload):
            return {
                "messages": [
                    SimpleNamespace(
                        content="""{"latex":"s = x^2 + y","role":"definition","plain_english":"The score adds the squared input to the target.","steps":["Square x.","Add y."],"symbols":[{"symbol":"x","meaning":"input","source":"paper"},{"symbol":"y","meaning":"target","source":"paper"}],"intuition":"The score combines both terms.","dimensional_analysis":"x squared and y must be add-compatible.","implementation_view":"s = x*x + y","paper_evidence":["The paper defines x as input and y as target."],"context_fit":"This defines the method score.","assumptions_or_missing_details":[]}"""
                    )
                ]
            }

    workspace = Workspace(tmp_path / "workspace")

    def fake_judge(*args, **kwargs):
        return MathJudgeResult(
            verdict="pass",
            overall_score=5.0,
            dimensions={},
            summary="Good explanation.",
            issues=[],
            deterministic_checks={},
            judge_model="ollama:judge",
            evaluated_at="2026-01-01T00:00:00+00:00",
        )

    first = explain_equation(
        parsed_math_paper(),
        workspace,
        1,
        "math-1-1",
        "s = x^2 + y",
        chat_model=object(),
        agent_factory=lambda **kwargs: FakeAgent(),
        judge_fn=fake_judge,
    )
    second = explain_equation(
        parsed_math_paper(),
        workspace,
        1,
        "math-1-1",
        "s = x^2 + y",
        chat_model=object(),
        agent_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache was not used")),
        judge_fn=fake_judge,
    )

    assert first.status == "agent_explained"
    assert first.display_latex
    assert first.evaluation["verdict"] == "pass"
    assert first.loop["stopReason"] == "passed_initial"
    assert first.symbols[0]["meaning"] == "input"
    assert first.steps == ["Square x.", "Add y."]
    assert second.plain_english == first.plain_english


def test_math_agent_failure_returns_grounded_focal_loss_fallback(tmp_path):
    latex = (
        r"\mathcal{L}_{\mathrm{FL}} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} "
        r"w_{t,s}(1-p_{t,s}^{(\mathrm{correct})})^\gamma \ell_{\mathrm{BCE}}(I_{t,s}, y_{t,s})"
    )
    text = (
        "We use focal loss. Let S be the samples in the batch, w_t,s the per-sample weight, "
        "and p(correct) the confidence on the correct class. L_BCE is binary cross-entropy, "
        "gamma = 2 is the focusing parameter, I_t,s is the index score, and y_t,s is the binary label."
    )
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Focal Paper", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text=text,
        page_text={1: text},
        sections={"method": text},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )

    class FailedAgent:
        def invoke(self, payload):
            raise TimeoutError("local model timed out")

    def fake_judge(*args, **kwargs):
        return MathJudgeResult(
            verdict="pass",
            overall_score=4.5,
            dimensions={},
            summary="Grounded fallback is complete.",
            issues=[],
            deterministic_checks={},
            judge_model="test-judge",
            evaluated_at="2026-01-01T00:00:00+00:00",
        )

    result = explain_equation(
        parsed,
        Workspace(tmp_path / "workspace"),
        1,
        "math-focal",
        "corrupt PDF glyphs",
        layout_evidence={"geometry_latex": latex},
        chat_model=object(),
        agent_factory=lambda **kwargs: FailedAgent(),
        judge_fn=fake_judge,
    )

    assert result.status == "grounded_fallback"
    assert result.role == "sample-weighted focal-loss training objective"
    assert len(result.steps) == 4
    assert any(item["symbol"] == r"\gamma" and item["source"] == "paper" for item in result.symbols)
    assert "local model timed out" in " ".join(result.assumptions_or_missing_details)


def test_math_loop_repairs_and_rejudges_weak_explanation(tmp_path, monkeypatch):
    responses = [
        math_payload("The equation combines two values."),
        math_payload(
            "The equation defines s by squaring the paper's input x and then adding target y. "
            "The first operation captures the nonlinear x contribution; the second combines it "
            "with y to produce the method's score."
        ),
    ]
    judge_results = [
        judge_result("review", 3.0, "The computation is too generic."),
        judge_result("pass", 4.4),
    ]
    agent_calls = 0

    class SequencedAgent:
        def invoke(self, payload):
            nonlocal agent_calls
            response = responses[agent_calls]
            agent_calls += 1
            return {"messages": [SimpleNamespace(content=json.dumps(response))]}

    def fake_judge(*args, **kwargs):
        return judge_results.pop(0)

    monkeypatch.setenv("PAPER_READER_MATH_REPAIR_ATTEMPTS", "1")
    result = explain_equation(
        parsed_math_paper(),
        Workspace(tmp_path / "workspace"),
        1,
        "math-loop-improves",
        "s = x^2 + y",
        chat_model=object(),
        agent_factory=lambda **kwargs: SequencedAgent(),
        judge_fn=fake_judge,
    )

    assert agent_calls == 2
    assert result.status == "agent_repaired"
    assert result.evaluation["verdict"] == "pass"
    assert result.loop["stopReason"] == "passed_after_repair"
    assert result.loop["acceptedRepairs"] == 1
    assert result.loop["initialScore"] == 3.0
    assert result.loop["finalScore"] == 4.4
    assert "direct definition" in result.derivation_notes
    assert "score is 5" in result.toy_example


def test_math_loop_stops_when_repair_repeats_output(tmp_path, monkeypatch):
    payload = math_payload("The equation combines two values.")
    judge_calls = 0

    class RepeatingAgent:
        def invoke(self, request):
            return {"messages": [SimpleNamespace(content=json.dumps(payload))]}

    def fake_judge(*args, **kwargs):
        nonlocal judge_calls
        judge_calls += 1
        return judge_result("review", 3.0, "Add a more concrete explanation.")

    monkeypatch.setenv("PAPER_READER_MATH_REPAIR_ATTEMPTS", "2")
    result = explain_equation(
        parsed_math_paper(),
        Workspace(tmp_path / "workspace"),
        1,
        "math-loop-repeat",
        "s = x^2 + y",
        chat_model=object(),
        agent_factory=lambda **kwargs: RepeatingAgent(),
        judge_fn=fake_judge,
    )

    assert judge_calls == 1
    assert result.loop["attemptsUsed"] == 1
    assert result.loop["acceptedRepairs"] == 0
    assert result.loop["stopReason"] == "repeated_output"


def test_math_loop_rejects_repair_without_score_gain(tmp_path, monkeypatch):
    responses = [
        math_payload("The equation combines two values."),
        math_payload("The equation combines x and y into a score."),
    ]
    judge_results = [
        judge_result("review", 3.0, "The explanation is too generic."),
        judge_result("review", 3.0, "The replacement remains too generic."),
    ]
    agent_calls = 0

    class SequencedAgent:
        def invoke(self, request):
            nonlocal agent_calls
            response = responses[agent_calls]
            agent_calls += 1
            return {"messages": [SimpleNamespace(content=json.dumps(response))]}

    monkeypatch.setenv("PAPER_READER_MATH_REPAIR_ATTEMPTS", "2")
    monkeypatch.setenv("PAPER_READER_MATH_MIN_IMPROVEMENT", "0.1")
    result = explain_equation(
        parsed_math_paper(),
        Workspace(tmp_path / "workspace"),
        1,
        "math-loop-no-gain",
        "s = x^2 + y",
        chat_model=object(),
        agent_factory=lambda **kwargs: SequencedAgent(),
        judge_fn=lambda *args, **kwargs: judge_results.pop(0),
    )

    assert agent_calls == 2
    assert result.plain_english == "The equation combines two values."
    assert result.loop["attemptsUsed"] == 1
    assert result.loop["acceptedRepairs"] == 0
    assert result.loop["stopReason"] == "no_improvement"


def test_math_loop_honors_disabled_repair_budget(tmp_path, monkeypatch):
    agent_calls = 0

    class SingleAgent:
        def invoke(self, request):
            nonlocal agent_calls
            agent_calls += 1
            return {"messages": [SimpleNamespace(content=json.dumps(math_payload("Too brief.")))]}

    monkeypatch.setenv("PAPER_READER_MATH_REPAIR_ATTEMPTS", "0")
    result = explain_equation(
        parsed_math_paper(),
        Workspace(tmp_path / "workspace"),
        1,
        "math-loop-disabled",
        "s = x^2 + y",
        chat_model=object(),
        agent_factory=lambda **kwargs: SingleAgent(),
        judge_fn=lambda *args, **kwargs: judge_result("review", 3.0),
    )

    assert agent_calls == 1
    assert result.loop["attemptBudget"] == 0
    assert result.loop["attemptsUsed"] == 0
    assert result.loop["stopReason"] == "repair_budget_disabled"


def test_repair_must_resolve_required_deterministic_feedback():
    previous = judge_result("review", 3.0)
    previous.deterministic_checks = {
        "required_fields_present": False,
        "missing_fields": ["paper_evidence"],
        "inline_math_valid": False,
        "inline_math_issues": ["Inline math uses unsupported commands: \\bad"],
    }
    unresolved = judge_result("review", 3.8)
    unresolved.deterministic_checks = dict(previous.deterministic_checks)

    accepted, blockers, applied = _required_feedback_resolved(previous, unresolved)

    assert accepted is False
    assert any("paper_evidence" in item for item in blockers)
    assert applied == []

    repaired = judge_result("pass", 4.2)
    repaired.deterministic_checks = {
        "required_fields_present": True,
        "missing_fields": [],
        "inline_math_valid": True,
        "inline_math_issues": [],
    }

    accepted, blockers, applied = _required_feedback_resolved(previous, repaired)

    assert accepted is True
    assert blockers == []
    assert "Added the required paper evidence." in applied
    assert "Corrected the flagged inline math notation." in applied
