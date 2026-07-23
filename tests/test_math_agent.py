from types import SimpleNamespace

from paper_agent.math_agent import _clean_generated_list, analyze_expression, explain_equation
from paper_agent.math_judge import MathJudgeResult
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.workspace import Workspace


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
