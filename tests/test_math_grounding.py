from paper_agent.math_grounding import (
    build_math_context,
    deterministic_math_fallback,
    extract_latex_symbols,
)
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata


FOCAL_LATEX = (
    r"\mathcal{L}_{\text{FL}} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} "
    r"w_{t,s}(1-p_{t,s}^{(\text{correct})})^\gamma "
    r"\ell_{\text{BCE}}(I_{t,s}, y_{t,s})"
)


def focal_paper() -> ParsedPaper:
    page = (
        "We train the Memory Indexer using focal loss. Let S be the samples in the batch, "
        "w_t,s the per-sample weight, and p(correct) the predicted confidence on the correct class. "
        "The focusing parameter gamma is set to 2. L_BCE is binary cross-entropy between the "
        "lookahead index score I_t,s and binary label y_t,s."
    )
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Memory Indexer", authors_guess=[], page_count=7, source_pdf="paper.pdf"
        ),
        full_text=page,
        page_text={7: page},
        sections={"method": page},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


def test_extracts_complete_focal_loss_symbol_set():
    symbols = extract_latex_symbols(FOCAL_LATEX)

    assert r"\mathcal{L}_{FL}" in symbols
    assert r"\mathcal{S}" in symbols
    assert r"p_{t,s}^{(correct)}" in symbols
    assert r"\ell_{BCE}" in symbols
    assert r"I_{t,s}" in symbols


def test_position_sensitive_grounding_resolves_paper_definitions():
    bundle = build_math_context(focal_paper(), 7, FOCAL_LATEX, "focal loss")
    meanings = {item.symbol: item for item in bundle.symbols}

    assert meanings[r"\gamma"].source == "paper"
    assert "focusing" in meanings[r"\gamma"].meaning
    assert meanings[r"\ell_{BCE}"].source == "paper"
    assert "binary cross-entropy" in meanings[r"\ell_{BCE}"].meaning.lower()
    assert all(item.evidence for item in bundle.symbols)


def test_grounded_fallback_remains_useful_when_model_fails():
    bundle = build_math_context(focal_paper(), 7, FOCAL_LATEX, "focal loss")
    fallback = deterministic_math_fallback(FOCAL_LATEX, bundle)

    assert fallback["role"] == "sample-weighted focal-loss training objective"
    assert len(fallback["steps"]) == 4
    assert "difficult" in fallback["intuition"]
    assert "scalar" in fallback["dimensional_analysis"]
