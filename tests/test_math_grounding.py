from paper_agent.math_grounding import (
    build_math_context,
    contextual_math_evidence,
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


def test_voting_equation_filters_environment_noise_and_resolves_symbols():
    latex = r"V_{i,s} = \sum_{l=1}^{L} \mathbb{I}(s \in M_{i,l})"
    page = (
        "Step 2: Top-p Thresholding. An entry s is marked as selected by layer l. "
        "Step 3: Cross-Layer Majority Voting. We aggregate selection hits across all L layers. "
        "The voting score V_i,s for entry s at token step i counts how many layers selected it."
    )
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Voting Paper", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text=page,
        page_text={1: page},
        sections={"method": page},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )

    assert extract_latex_symbols(r"\begin{aligned} " + latex + r" \end{aligned}") == [
        r"V_{i,s}",
        r"\mathbb{I}",
        r"M_{i,l}",
        "L",
    ]

    bundle = build_math_context(parsed, 1, latex, "cross-layer voting")
    meanings = {item.symbol: item for item in bundle.symbols}

    assert "voting score" in meanings[r"V_{i,s}"].meaning.lower()
    assert "indicator" in meanings[r"\mathbb{I}"].meaning.lower()
    assert "top-p" in meanings[r"M_{i,l}"].meaning.lower()
    assert "layers" in meanings["L"].meaning.lower()
    assert all(len(item) <= 523 for item in bundle.evidence)


def test_union_equation_resolves_window_and_label_set_from_paper():
    latex = (
        r"Y_{t}^{+} = \bigcup_{i=t}^{t+\tau-1} "
        r"A_{i}^{\mathrm{golden}}"
    )
    page = (
        "For each lookahead evaluation window, the positive ground-truth label set "
        "Y+t is established by taking the union of denoised golden entries A golden "
        "over the future lookahead window of tau decoding steps."
    )
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Union Paper", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text=page,
        page_text={1: page},
        sections={"method": page},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )

    bundle = build_math_context(parsed, 1, latex, "positive label set")
    meanings = {item.symbol: item for item in bundle.symbols}

    assert extract_latex_symbols(latex) == [
        r"Y_{t}^{+}",
        r"A_{i}^{\mathrm{golden}}",
        r"\tau",
    ]
    assert "positive ground-truth" in meanings[r"Y_{t}^{+}"].meaning.lower()
    assert "golden-entry" in meanings[r"A_{i}^{\mathrm{golden}}"].meaning.lower()
    assert "future" in meanings[r"\tau"].meaning.lower()
    assert "decoding steps" in meanings[r"\tau"].meaning.lower()
    assert bundle.evidence


def test_contextual_evidence_prefers_definition_sentences():
    evidence = contextual_math_evidence(
        "Current page 5:\nBackground text without a definition. "
        "The positive label set is established by taking the union over the lookahead window."
    )

    assert evidence
    assert "union" in evidence[0].lower()


def test_grounding_generalizes_to_unseen_linear_model_notation():
    latex = r"z_i = W x_i + b"
    page = (
        "For every item i, x_i denotes its input feature vector. "
        "W is the learned projection matrix, b is the learned bias vector, "
        "and z_i denotes the resulting projected representation."
    )
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Projection Model", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text=page,
        page_text={1: page},
        sections={"method": page},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )

    bundle = build_math_context(parsed, 1, latex, "linear projection")
    meanings = {item.symbol: item for item in bundle.symbols}

    assert "projected representation" in meanings[r"z_i"].meaning.lower()
    assert "projection matrix" in meanings["W"].meaning.lower()
    assert "feature vector" in meanings[r"x_i"].meaning.lower()
    assert "bias vector" in meanings["b"].meaning.lower()
    assert all(item.source == "paper" for item in meanings.values())
