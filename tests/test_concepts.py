from paper_agent.concepts import (
    _clean_evidence_sentence,
    create_deterministic_concept_card,
    sentence_windows,
)
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.workspace import Workspace


def sparse_attention_paper() -> ParsedPaper:
    text = (
        "While modern sparse attention mechanisms reduce computational FLOPs per decoding step, the KV cache "
        "still grows linearly.\nTo resolve this dilemma, we present Lookahead Sparse Attention (LSA), a predictive "
        "variant that fetches selected chunks."
    )
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="LSA Paper", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text=text,
        page_text={1: text},
        sections={"introduction": text},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


def test_sentence_windows_rejoin_pdf_line_wrapped_sentences():
    text = "Sparse attention reduces compute but the KV cache\nstill grows with sequence length. Next sentence."

    evidence = sentence_windows(text, "Sparse attention")

    assert evidence == [
        "Sparse attention reduces compute but the KV cache still grows with sequence length."
    ]


def test_generic_term_is_not_presented_as_the_paper_specific_method(tmp_path):
    card = create_deterministic_concept_card(
        sparse_attention_paper(), Workspace(tmp_path / "workspace"), "Sparse Attention"
    )

    assert "broader mechanism family" in card.paper_specific_meaning
    assert "Lookahead Sparse Attention" in card.paper_specific_meaning


def test_extracted_equation_tail_is_removed_from_prose_evidence():
    evidence = _clean_evidence_sentence(
        "It applies Multi-Query Attention to select the final entries: "
        "CCoreComp i = CComp s ∈ CMemComp t Score(i, s) ∈ Top-k."
    )

    assert evidence == "It applies Multi-Query Attention to select the final entries."
