import fitz

from paper_agent.parser import parse_paper
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.term_highlighter import extract_significant_terms


def test_significant_terms_include_ml_or_method_terms(tmp_path, sample_pdf):
    parsed = parse_paper(sample_pdf, tmp_path / "report")
    terms = extract_significant_terms(parsed, max_terms=40)

    assert terms
    assert any(term.category in {"math", "method", "result", "concept"} for term in terms)
    assert all("paper-term-highlighter" in term.skills for term in terms)


def test_user_taught_term_outranks_heuristic_terms(tmp_path, sample_pdf):
    parsed = parse_paper(sample_pdf, tmp_path / "report")
    learned = {
        "model": {
            "term": "model",
            "category": "method",
            "correction_count": 2,
        }
    }

    terms = extract_significant_terms(parsed, max_terms=100, learned_terms=learned)
    model = next(term for term in terms if term.term.lower() == "model")

    assert model.learned is True
    assert model.correction_count == 2
    assert model.category == "method"


def test_phrase_first_extraction_finds_research_spans_and_ignores_metadata(tmp_path):
    pdf = tmp_path / "research-phrases.pdf"
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((18, 500), "arXiv:2305.18290v3", fontsize=9, rotate=90)
    page.insert_text((54, 80), "Abstract", fontsize=14)
    page.insert_text(
        (54, 110),
        "Large unsupervised language models learn broad capabilities from training data.",
        fontsize=10,
    )
    page.insert_text(
        (54, 130),
        "We fit a reward model and then use reinforcement learning from human feedback.",
        fontsize=10,
    )
    document.save(pdf)
    document.close()

    parsed = parse_paper(pdf, tmp_path / "report")
    terms = extract_significant_terms(parsed, max_terms=100, learned_terms={}, vocabulary=[])
    names = {term.term.casefold() for term in terms}

    assert "large unsupervised language models" in names
    assert "reward model" in names
    assert "reinforcement learning" in names
    assert "arxiv" not in names
    assert "model" not in names
    assert "training" not in names


def _parsed_text(text: str, sections: dict[str, str] | None = None) -> ParsedPaper:
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Atlas Retrieval Network",
            authors_guess=[],
            page_count=1,
            source_pdf="paper.pdf",
        ),
        full_text=text,
        page_text={1: text},
        sections=sections or {"abstract": text, "method": text},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


def test_relevance_ranking_keeps_defined_methods_and_rejects_sentence_fragments():
    text = (
        "We introduce Atlas Retrieval Network (ARN), a sparse retrieval architecture. "
        "We generate data with a procedure we call Agentic Self-Instruct. "
        "ARN uses a contrastive objective, and the contrastive objective improves retrieval. "
        "ARN outperforms baselines throughout training. Results comparing training are reported. "
        "The implementation emits JSON records, and JSON validation runs before evaluation."
    )

    terms = extract_significant_terms(
        _parsed_text(text), max_terms=50, learned_terms={}, vocabulary=[]
    )
    names = {item.term.casefold() for item in terms}

    assert "atlas retrieval network" in names
    assert "agentic self-instruct" in names
    assert "arn" in names
    assert "contrastive objective" in names
    assert "json" not in names
    assert "throughout training" not in names
    assert "results comparing training" not in names
    assert (
        next(item for item in terms if item.term.casefold() == "atlas retrieval network").category
        == "method"
    )


def test_acronym_expansion_uses_shortest_matching_long_form():
    text = (
        "We minimize a standard element-wise Binary Cross-Entropy (BCE) objective. "
        "BCE is stable, and BCE is used for every sample."
    )

    terms = extract_significant_terms(
        _parsed_text(text), max_terms=50, learned_terms={}, vocabulary=[]
    )
    names = {item.term.casefold() for item in terms}

    assert "binary cross-entropy" in names
    assert "bce" in names
    assert not any(name.startswith("we minimize") for name in names)


def test_spacing_and_hyphen_aliases_share_one_ranked_entry():
    text = (
        "CoT-Self-Instruct is our data generation framework. "
        "CoT Self-Instruct creates training examples. "
        "CoT-Self-Instruct improves the resulting model."
    )

    terms = extract_significant_terms(
        _parsed_text(text), max_terms=50, learned_terms={}, vocabulary=[]
    )
    aliases = [
        item
        for item in terms
        if "".join(char for char in item.term.casefold() if char.isalnum()) == "cotselfinstruct"
    ]

    assert len(aliases) == 1
