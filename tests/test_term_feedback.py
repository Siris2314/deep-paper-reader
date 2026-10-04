from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.term_feedback import (
    load_term_feedback,
    record_term_feedback,
    suppressed_relevance_keys,
)
from paper_agent.term_highlighter import save_significant_terms
from paper_agent.term_relevance_eval import evaluate_workspace_feedback
from paper_agent.workspace import Workspace


def _paper() -> ParsedPaper:
    text = (
        "We introduce Atlas Retrieval Network (ARN). ARN is our retrieval model. "
        "The implementation also emits JSON records."
    )
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Atlas Retrieval Network",
            authors_guess=[],
            page_count=1,
            source_pdf="paper.pdf",
        ),
        full_text=text,
        page_text={1: text},
        sections={"abstract": text, "method": text},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


def test_paper_local_irrelevant_feedback_suppresses_term_and_can_be_revised(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr("paper_agent.term_highlighter.load_learned_terms", lambda: {})
    monkeypatch.setattr("paper_agent.term_highlighter.load_method_vocabulary", lambda: [])
    parsed = _paper()
    initial = save_significant_terms(parsed, workspace)
    target = next(item for item in initial if item.term == "ARN")

    first = record_term_feedback(
        workspace,
        target.term,
        "irrelevant",
        category=target.category,
        score=target.score,
        vocabulary_source=target.vocabulary_source,
        page=1,
    )
    suppressed = save_significant_terms(parsed, workspace)

    assert first["revision_count"] == 1
    assert "arn" in suppressed_relevance_keys(workspace)
    assert all(item.term != "ARN" for item in suppressed)
    assert evaluate_workspace_feedback(workspace, suppressed)["true_negatives"] == 1

    revised = record_term_feedback(workspace, "ARN", "relevant", page=1)
    restored = save_significant_terms(parsed, workspace)

    assert revised["revision_count"] == 2
    assert not suppressed_relevance_keys(workspace)
    assert any(item.term == "ARN" for item in restored)
    assert load_term_feedback(workspace)["labels"]["arn"]["label"] == "relevant"
    assert evaluate_workspace_feedback(workspace, restored)["recall"] == 1.0


def test_feedback_is_scoped_to_its_workspace(tmp_path):
    first = Workspace(tmp_path / "first")
    second = Workspace(tmp_path / "second")

    record_term_feedback(first, "reward model", "irrelevant")

    assert suppressed_relevance_keys(first) == {"rewardmodel"}
    assert suppressed_relevance_keys(second) == set()
