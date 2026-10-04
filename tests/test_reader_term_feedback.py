from paper_agent.parser import parse_paper
from paper_agent.term_feedback import load_term_feedback, record_term_feedback
from paper_agent.term_highlighter import save_significant_terms
from paper_agent.workspace import Workspace
from ui.paper_reader_server import reset_uploaded_workspace, suppress_highlight


def test_reader_suppression_removes_term_and_returns_feedback_metrics(
    tmp_path, sample_pdf, monkeypatch
):
    monkeypatch.setenv("PAPER_TERM_MEMORY_PATH", str(tmp_path / "global-term-memory.json"))
    workspace = Workspace(tmp_path / "workspace")
    parsed = parse_paper(sample_pdf, workspace.root)
    terms = save_significant_terms(parsed, workspace)
    target = next(item for item in terms if item.term.casefold() == "sparse attention")

    result = suppress_highlight(workspace, parsed, target.term, page=1)

    assert result["ok"] is True
    assert all(item["term"].casefold() != "sparse attention" for item in result["state"]["terms"])
    assert result["state"]["termFeedback"]["irrelevant_labels"] == 1
    assert result["state"]["termFeedback"]["true_negatives"] == 1


def test_same_pdf_workspace_reset_preserves_reviewed_term_labels(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    workspace.write_text("scratch/transient.txt", "discard me")
    record_term_feedback(workspace, "reward model", "irrelevant", page=2)

    reset_uploaded_workspace(workspace)

    assert not workspace.path("scratch/transient.txt").exists()
    assert load_term_feedback(workspace)["labels"]["rewardmodel"]["label"] == "irrelevant"
