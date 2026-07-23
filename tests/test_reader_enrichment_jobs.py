import time

from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.workspace import Workspace
from ui import paper_reader_server as reader


def test_enrichment_runs_as_background_job(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Paper", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text="KV cache evidence.",
        page_text={1: "KV cache evidence."},
        sections={"method": "KV cache evidence."},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )

    def fake_enrich(workspace, parsed, term, progress=None):
        progress("tavily_research", "Researching")
        progress("paper_agent", "Analyzing")
        return {"ok": True, "message": "Complete", "term": term}

    monkeypatch.setattr(reader, "enrich_term_with_tavily", fake_enrich)
    with reader.ENRICHMENT_LOCK:
        reader.ENRICHMENT_JOBS.clear()

    started = reader.start_enrichment_job(workspace, parsed, "KV cache")
    deadline = time.monotonic() + 2
    job = started
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        job = reader.get_enrichment_job(str(started["jobId"]))

    assert job["status"] == "completed"
    assert job["message"] == "Complete"


def test_stalled_enrichment_job_gets_a_terminal_timeout(monkeypatch):
    monkeypatch.setattr(reader, "ENRICHMENT_JOB_TIMEOUT", 0.01)
    job_id = "stalled-job"
    with reader.ENRICHMENT_LOCK:
        reader.ENRICHMENT_JOBS[job_id] = {
            "id": job_id,
            "key": "workspace::term",
            "status": "running",
            "stage": "tavily_research",
            "message": "Waiting",
            "started_monotonic": time.monotonic() - 1,
            "updated_at": time.time(),
            "result": None,
        }

    job = reader.get_enrichment_job(job_id)

    assert job["status"] == "failed"
    assert job["stage"] == "timeout"


def test_unexpected_enrichment_exception_becomes_terminal_failure(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Paper", authors_guess=[], page_count=1, source_pdf="paper.pdf"
        ),
        full_text="Sparse attention evidence.",
        page_text={1: "Sparse attention evidence."},
        sections={"method": "Sparse attention evidence."},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )
    monkeypatch.setattr(
        reader,
        "enrich_term_with_tavily",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("worker crashed")),
    )
    with reader.ENRICHMENT_LOCK:
        reader.ENRICHMENT_JOBS.clear()

    started = reader.start_enrichment_job(workspace, parsed, "Sparse Attention")
    deadline = time.monotonic() + 2
    job = started
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        job = reader.get_enrichment_job(str(started["jobId"]))

    assert job["status"] == "failed"
    assert "worker crashed" in job["message"]
