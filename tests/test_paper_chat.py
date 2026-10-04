from __future__ import annotations

import time

from paper_agent.paper_chat import (
    ChatEvidence,
    RouteDecision,
    SpecialistOutput,
    _paper_chat_model,
    decide_chat_route,
    load_chat_thread,
    retrieve_paper_evidence,
    run_paper_chat,
    verify_chat_answer,
)
from paper_agent.config import RunConfig
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import EquationCard, FigureCard, PaperMetadata, TableCard
from paper_agent.workspace import Workspace
from ui import paper_reader_server as reader


def sample_paper() -> ParsedPaper:
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="A Better Softmax Model",
            authors_guess=[],
            page_count=3,
            source_pdf="paper.pdf",
        ),
        full_text="Softmax objective method benchmark results.",
        page_text={
            1: "Abstract\nWe introduce a calibrated softmax objective for language model training. "
            * 3,
            2: "Method\nThe loss normalizes every logit with a temperature parameter. " * 3,
            3: "Results\nTable 1 shows that the method improves accuracy over the baseline. " * 3,
        },
        sections={
            "abstract": "We introduce a calibrated softmax objective.",
            "method": "The loss normalizes logits.",
            "results": "The method improves accuracy.",
        },
        equation_cards=[
            EquationCard(id="eq_001", page=2, raw="p_i = exp(z_i / T) / sum_j exp(z_j / T)")
        ],
        figure_cards=[FigureCard(id="figure_1", page=3, caption="Accuracy by temperature")],
        table_cards=[
            TableCard(id="table_1", page=3, caption="Main results", raw_text="ours 91 baseline 87")
        ],
    )


def test_route_uses_one_specialist_on_fast_path():
    route = decide_chat_route("Derive the softmax objective and explain its symbols", "fast")

    assert route.specialists == ["math"]
    assert "math-walkthrough" in route.skills
    assert route.needs_web is False


def test_explore_route_is_bounded_and_enables_web():
    route = decide_chat_route("Analyze the paper", "explore")

    assert route.specialists == ["math", "implementation", "experiments", "concept"]
    assert route.needs_web is True


def test_retrieval_returns_page_and_equation_evidence():
    route = RouteDecision(
        specialists=["math"],
        skills=["math-walkthrough"],
        reason="test",
    )
    evidence = retrieve_paper_evidence(
        sample_paper(), "How does the softmax temperature work?", route
    )

    assert any(item.kind == "page" and item.page in {1, 2} for item in evidence)
    assert any(item.id == "EQ:eq_001" for item in evidence)


def test_verifier_rejects_unresolved_citations():
    evidence = [ChatEvidence(id="P2.1", source="paper", kind="page", text="Paper evidence", page=2)]

    result = verify_chat_answer(
        "This is a sufficiently detailed answer grounded in the paper.", ["P99.1"], evidence
    )

    assert result.passed is False
    assert result.checks["citations_resolve"] is False
    assert result.checks["paper_grounded"] is False


def test_paper_chat_model_enforces_generation_budget_and_keep_alive(tmp_path, monkeypatch):
    updates = {}

    class FakeModel:
        def model_copy(self, *, update):
            updates.update(update)
            return self

    monkeypatch.setattr(
        "paper_agent.paper_chat.build_direct_chat_model", lambda config: FakeModel()
    )
    monkeypatch.setenv("PAPER_CHAT_MAX_TOKENS", "5000")
    monkeypatch.setenv("PAPER_READER_AGENT_KEEP_ALIVE", "20m")
    config = RunConfig(
        pdf_path=tmp_path / "paper.pdf",
        output_dir=tmp_path / "workspace",
        model_provider="ollama",
        model="test-model",
    )

    _paper_chat_model(config)

    assert updates == {"keep_alive": "20m", "num_predict": 1600}


def test_graph_persists_a_chat_turn_without_external_models(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")

    def fake_specialist(parsed, workspace, specialist, question, history, evidence, memory):
        passage = next(item for item in evidence if item.source == "paper" and item.kind == "page")
        citation = passage.id
        claim = passage.text.splitlines()[-1].split(".")[0]
        return SpecialistOutput(
            specialist=specialist,
            answer=f"{claim} [{citation}].",
            cited_evidence_ids=[citation],
            claims=["The paper uses calibrated softmax."],
            model="fake-model",
        )

    monkeypatch.setattr("paper_agent.paper_chat._invoke_specialist", fake_specialist)

    def no_judge(self, claims):
        raise AssertionError("Verbatim evidence should not need a model call")

    monkeypatch.setattr("paper_agent.paper_chat.PaperChatWorkflow._judge_claims", no_judge)
    result = run_paper_chat(
        sample_paper(),
        workspace,
        "What is the main objective?",
        thread_id="thread-one",
    )

    thread = load_chat_thread(workspace, "thread-one")
    assert result.verification.passed is True
    assert len(result.citations) == 1
    assert [item["role"] for item in thread["messages"]] == ["user", "assistant"]


def test_reader_chat_job_reports_progress_and_completes(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")

    class FakeResult:
        def model_dump(self):
            return {"answer": "done", "thread_id": "job-thread"}

    def fake_run(parsed, workspace, question, mode, thread_id, progress):
        progress("retrieval", "Retrieved paper evidence.")
        return FakeResult()

    monkeypatch.setattr(reader, "run_paper_chat", fake_run)
    with reader.CHAT_LOCK:
        reader.CHAT_JOBS.clear()

    started = reader.start_chat_job(workspace, sample_paper(), "Question", "fast", "job-thread")
    deadline = time.monotonic() + 2
    job = started
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        job = reader.get_chat_job(str(started["jobId"]))

    assert job["status"] == "completed"
    assert job["result"]["answer"] == "done"
