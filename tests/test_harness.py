import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

import paper_agent.harness as harness
from paper_agent.agents import _artifact_page, _retrieval_slice
from paper_agent.harness import (
    HarnessLimitExceeded,
    HarnessPolicy,
    PaperHarnessMiddleware,
    create_harness_agent,
    harness_scope,
    run_model_call,
)
from paper_agent.observability import invoke_observed
from paper_agent.workspace import Workspace


def test_nested_workflows_share_budget_and_record_failure_without_prompt(tmp_path):
    workspace = Workspace(tmp_path)
    calls = []
    with pytest.raises(HarnessLimitExceeded, match="model-call"):
        with harness_scope(workspace, "math", HarnessPolicy(model_calls=1)) as run:
            run_model_call("explain", "fake", "PRIVATE PROMPT", lambda: calls.append(1))
            with harness_scope(workspace, "math") as judge_run:
                assert judge_run is run
                run_model_call("judge", "fake", "PRIVATE PROMPT", lambda: calls.append(2))
    assert calls == [1]
    paths = list(tmp_path.glob("logs/harness/*.json"))
    assert len(paths) == 1
    text = paths[0].read_text()
    assert "PRIVATE PROMPT" not in text
    report = json.loads(text)
    assert report["status"] == "limited"
    assert report["modelCalls"] == 1


def test_parallel_reservations_cannot_exceed_shared_call_budget(tmp_path):
    with harness_scope(Workspace(tmp_path), "chat", HarnessPolicy(model_calls=1)) as run:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    copy_context().run, run_model_call, "call", "fake", "prompt", lambda: "ok"
                )
                for _ in range(2)
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except HarnessLimitExceeded:
                    outcomes.append("limited")
        assert "limited" in outcomes
        assert outcomes.count("ok") <= 1
        assert run.model_calls == 1


def test_deadline_and_token_limits_prevent_dispatch(tmp_path):
    for policy in (HarnessPolicy(seconds=0), HarnessPolicy(input_tokens=1)):
        with harness_scope(Workspace(tmp_path), "chat", policy):
            with pytest.raises(HarnessLimitExceeded):
                run_model_call(
                    "call",
                    "fake",
                    "long enough to exceed a token",
                    lambda: pytest.fail("Dispatched beyond budget"),
                )


def test_failed_calls_are_counted_and_not_retried(tmp_path):
    calls = []

    def fail():
        calls.append(1)
        raise ValueError("invalid response")

    with harness_scope(Workspace(tmp_path), "chat") as run:
        with pytest.raises(ValueError):
            run_model_call("call", "fake", "prompt", fail)
        assert run.model_calls == 1
        assert run.events[0]["errorType"] == "ValueError"
    assert calls == [1]


def test_local_model_gate_is_shared_across_workflows(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_LOCAL_GATE", threading.BoundedSemaphore(1))
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    peak = 0

    def model():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    def worker(index):
        with harness_scope(Workspace(tmp_path / str(index)), "chat"):
            barrier.wait(timeout=3)
            run_model_call("local", "ollama:test", "prompt", model)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))
    assert peak == 1


def test_middleware_spills_tool_result_and_stops_repeated_calls(tmp_path):
    with harness_scope(
        Workspace(tmp_path), "report", HarnessPolicy(tool_result_chars=40, repeated_tool_calls=2)
    ) as run:
        middleware = PaperHarnessMiddleware()
        request = SimpleNamespace(tool_call={"name": "read", "args": {"file": "same"}})

        def handler(_):
            return ToolMessage(content="evidence " * 100, tool_call_id="call1")

        result = middleware.wrap_tool_call(request, handler)
        assert result.tool_call_id == "call1"
        assert "Truncated" in result.content
        artifact = next(tmp_path.glob("scratch/harness/*.txt"))
        assert artifact.read_text() == "evidence " * 100
        middleware.wrap_tool_call(request, handler)
        with pytest.raises(HarnessLimitExceeded, match="Repeated"):
            middleware.wrap_tool_call(request, handler)
        assert run.tool_calls == 2


def test_spilled_json_remains_json_with_complete_artifact(tmp_path):
    original = {"cards": [{"raw": "equation " * 200}]}
    with harness_scope(Workspace(tmp_path), "report", HarnessPolicy(tool_result_chars=40)):
        middleware = PaperHarnessMiddleware()
        request = SimpleNamespace(tool_call={"name": "get_equation_cards", "args": {}})
        result = middleware.wrap_tool_call(
            request,
            lambda _: ToolMessage(
                content=json.dumps(original),
                tool_call_id="cards",
            ),
        )
        payload = json.loads(result.content)
        assert payload["truncated"] is True
        assert json.loads((tmp_path / payload["artifact"]).read_text()) == original


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_real_agent_graph_accounts_for_each_model_and_tool_once(tmp_path):
    @tool
    def retrieve() -> str:
        """Retrieve paper evidence."""
        return "Evidence from page 2."

    model = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "retrieve", "args": {}, "id": "one"}]),
            AIMessage(content="The answer is grounded in page 2."),
        ]
    )
    with harness_scope(Workspace(tmp_path), "chat", HarnessPolicy(model_calls=2)) as run:
        agent = create_harness_agent(create_agent, model=model, tools=[retrieve])
        result = invoke_observed(
            agent, {"messages": [("user", "Explain this paper")]}, name="test", model="fake"
        )
        assert result["messages"][-1].content == "The answer is grounded in page 2."
        assert run.model_calls == 2
        assert run.tool_calls == 1


def test_report_delegates_have_shared_middleware(tmp_path):
    with harness_scope(Workspace(tmp_path), "report") as run:
        options = create_harness_agent(
            lambda **kwargs: kwargs,
            subagents=[{"name": "math", "description": "Math", "system_prompt": "Explain math."}],
        )
        assert options["middleware"][0].run is run
        assert {spec["name"] for spec in options["subagents"]} == {"math", "general-purpose"}
        assert all(spec["middleware"][0].run is run for spec in options["subagents"])


def test_deep_report_delegation_uses_one_budget(tmp_path):
    from deepagents import create_deep_agent

    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"subagent_type": "math", "description": "Explain x=1"},
                        "id": "delegate",
                    }
                ],
            ),
            AIMessage(content="Report complete."),
        ]
    )
    child = ToolModel(responses=[AIMessage(content="x is defined to be 1.")])
    with harness_scope(Workspace(tmp_path), "report", HarnessPolicy(model_calls=3)) as run:
        agent = create_harness_agent(
            create_deep_agent,
            model=parent,
            tools=[],
            subagents=[
                {
                    "name": "math",
                    "model": child,
                    "description": "Explain math",
                    "system_prompt": "Explain equations.",
                }
            ],
        )
        result = invoke_observed(
            agent, {"messages": [("user", "Explain this equation")]}, name="report", model="fake"
        )
        assert result["messages"][-1].content == "Report complete."
        assert run.model_calls == 3
        assert run.tool_calls == 1


def test_retrieval_slices_keep_tail_accessible_and_artifact_json_valid():
    text = "x" * 7000 + "CRITICAL END"
    assert "offset=6000" in _retrieval_slice(text, 100_000, 0)
    assert "CRITICAL END" in _retrieval_slice(text, 100_000, 6000)
    cards = [SimpleNamespace(model_dump=lambda i=i: {"id": i}) for i in range(16)]
    first = json.loads(_artifact_page(cards, 0, 100))
    second = json.loads(_artifact_page(cards, first["next_offset"], 100))
    assert len(first["cards"]) == 12
    assert second["cards"][-1]["id"] == 15
    assert second["next_offset"] is None
