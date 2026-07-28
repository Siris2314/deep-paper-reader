import json
from contextlib import nullcontext
from types import SimpleNamespace

import paper_agent.observability as observability


class FakeSpan:
    def __init__(self):
        self.trace_id = "trace-123"
        self.updates = []
        self.scores = []

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def score_trace(self, **kwargs):
        self.scores.append(kwargs)


class FakeClient:
    def __init__(self, *, fail_start=False):
        self.fail_start = fail_start
        self.observations = []
        self.session_scores = []
        self.flush_count = 0

    def start_as_current_observation(self, **kwargs):
        if self.fail_start:
            raise RuntimeError("telemetry unavailable")
        span = FakeSpan()
        self.observations.append((kwargs, span))
        return nullcontext(span)

    def create_score(self, **kwargs):
        self.session_scores.append(kwargs)

    def flush(self):
        self.flush_count += 1


def fake_propagate(**kwargs):
    return nullcontext(kwargs)


def test_observability_requires_opt_in_sdk_and_both_keys(monkeypatch):
    monkeypatch.setattr(observability, "_sdk_installed", lambda: True)
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert observability.observability_status()["enabled"] is False

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")

    assert observability.observability_status()["enabled"] is True


def test_default_trace_payload_hashes_content(monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "false")

    payload = observability.trace_payload(
        {"prompt": "private paper excerpt", "messages": ["private answer"]}
    )
    serialized = json.dumps(payload)

    assert "private paper excerpt" not in serialized
    assert "private answer" not in serialized
    assert payload["prompt"]["characters"] == 21
    assert payload["prompt"]["sha256"]


def test_runtime_suppresses_real_telemetry_during_pytest(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_observability.py::test")
    monkeypatch.setenv("LANGFUSE_ENABLE_IN_TESTS", "false")
    monkeypatch.setattr(
        observability,
        "observability_status",
        lambda: {"enabled": True},
    )

    assert observability._runtime() is None


def test_workflow_trace_updates_and_scores_fake_span(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(observability, "_runtime", lambda: (client, fake_propagate))
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "false")
    monkeypatch.setenv("PAPER_READER_RELEASE", "test-release")

    with observability.workflow_trace(
        "math-explanation",
        input_data={"equation": "secret equation"},
        session_id="paper-1",
        tags=["math"],
    ) as trace:
        trace.update(output={"answer": "secret answer"}, metadata={"status": "pass"})
        trace.score(
            "math_passed",
            1.0,
            data_type="BOOLEAN",
            comment="private judge explanation",
        )

    assert trace.trace_id == "trace-123"
    assert len(client.observations) == 1
    observation, span = client.observations[0]
    assert observation["name"] == "math-explanation"
    assert "secret equation" not in json.dumps(observation["input"])
    assert "secret answer" not in json.dumps(span.updates)
    assert span.scores[0]["name"] == "math_passed"
    assert span.scores[0]["metadata"]["release"] == "test-release"
    assert "private judge explanation" not in json.dumps(span.scores)


def test_observed_stage_records_a_named_child_observation(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(observability, "_runtime", lambda: (client, fake_propagate))

    with observability.observed_stage(
        "parse-equation",
        input_data={"equation": "private equation"},
        metadata={"page": 4},
        as_type="tool",
    ) as stage:
        stage.update(output={"status": "parsed"}, metadata={"symbolCount": 3})

    observation, span = client.observations[0]
    assert observation["name"] == "parse-equation"
    assert observation["as_type"] == "tool"
    assert observation["metadata"]["page"] == 4
    assert "private equation" not in json.dumps(observation["input"])
    assert span.updates[0]["metadata"]["symbolCount"] == 3


def test_observed_invoke_calls_model_once_and_records_usage(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(observability, "_runtime", lambda: (client, fake_propagate))
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "false")

    class Runnable:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return SimpleNamespace(
                content="private response",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
            )

    runnable = Runnable()
    result = observability.invoke_observed(
        runnable,
        "private prompt",
        name="generation",
        model="ollama:test",
    )

    assert result.content == "private response"
    assert runnable.calls == 1
    assert len(client.observations) == 1
    _, span = client.observations[0]
    assert span.updates[0]["usage_details"]["total_tokens"] == 14
    assert "private response" not in json.dumps(span.updates, default=str)


def test_telemetry_setup_failure_does_not_retry_or_block_model(monkeypatch):
    client = FakeClient(fail_start=True)
    monkeypatch.setattr(observability, "_runtime", lambda: (client, fake_propagate))
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "false")

    class Runnable:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return "answer"

    runnable = Runnable()

    assert observability.invoke_observed(runnable, "prompt", name="generation") == "answer"
    assert runnable.calls == 1


def test_session_feedback_score_and_flush_are_optional(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(observability, "_runtime", lambda: (client, fake_propagate))

    observability.score_session(
        "paper-thread",
        "user_helpful",
        0.0,
        data_type="BOOLEAN",
        comment="Needs correction.",
    )
    observability.flush_observability()

    assert client.session_scores[0]["session_id"] == "paper-thread"
    assert client.session_scores[0]["name"] == "user_helpful"
    assert client.flush_count == 1
