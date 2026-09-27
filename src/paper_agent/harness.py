"""Shared budgets for synchronous paper workflows and their model/tool calls.

Deadlines are checked before dispatch; transport timeouts bound in-flight requests.
No automatic retries: callers already own fallback and semantic repair policies.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph.state import CompiledStateGraph

from paper_agent.context_diagnostics import estimate_tokens
from paper_agent.workspace import Workspace


class HarnessLimitExceeded(RuntimeError):
    """The run cannot dispatch more work within its configured budget."""


def bounded_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


@dataclass(frozen=True)
class HarnessPolicy:
    model_calls: int = 8
    tool_calls: int = 24
    input_tokens: int = 60_000
    seconds: int = 300
    repeated_tool_calls: int = 3
    tool_result_chars: int = 8000

    @classmethod
    def for_workflow(cls, workflow: str) -> HarnessPolicy:
        defaults = {
            "chat": (8, 24, 60_000, 300),
            "math": (12, 24, 100_000, 480),
            "concept": (4, 12, 30_000, 180),
            "report": (64, 120, 600_000, 1800),
        }
        calls, tools, tokens, seconds = defaults[workflow]
        prefix = f"PAPER_HARNESS_{workflow.upper()}"
        return cls(
            model_calls=bounded_env(f"{prefix}_MODEL_CALLS", calls, 1, 256),
            tool_calls=bounded_env(f"{prefix}_TOOL_CALLS", tools, 1, 512),
            input_tokens=bounded_env(f"{prefix}_INPUT_TOKENS", tokens, 1000, 2_000_000),
            seconds=bounded_env(f"{prefix}_SECONDS", seconds, 1, 7200),
        )


def _serialized(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


@dataclass
class HarnessRun:
    workspace: Workspace
    workflow: str
    policy: HarnessPolicy
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    tool_calls: int = 0
    estimated_input_tokens: int = 0
    stop_reason: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    _last_tool: str = ""
    _repeats: int = 0
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    def reject(self, reason: str) -> None:
        self.stop_reason = reason
        raise HarnessLimitExceeded(reason)

    def check_deadline(self) -> None:
        if self.stop_reason:
            raise HarnessLimitExceeded(self.stop_reason)
        if time.monotonic() - self.started >= self.policy.seconds:
            self.reject("Workflow deadline reached before dispatch.")

    def reserve(self, kind: str, name: str, payload: Any) -> dict[str, Any]:
        serialized = _serialized(payload)
        tokens = estimate_tokens(serialized) if kind == "model" else 0
        with self._lock:
            self.check_deadline()
            if kind == "model":
                if self.model_calls >= self.policy.model_calls:
                    self.reject("Workflow model-call budget exhausted.")
                if self.estimated_input_tokens + tokens > self.policy.input_tokens:
                    self.reject("Workflow estimated input-token budget exhausted.")
                self.model_calls += 1
                self.estimated_input_tokens += tokens
            else:
                if self.tool_calls >= self.policy.tool_calls:
                    self.reject("Workflow tool-call budget exhausted.")
                fingerprint = hashlib.sha256((name + serialized).encode()).hexdigest()
                repeats = self._repeats + 1 if fingerprint == self._last_tool else 1
                if repeats > self.policy.repeated_tool_calls:
                    self.reject("Repeated identical tool calls made no progress.")
                self._last_tool, self._repeats = fingerprint, repeats
                self.tool_calls += 1
            event = {
                "kind": kind,
                "name": name,
                "estimatedInputTokens": tokens,
                "status": "running",
            }
            self.events.append(event)
            return event

    def execute(
        self,
        kind: str,
        name: str,
        payload: Any,
        call: Callable[[], Any],
        *,
        local_model: bool = False,
    ) -> Any:
        event = self.reserve(kind, name, payload)
        queued = time.monotonic()
        gate = _local_gate() if local_model else None
        acquired = False
        started = queued
        try:
            if gate is not None:
                remaining = max(0.0, self.policy.seconds - (queued - self.started))
                acquired = gate.acquire(timeout=remaining)
                if not acquired:
                    self.reject("Workflow deadline reached while waiting for a local model.")
            started = time.monotonic()
            event["queueSeconds"] = round(started - queued, 4)
            self.check_deadline()
            result = call()
            event["status"] = "error" if getattr(result, "status", None) == "error" else "ok"
            # Structured-output parsers may discard token usage. Do not invent it.
            messages = getattr(result, "result", [result])
            usage = [getattr(item, "usage_metadata", None) for item in messages]
            event["usage"] = [item for item in usage if isinstance(item, dict)]
            return result
        except Exception as exc:
            event.update(status="error", errorType=type(exc).__name__)
            raise
        finally:
            event["durationSeconds"] = round(time.monotonic() - started, 4)
            if acquired:
                gate.release()

    def persist(self, status: str) -> None:
        self.workspace.write_json(
            f"logs/harness/{self.run_id}.json",
            {
                "runId": self.run_id,
                "workflow": self.workflow,
                "status": "limited" if self.stop_reason else status,
                "stopReason": self.stop_reason,
                "policy": asdict(self.policy),
                "modelCalls": self.model_calls,
                "toolCalls": self.tool_calls,
                "estimatedInputTokens": self.estimated_input_tokens,
                "elapsedSeconds": round(time.monotonic() - self.started, 4),
                "events": self.events,
            },
        )


_CURRENT: ContextVar[HarnessRun | None] = ContextVar("paper_harness", default=None)
_GATE_LOCK = threading.Lock()
_LOCAL_GATE: threading.BoundedSemaphore | None = None


def _local_gate() -> threading.BoundedSemaphore:
    global _LOCAL_GATE
    with _GATE_LOCK:
        if _LOCAL_GATE is None:
            _LOCAL_GATE = threading.BoundedSemaphore(
                bounded_env("PAPER_HARNESS_LOCAL_CONCURRENCY", 1, 1, 16)
            )
        return _LOCAL_GATE


@contextmanager
def harness_scope(workspace: Workspace, workflow: str, policy: HarnessPolicy | None = None):
    existing = _CURRENT.get()
    if existing is not None:
        yield existing
        return
    run = HarnessRun(workspace, workflow, policy or HarnessPolicy.for_workflow(workflow))
    token = _CURRENT.set(run)
    status = "completed"
    try:
        yield run
    except Exception:
        status = "failed"
        raise
    finally:
        _CURRENT.reset(token)
        try:
            run.persist(status)
        except OSError:
            logging.getLogger(__name__).warning("Could not persist harness metrics", exc_info=True)


def harness_workflow(workflow: str):
    def decorate(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            workspace = arguments.get("workspace")
            if workspace is None:
                workspace = Workspace(arguments["config"].output_dir)
            with harness_scope(workspace, workflow):
                return fn(*args, **kwargs)

        return wrapped

    return decorate


def run_model_call(name: str, model: str | None, payload: Any, call: Callable[[], Any]):
    run = _CURRENT.get()
    if run is None:
        return call()
    return run.execute(
        "model", name, payload, call, local_model=(model or "").startswith("ollama:")
    )


def invoke_with_harness(runnable, payload, call, *, name, model):
    # Graph internals are accounted for by middleware, never by a second outer charge.
    if isinstance(runnable, CompiledStateGraph):
        return call(runnable.with_config(recursion_limit=160), payload)
    return run_model_call(name, model, payload, lambda: call(runnable, payload))


class PaperHarnessMiddleware(AgentMiddleware):
    def __init__(self):
        self.run = _CURRENT.get()

    def wrap_model_call(self, request, handler):
        run = self.run or _CURRENT.get()
        if run is None:
            return handler(request)
        model = request.model
        payload = {
            "system": request.system_message,
            "messages": request.messages,
            "tools": [convert_to_openai_tool(tool) for tool in request.tools],
            "response_format": request.response_format,
        }
        model_name = getattr(model, "model", None) or getattr(
            model, "model_name", type(model).__name__
        )
        return run.execute(
            "model",
            f"agent-model:{model_name}",
            payload,
            lambda: handler(request),
            local_model=type(model).__name__ == "ChatOllama",
        )

    def wrap_tool_call(self, request, handler):
        run = self.run or _CURRENT.get()
        if run is None:
            return handler(request)
        tool_call = request.tool_call
        result = run.execute(
            "tool", tool_call["name"], tool_call.get("args", {}), lambda: handler(request)
        )
        if isinstance(result, ToolMessage) and isinstance(result.content, str):
            limit = run.policy.tool_result_chars
            if len(result.content) > limit:
                path = f"scratch/harness/{run.run_id}-{uuid.uuid4().hex}.txt"
                run.workspace.write_text(path, result.content)
                try:
                    json.loads(result.content)
                except (ValueError, TypeError):
                    content = result.content[:limit] + (
                        f"\n[Truncated. Full result: {path}; retrieve a relevant slice.]"
                    )
                else:
                    content = json.dumps(
                        {
                            "truncated": True,
                            "artifact": path,
                            "preview": result.content[:limit],
                            "instruction": "Retrieve a relevant slice from the full artifact.",
                        }
                    )
                result = result.model_copy(update={"content": content})
        return result


def create_harness_agent(factory, **kwargs):
    """Attach a shared budget to each explicit agent and to report delegation."""
    kwargs["middleware"] = [*kwargs.get("middleware", []), PaperHarnessMiddleware()]
    if "subagents" in kwargs:
        specs = list(kwargs["subagents"])
        if not any(spec["name"] == "general-purpose" for spec in specs):
            specs.append(
                {
                    "name": "general-purpose",
                    "description": "Focused paper-evidence research.",
                    "system_prompt": "Retrieve relevant paper evidence and report supported findings.",
                }
            )
        kwargs["subagents"] = [
            {**spec, "middleware": [*spec.get("middleware", []), PaperHarnessMiddleware()]}
            for spec in specs
        ]
    return factory(**kwargs)
