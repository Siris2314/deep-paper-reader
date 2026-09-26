from __future__ import annotations

import hashlib
import os
import platform
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal


ObservationType = Literal[
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
    "generation",
]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def capture_content() -> bool:
    return _truthy(os.getenv("LANGFUSE_CAPTURE_CONTENT", "false"))


def _sdk_installed() -> bool:
    try:
        import langfuse  # noqa: F401

        return True
    except ImportError:
        return False


def observability_status() -> dict[str, object]:
    requested = _truthy(os.getenv("LANGFUSE_ENABLED", "false"))
    configured = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    )
    installed = _sdk_installed()
    return {
        "backend": "langfuse",
        "enabled": requested and configured and installed,
        "requested": requested,
        "configured": configured,
        "sdkInstalled": installed,
        "captureContent": capture_content(),
        "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
        "baseUrl": os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        "release": release_version(),
    }


def paper_session_id(workspace: object, thread_id: str | None = None) -> str:
    root = getattr(workspace, "root", workspace)
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
    suffix = ""
    if thread_id:
        clean = "".join(character for character in thread_id if ord(character) < 128)
        clean = clean[:80].strip()
        suffix = f"-{clean}" if clean else ""
    return f"paper-{digest}{suffix}"


def _text_summary(value: str) -> dict[str, object]:
    encoded = value.encode("utf-8")
    return {
        "type": "text",
        "characters": len(value),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest()[:20],
    }


def _serializable(value: Any, *, include_content: bool, depth: int = 0) -> Any:
    if depth > 6:
        return {"type": type(value).__name__, "truncated": True}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:20_000] if include_content else _text_summary(value)
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest()[:20],
        }
    if isinstance(value, Path):
        return str(value) if include_content else _text_summary(str(value))
    if is_dataclass(value) and not isinstance(value, type):
        return _serializable(asdict(value), include_content=include_content, depth=depth + 1)
    if hasattr(value, "model_dump"):
        return _serializable(value.model_dump(), include_content=include_content, depth=depth + 1)
    if isinstance(value, dict):
        return {
            str(key)[:120]: _serializable(item, include_content=include_content, depth=depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return {
            "count": len(items),
            "items": [
                _serializable(item, include_content=include_content, depth=depth + 1)
                for item in items[:30]
            ],
            "truncated": len(items) > 30,
        }
    return _serializable(str(value), include_content=include_content, depth=depth + 1)


def trace_payload(value: Any) -> Any:
    return _serializable(value, include_content=capture_content())


def _score_comment(comment: str | None) -> str | None:
    if not comment:
        return None
    if capture_content():
        return comment[:500]
    summary = _text_summary(comment)
    return f"redacted text: characters={summary['characters']}, sha256={summary['sha256']}"


@lru_cache(maxsize=1)
def _source_release() -> str:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        *sorted((root / "src" / "paper_agent").glob("*.py")),
        root / "ui" / "paper_reader_server.py",
        root / "llmops" / "gate.json",
    ]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return f"dev-{digest.hexdigest()[:12]}"


def release_version() -> str:
    explicit = (
        os.getenv("PAPER_READER_RELEASE", "").strip()
        or os.getenv("LANGFUSE_RELEASE", "").strip()
        or os.getenv("GITHUB_SHA", "").strip()
    )
    return (explicit or _source_release())[:120]


def _base_metadata(metadata: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "application": "deep-paper-reader",
        "release": release_version(),
        "python": platform.python_version(),
        "platform": platform.system(),
        **(metadata or {}),
    }


def _runtime() -> tuple[Any, Any] | None:
    if os.getenv("PYTEST_CURRENT_TEST") and not _truthy(
        os.getenv("LANGFUSE_ENABLE_IN_TESTS", "false")
    ):
        return None
    if not observability_status()["enabled"]:
        return None
    try:
        from langfuse import get_client, propagate_attributes

        return get_client(), propagate_attributes
    except Exception:
        return None


@dataclass
class TraceHandle:
    span: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.span is not None

    @property
    def trace_id(self) -> str | None:
        value = getattr(self.span, "trace_id", None)
        return str(value) if value else None

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, object] | None = None,
        status_message: str | None = None,
        level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
    ) -> None:
        if self.span is None:
            return
        values: dict[str, object] = {}
        if output is not None:
            values["output"] = trace_payload(output)
        if metadata is not None:
            values["metadata"] = _base_metadata(metadata)
        if status_message:
            values["status_message"] = status_message[:500]
        if level:
            values["level"] = level
        try:
            self.span.update(**values)
        except Exception:
            return

    def score(
        self,
        name: str,
        value: float | str,
        *,
        data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT"] = "NUMERIC",
        comment: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.span is None:
            return
        try:
            self.span.score_trace(
                name=name,
                value=value,
                data_type=data_type,
                comment=_score_comment(comment),
                metadata={
                    "release": release_version(),
                    "details": trace_payload(metadata) if metadata else {},
                },
            )
        except Exception:
            return


def score_session(
    session_id: str,
    name: str,
    value: float | str,
    *,
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT"] = "NUMERIC",
    comment: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    runtime = _runtime()
    if runtime is None:
        return
    client, _ = runtime
    try:
        client.create_score(
            name=name,
            value=value,
            session_id=session_id,
            data_type=data_type,
            comment=_score_comment(comment),
            metadata={
                "release": release_version(),
                "details": trace_payload(metadata) if metadata else {},
            },
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
        )
    except Exception:
        return


def flush_observability() -> None:
    runtime = _runtime()
    if runtime is None:
        return
    client, _ = runtime
    try:
        client.flush()
    except Exception:
        return


@contextmanager
def workflow_trace(
    name: str,
    *,
    input_data: Any | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    as_type: ObservationType = "agent",
) -> Iterator[TraceHandle]:
    runtime = _runtime()
    if runtime is None:
        yield TraceHandle()
        return

    client, propagate_attributes = runtime
    stack = ExitStack()
    try:
        stack.enter_context(
            propagate_attributes(
                session_id=session_id,
                tags=tags or [],
                version=release_version(),
                environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
                metadata=_base_metadata(metadata),
                trace_name=name,
            )
        )
        span = stack.enter_context(
            client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=trace_payload(input_data) if input_data is not None else None,
                metadata=_base_metadata(metadata),
                version=release_version(),
            )
        )
    except Exception:
        stack.close()
        yield TraceHandle()
        return

    handle = TraceHandle(span)
    try:
        yield handle
    except Exception as exc:
        handle.update(
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}",
            metadata={"errorType": type(exc).__name__},
        )
        raise
    finally:
        try:
            stack.close()
        except Exception:
            pass


@contextmanager
def observed_stage(
    name: str,
    *,
    input_data: Any | None = None,
    metadata: dict[str, object] | None = None,
    as_type: ObservationType = "span",
) -> Iterator[TraceHandle]:
    """Create a child observation inside the active workflow trace."""
    runtime = _runtime()
    if runtime is None:
        yield TraceHandle()
        return

    client, _ = runtime
    stack = ExitStack()
    try:
        span = stack.enter_context(
            client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=trace_payload(input_data) if input_data is not None else None,
                metadata=_base_metadata(metadata),
                version=release_version(),
            )
        )
    except Exception:
        stack.close()
        yield TraceHandle()
        return

    handle = TraceHandle(span)
    try:
        yield handle
    except Exception as exc:
        handle.update(
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}",
            metadata={"errorType": type(exc).__name__},
        )
        raise
    finally:
        try:
            stack.close()
        except Exception:
            pass


def _usage_details(result: Any) -> dict[str, int] | None:
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        mapping = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        return {key: int(value) for key, value in mapping.items() if value is not None}
    response = getattr(result, "response_metadata", None)
    if isinstance(response, dict):
        prompt = response.get("prompt_eval_count")
        completion = response.get("eval_count")
        details: dict[str, int] = {}
        if prompt is not None:
            details["input_tokens"] = int(prompt)
        if completion is not None:
            details["output_tokens"] = int(completion)
        if prompt is not None or completion is not None:
            details["total_tokens"] = int(prompt or 0) + int(completion or 0)
        return details or None
    return None


def invoke_observed(
    runnable: Any,
    payload: Any,
    *,
    name: str,
    model: str | None = None,
    metadata: dict[str, object] | None = None,
    as_type: ObservationType = "generation",
) -> Any:
    runtime = _runtime()
    if runtime is None:
        return runnable.invoke(payload)

    if capture_content():
        try:
            from langfuse.langchain import CallbackHandler

            handler = CallbackHandler()
        except Exception:
            handler = None
        if handler is not None:
            return runnable.invoke(
                payload,
                config={
                    "callbacks": [handler],
                    "run_name": name,
                    "metadata": _base_metadata(metadata),
                },
            )

    client, _ = runtime
    stack = ExitStack()
    try:
        span = stack.enter_context(
            client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=trace_payload(payload),
                metadata=_base_metadata(metadata),
                model=model,
                version=release_version(),
            )
        )
    except Exception:
        stack.close()
        return runnable.invoke(payload)

    try:
        result = runnable.invoke(payload)
    except Exception as exc:
        try:
            span.update(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}"[:500],
            )
        except Exception:
            pass
        try:
            stack.close()
        except Exception:
            pass
        raise

    try:
        update: dict[str, object] = {"output": trace_payload(result)}
        usage = _usage_details(result)
        if usage:
            update["usage_details"] = usage
        span.update(**update)
    except Exception:
        pass
    try:
        stack.close()
    except Exception:
        pass
    return result
