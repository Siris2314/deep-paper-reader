from __future__ import annotations

import json
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_agent.workspace import Workspace


def safe_serialize(obj: Any) -> Any:
    """Best-effort JSON serialization for LangChain/DeepAgents objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if is_dataclass(obj):
        return safe_serialize(asdict(obj))
    if hasattr(obj, "model_dump"):
        try:
            return safe_serialize(obj.model_dump())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(x) for x in obj]
    # LangChain messages normally have type/content/tool_calls/metadata attributes.
    attrs: dict[str, Any] = {}
    for name in [
        "type",
        "name",
        "content",
        "tool_calls",
        "invalid_tool_calls",
        "response_metadata",
        "additional_kwargs",
        "id",
    ]:
        if hasattr(obj, name):
            try:
                attrs[name] = safe_serialize(getattr(obj, name))
            except Exception:
                attrs[name] = "<unserializable>"
    if attrs:
        attrs["__class__"] = obj.__class__.__name__
        return attrs
    return repr(obj)


def write_debug_json(workspace: Workspace, rel_path: str, payload: Any) -> Path:
    return workspace.write_text(
        rel_path, json.dumps(safe_serialize(payload), indent=2, ensure_ascii=False)
    )


def write_run_config(workspace: Workspace, config: Any) -> Path:
    payload = safe_serialize(config)
    payload["timestamp_utc"] = (
        datetime.now(timezone.utc).isoformat()
        if isinstance(payload, dict)
        else datetime.now(timezone.utc).isoformat()
    )
    return write_debug_json(workspace, "logs/run_config.json", payload)


def write_exception(
    workspace: Workspace, exc: BaseException, rel_path: str = "logs/last_error.txt"
) -> Path:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return workspace.write_text(rel_path, text)


def write_agent_messages_markdown(workspace: Workspace, result: Any) -> Path:
    serialized = safe_serialize(result)
    messages = []
    if isinstance(serialized, dict):
        maybe_messages = serialized.get("messages")
        if isinstance(maybe_messages, list):
            messages = maybe_messages

    lines = ["# Agent Messages", ""]
    if not messages:
        lines.extend(
            [
                "No messages found in result payload.",
                "",
                "```json",
                json.dumps(serialized, indent=2, ensure_ascii=False)[:20000],
                "```",
            ]
        )
    else:
        for i, msg in enumerate(messages, start=1):
            role = msg.get("type") or msg.get("__class__") or "message"
            name = msg.get("name") or ""
            lines.append(f"## {i}. {role} {name}".strip())
            lines.append("")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, indent=2, ensure_ascii=False)
            lines.append(str(content) if content else "[no content]")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                lines.append("")
                lines.append("Tool calls:")
                lines.append("```json")
                lines.append(json.dumps(tool_calls, indent=2, ensure_ascii=False))
                lines.append("```")
            metadata = msg.get("response_metadata")
            if metadata:
                lines.append("")
                lines.append("Response metadata:")
                lines.append("```json")
                lines.append(json.dumps(metadata, indent=2, ensure_ascii=False)[:8000])
                lines.append("```")
            lines.append("")
    return workspace.write_text("logs/agent_messages.md", "\n".join(lines))
