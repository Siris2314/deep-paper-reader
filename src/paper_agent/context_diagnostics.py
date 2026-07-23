from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_agent.config import DEFAULT_OLLAMA_MODEL
from paper_agent.workspace import Workspace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONTEXT_LOG = "logs/context_usage.json"
_LOG_LOCK = threading.RLock()
_PROBE_LOCK = threading.RLock()
_PROBE_CACHE: dict[tuple[str, str], tuple[float, int | None]] = {}
_RUNTIME_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def estimate_tokens(text: str) -> int:
    """Return a model-independent token estimate suitable for diagnostics, not billing."""

    clean = str(text or "")
    if not clean:
        return 0
    # Four characters per token is a useful English/code approximation. Whitespace-delimited
    # terms keep short prompts from being underestimated too aggressively.
    character_estimate = math.ceil(len(clean) / 4)
    word_estimate = math.ceil(len(clean.split()) * 1.25)
    return max(1, character_estimate, word_estimate)


def response_schema_text(schema: type[Any]) -> str:
    try:
        payload = schema.model_json_schema()
    except AttributeError:
        try:
            payload = schema.schema()
        except AttributeError:
            return ""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configured_context_window(provider: str) -> int | None:
    if provider.strip().lower() == "ollama":
        return max(1, int(os.getenv("OLLAMA_NUM_CTX", "8192")))
    value = os.getenv("OPENAI_CONTEXT_WINDOW")
    return max(1, int(value)) if value else None


def _read_records(workspace: Workspace) -> list[dict[str, Any]]:
    path = workspace.path(_CONTEXT_LOG)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return [item for item in records if isinstance(item, dict)]


def record_context_usage(
    workspace: Workspace,
    *,
    workflow: str,
    provider: str,
    model: str,
    components: dict[str, str],
    reserved_output_tokens: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    component_usage = [
        {
            "name": name,
            "characters": len(str(content or "")),
            "estimatedTokens": estimate_tokens(str(content or "")),
        }
        for name, content in components.items()
        if str(content or "")
    ]
    input_tokens = sum(int(item["estimatedTokens"]) for item in component_usage)
    configured_window = configured_context_window(provider)
    planned_tokens = input_tokens + max(0, int(reserved_output_tokens))
    record: dict[str, Any] = {
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "provider": provider,
        "model": model,
        "configuredWindowTokens": configured_window,
        "estimatedInputTokens": input_tokens,
        "reservedOutputTokens": max(0, int(reserved_output_tokens)),
        "estimatedPlannedTokens": planned_tokens,
        "estimatedConfiguredUsagePercent": (
            round(planned_tokens / configured_window * 100, 1) if configured_window else None
        ),
        "components": component_usage,
        "metadata": metadata or {},
        "estimateNote": "Visible prompt and response-schema estimate; framework and image-token overhead are excluded.",
    }
    with _LOG_LOCK:
        records = _read_records(workspace)
        records.append(record)
        workspace.write_json(_CONTEXT_LOG, {"version": 1, "records": records[-120:]})
    return record


def _native_context_from_show(payload: dict[str, Any]) -> int | None:
    model_info = payload.get("model_info")
    if not isinstance(model_info, dict):
        return None
    values: list[int] = []
    for key, value in model_info.items():
        if not str(key).lower().endswith("context_length"):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            values.append(parsed)
    return max(values) if values else None


def probe_ollama_native_context(model: str, base_url: str) -> int | None:
    if os.getenv("CONTEXT_DEBUG_PROBE_OLLAMA", "true").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    key = (base_url.rstrip("/"), model)
    now = time.monotonic()
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
        if cached and now - cached[0] < 60:
            return cached[1]

    request = urllib.request.Request(
        f"{key[0]}/api/show",
        data=json.dumps({"model": model}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "deep-paper-agent/0.1"},
        method="POST",
    )
    try:
        timeout = max(0.2, min(float(os.getenv("CONTEXT_DEBUG_PROBE_TIMEOUT", "1.0")), 5.0))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        native_context = _native_context_from_show(payload)
    except Exception:
        native_context = None
    with _PROBE_LOCK:
        _PROBE_CACHE[key] = (now, native_context)
    return native_context


def probe_ollama_runtime(base_url: str) -> dict[str, Any]:
    if os.getenv("CONTEXT_DEBUG_PROBE_OLLAMA", "true").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return {"available": False, "models": [], "totalVramBytes": 0}
    normalized_url = base_url.rstrip("/")
    now = time.monotonic()
    with _PROBE_LOCK:
        cached = _RUNTIME_CACHE.get(normalized_url)
        if cached and now - cached[0] < 5:
            return cached[1]
    request = urllib.request.Request(
        f"{normalized_url}/api/ps",
        headers={"Accept": "application/json", "User-Agent": "deep-paper-agent/0.1"},
    )
    try:
        timeout = max(0.2, min(float(os.getenv("CONTEXT_DEBUG_PROBE_TIMEOUT", "1.0")), 5.0))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = []
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            models.append(
                {
                    "model": str(item.get("model") or item.get("name") or "unknown"),
                    "sizeVramBytes": int(item.get("size_vram") or 0),
                    "contextLength": int(item.get("context_length") or 0),
                    "expiresAt": str(item.get("expires_at") or ""),
                }
            )
        result = {
            "available": True,
            "models": models,
            "totalVramBytes": sum(int(item["sizeVramBytes"]) for item in models),
        }
    except Exception:
        result = {"available": False, "models": [], "totalVramBytes": 0}
    with _PROBE_LOCK:
        _RUNTIME_CACHE[normalized_url] = (now, result)
    return result


def configured_workflows() -> list[dict[str, Any]]:
    default_provider = os.getenv("MODEL_PROVIDER", "ollama")
    reader_provider = os.getenv("PAPER_READER_AGENT_PROVIDER", default_provider)
    default_model = os.getenv("MODEL_NAME", os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    return [
        {
            "workflow": "concept_relevance",
            "label": "Concept agent",
            "provider": reader_provider,
            "model": os.getenv("PAPER_READER_AGENT_MODEL", default_model),
            "reservedOutputTokens": int(os.getenv("PAPER_READER_AGENT_MAX_TOKENS", "220")),
        },
        {
            "workflow": "math_explanation",
            "label": "Math agent",
            "provider": reader_provider,
            "model": os.getenv(
                "PAPER_READER_MATH_MODEL", os.getenv("PAPER_READER_AGENT_MODEL", "qwen3.5:4b")
            ),
            "reservedOutputTokens": int(os.getenv("PAPER_READER_MATH_MAX_TOKENS", "1200")),
        },
        {
            "workflow": "math_judge",
            "label": "Math judge",
            "provider": os.getenv("PAPER_READER_JUDGE_PROVIDER", reader_provider),
            "model": os.getenv("PAPER_READER_JUDGE_MODEL", "qwen2.5:7b"),
            "reservedOutputTokens": int(os.getenv("PAPER_READER_JUDGE_MAX_TOKENS", "700")),
        },
        {
            "workflow": "paper_chat",
            "workflowAliases": [
                "paper_chat_general",
                "paper_chat_implementation",
                "paper_chat_experiments",
                "paper_chat_concept",
                "paper_chat_synthesis",
            ],
            "label": "Paper chat",
            "provider": reader_provider,
            "model": os.getenv("PAPER_READER_AGENT_MODEL", default_model),
            "reservedOutputTokens": int(os.getenv("PAPER_CHAT_MAX_TOKENS", "1400")),
        },
        {
            "workflow": "paper_chat_math",
            "label": "Paper chat math",
            "provider": reader_provider,
            "model": os.getenv(
                "PAPER_READER_MATH_MODEL", os.getenv("PAPER_READER_AGENT_MODEL", "qwen3.5:4b")
            ),
            "reservedOutputTokens": int(os.getenv("PAPER_CHAT_MAX_TOKENS", "1400")),
        },
    ]


def _hardware_profile_summary() -> dict[str, Any] | None:
    path = PROJECT_ROOT / "agent_memory" / "hardware_profile.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    profile = payload.get("profile") if isinstance(payload, dict) else None
    hardware = payload.get("hardware") if isinstance(payload, dict) else None
    if not isinstance(profile, dict) or not isinstance(hardware, dict):
        return None
    return {
        "name": profile.get("name"),
        "contextLength": profile.get("context_length"),
        "maxLoadedModels": profile.get("max_loaded_models"),
        "gpu": (hardware.get("gpus") or [None])[0],
        "systemRamGb": hardware.get("system_ram_gb"),
        "configuredAt": payload.get("configuredAt"),
        "missingModels": payload.get("missingModels", []),
    }


def context_diagnostics_payload(workspace: Workspace | None = None) -> dict[str, Any]:
    records = _read_records(workspace) if workspace else []
    latest_by_workflow: dict[str, dict[str, Any]] = {}
    for record in records:
        latest_by_workflow[str(record.get("workflow") or "")] = record

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    specs = configured_workflows()
    probe_models = {
        str(spec["model"]) for spec in specs if str(spec["provider"]).lower() == "ollama"
    }
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(probe_models)))) as pool:
        native_cache = dict(
            zip(
                probe_models,
                pool.map(lambda model: probe_ollama_native_context(model, base_url), probe_models),
            )
        )
    models: list[dict[str, Any]] = []
    for spec in specs:
        aliases = [str(spec["workflow"]), *[str(item) for item in spec.get("workflowAliases", [])]]
        latest = next(
            (
                record
                for record in reversed(records)
                if str(record.get("workflow") or "") in aliases
            ),
            None,
        )
        model = str(spec["model"])
        provider = str(spec["provider"])
        latest_model = str(latest.get("model")) if latest else None
        configured = configured_context_window(provider)
        native: int | None = None
        if provider.lower() == "ollama":
            native = native_cache[model]
        effective = min(configured, native) if configured and native else configured or native
        planned = int(latest.get("estimatedPlannedTokens") or 0) if latest else 0
        models.append(
            {
                **spec,
                "provider": provider,
                "model": model,
                "latestModel": latest_model,
                "configuredWindowTokens": configured,
                "nativeWindowTokens": native,
                "effectiveWindowTokens": effective,
                "latest": latest,
                "estimatedEffectiveUsagePercent": (
                    round(planned / effective * 100, 1) if planned and effective else None
                ),
            }
        )
    return {
        "workspace": str(workspace.root) if workspace else None,
        "models": models,
        "ollamaRuntime": probe_ollama_runtime(base_url),
        "hardwareProfile": _hardware_profile_summary(),
        "recordCount": len(records),
        "estimateNote": (
            "Token counts estimate visible prompts and response schemas. Framework overhead and image tokens "
            "are not included; Ollama native capacity is probed from /api/show when available."
        ),
    }
