from paper_agent import context_diagnostics as diagnostics
from paper_agent.workspace import Workspace


def test_native_context_is_read_from_ollama_model_info():
    payload = {
        "model_info": {
            "general.architecture": "qwen3",
            "qwen3.context_length": 262144,
        }
    }

    assert diagnostics._native_context_from_show(payload) == 262144


def test_context_usage_is_recorded_and_exposed(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("PAPER_READER_MATH_MODEL", "qwen3.5:4b")
    monkeypatch.setattr(diagnostics, "probe_ollama_native_context", lambda model, base_url: 262144)
    monkeypatch.setattr(
        diagnostics,
        "probe_ollama_runtime",
        lambda base_url: {"available": True, "models": [], "totalVramBytes": 0},
    )

    record = diagnostics.record_context_usage(
        workspace,
        workflow="math_explanation",
        provider="ollama",
        model="qwen3.5:4b",
        components={"system prompt": "Explain math carefully.", "paper context": "x " * 800},
        reserved_output_tokens=1200,
        metadata={"page": 3},
    )
    payload = diagnostics.context_diagnostics_payload(workspace)
    math = next(item for item in payload["models"] if item["workflow"] == "math_explanation")

    assert record["estimatedInputTokens"] > 0
    assert record["configuredWindowTokens"] == 8192
    assert math["nativeWindowTokens"] == 262144
    assert math["effectiveWindowTokens"] == 8192
    assert math["latest"]["metadata"]["page"] == 3
    assert math["estimatedEffectiveUsagePercent"] > 0
