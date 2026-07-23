from scripts.benchmark_ollama import _result


def test_benchmark_result_converts_ollama_nanosecond_metrics():
    result = _result(
        "qwen3.5:4b",
        "warm",
        {
            "load_duration": 250_000_000,
            "prompt_eval_count": 100,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 50,
            "eval_duration": 2_000_000_000,
            "total_duration": 3_000_000_000,
        },
    )

    assert result.load_seconds == 0.25
    assert result.prompt_tokens_per_second == 200.0
    assert result.output_tokens_per_second == 25.0
    assert result.total_seconds == 3.0
