from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    run: str
    load_seconds: float
    prompt_tokens: int
    prompt_tokens_per_second: float
    output_tokens: int
    output_tokens_per_second: float
    total_seconds: float


def _post(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "deep-paper-agent-benchmark/0.1",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned a non-object benchmark response.")
    return result


def _seconds(nanoseconds: object) -> float:
    try:
        return max(0.0, float(nanoseconds) / 1_000_000_000)
    except (TypeError, ValueError):
        return 0.0


def _rate(count: int, duration_seconds: float) -> float:
    return round(count / duration_seconds, 2) if count > 0 and duration_seconds > 0 else 0.0


def _result(model: str, run: str, payload: dict[str, Any]) -> BenchmarkResult:
    prompt_count = int(payload.get("prompt_eval_count") or 0)
    output_count = int(payload.get("eval_count") or 0)
    prompt_seconds = _seconds(payload.get("prompt_eval_duration"))
    output_seconds = _seconds(payload.get("eval_duration"))
    return BenchmarkResult(
        model=model,
        run=run,
        load_seconds=round(_seconds(payload.get("load_duration")), 3),
        prompt_tokens=prompt_count,
        prompt_tokens_per_second=_rate(prompt_count, prompt_seconds),
        output_tokens=output_count,
        output_tokens_per_second=_rate(output_count, output_seconds),
        total_seconds=round(_seconds(payload.get("total_duration")), 3),
    )


def benchmark_model(args: argparse.Namespace, model: str) -> list[BenchmarkResult]:
    if args.cold:
        _post(
            args.base_url,
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            args.timeout,
        )
    request = {
        "model": model,
        "prompt": args.prompt,
        "stream": False,
        "think": False,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": 0,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "seed": 42,
        },
    }
    first = _post(args.base_url, request, args.timeout)
    second = _post(args.base_url, request, args.timeout)
    return [_result(model, "first", first), _result(model, "warm", second)]


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'MODEL':<18} {'RUN':<7} {'LOAD_S':>8} {'PROMPT_T/S':>11} "
        f"{'OUTPUT_T/S':>11} {'OUT_TOK':>8} {'TOTAL_S':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.model:<18} {item.run:<7} {item.load_seconds:>8.3f} "
            f"{item.prompt_tokens_per_second:>11.2f} {item.output_tokens_per_second:>11.2f} "
            f"{item.output_tokens:>8} {item.total_seconds:>8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cold/warm Ollama inference for reader models."
    )
    parser.add_argument("models", nargs="*", default=["qwen3.5:4b", "qwen2.5:7b"])
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--cold", action="store_true", help="Unload each model before its first run."
    )
    parser.add_argument(
        "--prompt",
        default=(
            "In four concise sentences, explain why a quantized KV cache can reduce memory use "
            "during transformer inference while preserving most answer quality."
        ),
    )
    args = parser.parse_args()
    results: list[BenchmarkResult] = []
    for model in args.models:
        try:
            results.extend(benchmark_model(args, model))
        except Exception as exc:
            print(f"{model}: benchmark failed: {exc}")
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
