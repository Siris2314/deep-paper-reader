from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from paper_agent.observability import observability_status, release_version


@dataclass
class MetricGateResult:
    name: str
    status: str
    samples: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    minimum_mean: float | None
    maximum_mean: float | None
    minimum_samples: int
    reason: str


@dataclass
class GateReport:
    passed: bool
    status: str
    environment: str
    release: str | None
    lookback_hours: int
    from_timestamp: str
    evaluated_at: str
    score_count: int
    fetched_score_count: int
    metrics: list[MetricGateResult]
    failures: list[str]


def load_gate_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise ValueError("The LLMOps gate config must contain a metrics object.")
    return payload


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def evaluate_score_gate(
    scores: list[dict[str, object]],
    config: dict[str, Any],
    *,
    allow_insufficient_data: bool = False,
    now: datetime | None = None,
) -> GateReport:
    current = now or datetime.now(timezone.utc)
    lookback_hours = max(1, int(config.get("lookbackHours", 168)))
    environment = str(config.get("environment") or "development")
    configured_release = str(config.get("release") or "").strip() or None
    release = (
        release_version()
        if configured_release and configured_release.casefold() == "current"
        else configured_release
    )
    scoped_scores = [
        item
        for item in scores
        if not release
        or str(
            item.get("release")
            or (
                item.get("metadata", {}).get("release")
                if isinstance(item.get("metadata"), dict)
                else ""
            )
        )
        == release
    ]
    results: list[MetricGateResult] = []
    failures: list[str] = []

    for name, raw_rule in config["metrics"].items():
        rule = raw_rule if isinstance(raw_rule, dict) else {}
        values = [
            numeric
            for item in scoped_scores
            if str(item.get("name") or "") == name
            and (numeric := _numeric(item.get("value"))) is not None
        ]
        minimum_samples = max(1, int(rule.get("minimumSamples", 1)))
        minimum_mean = float(rule["minimumMean"]) if rule.get("minimumMean") is not None else None
        maximum_mean = float(rule["maximumMean"]) if rule.get("maximumMean") is not None else None
        required = bool(rule.get("required", True))
        mean = round(sum(values) / len(values), 4) if values else None
        minimum = min(values) if values else None
        maximum = max(values) if values else None

        if len(values) < minimum_samples:
            status = "insufficient"
            reason = f"{len(values)}/{minimum_samples} required samples"
            if required and not allow_insufficient_data:
                failures.append(f"{name}: {reason}")
        elif minimum_mean is not None and (mean is None or mean < minimum_mean):
            status = "fail"
            reason = f"mean {mean} is below minimum {minimum_mean}"
            failures.append(f"{name}: {reason}")
        elif maximum_mean is not None and (mean is None or mean > maximum_mean):
            status = "fail"
            reason = f"mean {mean} is above maximum {maximum_mean}"
            failures.append(f"{name}: {reason}")
        else:
            status = "pass"
            reason = f"mean {mean} satisfies configured thresholds"

        results.append(
            MetricGateResult(
                name=name,
                status=status,
                samples=len(values),
                mean=mean,
                minimum=minimum,
                maximum=maximum,
                minimum_mean=minimum_mean,
                maximum_mean=maximum_mean,
                minimum_samples=minimum_samples,
                reason=reason,
            )
        )

    if failures:
        status = "fail"
    elif any(item.status == "insufficient" for item in results):
        status = "pass_with_insufficient_data"
    else:
        status = "pass"
    return GateReport(
        passed=not failures,
        status=status,
        environment=environment,
        release=release,
        lookback_hours=lookback_hours,
        from_timestamp=(current - timedelta(hours=lookback_hours)).isoformat(),
        evaluated_at=current.isoformat(),
        score_count=len(scoped_scores),
        fetched_score_count=len(scores),
        metrics=results,
        failures=failures,
    )


def fetch_langfuse_scores(
    names: list[str],
    *,
    environment: str,
    lookback_hours: int,
) -> list[dict[str, object]]:
    status = observability_status()
    if not status["enabled"]:
        raise RuntimeError(
            "Langfuse is not ready. Install the observability extra, set both keys, "
            "and enable LANGFUSE_ENABLED."
        )
    from langfuse import get_client

    client = get_client()
    cursor: str | None = None
    output: list[dict[str, object]] = []
    from_timestamp = datetime.now(timezone.utc) - timedelta(hours=max(1, lookback_hours))
    while True:
        response = client.api.scores_v3.get_many_v3(
            name=",".join(names),
            environment=environment,
            from_timestamp=from_timestamp,
            fields="details,subject",
            limit=100,
            cursor=cursor,
        )
        for score in response.data:
            metadata = getattr(score, "metadata", None)
            if hasattr(metadata, "model_dump"):
                metadata = metadata.model_dump()
            if not isinstance(metadata, dict):
                metadata = {}
            output.append(
                {
                    "id": score.id,
                    "name": score.name,
                    "value": score.value,
                    "timestamp": score.timestamp.isoformat(),
                    "comment": score.comment,
                    "metadata": metadata,
                    "release": metadata.get("release"),
                }
            )
        cursor = response.meta.cursor
        if not cursor:
            break
    return output


def run_langfuse_gate(
    config_path: str | Path,
    output_path: str | Path,
    *,
    allow_insufficient_data: bool = False,
    lookback_hours: int | None = None,
) -> GateReport:
    config = load_gate_config(config_path)
    if lookback_hours is not None:
        config["lookbackHours"] = max(1, lookback_hours)
    names = [str(name) for name in config["metrics"]]
    scores = fetch_langfuse_scores(
        names,
        environment=str(config.get("environment") or "development"),
        lookback_hours=int(config.get("lookbackHours", 168)),
    )
    report = evaluate_score_gate(
        scores,
        config,
        allow_insufficient_data=allow_insufficient_data,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return report
