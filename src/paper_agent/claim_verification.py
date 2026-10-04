"""Bounded claim checks. Citation presence is never treated as entailment."""

from __future__ import annotations

import re
from typing import Callable, Literal

from paper_agent.schemas import BaseModel

CITATION = re.compile(r"\[([A-Za-z][\w:.\-/]*)\]")
MAX_CLAIMS = 24


class ClaimCheck(BaseModel):
    claim: str
    evidence_ids: list[str] = []
    status: Literal["supported", "contradicted", "unresolved", "inference"] = "unresolved"
    reason: str = "No supporting passage was verified."
    quote: str = ""


def normalized(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .\"'*")


def check_claims(
    answer: str,
    passages: dict[str, str],
    judge: Callable[[list[dict]], list[dict]] | None = None,
) -> tuple[list[ClaimCheck], bool]:
    # Sentence-sized units include bullet items and displayed math. Do not use the
    # answering model's self-selected claims: it could omit its own bad assertions.
    units = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", answer)
    units = [u.strip() for u in units if u.strip()]
    results: list[ClaimCheck] = []
    pending: list[dict] = []
    for unit in units[:MAX_CLAIMS]:
        ids = list(dict.fromkeys(CITATION.findall(unit)))
        claim = CITATION.sub("", unit).strip(" -*#")
        result = ClaimCheck(claim=claim, evidence_ids=ids)
        sources = {key: passages[key] for key in ids if key in passages}
        if not ids or len(sources) != len(ids):
            result.reason = "Missing or unresolvable inline evidence citation."
        elif (
            any(
                normalized(claim) == normalized(sentence)
                for passage in sources.values()
                for sentence in re.split(r"\n+|(?<=[.!?])\s+", passage)
            )
            and len(claim) > 15
        ):
            result.status = "supported"
            result.reason = "The claim appears verbatim in the cited passage."
            result.quote = claim
        else:
            pending.append({"index": len(results), "claim": claim, "evidence": sources})
        results.append(result)
    if pending and judge is not None:
        # The caller handles provider failure; malformed or omitted findings fail closed.
        findings = judge(pending)
        requests = {item["index"]: item for item in pending}
        seen: set[int] = set()
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            index = finding.get("index")
            if not isinstance(index, int) or index not in requests or index in seen:
                continue
            seen.add(index)
            result = results[index]
            status = finding.get("status")
            quote = str(finding.get("quote", ""))
            actual_quote = len(quote.strip()) >= 12 and any(
                normalized(quote) in normalized(p) for p in requests[index]["evidence"].values()
            )
            if status in {"supported", "contradicted"} and actual_quote:
                result.status = status
                result.quote = quote
            elif status == "inference":
                result.status = "inference"
            else:
                result.status = "unresolved"
            result.reason = str(finding.get("reason", result.reason))[:600]
            if status in {"supported", "contradicted"} and not actual_quote:
                result.reason = "The verifier did not return a matching source quotation."
            if result.status == "supported":
                numbers = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?", result.claim))
                source_numbers = set(
                    re.findall(
                        r"(?<!\w)\d+(?:\.\d+)?", " ".join(requests[index]["evidence"].values())
                    )
                )
                if numbers - source_numbers:
                    result.status = "unresolved"
                    result.reason = "A numerical value in the claim is absent from its cited passages; derived values need separate validation."
    return results, len(units) <= MAX_CLAIMS
