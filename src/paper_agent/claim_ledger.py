from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from paper_agent.schemas import ClaimRecord
from paper_agent.workspace import Workspace

DEFAULT_CLAIMS: list[ClaimRecord] = []


def claim_ledger_path(workspace: Workspace) -> Path:
    return workspace.path("final/claim_ledger.json")


def load_claim_ledger(workspace: Workspace) -> list[ClaimRecord]:
    path = claim_ledger_path(workspace)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ClaimRecord.model_validate(item) for item in raw]


def save_claim_ledger(workspace: Workspace, claims: list[ClaimRecord]) -> Path:
    path = claim_ledger_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([claim.model_dump() for claim in claims], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def add_claim(workspace: Workspace, claim: ClaimRecord) -> Path:
    claims = load_claim_ledger(workspace)
    claims.append(claim)
    return save_claim_ledger(workspace, claims)


def validate_claims_json(raw_json: str) -> tuple[bool, str]:
    try:
        raw = json.loads(raw_json)
        if not isinstance(raw, list):
            return False, "Claim ledger must be a JSON list."
        [ClaimRecord.model_validate(item) for item in raw]
        return True, "valid"
    except (json.JSONDecodeError, ValidationError) as exc:
        return False, str(exc)
