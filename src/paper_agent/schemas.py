from __future__ import annotations

from typing import Literal

from pydantic import BaseModel as _PydanticBaseModel, Field


class BaseModel(_PydanticBaseModel):
    """Pydantic v1/v2 compatibility wrapper.

    Some local/conda environments still resolve pydantic v1, where `.model_dump()`
    and `.model_dump_json()` do not exist. The app code uses the v2 names, so this
    wrapper supplies them on v1 while preserving v2 behavior.
    """

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        parent = super()
        if hasattr(parent, "model_dump"):
            return parent.model_dump(*args, **kwargs)
        return self.dict(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):  # type: ignore[override]
        parent = super()
        if hasattr(parent, "model_dump_json"):
            return parent.model_dump_json(*args, **kwargs)
        return self.json(*args, **kwargs)


SourceType = Literal["paper", "web", "inference", "unsupported"]
SupportLabel = Literal[
    "supported", "partially_supported", "unsupported", "needs_external_verification"
]


class ClaimRecord(BaseModel):
    claim: str = Field(..., description="Atomic claim made in the final report.")
    source_type: SourceType
    support: SupportLabel
    evidence: str = Field(..., description="Paper page/section, web URL, or reasoning note.")
    confidence: Literal["low", "medium", "high"] = "medium"


class EquationCard(BaseModel):
    id: str
    page: int | None = None
    raw: str
    purpose: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)
    plain_english: str | None = None


class FigureCard(BaseModel):
    id: str
    page: int
    caption: str
    surrounding_text: str = ""


class TableCard(BaseModel):
    id: str
    page: int
    caption: str
    raw_text: str


class PaperMetadata(BaseModel):
    title_guess: str | None
    authors_guess: list[str] = Field(default_factory=list)
    page_count: int
    source_pdf: str
