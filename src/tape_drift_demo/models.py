from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Fact(BaseModel):
    id: str
    category: str
    statement: str
    keywords: list[str]


class Scenario(BaseModel):
    scenario: str
    facts: list[Fact]
    round_updates: list[str]


class ClaimStatus(StrEnum):
    preserved = "preserved"
    contradicted = "contradicted"
    missing = "missing"


class FactAssessment(BaseModel):
    fact_id: str
    status: ClaimStatus
    explanation: str
    quote: str | None = None


class Grade(BaseModel):
    assessments: list[FactAssessment]
    unsupported_claims: list[str] = Field(default_factory=list)


class SummaryPayload(BaseModel):
    summary: str
    source_ids: list[str] = Field(default_factory=list)


class StrategyName(StrEnum):
    recursive = "recursive"
    grounded = "grounded"


class RoundResult(BaseModel):
    strategy: StrategyName
    round: int
    anchor: str
    summary: str
    source_ids: list[str]
    grade: Grade
    deterministic_recall: float
    semantic_recall: float
    contradicted: int
    unsupported: int
    summary_chars: int


class ExperimentResult(BaseModel):
    model: str
    judge_model: str
    rounds: int
    scenario: str
    results: list[RoundResult]
    tape_dir: str
    type: Literal["tape-summary-drift"] = "tape-summary-drift"
