"""The replay result contract.

Three outcomes, structurally distinct. The brief's glossary names conflating a
business outcome with a failure as the most common design mistake in this
problem; representing them as one union of three types makes it impossible
rather than merely discouraged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Success(BaseModel):
    kind: Literal["success"] = "success"
    capability_id: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    steps_executed: int = 0
    tier_escalations: int = 0
    evidence_dir: str | None = None


class BusinessOutcome(BaseModel):
    """A legitimate answer, not a crash. 'No such member' belongs here."""

    kind: Literal["business_outcome"] = "business_outcome"
    capability_id: str
    code: str
    message: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    at_step_id: str | None = None
    evidence_dir: str | None = None


class Failure(BaseModel):
    """Something is actually broken. Debuggable by construction: every field
    here answers a question the on-call engineer will ask."""

    kind: Literal["failure"] = "failure"
    capability_id: str
    code: str
    at_step_id: str | None = None
    step_intent: str | None = None
    expected: str = ""
    observed: str = ""
    resolved_tier: int | None = None
    candidates: int | None = None
    evidence_dir: str | None = None
    escalated: bool = False


ReplayOutcome = Success | BusinessOutcome | Failure
