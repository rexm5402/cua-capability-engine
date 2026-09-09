"""Replay telemetry, deliberately separated from the artifact.

Stability data mutates on every replay. Storing it on the Capability would
churn the hash of an immutable, versioned, reviewable document without its
behaviour changing -- and would make "has this artifact been modified?"
unanswerable. It lives here instead, keyed by capability id + version, and the
approval workflow reads it alongside the artifact.

It is also where drift surfaces: a step resolving above its RECORDED BASELINE
tier is the early-warning signal that a tenant needs an overlay.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class StepTelemetry(BaseModel):
    step_id: str
    baseline_tier: int | None = None
    observed_tiers: dict[str, int] = Field(default_factory=dict)

    @property
    def escalations(self) -> int:
        """Resolutions strictly worse than this step's own baseline."""
        if self.baseline_tier is None:
            return 0
        return sum(
            count
            for tier, count in self.observed_tiers.items()
            if int(tier) > self.baseline_tier
        )


class CapabilityTelemetry(BaseModel):
    capability_id: str
    version: str
    tenant_id: str | None = None
    replays: int = 0
    successes: int = 0
    business_outcomes: dict[str, int] = Field(default_factory=dict)
    failures: dict[str, int] = Field(default_factory=dict)
    steps: dict[str, StepTelemetry] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.successes / self.replays if self.replays else 0.0

    def record_resolution(self, step_id: str, tier: int, baseline: int | None) -> bool:
        st = self.steps.setdefault(
            step_id, StepTelemetry(step_id=step_id, baseline_tier=baseline)
        )
        if st.baseline_tier is None:
            st.baseline_tier = baseline
        st.observed_tiers[str(tier)] = st.observed_tiers.get(str(tier), 0) + 1
        return st.baseline_tier is not None and tier > st.baseline_tier

    def drifting_steps(self) -> list[str]:
        return [sid for sid, st in self.steps.items() if st.escalations > 0]


class TelemetryStore:
    def __init__(self, root: str | Path = "evidence/telemetry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cap_id: str, version: str, tenant: str | None) -> Path:
        return self.root / f"{cap_id}@{version}{'#' + tenant if tenant else ''}.json"

    def load(
        self, cap_id: str, version: str, tenant: str | None = None
    ) -> CapabilityTelemetry:
        p = self._path(cap_id, version, tenant)
        if p.exists():
            return CapabilityTelemetry.model_validate_json(p.read_text())
        return CapabilityTelemetry(
            capability_id=cap_id, version=version, tenant_id=tenant
        )

    def save(self, t: CapabilityTelemetry) -> None:
        p = self._path(t.capability_id, t.version, t.tenant_id)
        p.write_text(json.dumps(t.model_dump(), indent=2))
