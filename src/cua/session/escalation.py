"""Escalation: the one channel by which a run leaves the machine and reaches a
person.

There is deliberately ONE queue for two things that look different but are not:

  * "I am stuck"      -- discovery made no state progress, or replay produced a
                         Failure.
  * "please approve"  -- the policy gate returned RequireApproval for an
                         irreversible action.

Both are the same event from the operator's point of view: the run has stopped
and a human must look at it and decide. Giving them separate queues would mean
two consoles, two sets of notifications, and two chances to leave one of them
unstaffed. `cua.policy.gate.RequireApproval` says the same thing in its
docstring; this module is the other end of that wire.

Like the lease, the store is on the filesystem, because the console is a
different process:

    <root>/requests/<request_id>.json      written by the engine, read by all
    <root>/resolutions/<request_id>.json   written by the console
    <root>/interventions/<request_id>.json what the human actually did

Requests are written temp-then-`os.replace` so a console listing never picks up
a half-written file.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ReasonCode(str, Enum):
    """Why we stopped. The three trigger sources the engine must support, plus
    an explicit catch-all so an unmapped cause is visibly unmapped rather than
    mislabelled as one of the real ones."""

    NO_STATE_PROGRESS = "no_state_progress"      # discovery is looping
    REPLAY_FAILURE = "replay_failure"            # replay produced a Failure
    APPROVAL_REQUIRED = "approval_required"      # gate returned RequireApproval
    OTHER = "other"


class PermittedAction(str, Enum):
    RESUME = "resume"
    ABORT = "abort"
    FAIL = "fail"


class InterventionRequest(BaseModel):
    """Everything a person needs to act WITHOUT reading the codebase.

    `step_intent` is the load-bearing field. An operator handed
    "click(#btn-3f2a) failed" cannot help. An operator handed "confirm the
    beneficiary change so the policy is written back" can decide in seconds.
    """

    request_id: str = Field(default_factory=lambda: f"ir_{uuid.uuid4().hex[:12]}")
    run_id: str
    capability_id: str
    capability_name: str
    step_id: str | None = None
    step_intent: str = Field(
        description="Prose. What the automation was trying to achieve here."
    )
    reason_code: ReasonCode
    why: str = Field(description="Human-readable explanation of what went wrong.")
    current_url: str = ""
    screenshot_path: str | None = None
    observation_digest: str = ""
    permitted_actions: list[PermittedAction] = Field(
        default_factory=lambda: [
            PermittedAction.RESUME,
            PermittedAction.ABORT,
            PermittedAction.FAIL,
        ]
    )
    risk_class: str | None = None
    proposed_action: str | None = Field(
        default=None,
        description="For APPROVAL_REQUIRED: the action awaiting a yes/no.",
    )
    created_at: str = Field(default_factory=_now_iso)
    deadline: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


# -- resolutions ----------------------------------------------------------


class Resumed(BaseModel):
    """The human fixed/approved it; automation should continue -- but only
    after re-verifying the step's post-condition. See handoff.py."""

    kind: Literal["resumed"] = "resumed"
    request_id: str
    operator_id: str
    note: str = ""
    resolved_at: str = Field(default_factory=_now_iso)


class Aborted(BaseModel):
    """Stop the run. Not an error: a deliberate human decision."""

    kind: Literal["aborted"] = "aborted"
    request_id: str
    operator_id: str
    note: str = ""
    resolved_at: str = Field(default_factory=_now_iso)


class MarkedFailed(BaseModel):
    """The human confirms this is genuinely broken; surface a Failure."""

    kind: Literal["marked_failed"] = "marked_failed"
    request_id: str
    operator_id: str
    code: str = "human_marked_failed"
    note: str = ""
    resolved_at: str = Field(default_factory=_now_iso)


Resolution = Resumed | Aborted | MarkedFailed

_RESOLUTION_TYPES = {
    "resumed": Resumed,
    "aborted": Aborted,
    "marked_failed": MarkedFailed,
}


def parse_resolution(payload: dict[str, Any]) -> Resolution:
    kind = payload.get("kind")
    if kind not in _RESOLUTION_TYPES:
        raise ValueError(f"unknown resolution kind {kind!r}")
    return _RESOLUTION_TYPES[kind].model_validate(payload)


class HumanIntervention(BaseModel):
    """The audit record: what the operator actually did while holding the
    lease. Attached to evidence via `EvidenceRecorder.step(...)`.

    `actions_inferred` is honest about its own limits -- we do not keystroke-log
    the human. We diff the observation before and after and describe the delta.
    """

    request_id: str
    run_id: str
    operator_id: str
    resolution_kind: str
    started_at: str
    ended_at: str
    duration_s: float
    actions_inferred: list[str] = Field(default_factory=list)
    url_before: str = ""
    url_after: str = ""
    observation_delta: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


# -- store ----------------------------------------------------------------


class RequestStore:
    """A directory of JSON files. Chosen for the same reason as the lease file:
    the console is another process, and a shared directory is the smallest
    thing both can see. Swapping this for Postgres changes no caller."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.requests_dir = self.root / "requests"
        self.resolutions_dir = self.root / "resolutions"
        self.interventions_dir = self.root / "interventions"
        for d in (self.requests_dir, self.resolutions_dir, self.interventions_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        os.replace(tmp, path)

    # requests

    def put_request(self, req: InterventionRequest) -> Path:
        p = self.requests_dir / f"{req.request_id}.json"
        self._atomic_write(p, req.model_dump(mode="json"))
        return p

    def get_request(self, request_id: str) -> InterventionRequest | None:
        p = self.requests_dir / f"{request_id}.json"
        if not p.exists():
            return None
        return InterventionRequest.model_validate(json.loads(p.read_text()))

    def list_requests(self, *, open_only: bool = True) -> list[InterventionRequest]:
        out: list[InterventionRequest] = []
        for p in sorted(self.requests_dir.glob("*.json")):
            try:
                req = InterventionRequest.model_validate(json.loads(p.read_text()))
            except (ValueError, OSError):
                continue  # a request being written right now; next poll gets it
            if open_only and self.get_resolution(req.request_id) is not None:
                continue
            out.append(req)
        out.sort(key=lambda r: r.created_at)
        return out

    # resolutions

    def put_resolution(self, res: Resolution) -> Path:
        p = self.resolutions_dir / f"{res.request_id}.json"
        self._atomic_write(p, res.model_dump(mode="json"))
        return p

    def get_resolution(self, request_id: str) -> Resolution | None:
        p = self.resolutions_dir / f"{request_id}.json"
        if not p.exists():
            return None
        try:
            return parse_resolution(json.loads(p.read_text()))
        except (ValueError, OSError):
            return None

    # interventions

    def put_intervention(self, rec: HumanIntervention) -> Path:
        p = self.interventions_dir / f"{rec.request_id}.json"
        self._atomic_write(p, rec.model_dump(mode="json"))
        return p

    def get_intervention(self, request_id: str) -> HumanIntervention | None:
        p = self.interventions_dir / f"{request_id}.json"
        if not p.exists():
            return None
        return HumanIntervention.model_validate(json.loads(p.read_text()))


class EscalationTimeout(TimeoutError):
    """Nobody answered before the deadline. The caller decides what that means;
    this module refuses to pick 'just continue' on a human's behalf."""


class Escalator:
    """Engine-side API. Three named constructors, one channel."""

    def __init__(
        self,
        store: RequestStore,
        *,
        run_id: str = "unknown-run",
        default_deadline_s: float | None = 1800.0,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.default_deadline_s = default_deadline_s

    def raise_request(
        self,
        *,
        capability_id: str,
        capability_name: str,
        step_intent: str,
        reason_code: ReasonCode,
        why: str,
        step_id: str | None = None,
        current_url: str = "",
        screenshot_path: str | None = None,
        observation_digest: str = "",
        permitted_actions: list[PermittedAction] | None = None,
        risk_class: str | None = None,
        proposed_action: str | None = None,
        deadline_s: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> InterventionRequest:
        secs = self.default_deadline_s if deadline_s is None else deadline_s
        deadline = None
        if secs is not None:
            deadline = datetime.fromtimestamp(
                time.time() + secs, tz=timezone.utc
            ).isoformat(timespec="milliseconds")
        req = InterventionRequest(
            run_id=self.run_id,
            capability_id=capability_id,
            capability_name=capability_name,
            step_id=step_id,
            step_intent=step_intent,
            reason_code=reason_code,
            why=why,
            current_url=current_url,
            screenshot_path=screenshot_path,
            observation_digest=observation_digest,
            permitted_actions=permitted_actions
            or [PermittedAction.RESUME, PermittedAction.ABORT, PermittedAction.FAIL],
            risk_class=risk_class,
            proposed_action=proposed_action,
            deadline=deadline,
            context=context or {},
        )
        self.store.put_request(req)
        return req

    # -- the three trigger sources ----------------------------------------

    def for_no_progress(
        self,
        *,
        capability_id: str,
        capability_name: str,
        step_intent: str,
        steps_without_progress: int,
        **kw: Any,
    ) -> InterventionRequest:
        """Trigger 1: discovery has made no state progress for N steps."""
        return self.raise_request(
            capability_id=capability_id,
            capability_name=capability_name,
            step_intent=step_intent,
            reason_code=ReasonCode.NO_STATE_PROGRESS,
            why=(
                f"Discovery made no observable state progress for "
                f"{steps_without_progress} consecutive steps. The screen is "
                f"probably waiting on something the agent cannot see or cannot do."
            ),
            context={"steps_without_progress": steps_without_progress},
            **kw,
        )

    def for_replay_failure(
        self,
        *,
        failure: Any,
        capability_name: str,
        **kw: Any,
    ) -> InterventionRequest:
        """Trigger 2: replay produced a `cua.outcomes.Failure`."""
        return self.raise_request(
            capability_id=getattr(failure, "capability_id", "unknown"),
            capability_name=capability_name,
            step_id=getattr(failure, "at_step_id", None),
            step_intent=getattr(failure, "step_intent", None) or "(no intent recorded)",
            reason_code=ReasonCode.REPLAY_FAILURE,
            why=(
                f"Replay failed with code {getattr(failure, 'code', '?')!r}. "
                f"Expected: {getattr(failure, 'expected', '')}. "
                f"Observed: {getattr(failure, 'observed', '')}."
            ),
            context={
                "failure_code": getattr(failure, "code", None),
                "resolved_tier": getattr(failure, "resolved_tier", None),
                "candidates": getattr(failure, "candidates", None),
                "evidence_dir": getattr(failure, "evidence_dir", None),
            },
            **kw,
        )

    def for_approval(
        self,
        *,
        capability_id: str,
        capability_name: str,
        step_intent: str,
        decision: Any,
        proposed_action: str,
        risk_class: str | None = None,
        **kw: Any,
    ) -> InterventionRequest:
        """Trigger 3: the policy gate returned `RequireApproval`.

        Same queue as the stuck cases, on purpose -- see the module docstring.
        """
        return self.raise_request(
            capability_id=capability_id,
            capability_name=capability_name,
            step_intent=step_intent,
            reason_code=ReasonCode.APPROVAL_REQUIRED,
            why=(
                "The policy gate requires a human to approve this action: "
                f"{getattr(decision, 'reason', str(decision))}"
            ),
            proposed_action=proposed_action,
            risk_class=risk_class,
            **kw,
        )

    # -- waiting -----------------------------------------------------------

    def await_resolution(
        self,
        request_id: str,
        timeout: float,
        *,
        poll_s: float = 0.02,
    ) -> Resolution:
        deadline = time.time() + timeout
        while True:
            res = self.store.get_resolution(request_id)
            if res is not None:
                return res
            if time.time() >= deadline:
                raise EscalationTimeout(
                    f"no operator resolved {request_id} within {timeout}s"
                )
            time.sleep(poll_s)

    # -- console-side ------------------------------------------------------

    def resolve(self, res: Resolution) -> Resolution:
        self.store.put_resolution(res)
        return res


__all__ = [
    "InterventionRequest",
    "ReasonCode",
    "PermittedAction",
    "Resolution",
    "Resumed",
    "Aborted",
    "MarkedFailed",
    "parse_resolution",
    "HumanIntervention",
    "RequestStore",
    "Escalator",
    "EscalationTimeout",
]
