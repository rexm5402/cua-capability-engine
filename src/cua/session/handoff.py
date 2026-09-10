"""Resume semantics: the rule that makes human handoff safe.

    with handoff(lease, escalator, surface, request) as outcome:
        if outcome.resumed:
            ...continue the run...

The context manager pauses automation, transfers the lease to the operator,
blocks until the console writes a resolution, then reacquires.

THE IMPORTANT RULE
------------------
On resume, automation RE-OBSERVES and RE-VERIFIES the current step's
post-condition before continuing. Never assume the human left the session where
the automation expected it -- they may have fixed the problem, or navigated
three screens away, or logged out, or fixed it a different way than the step
anticipated. "The operator clicked Resume" is a statement about the operator's
belief, not about the DOM.

If the post-condition does not hold, that is a fresh `Failure`
(`post_condition_unmet_after_handoff`), surfaced -- not a silent continue. A
silent continue here is the single most dangerous thing this whole subsystem
could do, because the next step would act on a screen nobody has verified.

We also diff the observation before and after the human held the lease, and
record the delta as a `HumanIntervention` for evidence. We do not keystroke-log
the operator; the delta is inference, and is labelled as such.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from cua.artifact import Checkpoint, Step
from cua.outcomes import Failure
from cua.session.escalation import (
    Aborted,
    Escalator,
    HumanIntervention,
    InterventionRequest,
    MarkedFailed,
    Resolution,
    Resumed,
)
from cua.session.lease import LeaseGrant, LeaseState, SessionLease

POST_CONDITION_UNMET = "post_condition_unmet_after_handoff"
OBSERVE_FAILED = "observe_failed_after_handoff"


# -- observation diffing ---------------------------------------------------


def _node_key(n: Any) -> tuple[str, str]:
    return (getattr(n, "role", ""), getattr(n, "name", ""))


def diff_observations(before: Any, after: Any) -> dict[str, Any]:
    """A structural delta between two `Observation`s.

    Deliberately coarse: role+name identity rather than `ref`, because refs are
    per-snapshot handles and would report every node as changed.
    """
    b_nodes = {_node_key(n): n for n in getattr(before, "nodes", ())}
    a_nodes = {_node_key(n): n for n in getattr(after, "nodes", ())}

    appeared = sorted(f"{r} {n!r}" for (r, n) in a_nodes.keys() - b_nodes.keys())
    disappeared = sorted(f"{r} {n!r}" for (r, n) in b_nodes.keys() - a_nodes.keys())

    value_changes: list[dict[str, Any]] = []
    for key in b_nodes.keys() & a_nodes.keys():
        bv = getattr(b_nodes[key], "value", None)
        av = getattr(a_nodes[key], "value", None)
        if bv != av:
            value_changes.append(
                {"role": key[0], "name": key[1], "before": bv, "after": av}
            )
    value_changes.sort(key=lambda d: (d["role"], d["name"]))

    return {
        "url_changed": getattr(before, "url", None) != getattr(after, "url", None),
        "url_before": getattr(before, "url", ""),
        "url_after": getattr(after, "url", ""),
        "title_before": getattr(before, "title", ""),
        "title_after": getattr(after, "title", ""),
        "text_digest_changed": (
            getattr(before, "text_digest", "") != getattr(after, "text_digest", "")
        ),
        "controls_appeared": appeared,
        "controls_disappeared": disappeared,
        "value_changes": value_changes,
    }


def infer_actions(delta: dict[str, Any]) -> list[str]:
    """Turn a delta into prose an auditor can read. Inference, not a log."""
    out: list[str] = []
    if delta.get("url_changed"):
        out.append(
            f"navigated from {delta['url_before']!r} to {delta['url_after']!r}"
        )
    for vc in delta.get("value_changes", []):
        out.append(
            f"changed {vc['role']} {vc['name']!r}: {vc['before']!r} -> {vc['after']!r}"
        )
    for c in delta.get("controls_appeared", []):
        out.append(f"caused control to appear: {c}")
    for c in delta.get("controls_disappeared", []):
        out.append(f"caused control to disappear: {c}")
    if not out:
        if delta.get("text_digest_changed"):
            out.append("changed page text but no control-level change was detectable")
        else:
            out.append("no observable change (operator may have acted outside the app)")
    return out


# -- outcome ---------------------------------------------------------------


@dataclass
class HandoffOutcome:
    """What the engine inspects after the `with` block opens."""

    resolution: Optional[Resolution] = None
    resumed: bool = False
    verified: bool = False
    failure: Optional[Failure] = None
    intervention: Optional[HumanIntervention] = None
    grant: Optional[LeaseGrant] = None
    delta: dict[str, Any] = field(default_factory=dict)

    @property
    def may_continue(self) -> bool:
        """The ONLY thing the engine should branch on. True requires both a
        human Resume AND a re-verified post-condition."""
        return self.resumed and self.verified and self.failure is None


@contextmanager
def handoff(
    lease: SessionLease,
    escalator: Escalator,
    surface: Any,
    request: InterventionRequest,
    *,
    grant: LeaseGrant,
    operator_holder: str = "operator-console",
    timeout: float = 1800.0,
    post_condition: Checkpoint | None = None,
    step: Step | None = None,
    verify_timeout_ms: int = 10_000,
    recorder: Any = None,
    operator_ttl_s: float | None = None,
) -> Iterator[HandoffOutcome]:
    """Pause, hand over, block, reacquire, re-verify.

    `grant` is the engine's current lease grant; the engine must be holding the
    lease in AUTOMATION when this is called.

    Note on who reclaims: the engine performs both transfers and therefore
    keeps the operator-side token it minted. The console authenticates by
    writing a resolution into the store, not by holding a lease token. That is
    a simplification -- a console that itself acquires the lease would call
    `lease.transfer(operator_holder, engine_holder)` with its own token -- but
    the invariant is unchanged either way: at every instant the lease file names
    exactly one holder, and the fence rejects any superseded one.
    """
    check = post_condition or (step.post_condition if step is not None else None)
    outcome = HandoffOutcome()
    started = time.time()
    engine_holder = grant.holder

    before = surface.observe()

    op_grant = lease.transfer(
        engine_holder,
        operator_holder,
        token=grant.token,
        to_state=LeaseState.OPERATOR,
        ttl_s=operator_ttl_s,
        note=f"handoff for {request.request_id}",
    )
    if recorder is not None:
        recorder.note(
            f"handoff: lease -> OPERATOR for {request.request_id} "
            f"({request.reason_code.value})"
        )

    try:
        resolution = escalator.await_resolution(request.request_id, timeout)
    finally:
        # Whatever happened -- resolution, timeout, operator walked away -- the
        # engine takes the lease back so the session is never left orphaned in
        # OPERATOR. If the operator's TTL lapsed first, reclaim it explicitly.
        try:
            new_grant = lease.transfer(
                operator_holder,
                engine_holder,
                token=op_grant.token,
                to_state=LeaseState.AUTOMATION,
                note=f"resume after {request.request_id}",
            )
        except Exception:
            lease.expire_if_stale(note=f"operator lapsed during {request.request_id}")
            new_grant = lease.acquire(
                engine_holder,
                target_state=LeaseState.AUTOMATION,
                expected_state=LeaseState.SUSPENDED,
                break_expired=True,
                note=f"forced resume after {request.request_id}",
            )
        outcome.grant = new_grant

    outcome.resolution = resolution
    operator_id = getattr(resolution, "operator_id", "unknown")

    after = surface.observe()
    delta = diff_observations(before, after)
    outcome.delta = delta

    ended = time.time()
    outcome.intervention = HumanIntervention(
        request_id=request.request_id,
        run_id=request.run_id,
        operator_id=operator_id,
        resolution_kind=resolution.kind,
        started_at=request.created_at,
        ended_at=getattr(resolution, "resolved_at", ""),
        duration_s=round(ended - started, 3),
        actions_inferred=infer_actions(delta),
        url_before=delta["url_before"],
        url_after=delta["url_after"],
        observation_delta=delta,
        note=getattr(resolution, "note", ""),
    )
    escalator.store.put_intervention(outcome.intervention)

    if isinstance(resolution, Resumed):
        outcome.resumed = True
        # THE RULE. A human clicking Resume is not evidence about the screen.
        if check is None:
            # No declared post-condition to check. We do not pretend we
            # verified something; we say so, and treat it as verified only
            # because there is nothing to assert.
            outcome.verified = True
            if recorder is not None:
                recorder.note(
                    "resume: no post_condition declared for this step; nothing "
                    "to re-verify",
                    level="WARN",
                )
        else:
            try:
                ok = bool(surface.wait_for(check, verify_timeout_ms))
            except Exception as exc:  # a surface that cannot even look
                outcome.verified = False
                outcome.failure = _failure(
                    request, check, f"observation failed: {exc}", code=OBSERVE_FAILED
                )
                ok = False
            if ok:
                outcome.verified = True
            elif outcome.failure is None:
                outcome.verified = False
                outcome.failure = _failure(
                    request,
                    check,
                    _observed_summary(after, delta),
                    code=POST_CONDITION_UNMET,
                )
    elif isinstance(resolution, Aborted):
        outcome.resumed = False
    elif isinstance(resolution, MarkedFailed):
        outcome.resumed = False
        outcome.failure = Failure(
            capability_id=request.capability_id,
            code=resolution.code,
            at_step_id=request.step_id,
            step_intent=request.step_intent,
            expected="operator judgement",
            observed=resolution.note or "operator marked this run failed",
            escalated=True,
        )

    if recorder is not None:
        recorder.step(
            step_id=request.step_id or "handoff",
            intent=request.step_intent,
            decision={"resolution": resolution.kind, "operator": operator_id},
            outcome=outcome.failure,
            human_intervention=outcome.intervention.model_dump(mode="json"),
        )

    yield outcome


def _observed_summary(after: Any, delta: dict[str, Any]) -> str:
    return (
        f"after the operator resumed, url={getattr(after, 'url', '')!r}, "
        f"title={getattr(after, 'title', '')!r}; "
        f"operator changes inferred: {'; '.join(infer_actions(delta))}"
    )


def _failure(
    request: InterventionRequest,
    check: Checkpoint,
    observed: str,
    *,
    code: str,
) -> Failure:
    return Failure(
        capability_id=request.capability_id,
        code=code,
        at_step_id=request.step_id,
        step_intent=request.step_intent,
        expected=(
            f"post-condition of the escalated step still holds: "
            f"{check.description}"
        ),
        observed=observed,
        escalated=True,
    )


__all__ = [
    "handoff",
    "HandoffOutcome",
    "diff_observations",
    "infer_actions",
    "POST_CONDITION_UNMET",
    "OBSERVE_FAILED",
]
