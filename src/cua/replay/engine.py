"""The deterministic replay engine -- the production execution path.

An AI agent calls `replay(capability, inputs, surface)` and gets back one of
three structurally distinct outcomes. No model is consulted, nothing is
inferred, and no probabilistic component sits anywhere on this path: replay is
a typed function over a reviewed artifact. That is the whole product claim, and
it is enforced here by the absence of any LLM import.

Execution order per step, and the reason for it:

    1.  bind the step's value (late binding; sensitive values never logged)
    2.  policy gate       -- the engine cannot reach a surface without a Decision
    3.  act
    4.  SIGNAL SCAN      -- classify what the world said BEFORE judging it
    5.  post-condition   -- prove the action landed; never a blind sleep

Step 4 precedes step 5 deliberately. A failed checkpoint and a legitimate
"record not found" page are the same observation from the checkpoint's point of
view; scanning first is what keeps a business answer from being reported as a
crash.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from cua.artifact import (
    ActionType,
    Capability,
    Checkpoint,
    OutputSpec,
    ParamSpec,
    RiskClass,
    Sensitivity,
    Step,
    ValueRef,
)
from cua.evidence.recorder import EvidenceRecorder
from cua.locator.resolve import resolve
from cua.outcomes import BusinessOutcome, Failure, ReplayOutcome, Success
from cua.policy.gate import Allow, Deny, PolicyGate, RequireApproval
from cua.policy.redaction import Redactor
from cua.replay.signals import SignalMatch, find_dismiss_control, scan
from cua.surface.base import Action, ActResult, Observation, Surface
from cua.telemetry import CapabilityTelemetry, TelemetryStore

__all__ = [
    "replay",
    "Escalator",
    "DryRunSuccess",
    "ReplayError",
    "INVALID_INPUT",
    "POLICY_DENIED",
    "APPROVAL_REQUIRED",
    "SUCCESS_CHECKPOINT_FAILED",
]

# -- stable failure codes ---------------------------------------------------

INVALID_INPUT = "invalid_input"
POLICY_DENIED = "policy_denied"
APPROVAL_REQUIRED = "approval_required"
ACTION_FAILED = "action_failed"
LOCATOR_UNRESOLVED = "locator_unresolved"
POST_CONDITION_FAILED = "post_condition_failed"
SUCCESS_CHECKPOINT_FAILED = "success_checkpoint_failed"
UNRECOVERED_SIGNAL = "unrecovered_signal"
DRY_RUN_UNREACHABLE = "dry_run_unreachable"
REAUTH_FAILED = "reauth_failed"
UNKNOWN_STEP = "unknown_step"

#: Hard ceiling on inner iterations for one step, independent of any rule's
#: `max_attempts`. A recoverable condition that keeps re-firing must terminate
#: the run, not the process.
MAX_STEP_ITERATIONS = 25


class ReplayError(RuntimeError):
    """Only raised for programmer error (e.g. a bad `dry_run_from`). Runtime
    conditions are returned as outcomes, never raised."""


class DryRunSuccess(Success):
    """A `Success` that also says the run was a dry run.

    `Success` is a fixed contract with no room for the flag, and smuggling it
    into `outputs` would corrupt the typed result the caller reads. Subclassing
    keeps `isinstance(outcome, Success)` true for every existing caller while
    making the distinction inspectable.
    """

    dry_run: bool = True
    dry_run_from: str | None = None
    resolved_without_acting: list[str] = []


@runtime_checkable
class Escalator(Protocol):
    """Where a run goes when a human must look at it.

    The same queue serves policy approvals and stuck states -- one place a
    person watches, rather than two.
    """

    def request_approval(
        self,
        *,
        capability: Capability,
        step: Step,
        reason: str,
        observation: Observation,
    ) -> bool:
        """Return True to proceed with the action, False to stop the run."""
        ...


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def _type_ok(spec: ParamSpec, value: Any) -> bool:
    if spec.type == "boolean":
        return isinstance(value, bool)
    if spec.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if spec.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


def validate_inputs(capability: Capability, inputs: dict[str, Any]) -> str | None:
    """Return a human-readable problem, or None when the inputs are valid.

    Runs to completion before ANY surface interaction: a typo in an argument
    must not leave a half-finished record behind.
    """
    problems: list[str] = []
    declared = {p.name: p for p in capability.inputs}

    for name, spec in declared.items():
        if name not in inputs or inputs[name] is None:
            if spec.required:
                problems.append(f"missing required input {name!r}")
            continue
        value = inputs[name]
        if not _type_ok(spec, value):
            problems.append(
                f"input {name!r} must be {spec.type}, got "
                f"{type(value).__name__}"
            )
            continue
        if spec.pattern:
            try:
                rx = re.compile(spec.pattern)
            except re.error as exc:
                problems.append(f"input {name!r} has an unusable pattern: {exc}")
                continue
            if not rx.search(str(value)):
                # The value itself is never quoted here: it may be sensitive.
                problems.append(
                    f"input {name!r} does not match pattern {spec.pattern!r}"
                )

    unknown = sorted(set(inputs) - set(declared))
    if unknown:
        problems.append(f"unknown input(s): {', '.join(unknown)}")

    return "; ".join(problems) if problems else None


# --------------------------------------------------------------------------
# Output coercion
# --------------------------------------------------------------------------


def _coerce_scalar(kind: str, raw: Any) -> Any:
    if raw is None:
        return None
    if kind == "string":
        return raw if isinstance(raw, str) else json.dumps(raw)
    if kind == "boolean":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().casefold() in {"true", "yes", "y", "1", "on"}
    text = raw if isinstance(raw, str) else str(raw)
    cleaned = re.sub(r"[^0-9eE+\-.]", "", text.strip())
    try:
        if kind == "integer":
            return int(float(cleaned)) if cleaned else None
        if kind == "number":
            return float(cleaned) if cleaned else None
    except ValueError:
        return None
    return raw


def _coerce_object(raw: Any, shape: dict[str, str] | None) -> Any:
    obj = raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = {"value": raw}
    if not isinstance(obj, dict):
        obj = {"value": obj}
    if shape:
        return {k: _coerce_scalar(t, obj.get(k)) for k, t in shape.items()}
    return obj


def _split_list(raw: str) -> list[Any]:
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    if "\n" in raw:
        return [line.strip() for line in raw.splitlines() if line.strip()]
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [raw] if raw.strip() else []


def coerce_output(spec: OutputSpec, raws: list[Any]) -> Any:
    """Turn the raw string(s) a READ produced into the declared type.

    Several READ steps may fill one `array` output -- that is how a capability
    returns a list of sub-accounts -- so the engine accumulates per name and
    coerces once, at the end.
    """
    if not raws:
        return None
    if spec.type == "array":
        items: list[Any] = []
        for raw in raws:
            items.extend(_split_list(raw) if isinstance(raw, str) else [raw])
        item_type = spec.item_type or "string"
        if item_type == "object":
            return [_coerce_object(i, spec.shape) for i in items]
        return [_coerce_scalar(item_type, i) for i in items]
    if spec.type == "object":
        return _coerce_object(raws[-1], spec.shape)
    return _coerce_scalar(spec.type, raws[-1])


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


class _Run:
    """Mutable per-run state. Kept off the module so `replay` is reentrant."""

    def __init__(
        self,
        capability: Capability,
        inputs: dict[str, Any],
        surface: Surface,
        gate: PolicyGate | None,
        recorder: EvidenceRecorder | None,
        telemetry: TelemetryStore | None,
        escalator: Escalator | None,
        dry_run_from: str | None,
    ) -> None:
        self.cap = capability
        self.inputs = inputs
        self.surface = surface
        self.gate = gate
        self.recorder = recorder
        self.escalator = escalator
        self.dry_run_from = dry_run_from

        self.sensitive = {
            p.name for p in capability.inputs if p.sensitivity is not Sensitivity.PUBLIC
        }
        self.outputs_by_name = {o.name: o for o in capability.outputs}
        self.raw_outputs: dict[str, list[Any]] = {}
        self.steps_executed = 0
        self.tier_escalations = 0
        self.resolved_without_acting: list[str] = []

        self.telemetry_store = telemetry
        self.telemetry: CapabilityTelemetry | None = None
        if telemetry is not None:
            tenant = capability.app_profile.tenant_id
            self.telemetry = telemetry.load(capability.id, capability.version, tenant)
            self.telemetry.replays += 1

        if recorder is not None and recorder.redactor is None:
            recorder.attach_redactor(Redactor.for_capability(capability, inputs))

    # -- small helpers -----------------------------------------------------

    @property
    def evidence_dir(self) -> str | None:
        return str(self.recorder.dir) if self.recorder is not None else None

    def note(self, message: str, level: str = "INFO") -> None:
        if self.recorder is not None:
            self.recorder.note(message, level=level)

    def observe(self) -> Observation:
        return self.surface.observe()

    def current_url(self) -> str:
        try:
            return self.observe().url
        except Exception:
            return self.cap.entry_url

    def record_step(self, **kw: Any) -> None:
        if self.recorder is not None:
            self.recorder.step(**kw)

    def record_resolution(self, step: Step, res: ActResult) -> None:
        """Telemetry + drift. Drift is deviation from THIS step's baseline, not
        from tier 1: a hostile surface may legitimately baseline at tier 2."""
        if res.resolved_tier is None:
            return
        tier = int(res.resolved_tier)
        drifted = False
        if self.telemetry is not None:
            drifted = self.telemetry.record_resolution(
                step.id, tier, step.baseline_tier
            )
        elif step.baseline_tier is not None:
            drifted = tier > step.baseline_tier
        if drifted:
            self.tier_escalations += 1
            self.note(
                f"DRIFT step={step.id} resolved at tier {tier}, baseline "
                f"{step.baseline_tier} (candidates={res.candidates})",
                level="WARN",
            )

    # -- outcome builders --------------------------------------------------

    def snapshot(self, label: str) -> str | None:
        try:
            path = self.surface.snapshot(label)
        except Exception:
            return None
        if path and self.recorder is not None:
            try:
                self.recorder.attach(path, label=label)
            except Exception:
                pass
        return path

    def fail(
        self,
        code: str,
        *,
        step: Step | None = None,
        expected: str = "",
        observed: str = "",
        resolved_tier: int | None = None,
        candidates: int | None = None,
        escalated: bool = False,
        capture: bool = True,
    ) -> Failure:
        if capture:
            self.snapshot(f"failure-{code}-{step.id if step else 'run'}")
        if self.telemetry is not None:
            self.telemetry.failures[code] = self.telemetry.failures.get(code, 0) + 1
        self.note(
            f"FAILURE code={code} step={step.id if step else '-'} "
            f"expected={expected!r} observed={observed!r}",
            level="ERROR",
        )
        return Failure(
            capability_id=self.cap.id,
            code=code,
            at_step_id=step.id if step else None,
            step_intent=step.intent if step else None,
            expected=expected,
            observed=observed,
            resolved_tier=resolved_tier,
            candidates=candidates,
            evidence_dir=self.evidence_dir,
            escalated=escalated,
        )

    def business(self, match: SignalMatch, step: Step | None) -> BusinessOutcome:
        if self.telemetry is not None:
            code = match.code
            self.telemetry.business_outcomes[code] = (
                self.telemetry.business_outcomes.get(code, 0) + 1
            )
        self.note(f"BUSINESS OUTCOME {match.code}: {match.message}")
        return BusinessOutcome(
            capability_id=self.cap.id,
            code=match.code,
            message=match.message,
            outputs=self.collect_outputs(),
            at_step_id=step.id if step else None,
            evidence_dir=self.evidence_dir,
        )

    def collect_outputs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, raws in self.raw_outputs.items():
            spec = self.outputs_by_name.get(name)
            out[name] = coerce_output(spec, raws) if spec else (raws[-1] if raws else None)
        return out

    # -- binding -----------------------------------------------------------

    def bind(self, ref: ValueRef | None) -> tuple[str | None, bool, str | None]:
        """Resolve a ValueRef at the moment of action.

        Returns (value, is_sensitive, error). The real value is produced here
        and nowhere else, which is what keeps it out of the artifact, the logs
        and the evidence bundle.
        """
        if ref is None:
            return None, False, None
        if ref.literal is not None:
            return ref.literal, False, None
        name = ref.param or ""
        if name not in self.inputs or self.inputs[name] is None:
            return None, False, f"step references unbound input {name!r}"
        value = self.inputs[name]
        text = "true" if value is True else "false" if value is False else str(value)
        return text, name in self.sensitive, None

    def loggable(self, value: str | None, sensitive: bool) -> str | None:
        return "<redacted>" if (sensitive and value is not None) else value

    # -- gate --------------------------------------------------------------

    def gate_check(self, action: Action, step: Step) -> Failure | None:
        """Deny -> Failure. RequireApproval -> escalator, or Failure.

        A `None` gate means the caller opted out of policy for this run (a
        recorded, hermetic test surface, typically); a configured gate with an
        empty policy still denies everything, because `PolicyGate` fails closed.
        """
        if self.gate is None:
            return None
        decision = self.gate.check(
            action, current_url=self.current_url(), risk=step.risk
        )
        if isinstance(decision, Allow):
            return None
        if isinstance(decision, Deny):
            self.record_step(
                step_id=step.id,
                intent=step.intent,
                action={"type": action.type.value},
                decision=decision,
                outcome="denied",
            )
            return self.fail(
                POLICY_DENIED,
                step=step,
                expected=f"policy to allow {action.type.value} at risk {step.risk.value}",
                observed=f"{decision.code}: {decision.reason}",
                capture=False,
            )
        if isinstance(decision, RequireApproval):
            if self.escalator is None:
                self.record_step(
                    step_id=step.id,
                    intent=step.intent,
                    action={"type": action.type.value},
                    decision=decision,
                    outcome="approval_required",
                )
                return self.fail(
                    APPROVAL_REQUIRED,
                    step=step,
                    expected="a human approval channel for this action",
                    observed=decision.reason,
                    escalated=False,
                    capture=False,
                )
            approved = bool(
                self.escalator.request_approval(
                    capability=self.cap,
                    step=step,
                    reason=decision.reason,
                    observation=self.observe(),
                )
            )
            self.note(
                f"escalated step={step.id} for approval: "
                f"{'granted' if approved else 'refused'}"
            )
            if approved:
                return None
            return self.fail(
                APPROVAL_REQUIRED,
                step=step,
                expected="human approval",
                observed=f"approval refused: {decision.reason}",
                escalated=True,
                capture=False,
            )
        return self.fail(
            POLICY_DENIED,
            step=step,
            expected="a policy decision",
            observed=f"unrecognised decision {decision!r}",
            capture=False,
        )

    # -- action construction ----------------------------------------------

    def build_action(self, step: Step) -> tuple[Action | None, bool, str | None]:
        value, sensitive, err = self.bind(step.value)
        if err:
            return None, False, err
        if step.action is ActionType.NAVIGATE:
            return (
                Action(type=step.action, url=value or self.cap.entry_url, value=value),
                sensitive,
                None,
            )
        return Action(type=step.action, target=step.target, value=value), sensitive, None

    # -- reauth ------------------------------------------------------------

    def reauth(self, step: Step) -> bool:
        """Re-establish the session, then resume from the current step.

        Authentication is a precondition, not steps, so the engine asks the
        surface's session provider rather than replaying a login it was never
        allowed to record.
        """
        provider = getattr(self.surface, "establish_session", None)
        if callable(provider):
            try:
                provider(self.cap.auth)
                self.note(f"re-authenticated via surface session provider (step {step.id})")
                return True
            except Exception as exc:
                self.note(f"session provider failed: {exc!r}", level="ERROR")
                return False
        action = Action(type=ActionType.NAVIGATE, url=self.cap.entry_url)
        if self.gate_check(action, step) is not None:
            return False
        ok = self.surface.act(action).ok
        self.note(f"re-auth by returning to entry_url: {'ok' if ok else 'failed'}")
        return ok


# --------------------------------------------------------------------------
# Step execution
# --------------------------------------------------------------------------


def _checkpoint_expectation(cp: Checkpoint) -> str:
    bits = [cp.description] if cp.description else []
    if cp.text_present:
        bits.append(f"text present {cp.text_present!r}")
    if cp.text_absent:
        bits.append(f"text absent {cp.text_absent!r}")
    if cp.url_matches:
        bits.append(f"url matching {cp.url_matches!r}")
    if cp.locator:
        bits.append(f"control {cp.locator.description!r}")
    return "; ".join(bits)


def _observed_summary(obs: Observation) -> str:
    names = ", ".join(f"{n.role}:{n.name}" for n in obs.nodes[:8])
    return f"url={obs.url!r} title={obs.title!r} controls=[{names}]"


def _execute_step(run: _Run, step: Step) -> ReplayOutcome | None:
    """Run one step to a settled state.

    Returns a terminal outcome, or None meaning "carry on to the next step".
    """
    action, sensitive, bind_error = run.build_action(step)
    if bind_error:
        return run.fail(
            INVALID_INPUT,
            step=step,
            expected="every referenced input to be bound",
            observed=bind_error,
            capture=False,
        )
    assert action is not None

    budget: dict[str, int] = {}
    do_action = True
    last_result: ActResult | None = None
    started = time.monotonic()

    for _ in range(MAX_STEP_ITERATIONS):
        if do_action:
            denial = run.gate_check(action, step)
            if denial is not None:
                return denial

            result = run.surface.act(action)
            last_result = result
            run.record_resolution(step, result)
            run.record_step(
                step_id=step.id,
                intent=step.intent,
                action={
                    "type": action.type.value,
                    "target": step.target.description if step.target else None,
                    "value": run.loggable(action.value, sensitive),
                },
                resolved_tier=result.resolved_tier,
                candidates=result.candidates,
                duration_ms=(time.monotonic() - started) * 1000.0,
                outcome="ok" if result.ok else "act_failed",
                detail=result.detail,
            )
            if result.ok and step.action is ActionType.READ:
                if step.output_name:
                    run.raw_outputs.setdefault(step.output_name, []).append(
                        result.read_value
                    )
        do_action = True

        obs = run.observe()

        # --- the signal scan ------------------------------------------------
        match = scan(run.cap.signals, obs)
        if match is not None:
            if match.is_business:
                return run.business(match, step)
            if match.is_hard:
                return run.fail(
                    match.code,
                    step=step,
                    expected=step.intent,
                    observed=f"{match.message} ({match.matched_on})",
                    resolved_tier=match.resolved_tier,
                    candidates=match.candidates,
                )
            # RECOVERABLE: the caller must never hear about this.
            key = f"{step.id}:{match.code}"
            budget[key] = budget.get(key, 0) + 1
            if budget[key] > match.rule.max_attempts:
                return run.fail(
                    UNRECOVERED_SIGNAL,
                    step=step,
                    expected=(
                        f"recoverable signal {match.code!r} to clear within "
                        f"{match.rule.max_attempts} attempt(s)"
                    ),
                    observed=(
                        f"{match.message} still present after {budget[key] - 1} "
                        f"attempt(s): {match.observed}"
                    ),
                    resolved_tier=match.resolved_tier,
                    candidates=match.candidates,
                )
            handler = match.rule.handler or "retry"
            run.note(
                f"recoverable {match.code!r} at step {step.id}: handler={handler} "
                f"attempt={budget[key]}/{match.rule.max_attempts}"
            )
            if handler == "abort":
                return run.fail(
                    match.code,
                    step=step,
                    expected=step.intent,
                    observed=f"{match.message} (handler=abort)",
                    resolved_tier=match.resolved_tier,
                    candidates=match.candidates,
                )
            if handler == "dismiss":
                node = find_dismiss_control(match, obs)
                if node is None:
                    return run.fail(
                        UNRECOVERED_SIGNAL,
                        step=step,
                        expected=f"a control to dismiss {match.code!r}",
                        observed=f"no continue control found: {match.observed}",
                    )
                dismiss = Action(type=ActionType.CLICK, ref=node.ref)
                denial = run.gate_check(dismiss, step)
                if denial is not None:
                    return denial
                res = run.surface.act(dismiss)
                run.record_step(
                    step_id=step.id,
                    intent=f"dismiss interstitial {match.code!r}",
                    action={"type": "click", "target": node.name},
                    outcome="ok" if res.ok else "dismiss_failed",
                    detail=res.detail,
                )
                if not res.ok:
                    return run.fail(
                        UNRECOVERED_SIGNAL,
                        step=step,
                        expected=f"dismissing {match.code!r} to clear the page",
                        observed=res.detail,
                    )
                # Whether to re-perform the original action depends on whether
                # it actually landed. If it succeeded and the interstitial only
                # appeared afterwards, redoing it would double the effect. If it
                # FAILED because the interstitial was covering the control, the
                # step has not happened yet and must be retried now that the
                # page is clear.
                do_action = last_result is not None and not last_result.ok
                continue
            if handler == "reauth":
                if not run.reauth(step):
                    return run.fail(
                        REAUTH_FAILED,
                        step=step,
                        expected="session re-established",
                        observed=match.message,
                    )
                continue
            # "retry" (and "return", which has no inline meaning here): redo it.
            continue

        # --- the action itself must have worked -----------------------------
        if last_result is not None and not last_result.ok:
            code = (
                LOCATOR_UNRESOLVED
                if "resolve" in last_result.detail.lower()
                else ACTION_FAILED
            )
            return run.fail(
                code,
                step=step,
                expected=step.intent,
                observed=last_result.detail or "surface reported failure",
                resolved_tier=last_result.resolved_tier,
                candidates=last_result.candidates,
            )

        # --- checkpoint ------------------------------------------------------
        if step.post_condition is None:
            run.steps_executed += 1
            return None
        if run.surface.wait_for(step.post_condition, step.post_condition.timeout_ms):
            run.steps_executed += 1
            return None

        # The checkpoint failed and nothing in the scan explained it. Re-scan
        # once against the settled observation before calling it a failure --
        # `wait_for` may have advanced the surface.
        settled = run.observe()
        late = scan(run.cap.signals, settled)
        if late is not None and late is not match:
            do_action = False
            continue
        return run.fail(
            POST_CONDITION_FAILED,
            step=step,
            expected=_checkpoint_expectation(step.post_condition),
            observed=_observed_summary(settled),
            resolved_tier=last_result.resolved_tier if last_result else None,
            candidates=last_result.candidates if last_result else None,
        )

    return run.fail(
        UNRECOVERED_SIGNAL,
        step=step,
        expected="the step to settle",
        observed=f"gave up after {MAX_STEP_ITERATIONS} iterations",
    )


def _dry_run_remaining(run: _Run, steps: list[Step]) -> Failure | None:
    """Resolve, do not act.

    This is how a write capability is verified without creating a second real
    record: everything up to the mutating step really runs, and from there the
    engine only proves each remaining control is reachable.
    """
    for step in steps:
        if step.target is None:
            run.note(f"dry run: step {step.id} has no locator to resolve; skipped")
            continue
        obs = run.observe()
        try:
            res = resolve(step.target, obs)
        except Exception as exc:
            res = None
            reason = f"resolve raised {exc!r}"
        else:
            reason = res.reason
        node = getattr(res, "node", None)
        tier = int(res.tier) if res is not None and res.tier is not None else None
        candidates = res.candidates if res is not None else 0
        run.record_step(
            step_id=step.id,
            intent=step.intent,
            action={"type": step.action.value, "target": step.target.description},
            resolved_tier=tier,
            candidates=candidates,
            outcome="dry_run_resolved" if node is not None else "dry_run_unreachable",
            detail=reason,
        )
        if node is None:
            return run.fail(
                DRY_RUN_UNREACHABLE,
                step=step,
                expected=f"to locate {step.target.description!r} without acting",
                observed=reason,
                resolved_tier=tier,
                candidates=candidates,
            )
        if tier is not None:
            run.record_resolution(step, ActResult(ok=True, resolved_tier=tier, candidates=candidates))
        run.resolved_without_acting.append(step.id)
    return None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def replay(
    capability: Capability,
    inputs: dict[str, Any],
    surface: Surface,
    *,
    gate: PolicyGate | None = None,
    recorder: EvidenceRecorder | None = None,
    telemetry: TelemetryStore | None = None,
    escalator: "Escalator | None" = None,
    dry_run_from: str | None = None,
) -> ReplayOutcome:
    """Execute a reviewed capability deterministically. No model involved."""
    run = _Run(
        capability, inputs, surface, gate, recorder, telemetry, escalator, dry_run_from
    )

    step_ids = [s.id for s in capability.steps]
    if dry_run_from is not None and dry_run_from not in step_ids:
        raise ReplayError(
            f"dry_run_from={dry_run_from!r} is not a step of {capability.id!r}; "
            f"known steps: {step_ids}"
        )

    outcome: ReplayOutcome

    # 1. Validate BEFORE touching the surface.
    problem = validate_inputs(capability, inputs)
    if problem is not None:
        outcome = run.fail(
            INVALID_INPUT,
            expected="inputs matching the declared contract",
            observed=problem,
            capture=False,
        )
        return _finish(run, outcome)

    run.note(
        f"replay {capability.id}@{capability.version} "
        f"({'dry run from ' + dry_run_from if dry_run_from else 'full'})"
    )

    cut = step_ids.index(dry_run_from) if dry_run_from is not None else len(step_ids)

    # 2. Steps.
    for step in capability.steps[:cut]:
        result = _execute_step(run, step)
        if result is not None:
            return _finish(run, result)

    # 3. Dry run tail: resolve without acting, then stop.
    if dry_run_from is not None:
        failure = _dry_run_remaining(run, capability.steps[cut:])
        if failure is not None:
            return _finish(run, failure)
        run.note(
            "dry run complete; terminal checkpoint deliberately NOT asserted "
            "because the terminal state was never reached"
        )
        return _finish(
            run,
            DryRunSuccess(
                capability_id=capability.id,
                outputs=run.collect_outputs(),
                steps_executed=run.steps_executed,
                tier_escalations=run.tier_escalations,
                evidence_dir=run.evidence_dir,
                dry_run_from=dry_run_from,
                resolved_without_acting=list(run.resolved_without_acting),
            ),
        )

    # 4. Never report a success we did not verify.
    if not surface.wait_for(capability.success, capability.success.timeout_ms):
        settled = run.observe()
        late = scan(capability.signals, settled)
        if late is not None and late.is_business:
            return _finish(run, run.business(late, capability.steps[-1] if capability.steps else None))
        return _finish(
            run,
            run.fail(
                SUCCESS_CHECKPOINT_FAILED,
                step=capability.steps[-1] if capability.steps else None,
                expected=_checkpoint_expectation(capability.success),
                observed=_observed_summary(settled),
            ),
        )

    if run.telemetry is not None:
        run.telemetry.successes += 1
    return _finish(
        run,
        Success(
            capability_id=capability.id,
            outputs=run.collect_outputs(),
            steps_executed=run.steps_executed,
            tier_escalations=run.tier_escalations,
            evidence_dir=run.evidence_dir,
        ),
    )


def _finish(run: _Run, outcome: ReplayOutcome) -> ReplayOutcome:
    if run.telemetry_store is not None and run.telemetry is not None:
        run.telemetry_store.save(run.telemetry)
    if run.recorder is not None:
        run.recorder.finish(outcome)
    return outcome


def new_run_id(prefix: str = "replay") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
