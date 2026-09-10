"""Verify before save: an unverified artifact never reaches disk.

This is the mechanism the whole product claim rests on. A model wrote the
trajectory; a compiler guessed the locators and the checkpoints. Neither is
trusted. Before a `Capability` becomes a file, it is replayed *by the
deterministic engine, with no model anywhere in the loop*, against a FRESH
surface.

Two verification modes, chosen by risk, and the choice matters:

* ``read_only`` -> **full_replay with a DIFFERENT input than discovery used.**
  Replaying with the same inputs would pass even if the compiler baked a
  literal in where a parameter belongs; all it would prove is that the flow
  runs. Changing the input is what proves it is *parameterized*. We also
  replay the discovery input and require the outputs to differ, because a
  capability that returns the same record for two different members is reading
  something that is not the record.

* ``reversible`` / ``irreversible`` -> **dry_run from the first unsafe step.**
  Everything up to the mutation really executes; from there the engine only
  proves each remaining control is reachable. Verifying "open a sub-account"
  by replaying it would open a second sub-account.

On failure: nothing is written and ``None`` is returned. There is no partial
write path, because there is no code that writes before the outcome is known.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any, Callable

import yaml

from cua.artifact import Capability, ParamSpec, RiskClass, Sensitivity
from cua.evidence.recorder import EvidenceRecorder
from cua.outcomes import Success
from cua.policy.gate import PolicyGate
from cua.replay.engine import DryRunSuccess, replay
from cua.surface.base import Surface

__all__ = [
    "verify_and_save",
    "verify",
    "VerificationReport",
    "derive_different_inputs",
    "dump_capability_yaml",
]

SurfaceFactory = Callable[..., Surface]


class VerificationReport:
    """Why an artifact was (or was not) allowed to exist."""

    def __init__(
        self,
        *,
        ok: bool,
        mode: str,
        reason: str = "",
        inputs: dict[str, Any] | None = None,
        outcomes: list[Any] | None = None,
    ) -> None:
        self.ok = ok
        self.mode = mode
        self.reason = reason
        self.inputs = inputs or {}
        self.outcomes = outcomes or []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<VerificationReport ok={self.ok} mode={self.mode} reason={self.reason!r}>"


# --------------------------------------------------------------------------
# Deriving a genuinely different input
# --------------------------------------------------------------------------


def _shift_digits(text: str) -> str:
    shifted = "".join(str((int(c) + 1) % 10) if c.isdigit() else c for c in text)
    return shifted


def _different_value(spec: ParamSpec, value: Any) -> Any | None:
    if spec.type == "boolean":
        return not bool(value)
    if spec.type == "integer":
        return int(value) + 1
    if spec.type == "number":
        return float(value) + 1.0
    text = str(value)
    for candidate in (_shift_digits(text), text + "X", text.upper()):
        if candidate != text:
            return candidate
    return None


def derive_different_inputs(
    cap: Capability, discovery_inputs: dict[str, Any]
) -> dict[str, Any] | None:
    """A second, distinct input set of the same shape.

    Returns ``None`` when no declared input could be varied -- in which case we
    cannot prove parameterization and must not pretend we did.
    """
    declared = {p.name: p for p in cap.inputs}
    if not declared:
        return {}
    out: dict[str, Any] = {}
    varied = False
    for name, spec in declared.items():
        original = discovery_inputs.get(name)
        alt = _different_value(spec, original) if original is not None else None
        if alt is not None and alt != original:
            import re  # noqa: PLC0415

            if spec.pattern and not re.search(spec.pattern, str(alt)):
                out[name] = original
                continue
            out[name] = alt
            varied = True
        else:
            out[name] = original
    return out if varied else None


def _filtered(cap: Capability, values: dict[str, Any]) -> dict[str, Any]:
    """`validate_inputs` rejects unknown keys, so hand it only what is declared."""
    declared = {p.name for p in cap.inputs}
    return {k: v for k, v in (values or {}).items() if k in declared}


def _make_surface(factory: SurfaceFactory, inputs: dict[str, Any]) -> Surface:
    """A factory may take the inputs (so a recorded fixture can serve the right
    record) or take nothing at all."""
    try:
        sig = inspect.signature(factory)
        takes_arg = len(sig.parameters) > 0
    except (TypeError, ValueError):  # pragma: no cover - builtins/callables
        takes_arg = False
    return factory(inputs) if takes_arg else factory()


def _first_unsafe_step_id(cap: Capability) -> str | None:
    for step in cap.steps:
        if step.risk is not RiskClass.READ_ONLY:
            return step.id
    return None


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify(
    cap: Capability,
    surface_factory: SurfaceFactory,
    *,
    discovery_inputs: dict[str, Any],
    verify_inputs: dict[str, Any] | None = None,
    gate: PolicyGate | None = None,
    recorder: EvidenceRecorder | None = None,
    require_outputs_differ: bool = True,
) -> VerificationReport:
    """Replay the artifact deterministically. No LLM is involved, by design."""
    discovery_inputs = dict(discovery_inputs or {})

    if cap.risk_class is RiskClass.READ_ONLY:
        alt = verify_inputs if verify_inputs is not None else derive_different_inputs(
            cap, discovery_inputs
        )
        if alt is None:
            return VerificationReport(
                ok=False,
                mode="full_replay",
                reason=(
                    "cannot derive an input different from the one discovery "
                    "used, so parameterization cannot be proven; refusing to "
                    "verify with the discovery input"
                ),
            )
        alt = _filtered(cap, alt)
        base = _filtered(cap, discovery_inputs)
        if cap.inputs and alt == base:
            return VerificationReport(
                ok=False,
                mode="full_replay",
                reason="verification inputs are identical to discovery inputs",
            )

        outcomes: list[Any] = []
        second = replay(
            cap, alt, _make_surface(surface_factory, alt), gate=gate, recorder=recorder
        )
        outcomes.append(second)
        if not isinstance(second, Success):
            return VerificationReport(
                ok=False,
                mode="full_replay",
                reason=f"replay with a different input did not succeed: {second!r}",
                inputs=alt,
                outcomes=outcomes,
            )

        if cap.outputs and require_outputs_differ:
            first = replay(
                cap, base, _make_surface(surface_factory, base), gate=gate
            )
            outcomes.append(first)
            if not isinstance(first, Success):
                return VerificationReport(
                    ok=False,
                    mode="full_replay",
                    reason=f"replay with the discovery input did not succeed: {first!r}",
                    inputs=alt,
                    outcomes=outcomes,
                )
            if first.outputs == second.outputs:
                return VerificationReport(
                    ok=False,
                    mode="full_replay",
                    reason=(
                        "two different inputs produced identical outputs; the "
                        "capability is reading something that does not depend "
                        "on its parameters (a literal was probably baked in "
                        "where a parameter belongs)"
                    ),
                    inputs=alt,
                    outcomes=outcomes,
                )
        return VerificationReport(ok=True, mode="full_replay", inputs=alt, outcomes=outcomes)

    # Write capability: prove reachability, never consequence.
    cut = _first_unsafe_step_id(cap)
    if cut is None:
        return VerificationReport(
            ok=False,
            mode="dry_run",
            reason=(
                f"risk_class is {cap.risk_class.value!r} but no step is marked "
                "unsafe; the artifact contradicts itself"
            ),
        )
    alt = verify_inputs if verify_inputs is not None else derive_different_inputs(
        cap, discovery_inputs
    )
    if alt is None:
        alt = dict(discovery_inputs)
    alt = _filtered(cap, alt)
    outcome = replay(
        cap,
        alt,
        _make_surface(surface_factory, alt),
        gate=gate,
        recorder=recorder,
        dry_run_from=cut,
    )
    if not isinstance(outcome, DryRunSuccess):
        return VerificationReport(
            ok=False,
            mode="dry_run",
            reason=f"dry run from {cut!r} did not succeed: {outcome!r}",
            inputs=alt,
            outcomes=[outcome],
        )
    return VerificationReport(ok=True, mode="dry_run", inputs=alt, outcomes=[outcome])


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def dump_capability_yaml(cap: Capability) -> str:
    """Stable, sorted YAML so two artifacts diff cleanly in review."""
    return yaml.safe_dump(
        cap.model_dump(mode="json"),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


def verify_and_save(
    cap: Capability,
    surface_factory: SurfaceFactory,
    out_path: str | Path,
    *,
    discovery_inputs: dict[str, Any],
    verify_inputs: dict[str, Any] | None = None,
    gate: PolicyGate | None = None,
    recorder: EvidenceRecorder | None = None,
    require_outputs_differ: bool = True,
    on_report: "Callable[[VerificationReport], None] | None" = None,
) -> Path | None:
    """Verify, then write. Returns the path, or ``None`` if nothing was written.

    The ordering is the entire point: there is no code path that writes before
    `verify` has returned ``ok``.
    """
    try:
        report = verify(
            cap,
            surface_factory,
            discovery_inputs=discovery_inputs,
            verify_inputs=verify_inputs,
            gate=gate,
            recorder=recorder,
            require_outputs_differ=require_outputs_differ,
        )
    except Exception as exc:  # a crash during verification is a failed verification
        report = VerificationReport(ok=False, mode="none", reason=f"verification raised {exc!r}")

    if on_report is not None:
        on_report(report)

    if not report.ok:
        if recorder is not None:
            recorder.note(
                f"VERIFICATION FAILED ({report.mode}): {report.reason}; nothing written",
                level="ERROR",
            )
        return None

    sensitive = {p.name for p in cap.inputs if p.sensitivity is not Sensitivity.PUBLIC}
    cap.provenance.verified_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cap.provenance.verification_mode = report.mode  # type: ignore[assignment]
    cap.provenance.verified_with_inputs = {
        k: str(v) for k, v in report.inputs.items() if k not in sensitive
    }

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_capability_yaml(cap), encoding="utf-8")
    if recorder is not None:
        recorder.note(f"verified ({report.mode}) and saved to {path}")
    return path
