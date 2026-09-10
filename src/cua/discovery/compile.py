"""Trajectory -> `Capability`: the compiler.

The discovery loop produces a *narrative* (what the model did, and why). This
module turns it into the *contract* (what an agent may invoke, and what the
replay engine must prove at every step). Three things are synthesized here
that the model is never asked for, because a model asked to invent them will
invent something plausible rather than something true:

* **Locators.** Every acted-on node goes through `build_bundle`, which records
  only strategies it has proven unique against the very observation they came
  from. The tier that actually resolved becomes the step's ``baseline_tier``,
  so drift later is measured against what this app really does, not against an
  idealized tier 1.
* **Post-conditions.** Derived from the observation that FOLLOWED each
  state-changing step -- a URL change, a control that appeared, text that was
  not there before. Discriminators that echo a recorded input value or a piece
  of page *data* are rejected: those would pass verification with the discovery
  input and fail on every other one, which is precisely backwards.
* **Route patterns.** ``/member/12345`` becomes ``/member/:member_id`` wherever
  a recorded input value shows up in a URL, so checkpoints stay true for a
  different member.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from cua.artifact import (
    ActionType,
    AppProfileRef,
    Capability,
    Checkpoint,
    LocatorBundle,
    OutcomeSpec,
    OutputSpec,
    ParamSpec,
    Provenance,
    RiskClass,
    SignalRule,
    Step,
)
from cua.discovery.agent import Trajectory, TrajectoryStep
from cua.locator.record import build_bundle
from cua.locator.resolve import resolve
from cua.surface.base import Node, Observation

__all__ = [
    "compile_trajectory",
    "AppProfile",
    "canonicalize_route",
    "route_regex",
    "CompileError",
]

_RISK_ORDER = {
    RiskClass.READ_ONLY: 0,
    RiskClass.REVERSIBLE: 1,
    RiskClass.IRREVERSIBLE: 2,
}

_STATE_CHANGING = {
    ActionType.CLICK,
    ActionType.TYPE,
    ActionType.SELECT,
    ActionType.PRESS,
    ActionType.NAVIGATE,
}


class CompileError(ValueError):
    """The trajectory cannot become a valid artifact."""


@dataclass
class AppProfile:
    """Per-application knowledge that does not belong to any one capability.

    Signals and outcomes live here because 'the error banner says No records
    located' is a fact about the vendor's product, not about this flow. Keeping
    them in a profile is what makes a differently-worded tenant overlay a data
    change instead of a recompile.
    """

    app_id: str = "unknown"
    vendor_product: str = "unknown"
    product_version: str | None = None
    tenant_id: str | None = None
    signals: list[SignalRule] = field(default_factory=list)
    outcomes: list[OutcomeSpec] = field(default_factory=list)

    def ref(self) -> AppProfileRef:
        return AppProfileRef(
            app_id=self.app_id,
            vendor_product=self.vendor_product,
            product_version=self.product_version,
            tenant_id=self.tenant_id,
        )


# --------------------------------------------------------------------------
# Route canonicalization
# --------------------------------------------------------------------------

_SLOT = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")


def canonicalize_route(url: str, inputs: dict[str, Any]) -> str:
    """Replace recorded input values appearing in ``url`` with ``:param`` slots.

    ``/member/12345`` recorded with ``member_id="12345"`` becomes
    ``/member/:member_id``. Longest values first, so a short value that happens
    to be a substring of a longer one cannot shadow it.
    """
    if not url:
        return url
    items = [
        (str(v), k)
        for k, v in (inputs or {}).items()
        if v is not None and len(str(v)) >= 2
    ]
    items.sort(key=lambda t: len(t[0]), reverse=True)
    out = url
    for value, name in items:
        if value in out:
            out = out.replace(value, f":{name}")
    return out


def route_regex(pattern: str) -> str:
    """Turn a canonical route into an anchored regex whose slots match anything
    that is not a path or query separator."""
    parts: list[str] = []
    last = 0
    for m in _SLOT.finditer(pattern):
        parts.append(re.escape(pattern[last : m.start()]))
        parts.append(r"[^/?&#]+")
        last = m.end()
    parts.append(re.escape(pattern[last:]))
    return "^" + "".join(parts) + "$"


# --------------------------------------------------------------------------
# Discriminator synthesis
# --------------------------------------------------------------------------


def _forbidden_strings(inputs: dict[str, Any], *observations: Observation | None) -> list[str]:
    """Text we must never build a checkpoint out of.

    Two families: the recorded input values (a checkpoint asserting "12345" is
    on the page is a checkpoint that fails for member 67890), and the *data*
    the page displayed -- node values -- which vary per record for exactly the
    same reason. Labels and chrome are fair game; data is not.
    """
    out: list[str] = []
    for v in (inputs or {}).values():
        if v is None:
            continue
        text = str(v).strip()
        if len(text) >= 2:
            out.append(text.casefold())
    for obs in observations:
        if obs is None:
            continue
        for n in obs.nodes:
            if n.value and len(n.value.strip()) >= 2:
                out.append(n.value.strip().casefold())
    return out


def _tainted(text: str, forbidden: Sequence[str]) -> bool:
    low = text.casefold()
    return any(f in low for f in forbidden)


def _sentences(obs: Observation) -> list[str]:
    digest = obs.text_digest or ""
    return [s.strip() for s in re.split(r"[.\n]", digest) if len(s.strip()) >= 4]


def _new_control(before: Observation, after: Observation) -> Node | None:
    seen = {(n.role, n.name) for n in before.nodes}
    for n in after.nodes:
        if (n.role, n.name) not in seen and (n.name or n.text):
            return n
    return None


def _new_text(
    before: Observation, after: Observation, forbidden: Sequence[str]
) -> str | None:
    old = set(_sentences(before)) | {n.name for n in before.nodes if n.name}
    for cand in _sentences(after):
        if cand not in old and not _tainted(cand, forbidden):
            return cand
    for n in after.nodes:
        label = (n.name or n.text or "").strip()
        if label and label not in old and not _tainted(label, forbidden):
            return label
    return None


def synthesize_checkpoint(
    before: Observation | None,
    after: Observation,
    inputs: dict[str, Any],
    *,
    what: str,
) -> Checkpoint:
    """A stable discriminator for "we really did land here".

    Ordered by durability, not by convenience: a URL change is a fact about the
    application's routing; a control appearing is a fact about its structure;
    free text is the weakest of the three and is tried last. There is always a
    fallback, because the schema requires a post_condition on every
    state-changing step and an artifact without one must not exist.
    """
    canonical = canonicalize_route(after.url, inputs)
    forbidden = _forbidden_strings(inputs, before, after)

    if before is not None and before.url != after.url:
        return Checkpoint(
            description=f"{what}: landed on {canonical}",
            url_matches=route_regex(canonical),
        )

    if before is not None:
        node = _new_control(before, after)
        if node is not None:
            try:
                bundle = build_bundle(node, after)
            except ValueError:
                bundle = None
            if bundle is not None:
                return Checkpoint(
                    description=f"{what}: {bundle.description} is present",
                    locator=bundle,
                )
        text = _new_text(before, after, forbidden)
        if text is not None:
            return Checkpoint(
                description=f"{what}: the page now says {text!r}",
                text_present=text,
            )

    # Nothing observably changed (typing into a field, for instance). Assert
    # the weakest true thing rather than inventing a stronger false one.
    return Checkpoint(
        description=f"{what}: still on {canonical}",
        url_matches=route_regex(canonical),
    )


def _success_checkpoint(traj: Trajectory, inputs: dict[str, Any]) -> Checkpoint:
    final = traj.final_observation
    if final is None:
        raise CompileError("trajectory has no final observation to assert on")
    first = traj.steps[0].observation_before if traj.steps else None
    return synthesize_checkpoint(first, final, inputs, what="capability succeeded")


# --------------------------------------------------------------------------
# The compiler
# --------------------------------------------------------------------------


def _bundle_for(step: TrajectoryStep) -> tuple[LocatorBundle | None, int | None]:
    if step.node is None:
        return None, None
    obs = step.observation_before
    bundle = build_bundle(step.node, obs)
    res = resolve(bundle, obs)
    tier = int(res.tier) if res.tier is not None else None
    return bundle, tier


def compile_trajectory(
    traj: Trajectory,
    goal: str,
    *,
    capability_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    profile: AppProfile | None = None,
    app_profile: AppProfileRef | None = None,
    entry_url: str | None = None,
) -> Capability:
    """Compile a discovery trajectory into a reviewable, replayable artifact.

    Raises `CompileError` when the trajectory cannot support one -- an empty
    run, a stuck run, or a node that cannot be located uniquely. Producing a
    plausible-looking artifact from a failed discovery is the one thing this
    function must never do.
    """
    if traj.stop_reason != "finished":
        raise CompileError(
            f"refusing to compile a trajectory that ended in {traj.stop_reason!r}; "
            "only a finished run describes a capability"
        )
    acted = traj.executed_steps()
    if not acted:
        raise CompileError("trajectory contains no successfully executed steps")

    inputs = dict(traj.input_values)
    prof = profile or AppProfile()

    steps: list[Step] = []
    outputs: list[OutputSpec] = []
    used_params: set[str] = set()
    max_risk = RiskClass.READ_ONLY

    for i, ts in enumerate(acted, start=1):
        step_id = f"s{i}"
        assert ts.action_type is not None
        bundle, tier = _bundle_for(ts)

        value = ts.value
        if ts.action_type is ActionType.NAVIGATE:
            from cua.artifact import ValueRef  # noqa: PLC0415

            value = ValueRef(literal=ts.url or traj.entry_url)
        if value is not None and value.param:
            used_params.add(value.param)

        post: Checkpoint | None = None
        if ts.action_type in _STATE_CHANGING:
            after = ts.observation_after or ts.observation_before
            post = synthesize_checkpoint(
                ts.observation_before, after, inputs, what=f"step {step_id}"
            )

        if _RISK_ORDER[ts.risk] > _RISK_ORDER[max_risk]:
            max_risk = ts.risk

        steps.append(
            Step(
                id=step_id,
                intent=ts.why or f"{ts.tool} recorded during discovery",
                action=ts.action_type,
                target=bundle,
                value=value,
                output_name=ts.output_name or None,
                post_condition=post,
                risk=ts.risk,
                baseline_tier=tier,
            )
        )

        if ts.action_type is ActionType.READ and ts.output_name:
            outputs.append(
                OutputSpec(
                    name=ts.output_name,
                    type="string",
                    description=ts.why or f"Value read at {step_id}.",
                    source_step_id=step_id,
                )
            )

    declared_inputs: list[ParamSpec] = [p for p in traj.params if p.name in used_params]

    cap_id = capability_id or f"cap-{traj.run_id}"
    return Capability(
        id=cap_id,
        name=name or re.sub(r"[^a-z0-9]+", "_", goal.casefold()).strip("_")[:60] or cap_id,
        version=version,
        description=description or goal,
        app_profile=app_profile or prof.ref(),
        entry_url=entry_url or traj.entry_url,
        inputs=declared_inputs,
        outputs=outputs,
        steps=steps,
        success=_success_checkpoint(traj, inputs),
        signals=list(prof.signals),
        possible_outcomes=list(prof.outcomes),
        risk_class=max_risk,
        provenance=Provenance(
            recorded_by_model=traj.model,
            discovery_run_id=traj.run_id,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
