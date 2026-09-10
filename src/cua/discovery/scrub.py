"""Strip sensitive input values out of a compiled artifact.

An artifact is a reviewable document that gets committed to version control and
handed to a calling agent. A value the schema itself tags `pii` or `secret` must
not appear in it -- and it can arrive by two very different routes, which need
two different treatments:

1. **Descriptive text** -- the capability description (derived from the goal),
   step intents, checkpoint and locator descriptions. Here the value is only
   prose, so it is replaced with a `<param:name>` placeholder. Nothing breaks.

2. **Locator match values** -- a `SemanticLocator.name` or `TextLocator.text`
   that happens to contain the value. These cannot be rewritten, because the
   string IS the matching key. But a locator that matches on a specific member's
   id was never reusable in the first place: it would fail for every other
   input. So a contaminated strategy is DROPPED, provided a clean strategy
   remains to carry the bundle. That makes this a correctness fix as much as a
   privacy one.

A bundle whose every strategy is contaminated cannot be made safe or reusable,
so compilation fails rather than emitting a capability that only works for the
one record it was recorded against.
"""

from __future__ import annotations

from cua.artifact import (
    Capability,
    Checkpoint,
    LocatorBundle,
    ParamSpec,
    Sensitivity,
    Step,
)

SENSITIVE = {Sensitivity.PII, Sensitivity.SECRET}
#: Below this length a "value" is too short to redact without shredding
#: unrelated text (a member id of "7" would scrub every 7 in the document).
MIN_SCRUBBABLE = 3


class ContaminatedLocator(RuntimeError):
    """Every strategy for a control embeds a sensitive value."""


def _bindings(params: list[ParamSpec], inputs: dict[str, object]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in params:
        if p.sensitivity not in SENSITIVE:
            continue
        raw = inputs.get(p.name)
        if raw is None:
            continue
        text = str(raw)
        if len(text) >= MIN_SCRUBBABLE:
            out[p.name] = text
    return out


def scrub_text(text: str | None, bindings: dict[str, str]) -> str | None:
    if not text:
        return text
    for name, value in bindings.items():
        text = text.replace(value, f"<param:{name}>")
    return text


def _contaminated(strategy, bindings: dict[str, str]) -> bool:
    """Does this strategy MATCH on a sensitive value?"""
    for field in ("name", "text", "anchor_text"):
        val = getattr(strategy, field, None)
        if isinstance(val, str) and any(v in val for v in bindings.values()):
            return True
    return False


def _clean_bundle(bundle: LocatorBundle, bindings: dict[str, str], where: str) -> LocatorBundle:
    kept = [s for s in bundle.strategies if not _contaminated(s, bindings)]
    if not kept:
        raise ContaminatedLocator(
            f"{where}: every locator strategy for {bundle.description!r} matches "
            "on a sensitive input value, so the control can only ever be found "
            "for the record it was recorded against"
        )
    scope = bundle.scope_text
    if scope and any(v in scope for v in bindings.values()):
        scope = None  # a row scoped by one member's id is not reusable either
    return bundle.model_copy(
        update={
            "description": scrub_text(bundle.description, bindings),
            "strategies": kept,
            "scope_text": scope,
        }
    )


def _clean_checkpoint(cp: Checkpoint | None, bindings: dict[str, str], where: str):
    if cp is None:
        return None
    update: dict[str, object] = {"description": scrub_text(cp.description, bindings)}
    if cp.locator is not None:
        update["locator"] = _clean_bundle(cp.locator, bindings, where)
    # A checkpoint asserting a specific member's text is not a checkpoint, it is
    # a coincidence. Drop it rather than ship a capability that passes for one
    # input and fails for every other.
    for field in ("text_present", "text_absent"):
        val = getattr(cp, field)
        if isinstance(val, str) and any(v in val for v in bindings.values()):
            update[field] = None
    return cp.model_copy(update=update)


def _clean_step(step: Step, bindings: dict[str, str]) -> Step:
    where = f"step {step.id}"
    update: dict[str, object] = {"intent": scrub_text(step.intent, bindings)}
    if step.target is not None:
        update["target"] = _clean_bundle(step.target, bindings, where)
    update["post_condition"] = _clean_checkpoint(step.post_condition, bindings, where)
    return step.model_copy(update=update)


def scrub_capability(cap: Capability, inputs: dict[str, object]) -> Capability:
    """Return a copy of `cap` with no sensitive input value anywhere in it.

    Applied at compile time so an artifact is clean from birth -- there is no
    window in which a contaminated capability exists on disk.
    """
    bindings = _bindings(cap.inputs, inputs)
    if not bindings:
        return cap

    return cap.model_copy(
        update={
            "description": scrub_text(cap.description, bindings),
            "steps": [_clean_step(s, bindings) for s in cap.steps],
            "success": _clean_checkpoint(cap.success, bindings, "success checkpoint"),
        }
    )
