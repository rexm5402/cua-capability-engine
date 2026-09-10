"""The LLM-driven discovery agent: observe -> decide -> act, once per capability.

This is the only place in the system where a model touches a surface. Three
rules make that safe enough to ship:

1.  **Tool calls, never free-form JSON.** The model chooses from a fixed,
    schema-checked vocabulary. A malformed plan is a rejected tool call, not a
    parse error halfway through a mutation.
2.  **Late binding, enforced at the boundary.** ``type_text``/``select`` accept
    a *reference* to a declared input parameter, never a value. The real value
    is substituted by this engine at the moment of action, so a member's SSN
    never enters the model transcript. A model that emits a raw value anyway
    has its tool call REJECTED and fed back; the surface never sees it.
3.  **Every action passes the policy gate first.** A denied action becomes an
    observation ("that action is not permitted"), not a side effect.

The accessibility tree is the primary signal *by design*: the loop is fully
functional against a vision-less model, and screenshots are a disambiguation
aid requested only when ``llm.supports_vision`` is true.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from cua.artifact import ActionType, ParamSpec, RiskClass, Sensitivity, ValueRef
from cua.evidence.recorder import EvidenceRecorder
from cua.llm.client import LLMClient, ToolCall
from cua.policy.gate import Allow, Deny, PolicyGate, RequireApproval
from cua.surface.base import Action, ActResult, Node, Observation, Surface

__all__ = [
    "discover",
    "Trajectory",
    "TrajectoryStep",
    "StopReason",
    "tool_definitions",
    "parse_value_ref",
    "risk_of",
    "infer_params",
    "LITERAL_PREFIX",
]

LITERAL_PREFIX = "literal:"

#: Verbs that make a control's activation impossible to undo. Conservative on
#: purpose: mis-classifying a read as a write costs a dry run, the reverse
#: costs a real record.
IRREVERSIBLE_WORDS = frozenset(
    {
        "delete",
        "remove",
        "terminate",
        "transfer",
        "pay",
        "purchase",
        "submit",
        "send",
        "confirm",
        "finalize",
        "post",
        "issue",
        "void",
    }
)

#: Verbs that change state but can be walked back.
REVERSIBLE_WORDS = frozenset(
    {
        "save",
        "create",
        "add",
        "update",
        "edit",
        "apply",
        "assign",
        "enable",
        "disable",
        "activate",
        "deactivate",
        "open",
    }
)

#: Parameter-name fragments that imply the value must never be transcribed.
_SECRET_HINTS = ("password", "secret", "token", "pin", "api_key", "apikey")
_PII_HINTS = ("ssn", "social", "dob", "birth", "member_id", "account", "email", "phone")


class StopReason:
    FINISHED = "finished"
    STUCK = "stuck"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    NO_PROGRESS = "no_progress"
    DEAD_END = "dead_end"
    NO_TOOL_CALL = "no_tool_call"


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    """One turn of the loop, with everything the compiler will need.

    Rejected and denied turns are kept: an artifact reviewer should be able to
    see what the model *tried* to do, not merely what it was allowed to do.
    """

    index: int
    tool: str
    why: str
    observation_before: Observation
    arguments: dict[str, Any] = field(default_factory=dict)
    ref: str | None = None
    node: Node | None = None
    value: ValueRef | None = None
    output_name: str | None = None
    key: str | None = None
    url: str | None = None
    action_type: ActionType | None = None
    risk: RiskClass = RiskClass.READ_ONLY
    result: ActResult | None = None
    observation_after: Observation | None = None
    read_value: str | None = None
    rationale: str = ""
    screenshot_path: str | None = None
    rejected: bool = False
    rejection: str | None = None
    denied: bool = False
    denial: str | None = None

    @property
    def executed(self) -> bool:
        return (
            not self.rejected
            and not self.denied
            and self.result is not None
            and self.result.ok
        )


@dataclass
class Trajectory:
    goal: str
    entry_url: str
    run_id: str
    model: str
    params: list[ParamSpec] = field(default_factory=list)
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_observation: Observation | None = None
    stop_reason: str = StopReason.MAX_STEPS
    summary: str = ""
    stuck_reason: str | None = None
    escalate: bool = False
    input_values: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == StopReason.FINISHED

    def executed_steps(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.executed]

    def param_names(self) -> list[str]:
        return [p.name for p in self.params]


# --------------------------------------------------------------------------
# Tool vocabulary
# --------------------------------------------------------------------------


def _value_ref_schema(param_names: Sequence[str]) -> dict[str, Any]:
    """Late binding, expressed in the schema itself rather than in prose.

    A model is far more likely to obey a constraint the tool definition can
    state than one buried in a system prompt, so ``value_ref`` is an enum of
    declared parameter names plus an explicit escape hatch for genuinely
    non-sensitive constants.
    """
    options: list[dict[str, Any]] = []
    if param_names:
        options.append({"type": "string", "enum": list(param_names)})
    options.append(
        {
            "type": "string",
            "pattern": f"^{LITERAL_PREFIX}.+",
            "description": (
                "Only for a non-sensitive constant that is part of the flow "
                "itself, e.g. 'literal:Active' for a dropdown option."
            ),
        }
    )
    return {
        "type": "string",
        "anyOf": options,
        "description": (
            "A REFERENCE, never a value. Use one of the declared parameter "
            f"names ({', '.join(param_names) or 'none declared'}) so the engine "
            "can substitute the real value at execution time, or "
            f"'{LITERAL_PREFIX}<constant>' for a fixed, non-sensitive option. "
            "Supplying an actual data value here will be rejected."
        ),
    }


def _fn(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        },
    }


_WHY = {
    "type": "string",
    "description": (
        "Why this step exists, in one sentence of plain prose. This becomes "
        "the step's permanent intent and is what a human reviewer reads when "
        "approving the capability."
    ),
}


def tool_definitions(param_names: Sequence[str]) -> list[dict[str, Any]]:
    ref = {"type": "string", "description": "The [ref] of a control from CONTROLS."}
    vr = _value_ref_schema(param_names)
    return [
        _fn(
            "click",
            "Activate a control (button, link, tab, checkbox).",
            {"ref": ref, "why": _WHY},
            ["ref", "why"],
        ),
        _fn(
            "type_text",
            "Type into a text field. The value is late-bound by reference.",
            {"ref": ref, "value_ref": vr, "why": _WHY},
            ["ref", "value_ref", "why"],
        ),
        _fn(
            "select",
            "Choose an option in a dropdown/listbox. Late-bound by reference.",
            {"ref": ref, "value_ref": vr, "why": _WHY},
            ["ref", "value_ref", "why"],
        ),
        _fn(
            "press",
            "Send a key to a control, e.g. Enter or Tab.",
            {"ref": ref, "key": {"type": "string"}, "why": _WHY},
            ["ref", "key", "why"],
        ),
        _fn(
            "navigate",
            "Go directly to a URL within the application.",
            {"url": {"type": "string"}, "why": _WHY},
            ["url", "why"],
        ),
        _fn(
            "read_value",
            "Read a value off the page and declare it as a named output of "
            "this capability.",
            {
                "ref": ref,
                "output_name": {
                    "type": "string",
                    "description": "snake_case name for the declared output.",
                },
                "why": _WHY,
            },
            ["ref", "output_name", "why"],
        ),
        _fn(
            "finish",
            "The goal has been achieved. Call this once, last.",
            {"summary": {"type": "string"}},
            ["summary"],
        ),
        _fn(
            "stuck",
            "The goal cannot be achieved from here. Ends the run and escalates "
            "to a human. Preferable to guessing.",
            {"reason": {"type": "string"}},
            ["reason"],
        ),
    ]


SYSTEM_PROMPT = """\
You are recording a REUSABLE capability by driving a real application through \
an accessibility tree. You are not doing a one-off task: everything you do is \
compiled into a parameterized artifact that will be replayed later, \
deterministically, with DIFFERENT input values and no model in the loop.

Rules, in order of importance:

1. Act only through the provided tools. Every call needs a `why` -- one \
sentence of prose that becomes the permanent, human-reviewed intent of the step.
2. NEVER put a data value in `value_ref`. Pass the NAME of a declared input \
parameter; the engine substitutes the real value at the moment of action. \
Sensitive values must never appear in this conversation. Use \
'literal:<constant>' only for a fixed, non-sensitive part of the flow such as \
a dropdown option that never varies between runs.
3. Address controls by the [ref] shown in CONTROLS. Do not invent refs.
4. Prefer the shortest path that a later replay could follow reliably.
5. When the goal is achieved, call `finish`. If it cannot be achieved -- a \
control is missing, the page is a dead end, you would have to guess -- call \
`stuck`. Guessing produces an artifact that fails in production; `stuck` \
produces a human review. `stuck` is the better outcome.
"""


# --------------------------------------------------------------------------
# Late binding enforcement
# --------------------------------------------------------------------------


def _looks_like_value(raw: str, input_values: dict[str, Any]) -> str | None:
    """Return the parameter whose real value ``raw`` reproduces, if any."""
    probe = raw.strip().casefold()
    if not probe:
        return None
    for name, value in input_values.items():
        if value is None:
            continue
        text = str(value).strip().casefold()
        if text and (text == probe or text in probe):
            return name
    return None


def parse_value_ref(
    raw: Any,
    declared: Sequence[str],
    input_values: dict[str, Any] | None = None,
    sensitive: Iterable[str] = (),
) -> tuple[ValueRef | None, str | None]:
    """Turn a model-supplied ``value_ref`` into a `ValueRef`, or refuse.

    Returns ``(value_ref, None)`` on success and ``(None, reason)`` on
    rejection. A rejection is fed back to the model and the action is NOT
    performed -- which is the whole point: the surface must never receive a
    value the model typed out of its own context.
    """
    values = input_values or {}
    sensitive_names = set(sensitive)

    if not isinstance(raw, str) or not raw:
        return None, (
            "value_ref must be a string naming a declared input parameter "
            f"(one of: {', '.join(declared) or 'none'}) or "
            f"'{LITERAL_PREFIX}<constant>'."
        )

    if raw in declared:
        return ValueRef(param=raw), None

    if raw.startswith(LITERAL_PREFIX):
        literal = raw[len(LITERAL_PREFIX) :]
        if not literal:
            return None, f"'{LITERAL_PREFIX}' needs a constant after the colon."
        hit = _looks_like_value(literal, values)
        if hit is not None:
            kind = "sensitive " if hit in sensitive_names else ""
            return None, (
                f"REJECTED: that literal reproduces the {kind}value of the "
                f"declared input {hit!r}. A value that varies between runs is a "
                f"parameter, not a constant -- pass value_ref={hit!r}."
            )
        return ValueRef(literal=literal), None

    hit = _looks_like_value(raw, values)
    if hit is not None:
        kind = "SENSITIVE " if hit in sensitive_names else ""
        return None, (
            f"REJECTED: you supplied a raw {kind}data value instead of a "
            f"reference. The action was NOT performed and the value was "
            f"discarded. Pass value_ref={hit!r} so the engine binds the real "
            "value at execution time."
        )

    return None, (
        f"REJECTED: value_ref={raw!r} is not a declared input parameter "
        f"(declared: {', '.join(declared) or 'none'}) and does not start with "
        f"'{LITERAL_PREFIX}'. The action was NOT performed. Raw data values are "
        "never accepted here."
    )


# --------------------------------------------------------------------------
# Risk inference
# --------------------------------------------------------------------------


def risk_of(action: ActionType, node: Node | None) -> RiskClass:
    """Classify a step's blast radius from the control it touches.

    Navigation and reads are read-only. Everything else is judged by the verb
    printed on the control, because that verb is the only thing the application
    tells us about consequence.
    """
    if action in (ActionType.NAVIGATE, ActionType.READ, ActionType.WAIT):
        return RiskClass.READ_ONLY
    label = " ".join(
        part for part in ((node.name if node else ""), (node.text if node else "")) if part
    ).casefold()
    words = {w.strip(".,:;!?-_/") for w in label.split()}
    if words & IRREVERSIBLE_WORDS:
        return RiskClass.IRREVERSIBLE
    if words & REVERSIBLE_WORDS:
        return RiskClass.REVERSIBLE
    return RiskClass.READ_ONLY


def infer_params(inputs: dict[str, Any] | None) -> list[ParamSpec]:
    """Best-effort ParamSpecs when the caller supplies only raw inputs.

    Sensitivity defaults *up*, not down: an unrecognised name that carries a
    value we cannot classify is still described honestly rather than declared
    public by omission.
    """
    out: list[ParamSpec] = []
    for name, value in (inputs or {}).items():
        low = name.casefold()
        if any(h in low for h in _SECRET_HINTS):
            sens = Sensitivity.SECRET
        elif any(h in low for h in _PII_HINTS):
            sens = Sensitivity.PII
        else:
            sens = Sensitivity.PUBLIC
        if isinstance(value, bool):
            kind = "boolean"
        elif isinstance(value, int):
            kind = "integer"
        elif isinstance(value, float):
            kind = "number"
        else:
            kind = "string"
        out.append(
            ParamSpec(
                name=name,
                type=kind,  # type: ignore[arg-type]
                description=f"Input {name!r}, supplied by the calling agent.",
                sensitivity=sens,
            )
        )
    return out


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def _state_key(obs: Observation) -> tuple[Any, ...]:
    return (
        obs.url,
        obs.title,
        tuple((n.ref, n.role, n.name, n.value, n.enabled) for n in obs.nodes),
    )


def _find_node(obs: Observation, ref: str | None) -> Node | None:
    if not ref:
        return None
    for n in obs.nodes:
        if n.ref == ref:
            return n
    return None


_TOOL_TO_ACTION = {
    "click": ActionType.CLICK,
    "type_text": ActionType.TYPE,
    "select": ActionType.SELECT,
    "press": ActionType.PRESS,
    "navigate": ActionType.NAVIGATE,
    "read_value": ActionType.READ,
}


def _history_block(steps: list[TrajectoryStep], limit: int = 8) -> str:
    if not steps:
        return "(nothing yet)"
    lines = []
    for s in steps[-limit:]:
        status = (
            "REJECTED"
            if s.rejected
            else "DENIED"
            if s.denied
            else ("ok" if s.executed else "failed")
        )
        lines.append(f"  {s.index}. {s.tool}({_arg_digest(s)}) -> {status}")
    return "\n".join(lines)


def _arg_digest(step: TrajectoryStep) -> str:
    bits = []
    if step.ref:
        bits.append(step.ref)
    if step.value is not None:
        bits.append(
            f"param={step.value.param}" if step.value.param else f"literal={step.value.literal!r}"
        )
    if step.output_name:
        bits.append(f"->{step.output_name}")
    if step.url:
        bits.append(step.url)
    if step.key:
        bits.append(f"key={step.key}")
    return ", ".join(bits)


def discover(
    goal: str,
    entry_url: str,
    surface: Surface,
    *,
    llm: LLMClient,
    inputs: dict[str, Any] | None = None,
    gate: PolicyGate | None = None,
    recorder: EvidenceRecorder | None = None,
    max_steps: int = 25,
    timeout_s: int = 300,
    params: Sequence[ParamSpec] | None = None,
    no_progress_limit: int = 3,
    dead_end_limit: int = 3,
    run_id: str | None = None,
) -> Trajectory:
    """Drive ``surface`` toward ``goal`` and return what happened.

    The loop never returns a partial artifact -- only a `Trajectory`. Turning
    that into a `Capability` is `cua.discovery.compile`'s job, and proving the
    result actually works is `cua.discovery.verify`'s.
    """
    input_values = dict(inputs or {})
    declared = list(params) if params is not None else infer_params(input_values)
    declared_names = [p.name for p in declared]
    sensitive = {p.name for p in declared if p.sensitivity is not Sensitivity.PUBLIC}

    traj = Trajectory(
        goal=goal,
        entry_url=entry_url,
        run_id=run_id or f"discovery-{uuid.uuid4().hex[:12]}",
        model=getattr(llm, "model", "unknown"),
        params=declared,
        input_values=input_values,
    )

    def note(msg: str, level: str = "INFO") -> None:
        if recorder is not None:
            recorder.note(msg, level=level)

    tools = tool_definitions(declared_names)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"GOAL: {goal}\n"
                f"ENTRY URL: {entry_url}\n"
                f"DECLARED INPUT PARAMETERS (names only -- values are withheld "
                f"deliberately): {', '.join(declared_names) or '(none)'}\n"
                "Drive the application to achieve the goal, then call finish."
            ),
        }
    ]

    started = time.monotonic()
    stale = 0
    consecutive_failures = 0
    empty_replies = 0
    obs = surface.observe()

    for turn in range(1, max_steps + 1):
        if timeout_s is not None and (time.monotonic() - started) >= timeout_s:
            traj.stop_reason = StopReason.TIMEOUT
            note(f"discovery timed out after {timeout_s}s", level="WARN")
            break

        obs = surface.observe()
        shot: str | None = None
        if getattr(llm, "supports_vision", False):
            try:
                shot = surface.snapshot(f"discovery-turn-{turn}")
            except Exception:
                shot = None

        content = (
            f"STEP {turn} of at most {max_steps}.\n\n"
            f"{obs.render()}\n\n"
            f"ACTIONS SO FAR:\n{_history_block(traj.steps)}\n\n"
            f"DECLARED INPUT PARAMETERS: {', '.join(declared_names) or '(none)'}"
        )
        if shot:
            content += f"\n\nSCREENSHOT: {shot}"
        messages.append({"role": "user", "content": content})

        response = llm.complete(system=SYSTEM_PROMPT, messages=messages, tools=tools)

        if not response.tool_calls:
            empty_replies += 1
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You must respond with a tool call. Prose is not an "
                        "action. If you cannot proceed, call stuck."
                    ),
                }
            )
            if empty_replies >= 2:
                traj.stop_reason = StopReason.NO_TOOL_CALL
                break
            continue
        empty_replies = 0

        call: ToolCall = response.tool_calls[0]
        args = dict(call.arguments or {})
        why = str(args.get("why") or args.get("summary") or args.get("reason") or "")

        step = TrajectoryStep(
            index=turn,
            tool=call.name,
            why=why,
            arguments=args,
            observation_before=obs,
            rationale=response.text or why,
            screenshot_path=shot,
        )

        # -- terminal tools ------------------------------------------------
        if call.name == "finish":
            traj.stop_reason = StopReason.FINISHED
            traj.summary = str(args.get("summary") or "")
            note(f"discovery finished: {traj.summary}")
            break
        if call.name == "stuck":
            traj.stop_reason = StopReason.STUCK
            traj.stuck_reason = str(args.get("reason") or "")
            traj.escalate = True
            traj.steps.append(step)
            note(f"discovery STUCK, escalating: {traj.stuck_reason}", level="WARN")
            break

        action_type = _TOOL_TO_ACTION.get(call.name)
        if action_type is None:
            step.rejected = True
            step.rejection = f"unknown tool {call.name!r}"
            traj.steps.append(step)
            messages.append({"role": "user", "content": step.rejection})
            continue
        step.action_type = action_type

        # -- argument validation, including late binding -------------------
        node: Node | None = None
        if action_type is not ActionType.NAVIGATE:
            step.ref = str(args.get("ref") or "")
            node = _find_node(obs, step.ref)
            if node is None:
                step.rejected = True
                step.rejection = (
                    f"REJECTED: ref {step.ref!r} is not a control in the current "
                    "observation. Use a [ref] exactly as printed in CONTROLS."
                )
        else:
            step.url = str(args.get("url") or "")
            if not step.url:
                step.rejected = True
                step.rejection = "REJECTED: navigate requires a url."

        if not step.rejected and action_type in (ActionType.TYPE, ActionType.SELECT):
            vref, problem = parse_value_ref(
                args.get("value_ref"), declared_names, input_values, sensitive
            )
            if problem is not None:
                step.rejected = True
                step.rejection = problem
            else:
                step.value = vref

        if not step.rejected and action_type is ActionType.PRESS:
            step.key = str(args.get("key") or "")
            if not step.key:
                step.rejected = True
                step.rejection = "REJECTED: press requires a key."

        if not step.rejected and action_type is ActionType.READ:
            step.output_name = str(args.get("output_name") or "")
            if not step.output_name:
                step.rejected = True
                step.rejection = "REJECTED: read_value requires an output_name."

        if not step.rejected and not why:
            step.rejected = True
            step.rejection = (
                "REJECTED: every action needs a `why`; it becomes the step's "
                "permanent intent."
            )

        if step.rejected:
            traj.steps.append(step)
            note(f"rejected tool call {call.name}: {step.rejection}", level="WARN")
            messages.append({"role": "assistant", "content": f"{call.name}({_arg_digest(step)})"})
            messages.append({"role": "user", "content": str(step.rejection)})
            stale += 1
            if stale >= no_progress_limit:
                traj.stop_reason = StopReason.NO_PROGRESS
                break
            continue

        step.node = node
        step.risk = risk_of(action_type, node)

        # -- bind the value HERE, and nowhere earlier ----------------------
        bound: str | None = None
        if step.value is not None:
            bound = (
                str(input_values.get(step.value.param))
                if step.value.param is not None
                else step.value.literal
            )

        action = Action(
            type=action_type,
            ref=step.ref or None,
            value=step.key if action_type is ActionType.PRESS else bound,
            url=step.url,
        )

        # -- the gate, before the surface ----------------------------------
        if gate is not None:
            decision = gate.check(action, current_url=obs.url, risk=step.risk)
            if not isinstance(decision, Allow):
                reason = getattr(decision, "reason", str(decision))
                code = getattr(decision, "code", "denied")
                step.denied = True
                step.denial = (
                    f"That action is not permitted: {code}: {reason}. It was NOT "
                    "performed. Find another way, or call stuck."
                )
                traj.steps.append(step)
                note(f"policy denied {call.name} at turn {turn}: {code}", level="WARN")
                if recorder is not None:
                    recorder.step(
                        step_id=f"d{turn}",
                        intent=why,
                        action={"type": action_type.value, "ref": step.ref},
                        decision=decision,
                        outcome="denied",
                    )
                messages.append(
                    {"role": "assistant", "content": f"{call.name}({_arg_digest(step)})"}
                )
                messages.append({"role": "user", "content": step.denial})
                stale += 1
                if stale >= no_progress_limit:
                    traj.stop_reason = StopReason.NO_PROGRESS
                    break
                continue

        # -- act -----------------------------------------------------------
        result = surface.act(action)
        step.result = result
        if result.ok and action_type is ActionType.READ:
            step.read_value = result.read_value
        after = surface.observe()
        step.observation_after = after
        traj.steps.append(step)

        if recorder is not None:
            recorder.step(
                step_id=f"d{turn}",
                intent=why,
                action={
                    "type": action_type.value,
                    "ref": step.ref,
                    "value": (
                        f"<param:{step.value.param}>"
                        if step.value is not None and step.value.param
                        else (step.value.literal if step.value is not None else None)
                    ),
                },
                resolved_tier=result.resolved_tier,
                candidates=result.candidates,
                outcome="ok" if result.ok else "act_failed",
                detail=result.detail,
            )

        if result.ok:
            consecutive_failures = 0
            feedback = f"Action performed: {result.detail or 'ok'}."
            if action_type is ActionType.READ:
                feedback += (
                    f" Declared output {step.output_name!r} was captured "
                    "(its value is withheld from this transcript)."
                )
        else:
            consecutive_failures += 1
            feedback = (
                f"The action FAILED: {result.detail}. The page is unchanged. "
                "Try a different control, or call stuck."
            )

        messages.append({"role": "assistant", "content": f"{call.name}({_arg_digest(step)})"})
        messages.append({"role": "user", "content": feedback})

        if consecutive_failures >= dead_end_limit:
            traj.stop_reason = StopReason.DEAD_END
            note("discovery hit a dead end: repeated action failures", level="WARN")
            break

        if _state_key(after) == _state_key(obs):
            stale += 1
        else:
            stale = 0
        if stale >= no_progress_limit:
            traj.stop_reason = StopReason.NO_PROGRESS
            note(
                f"no state progress for {stale} consecutive steps; stopping",
                level="WARN",
            )
            break
    else:
        traj.stop_reason = StopReason.MAX_STEPS

    traj.final_observation = surface.observe()
    if traj.stop_reason == StopReason.STUCK:
        traj.escalate = True
    return traj
