"""Hermetic tests for the discovery agent, the compiler, and verify-before-save.

No network, no API key, no browser: a `RecordedSurface` plays the application
and a scripted fake `LLMClient` plays the model. The two assertions that carry
the most weight are (a) a raw sensitive value in a tool call is rejected and
never reaches the surface, and (b) verification replays with a DIFFERENT input
than discovery used -- without which "verified" would mean nothing more than
"the flow ran once".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cua.artifact import (
    ActionType,
    Capability,
    ParamSpec,
    RiskClass,
    Sensitivity,
)
from cua.discovery.agent import StopReason, discover, parse_value_ref
from cua.discovery.compile import (
    AppProfile,
    canonicalize_route,
    compile_trajectory,
    route_regex,
)
from cua.discovery.verify import derive_different_inputs, verify_and_save
from cua.llm.client import LLMResponse, ToolCall
from cua.policy.gate import Policy, PolicyGate
from cua.surface.base import Node, Observation
from cua.surface.recorded import RecordedSurface

SEARCH_URL = "https://app.example.com/members/search"
RESULTS_URL = "https://app.example.com/members/results"

GOAL = "Look up a member by id and return their name"

#: Discovery uses one member; verification must use another.
DISCOVERY_ID = "123456"

NAMES = {"123456": "Ada Lovelace", "234567": "Grace Hopper"}


# --------------------------------------------------------------------------
# The fake model
# --------------------------------------------------------------------------


class FakeLLM:
    """Returns scripted tool calls. Records every transcript it was shown, so a
    test can assert what the model was -- and was not -- told."""

    def __init__(self, script: list[ToolCall] | list[list[ToolCall]], *, vision: bool = False):
        self.script = list(script)
        self._vision = vision
        self.transcripts: list[list[dict[str, Any]]] = []
        self.systems: list[str] = []
        self.tools_seen: list[list[dict[str, Any]]] = []
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-model-1"

    @property
    def supports_vision(self) -> bool:
        return self._vision

    def complete(self, *, system, messages, tools=None, max_tokens=2048) -> LLMResponse:
        self.calls += 1
        self.systems.append(system)
        self.transcripts.append([dict(m) for m in messages])
        self.tools_seen.append(list(tools or []))
        if not self.script:
            return LLMResponse(tool_calls=[ToolCall(name="stuck", arguments={"reason": "script exhausted"})])
        item = self.script.pop(0)
        calls = item if isinstance(item, list) else [item]
        return LLMResponse(tool_calls=calls)

    def transcript_text(self) -> str:
        return "\n".join(
            str(m.get("content", "")) for turn in self.transcripts for m in turn
        )


def tc(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, arguments=args)


# --------------------------------------------------------------------------
# The fake application
# --------------------------------------------------------------------------


def frames_for(member_id: str) -> list[Observation]:
    name = NAMES.get(member_id, f"Member {member_id}")
    search = Observation(
        url=SEARCH_URL,
        title="Member Search",
        text_digest="Member Search. Enter a member id and press Search.",
        nodes=(
            Node(ref="n1", role="textbox", name="Member ID", value="", bbox=(120, 40, 200, 20)),
            Node(ref="n2", role="button", name="Search", bbox=(340, 40, 80, 20)),
        ),
    )
    results = Observation(
        url=RESULTS_URL,
        title="Member Detail",
        text_digest="Member detail loaded. Coverage is active.",
        nodes=(
            Node(ref="r1", role="text", name="Member Name", value=name, bbox=(120, 40, 200, 20)),
            Node(ref="r2", role="text", name="Plan Code", value="PPO-1", bbox=(120, 70, 200, 20)),
        ),
    )
    return [search, results]


def make_surface(inputs: dict[str, Any] | None = None) -> RecordedSurface:
    member_id = str((inputs or {}).get("member_id") or DISCOVERY_ID)
    return RecordedSurface(
        frames=frames_for(member_id),
        # Typing leaves you on the same screen; only Search advances.
        transitions={(0, ActionType.TYPE): 0},
    )


def stuck_surface(inputs: dict[str, Any] | None = None) -> RecordedSurface:
    """An application that never leaves the search page -- clicks go nowhere."""
    return RecordedSurface(frames=frames_for(DISCOVERY_ID)[:1], transitions={0: 0})


MEMBER_ID_PARAM = ParamSpec(
    name="member_id",
    type="string",
    description="The member id to look up.",
    sensitivity=Sensitivity.PII,
    example="000000",
)


HAPPY_SCRIPT = [
    tc("type_text", ref="n1", value_ref="member_id", why="Enter the member id we were asked about"),
    tc("click", ref="n2", why="Run the search to reach the member's detail page"),
    tc("read_value", ref="r1", output_name="member_name", why="Return the member's name"),
    tc("finish", summary="Member detail located and the name returned"),
]


def run_discovery(**kw: Any):
    llm = kw.pop("llm", None) or FakeLLM(list(HAPPY_SCRIPT))
    surface = kw.pop("surface", None) or make_surface({"member_id": DISCOVERY_ID})
    traj = discover(
        GOAL,
        SEARCH_URL,
        surface,
        llm=llm,
        inputs={"member_id": DISCOVERY_ID},
        params=[MEMBER_ID_PARAM],
        **kw,
    )
    return traj, llm, surface


def compiled() -> Capability:
    traj, _, _ = run_discovery()
    return compile_trajectory(
        traj,
        GOAL,
        capability_id="cap-member-lookup",
        name="member_lookup",
        profile=AppProfile(app_id="demo", vendor_product="DemoCare 9"),
    )


# --------------------------------------------------------------------------
# Part 1 -- the loop
# --------------------------------------------------------------------------


def test_loop_drives_a_scripted_plan_to_completion():
    traj, llm, surface = run_discovery()

    assert traj.stop_reason == StopReason.FINISHED
    assert traj.succeeded and not traj.escalate
    assert [s.tool for s in traj.executed_steps()] == ["type_text", "click", "read_value"]

    typed, clicked, read = traj.executed_steps()
    # The trajectory carries what the compiler needs: node, before-observation,
    # result and rationale.
    assert typed.node is not None and typed.node.ref == "n1"
    assert typed.observation_before.url == SEARCH_URL
    assert clicked.observation_after is not None
    assert clicked.observation_after.url == RESULTS_URL
    assert read.read_value == NAMES[DISCOVERY_ID]
    assert all(s.why for s in traj.executed_steps())  # every `why` is captured

    # Late binding: the reference is recorded, never the value.
    assert typed.value is not None and typed.value.param == "member_id"
    assert DISCOVERY_ID not in llm.transcript_text()
    # The model was handed real tool definitions, not asked for free-form JSON.
    assert {t["name"] for t in llm.tools_seen[0]} >= {"click", "type_text", "finish", "stuck"}


def test_value_ref_schema_enumerates_declared_params_and_a_literal_form():
    _, llm, _ = run_discovery()
    type_tool = next(t for t in llm.tools_seen[0] if t["name"] == "type_text")
    schema = type_tool["parameters"]["properties"]["value_ref"]
    options = schema["anyOf"]
    assert options[0]["enum"] == ["member_id"]
    assert options[1]["pattern"].startswith("^literal:")


def test_raw_sensitive_value_is_rejected_and_never_reaches_the_surface():
    ssn = "123-45-6789"
    llm = FakeLLM(
        [
            tc("type_text", ref="n1", value_ref=ssn, why="Type the member's SSN"),
            tc("finish", summary="done"),
        ]
    )
    surface = make_surface()
    traj = discover(
        GOAL,
        SEARCH_URL,
        surface,
        llm=llm,
        inputs={"ssn": ssn},
        params=[
            ParamSpec(
                name="ssn",
                type="string",
                description="Member SSN.",
                sensitivity=Sensitivity.SECRET,
            )
        ],
    )

    rejected = [s for s in traj.steps if s.rejected]
    assert len(rejected) == 1
    assert "SENSITIVE" in rejected[0].rejection
    assert "ssn" in rejected[0].rejection

    # The surface never saw a type action at all.
    assert not any(entry.startswith("type") for entry in surface.log)
    assert traj.executed_steps() == []
    # And the rejection was fed back to the model.
    assert "REJECTED" in llm.transcript_text()


def test_literal_prefix_is_accepted_for_a_genuine_constant():
    vref, err = parse_value_ref("literal:Active", ["member_id"], {"member_id": "123456"})
    assert err is None and vref.literal == "Active"

    vref, err = parse_value_ref("literal:123456", ["member_id"], {"member_id": "123456"})
    assert vref is None and "parameter, not a constant" in err


def test_gate_denied_action_is_fed_back_and_not_executed():
    llm = FakeLLM(
        [
            tc("click", ref="n2", why="Run the search"),
            tc("finish", summary="gave up on clicking"),
        ]
    )
    surface = make_surface()
    traj = discover(
        GOAL,
        SEARCH_URL,
        surface,
        llm=llm,
        inputs={"member_id": DISCOVERY_ID},
        params=[MEMBER_ID_PARAM],
        gate=PolicyGate(Policy.empty()),  # deny-by-default
    )

    denied = [s for s in traj.steps if s.denied]
    assert len(denied) == 1
    assert "not permitted" in denied[0].denial
    assert not any(entry.startswith("click") for entry in surface.log)
    assert "not permitted" in llm.transcript_text()
    assert surface.index == 0  # the world never moved


def test_max_steps_stops_the_run():
    llm = FakeLLM([tc("click", ref="n2", why="try again") for _ in range(20)])
    surface = RecordedSurface(frames=frames_for(DISCOVERY_ID), transitions={0: 0})
    traj = discover(
        GOAL,
        SEARCH_URL,
        surface,
        llm=llm,
        inputs={},
        max_steps=3,
        no_progress_limit=99,
    )
    assert traj.stop_reason == StopReason.MAX_STEPS
    assert len(traj.executed_steps()) == 3


def test_no_progress_stops_the_run():
    llm = FakeLLM([tc("click", ref="n2", why="try again") for _ in range(20)])
    surface = RecordedSurface(frames=frames_for(DISCOVERY_ID), transitions={0: 0})
    traj = discover(
        GOAL,
        SEARCH_URL,
        surface,
        llm=llm,
        inputs={},
        max_steps=20,
        no_progress_limit=2,
    )
    assert traj.stop_reason == StopReason.NO_PROGRESS
    assert len(traj.executed_steps()) == 2


def test_timeout_stops_the_run_before_any_action():
    llm = FakeLLM([tc("click", ref="n2", why="try")])
    surface = make_surface()
    traj = discover(
        GOAL, SEARCH_URL, surface, llm=llm, inputs={}, timeout_s=0
    )
    assert traj.stop_reason == StopReason.TIMEOUT
    assert llm.calls == 0
    assert surface.log == []


def test_stuck_ends_the_run_flagged_for_escalation():
    llm = FakeLLM([tc("stuck", reason="No control on this page opens a member record")])
    traj, _, _ = run_discovery(llm=llm)
    assert traj.stop_reason == StopReason.STUCK
    assert traj.escalate is True
    assert "member record" in traj.stuck_reason
    assert traj.executed_steps() == []


def test_loop_works_with_a_visionless_client():
    llm = FakeLLM(list(HAPPY_SCRIPT), vision=False)
    traj, _, _ = run_discovery(llm=llm)
    assert traj.succeeded
    assert all(s.screenshot_path is None for s in traj.steps)
    assert "SCREENSHOT" not in llm.transcript_text()


def test_vision_client_gets_a_screenshot_but_the_tree_still_drives():
    llm = FakeLLM(list(HAPPY_SCRIPT), vision=True)
    traj, _, _ = run_discovery(llm=llm)
    assert traj.succeeded
    assert "CONTROLS:" in llm.transcript_text()


# --------------------------------------------------------------------------
# Part 2 -- the compiler
# --------------------------------------------------------------------------


def test_compiler_yields_a_schema_valid_capability():
    cap = compiled()
    Capability.model_validate(cap.model_dump())  # round-trips through the schema

    assert [s.action for s in cap.steps] == [
        ActionType.TYPE,
        ActionType.CLICK,
        ActionType.READ,
    ]
    state_changing = {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT, ActionType.PRESS, ActionType.NAVIGATE}
    for step in cap.steps:
        if step.action in state_changing:
            assert step.post_condition is not None, step.id
        if step.target is not None:
            assert step.baseline_tier is not None, step.id

    # `why` became `intent` -- that is why the tool asks for it.
    assert cap.steps[0].intent.startswith("Enter the member id")
    # Late binding survived compilation.
    assert cap.steps[0].value.param == "member_id"
    assert [p.name for p in cap.inputs] == ["member_id"]
    assert [o.name for o in cap.outputs] == ["member_name"]
    assert cap.outputs[0].source_step_id == cap.steps[2].id
    assert cap.risk_class is RiskClass.READ_ONLY
    assert cap.provenance.verified_at is None  # not verified yet, not saved yet


def test_post_conditions_do_not_bake_in_the_discovery_value():
    cap = compiled()
    blob = cap.model_dump_json()
    assert DISCOVERY_ID not in blob
    assert NAMES[DISCOVERY_ID] not in blob
    # The click's post-condition asserts the route change it actually caused.
    assert cap.steps[1].post_condition.url_matches is not None


def test_app_profile_supplies_signals_and_outcomes():
    from cua.artifact import OutcomeClass, OutcomeSpec, SignalRule

    traj, _, _ = run_discovery()
    profile = AppProfile(
        app_id="demo",
        vendor_product="DemoCare 9",
        signals=[
            SignalRule(
                code="record_not_found",
                classification=OutcomeClass.BUSINESS,
                text_present="No records located",
                message="No member matches that id.",
                handler="return",
            )
        ],
        outcomes=[OutcomeSpec(code="record_not_found", description="No such member.")],
    )
    cap = compile_trajectory(traj, GOAL, profile=profile)
    assert [s.code for s in cap.signals] == ["record_not_found"]
    assert [o.code for o in cap.possible_outcomes] == ["record_not_found"]
    assert cap.app_profile.vendor_product == "DemoCare 9"


def test_url_canonicalization_turns_a_recorded_id_into_a_slot():
    canonical = canonicalize_route(
        "https://app.example.com/member/12345/detail", {"member_id": "12345"}
    )
    assert canonical == "https://app.example.com/member/:member_id/detail"

    import re

    rx = route_regex(canonical)
    assert re.search(rx, "https://app.example.com/member/99999/detail")
    assert not re.search(rx, "https://app.example.com/member/99999/other")


def test_compiler_refuses_a_run_that_did_not_finish():
    from cua.discovery.compile import CompileError

    llm = FakeLLM([tc("stuck", reason="dead end")])
    traj, _, _ = run_discovery(llm=llm)
    with pytest.raises(CompileError):
        compile_trajectory(traj, GOAL)


# --------------------------------------------------------------------------
# Part 3 -- verify before save
# --------------------------------------------------------------------------


def test_derive_different_inputs_produces_a_distinct_value():
    cap = compiled()
    alt = derive_different_inputs(cap, {"member_id": DISCOVERY_ID})
    assert alt["member_id"] != DISCOVERY_ID
    assert alt["member_id"] == "234567"  # digit-shifted, same shape


def test_verify_and_save_uses_a_different_input_than_discovery(tmp_path: Path):
    cap = compiled()
    seen: list[str] = []

    def factory(inputs: dict[str, Any]) -> RecordedSurface:
        seen.append(str(inputs.get("member_id")))
        return make_surface(inputs)

    out = tmp_path / "member_lookup.yaml"
    written = verify_and_save(
        cap, factory, out, discovery_inputs={"member_id": DISCOVERY_ID}
    )

    assert written == out and out.exists()
    # THE point of the exercise: the artifact was proven with an input the
    # discovery run never used.
    assert seen, "verification never built a surface"
    assert seen[0] != DISCOVERY_ID
    assert DISCOVERY_ID in seen  # and the pair was compared, not just re-run
    assert cap.provenance.verification_mode == "full_replay"
    assert cap.provenance.verified_at is not None
    # member_id is PII, so the value used is not written into the artifact.
    assert cap.provenance.verified_with_inputs == {}

    text = out.read_text()
    assert "schema_version" in text
    # Stable, sorted YAML so artifacts diff cleanly.
    keys = [
        line.split(":")[0]
        for line in text.splitlines()
        if line and not line[0].isspace() and not line.startswith("-")
    ]
    assert keys == sorted(keys)


def test_verify_and_save_writes_nothing_when_verification_fails(tmp_path: Path):
    cap = compiled()
    out = tmp_path / "nested" / "member_lookup.yaml"

    # An application whose search button no longer goes anywhere.
    written = verify_and_save(
        cap, stuck_surface, out, discovery_inputs={"member_id": DISCOVERY_ID}
    )

    assert written is None
    assert not out.exists()
    assert cap.provenance.verified_at is None
    assert cap.provenance.verification_mode == "none"


def test_verify_fails_when_outputs_do_not_depend_on_the_input(tmp_path: Path):
    """A baked-in literal would make two different members return one name."""
    cap = compiled()

    def frozen_factory(inputs: dict[str, Any]) -> RecordedSurface:
        return make_surface({"member_id": DISCOVERY_ID})  # ignores the input

    out = tmp_path / "frozen.yaml"
    assert verify_and_save(
        cap, frozen_factory, out, discovery_inputs={"member_id": DISCOVERY_ID}
    ) is None
    assert not out.exists()


def test_write_capability_is_verified_by_dry_run_not_by_doing_it(tmp_path: Path):
    """Replaying 'Submit Adjustment' for real would create a second record."""
    submit = Node(ref="n3", role="button", name="Submit Adjustment", bbox=(440, 40, 140, 20))
    search = Observation(
        url=SEARCH_URL,
        title="Member Search",
        text_digest="Member Search. Enter a member id and press Search.",
        nodes=(
            Node(ref="n1", role="textbox", name="Member ID", value="", bbox=(120, 40, 200, 20)),
            submit,
        ),
    )
    done = Observation(
        url=RESULTS_URL,
        title="Adjustment Posted",
        text_digest="Adjustment posted successfully.",
        nodes=(Node(ref="d1", role="text", name="Confirmation Number", value="A-1"),),
    )

    def factory(inputs: dict[str, Any] | None = None) -> RecordedSurface:
        return RecordedSurface(frames=[search, done], transitions={(0, ActionType.TYPE): 0})

    llm = FakeLLM(
        [
            tc("type_text", ref="n1", value_ref="member_id", why="Identify the member"),
            tc("click", ref="n3", why="Post the adjustment"),
            tc("finish", summary="Adjustment posted"),
        ]
    )
    traj = discover(
        "Post an adjustment",
        SEARCH_URL,
        factory(),
        llm=llm,
        inputs={"member_id": DISCOVERY_ID},
        params=[MEMBER_ID_PARAM],
    )
    cap = compile_trajectory(traj, "Post an adjustment", capability_id="cap-adjust")
    assert cap.risk_class is RiskClass.IRREVERSIBLE

    out = tmp_path / "adjust.yaml"
    written = verify_and_save(
        cap, factory, out, discovery_inputs={"member_id": DISCOVERY_ID}
    )
    assert written == out
    assert cap.provenance.verification_mode == "dry_run"
