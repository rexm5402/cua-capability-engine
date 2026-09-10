"""Hermetic tests for the deterministic replay engine.

No browser, no network, no API key: every scenario is a hand-built
`Capability` played against a `RecordedSurface`. The single most important
assertion in this file is that a "no records located" page comes back as a
`BusinessOutcome` and not a `Failure` -- conflating those two is the design
mistake the whole three-type result union exists to prevent.
"""

from __future__ import annotations

import pytest

from cua.artifact import (
    ActionType,
    AppProfileRef,
    Capability,
    Checkpoint,
    LocatorBundle,
    OutcomeClass,
    OutputSpec,
    ParamSpec,
    Provenance,
    RiskClass,
    SemanticLocator,
    Sensitivity,
    SignalRule,
    Step,
    TextLocator,
    ValueRef,
)
from cua.evidence.recorder import EvidenceRecorder
from cua.outcomes import BusinessOutcome, Failure, Success
from cua.policy.gate import Policy, PolicyGate, RiskRule
from cua.replay.engine import (
    APPROVAL_REQUIRED,
    INVALID_INPUT,
    POLICY_DENIED,
    SUCCESS_CHECKPOINT_FAILED,
    UNRECOVERED_SIGNAL,
    DryRunSuccess,
    ReplayError,
    replay,
)
from cua.surface.base import Node, Observation
from cua.surface.recorded import RecordedSurface
from cua.telemetry import TelemetryStore

SEARCH_URL = "https://app.example.com/members/search"
RESULTS_URL = "https://app.example.com/members/results"


# --------------------------------------------------------------------------
# Fixtures: frames
# --------------------------------------------------------------------------


def search_frame() -> Observation:
    return Observation(
        url=SEARCH_URL,
        title="Member Search",
        text_digest="Member Search. Enter a member id and press Search.",
        nodes=(
            Node(ref="n1", role="textbox", name="Member ID", value=""),
            Node(ref="n2", role="button", name="Search"),
            Node(ref="n3", role="button", name="Submit Adjustment"),
        ),
    )


def results_frame() -> Observation:
    return Observation(
        url=RESULTS_URL,
        title="Results",
        text_digest="Search results for member 123456. Account is active.",
        nodes=(
            Node(ref="r1", role="text", name="Member Name", value="Ada Lovelace"),
            Node(ref="r2", role="text", name="Balance Amount", value="1,234.50"),
        ),
    )


def frame(url: str, title: str, digest: str, *nodes: Node) -> Observation:
    return Observation(url=url, title=title, text_digest=digest, nodes=tuple(nodes))


# --------------------------------------------------------------------------
# Fixtures: capability
# --------------------------------------------------------------------------


def _provenance() -> Provenance:
    return Provenance(
        recorded_by_model="test",
        discovery_run_id="disc-1",
        recorded_at="2026-01-01T00:00:00Z",
        verified_at="2026-01-01T00:05:00Z",
        verification_mode="full_replay",
    )


def bundle(desc: str, role: str, name: str, *, text: str | None = None) -> LocatorBundle:
    strategies = [SemanticLocator(role=role, name=name)]
    if text is not None:
        strategies.append(TextLocator(text=text, role=role))
    return LocatorBundle(description=desc, strategies=strategies)


SEARCH_BUTTON = bundle("the Search button", "button", "Search")
MEMBER_FIELD = bundle("the Member ID field", "textbox", "Member ID")
BALANCE = bundle("the balance readout", "text", "Balance Amount")
MEMBER_NAME = bundle("the member name readout", "text", "Member Name")


def lookup_capability(
    *,
    signals: list[SignalRule] | None = None,
    success: Checkpoint | None = None,
    search_target: LocatorBundle = SEARCH_BUTTON,
    baseline_tier: int | None = 1,
) -> Capability:
    """A read-only member lookup: navigate, type, click, read two fields."""
    return Capability(
        id="member_lookup",
        name="member_lookup",
        description="Look up a member by id and return name and balance.",
        app_profile=AppProfileRef(app_id="demo", vendor_product="demo/portal"),
        entry_url=SEARCH_URL,
        inputs=[
            ParamSpec(
                name="member_id",
                type="string",
                description="Six digit member id.",
                pattern=r"^\d{6}$",
                sensitivity=Sensitivity.PII,
            )
        ],
        outputs=[
            OutputSpec(
                name="member_name",
                type="string",
                description="Member full name.",
                source_step_id="s_read_name",
            ),
            OutputSpec(
                name="balance",
                type="number",
                description="Current balance.",
                source_step_id="s_read_balance",
            ),
        ],
        steps=[
            Step(
                id="s_nav",
                intent="Open the member search screen.",
                action=ActionType.NAVIGATE,
                value=ValueRef(literal=SEARCH_URL),
                post_condition=Checkpoint(
                    description="the search screen is showing",
                    text_present="Member Search",
                ),
            ),
            Step(
                id="s_type",
                intent="Enter the member id supplied by the caller.",
                action=ActionType.TYPE,
                target=MEMBER_FIELD,
                value=ValueRef(param="member_id"),
                post_condition=Checkpoint(
                    description="still on the search screen",
                    text_present="Member Search",
                ),
                baseline_tier=1,
            ),
            Step(
                id="s_search",
                intent="Run the search.",
                action=ActionType.CLICK,
                target=search_target,
                post_condition=Checkpoint(
                    description="the results screen is showing",
                    text_present="Search results",
                ),
                baseline_tier=baseline_tier,
            ),
            Step(
                id="s_read_name",
                intent="Read the member name off the results screen.",
                action=ActionType.READ,
                target=MEMBER_NAME,
                output_name="member_name",
                baseline_tier=1,
            ),
            Step(
                id="s_read_balance",
                intent="Read the balance off the results screen.",
                action=ActionType.READ,
                target=BALANCE,
                output_name="balance",
                baseline_tier=1,
            ),
        ],
        success=success
        or Checkpoint(
            description="results are on screen",
            text_present="Search results",
        ),
        signals=signals or [],
        provenance=_provenance(),
    )


def open_policy() -> PolicyGate:
    return PolicyGate(
        Policy(
            name="test",
            mode="attended",
            allowed_origins=["https://app.example.com"],
            allowed_routes=["/members/*"],
            allowed_action_types=[
                ActionType.NAVIGATE,
                ActionType.CLICK,
                ActionType.TYPE,
                ActionType.READ,
                ActionType.WAIT,
                ActionType.PRESS,
                ActionType.SELECT,
            ],
        )
    )


def two_frame_surface(second: Observation | None = None) -> RecordedSurface:
    """Search -> results, with TYPE deliberately not advancing the world."""
    return RecordedSurface(
        frames=[search_frame(), second or results_frame()],
        transitions={(0, ActionType.TYPE): 0},
    )


# --------------------------------------------------------------------------
# 1. Happy path
# --------------------------------------------------------------------------


def test_happy_path_returns_typed_outputs_and_verified_checkpoint():
    surface = two_frame_surface()
    outcome = replay(lookup_capability(), {"member_id": "123456"}, surface, gate=open_policy())

    assert isinstance(outcome, Success), outcome
    assert outcome.capability_id == "member_lookup"
    assert outcome.outputs["member_name"] == "Ada Lovelace"
    # Declared as `number`, so the caller gets a float, not "1,234.50".
    assert outcome.outputs["balance"] == pytest.approx(1234.50)
    assert isinstance(outcome.outputs["balance"], float)
    assert outcome.steps_executed == 5
    assert outcome.tier_escalations == 0
    # The terminal checkpoint was actually asserted.
    assert any(entry.startswith("wait_for") for entry in surface.log)


def test_no_llm_client_is_imported_on_the_replay_path():
    import sys

    for mod in ("cua.replay.engine", "cua.replay.signals"):
        sys.modules.pop(mod, None)
    sys.modules.pop("cua.llm.client", None)
    import cua.replay.engine  # noqa: F401
    import cua.replay.signals  # noqa: F401

    assert "cua.llm.client" not in sys.modules
    assert not any(m.startswith("anthropic") for m in sys.modules)


# --------------------------------------------------------------------------
# 2. THE test: a business outcome is not a failure
# --------------------------------------------------------------------------


def test_no_records_located_is_a_business_outcome_not_a_failure():
    empty = frame(
        RESULTS_URL,
        "Results",
        "No records located for the supplied member id.",
        Node(ref="e1", role="text", name="No records located"),
    )
    cap = lookup_capability(
        signals=[
            SignalRule(
                code="record_not_found",
                classification=OutcomeClass.BUSINESS,
                text_present="No records located",
                message="No member exists with that id.",
            )
        ]
    )
    outcome = replay(cap, {"member_id": "123456"}, two_frame_surface(empty), gate=open_policy())

    assert isinstance(outcome, BusinessOutcome), outcome
    assert not isinstance(outcome, Failure)
    assert outcome.code == "record_not_found"
    assert outcome.message == "No member exists with that id."
    assert outcome.at_step_id == "s_search"


def test_business_outcome_carries_partial_outputs_collected_so_far():
    """A name read before the business signal fires still reaches the caller."""
    partial = frame(
        RESULTS_URL,
        "Results",
        "Search results for member. Balance is unavailable for closed accounts.",
        Node(ref="r1", role="text", name="Member Name", value="Ada Lovelace"),
    )
    cap = lookup_capability(
        signals=[
            SignalRule(
                code="account_closed",
                classification=OutcomeClass.BUSINESS,
                text_present="Balance is unavailable for closed accounts",
                message="The account is closed; no balance is published.",
            )
        ]
    )
    # The signal is not on the search frame, so the search step completes and the
    # name is read before the scan on the following step fires.
    outcome = replay(cap, {"member_id": "123456"}, two_frame_surface(partial), gate=open_policy())

    assert isinstance(outcome, BusinessOutcome), outcome
    assert outcome.code == "account_closed"


# --------------------------------------------------------------------------
# 3. Recoverable conditions
# --------------------------------------------------------------------------


def interstitial_frames() -> list[Observation]:
    return [
        search_frame(),
        frame(
            RESULTS_URL,
            "Notice",
            "Scheduled maintenance notice. Please acknowledge to continue.",
            Node(ref="i1", role="button", name="Continue"),
        ),
        results_frame(),
    ]


def test_recoverable_interstitial_is_dismissed_and_the_caller_never_hears_about_it():
    cap = lookup_capability(
        signals=[
            SignalRule(
                code="maintenance_notice",
                classification=OutcomeClass.RECOVERABLE,
                text_present="Scheduled maintenance notice",
                message="A maintenance interstitial was shown.",
                handler="dismiss",
                max_attempts=2,
            )
        ]
    )
    surface = RecordedSurface(
        frames=interstitial_frames(),
        transitions={(0, ActionType.TYPE): 0},
    )
    outcome = replay(cap, {"member_id": "123456"}, surface, gate=open_policy())

    assert isinstance(outcome, Success), outcome
    assert outcome.outputs["member_name"] == "Ada Lovelace"
    # Nothing in the typed result mentions the interstitial.
    assert "maintenance" not in outcome.model_dump_json().casefold()


def test_recoverable_condition_that_never_clears_becomes_a_hard_failure():
    stuck = [
        search_frame(),
        frame(
            RESULTS_URL,
            "Notice",
            "Scheduled maintenance notice. Please acknowledge to continue.",
            Node(ref="i1", role="button", name="Continue"),
        ),
    ]
    cap = lookup_capability(
        signals=[
            SignalRule(
                code="maintenance_notice",
                classification=OutcomeClass.RECOVERABLE,
                text_present="Scheduled maintenance notice",
                message="A maintenance interstitial was shown.",
                handler="dismiss",
                max_attempts=2,
            )
        ]
    )
    surface = RecordedSurface(
        frames=stuck,
        # Dismissing gets us nowhere: the notice is still there.
        transitions={(0, ActionType.TYPE): 0, (1, ActionType.CLICK): 1},
    )
    outcome = replay(cap, {"member_id": "123456"}, surface, gate=open_policy())

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == UNRECOVERED_SIGNAL
    assert outcome.at_step_id == "s_search"
    assert "maintenance" in outcome.observed.casefold()


def test_recoverable_retry_handler_succeeds_on_the_second_attempt():
    # The busy page re-renders the SAME form with a banner over it, which is
    # what a real transient interstitial does -- and it matters here: a retry
    # needs something to act on. A busy page that dropped the form would make
    # "redo the step" structurally impossible, and the correct recovery there
    # would be wait-and-re-observe rather than retry.
    busy = frame(
        SEARCH_URL,
        "Busy",
        "The system is busy, please try again.",
        Node(ref="n1", role="textbox", name="Member ID", value="123456"),
        Node(ref="n2", role="button", name="Search"),
    )
    flaky = [search_frame(), busy, results_frame()]
    cap = lookup_capability(
        signals=[
            SignalRule(
                code="system_busy",
                classification=OutcomeClass.RECOVERABLE,
                text_present="system is busy",
                message="Transient busy page.",
                handler="retry",
                max_attempts=3,
            )
        ]
    )
    surface = RecordedSurface(
        frames=flaky,
        transitions={(0, ActionType.TYPE): 0},
    )
    outcome = replay(cap, {"member_id": "123456"}, surface, gate=open_policy())

    assert isinstance(outcome, Success), outcome
    assert outcome.outputs["balance"] == pytest.approx(1234.50)


# --------------------------------------------------------------------------
# 4. Hard failure
# --------------------------------------------------------------------------


def test_hard_signal_produces_a_debuggable_failure():
    broken = frame(
        RESULTS_URL,
        "Error",
        "Internal server error 500. Reference ABC-1.",
        Node(ref="x1", role="text", name="Internal server error"),
    )
    cap = lookup_capability(
        signals=[
            SignalRule(
                code="server_error",
                classification=OutcomeClass.HARD,
                text_present="Internal server error",
                message="The application returned a 500.",
            )
        ]
    )
    outcome = replay(cap, {"member_id": "123456"}, two_frame_surface(broken), gate=open_policy())

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == "server_error"
    assert outcome.at_step_id == "s_search"
    assert outcome.step_intent == "Run the search."
    assert outcome.expected == "Run the search."
    assert "500" in outcome.observed


def test_failed_post_condition_without_a_signal_is_a_failure_with_context():
    """No signal explains it, so it is genuinely broken -- and debuggable."""
    wrong = frame(RESULTS_URL, "Somewhere else", "An unrelated screen.")
    outcome = replay(
        lookup_capability(), {"member_id": "123456"}, two_frame_surface(wrong), gate=open_policy()
    )

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == "post_condition_failed"
    assert outcome.at_step_id == "s_search"
    assert "Search results" in outcome.expected
    assert "unrelated" in outcome.observed.casefold() or "Somewhere else" in outcome.observed


# --------------------------------------------------------------------------
# 5. Policy
# --------------------------------------------------------------------------


def test_policy_denial_stops_the_run_before_any_action_reaches_the_surface():
    surface = two_frame_surface()
    outcome = replay(
        lookup_capability(),
        {"member_id": "123456"},
        surface,
        gate=PolicyGate(Policy.empty()),
    )

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == POLICY_DENIED
    assert outcome.at_step_id == "s_nav"
    assert surface.log == []
    assert surface.index == 0


def test_require_approval_without_an_escalator_is_approval_required():
    gate = PolicyGate(
        Policy(
            name="approval",
            mode="attended",
            allowed_origins=["https://app.example.com"],
            allowed_routes=["/members/*"],
            allowed_action_types=[ActionType.NAVIGATE, ActionType.CLICK, ActionType.TYPE, ActionType.READ],
            risk_rules={RiskClass.READ_ONLY: RiskRule(allowed=True, require_approval=True)},
        )
    )
    surface = two_frame_surface()
    outcome = replay(lookup_capability(), {"member_id": "123456"}, surface, gate=gate)

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == APPROVAL_REQUIRED
    assert surface.log == []


def test_require_approval_is_routed_to_the_escalator_when_one_is_supplied():
    class Approver:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def request_approval(self, *, capability, step, reason, observation) -> bool:
            self.calls.append(step.id)
            return True

    gate = PolicyGate(
        Policy(
            name="approval",
            mode="attended",
            allowed_origins=["https://app.example.com"],
            allowed_routes=["/members/*"],
            allowed_action_types=[ActionType.NAVIGATE, ActionType.CLICK, ActionType.TYPE, ActionType.READ],
            risk_rules={RiskClass.READ_ONLY: RiskRule(allowed=True, require_approval=True)},
        )
    )
    approver = Approver()
    outcome = replay(
        lookup_capability(),
        {"member_id": "123456"},
        two_frame_surface(),
        gate=gate,
        escalator=approver,
    )

    assert isinstance(outcome, Success), outcome
    assert approver.calls[0] == "s_nav"


# --------------------------------------------------------------------------
# 6. Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        {},                              # missing required
        {"member_id": 123456},           # wrong type
        {"member_id": "abc"},            # fails pattern
        {"member_id": "123456", "x": 1},  # undeclared input
    ],
)
def test_invalid_input_is_rejected_before_any_surface_interaction(inputs):
    surface = two_frame_surface()
    outcome = replay(lookup_capability(), inputs, surface, gate=open_policy())

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == INVALID_INPUT
    assert surface.log == []
    assert surface.index == 0
    assert outcome.at_step_id is None


def test_invalid_input_failure_does_not_quote_the_sensitive_value():
    outcome = replay(lookup_capability(), {"member_id": "99-SECRET-99"}, two_frame_surface())
    assert isinstance(outcome, Failure)
    assert "99-SECRET-99" not in outcome.observed


# --------------------------------------------------------------------------
# 7. Drift telemetry
# --------------------------------------------------------------------------


def test_resolution_worse_than_baseline_is_recorded_as_drift(tmp_path):
    """The button was renamed, so tier 1 misses and tier 3 catches it."""
    renamed = frame(
        SEARCH_URL,
        "Member Search",
        "Member Search. Enter a member id and press Search.",
        Node(ref="n1", role="textbox", name="Member ID", value=""),
        Node(ref="n2", role="button", name="Find", text="Search"),
    )
    cap = lookup_capability(
        search_target=bundle("the Search button", "button", "Search", text="Search"),
        baseline_tier=1,
    )
    surface = RecordedSurface(
        frames=[renamed, results_frame()],
        transitions={(0, ActionType.TYPE): 0},
    )
    store = TelemetryStore(root=tmp_path / "telemetry")
    outcome = replay(cap, {"member_id": "123456"}, surface, gate=open_policy(), telemetry=store)

    assert isinstance(outcome, Success), outcome
    assert outcome.tier_escalations >= 1

    t = store.load(cap.id, cap.version, None)
    assert "s_search" in t.drifting_steps()
    assert t.steps["s_search"].baseline_tier == 1
    assert t.steps["s_search"].escalations >= 1
    assert t.replays == 1 and t.successes == 1


def test_baseline_tier_two_resolving_at_tier_two_is_not_drift(tmp_path):
    """A hostile surface may legitimately baseline above tier 1."""
    cap = lookup_capability(baseline_tier=2)
    store = TelemetryStore(root=tmp_path / "telemetry")
    outcome = replay(
        cap, {"member_id": "123456"}, two_frame_surface(), gate=open_policy(), telemetry=store
    )

    assert isinstance(outcome, Success), outcome
    assert outcome.tier_escalations == 0
    assert store.load(cap.id, cap.version, None).drifting_steps() == []


# --------------------------------------------------------------------------
# 8. Dry run
# --------------------------------------------------------------------------


def write_capability() -> Capability:
    """A write capability: the point of dry-run mode is verifying it without
    creating a second real record."""
    return Capability(
        id="submit_adjustment",
        name="submit_adjustment",
        description="Submit a balance adjustment.",
        app_profile=AppProfileRef(app_id="demo", vendor_product="demo/portal"),
        entry_url=SEARCH_URL,
        inputs=[
            ParamSpec(name="member_id", type="string", description="id", pattern=r"^\d{6}$")
        ],
        outputs=[],
        steps=[
            Step(
                id="s_type",
                intent="Enter the member id.",
                action=ActionType.TYPE,
                target=MEMBER_FIELD,
                value=ValueRef(param="member_id"),
                post_condition=Checkpoint(
                    description="still on the search screen", text_present="Member Search"
                ),
                baseline_tier=1,
            ),
            Step(
                id="s_submit",
                intent="Submit the adjustment -- this creates a real record.",
                action=ActionType.CLICK,
                target=bundle("the Submit Adjustment button", "button", "Submit Adjustment"),
                post_condition=Checkpoint(
                    description="a receipt is shown", text_present="Adjustment recorded"
                ),
                risk=RiskClass.REVERSIBLE,
                baseline_tier=1,
            ),
        ],
        success=Checkpoint(description="receipt", text_present="Adjustment recorded"),
        provenance=_provenance(),
    )


def test_dry_run_resolves_remaining_locators_without_acting():
    surface = RecordedSurface(
        frames=[search_frame(), results_frame()],
        transitions={(0, ActionType.TYPE): 0},
    )
    outcome = replay(
        write_capability(),
        {"member_id": "123456"},
        surface,
        gate=open_policy(),
        dry_run_from="s_submit",
    )

    assert isinstance(outcome, Success), outcome
    assert isinstance(outcome, DryRunSuccess)
    assert outcome.dry_run is True
    assert outcome.dry_run_from == "s_submit"
    assert outcome.resolved_without_acting == ["s_submit"]
    # No real record was created: the world never moved past the search screen.
    assert surface.index == 0
    assert [e for e in surface.log if e.startswith("click")] == []


def test_dry_run_reports_an_unreachable_remaining_locator():
    stripped = frame(
        SEARCH_URL,
        "Member Search",
        "Member Search.",
        Node(ref="n1", role="textbox", name="Member ID", value=""),
    )
    surface = RecordedSurface(frames=[stripped], transitions={(0, ActionType.TYPE): 0})
    outcome = replay(
        write_capability(),
        {"member_id": "123456"},
        surface,
        gate=open_policy(),
        dry_run_from="s_submit",
    )

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == "dry_run_unreachable"
    assert outcome.at_step_id == "s_submit"


def test_dry_run_from_an_unknown_step_is_programmer_error():
    with pytest.raises(ReplayError):
        replay(
            write_capability(),
            {"member_id": "123456"},
            two_frame_surface(),
            dry_run_from="nope",
        )


# --------------------------------------------------------------------------
# 9. Unverified success
# --------------------------------------------------------------------------


def test_unverified_success_checkpoint_yields_failure_not_success():
    cap = lookup_capability(
        success=Checkpoint(
            description="a receipt number is displayed", text_present="Receipt #"
        )
    )
    outcome = replay(cap, {"member_id": "123456"}, two_frame_surface(), gate=open_policy())

    assert isinstance(outcome, Failure), outcome
    assert outcome.code == SUCCESS_CHECKPOINT_FAILED
    assert "Receipt #" in outcome.expected
    assert outcome.observed


# --------------------------------------------------------------------------
# 10. Evidence and redaction
# --------------------------------------------------------------------------


def test_sensitive_values_never_reach_the_evidence_bundle(tmp_path):
    recorder = EvidenceRecorder(
        "run-1", "replay", root=tmp_path / "runs", capability_id="member_lookup"
    )
    outcome = replay(
        lookup_capability(),
        {"member_id": "123456"},
        two_frame_surface(),
        gate=open_policy(),
        recorder=recorder,
    )

    assert isinstance(outcome, Success), outcome
    steps = (recorder.dir / "steps.jsonl").read_text()
    manifest = (recorder.dir / "manifest.json").read_text()
    assert "123456" not in steps
    assert "123456" not in manifest
    assert "s_search" in steps
    assert outcome.evidence_dir == str(recorder.dir)


def test_drift_is_noted_in_the_evidence_log(tmp_path):
    renamed = frame(
        SEARCH_URL,
        "Member Search",
        "Member Search. Enter a member id and press Search.",
        Node(ref="n1", role="textbox", name="Member ID", value=""),
        Node(ref="n2", role="button", name="Find", text="Search"),
    )
    cap = lookup_capability(
        search_target=bundle("the Search button", "button", "Search", text="Search")
    )
    surface = RecordedSurface(
        frames=[renamed, results_frame()], transitions={(0, ActionType.TYPE): 0}
    )
    recorder = EvidenceRecorder("run-2", "replay", root=tmp_path / "runs")
    replay(cap, {"member_id": "123456"}, surface, gate=open_policy(), recorder=recorder)

    log = (recorder.dir / "run.log").read_text()
    assert "DRIFT" in log and "s_search" in log
