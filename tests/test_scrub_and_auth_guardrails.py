"""Regression tests for two audit findings.

C1: a value the schema tags pii/secret reached the compiled artifact.
C2: authentication called surface.act directly, bypassing the policy gate --
    and authentication is also the `reauth` recovery handler, so the bypass was
    reachable mid-replay, not only at pre-flight.

Both tests fail if their fix is reverted.
"""

from __future__ import annotations

import pytest

from cua.artifact import (
    ActionType,
    Capability,
    Checkpoint,
    LocatorBundle,
    OutputSpec,
    ParamSpec,
    Provenance,
    SemanticLocator,
    Sensitivity,
    Step,
    TextLocator,
    ValueRef,
)
from cua.discovery.scrub import ContaminatedLocator, scrub_capability
from cua.policy.gate import Policy, PolicyGate
from cua.session.provider import AuthPolicyDenied, authenticate
from cua.surface.base import ActResult, Node, Observation, SurfaceInfo

MEMBER = "10001"


def _cap(*, target: LocatorBundle, checkpoint: Checkpoint) -> Capability:
    return Capability(
        id="c1",
        name="lookup",
        description=f"look up member {MEMBER} and read their savings balance",
        app_profile={"app_id": "a", "vendor_product": "v"},
        entry_url="http://localhost:5055/search",
        inputs=[
            ParamSpec(
                name="member_id",
                type="string",
                description="Member id",
                sensitivity=Sensitivity.PII,
            )
        ],
        outputs=[
            OutputSpec(
                name="savings", type="string", description="bal", source_step_id="s1"
            )
        ],
        steps=[
            Step(
                id="s1",
                intent=f"Type member {MEMBER} into the Member ID field",
                action=ActionType.TYPE,
                target=target,
                value=ValueRef(param="member_id"),
                post_condition=checkpoint,
            )
        ],
        success=Checkpoint(
            description=f"member {MEMBER} detail shown", text_present=f"Member {MEMBER}"
        ),
        provenance=Provenance(
            recorded_by_model="m", discovery_run_id="r", recorded_at="2026-01-01T00:00:00Z"
        ),
    )


def _dump(cap: Capability) -> str:
    return cap.model_dump_json()


# --------------------------------------------------------------- C1


def test_sensitive_value_never_survives_into_the_artifact():
    """The whole finding: the value appeared in descriptions and checkpoints."""
    cap = _cap(
        target=LocatorBundle(
            description=f"the Member ID box next to {MEMBER}",
            strategies=[
                SemanticLocator(role="textbox", name="Member ID"),
                TextLocator(text="Member ID"),
            ],
        ),
        checkpoint=Checkpoint(
            description=f"the row reading Member ID {MEMBER} is present",
            text_present=f"Member ID {MEMBER}",
        ),
    )
    assert MEMBER in _dump(cap), "fixture must start contaminated"

    clean = scrub_capability(cap, {"member_id": MEMBER})

    assert MEMBER not in _dump(clean)
    assert "<param:member_id>" in clean.description
    assert "<param:member_id>" in clean.steps[0].intent
    # a checkpoint asserting one member's text is a coincidence, not a checkpoint
    assert clean.steps[0].post_condition.text_present is None
    assert clean.success.text_present is None


def test_contaminated_locator_strategy_is_dropped_not_rewritten():
    """Rewriting a match value would break resolution; a locator keyed on one
    member's id was never reusable anyway, so it is dropped."""
    cap = _cap(
        target=LocatorBundle(
            description="the field",
            strategies=[
                TextLocator(text=f"Member ID {MEMBER}"),  # contaminated
                SemanticLocator(role="textbox", name="Member ID"),  # clean
            ],
        ),
        checkpoint=Checkpoint(description="ok", url_matches="^http://localhost:5055/"),
    )
    clean = scrub_capability(cap, {"member_id": MEMBER})
    kept = clean.steps[0].target.strategies

    assert len(kept) == 1
    assert isinstance(kept[0], SemanticLocator)
    assert MEMBER not in _dump(clean)


def test_fully_contaminated_locator_fails_compilation_rather_than_shipping():
    cap = _cap(
        target=LocatorBundle(
            description="the field",
            strategies=[TextLocator(text=f"Member ID {MEMBER}")],
        ),
        checkpoint=Checkpoint(description="ok", url_matches="^http://"),
    )
    with pytest.raises(ContaminatedLocator):
        scrub_capability(cap, {"member_id": MEMBER})


def test_non_sensitive_values_are_left_alone():
    cap = _cap(
        target=LocatorBundle(
            description="the field",
            strategies=[SemanticLocator(role="textbox", name="Member ID")],
        ),
        checkpoint=Checkpoint(description="ok", url_matches="^http://"),
    )
    cap.inputs[0].sensitivity = Sensitivity.PUBLIC
    clean = scrub_capability(cap, {"member_id": MEMBER})
    assert MEMBER in _dump(clean), "public values must not be scrubbed"


# --------------------------------------------------------------- C2


class _FakeSurface:
    """Records every action that actually reached the surface."""

    def __init__(self) -> None:
        self.acted: list = []

    def observe(self) -> Observation:
        return Observation(
            url="http://evil.example.com/login",
            title="Login",
            nodes=(
                Node(ref="n1", role="textbox", name="Operator ID"),
                Node(ref="n2", role="textbox", name="Password"),
                Node(ref="n3", role="button", name="Sign On"),
            ),
            text_digest="Sign on",
        )

    def act(self, action) -> ActResult:
        self.acted.append(action)
        return ActResult(ok=True)

    def read(self, target):
        return None

    def wait_for(self, checkpoint, timeout_ms):
        return True

    def snapshot(self, label):
        return None

    def describe(self):
        return SurfaceInfo(kind="fake", app_id="a")


def _localhost_only_gate() -> PolicyGate:
    return PolicyGate(
        Policy(
            name="t",
            mode="unattended",
            allowed_origins=["http://localhost:5055"],
            allowed_routes=["/", "/login", "/search"],
            allowed_action_types=["navigate", "click", "type", "read", "wait"],
        )
    )


def test_authentication_is_refused_off_the_allowlist(monkeypatch):
    """The finding: login ran un-gated, so credentials could be typed into a
    page on any origin the surface happened to be showing."""
    monkeypatch.setenv("CUA_APP_USER", "demo")
    monkeypatch.setenv("CUA_APP_PASS", "demo")
    surface = _FakeSurface()

    with pytest.raises(AuthPolicyDenied):
        authenticate(surface, gate=_localhost_only_gate(), ready_text="Search Criteria")

    assert surface.acted == [], "no credential may reach an off-allowlist page"


def test_authentication_without_a_gate_still_works_for_hermetic_use(monkeypatch):
    monkeypatch.setenv("CUA_APP_USER", "demo")
    monkeypatch.setenv("CUA_APP_PASS", "demo")
    surface = _FakeSurface()
    with pytest.raises(Exception):
        # no gate: actions reach the surface, and it fails later on ready_text
        authenticate(surface, ready_text="Search Criteria")
    assert len(surface.acted) == 3, "user, password, submit"
