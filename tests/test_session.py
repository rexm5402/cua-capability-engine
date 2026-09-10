"""Hermetic tests for the session subsystem. No browser, no network.

The concurrency tests use real threads AND a real subprocess, because the
claim being tested is cross-process mutual exclusion. A test that only used a
lock object inside one interpreter would prove nothing about the design.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from cua.artifact import Checkpoint
from cua.outcomes import Failure
from cua.session.escalation import (
    Aborted,
    EscalationTimeout,
    Escalator,
    InterventionRequest,
    MarkedFailed,
    PermittedAction,
    ReasonCode,
    RequestStore,
    Resumed,
)
from cua.session.handoff import POST_CONDITION_UNMET, diff_observations, handoff
from cua.session.lease import (
    LeaseExpired,
    LeaseState,
    LeaseStateMismatch,
    NotHolder,
    SessionLease,
)
from cua.surface.base import Node, Observation

REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")


# ==========================================================================
# Fakes
# ==========================================================================


class FakeSurface:
    """Six methods, no Playwright. `wait_for` answers from the current
    observation only -- it never waits, which keeps the tests fast."""

    def __init__(self, observation: Observation) -> None:
        self.observation = observation
        self.observe_calls = 0
        self.wait_for_calls: list[Checkpoint] = []

    def observe(self) -> Observation:
        self.observe_calls += 1
        return self.observation

    def act(self, action):  # pragma: no cover - unused here
        raise NotImplementedError

    def read(self, target):  # pragma: no cover - unused here
        raise NotImplementedError

    def wait_for(self, checkpoint: Checkpoint, timeout_ms: int) -> bool:
        self.wait_for_calls.append(checkpoint)
        obs = self.observation
        if checkpoint.url_matches and checkpoint.url_matches not in obs.url:
            return False
        if checkpoint.text_present and checkpoint.text_present not in obs.text_digest:
            return False
        if checkpoint.text_absent and checkpoint.text_absent in obs.text_digest:
            return False
        return True

    def snapshot(self, label: str):  # pragma: no cover - unused here
        return None

    def describe(self):  # pragma: no cover - unused here
        raise NotImplementedError


def obs(url="https://app.test/confirm", title="Confirm", text="", nodes=()) -> Observation:
    return Observation(url=url, title=title, nodes=tuple(nodes), text_digest=text)


@pytest.fixture()
def lease(tmp_path: Path) -> SessionLease:
    return SessionLease(tmp_path / "session.lease", ttl_s=60.0)


@pytest.fixture()
def store(tmp_path: Path) -> RequestStore:
    return RequestStore(tmp_path / "escalations")


def make_request(store: RequestStore, **kw) -> InterventionRequest:
    esc = Escalator(store, run_id="run-1")
    defaults = dict(
        capability_id="cap.confirm_change",
        capability_name="confirm_beneficiary_change",
        step_id="s7",
        step_intent=(
            "Confirm the beneficiary change so the policy record is written back"
        ),
        reason_code=ReasonCode.APPROVAL_REQUIRED,
        why="irreversible action requires human approval",
        current_url="https://app.test/confirm",
    )
    defaults.update(kw)
    return esc.raise_request(**defaults)


# ==========================================================================
# PART 1 -- the lease
# ==========================================================================


def test_two_concurrent_thread_acquires_exactly_one_wins(lease: SessionLease) -> None:
    barrier = threading.Barrier(8)
    winners: list[str] = []
    losers: list[Exception] = []
    lock = threading.Lock()

    def attempt(i: int) -> None:
        barrier.wait()
        try:
            g = lease.acquire(f"engine-{i}", expected_state=LeaseState.SUSPENDED)
            with lock:
                winners.append(g.holder)
        except Exception as exc:
            with lock:
                losers.append(exc)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert len(winners) == 1, f"expected exactly one winner, got {winners}"
    assert len(losers) == 7
    assert all(isinstance(e, (LeaseHeldOrMismatch)) for e in losers), losers

    rec = lease.read()
    assert rec.state is LeaseState.AUTOMATION
    assert rec.holder == winners[0]
    assert rec.fence == 1


# Both refusals are correct outcomes of a lost CAS: the loser either sees the
# winner's state (mismatch) or races into the held branch.
LeaseHeldOrMismatch = (LeaseStateMismatch,)


_CHILD = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from cua.session.lease import SessionLease, LeaseState, LeaseError
    lease = SessionLease(sys.argv[2])
    try:
        g = lease.acquire(sys.argv[3], expected_state=LeaseState.SUSPENDED)
        print(json.dumps({"ok": True, "holder": g.holder, "fence": g.fence}))
    except LeaseError as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
    """
)


def test_concurrent_acquires_across_processes_exactly_one_wins(tmp_path: Path) -> None:
    """The real claim: mutual exclusion between the engine and the console,
    which are separate OS processes."""
    path = tmp_path / "session.lease"
    script = tmp_path / "child.py"
    script.write_text(_CHILD)

    procs = [
        subprocess.Popen(
            [sys.executable, str(script), REPO_SRC, str(path), f"proc-{i}"],
            stdout=subprocess.PIPE,
            text=True,
        )
        for i in range(6)
    ]
    results = []
    for p in procs:
        out, _ = p.communicate(timeout=60)
        results.append(json.loads(out.strip()))

    winners = [r for r in results if r["ok"]]
    assert len(winners) == 1, results
    assert winners[0]["fence"] == 1
    assert SessionLease(path).read().holder == winners[0]["holder"]


def test_preempted_holder_cannot_resume(lease: SessionLease) -> None:
    g = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)

    # Operator takes over, then hands to a *different* engine acquisition.
    op = lease.transfer("engine", "operator", token=g.token)
    lease.transfer("operator", "engine", token=op.token,
                   to_state=LeaseState.AUTOMATION)

    # The original grant is stale by two fences. It must not be able to act...
    with pytest.raises(NotHolder):
        lease.validate("engine", g.token)
    # ...nor release, nor transfer, even though its *holder name* still matches.
    with pytest.raises(NotHolder):
        lease.release("engine", token=g.token)
    with pytest.raises(NotHolder):
        lease.transfer("engine", "somebody", token=g.token)

    # And the fence alone rejects it even if it somehow learned the token.
    cur = lease.read()
    with pytest.raises(NotHolder):
        lease.validate("engine", cur.holder_token, fence=g.fence)


def test_ttl_expiry_is_explicit_and_recoverable(tmp_path: Path) -> None:
    lease = SessionLease(tmp_path / "s.lease", ttl_s=0.05)
    g = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)
    time.sleep(0.08)

    # Explicit, not silent: the lease still reports itself HELD and EXPIRED.
    rec = lease.read()
    assert rec.state is LeaseState.AUTOMATION
    assert rec.holder == "engine"
    assert rec.is_expired()

    # The holder is told it lapsed rather than being allowed to carry on.
    with pytest.raises(LeaseExpired):
        lease.validate("engine", g.token)

    # A newcomer must not silently steal it either.
    with pytest.raises(LeaseExpired):
        lease.acquire("operator", expected_state=LeaseState.SUSPENDED)

    # Recovery is a named operation, and it bumps the fence.
    reclaimed = lease.expire_if_stale()
    assert reclaimed.state is LeaseState.SUSPENDED
    assert reclaimed.fence == g.fence + 1
    assert "expired" in reclaimed.note

    g2 = lease.acquire("operator", expected_state=LeaseState.SUSPENDED)
    assert g2.fence == g.fence + 2
    # The crashed holder still cannot come back.
    with pytest.raises(NotHolder):
        lease.validate("engine", g.token)


def test_heartbeat_keeps_a_live_holder_alive(tmp_path: Path) -> None:
    lease = SessionLease(tmp_path / "s.lease", ttl_s=0.08)
    g = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)
    time.sleep(0.05)
    lease.heartbeat("engine", g.token)
    time.sleep(0.05)
    assert not lease.read().is_expired()
    assert lease.validate("engine", g.token).fence == g.fence


def test_state_machine_rejects_illegal_transitions(lease: SessionLease) -> None:
    # Cannot acquire into SUSPENDED -- that is the unheld state.
    with pytest.raises(LeaseStateMismatch):
        lease.acquire("engine", target_state=LeaseState.SUSPENDED)

    g = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)

    # AUTOMATION -> AUTOMATION is not a transition; it is a double-acquire.
    with pytest.raises(LeaseStateMismatch):
        lease.transfer("engine", "engine-2", token=g.token, to_state=LeaseState.AUTOMATION)

    # A second party cannot acquire while it is held.
    with pytest.raises(LeaseStateMismatch):
        lease.acquire("operator", expected_state=LeaseState.SUSPENDED)

    # Wrong expected_state on a CAS is refused.
    with pytest.raises(LeaseStateMismatch):
        lease.acquire("operator", expected_state=LeaseState.OPERATOR)

    op = lease.transfer("engine", "operator", token=g.token)
    assert lease.read().state is LeaseState.OPERATOR
    with pytest.raises(LeaseStateMismatch):
        lease.transfer("operator", "op2", token=op.token, to_state=LeaseState.OPERATOR)

    # A non-holder may not release.
    with pytest.raises(NotHolder):
        lease.release("engine", token=g.token)

    lease.release("operator", token=op.token)
    assert lease.read().state is LeaseState.SUSPENDED
    # Releasing an unheld lease is a refusal, not a no-op.
    with pytest.raises(NotHolder):
        lease.release("operator", token=op.token)


def test_wait_for_state_sees_another_thread_transition(lease: SessionLease) -> None:
    g = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)

    def later() -> None:
        time.sleep(0.05)
        lease.transfer("engine", "operator", token=g.token)

    threading.Thread(target=later).start()
    rec = lease.wait_for_state(LeaseState.OPERATOR, timeout=5.0)
    assert rec.holder == "operator"

    with pytest.raises(TimeoutError):
        lease.wait_for_state(LeaseState.SUSPENDED, timeout=0.1)


def test_corrupt_lease_file_fails_closed(tmp_path: Path) -> None:
    p = tmp_path / "s.lease"
    p.write_text("{not json")
    with pytest.raises(Exception):
        SessionLease(p).read()


# ==========================================================================
# PART 2 -- escalation
# ==========================================================================


def test_request_round_trips_to_a_separate_reader_with_step_intent(
    tmp_path: Path, store: RequestStore
) -> None:
    req = make_request(store)

    # A *different* store object, as the console process would construct.
    console_store = RequestStore(store.root)
    listed = console_store.list_requests(open_only=True)
    assert [r.request_id for r in listed] == [req.request_id]

    fetched = console_store.get_request(req.request_id)
    assert fetched is not None
    assert fetched.step_intent == (
        "Confirm the beneficiary change so the policy record is written back"
    )
    assert fetched.reason_code is ReasonCode.APPROVAL_REQUIRED
    assert PermittedAction.RESUME in fetched.permitted_actions

    # And the on-disk payload itself carries the intent -- an operator reading
    # raw JSON with no code at hand can still act.
    raw = json.loads((store.requests_dir / f"{req.request_id}.json").read_text())
    assert "beneficiary change" in raw["step_intent"]
    assert raw["current_url"] == "https://app.test/confirm"


def test_resolving_a_request_removes_it_from_the_open_list(store: RequestStore) -> None:
    req = make_request(store)
    assert store.list_requests(open_only=True)
    RequestStore(store.root).put_resolution(
        Resumed(request_id=req.request_id, operator_id="alice")
    )
    assert store.list_requests(open_only=True) == []
    assert len(store.list_requests(open_only=False)) == 1


@pytest.mark.parametrize(
    "resolution_factory, expected_kind",
    [
        (lambda rid: Resumed(request_id=rid, operator_id="alice"), "resumed"),
        (lambda rid: Aborted(request_id=rid, operator_id="bob"), "aborted"),
        (lambda rid: MarkedFailed(request_id=rid, operator_id="carol"), "marked_failed"),
    ],
)
def test_await_resolution_returns_each_kind(
    store: RequestStore, resolution_factory, expected_kind: str
) -> None:
    esc = Escalator(store, run_id="run-1")
    req = make_request(store)

    def console() -> None:
        time.sleep(0.05)
        RequestStore(store.root).put_resolution(resolution_factory(req.request_id))

    threading.Thread(target=console).start()
    res = esc.await_resolution(req.request_id, timeout=5.0)
    assert res.kind == expected_kind
    assert res.request_id == req.request_id


def test_await_resolution_times_out_rather_than_assuming_yes(store: RequestStore) -> None:
    esc = Escalator(store, run_id="run-1")
    req = make_request(store)
    with pytest.raises(EscalationTimeout):
        esc.await_resolution(req.request_id, timeout=0.1)


def test_three_trigger_sources_share_one_queue(store: RequestStore) -> None:
    esc = Escalator(store, run_id="run-1")
    esc.for_no_progress(
        capability_id="cap.a",
        capability_name="search_member",
        step_intent="find the member record",
        steps_without_progress=5,
    )
    esc.for_replay_failure(
        capability_name="search_member",
        failure=Failure(
            capability_id="cap.a",
            code="locator_ambiguous",
            at_step_id="s3",
            step_intent="click the View button on the member row",
            expected="one match",
            observed="eight matches",
        ),
    )

    @dataclass
    class _Decision:
        reason: str = "irreversible action requires human approval"

    esc.for_approval(
        capability_id="cap.a",
        capability_name="search_member",
        step_intent="submit the change",
        decision=_Decision(),
        proposed_action="click Submit",
        risk_class="irreversible",
    )

    reqs = RequestStore(store.root).list_requests()
    assert {r.reason_code for r in reqs} == {
        ReasonCode.NO_STATE_PROGRESS,
        ReasonCode.REPLAY_FAILURE,
        ReasonCode.APPROVAL_REQUIRED,
    }
    # Every one of them carries the intent, which is the operator's real input.
    assert all(r.step_intent for r in reqs)
    failure_req = next(r for r in reqs if r.reason_code is ReasonCode.REPLAY_FAILURE)
    assert failure_req.step_id == "s3"
    assert "eight matches" in failure_req.why


# ==========================================================================
# PART 3 -- handoff and resume semantics
# ==========================================================================


def _resolve_soon(store: RequestStore, request_id: str, resolution) -> None:
    def go() -> None:
        time.sleep(0.03)
        RequestStore(store.root).put_resolution(resolution)

    threading.Thread(target=go).start()


CHECK = Checkpoint(
    description="the confirmation receipt is on screen",
    url_matches="/receipt",
    text_present="Change confirmed",
)


def test_resume_reverifies_post_condition_and_fails_when_human_left_wrong_state(
    lease: SessionLease, store: RequestStore
) -> None:
    """The human clicked Resume, but left the session on the wrong screen.
    The engine must surface a Failure, not continue."""
    esc = Escalator(store, run_id="run-1")
    req = make_request(store)
    grant = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)

    surface = FakeSurface(obs(url="https://app.test/confirm", text="Confirm change?"))

    def after_human() -> None:
        time.sleep(0.03)
        # The operator wandered off to an unrelated screen and hit Resume.
        surface.observation = obs(
            url="https://app.test/dashboard", title="Dashboard", text="Welcome back"
        )
        RequestStore(store.root).put_resolution(
            Resumed(request_id=req.request_id, operator_id="alice", note="looks fine")
        )

    threading.Thread(target=after_human).start()

    with handoff(
        lease, esc, surface, req, grant=grant, timeout=5.0, post_condition=CHECK
    ) as outcome:
        assert outcome.resumed is True, "the human did say resume"
        assert outcome.verified is False
        assert outcome.may_continue is False
        assert isinstance(outcome.failure, Failure)
        assert outcome.failure.code == POST_CONDITION_UNMET
        assert outcome.failure.escalated is True
        assert outcome.failure.at_step_id == "s7"
        assert "beneficiary change" in (outcome.failure.step_intent or "")
        assert "dashboard" in outcome.failure.observed

    # The lease came back to the engine, on a new fence.
    rec = lease.read()
    assert rec.state is LeaseState.AUTOMATION
    assert rec.holder == "engine"
    assert rec.fence == grant.fence + 2


def test_resume_continues_when_the_human_left_the_expected_state(
    lease: SessionLease, store: RequestStore
) -> None:
    esc = Escalator(store, run_id="run-1")
    req = make_request(store)
    grant = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)
    surface = FakeSurface(obs(url="https://app.test/confirm", text="Confirm change?"))

    def after_human() -> None:
        time.sleep(0.03)
        surface.observation = obs(
            url="https://app.test/receipt",
            title="Receipt",
            text="Change confirmed. Reference 88213.",
        )
        RequestStore(store.root).put_resolution(
            Resumed(request_id=req.request_id, operator_id="alice")
        )

    threading.Thread(target=after_human).start()

    with handoff(
        lease, esc, surface, req, grant=grant, timeout=5.0, post_condition=CHECK
    ) as outcome:
        assert outcome.may_continue is True
        assert outcome.failure is None
        assert surface.wait_for_calls == [CHECK], "post-condition must be re-checked"
        assert surface.observe_calls >= 2, "must re-observe, not reuse the old view"


def test_lease_is_held_by_the_operator_while_the_human_is_driving(
    lease: SessionLease, store: RequestStore
) -> None:
    esc = Escalator(store, run_id="run-1")
    req = make_request(store)
    grant = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)
    surface = FakeSurface(obs(url="https://app.test/receipt", text="Change confirmed"))
    seen: list[tuple[str, str | None]] = []

    def console() -> None:
        # A separate reader observes the lease mid-handoff.
        lease.wait_for_state(LeaseState.OPERATOR, timeout=5.0)
        rec = lease.read()
        seen.append((rec.state.value, rec.holder))
        # ...and the engine cannot act while the operator holds it.
        try:
            lease.validate("engine", grant.token)
            seen.append(("engine-could-still-act", None))
        except NotHolder:
            seen.append(("engine-locked-out", None))
        RequestStore(store.root).put_resolution(
            Resumed(request_id=req.request_id, operator_id="alice")
        )

    t = threading.Thread(target=console)
    t.start()
    with handoff(
        lease, esc, surface, req, grant=grant, timeout=5.0, post_condition=CHECK
    ) as outcome:
        assert outcome.may_continue is True
    t.join(timeout=5)

    assert seen[0] == ("operator", "operator-console")
    assert seen[1][0] == "engine-locked-out"


def test_abort_and_marked_failed_do_not_continue(
    lease: SessionLease, store: RequestStore
) -> None:
    esc = Escalator(store, run_id="run-1")
    surface = FakeSurface(obs(url="https://app.test/receipt", text="Change confirmed"))

    req1 = make_request(store)
    g1 = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)
    _resolve_soon(store, req1.request_id, Aborted(request_id=req1.request_id, operator_id="bob"))
    with handoff(lease, esc, surface, req1, grant=g1, timeout=5.0, post_condition=CHECK) as o:
        assert o.resumed is False and o.may_continue is False and o.failure is None
        assert surface.wait_for_calls == [], "no point verifying an aborted run"
    g1b = o.grant

    req2 = make_request(store)
    _resolve_soon(
        store,
        req2.request_id,
        MarkedFailed(request_id=req2.request_id, operator_id="carol", note="app is broken"),
    )
    with handoff(lease, esc, surface, req2, grant=g1b, timeout=5.0, post_condition=CHECK) as o2:
        assert o2.may_continue is False
        assert isinstance(o2.failure, Failure)
        assert o2.failure.code == "human_marked_failed"
        assert o2.failure.observed == "app is broken"
        assert o2.failure.escalated is True


def test_human_intervention_records_the_observation_delta(
    lease: SessionLease, store: RequestStore
) -> None:
    esc = Escalator(store, run_id="run-1")
    req = make_request(store)
    grant = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)

    before_nodes = (
        Node(ref="n1", role="textbox", name="Beneficiary", value=""),
        Node(ref="n2", role="button", name="Confirm"),
    )
    surface = FakeSurface(
        obs(url="https://app.test/confirm", text="Confirm change?", nodes=before_nodes)
    )

    def after_human() -> None:
        time.sleep(0.03)
        surface.observation = obs(
            url="https://app.test/receipt",
            title="Receipt",
            text="Change confirmed. Reference 88213.",
            nodes=(
                Node(ref="x1", role="textbox", name="Beneficiary", value="J. Okafor"),
                Node(ref="x3", role="link", name="Print receipt"),
            ),
        )
        RequestStore(store.root).put_resolution(
            Resumed(request_id=req.request_id, operator_id="alice", note="typed it manually")
        )

    threading.Thread(target=after_human).start()
    with handoff(
        lease, esc, surface, req, grant=grant, timeout=5.0, post_condition=CHECK
    ) as outcome:
        rec = outcome.intervention

    assert rec is not None
    assert rec.operator_id == "alice"
    assert rec.resolution_kind == "resumed"
    assert rec.duration_s >= 0
    assert rec.url_before == "https://app.test/confirm"
    assert rec.url_after == "https://app.test/receipt"

    delta = rec.observation_delta
    assert delta["url_changed"] is True
    assert delta["value_changes"] == [
        {"role": "textbox", "name": "Beneficiary", "before": "", "after": "J. Okafor"}
    ]
    assert "link 'Print receipt'" in delta["controls_appeared"]
    assert "button 'Confirm'" in delta["controls_disappeared"]

    joined = " | ".join(rec.actions_inferred)
    assert "J. Okafor" in joined and "navigated" in joined

    # It is durable and visible to a separate reader (evidence/audit path).
    persisted = RequestStore(store.root).get_intervention(req.request_id)
    assert persisted is not None
    assert persisted.actions_inferred == rec.actions_inferred


def test_diff_observations_reports_no_change_honestly() -> None:
    o = obs(url="https://app.test/x", text="same")
    delta = diff_observations(o, o)
    assert delta["url_changed"] is False
    assert delta["controls_appeared"] == []
    from cua.session.handoff import infer_actions

    assert "no observable change" in infer_actions(delta)[0]


def test_missing_post_condition_is_flagged_not_silently_passed(
    lease: SessionLease, store: RequestStore, tmp_path: Path
) -> None:
    """A step with nothing to assert cannot be *verified*; we say so in the
    evidence rather than pretending."""
    from cua.evidence.recorder import EvidenceRecorder

    esc = Escalator(store, run_id="run-1")
    req = make_request(store)
    grant = lease.acquire("engine", expected_state=LeaseState.SUSPENDED)
    surface = FakeSurface(obs())
    _resolve_soon(store, req.request_id, Resumed(request_id=req.request_id, operator_id="al"))

    rec = EvidenceRecorder("run-1", "replay", root=tmp_path / "ev")
    with handoff(
        lease, esc, surface, req, grant=grant, timeout=5.0, post_condition=None, recorder=rec
    ) as outcome:
        assert outcome.may_continue is True
    rec.finish(None)

    log = (rec.dir / "run.log").read_text()
    assert "no post_condition declared" in log
    steps = [json.loads(line) for line in (rec.dir / "steps.jsonl").read_text().splitlines()]
    assert steps[-1]["step_id"] == "s7"
    assert steps[-1]["extra"]["human_intervention"]["operator_id"] == "al"


# ==========================================================================
# PART 4 -- console
# ==========================================================================


def test_console_lists_and_resolves_through_the_store(store: RequestStore) -> None:
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("multipart", reason="python-multipart needed for form posts")
    from fastapi.testclient import TestClient

    from cua.session.console import create_app

    req = make_request(store)
    # The console builds its OWN store handle -- it shares nothing but the dir.
    client = TestClient(create_app(RequestStore(store.root)))

    body = client.get("/").text
    assert req.request_id in body
    assert "beneficiary change" in body

    detail = client.get(f"/requests/{req.request_id}").text
    assert "STEP INTENT" in detail
    assert "https://app.test/confirm" in detail

    r = client.post(
        f"/requests/{req.request_id}/resume",
        data={"operator_id": "alice"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    res = store.get_resolution(req.request_id)
    assert res is not None and res.kind == "resumed" and res.operator_id == "alice"

    # First resolution wins: a second operator clicking must not overwrite it.
    r2 = client.post(f"/requests/{req.request_id}/abort", data={"operator_id": "bob"})
    assert r2.status_code == 409
    assert store.get_resolution(req.request_id).kind == "resumed"
    assert client.get("/").text.count(req.request_id) == 0
