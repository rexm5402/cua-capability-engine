"""Command line entry point.

Three verbs matching the three things this system does: discover a flow once
with a model, replay it forever without one, and hand a stuck session to a
human.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

DEFAULT_APP = "http://localhost:5055"
# One shared directory the engine and the console process both see. The lease
# and the request store live here for the same reason: the operator console is
# a different process, so in-memory state would make the handoff imaginary.
SESSION_ROOT = "evidence/session"


def _run_id(kind: str) -> str:
    from datetime import datetime, timezone

    return f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def _parse_inputs(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--input expects name=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


def _load_profile(path: str | None):
    """App-level knowledge (signals, outcomes, product identity) kept separate
    from any one capability, so another institution on the same vendor product
    is a data overlay rather than a re-recording."""
    if not path:
        return None
    import yaml

    from cua.artifact import OutcomeSpec, SignalRule
    from cua.discovery.compile import AppProfile

    d = yaml.safe_load(Path(path).read_text()) or {}
    return AppProfile(
        app_id=d.get("app_id", "unknown"),
        vendor_product=d.get("vendor_product", "unknown"),
        product_version=d.get("product_version"),
        tenant_id=d.get("tenant_id"),
        signals=[SignalRule.model_validate(x) for x in d.get("signals", [])],
        outcomes=[OutcomeSpec.model_validate(x) for x in d.get("outcomes", [])],
    )


def _load_capability(path: str):
    import yaml

    from cua.artifact import Capability

    data = yaml.safe_load(Path(path).read_text())
    return Capability.model_validate(data)


def _gate(policy_path: str | None):
    from cua.policy.gate import Policy, PolicyGate

    if policy_path:
        return PolicyGate(Policy.from_yaml(policy_path))
    default = Path(__file__).parent / "policy" / "policy.yaml"
    return PolicyGate(Policy.from_yaml(default)) if default.exists() else PolicyGate()


def _surface(url: str, *, headless: bool, app_id: str, tenant: str | None):
    from cua.surface.web import WebSurface, WebSurfaceConfig

    return WebSurface.launch(
        app_id,
        start_url=url,
        tenant_id=tenant,
        config=WebSurfaceConfig(headless=headless),
    )


def _print_outcome(outcome: Any) -> int:
    """Exit codes distinguish the three result types, so a caller -- or CI --
    can tell a legitimate business answer from a broken run."""
    print(json.dumps(outcome.model_dump(), indent=2, default=str))
    kind = getattr(outcome, "kind", "")
    if kind == "success":
        return 0
    if kind == "business_outcome":
        print(f"\nBusiness outcome: {outcome.code} -- {outcome.message}", file=sys.stderr)
        return 3
    print(
        f"\nFAILED at step {outcome.at_step_id}: expected {outcome.expected!r}, "
        f"observed {outcome.observed!r}",
        file=sys.stderr,
    )
    return 1


# --------------------------------------------------------------------------


def cmd_discover(a: argparse.Namespace) -> int:
    from cua.discovery.agent import discover
    from cua.discovery.compile import compile_trajectory
    from cua.discovery.verify import verify_and_save
    from cua.evidence.recorder import EvidenceRecorder
    from cua.llm.client import build_client
    from cua.session.provider import authenticate

    llm = build_client(a.provider, a.model)
    print(f"discovery: model={llm.model} vision={llm.supports_vision}")

    inputs = _parse_inputs(a.input)
    gate = _gate(a.policy)

    with EvidenceRecorder(_run_id("discovery"), "discovery") as rec:
        with _surface(a.url, headless=a.headless, app_id=a.app_id, tenant=a.tenant) as s:
            if not a.no_auth:
                # Authenticate FIRST so the recorded flow begins post-login and
                # no credential can reach the artifact.
                authenticate(s)
            traj = discover(
                a.goal, a.url, s, llm=llm, inputs=inputs, gate=gate,
                recorder=rec, max_steps=a.max_steps,
            )
        if not traj.succeeded:
            print(f"discovery did not complete: {traj.stop_reason}", file=sys.stderr)
            return 1
        cap = compile_trajectory(
            traj, a.goal,
            profile=_load_profile(a.profile),
            name=a.name,
            entry_url=a.url,
        )

    # Verify BEFORE saving: an artifact reaches disk only once it has been
    # proven to run with the model switched off.
    # Each verification replay gets a FRESH, separately-authenticated session:
    # reusing the discovery session would let leftover state make a broken
    # artifact look reproducible.
    # Verification replays run SEQUENTIALLY, and Playwright's sync API cannot
    # be nested in one thread, so each new session closes the previous one.
    slot: list[ExitStack] = []

    def factory(*_args, **_kw):
        while slot:
            slot.pop().close()
        stack = ExitStack()
        slot.append(stack)
        s = stack.enter_context(
            _surface(a.url, headless=True, app_id=a.app_id, tenant=a.tenant)
        )
        if cap.auth.required and not a.no_auth:
            authenticate(s)
        return s

    def _report(r):
        print(f"verification: mode={r.mode} ok={r.ok}")
        if r.reason:
            print(f"  reason: {r.reason}")

    try:
        saved = verify_and_save(
            cap, factory, a.out,
            discovery_inputs=inputs,
            verify_inputs=_parse_inputs(a.verify_input) or None,
            on_report=_report,
        )
    finally:
        while slot:
            slot.pop().close()
    if saved is None:
        print(
            "verification replay FAILED -- artifact not written. The flow the "
            "model found could not be reproduced without it.",
            file=sys.stderr,
        )
        return 1
    print(f"verified and saved: {saved}")
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    from cua.evidence.recorder import EvidenceRecorder
    from cua.replay.engine import replay
    from cua.session.provider import authenticate
    from cua.telemetry import TelemetryStore

    cap = _load_capability(a.artifact)
    inputs = _parse_inputs(a.input)
    url = cap.entry_url

    with EvidenceRecorder(
        _run_id("replay"), "replay", capability_id=cap.id
    ) as rec:
        with _surface(url, headless=a.headless, app_id=cap.app_profile.app_id,
                      tenant=a.tenant) as s:
            if cap.auth.required and not a.no_auth:
                authenticate(s)
            if a.inject:
                # Arm the fault AFTER authenticating. The fixture's universal
                # modes fire on the next request, so arming it on the entry URL
                # would break the login rather than the flow under test.
                from cua.artifact import ActionType
                from cua.surface.base import Action

                base = url.split("/search")[0]
                s.act(Action(type=ActionType.NAVIGATE,
                             url=f"{base}/admin/inject/{a.inject}"))
                s.act(Action(type=ActionType.NAVIGATE, url=url))
            outcome = replay(
                cap, inputs, s, gate=_gate(a.policy), recorder=rec,
                telemetry=TelemetryStore(),
            )
    return _print_outcome(outcome)


def cmd_console(a: argparse.Namespace) -> int:
    import uvicorn

    from cua.session.console import create_app
    from cua.session.escalation import RequestStore
    from cua.session.lease import SessionLease

    store = RequestStore(a.session_root)
    lease = SessionLease(Path(a.session_root) / "session.lease")
    print(f"operator console on http://127.0.0.1:{a.port}  (watching {a.session_root})")
    uvicorn.run(
        create_app(store, lease, screenshot_root="evidence"),
        host="127.0.0.1",
        port=a.port,
        log_level="warning",
    )
    return 0


def cmd_catalog(a: argparse.Namespace) -> int:
    """Every saved artifact as a callable tool definition -- what an AI agent
    would discover and invoke by name with typed args."""
    tools = []
    for p in sorted(Path(a.dir).glob("*.yaml")):
        try:
            tools.append(_load_capability(str(p)).tool_schema())
        except Exception as e:  # noqa: BLE001
            print(f"skipping {p.name}: {e}", file=sys.stderr)
    print(json.dumps(tools, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("cua", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--policy", help="path to a policy YAML")
        sp.add_argument("--tenant", default=None)
        sp.add_argument("--headless", action="store_true",
                        help="headed by default so a human can take over")
        sp.add_argument("--input", action="append", default=[], metavar="NAME=VALUE")
        sp.add_argument("--no-auth", action="store_true",
                        help="skip pre-authentication")

    d = sub.add_parser("discover", help="LLM-driven discovery run")
    d.add_argument("--goal", required=True)
    d.add_argument("--url", default=f"{DEFAULT_APP}/search")
    d.add_argument("--out", required=True)
    d.add_argument("--app-id", default="corebanking")
    d.add_argument("--provider", default=None, help="xai (default) | openai")
    d.add_argument("--model", default=None)
    d.add_argument("--max-steps", type=int, default=25)
    d.add_argument("--profile", default=None,
                   help="app profile YAML supplying signals, outcomes and "
                        "product identity (see profiles/)")
    d.add_argument("--name", default=None, help="capability name")
    d.add_argument("--verify-input", action="append", default=[], metavar="NAME=VALUE",
                   help="a SECOND, real input used for the verification replay. "
                        "Verification deliberately replays with a different "
                        "value so a literal baked in place of a parameter "
                        "fails; a derived value can match the shape of a real "
                        "one but cannot know which values actually exist in "
                        "the system, so the operator supplies it.")
    common(d)
    d.set_defaults(func=cmd_discover)

    r = sub.add_parser("replay", help="deterministic replay -- no LLM")
    r.add_argument("artifact")
    r.add_argument("--inject", default=None,
                   help="fault to inject into the fixture app (not_found, "
                        "dialog, perm_denied, validation, timeout, slow, error)")
    common(r)
    r.set_defaults(func=cmd_replay)

    c = sub.add_parser("console", help="operator console for human handoff")
    c.add_argument("--port", type=int, default=5056)
    c.add_argument("--session-root", default=SESSION_ROOT)
    c.set_defaults(func=cmd_console)

    k = sub.add_parser("catalog", help="saved artifacts as callable tool schemas")
    k.add_argument("--dir", default="artifacts")
    k.set_defaults(func=cmd_catalog)

    a = p.parse_args(argv)
    return a.func(a)
