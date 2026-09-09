"""Surface tests.

Everything here is hermetic: no browser, no network, no API key.  The one test
that needs a real Chromium is guarded by `requires_browser` and skips cleanly
when Playwright (or its browser binary) is absent.
"""

from __future__ import annotations

import importlib.util

import pytest

from cua.artifact import ActionType, Checkpoint, LocatorBundle, SemanticLocator
from cua.surface.base import Action, Node, Observation, Surface, SurfaceInfo
from cua.surface.factory import build_surface
from cua.surface.recorded import (
    RecordedSurface,
    dump_observations,
    load_observations,
    observation_from_dict,
    observation_to_dict,
)
from cua.surface.web import WebSurface, WebSurfaceConfig


# --------------------------------------------------------------------------
# Availability guards
# --------------------------------------------------------------------------

_HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


def _browser_ok() -> bool:
    if not _HAS_PLAYWRIGHT:
        return False
    try:
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        try:
            b = pw.chromium.launch(headless=True)
            b.close()
            return True
        finally:
            pw.stop()
    except Exception:
        return False


requires_browser = pytest.mark.skipif(
    not _browser_ok(), reason="Playwright and/or its Chromium binary is not available"
)

_HAS_RESOLVER = importlib.util.find_spec("cua.locator.resolve") is not None
requires_resolver = pytest.mark.skipif(
    not _HAS_RESOLVER, reason="cua.locator.resolve not implemented yet"
)


# --------------------------------------------------------------------------
# Fixtures: a hand-built two-screen legacy app (frameset + nested table).
# --------------------------------------------------------------------------


def _login_frame() -> Observation:
    return Observation(
        url="http://legacy.test/login",
        title="ACME Legacy :: Sign In",
        text_digest="Sign In User ID Password Submit",
        screenshot_path="evidence/0000-login.png",
        nodes=(
            Node(
                ref="n1",
                role="textbox",
                name="User ID",
                value="",
                frame_path=("mainFrame",),
                bbox=(0.30, 0.40, 0.20, 0.03),
                dom_path="html/body/table/tr[2]/td[2]/input",
                text="User ID",
            ),
            Node(
                ref="n2",
                role="textbox",
                name="Password",
                value="",
                frame_path=("mainFrame",),
                bbox=(0.30, 0.45, 0.20, 0.03),
                dom_path="html/body/table/tr[3]/td[2]/input",
                text="Password",
            ),
            Node(
                ref="n3",
                role="button",
                name="Submit",
                frame_path=("mainFrame",),
                bbox=(0.30, 0.52, 0.08, 0.03),
                dom_path="html/body/table/tr[4]/td[2]/input",
                text="Submit",
            ),
            Node(
                ref="n4",
                role="link",
                name="Help",
                frame_path=("navFrame",),
                bbox=(0.02, 0.10, 0.10, 0.02),
                dom_path="html/body/a",
            ),
        ),
    )


def _account_frame() -> Observation:
    return Observation(
        url="http://legacy.test/account",
        title="ACME Legacy :: Account",
        text_digest="Account Summary Balance 1042.55 Sign Out",
        screenshot_path="evidence/0001-account.png",
        nodes=(
            Node(
                ref="n1",
                role="heading",
                name="Account Summary",
                frame_path=("mainFrame",),
                bbox=(0.05, 0.08, 0.4, 0.04),
            ),
            Node(
                ref="n2",
                role="textbox",
                name="Balance",
                value="1042.55",
                frame_path=("mainFrame", "detailFrame"),
                bbox=(0.35, 0.30, 0.15, 0.03),
                dom_path="html/body/table/tr[1]/td[2]/input",
                text="Balance 1042.55",
            ),
            Node(
                ref="n3",
                role="button",
                name="Sign Out",
                frame_path=("navFrame",),
                bbox=(0.85, 0.02, 0.10, 0.03),
            ),
            Node(
                ref="n4",
                role="button",
                name="Archive",
                frame_path=("mainFrame",),
                enabled=False,
            ),
        ),
    )


@pytest.fixture
def frames() -> list[Observation]:
    return [_login_frame(), _account_frame()]


@pytest.fixture
def rec(frames) -> RecordedSurface:
    return RecordedSurface(frames=frames, app_id="acme-legacy", tenant_id="t1")


def _bundle(role: str, name: str) -> LocatorBundle:
    return LocatorBundle(
        description=f"the {name} {role}",
        strategies=[SemanticLocator(role=role, name=name)],
    )


# --------------------------------------------------------------------------
# The seam: both implementations satisfy the protocol.
# --------------------------------------------------------------------------


def test_recorded_satisfies_surface_protocol(rec):
    assert isinstance(rec, Surface)


def test_web_satisfies_surface_protocol():
    # No browser needed: the protocol check is structural, and WebSurface's
    # constructor tolerates a page object that does nothing.
    surface = WebSurface(page=_DeadPage(), app_id="acme-legacy")
    assert isinstance(surface, Surface)


class _DeadPage:
    """Minimal stand-in so WebSurface can be constructed without Playwright."""

    context = None

    def set_default_timeout(self, ms):  # pragma: no cover - trivial
        raise RuntimeError("no browser")


def test_both_surfaces_are_interchangeable_to_upstream(rec):
    for s in (rec, WebSurface(page=_DeadPage(), app_id="x")):
        info = s.describe()
        assert isinstance(info, SurfaceInfo)
        assert info.kind in {"web", "recorded"}
        assert info.app_id


def test_describe_recorded(rec):
    info = rec.describe()
    assert info.kind == "recorded"
    assert info.app_id == "acme-legacy"
    assert info.tenant_id == "t1"
    assert "replay" in info.capabilities


def test_web_describe_defaults_headed():
    s = WebSurface(page=_DeadPage(), app_id="x")
    assert s.config.headless is False  # human handoff is the default
    assert "human_handoff" in s.describe().capabilities
    assert s.describe().kind == "web"


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def test_factory_builds_recorded_from_frames(frames):
    s = build_surface("recorded", frames=frames, app_id="acme")
    assert isinstance(s, RecordedSurface)
    assert isinstance(s, Surface)
    assert s.describe().kind == "recorded"


def test_factory_builds_recorded_from_directory(frames, tmp_path):
    dump_observations(frames, tmp_path / "run")
    s = build_surface("recorded", directory=str(tmp_path / "run"))
    assert len(s.frames) == 2
    assert s.observe().url == "http://legacy.test/login"


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown surface kind"):
        build_surface("smoke-signals")


def test_factory_builds_web_without_launching(monkeypatch):
    s = build_surface("web", page=_DeadPage(), app_id="x", config={"headless": True})
    assert isinstance(s, WebSurface)
    assert isinstance(s.config, WebSurfaceConfig)
    assert s.config.headless is True


# --------------------------------------------------------------------------
# RecordedSurface: observe -> act -> advance
# --------------------------------------------------------------------------


def test_observe_is_side_effect_free(rec):
    a = rec.observe()
    b = rec.observe()
    assert a is b
    assert rec.index == 0


def test_click_advances_one_frame(rec):
    assert rec.observe().url.endswith("/login")
    res = rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert res.ok, res.detail
    assert rec.index == 1
    assert rec.observe().url.endswith("/account")


def test_type_sets_nothing_but_advances(rec):
    res = rec.act(Action(type=ActionType.TYPE, ref="n1", value="alice"))
    assert res.ok
    assert rec.index == 1


def test_advance_clamps_at_last_frame(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert rec.index == len(rec.frames) - 1


def test_transition_map_overrides_default_advance(frames):
    s = RecordedSurface(frames=frames, transitions={(0, ActionType.CLICK, "n3"): 1, 1: 0})
    assert s.act(Action(type=ActionType.CLICK, ref="n3")).ok
    assert s.index == 1
    assert s.act(Action(type=ActionType.CLICK, ref="n3")).ok
    assert s.index == 0  # bounced back to login


def test_navigate_jumps_by_url(rec):
    res = rec.act(Action(type=ActionType.NAVIGATE, url="http://legacy.test/account"))
    assert res.ok, res.detail
    assert rec.observe().title.endswith("Account")


def test_navigate_to_unrecorded_url_fails_debuggably(rec):
    res = rec.act(Action(type=ActionType.NAVIGATE, url="http://elsewhere.test/"))
    assert not res.ok
    assert "no matching recorded frame" in res.detail
    assert "legacy.test" in res.detail  # tells you what WAS recorded
    assert rec.index == 0


def test_unknown_ref_fails_without_advancing(rec):
    res = rec.act(Action(type=ActionType.CLICK, ref="n999"))
    assert not res.ok
    assert "n999" in res.detail
    assert rec.index == 0


def test_action_without_ref_or_target_fails(rec):
    res = rec.act(Action(type=ActionType.CLICK))
    assert not res.ok
    assert "neither ref nor target" in res.detail


def test_disabled_node_is_refused(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))  # to account frame
    res = rec.act(Action(type=ActionType.CLICK, ref="n4"))  # Archive, disabled
    assert not res.ok
    assert "disabled" in res.detail


def test_wait_action_is_a_noop(rec):
    res = rec.act(Action(type=ActionType.WAIT))
    assert res.ok
    assert rec.index == 0


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------


def test_read_returns_value_and_does_not_advance(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert rec.read_ref("n2") == "1042.55"
    assert rec.index == 1


def test_read_falls_back_to_text_then_name(rec):
    assert rec.read_ref("n3") == "Submit"


def test_read_missing_ref_returns_none(rec):
    assert rec.read_ref("nope") is None


@requires_resolver
def test_read_by_locator_bundle(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert rec.read(_bundle("textbox", "Balance")) == "1042.55"


def test_read_by_locator_without_resolver_is_a_clean_failure(rec):
    res = rec.act(Action(type=ActionType.READ, target=_bundle("textbox", "Balance")))
    if not _HAS_RESOLVER:
        assert not res.ok
        assert "resolver unavailable" in res.detail
    else:
        assert res.ok or "resolve" in res.detail


# --------------------------------------------------------------------------
# wait_for
# --------------------------------------------------------------------------


def test_wait_for_text_present_success(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert rec.wait_for(Checkpoint(description="landed", text_present="Account Summary"), 1000)


def test_wait_for_url_matches_success(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert rec.wait_for(Checkpoint(description="url", url_matches=r"/account$"), 1000)


def test_wait_for_text_absent_success(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert rec.wait_for(Checkpoint(description="gone", text_absent="Password"), 1000)


def test_wait_for_times_out_without_sleeping(rec):
    import time

    t0 = time.monotonic()
    ok = rec.wait_for(Checkpoint(description="never", text_present="Nirvana"), 5000)
    assert ok is False
    assert time.monotonic() - t0 < 0.5  # deterministic, no blind sleeps


def test_wait_for_empty_checkpoint_is_not_trivially_true(rec):
    assert rec.wait_for(Checkpoint(description="declares nothing"), 1000) is False


def test_wait_for_can_look_ahead_when_enabled(frames):
    s = RecordedSurface(frames=frames, advance_on_wait=True)
    assert s.index == 0
    assert s.wait_for(Checkpoint(description="eventual", text_present="Balance"), 3000)
    assert s.index == 1


def test_wait_for_lookahead_disabled_by_default(rec):
    assert rec.wait_for(Checkpoint(description="eventual", text_present="Balance"), 3000) is False
    assert rec.index == 0


# --------------------------------------------------------------------------
# Injected failure
# --------------------------------------------------------------------------


def test_injected_failure_blocks_act_and_freezes_cursor(rec):
    rec.inject_failure(0, "session expired banner")
    res = rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert not res.ok
    assert "injected failure at step 0" in res.detail
    assert "session expired banner" in res.detail
    assert rec.index == 0


def test_injected_failure_at_later_step_only(rec):
    rec.inject_failure(1, "boom")
    assert rec.act(Action(type=ActionType.CLICK, ref="n3")).ok
    assert rec.index == 1
    assert not rec.act(Action(type=ActionType.CLICK, ref="n3")).ok
    assert rec.index == 1


def test_reset_restores_the_cursor(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    rec.reset()
    assert rec.index == 0
    assert rec.log == []


def test_act_log_records_every_call(rec):
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    rec.read_ref("n2")
    assert rec.log == ["click@0", "read@1"]


def test_empty_recording_is_rejected():
    with pytest.raises(ValueError):
        RecordedSurface(frames=[])


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------


def test_snapshot_returns_recorded_path(rec):
    assert rec.snapshot("start") == "evidence/0000-login.png"
    rec.act(Action(type=ActionType.CLICK, ref="n3"))
    assert rec.snapshot("after") == "evidence/0001-account.png"


# --------------------------------------------------------------------------
# Serialization round-trip
# --------------------------------------------------------------------------


def test_observation_dict_round_trip(frames):
    for obs in frames:
        assert observation_from_dict(observation_to_dict(obs)) == obs


def test_directory_round_trip_preserves_frames_and_replay(frames, tmp_path):
    dump_observations(frames, tmp_path / "run")
    loaded = load_observations(tmp_path / "run")
    assert loaded == frames

    s = RecordedSurface.from_dir(tmp_path / "run")
    assert s.act(Action(type=ActionType.CLICK, ref="n3")).ok
    assert s.read_ref("n2") == "1042.55"


def test_load_from_empty_directory_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_observations(tmp_path / "empty")


def test_frame_path_and_bbox_survive_serialization(frames, tmp_path):
    dump_observations(frames, tmp_path / "run")
    loaded = load_observations(tmp_path / "run")
    balance = next(n for n in loaded[1].nodes if n.name == "Balance")
    assert balance.frame_path == ("mainFrame", "detailFrame")
    assert balance.bbox == (0.35, 0.30, 0.15, 0.03)
    assert all(0.0 <= v <= 1.0 for v in balance.bbox)


# --------------------------------------------------------------------------
# Observation.render()
# --------------------------------------------------------------------------


def test_render_is_compact_and_informative(frames):
    out = frames[0].render()
    assert "URL: http://legacy.test/login" in out
    assert "TITLE: ACME Legacy :: Sign In" in out
    assert "CONTROLS:" in out
    assert '[n3] button "Submit"' in out
    assert "(in mainFrame)" in out
    assert "(in navFrame)" in out
    assert len(out.splitlines()) == 3 + len(frames[0].nodes)


def test_render_marks_values_and_disabled(frames):
    out = frames[1].render()
    assert 'value="1042.55"' in out
    assert "[disabled]" in out
    assert "(in mainFrame/detailFrame)" in out


def test_render_truncates_and_says_so(frames):
    out = frames[1].render(limit=2)
    assert "... 2 more controls omitted" in out


# --------------------------------------------------------------------------
# Live browser (skipped unless Chromium is actually available)
# --------------------------------------------------------------------------


@requires_browser
def test_web_observe_traverses_frames(tmp_path):
    inner = tmp_path / "inner.html"
    inner.write_text(
        "<html><body><button>Inner Button</button></body></html>", encoding="utf-8"
    )
    outer = tmp_path / "outer.html"
    outer.write_text(
        f"<html><body><h1>Outer</h1>"
        f'<iframe name="detailFrame" src="{inner.as_uri()}"></iframe>'
        f"</body></html>",
        encoding="utf-8",
    )

    cfg = WebSurfaceConfig(headless=True, evidence_dir=str(tmp_path / "evidence"))
    with WebSurface.launch("test-app", start_url=outer.as_uri(), config=cfg) as s:
        obs = s.observe()
        inner_nodes = [n for n in obs.nodes if n.name == "Inner Button"]
        assert inner_nodes, obs.render()
        n = inner_nodes[0]
        assert n.frame_path == ("detailFrame",)
        assert n.bbox is not None and all(0.0 <= v <= 1.0 for v in n.bbox)
        assert s.snapshot("frames") is not None
