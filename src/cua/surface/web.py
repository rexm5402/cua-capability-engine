"""`WebSurface` -- the Playwright (sync API) implementation of `Surface`.

Design commitments, in order of importance:

1.  **The accessibility tree is the source of truth, not CSS.**  Every node in an
    `Observation` comes from Playwright's `page.accessibility.snapshot()`, which
    is the same representation a screen reader (and a desktop automation API)
    sees.  CSS selectors appear only as a *recorded, distrusted* `dom_path`
    (LocatorTier.STRUCTURAL).  This is what makes the surface seam portable to a
    native desktop app later.

2.  **Frames are traversed, not ignored.**  The target class of app is a hostile
    legacy surface: framesets, nested tables, no test ids.  A snapshot of the
    main frame alone would see nothing.  We walk `page.main_frame` and every
    descendant frame, recursively, and stamp every node with the `frame_path`
    that reached it.

3.  **Never guess.**  When an action is addressed by `LocatorBundle`, we
    re-observe and delegate to `cua.locator.resolve.resolve`.  Ambiguity or
    failure returns `ok=False` with a debuggable `detail` -- never a silent
    `first()`.

4.  **Headed by default.**  A human must be able to take over the same live
    session, so `launch()` defaults to `headless=False`.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from cua.artifact import ActionType, Checkpoint, LocatorBundle
from cua.surface.base import (
    Action,
    ActResult,
    Node,
    Observation,
    SurfaceInfo,
)

# --------------------------------------------------------------------------
# Resolver seam.  Another module owns `resolve`; we import it lazily so this
# file is importable (and testable) before that module lands.
# --------------------------------------------------------------------------


def _load_resolver():
    from cua.locator.resolve import resolve  # noqa: PLC0415

    return resolve


# Roles that carry no interactive or informational value for the LLM.  Dropping
# them keeps `Observation.render()` inside a sane token budget on legacy pages
# that nest fifteen tables deep.
_NOISE_ROLES = frozenset(
    {
        "generic",
        "none",
        "presentation",
        "InlineTextBox",
        "LineBreak",
        "RootWebArea",
    }
)

_VALUE_ROLES = frozenset(
    {"textbox", "combobox", "searchbox", "spinbutton", "slider", "checkbox", "radio"}
)

_WS = re.compile(r"\s+")


def _norm(s: str | None) -> str:
    return _WS.sub(" ", s or "").strip()


# --------------------------------------------------------------------------


@dataclass
class WebSurfaceConfig:
    headless: bool = False  # HEADED by default: a human takes over this session.
    viewport: tuple[int, int] = (1280, 900)
    evidence_dir: str = "evidence"
    max_nodes: int = 1500
    default_timeout_ms: int = 10_000
    trace: bool = False


class WebSurface:
    """A live browser page, seen as six buttons."""

    def __init__(
        self,
        page: Any,
        app_id: str,
        tenant_id: str | None = None,
        config: WebSurfaceConfig | None = None,
        context: Any = None,
        _owned: tuple[Any, ...] = (),
    ) -> None:
        self.page = page
        self.context = context if context is not None else getattr(page, "context", None)
        self.app_id = app_id
        self.tenant_id = tenant_id
        self.config = config or WebSurfaceConfig()
        self._owned = _owned
        self._last_obs: Observation | None = None
        self._ref_index: dict[str, tuple[Node, Any]] = {}
        self._tracing = False
        Path(self.config.evidence_dir).mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            self.page.set_default_timeout(self.config.default_timeout_ms)

    # ---------------------------------------------------------------- launch

    @classmethod
    @contextlib.contextmanager
    def launch(
        cls,
        app_id: str,
        start_url: str | None = None,
        tenant_id: str | None = None,
        config: WebSurfaceConfig | None = None,
    ) -> Iterator["WebSurface"]:
        """Boot Chromium and yield a ready `WebSurface`.

        Headed unless explicitly configured otherwise -- handoff to a human is a
        product requirement, not a debugging convenience.
        """
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        cfg = config or WebSurfaceConfig()
        Path(cfg.evidence_dir).mkdir(parents=True, exist_ok=True)
        pw = sync_playwright().start()
        browser = None
        context = None
        try:
            browser = pw.chromium.launch(headless=cfg.headless)
            context = browser.new_context(
                viewport={"width": cfg.viewport[0], "height": cfg.viewport[1]}
            )
            if cfg.trace:
                context.tracing.start(screenshots=True, snapshots=True, sources=False)
            page = context.new_page()
            surface = cls(
                page,
                app_id=app_id,
                tenant_id=tenant_id,
                config=cfg,
                context=context,
            )
            surface._tracing = cfg.trace
            if start_url:
                page.goto(start_url)
            yield surface
        finally:
            if cfg.trace and context is not None:
                with contextlib.suppress(Exception):
                    context.tracing.stop(
                        path=str(Path(cfg.evidence_dir) / f"{app_id}-trace.zip")
                    )
            if browser is not None:
                with contextlib.suppress(Exception):
                    browser.close()
            with contextlib.suppress(Exception):
                pw.stop()

    def close(self) -> None:
        for obj in reversed(self._owned):
            with contextlib.suppress(Exception):
                if hasattr(obj, "close"):
                    obj.close()
                elif hasattr(obj, "stop"):
                    obj.stop()

    # --------------------------------------------------------------- observe

    def observe(self) -> Observation:
        """Build an `Observation` from the accessibility tree of every frame.

        Frame traversal
        ---------------
        We start at ``page.main_frame`` with ``frame_path = ()`` and recurse
        through ``frame.child_frames``.  Each child contributes one segment to
        the path, chosen in this order of preference:

          1. the frame's ``name`` attribute (legacy framesets almost always name
             their frames -- ``navFrame``, ``mainFrame``);
          2. the frame's URL path basename;
          3. ``f{i}`` positional fallback.

        Segments are made unique among siblings by appending ``#n`` on
        collision, so the path is deterministic and replayable.  Every node
        emitted from a frame carries that frame's full path, outermost first.
        This is exactly the ``scope`` / ``frame_path`` that
        ``SemanticLocator`` and ``StructuralLocator`` consume, so resolution can
        re-enter the same frame on replay.

        bbox normalization
        ------------------
        Playwright reports element boxes in **page CSS pixels relative to the
        top-level viewport** (frame offsets already applied), so a node inside an
        iframe needs no manual offset arithmetic.  We divide by the *viewport*
        size, not the document scroll size, and add the current scroll offset so
        coordinates describe the document position:

            x = (box.x + scroll_x) / viewport_w  ... clamped into [0, 1]

        The result is resolution-independent and matches ``GeometryLocator``'s
        ``ge=0.0, le=1.0`` constraints.  Boxes are clamped rather than dropped:
        a partially off-screen control is still a real control.
        """
        vw, vh = self._viewport()
        nodes: list[Node] = []
        self._ref_index = {}
        counter = [0]

        self._walk_frame(self.page.main_frame, (), nodes, counter, vw, vh)

        obs = Observation(
            url=self._safe(lambda: self.page.url, ""),
            title=self._safe(lambda: self.page.title(), ""),
            nodes=tuple(nodes[: self.config.max_nodes]),
            text_digest=self._text_digest(),
        )
        self._last_obs = obs
        return obs

    # -- frame traversal ---------------------------------------------------

    def _frame_segment(self, frame: Any, index: int, taken: set[str]) -> str:
        name = _norm(getattr(frame, "name", "") or "")
        if not name:
            url = self._safe(lambda: frame.url, "") or ""
            tail = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
            name = tail or f"f{index}"
        seg = name
        n = 1
        while seg in taken:
            n += 1
            seg = f"{name}#{n}"
        taken.add(seg)
        return seg

    def _walk_frame(
        self,
        frame: Any,
        frame_path: tuple[str, ...],
        out: list[Node],
        counter: list[int],
        vw: float,
        vh: float,
    ) -> None:
        snap = self._safe(lambda: frame.accessibility.snapshot(interesting_only=False), None)
        if snap:
            self._flatten_ax(frame, snap, frame_path, out, counter, vw, vh, depth=0)

        taken: set[str] = set()
        children = self._safe(lambda: list(frame.child_frames), [])
        for i, child in enumerate(children):
            seg = self._frame_segment(child, i, taken)
            self._walk_frame(child, frame_path + (seg,), out, counter, vw, vh)

    def _flatten_ax(
        self,
        frame: Any,
        ax: dict,
        frame_path: tuple[str, ...],
        out: list[Node],
        counter: list[int],
        vw: float,
        vh: float,
        depth: int,
    ) -> None:
        if len(out) >= self.config.max_nodes:
            return
        role = ax.get("role") or ""
        name = _norm(ax.get("name"))
        keep = role not in _NOISE_ROLES and (name or role in _VALUE_ROLES or role == "button")

        if keep:
            counter[0] += 1
            ref = f"n{counter[0]}"
            handle = self._handle_for(frame, role, name)
            node = Node(
                ref=ref,
                role=role,
                name=name,
                value=self._value_of(ax, handle),
                frame_path=frame_path,
                bbox=self._bbox(handle, vw, vh),
                dom_path=self._dom_path(handle),
                text=self._nearby_text(handle) or name or None,
                enabled=not bool(ax.get("disabled")),
            )
            out.append(node)
            self._ref_index[ref] = (node, handle)

        for child in ax.get("children") or []:
            self._flatten_ax(frame, child, frame_path, out, counter, vw, vh, depth + 1)

    # -- per-node enrichment ----------------------------------------------

    def _handle_for(self, frame: Any, role: str, name: str) -> Any:
        """Best-effort live handle for an ax node, via ARIA role query only.

        We deliberately never synthesize a CSS selector here.  If the
        role+name pair is not unique in the frame we return ``None`` rather than
        picking one; the node still exists in the observation, it simply lacks
        geometry.  Guessing is the failure mode this whole design exists to
        avoid.
        """
        if not role:
            return None
        try:
            loc = frame.get_by_role(role, name=name, exact=True) if name else frame.get_by_role(role)
            if loc.count() != 1:
                return None
            return loc.first
        except Exception:
            return None

    def _bbox(self, handle: Any, vw: float, vh: float) -> tuple[float, float, float, float] | None:
        if handle is None or vw <= 0 or vh <= 0:
            return None
        box = self._safe(lambda: handle.bounding_box(), None)
        if not box:
            return None

        def clamp(v: float) -> float:
            return max(0.0, min(1.0, v))

        return (
            clamp(box["x"] / vw),
            clamp(box["y"] / vh),
            clamp(box["width"] / vw),
            clamp(box["height"] / vh),
        )

    def _dom_path(self, handle: Any) -> str | None:
        """Recorded for STRUCTURAL tier -- deliberately distrusted, never used
        as the primary way to find anything."""
        if handle is None:
            return None
        return self._safe(
            lambda: handle.evaluate(
                """el => { const p=[]; let n=el;
                    while (n && n.nodeType===1 && p.length<25) {
                      let s=n.nodeName.toLowerCase();
                      const par=n.parentNode;
                      if (par) { const sibs=[...par.children].filter(c=>c.nodeName===n.nodeName);
                        if (sibs.length>1) s += '[' + (sibs.indexOf(n)+1) + ']'; }
                      p.unshift(s); n=par; }
                    return p.join('/'); }"""
            ),
            None,
        )

    def _nearby_text(self, handle: Any) -> str | None:
        """Text of the nearest meaningful ancestor -- the row/cell label that is
        the only stable anchor on a table-based legacy screen."""
        if handle is None:
            return None
        raw = self._safe(
            lambda: handle.evaluate(
                """el => { let n=el;
                    for (let i=0;i<4 && n;i++) {
                      n = n.parentElement;
                      if (!n) break;
                      const t=(n.innerText||'').trim();
                      if (t && t.length<200) return t;
                    } return ''; }"""
            ),
            None,
        )
        raw = _norm(raw)
        return raw[:200] or None

    def _value_of(self, ax: dict, handle: Any) -> str | None:
        v = ax.get("value")
        if v not in (None, ""):
            return str(v)
        if handle is None:
            return None
        return self._safe(lambda: handle.input_value(), None) or None

    def _viewport(self) -> tuple[float, float]:
        vp = self._safe(lambda: self.page.viewport_size, None)
        if vp:
            return float(vp["width"]), float(vp["height"])
        return float(self.config.viewport[0]), float(self.config.viewport[1])

    def _text_digest(self, limit: int = 4000) -> str:
        """Concatenated visible text across ALL frames -- what `text_present` /
        `text_absent` checkpoints are evaluated against."""
        parts: list[str] = []

        def collect(frame: Any) -> None:
            t = self._safe(lambda: frame.inner_text("body"), "") or ""
            if t:
                parts.append(_norm(t))
            for child in self._safe(lambda: list(frame.child_frames), []):
                collect(child)

        collect(self.page.main_frame)
        return " \n".join(parts)[:limit]

    @staticmethod
    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    # ------------------------------------------------------------------ act

    def act(self, action: Action) -> ActResult:
        t = action.type

        if t is ActionType.NAVIGATE:
            url = action.url or (action.value or "")
            if not url:
                return ActResult(ok=False, detail="NAVIGATE requires a url")
            try:
                self.page.goto(url)
                return ActResult(ok=True, detail=f"navigated to {url}")
            except Exception as e:
                return ActResult(ok=False, detail=f"navigate failed: {e!r}")

        if t is ActionType.WAIT:
            # No blind sleeps: WAIT means 'settle the network', an explicit
            # browser-observable condition.
            try:
                self.page.wait_for_load_state("networkidle")
                return ActResult(ok=True, detail="load state networkidle")
            except Exception as e:
                return ActResult(ok=False, detail=f"wait failed: {e!r}")

        node, handle, res = self._locate(action)
        if node is None:
            return res  # already a debuggable failure

        tier = res.resolved_tier
        cands = res.candidates

        if t is ActionType.READ:
            val = node.value if node.value is not None else (node.text or node.name)
            if handle is not None:
                val = (
                    self._safe(lambda: handle.input_value(), None)
                    or self._safe(lambda: _norm(handle.inner_text()), None)
                    or val
                )
            return ActResult(
                ok=True, detail=f"read {node.ref}", resolved_tier=tier,
                candidates=cands, read_value=val,
            )

        if handle is None:
            return ActResult(
                ok=False,
                detail=(
                    f"resolved node {node.ref} ({node.role} '{node.name}' in "
                    f"{'/'.join(node.frame_path) or 'main'}) has no unique live handle; "
                    "refusing to guess which element to act on"
                ),
                resolved_tier=tier,
                candidates=cands,
            )

        try:
            if t is ActionType.CLICK:
                handle.click()
            elif t is ActionType.TYPE:
                handle.fill(action.value or "")
            elif t is ActionType.SELECT:
                handle.select_option(action.value or "")
            elif t is ActionType.PRESS:
                handle.press(action.value or "Enter")
            else:
                return ActResult(ok=False, detail=f"unsupported action type {t}")
        except Exception as e:
            return ActResult(
                ok=False,
                detail=f"{t.value} on {node.ref} failed: {e!r}",
                resolved_tier=tier,
                candidates=cands,
            )

        return ActResult(
            ok=True, detail=f"{t.value} on {node.ref}", resolved_tier=tier, candidates=cands
        )

    def _locate(self, action: Action) -> tuple[Node | None, Any, ActResult]:
        """Return (node, live_handle, result-carrying-tier) or a failure."""
        if action.ref:
            entry = self._ref_index.get(action.ref)
            if entry is None:
                return None, None, ActResult(
                    ok=False,
                    detail=(
                        f"ref {action.ref!r} not found in last observation "
                        f"({len(self._ref_index)} refs); observe() again before acting"
                    ),
                )
            node, handle = entry
            return node, handle, ActResult(ok=True, resolved_tier=None, candidates=1)

        if action.target is None:
            return None, None, ActResult(
                ok=False, detail="action has neither ref nor target locator bundle"
            )

        try:
            resolve = _load_resolver()
        except Exception as e:
            return None, None, ActResult(
                ok=False, detail=f"locator resolver unavailable: {e!r}"
            )

        obs = self.observe()  # always resolve against a FRESH observation
        try:
            r = resolve(action.target, obs)
        except Exception as e:
            return None, None, ActResult(
                ok=False,
                detail=f"resolve({action.target.description!r}) raised: {e!r}",
            )

        node = getattr(r, "node", None)
        tier = getattr(r, "tier", None)
        cands = getattr(r, "candidates", 0) or 0
        reason = getattr(r, "reason", "") or ""
        if node is None:
            return None, None, ActResult(
                ok=False,
                detail=(
                    f"could not uniquely resolve {action.target.description!r}: "
                    f"{reason} (candidates={cands}, url={obs.url})"
                ),
                resolved_tier=int(tier) if tier is not None else None,
                candidates=cands,
            )

        handle = self._ref_index.get(node.ref, (None, None))[1]
        return node, handle, ActResult(
            ok=True,
            resolved_tier=int(tier) if tier is not None else None,
            candidates=cands,
        )

    # ----------------------------------------------------------------- read

    def read(self, target: LocatorBundle) -> str | None:
        res = self.act(Action(type=ActionType.READ, target=target))
        return res.read_value if res.ok else None

    # ------------------------------------------------------------- wait_for

    def wait_for(self, checkpoint: Checkpoint, timeout_ms: int) -> bool:
        """Poll the DECLARED conditions until all hold, or time out.

        There is no blind sleep anywhere in this method: the only sleeping is
        the poll interval between explicit condition evaluations, and a
        checkpoint with no conditions is treated as unsatisfiable rather than
        as trivially true.
        """
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        interval = 0.15
        last = "no conditions declared on checkpoint"
        while True:
            ok, last = self._check(checkpoint)
            if ok:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)
            interval = min(interval * 1.5, 1.0)

    def _check(self, cp: Checkpoint) -> tuple[bool, str]:
        conditions = 0
        obs = self.observe()

        if cp.url_matches is not None:
            conditions += 1
            if not re.search(cp.url_matches, obs.url):
                return False, f"url {obs.url!r} !~ {cp.url_matches!r}"

        haystack = obs.text_digest
        if cp.text_present is not None:
            conditions += 1
            if cp.text_present.lower() not in haystack.lower():
                return False, f"text_present {cp.text_present!r} not found"

        if cp.text_absent is not None:
            conditions += 1
            if cp.text_absent.lower() in haystack.lower():
                return False, f"text_absent {cp.text_absent!r} still present"

        if cp.locator is not None:
            conditions += 1
            try:
                resolve = _load_resolver()
                r = resolve(cp.locator, obs)
            except Exception as e:
                return False, f"resolver unavailable/raised: {e!r}"
            if getattr(r, "node", None) is None:
                return False, f"locator {cp.locator.description!r} unresolved"

        if conditions == 0:
            return False, "checkpoint declares no conditions"
        return True, "all conditions satisfied"

    # ------------------------------------------------------------- snapshot

    def snapshot(self, label: str) -> str | None:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "snapshot"
        path = Path(self.config.evidence_dir) / f"{int(time.time() * 1000)}-{safe}.png"
        try:
            self.page.screenshot(path=str(path), full_page=True)
        except Exception:
            return None
        return str(path)

    # ------------------------------------------------------------- describe

    def describe(self) -> SurfaceInfo:
        return SurfaceInfo(
            kind="web",
            app_id=self.app_id,
            tenant_id=self.tenant_id,
            capabilities=frozenset(
                {
                    "navigate",
                    "click",
                    "type",
                    "select",
                    "press",
                    "read",
                    "wait",
                    "frames",
                    "screenshot",
                    "accessibility_tree",
                    "human_handoff" if not self.config.headless else "headless",
                }
            ),
        )
