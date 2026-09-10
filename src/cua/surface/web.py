"""`WebSurface` -- the Playwright (sync API) implementation of `Surface`.

Design commitments, in order of importance:

1.  **The accessibility tree is the source of truth, not CSS.**  Every node in an
    `Observation` comes from the accessibility tree -- read over CDP
    (`Accessibility.getFullAXTree`, per frame), because Playwright 1.62 removed
    the `page.accessibility` / `frame.accessibility` API entirely.  That tree is
    the same representation a screen reader (and a desktop automation API) sees.  CSS selectors appear only as a *recorded, distrusted* `dom_path`
    (LocatorTier.STRUCTURAL).  This is what makes the surface seam portable to a
    native desktop app later.

2.  **Frames are traversed, not ignored.**  The target class of app is a hostile
    legacy surface: framesets, nested tables, no test ids.  A snapshot of the
    main frame alone would see nothing -- and `Accessibility.getFullAXTree`
    with no `frameId` returns exactly that.  We walk `Page.getFrameTree`
    recursively, ask for the AX tree of EVERY frame id, and stamp every node
    with the `frame_path` that reached it.

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
from datetime import datetime, timezone
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

#: AX roles that carry printed text rather than a control.  We keep them as
#: nodes -- the anchor-relative tier is the workhorse on table-based legacy
#: screens and it needs the printed label to EXIST as a node -- but we skip the
#: DOM round trip that only pays off for something you can click or type into.
_TEXTUAL_ROLES = frozenset(
    {
        "StaticText",
        "text",
        "paragraph",
        "caption",
        "legend",
        "cell",
        "gridcell",
        "columnheader",
        "rowheader",
        "row",
        "table",
        "rowgroup",
        "listitem",
        "list",
        "heading",
        "label",
    }
)

_MARK_ATTR = "data-cua-ref"

_MARK_JS = """function(marker){
    const el = this.nodeType === 1 ? this : this.parentElement;
    if (!el) return false;
    el.setAttribute('%s', marker);
    return true;
}""" % _MARK_ATTR

_ELEMENT_INFO_JS = """function(){
    const el = this.nodeType === 1 ? this : this.parentElement;
    if (!el) return null;
    const p = []; let n = el;
    while (n && n.nodeType === 1 && p.length < 25) {
      let s = n.nodeName.toLowerCase();
      const par = n.parentNode;
      if (par && par.children) {
        const sibs = [...par.children].filter(c => c.nodeName === n.nodeName);
        if (sibs.length > 1) s += '[' + (sibs.indexOf(n) + 1) + ']';
      }
      p.unshift(s); n = par;
    }
    let near = '', m = el;
    for (let i = 0; i < 4 && m; i++) {
      m = m.parentElement;
      if (!m) break;
      const t = (m.innerText || '').trim();
      if (t && t.length < 200) { near = t.slice(0, 200); break; }
    }
    let val = null;
    try { if (typeof el.value === 'string') val = el.value; } catch (e) {}
    return {dom_path: p.join('/'), near: near, value: val};
}"""


@dataclass(frozen=True)
class _HandleSpec:
    """Everything needed to (lazily) re-acquire a unique live handle."""

    ref: str
    frame: Any
    role: str
    name: str
    backend_id: int | None


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
        self._ref_index: dict[str, tuple[Node, "_HandleSpec"]] = {}
        self._cdp_session: Any = None
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
        self._cdp_session = None
        for obj in reversed(self._owned):
            with contextlib.suppress(Exception):
                if hasattr(obj, "close"):
                    obj.close()
                elif hasattr(obj, "stop"):
                    obj.stop()

    # --------------------------------------------------------------- observe

    def _settle_frames(self, timeout_ms: int = 6000) -> None:
        """Wait for the frame tree to stop changing before snapshotting.

        A frameset's children are still empty for a beat after the parent
        navigates, so snapshotting immediately yields an EMPTY observation --
        which is precisely what happens on our target app's post-login screen.

        This is a bounded wait on an explicit condition (every frame has a URL
        and the set of frames has stopped changing), not a blind sleep. The
        determinism rule forbids sleeping a fixed interval and hoping; it does
        not forbid waiting for a stated condition with a deadline.
        """
        page = self.page
        deadline = time.monotonic() + timeout_ms / 1000.0
        last: tuple[str, ...] | None = None
        stable = 0
        while time.monotonic() < deadline:
            try:
                frames = tuple(sorted(f.url for f in page.frames))
                ready = all(f.url for f in page.frames)
            except Exception:  # noqa: BLE001 - page may be navigating
                return
            if ready and frames == last:
                stable += 1
                if stable >= 2:
                    return
            else:
                stable = 0
                last = frames
            try:
                page.wait_for_timeout(50)
            except Exception:  # noqa: BLE001
                return

    def observe(self) -> Observation:
        """Build an `Observation` from the accessibility tree of every frame.

        Frame traversal
        ---------------
        We start at the root of ``Page.getFrameTree`` with ``frame_path = ()``
        and recurse through ``childFrames``.  Each child contributes one segment
        to the path, chosen in this order of preference:

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
        ``DOM.getBoxModel`` reports boxes in **root-frame viewport CSS pixels**
        (frame offsets already applied), so a node inside an iframe needs no
        manual offset arithmetic.  We divide by the *viewport* size:

            x = box.x / viewport_w  ... clamped into [0, 1]

        The result is resolution-independent and matches ``GeometryLocator``'s
        ``ge=0.0, le=1.0`` constraints.  Boxes are clamped rather than dropped:
        a partially off-screen control is still a real control.
        """
        self._settle_frames()
        vw, vh = self._viewport()
        nodes: list[Node] = []
        self._ref_index = {}
        counter = [0]

        cdp = self._cdp()
        if cdp is not None:
            # Populate the DOM agent's node map so `backendNodeId` lookups
            # (box model, resolveNode) work for every frame, iframes included.
            self._safe(lambda: cdp.send("DOM.getDocument", {"depth": -1, "pierce": True}), None)
            tree = self._safe(lambda: cdp.send("Page.getFrameTree")["frameTree"], None)
            pw_frames = self._safe(lambda: list(self.page.frames), [])
            if tree:
                self._walk_frame_tree(cdp, tree, (), nodes, counter, vw, vh, pw_frames)

        obs = Observation(
            url=self._safe(lambda: self.page.url, ""),
            title=self._safe(lambda: self.page.title(), ""),
            nodes=tuple(nodes[: self.config.max_nodes]),
            text_digest=self._text_digest(),
        )
        self._last_obs = obs
        return obs

    # -- CDP session -------------------------------------------------------

    def _cdp(self) -> Any:
        """One CDP session per surface, created lazily and reused.

        Playwright 1.62 removed `page.accessibility` / `frame.accessibility`,
        so the accessibility tree now comes from the DevTools protocol
        directly.  Opening a session is not free, hence: exactly one, cached.
        """
        if self._cdp_session is not None:
            return self._cdp_session
        ctx = self.context if self.context is not None else getattr(self.page, "context", None)
        if ctx is None:
            return None
        sess = self._safe(lambda: ctx.new_cdp_session(self.page), None)
        if sess is None:
            return None
        for domain in ("Accessibility.enable", "Page.enable", "DOM.enable"):
            self._safe(lambda d=domain: sess.send(d), None)
        self._cdp_session = sess
        return sess

    # -- frame traversal ---------------------------------------------------

    def _segment(self, name: str | None, url: str | None, index: int, taken: set[str]) -> str:
        seg_name = _norm(name or "")
        if not seg_name:
            tail = (url or "").rstrip("/").rsplit("/", 1)[-1].split("?")[0]
            seg_name = tail or f"f{index}"
        seg = seg_name
        n = 1
        while seg in taken:
            n += 1
            seg = f"{seg_name}#{n}"
        taken.add(seg)
        return seg

    def _frame_segment(self, frame: Any, index: int, taken: set[str]) -> str:
        """Legacy naming scheme, kept verbatim: frame `name`, else URL
        basename, else positional `f{i}`, with `#2` on sibling collisions."""
        return self._segment(
            getattr(frame, "name", "") or "", self._safe(lambda: frame.url, "") or "", index, taken
        )

    def _match_pw_frame(self, cdp_frame: dict, pw_frames: list) -> Any:
        """Best-effort CDP frame id -> Playwright `Frame`.

        Matched on (name, url), then url, then name.  A frame we cannot match
        simply yields no live handles; its nodes are still observed.
        """
        url = cdp_frame.get("url") or ""
        name = _norm(cdp_frame.get("name") or "")
        both = [
            f for f in pw_frames
            if self._safe(lambda f=f: f.url, "") == url
            and _norm(getattr(f, "name", "") or "") == name
        ]
        if len(both) == 1:
            return both[0]
        by_url = [f for f in pw_frames if url and self._safe(lambda f=f: f.url, "") == url]
        if len(by_url) == 1:
            return by_url[0]
        by_name = [f for f in pw_frames if name and _norm(getattr(f, "name", "") or "") == name]
        if len(by_name) == 1:
            return by_name[0]
        return both[0] if both else None

    def _walk_frame_tree(
        self,
        cdp: Any,
        tree: dict,
        frame_path: tuple[str, ...],
        out: list[Node],
        counter: list[int],
        vw: float,
        vh: float,
        pw_frames: list,
    ) -> None:
        frame = tree.get("frame") or {}
        frame_id = frame.get("id")
        pw_frame = self._match_pw_frame(frame, pw_frames)
        if frame_id:
            self._emit_frame_nodes(cdp, frame_id, pw_frame, frame_path, out, counter, vw, vh)

        taken: set[str] = set()
        for i, child in enumerate(tree.get("childFrames") or []):
            cf = child.get("frame") or {}
            seg = self._segment(cf.get("name"), cf.get("url"), i, taken)
            self._walk_frame_tree(
                cdp, child, frame_path + (seg,), out, counter, vw, vh, pw_frames
            )

    # -- accessibility tree ------------------------------------------------

    @staticmethod
    def _ax_order(nodes: list[dict]) -> list[dict]:
        """Return the flat AX node list in stable pre-order.

        Tree order is not cosmetic: it is the fallback the anchor-relative and
        scope tiers use when a node has no geometry.
        """
        by_id = {n.get("nodeId"): n for n in nodes if n.get("nodeId") is not None}
        children = {c for n in nodes for c in (n.get("childIds") or [])}
        roots = [n for n in nodes if n.get("nodeId") not in children]
        ordered: list[dict] = []
        seen: set[str] = set()

        def dfs(n: dict) -> None:
            nid = n.get("nodeId")
            if nid in seen:
                return
            seen.add(nid)
            ordered.append(n)
            for cid in n.get("childIds") or []:
                child = by_id.get(cid)
                if child is not None:
                    dfs(child)

        for r in roots or nodes[:1]:
            dfs(r)
        for n in nodes:  # anything unreachable via childIds keeps list order
            if n.get("nodeId") not in seen:
                ordered.append(n)
                seen.add(n.get("nodeId"))
        return ordered

    @staticmethod
    def _ax_props(ax: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in ax.get("properties") or []:
            name = p.get("name")
            if name:
                out[name] = (p.get("value") or {}).get("value")
        return out

    def _emit_frame_nodes(
        self,
        cdp: Any,
        frame_id: str,
        pw_frame: Any,
        frame_path: tuple[str, ...],
        out: list[Node],
        counter: list[int],
        vw: float,
        vh: float,
    ) -> None:
        """Pull the AX tree for ONE frame.

        `Accessibility.getFullAXTree` without a `frameId` returns main-frame
        nodes only -- iframe content is silently missed, which is fatal on a
        frameset app.  We therefore ask per frame id.
        """
        raw = self._safe(
            lambda: cdp.send("Accessibility.getFullAXTree", {"frameId": frame_id})["nodes"],
            None,
        )
        if not raw:
            return

        for ax in self._ax_order(raw):
            if len(out) >= self.config.max_nodes:
                return
            if ax.get("ignored"):
                continue
            role = ((ax.get("role") or {}).get("value")) or ""
            if role in _NOISE_ROLES:
                continue
            name = _norm((ax.get("name") or {}).get("value"))
            if not (name or role in _VALUE_ROLES or role == "button"):
                continue

            props = self._ax_props(ax)
            backend_id = ax.get("backendDOMNodeId")
            info = (
                self._element_info(cdp, backend_id)
                if backend_id is not None and role not in _TEXTUAL_ROLES
                else {}
            )

            value = props.get("value")
            if value in (None, ""):
                value = (ax.get("value") or {}).get("value")
            if value in (None, "") and role in _VALUE_ROLES:
                value = info.get("value")
            value = None if value in (None, "") else str(value)

            counter[0] += 1
            ref = f"n{counter[0]}"
            node = Node(
                ref=ref,
                role=role,
                name=name,
                value=value,
                frame_path=frame_path,
                bbox=self._bbox(cdp, backend_id, vw, vh),
                dom_path=info.get("dom_path") or None,
                text=_norm(info.get("near")) or name or None,
                enabled=not bool(props.get("disabled")),
            )
            out.append(node)
            self._ref_index[ref] = (
                node,
                _HandleSpec(ref=ref, frame=pw_frame, role=role, name=name, backend_id=backend_id),
            )

    # -- per-node enrichment ----------------------------------------------

    def _bbox(
        self, cdp: Any, backend_id: int | None, vw: float, vh: float
    ) -> tuple[float, float, float, float] | None:
        """Normalized, clamped box from `DOM.getBoxModel`.

        CDP reports the box model in **root-frame viewport coordinates**, so a
        control inside an iframe already carries the iframe's offset and needs
        no arithmetic of ours.  A node we cannot box keeps ``bbox=None``: the
        locator engine has a tree-order fallback, so degrading beats crashing.
        """
        if cdp is None or backend_id is None or vw <= 0 or vh <= 0:
            return None
        quad = self._safe(
            lambda: cdp.send("DOM.getBoxModel", {"backendNodeId": backend_id})["model"]["border"],
            None,
        )
        if not quad or len(quad) < 8:
            return None
        xs, ys = quad[0::2], quad[1::2]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y

        def clamp(v: float) -> float:
            return max(0.0, min(1.0, v))

        return (clamp(x / vw), clamp(y / vh), clamp(w / vw), clamp(h / vh))

    def _element_info(self, cdp: Any, backend_id: int | None) -> dict:
        """One round trip for `dom_path`, nearby label text and live value.

        Only asked for nodes that can actually be acted upon -- resolving every
        StaticText would triple the cost of `observe()` for nothing.
        """
        if cdp is None or backend_id is None:
            return {}
        obj = self._safe(
            lambda: cdp.send("DOM.resolveNode", {"backendNodeId": backend_id})["object"], None
        )
        object_id = (obj or {}).get("objectId")
        if not object_id:
            return {}
        res = self._safe(
            lambda: cdp.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": object_id,
                    "functionDeclaration": _ELEMENT_INFO_JS,
                    "returnByValue": True,
                },
            ),
            None,
        )
        self._safe(lambda: cdp.send("Runtime.releaseObject", {"objectId": object_id}), None)
        val = ((res or {}).get("result") or {}).get("value")
        return val if isinstance(val, dict) else {}

    # -- live handles ------------------------------------------------------

    def _live_handle(self, spec: "_HandleSpec | None") -> Any:
        """A UNIQUE live handle for a node, or ``None``.

        Tried in order: the ARIA role+name query (unchanged from before), then
        a `backendDOMNodeId`-anchored marker attribute.  Both must resolve to
        exactly one element; anything else returns ``None`` so `act()` refuses
        rather than guessing.  Resolution is lazy -- `observe()` never pays for
        handles nobody asks for.
        """
        if spec is None or spec.frame is None:
            return None
        frame = spec.frame
        if spec.role:
            loc = self._safe(
                lambda: (
                    frame.get_by_role(spec.role, name=spec.name, exact=True)
                    if spec.name
                    else frame.get_by_role(spec.role)
                ),
                None,
            )
            if loc is not None and self._safe(lambda: loc.count(), -1) == 1:
                return loc.first

        if spec.backend_id is None:
            return None
        cdp = self._cdp()
        if cdp is None:
            return None
        obj = self._safe(
            lambda: cdp.send("DOM.resolveNode", {"backendNodeId": spec.backend_id})["object"],
            None,
        )
        object_id = (obj or {}).get("objectId")
        if not object_id:
            return None
        marker = f"cua-{spec.ref}"
        ok = self._safe(
            lambda: cdp.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": object_id,
                    "functionDeclaration": _MARK_JS,
                    "arguments": [{"value": marker}],
                    "returnByValue": True,
                },
            ),
            None,
        )
        self._safe(lambda: cdp.send("Runtime.releaseObject", {"objectId": object_id}), None)
        if not (((ok or {}).get("result") or {}).get("value")):
            return None
        loc = self._safe(lambda: frame.locator(f'[{_MARK_ATTR}="{marker}"]'), None)
        if loc is not None and self._safe(lambda: loc.count(), -1) == 1:
            return loc.first
        return None

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
            node, spec = entry
            return node, self._live_handle(spec), ActResult(ok=True, resolved_tier=None, candidates=1)

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

        spec = self._ref_index.get(node.ref, (None, None))[1]
        return node, self._live_handle(spec), ActResult(
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
        # A UTC timestamp rather than epoch milliseconds: a 13-digit run looks
        # like a card number to the redaction filter, which then scrubs the
        # filename and breaks the evidence reference it was meant to protect.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = Path(self.config.evidence_dir) / f"{stamp}-{safe}.png"
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
