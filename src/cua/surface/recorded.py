"""`RecordedSurface` -- a deterministic, hermetic implementation of `Surface`.

Why this is a first-class deliverable and not a mock
----------------------------------------------------
Two things fall out of having a second real implementation:

1.  The whole test suite -- discovery planning, replay, outcome classification,
    evidence -- runs with **no browser, no network, and no API key**.  CI stays
    fast and green.
2.  It is the *proof* that the `Surface` seam is real.  Upstream code cannot
    tell a `RecordedSurface` from a `WebSurface`; if it could, the seam would be
    a lie and the "a new surface is six methods" claim would be aspiration.

A real discovery run against a live browser can be captured with
`dump_observations()` and replayed forever with `RecordedSurface.from_dir()`.

The playback model (deliberately simple, and documented because tests depend on
it)
-----------------------------------------------------------------------------
* The surface holds an ordered list of `Observation`s and a cursor `index`,
  starting at 0.  `observe()` returns `frames[index]` and is side-effect free --
  you may call it as often as you like.
* Every *successful* mutating `act()` (CLICK / TYPE / SELECT / PRESS) advances
  the cursor by one, clamped at the last frame.  This is the default
  "each action moves the world forward one step" model.
* `transitions` overrides that: it maps a step key to the absolute index the
  cursor should jump to.  A key may be
    - an int: the source index (any action at that index jumps there), or
    - a `(source_index, ActionType)` tuple, or
    - a `(source_index, ActionType, ref_or_description)` tuple  -- most specific
      wins.
* `NAVIGATE` does not advance by one; it jumps to the first frame whose `url`
  matches (exact, then substring).  If nothing matches it fails -- a recording
  that does not contain the destination is a recording bug, not a runtime one.
* `READ` and `WAIT` never advance.
* `failures` injects errors: `{step_index: "detail"}` makes any `act()`
  attempted while the cursor is at that index return `ok=False` with that
  detail, and the cursor does not move.  That is how tests exercise error
  paths without a flaky browser.
* `wait_for()` evaluates the checkpoint against the *current* frame only, then
  against each subsequent frame if `advance_on_wait` is set; it never sleeps.
  `timeout_ms` is honoured semantically (it is the number of frames we are
  willing to look ahead when advancing), never as wall-clock delay.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cua.artifact import ActionType, Checkpoint, LocatorBundle
from cua.surface.base import Action, ActResult, Node, Observation, SurfaceInfo

_MUTATING = frozenset(
    {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT, ActionType.PRESS}
)


def _load_resolver():
    from cua.locator.resolve import resolve  # noqa: PLC0415

    return resolve


# --------------------------------------------------------------------------
# Serialization: capture a real run, replay it in CI.
# --------------------------------------------------------------------------


def node_to_dict(n: Node) -> dict[str, Any]:
    return {
        "ref": n.ref,
        "role": n.role,
        "name": n.name,
        "value": n.value,
        "frame_path": list(n.frame_path),
        "bbox": list(n.bbox) if n.bbox else None,
        "dom_path": n.dom_path,
        "text": n.text,
        "enabled": n.enabled,
    }


def node_from_dict(d: Mapping[str, Any]) -> Node:
    bbox = d.get("bbox")
    return Node(
        ref=d["ref"],
        role=d.get("role", ""),
        name=d.get("name", ""),
        value=d.get("value"),
        frame_path=tuple(d.get("frame_path") or ()),
        bbox=tuple(bbox) if bbox else None,  # type: ignore[arg-type]
        dom_path=d.get("dom_path"),
        text=d.get("text"),
        enabled=bool(d.get("enabled", True)),
    )


def observation_to_dict(o: Observation) -> dict[str, Any]:
    return {
        "url": o.url,
        "title": o.title,
        "screenshot_path": o.screenshot_path,
        "text_digest": o.text_digest,
        "nodes": [node_to_dict(n) for n in o.nodes],
    }


def observation_from_dict(d: Mapping[str, Any]) -> Observation:
    return Observation(
        url=d.get("url", ""),
        title=d.get("title", ""),
        nodes=tuple(node_from_dict(n) for n in d.get("nodes") or ()),
        screenshot_path=d.get("screenshot_path"),
        text_digest=d.get("text_digest", ""),
    )


def dump_observations(frames: Sequence[Observation], directory: str | Path) -> list[str]:
    """Write one `NNNN.json` per observation.  Zero-padded so lexical order is
    playback order."""
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    written = []
    for i, obs in enumerate(frames):
        f = p / f"{i:04d}.json"
        f.write_text(json.dumps(observation_to_dict(obs), indent=2), encoding="utf-8")
        written.append(str(f))
    return written


def load_observations(directory: str | Path) -> list[Observation]:
    p = Path(directory)
    files = sorted(p.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no observation JSON files in {p}")
    return [observation_from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]


# --------------------------------------------------------------------------


@dataclass
class RecordedSurface:
    """Plays back a recorded sequence of `Observation`s.  See module docstring
    for the playback model."""

    frames: list[Observation]
    app_id: str = "recorded"
    tenant_id: str | None = None
    transitions: dict[Any, int] = field(default_factory=dict)
    failures: dict[int, str] = field(default_factory=dict)
    advance_on_wait: bool = False
    index: int = 0

    #: Every act()/read()/wait_for() call, in order -- assertions live here.
    log: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.frames = list(self.frames)
        if not self.frames:
            raise ValueError("RecordedSurface requires at least one Observation")

    # ------------------------------------------------------------ builders

    @classmethod
    def from_dir(cls, directory: str | Path, **kw: Any) -> "RecordedSurface":
        return cls(frames=load_observations(directory), **kw)

    def save(self, directory: str | Path) -> list[str]:
        return dump_observations(self.frames, directory)

    def inject_failure(self, at_index: int, detail: str) -> "RecordedSurface":
        """Make any act() attempted from `at_index` fail with `detail`."""
        self.failures[at_index] = detail
        return self

    def reset(self) -> None:
        self.index = 0
        self.log.clear()

    # ------------------------------------------------------------- observe

    def observe(self) -> Observation:
        return self.frames[self.index]

    # ----------------------------------------------------------------- act

    def act(self, action: Action) -> ActResult:
        self.log.append(f"{action.type.value}@{self.index}")

        if self.index in self.failures:
            return ActResult(
                ok=False,
                detail=f"injected failure at step {self.index}: {self.failures[self.index]}",
            )

        if action.type is ActionType.NAVIGATE:
            url = action.url or action.value or ""
            target = self._frame_for_url(url)
            if target is None:
                return ActResult(
                    ok=False,
                    detail=(
                        f"NAVIGATE to {url!r} has no matching recorded frame; "
                        f"known urls: {[f.url for f in self.frames]}"
                    ),
                )
            self.index = target
            return ActResult(ok=True, detail=f"navigated to recorded frame {target}")

        if action.type is ActionType.WAIT:
            return ActResult(ok=True, detail="wait is a no-op on a recorded surface")

        node, res = self._locate(action)
        if node is None:
            return res

        if action.type is ActionType.READ:
            value = node.value if node.value is not None else (node.text or node.name)
            return ActResult(
                ok=True,
                detail=f"read {node.ref}",
                resolved_tier=res.resolved_tier,
                candidates=res.candidates,
                read_value=value,
            )

        if action.type not in _MUTATING:
            return ActResult(ok=False, detail=f"unsupported action type {action.type}")

        if not node.enabled:
            return ActResult(
                ok=False,
                detail=f"node {node.ref} ({node.role} '{node.name}') is disabled",
                resolved_tier=res.resolved_tier,
                candidates=res.candidates,
            )

        self.index = self._next_index(action, node)
        return ActResult(
            ok=True,
            detail=f"{action.type.value} on {node.ref} -> frame {self.index}",
            resolved_tier=res.resolved_tier,
            candidates=res.candidates,
        )

    def _next_index(self, action: Action, node: Node) -> int:
        key_ref = action.ref or (action.target.description if action.target else None)
        for key in (
            (self.index, action.type, key_ref),
            (self.index, action.type),
            self.index,
        ):
            if key in self.transitions:
                return self.transitions[key]
        return min(self.index + 1, len(self.frames) - 1)

    def _frame_for_url(self, url: str) -> int | None:
        for i, f in enumerate(self.frames):
            if f.url == url:
                return i
        for i, f in enumerate(self.frames):
            if url and (url in f.url or f.url in url):
                return i
        return None

    def _locate(self, action: Action) -> tuple[Node | None, ActResult]:
        obs = self.observe()
        if action.ref:
            for n in obs.nodes:
                if n.ref == action.ref:
                    return n, ActResult(ok=True, candidates=1)
            return None, ActResult(
                ok=False,
                detail=(
                    f"ref {action.ref!r} not present in recorded frame {self.index} "
                    f"({[n.ref for n in obs.nodes]})"
                ),
            )

        if action.target is None:
            return None, ActResult(
                ok=False, detail="action has neither ref nor target locator bundle"
            )

        try:
            resolve = _load_resolver()
        except Exception as e:
            return None, ActResult(ok=False, detail=f"locator resolver unavailable: {e!r}")

        try:
            r = resolve(action.target, obs)
        except Exception as e:
            return None, ActResult(
                ok=False, detail=f"resolve({action.target.description!r}) raised: {e!r}"
            )

        node = getattr(r, "node", None)
        tier = getattr(r, "tier", None)
        cands = getattr(r, "candidates", 0) or 0
        if node is None:
            return None, ActResult(
                ok=False,
                detail=(
                    f"could not uniquely resolve {action.target.description!r} in frame "
                    f"{self.index}: {getattr(r, 'reason', '')} (candidates={cands})"
                ),
                resolved_tier=int(tier) if tier is not None else None,
                candidates=cands,
            )
        return node, ActResult(
            ok=True,
            resolved_tier=int(tier) if tier is not None else None,
            candidates=cands,
        )

    # ---------------------------------------------------------------- read

    def read(self, target: LocatorBundle) -> str | None:
        res = self.act(Action(type=ActionType.READ, target=target))
        return res.read_value if res.ok else None

    def read_ref(self, ref: str) -> str | None:
        """Convenience for tests/fixtures that address by observation ref."""
        res = self.act(Action(type=ActionType.READ, ref=ref))
        return res.read_value if res.ok else None

    # ------------------------------------------------------------ wait_for

    def wait_for(self, checkpoint: Checkpoint, timeout_ms: int) -> bool:
        """Evaluate the checkpoint deterministically -- no sleeping, ever."""
        self.log.append(f"wait_for@{self.index}")
        if self._satisfied(checkpoint, self.frames[self.index]):
            return True
        if not self.advance_on_wait or timeout_ms <= 0:
            return False
        # Look ahead: one recorded frame per 1000ms of declared patience.
        lookahead = max(1, timeout_ms // 1000)
        for i in range(self.index + 1, min(self.index + 1 + lookahead, len(self.frames))):
            if self._satisfied(checkpoint, self.frames[i]):
                self.index = i
                return True
        return False

    def _satisfied(self, cp: Checkpoint, obs: Observation) -> bool:
        conditions = 0
        hay = (obs.text_digest or obs.render()).lower()

        if cp.url_matches is not None:
            conditions += 1
            if not re.search(cp.url_matches, obs.url):
                return False
        if cp.text_present is not None:
            conditions += 1
            if cp.text_present.lower() not in hay:
                return False
        if cp.text_absent is not None:
            conditions += 1
            if cp.text_absent.lower() in hay:
                return False
        if cp.locator is not None:
            conditions += 1
            try:
                r = _load_resolver()(cp.locator, obs)
            except Exception:
                return False
            if getattr(r, "node", None) is None:
                return False
        return conditions > 0

    # ------------------------------------------------------------ snapshot

    def snapshot(self, label: str) -> str | None:
        """Returns the recorded screenshot path, if the capture had one."""
        return self.frames[self.index].screenshot_path

    # ------------------------------------------------------------ describe

    def describe(self) -> SurfaceInfo:
        return SurfaceInfo(
            kind="recorded",
            app_id=self.app_id,
            tenant_id=self.tenant_id,
            capabilities=frozenset(
                {"navigate", "click", "type", "select", "press", "read", "wait", "replay"}
            ),
        )
