"""The universal remote: six methods, no Playwright.

Neither the discovery agent nor the replay engine may import a concrete
surface. That single rule is what makes the desktop story architectural fact
rather than aspiration -- a new surface is six methods, and nothing upstream
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cua.artifact import ActionType, LocatorBundle


@dataclass(frozen=True)
class Node:
    """One control, as perceived. Deliberately mirrors the accessibility tree,
    because that representation exists on legacy web AND native desktop."""

    ref: str
    role: str
    name: str
    value: str | None = None
    frame_path: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    dom_path: str | None = None
    text: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class Observation:
    """What the system can see right now. The only view of the world either
    engine gets."""

    url: str
    title: str
    nodes: tuple[Node, ...]
    screenshot_path: str | None = None
    text_digest: str = ""

    def render(self, limit: int = 120) -> str:
        """Compact text encoding handed to the LLM during discovery."""
        lines = [f"URL: {self.url}", f"TITLE: {self.title}", "CONTROLS:"]
        for n in self.nodes[:limit]:
            scope = "/".join(n.frame_path)
            val = f' value="{n.value}"' if n.value else ""
            state = "" if n.enabled else " [disabled]"
            lines.append(
                f'  [{n.ref}] {n.role} "{n.name}"{val}{state}'
                + (f"  (in {scope})" if scope else "")
            )
        if len(self.nodes) > limit:
            lines.append(f"  ... {len(self.nodes) - limit} more controls omitted")
        return "\n".join(lines)


@dataclass(frozen=True)
class Action:
    type: ActionType
    target: LocatorBundle | None = None
    ref: str | None = None
    value: str | None = None
    url: str | None = None


@dataclass
class ActResult:
    ok: bool
    detail: str = ""
    resolved_tier: int | None = None
    candidates: int = 0
    read_value: str | None = None


@dataclass
class SurfaceInfo:
    kind: str
    app_id: str
    tenant_id: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)


@runtime_checkable
class Surface(Protocol):
    """The six buttons."""

    def observe(self) -> Observation: ...

    def act(self, action: Action) -> ActResult: ...

    def read(self, target: LocatorBundle) -> str | None: ...

    def wait_for(self, checkpoint, timeout_ms: int) -> bool: ...

    def snapshot(self, label: str) -> str | None:
        """Capture richer evidence (screenshot/trace) and return its path."""
        ...

    def describe(self) -> SurfaceInfo: ...
