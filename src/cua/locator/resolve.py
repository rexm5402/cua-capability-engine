"""Locator resolution: turn a LocatorBundle into exactly one Node, or fail loudly.

Design rules, in order of importance:

1. Strategies are tried in ascending tier order (``bundle.by_tier()``).
2. A strategy succeeds only if it identifies EXACTLY ONE node. Two matches is
   ambiguity, and ambiguity is a failure of that tier -- never a silent
   ``first()``. We record the candidate count and fall through.
3. The tier that actually resolved is returned, so callers can count tier
   escalations as a drift early-warning signal.
4. Failure carries a reason that names every tier tried and what it saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cua.artifact import (
    AnchorRelativeLocator,
    GeometryLocator,
    LocatorBundle,
    Relation,
    SemanticLocator,
    StructuralLocator,
    TextLocator,
)
from cua.surface.base import Node, Observation

__all__ = ["Resolution", "resolve", "normalize"]


# --------------------------------------------------------------------------
# Geometry tolerances. All values are in surface pixels unless stated.
# --------------------------------------------------------------------------

#: Two nodes are "on the same row" when their vertical centres differ by no
#: more than half the taller box plus this slack. Half-height keeps the rule
#: scale-free (it survives a font-size or padding change), and the slack
#: absorbs baseline/border jitter between a label and its input.
ROW_SLACK_PX = 6.0

#: A target must start at least this far right of the anchor's right edge to
#: count as "to the right". Slightly negative so a control that visually abuts
#: (or overlaps by a hairline) its label still qualifies.
RIGHT_GAP_PX = -2.0

#: Same idea, vertically, for BELOW.
BELOW_GAP_PX = -2.0

#: SAME_CELL inflates the anchor box by this much before testing containment,
#: standing in for the (usually unexposed) table-cell element.
CELL_PAD_PX = 8.0

#: WITHIN allows this much bleed outside the anchor box ("closely bounded").
WITHIN_PAD_PX = 2.0

#: Nearest-wins is only trustworthy if there IS a nearest. Two candidates whose
#: distances differ by less than this are treated as a tie -> ambiguous.
DISTANCE_TIE_PX = 2.0

#: GEOMETRY tier: max normalized centre distance we will accept at all, and the
#: margin by which the winner must beat the runner-up.
GEOMETRY_MAX_DIST = 0.03
GEOMETRY_TIE_MARGIN = 0.01


@dataclass
class Resolution:
    """Outcome of resolving a bundle against one observation."""

    node: Node | None
    tier: int | None
    candidates: int
    reason: str

    @property
    def ok(self) -> bool:
        return self.node is not None


@dataclass
class _TierResult:
    """Internal per-strategy verdict."""

    node: Node | None
    candidates: int
    note: str
    detail: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")
#: Trailing label decoration: colon, asterisk (required marker), and friends.
_TRAILING_JUNK = " \t:*. -–—?"


def normalize(s: str | None) -> str:
    """casefold + collapse whitespace + strip trailing label punctuation.

    ``"Account Number:"``, ``"account   number *"`` and ``"ACCOUNT NUMBER"``
    all normalize to ``"account number"``. This is what lets a locator recorded
    at one tenant match a differently-styled label at another.
    """
    if not s:
        return ""
    out = _WS.sub(" ", s.replace(" ", " ")).strip()
    out = out.strip(_TRAILING_JUNK)
    return _WS.sub(" ", out).strip().casefold()


def _matches(value: str | None, wanted: str, mode: str) -> bool:
    if value is None:
        value = ""
    if mode == "exact":
        return value == wanted
    if mode == "contains":
        return normalize(wanted) in normalize(value)
    return normalize(value) == normalize(wanted)


def _node_texts(n: Node) -> tuple[str, ...]:
    return tuple(t for t in (n.name, n.text, n.value) if t)


def _role_eq(a: str, b: str) -> bool:
    return normalize(a).replace(" ", "") == normalize(b).replace(" ", "")


# --------------------------------------------------------------------------
# Scope handling
# --------------------------------------------------------------------------


def _scope_ok(node: Node, scope: list[str]) -> bool:
    """``scope`` matches when it is a suffix of, or a contiguous run inside,
    the node's frame path. Absent scope matches everything."""
    if not scope:
        return True
    want = [normalize(s) for s in scope]
    have = [normalize(s) for s in node.frame_path]
    n = len(want)
    if n > len(have):
        return False
    if have[-n:] == want:
        return True
    return any(have[i : i + n] == want for i in range(len(have) - n + 1))


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _cx(b: tuple[float, float, float, float]) -> float:
    return b[0] + b[2] / 2.0


def _cy(b: tuple[float, float, float, float]) -> float:
    return b[1] + b[3] / 2.0


def _same_row(a: tuple[float, float, float, float],
              t: tuple[float, float, float, float]) -> bool:
    tol = max(a[3], t[3]) / 2.0 + ROW_SLACK_PX
    if abs(_cy(a) - _cy(t)) > tol:
        return False
    # require genuine vertical overlap of the spans as well, so a tall anchor
    # cannot "capture" a control two rows down.
    return min(a[1] + a[3], t[1] + t[3]) - max(a[1], t[1]) > -ROW_SLACK_PX


def _v_overlap(a: tuple[float, float, float, float],
               t: tuple[float, float, float, float]) -> bool:
    return min(a[1] + a[3], t[1] + t[3]) - max(a[1], t[1]) > 0.0


def _h_overlap(a: tuple[float, float, float, float],
               t: tuple[float, float, float, float]) -> bool:
    return min(a[0] + a[2], t[0] + t[2]) - max(a[0], t[0]) > 0.0


def _contained(outer: tuple[float, float, float, float],
               inner: tuple[float, float, float, float], pad: float) -> bool:
    return (
        inner[0] >= outer[0] - pad
        and inner[1] >= outer[1] - pad
        and inner[0] + inner[2] <= outer[0] + outer[2] + pad
        and inner[1] + inner[3] <= outer[1] + outer[3] + pad
    )


def _viewport(obs: Observation) -> tuple[float, float]:
    w = max((n.bbox[0] + n.bbox[2] for n in obs.nodes if n.bbox), default=1.0)
    h = max((n.bbox[1] + n.bbox[3] for n in obs.nodes if n.bbox), default=1.0)
    return (max(w, 1.0), max(h, 1.0))


# --------------------------------------------------------------------------
# Scope-then-unique: narrow the pool to one container region, THEN require a
# unique match inside it.
#
# The region is derived exactly the way the anchor-relative tier reasons about
# rows: find the node(s) carrying ``scope_text``, then take the SAME_ROW band
# around each (same ``frame_path``, same-row per :func:`_same_row`, i.e. the
# same half-height + ROW_SLACK_PX tolerance the ANCHOR_RELATIVE tier uses).
# Nodes with no bbox cannot be banded geometrically, so they fall back to the
# tree-order run that follows the anchor -- the same dom-order fallback the
# anchor tier uses when geometry is unavailable.
# --------------------------------------------------------------------------


def _tree_band(anchor: Node, obs: Observation) -> list[Node]:
    """DOM-order banding: the contiguous run of same-frame nodes starting at
    the anchor and ending before the next node that plays the anchor's own
    role (i.e. the next row's identifying cell)."""
    nodes = list(obs.nodes)
    ai = next((i for i, n in enumerate(nodes) if n.ref == anchor.ref), None)
    if ai is None:
        return []
    band = [nodes[ai]]
    for n in nodes[ai + 1 :]:
        if n.frame_path != anchor.frame_path:
            break
        if _role_eq(n.role, anchor.role):
            break
        band.append(n)
    return band


def _band_of(anchor: Node, obs: Observation) -> list[Node]:
    """All nodes belonging to the anchor's region."""
    same_frame = [n for n in obs.nodes if n.frame_path == anchor.frame_path]
    if anchor.bbox is None:
        return _tree_band(anchor, obs)
    band = {
        n.ref: n
        for n in same_frame
        if n.bbox is not None and (n.ref == anchor.ref or _same_row(anchor.bbox, n.bbox))
    }
    # Nodes with no geometry cannot be banded by row; use the dom-order run.
    if any(n.bbox is None for n in same_frame):
        for n in _tree_band(anchor, obs):
            if n.bbox is None:
                band.setdefault(n.ref, n)
    return list(band.values())


def _scope_region(scope_text: str, obs: Observation) -> tuple[list[Node] | None, str]:
    """Return (region nodes, note). ``None`` means the scope did not yield
    exactly one region; the note says why. Never falls back to the global pool.
    """
    anchors = _find_anchors(scope_text, obs)
    if not anchors:
        return None, f"scope_text {scope_text!r} matched no node in this observation"

    groups: list[tuple[set[str], dict[str, Node]]] = []
    for anchor in anchors:
        band = _band_of(anchor, obs)
        refs = {n.ref for n in band} | {anchor.ref}
        merged: tuple[set[str], dict[str, Node]] | None = None
        for g in list(groups):
            if g[0] & refs:
                if merged is None:
                    g[0].update(refs)
                    g[1].update({n.ref: n for n in band})
                    merged = g
                else:
                    merged[0].update(g[0])
                    merged[1].update(g[1])
                    groups.remove(g)
        if merged is None:
            groups.append((set(refs), {n.ref: n for n in band}))

    if len(groups) > 1:
        return None, (
            f"scope_text {scope_text!r} is ambiguous: it matched "
            f"{len(anchors)} node(s) spanning {len(groups)} disjoint regions"
        )
    region = list(groups[0][1].values())
    order = {n.ref: i for i, n in enumerate(obs.nodes)}
    region.sort(key=lambda n: order.get(n.ref, 0))
    return region, f"scoped to region of {scope_text!r} ({len(region)} node(s))"


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def resolve(bundle: LocatorBundle, obs: Observation) -> Resolution:
    """Resolve ``bundle`` against ``obs``, tier by tier, unique-or-nothing.

    When ``bundle.scope_text`` is set the candidate pool is first narrowed to
    the single container region holding that text, and the unique-match rule is
    then applied WITHIN that region. If the scope text matches nothing, or
    matches several disjoint regions, resolution fails outright -- falling back
    to the global pool would quietly reintroduce guessing.
    """
    notes: list[str] = []
    last_candidates = 0
    viewport = _viewport(obs)
    scope_note = "unscoped"
    pool = obs
    if bundle.scope_text is not None:
        region, note = _scope_region(bundle.scope_text, obs)
        scope_note = note
        if region is None:
            return Resolution(
                node=None,
                tier=None,
                candidates=0,
                reason=(
                    f"unresolved for {bundle.description!r} on {obs.url!r}: {note}"
                ),
            )
        pool = Observation(
            url=obs.url,
            title=obs.title,
            nodes=tuple(region),
            screenshot_path=obs.screenshot_path,
            text_digest=obs.text_digest,
        )
    for strategy in bundle.by_tier():
        tier = int(strategy.tier)
        result = _apply(strategy, pool, viewport)
        last_candidates = result.candidates
        if result.node is not None:
            reason = (
                f"resolved at tier {tier} ({strategy.tier.name}) "
                f"[{scope_note}]: {result.note}"
            )
            if notes:
                reason = "; ".join(notes) + " -> " + reason
            return Resolution(
                node=result.node,
                tier=tier,
                candidates=result.candidates,
                reason=reason,
            )
        notes.append(
            f"tier {tier} ({strategy.tier.name}) found {result.candidates} "
            f"candidate(s): {result.note}"
        )
    return Resolution(
        node=None,
        tier=None,
        candidates=last_candidates,
        reason=(
            f"unresolved for {bundle.description!r} across "
            f"{len(bundle.strategies)} strategie(s) on {obs.url!r} "
            f"[{scope_note}]: " + " | ".join(notes)
            if notes
            else f"unresolved for {bundle.description!r} [{scope_note}]: "
            "bundle had no strategies"
        ),
    )


def _apply(
    strategy, obs: Observation, viewport: tuple[float, float] | None = None
) -> _TierResult:
    if isinstance(strategy, SemanticLocator):
        return _semantic(strategy, obs)
    if isinstance(strategy, AnchorRelativeLocator):
        return _anchor(strategy, obs)
    if isinstance(strategy, TextLocator):
        return _text(strategy, obs)
    if isinstance(strategy, StructuralLocator):
        return _structural(strategy, obs)
    if isinstance(strategy, GeometryLocator):
        return _geometry(strategy, obs, viewport)
    return _TierResult(None, 0, f"unknown strategy type {type(strategy).__name__}")


def _unique(matches: list[Node], what: str, nth: int = 0) -> _TierResult:
    """Shared unique-or-ambiguous verdict. ``nth`` > 0 is an explicit,
    recorded ordinal selection and is therefore allowed to index."""
    if not matches:
        return _TierResult(None, 0, f"no node matched {what}")
    if nth:
        if nth < len(matches):
            return _TierResult(matches[nth], len(matches), f"{what} [nth={nth}]")
        return _TierResult(
            None, len(matches), f"{what}: nth={nth} out of range ({len(matches)})"
        )
    if len(matches) > 1:
        names = ", ".join(repr(m.ref) for m in matches[:5])
        return _TierResult(
            None, len(matches), f"ambiguous {what}: {len(matches)} matches ({names})"
        )
    return _TierResult(matches[0], 1, what)


# --------------------------------------------------------------------------
# Tier 1: SEMANTIC -- role + accessible name
# --------------------------------------------------------------------------


def _semantic(s: SemanticLocator, obs: Observation) -> _TierResult:
    what = f"role={s.role!r} name={s.name!r} match={s.name_match}"
    if s.scope:
        what += f" scope={list(s.scope)!r}"
    matches = [
        n
        for n in obs.nodes
        if _role_eq(n.role, s.role)
        and _scope_ok(n, s.scope)
        and (s.name is None or _matches(n.name, s.name, s.name_match))
    ]
    return _unique(matches, what)


# --------------------------------------------------------------------------
# Tier 2: ANCHOR_RELATIVE -- the one that survives a reskin
# --------------------------------------------------------------------------


def _find_anchors(anchor_text: str, obs: Observation) -> list[Node]:
    want = normalize(anchor_text)
    if not want:
        return []
    exact = [n for n in obs.nodes if any(normalize(t) == want for t in _node_texts(n))]
    if exact:
        return exact
    return [n for n in obs.nodes if any(want in normalize(t) for t in _node_texts(n))]


def _relation_candidates(
    anchor: Node, s: AnchorRelativeLocator, obs: Observation
) -> tuple[list[tuple[float, Node]], str]:
    """Return (distance, node) pairs satisfying the relation, nearest first,
    plus a note describing which mode (geometric / dom-order) was used."""
    pool = [
        n
        for n in obs.nodes
        if n.ref != anchor.ref
        and _role_eq(n.role, s.target_role)
        and n.frame_path == anchor.frame_path
    ]

    if anchor.bbox is not None:
        ab = anchor.bbox
        scored: list[tuple[float, Node]] = []
        for n in pool:
            if n.bbox is None:  # geometry unavailable for this node: skip it
                continue
            tb = n.bbox
            if s.relation is Relation.SAME_ROW:
                if _same_row(ab, tb) and tb[0] >= ab[0] + ab[2] + RIGHT_GAP_PX:
                    scored.append((tb[0] - (ab[0] + ab[2]), n))
            elif s.relation is Relation.RIGHT_OF:
                if _v_overlap(ab, tb) and tb[0] >= ab[0] + ab[2] + RIGHT_GAP_PX:
                    scored.append((tb[0] - (ab[0] + ab[2]), n))
            elif s.relation is Relation.BELOW:
                if _h_overlap(ab, tb) and tb[1] >= ab[1] + ab[3] + BELOW_GAP_PX:
                    scored.append((tb[1] - (ab[1] + ab[3]), n))
            elif s.relation is Relation.WITHIN:
                if _contained(ab, tb, WITHIN_PAD_PX):
                    scored.append((abs(_cx(ab) - _cx(tb)) + abs(_cy(ab) - _cy(tb)), n))
            elif s.relation is Relation.SAME_CELL:
                if _contained(ab, tb, CELL_PAD_PX):
                    scored.append((abs(_cx(ab) - _cx(tb)) + abs(_cy(ab) - _cy(tb)), n))
        if scored:
            scored.sort(key=lambda p: (p[0], p[1].ref))
            return scored, "geometric"
        if any(n.bbox is not None for n in pool):
            # geometry was available and simply did not satisfy the relation
            return [], "geometric"

    # Fallback: no usable geometry -> DOM-order adjacency after the anchor.
    order = {n.ref: i for i, n in enumerate(obs.nodes)}
    ai = order.get(anchor.ref, -1)
    after = [(order[n.ref] - ai, n) for n in pool if order[n.ref] > ai]
    after.sort(key=lambda p: p[0])
    return [(float(d), n) for d, n in after], "dom-order fallback"


def _anchor(s: AnchorRelativeLocator, obs: Observation) -> _TierResult:
    what = (
        f"anchor={s.anchor_text!r} relation={s.relation.value} "
        f"target_role={s.target_role!r} nth={s.nth}"
    )
    anchors = _find_anchors(s.anchor_text, obs)
    if not anchors:
        return _TierResult(None, 0, f"{what}: no node carries the anchor text")

    picks: list[Node] = []
    total = 0
    mode = "geometric"
    for anchor in anchors:
        scored, mode = _relation_candidates(anchor, s, obs)
        total += len(scored)
        if len(scored) <= s.nth:
            continue
        # nearest-wins, but a genuine tie is ambiguity, not a coin flip.
        if s.nth == 0 and len(scored) > 1:
            if abs(scored[1][0] - scored[0][0]) < DISTANCE_TIE_PX:
                return _TierResult(
                    None,
                    len(scored),
                    f"{what}: tie between {scored[0][1].ref!r} and "
                    f"{scored[1][1].ref!r} at equal distance ({mode})",
                )
        picks.append(scored[s.nth][1])

    distinct = {n.ref: n for n in picks}
    if not distinct:
        return _TierResult(
            None,
            total,
            f"{what}: {len(anchors)} anchor(s) but nothing stands in that "
            f"relation ({mode})",
        )
    if len(distinct) > 1:
        return _TierResult(
            None,
            len(distinct),
            f"{what}: ambiguous, {len(anchors)} anchors selected "
            f"{len(distinct)} different targets",
        )
    return _TierResult(picks[0], 1, f"{what} via {mode}")


# --------------------------------------------------------------------------
# Tier 3: TEXT
# --------------------------------------------------------------------------


def _text(s: TextLocator, obs: Observation) -> _TierResult:
    what = f"text={s.text!r} match={s.match} role={s.role!r} nth={s.nth}"
    matches = [
        n
        for n in obs.nodes
        if (s.role is None or _role_eq(n.role, s.role))
        and any(_matches(t, s.text, s.match) for t in _node_texts(n))
    ]
    return _unique(matches, what, nth=s.nth)


# --------------------------------------------------------------------------
# Tier 4: STRUCTURAL -- recorded, distrusted
# --------------------------------------------------------------------------


def _structural(s: StructuralLocator, obs: Observation) -> _TierResult:
    what = f"frame_path={list(s.frame_path)!r} dom_path={s.dom_path!r}"
    want_frame = tuple(s.frame_path)
    matches = [
        n
        for n in obs.nodes
        if n.dom_path == s.dom_path and tuple(n.frame_path) == want_frame
    ]
    return _unique(matches, what)


# --------------------------------------------------------------------------
# Tier 5: GEOMETRY -- last resort, tightly bounded
# --------------------------------------------------------------------------


def _geometry(
    s: GeometryLocator, obs: Observation, viewport: tuple[float, float] | None = None
) -> _TierResult:
    # The viewport is always the FULL observation's extent, even when the pool
    # has been scoped -- otherwise scoping would silently rescale the recorded
    # normalized coordinates.
    vw, vh = viewport if viewport is not None else _viewport(obs)
    tx, ty = s.x + s.w / 2.0, s.y + s.h / 2.0
    what = f"normalized centre=({tx:.3f},{ty:.3f}) threshold={GEOMETRY_MAX_DIST}"
    scored = []
    for n in obs.nodes:
        if n.bbox is None:
            continue
        nx = _cx(n.bbox) / vw
        ny = _cy(n.bbox) / vh
        scored.append((((nx - tx) ** 2 + (ny - ty) ** 2) ** 0.5, n))
    if not scored:
        return _TierResult(None, 0, f"{what}: no node exposes a bbox")
    scored.sort(key=lambda p: (p[0], p[1].ref))
    near = [p for p in scored if p[0] <= GEOMETRY_MAX_DIST]
    if not near:
        return _TierResult(
            None, 0, f"{what}: nearest node {scored[0][1].ref!r} is "
            f"{scored[0][0]:.3f} away, beyond threshold"
        )
    if len(near) > 1 and near[1][0] - near[0][0] < GEOMETRY_TIE_MARGIN:
        return _TierResult(
            None,
            len(near),
            f"{what}: {len(near)} nodes within threshold and no clear winner",
        )
    return _TierResult(near[0][1], len(near), f"{what}: dist={near[0][0]:.4f}")
