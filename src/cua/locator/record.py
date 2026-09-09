"""Locator recording: synthesize a best-first LocatorBundle for a node.

The invariant here is self-validation. Every candidate strategy is run back
through :func:`cua.locator.resolve.resolve` against the very observation it was
derived from, and is kept ONLY if it uniquely returns the same node. We never
record a locator we have not proven unique at record time -- an ambiguous
strategy in a bundle is worse than no strategy, because it burns a tier and
teaches the drift counter a lie.
"""

from __future__ import annotations

from cua.artifact import (
    AnchorRelativeLocator,
    GeometryLocator,
    LocatorBundle,
    LocatorStrategy,
    Relation,
    SemanticLocator,
    StructuralLocator,
    TextLocator,
)
from cua.locator.resolve import normalize, resolve
from cua.surface.base import Node, Observation

__all__ = ["build_bundle", "describe", "find_anchor_candidates"]


#: Roles that typically carry a printed label rather than accept input.
LABELISH_ROLES = frozenset(
    {
        "label",
        "text",
        "statictext",
        "static text",
        "cell",
        "gridcell",
        "columnheader",
        "rowheader",
        "heading",
        "paragraph",
        "legend",
        "caption",
        "generic",
    }
)

#: Roles a user actually acts on. Used to reject them as anchors.
INTERACTIVE_ROLES = frozenset(
    {
        "textbox",
        "combobox",
        "listbox",
        "checkbox",
        "radio",
        "button",
        "link",
        "menuitem",
        "spinbutton",
        "slider",
        "searchbox",
        "switch",
        "tab",
    }
)

#: How far left / above a label may sit and still be considered "this field's
#: label". Generous enough for a wide table cell, tight enough to not steal a
#: label from the previous column.
MAX_ANCHOR_DX = 420.0
MAX_ANCHOR_DY = 90.0


def describe(node: Node) -> str:
    """Human-readable bundle description, e.g. ``the "Submit" button``."""
    role = (node.role or "control").strip()
    label = (node.name or node.text or "").strip()
    if label:
        return f'the "{label}" {role}'
    return f"the unnamed {role} at {node.dom_path or node.ref}"


def _anchor_text_of(n: Node) -> str | None:
    for candidate in (n.name, n.text):
        if candidate and normalize(candidate):
            return candidate.strip()
    return None


def find_anchor_candidates(node: Node, obs: Observation) -> list[tuple[Node, Relation]]:
    """Nearest label-ish nodes to the LEFT of (same row) or ABOVE the target,
    best first, each paired with the relation that describes the arrangement.

    This is how tier 2 gets recorded automatically: nobody hand-writes
    ``anchor_text="Account Number"``; we infer it from the printed label the
    designer already put next to the field.
    """
    out: list[tuple[float, int, Node, Relation]] = []
    if node.bbox is None:
        # No geometry: fall back to the nearest preceding label-ish sibling.
        order = list(obs.nodes)
        try:
            idx = order.index(node)
        except ValueError:
            idx = next(
                (i for i, n in enumerate(order) if n.ref == node.ref), len(order)
            )
        for back, n in enumerate(reversed(order[:idx]), start=1):
            if n.frame_path != node.frame_path:
                continue
            if not _is_labelish(n) or _anchor_text_of(n) is None:
                continue
            out.append((float(back), 0, n, Relation.SAME_ROW))
        out.sort(key=lambda t: (t[0], t[1]))
        return [(n, r) for _, _, n, r in out]

    tx, ty, tw, th = node.bbox
    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
    for n in obs.nodes:
        if n.ref == node.ref or n.bbox is None:
            continue
        if n.frame_path != node.frame_path:
            continue
        if not _is_labelish(n) or _anchor_text_of(n) is None:
            continue
        ax, ay, aw, ah = n.bbox
        acy = ay + ah / 2.0
        # Same row, to the left.
        row_tol = max(ah, th) / 2.0 + 6.0
        if abs(acy - tcy) <= row_tol and ax + aw <= tx + 2.0:
            dx = tx - (ax + aw)
            if dx <= MAX_ANCHOR_DX:
                out.append((dx, 0, n, Relation.SAME_ROW))
                continue
        # Directly above.
        if ay + ah <= ty + 2.0 and abs((ax + aw / 2.0) - tcx) <= max(aw, tw):
            dy = ty - (ay + ah)
            if dy <= MAX_ANCHOR_DY:
                out.append((dy, 1, n, Relation.BELOW))
    out.sort(key=lambda t: (t[1], t[0]))
    return [(n, r) for _, _, n, r in out]


def _is_labelish(n: Node) -> bool:
    role = normalize(n.role).replace(" ", "")
    if role in {r.replace(" ", "") for r in INTERACTIVE_ROLES}:
        return False
    return role in {r.replace(" ", "") for r in LABELISH_ROLES} or role == ""


def _validates(strategy: LocatorStrategy, node: Node, obs: Observation) -> bool:
    """A strategy is kept only if, on its own, it uniquely returns ``node``."""
    probe = LocatorBundle(description="validation probe", strategies=[strategy])
    res = resolve(probe, obs)
    return res.node is not None and res.node.ref == node.ref


def _candidates(node: Node, obs: Observation) -> list[LocatorStrategy]:
    out: list[LocatorStrategy] = []

    # Tier 1 -- semantic.
    if node.role:
        if node.name:
            out.append(
                SemanticLocator(role=node.role, name=node.name, name_match="normalized")
            )
            if node.frame_path:
                out.append(
                    SemanticLocator(
                        role=node.role,
                        name=node.name,
                        name_match="normalized",
                        scope=list(node.frame_path),
                    )
                )
        else:
            out.append(SemanticLocator(role=node.role))

    # Tier 2 -- anchor relative, inferred from the nearest printed label.
    for anchor, relation in find_anchor_candidates(node, obs)[:4]:
        text = _anchor_text_of(anchor)
        if text is None:
            continue
        out.append(
            AnchorRelativeLocator(
                anchor_text=text, relation=relation, target_role=node.role
            )
        )
        if relation is Relation.SAME_ROW:
            out.append(
                AnchorRelativeLocator(
                    anchor_text=text,
                    relation=Relation.RIGHT_OF,
                    target_role=node.role,
                )
            )

    # Tier 3 -- text.
    for text in (node.text, node.name):
        if text and normalize(text):
            out.append(TextLocator(text=text, match="normalized", role=node.role))
            break

    # Tier 4 -- structural (recorded, distrusted).
    if node.dom_path:
        out.append(
            StructuralLocator(
                frame_path=list(node.frame_path), dom_path=node.dom_path
            )
        )

    # Tier 5 -- geometry, normalized against the observation's extent.
    if node.bbox:
        vw = max(
            (n.bbox[0] + n.bbox[2] for n in obs.nodes if n.bbox), default=1.0
        ) or 1.0
        vh = max(
            (n.bbox[1] + n.bbox[3] for n in obs.nodes if n.bbox), default=1.0
        ) or 1.0
        x, y, w, h = node.bbox
        out.append(
            GeometryLocator(
                x=min(max(x / vw, 0.0), 1.0),
                y=min(max(y / vh, 0.0), 1.0),
                w=min(max(w / vw, 0.0), 1.0),
                h=min(max(h / vh, 0.0), 1.0),
            )
        )
    return out


def build_bundle(node: Node, obs: Observation) -> LocatorBundle:
    """Synthesize every applicable strategy for ``node``, best-first, keeping
    only those proven to uniquely resolve back to ``node`` in ``obs``.

    Raises:
        ValueError: if no strategy uniquely identifies the node -- meaning the
            control is genuinely indistinguishable and must not be recorded.
    """
    kept: list[LocatorStrategy] = []
    seen: set[str] = set()
    for strategy in _candidates(node, obs):
        key = strategy.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        if _validates(strategy, node, obs):
            kept.append(strategy)

    if not kept:
        raise ValueError(
            f"cannot record {describe(node)}: no candidate strategy uniquely "
            f"resolved it in this observation (ref={node.ref!r}). Recording an "
            "ambiguous locator is worse than recording none."
        )
    kept.sort(key=lambda s: int(s.tier))
    return LocatorBundle(description=describe(node), strategies=kept)
