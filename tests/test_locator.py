"""Locator engine tests.

Fixtures are hand-built accessibility trees for a legacy-bank style screen --
a label/input table, the way 1998 built forms. No browser is involved: the
locator engine only ever sees Observation/Node, which is the whole point of the
surface abstraction.
"""

from __future__ import annotations

import pytest

from cua.artifact import (
    AnchorRelativeLocator,
    GeometryLocator,
    LocatorBundle,
    Relation,
    SemanticLocator,
    StructuralLocator,
    TextLocator,
)
from cua.locator import build_bundle, resolve
from cua.locator.resolve import normalize
from cua.surface.base import Node, Observation


# --------------------------------------------------------------------------
# Fixtures: "Customer Search" screen, v1 (as recorded)
# --------------------------------------------------------------------------


def _n(ref, role, name, bbox, dom_path, **kw) -> Node:
    return Node(ref=ref, role=role, name=name, bbox=bbox, dom_path=dom_path, **kw)


@pytest.fixture
def obs_v1() -> Observation:
    """Three labelled rows plus two buttons. Crucially the inputs have NO
    accessible name -- exactly the legacy case where tier 1 cannot work."""
    nodes = (
        _n("h1", "heading", "Customer Search", (20, 40, 300, 30), "/html/body/h1"),
        _n("l_acct", "cell", "Account Number:", (20, 100, 140, 24),
           "/html/body/table/tr[1]/td[1]"),
        _n("i_acct", "textbox", "", (180, 100, 200, 24),
           "/html/body/table/tr[1]/td[2]/input"),
        _n("l_sort", "cell", "Sort Code", (20, 140, 140, 24),
           "/html/body/table/tr[2]/td[1]"),
        _n("i_sort", "textbox", "", (180, 140, 200, 24),
           "/html/body/table/tr[2]/td[2]/input"),
        _n("l_branch", "cell", "Branch", (20, 180, 140, 24),
           "/html/body/table/tr[3]/td[1]"),
        _n("i_branch", "textbox", "", (180, 180, 200, 24),
           "/html/body/table/tr[3]/td[2]/input"),
        _n("b_search", "button", "Search", (180, 220, 90, 30),
           "/html/body/table/tr[4]/td[2]/button[1]"),
        _n("b_reset", "button", "Reset", (280, 220, 90, 30),
           "/html/body/table/tr[4]/td[2]/button[2]"),
    )
    return Observation(url="https://bank.example/search", title="Customer Search",
                       nodes=nodes)


@pytest.fixture
def obs_v2() -> Observation:
    """The SAME screen next month at a different customer: the Sort Code and
    Branch rows are swapped, every row is taller and shifted right (restyle),
    and the markup was rebuilt so every dom_path changed. Only the printed
    labels survived -- which is exactly the bet tier 2 makes."""
    nodes = (
        _n("x0", "heading", "Customer Search", (50, 30, 340, 40),
           "/html/body/div[1]/header/h1"),
        _n("x1", "cell", "Account Number *", (50, 100, 160, 32),
           "/html/body/div[1]/form/div[1]/label"),
        _n("x2", "textbox", "", (240, 100, 220, 32),
           "/html/body/div[1]/form/div[1]/input"),
        # rows below reordered relative to v1
        _n("x3", "cell", "Branch:", (50, 150, 160, 32),
           "/html/body/div[1]/form/div[2]/label"),
        _n("x4", "textbox", "", (240, 150, 220, 32),
           "/html/body/div[1]/form/div[2]/input"),
        _n("x5", "cell", "  sort   code  ", (50, 200, 160, 32),
           "/html/body/div[1]/form/div[3]/label"),
        _n("x6", "textbox", "", (240, 200, 220, 32),
           "/html/body/div[1]/form/div[3]/input"),
        _n("x7", "button", "Search", (240, 260, 110, 36),
           "/html/body/div[1]/form/div[4]/button.primary"),
        _n("x8", "button", "Reset", (370, 260, 110, 36),
           "/html/body/div[1]/form/div[4]/button.ghost"),
    )
    return Observation(url="https://other.example/customer/search",
                       title="Customer Search", nodes=nodes)


def _bundle(*strategies) -> LocatorBundle:
    return LocatorBundle(description="test target", strategies=list(strategies))


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_normalize_strips_case_whitespace_and_label_punctuation():
    assert normalize("Account Number:") == "account number"
    assert normalize("  sort   code  ") == "sort code"
    assert normalize("Account Number *") == "account number"
    assert normalize(None) == ""


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------


def test_semantic_resolves_clean_match(obs_v1):
    res = resolve(_bundle(SemanticLocator(role="button", name="Search")), obs_v1)
    assert res.node is not None and res.node.ref == "b_search"
    assert res.tier == 1
    assert res.candidates == 1


def test_semantic_normalized_name_beats_restyled_label(obs_v1):
    res = resolve(_bundle(SemanticLocator(role="cell", name="account number")), obs_v1)
    assert res.node.ref == "l_acct"


def test_semantic_scope_is_respected():
    scoped = Node(ref="a", role="button", name="Save",
                  frame_path=("main", "dialog"), bbox=(0, 0, 10, 10))
    other = Node(ref="b", role="button", name="Save",
                 frame_path=("main", "sidebar"), bbox=(0, 50, 10, 10))
    obs = Observation(url="u", title="t", nodes=(scoped, other))
    res = resolve(_bundle(SemanticLocator(role="button", name="Save",
                                          scope=["dialog"])), obs)
    assert res.node.ref == "a"
    # without the scope it is genuinely ambiguous
    assert resolve(_bundle(SemanticLocator(role="button", name="Save")), obs).node is None


# --------------------------------------------------------------------------
# Ambiguity must fall through, never guess
# --------------------------------------------------------------------------


def test_tier1_ambiguity_falls_through_instead_of_guessing(obs_v1):
    """Three unnamed textboxes match role=textbox. Taking the first would be a
    50/50 wrong write into the wrong field; we must escalate to tier 2."""
    bundle = _bundle(
        SemanticLocator(role="textbox"),
        AnchorRelativeLocator(anchor_text="Sort Code", relation=Relation.SAME_ROW,
                              target_role="textbox"),
    )
    res = resolve(bundle, obs_v1)
    assert res.node.ref == "i_sort"
    assert res.tier == 2, "must report the tier that actually resolved"
    assert "ambiguous" in res.reason and "tier 1" in res.reason


def test_ambiguous_tier_alone_returns_no_node_with_candidate_count(obs_v1):
    res = resolve(_bundle(SemanticLocator(role="textbox")), obs_v1)
    assert res.node is None
    assert res.tier is None
    assert res.candidates == 3


# --------------------------------------------------------------------------
# Tier 2 -- the money test
# --------------------------------------------------------------------------


def test_anchor_relative_finds_input_in_same_table_row(obs_v1):
    res = resolve(
        _bundle(AnchorRelativeLocator(anchor_text="Account Number",
                                      relation=Relation.SAME_ROW,
                                      target_role="textbox")),
        obs_v1,
    )
    assert res.node.ref == "i_acct"
    assert res.tier == 2


def test_anchor_relative_survives_reorder_and_restyle(obs_v2):
    """THE durability test: rows reordered, geometry changed, dom paths all
    rewritten, label punctuation different -- same locator, right field."""
    for anchor, expected in (
        ("Account Number:", "x2"),
        ("Sort Code", "x6"),
        ("Branch", "x4"),
    ):
        res = resolve(
            _bundle(AnchorRelativeLocator(anchor_text=anchor,
                                          relation=Relation.SAME_ROW,
                                          target_role="textbox")),
            obs_v2,
        )
        assert res.node is not None, res.reason
        assert res.node.ref == expected, f"{anchor} -> {res.node.ref}: {res.reason}"
        assert res.tier == 2


def test_anchor_relative_skips_nodes_without_bbox_instead_of_crashing(obs_v1):
    nodes = obs_v1.nodes + (
        Node(ref="ghost", role="textbox", name="", bbox=None, dom_path="/ghost"),
    )
    obs = Observation(url=obs_v1.url, title=obs_v1.title, nodes=nodes)
    res = resolve(
        _bundle(AnchorRelativeLocator(anchor_text="Branch",
                                      relation=Relation.SAME_ROW,
                                      target_role="textbox")),
        obs,
    )
    assert res.node.ref == "i_branch"


def test_anchor_relative_falls_back_to_dom_order_without_geometry():
    """Native/desktop surfaces sometimes expose no bbox at all. Adjacency in
    tree order is then the only signal, and it must still work."""
    nodes = (
        Node(ref="l1", role="cell", name="Account Number:"),
        Node(ref="t1", role="textbox", name=""),
        Node(ref="l2", role="cell", name="Sort Code"),
        Node(ref="t2", role="textbox", name=""),
    )
    obs = Observation(url="u", title="t", nodes=nodes)
    res = resolve(
        _bundle(AnchorRelativeLocator(anchor_text="Sort Code",
                                      relation=Relation.SAME_ROW,
                                      target_role="textbox")),
        obs,
    )
    assert res.node.ref == "t2"
    assert "dom-order fallback" in res.reason


def test_anchor_relative_nth_selects_further_control():
    nodes = (
        Node(ref="lbl", role="cell", name="Date Range", bbox=(0, 0, 100, 20)),
        Node(ref="from", role="textbox", name="", bbox=(120, 0, 60, 20)),
        Node(ref="to", role="textbox", name="", bbox=(200, 0, 60, 20)),
    )
    obs = Observation(url="u", title="t", nodes=nodes)
    first = resolve(_bundle(AnchorRelativeLocator(
        anchor_text="Date Range", relation=Relation.SAME_ROW,
        target_role="textbox", nth=0)), obs)
    second = resolve(_bundle(AnchorRelativeLocator(
        anchor_text="Date Range", relation=Relation.SAME_ROW,
        target_role="textbox", nth=1)), obs)
    assert first.node.ref == "from"
    assert second.node.ref == "to"


def test_anchor_relative_within_and_below():
    panel = Node(ref="panel", role="group", name="Filters", bbox=(0, 0, 300, 200))
    inside = Node(ref="inner", role="checkbox", name="", bbox=(20, 40, 20, 20))
    outside = Node(ref="outer", role="checkbox", name="", bbox=(400, 40, 20, 20))
    header = Node(ref="hdr", role="cell", name="Notes", bbox=(0, 300, 100, 20))
    under = Node(ref="area", role="textbox", name="", bbox=(0, 330, 200, 60))
    obs = Observation(url="u", title="t",
                      nodes=(panel, inside, outside, header, under))
    within = resolve(_bundle(AnchorRelativeLocator(
        anchor_text="Filters", relation=Relation.WITHIN,
        target_role="checkbox")), obs)
    assert within.node.ref == "inner"
    below = resolve(_bundle(AnchorRelativeLocator(
        anchor_text="Notes", relation=Relation.BELOW,
        target_role="textbox")), obs)
    assert below.node.ref == "area"


def test_anchor_relative_reports_tie_as_ambiguous_not_a_coin_flip():
    nodes = (
        Node(ref="lbl", role="cell", name="Amount", bbox=(0, 0, 100, 20)),
        Node(ref="a", role="textbox", name="", bbox=(120, 0, 40, 20)),
        Node(ref="b", role="textbox", name="", bbox=(120, 2, 40, 20)),
    )
    obs = Observation(url="u", title="t", nodes=nodes)
    res = resolve(_bundle(AnchorRelativeLocator(
        anchor_text="Amount", relation=Relation.SAME_ROW,
        target_role="textbox")), obs)
    assert res.node is None
    assert "tie" in res.reason


# --------------------------------------------------------------------------
# Tier 3 / 4 / 5
# --------------------------------------------------------------------------


def test_text_locator_with_role_constraint_and_nth(obs_v1):
    res = resolve(_bundle(TextLocator(text="reset", role="button")), obs_v1)
    assert res.node.ref == "b_reset"
    assert res.tier == 3


def test_structural_matches_exact_frame_and_dom_path(obs_v1):
    res = resolve(
        _bundle(StructuralLocator(
            frame_path=[], dom_path="/html/body/table/tr[2]/td[2]/input")),
        obs_v1,
    )
    assert res.node.ref == "i_sort"
    assert res.tier == 4


def test_structural_breaks_on_dom_change_and_bundle_escalates(obs_v1, obs_v2):
    """A dom_path is a hostage to the vendor's next release. When it breaks the
    bundle must still resolve -- and must SAY it escalated, so the stability
    counter can flag drift before it becomes an outage."""
    structural = StructuralLocator(
        frame_path=[], dom_path="/html/body/table/tr[1]/td[2]/input")

    # v1: the structural locator alone is fine.
    assert resolve(_bundle(structural), obs_v1).node.ref == "i_acct"

    # v2: same screen, rebuilt markup -> structural alone is dead.
    dead = resolve(_bundle(structural), obs_v2)
    assert dead.node is None
    assert dead.candidates == 0

    # The full bundle survives via the anchor tier, and reports the escalation.
    bundle = _bundle(
        SemanticLocator(role="textbox"),  # ambiguous on this screen
        AnchorRelativeLocator(anchor_text="Account Number",
                              relation=Relation.SAME_ROW, target_role="textbox"),
        structural,
    )
    res = resolve(bundle, obs_v2)
    assert res.node.ref == "x2"
    assert res.tier == 2
    assert res.tier > 1, "resolving above SEMANTIC is the drift signal"


def test_geometry_accepts_only_a_close_unambiguous_match(obs_v1):
    # b_search occupies (180,220,90,30); viewport extent is 380 x 250.
    good = GeometryLocator(x=180 / 380, y=220 / 250, w=90 / 380, h=30 / 250)
    res = resolve(_bundle(good), obs_v1)
    assert res.node.ref == "b_search"
    assert res.tier == 5

    far = GeometryLocator(x=0.02, y=0.60, w=0.05, h=0.05)
    miss = resolve(_bundle(far), obs_v1)
    assert miss.node is None
    assert "beyond threshold" in miss.reason


# --------------------------------------------------------------------------
# Failure reporting
# --------------------------------------------------------------------------


def test_total_failure_returns_debuggable_reason(obs_v1):
    bundle = LocatorBundle(
        description="the Wire Transfer button",
        strategies=[
            SemanticLocator(role="button", name="Wire Transfer"),
            SemanticLocator(role="textbox"),
            AnchorRelativeLocator(anchor_text="IBAN", relation=Relation.SAME_ROW,
                                  target_role="textbox"),
            StructuralLocator(frame_path=[], dom_path="/nope"),
        ],
    )
    res = resolve(bundle, obs_v1)
    assert res.node is None and res.tier is None
    reason = res.reason
    assert "the Wire Transfer button" in reason
    for fragment in ("tier 1", "tier 2", "tier 4", "SEMANTIC", "ANCHOR_RELATIVE",
                     "STRUCTURAL", "0 candidate", "3 candidate", "no node carries"):
        assert fragment in reason, f"missing {fragment!r} in: {reason}"
    assert obs_v1.url in reason


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def test_build_bundle_round_trips_to_the_same_node(obs_v1):
    for node in obs_v1.nodes:
        bundle = build_bundle(node, obs_v1)
        res = resolve(bundle, obs_v1)
        assert res.node is not None, f"{node.ref}: {res.reason}"
        assert res.node.ref == node.ref
        # every recorded strategy must independently be unique too
        for s in bundle.strategies:
            one = resolve(LocatorBundle(description="x", strategies=[s]), obs_v1)
            assert one.node is not None and one.node.ref == node.ref, (
                f"{node.ref} recorded a non-unique tier {int(s.tier)} strategy"
            )


def test_build_bundle_infers_the_printed_label_as_anchor(obs_v1):
    target = next(n for n in obs_v1.nodes if n.ref == "i_acct")
    bundle = build_bundle(target, obs_v1)
    anchors = [s for s in bundle.strategies if isinstance(s, AnchorRelativeLocator)]
    assert anchors, "an unnamed input beside a label must record a tier-2 anchor"
    assert any(normalize(a.anchor_text) == "account number" for a in anchors)
    assert bundle.strategies[0].tier <= anchors[0].tier  # best-first ordering


def test_build_bundle_drops_strategies_that_would_be_ambiguous(obs_v1):
    target = next(n for n in obs_v1.nodes if n.ref == "i_acct")
    bundle = build_bundle(target, obs_v1)
    # role=textbox with no name matches three nodes; it must NOT be recorded.
    assert not any(
        isinstance(s, SemanticLocator) and not s.name for s in bundle.strategies
    )


def test_build_bundle_recorded_at_v1_still_resolves_at_v2(obs_v1, obs_v2):
    """End to end: record against one tenant, replay against another."""
    target = next(n for n in obs_v1.nodes if n.ref == "i_acct")
    bundle = build_bundle(target, obs_v1)
    res = resolve(bundle, obs_v2)
    assert res.node is not None, res.reason
    assert res.node.ref == "x2"
    assert res.tier == 2


def test_build_bundle_generates_human_readable_description(obs_v1):
    btn = next(n for n in obs_v1.nodes if n.ref == "b_search")
    assert build_bundle(btn, obs_v1).description == 'the "Search" button'


def test_build_bundle_refuses_to_record_an_indistinguishable_node():
    a = Node(ref="a", role="textbox", name="")
    b = Node(ref="b", role="textbox", name="")
    obs = Observation(url="u", title="t", nodes=(a, b))
    with pytest.raises(ValueError, match="no candidate strategy uniquely"):
        build_bundle(a, obs)
