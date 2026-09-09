"""The signal scan: the heart of the error taxonomy.

After every action the engine asks one question of the current observation --
"is the world telling me something?" -- and the answer is looked up in DATA
(`capability.signals`), never in code. That is what makes "no records located"
a *business answer* at one tenant and a differently worded business answer at
the next without a code change.

Nothing here imports an LLM client, Playwright, or any concrete surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from cua.artifact import LocatorBundle, OutcomeClass, SignalRule
from cua.locator.resolve import Resolution, resolve
from cua.surface.base import Node, Observation

__all__ = [
    "SignalMatch",
    "scan",
    "match_rule",
    "haystack",
    "find_dismiss_control",
    "DISMISS_NAMES",
]


#: Fallback continue-controls for a `dismiss` rule that declared only text.
DISMISS_NAMES = (
    "continue",
    "ok",
    "dismiss",
    "close",
    "acknowledge",
    "proceed",
    "got it",
    "i understand",
)


@dataclass(frozen=True)
class SignalMatch:
    """One fired rule, plus how it fired -- enough to build a Failure payload."""

    rule: SignalRule
    matched_on: str
    observed: str
    resolved_tier: int | None = None
    candidates: int = 0
    node: Node | None = None

    @property
    def code(self) -> str:
        return self.rule.code

    @property
    def classification(self) -> OutcomeClass:
        return self.rule.classification

    @property
    def message(self) -> str:
        return self.rule.message

    @property
    def is_business(self) -> bool:
        return self.rule.classification is OutcomeClass.BUSINESS

    @property
    def is_recoverable(self) -> bool:
        return self.rule.classification is OutcomeClass.RECOVERABLE

    @property
    def is_hard(self) -> bool:
        return self.rule.classification is OutcomeClass.HARD


def haystack(obs: Observation) -> str:
    """The text we match `text_present` against.

    `text_digest` when the surface supplies one, otherwise the rendered control
    list -- so a signal expressed as button/label text still fires on a surface
    that has no page text (a native desktop app, for instance).
    """
    digest = obs.text_digest or ""
    return f"{digest}\n{obs.render(limit=500)}".casefold()


def _resolve_locator(bundle: LocatorBundle, obs: Observation) -> Resolution:
    try:
        return resolve(bundle, obs)
    except Exception as exc:  # a broken locator must not crash the scan
        return Resolution(node=None, tier=None, candidates=0, reason=f"raised {exc!r}")


def match_rule(rule: SignalRule, obs: Observation) -> SignalMatch | None:
    """A rule with both a text and a locator condition requires BOTH.

    A rule with neither never fires: an always-true signal would classify every
    observation and is a recording bug, not a runtime condition.
    """
    if rule.text_present is None and rule.locator is None:
        return None

    matched_on: list[str] = []
    tier: int | None = None
    candidates = 0
    node: Node | None = None

    if rule.text_present is not None:
        if rule.text_present.casefold() not in haystack(obs):
            return None
        matched_on.append(f"text_present={rule.text_present!r}")

    if rule.locator is not None:
        res = _resolve_locator(rule.locator, obs)
        if res.node is None:
            return None
        matched_on.append(f"locator={rule.locator.description!r}")
        tier = int(res.tier) if res.tier is not None else None
        candidates = res.candidates
        node = res.node

    return SignalMatch(
        rule=rule,
        matched_on=" and ".join(matched_on),
        observed=_observed_summary(obs, rule),
        resolved_tier=tier,
        candidates=candidates,
        node=node,
    )


def _observed_summary(obs: Observation, rule: SignalRule) -> str:
    bits = [f"url={obs.url!r}", f"title={obs.title!r}"]
    if rule.text_present:
        bits.append(f"page contains {rule.text_present!r}")
    if rule.locator:
        bits.append(f"page shows {rule.locator.description!r}")
    return "; ".join(bits)


def scan(rules: list[SignalRule], obs: Observation) -> SignalMatch | None:
    """First declared rule that fires wins.

    Declaration order is the precedence order, which is why an artifact lists
    its specific signals before its catch-all ones.
    """
    for rule in rules:
        hit = match_rule(rule, obs)
        if hit is not None:
            return hit
    return None


def find_dismiss_control(match: SignalMatch, obs: Observation) -> Node | None:
    """The control that makes a recoverable interstitial go away.

    Preference order: the node the rule's own locator resolved to (an
    interstitial is usually declared by pointing at its continue button), then a
    button/link whose accessible name is a conventional acknowledgement.
    """
    if match.node is not None:
        return match.node
    for wanted in DISMISS_NAMES:
        for n in obs.nodes:
            if not n.enabled:
                continue
            if n.role not in ("button", "link", "menuitem"):
                continue
            if n.name.strip().casefold() == wanted:
                return n
    for wanted in DISMISS_NAMES:
        for n in obs.nodes:
            if n.enabled and wanted in n.name.strip().casefold():
                return n
    return None
