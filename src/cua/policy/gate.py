"""The policy gate: one chokepoint every action must pass through.

The design claim this module makes is structural, not advisory. Nothing in the
replay engine is trusted to "remember" to check an allowlist; the engine cannot
reach a surface without producing a `Decision` first, and the only constructor
of an `Allow` lives here. Deny-by-default is the base case: an empty policy
denies everything, so a misconfigured or truncated YAML fails closed rather
than silently opening the world.

Nothing here imports Playwright, an LLM client, or any surface-specific code.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field

from cua.artifact import ActionType, RiskClass
from cua.surface.base import Action

DEFAULT_POLICY_PATH = Path(__file__).with_name("policy.yaml")


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


class DenyCode(str):
    """String enum-ish namespace of stable machine-readable deny codes."""


NO_POLICY = "no_policy"
ORIGIN_NOT_ALLOWED = "origin_not_allowed"
ROUTE_NOT_ALLOWED = "route_not_allowed"
ACTION_TYPE_NOT_ALLOWED = "action_type_not_allowed"
IRREVERSIBLE_UNATTENDED = "irreversible_in_unattended_mode"
RISK_NOT_ALLOWED = "risk_class_not_allowed"
MALFORMED_URL = "malformed_url"


class Allow(BaseModel):
    kind: Literal["allow"] = "allow"
    reason: str = "allowlisted"


class Deny(BaseModel):
    kind: Literal["deny"] = "deny"
    reason: str
    code: str


class RequireApproval(BaseModel):
    """Deliberately routed through the same human-escalation channel used for
    stuck states: one queue for every moment a human must look at the run."""

    kind: Literal["require_approval"] = "require_approval"
    reason: str
    code: str = "approval_required"


Decision = Allow | Deny | RequireApproval


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


class RiskRule(BaseModel):
    """Per-risk-class override. Absent means: fall back to the built-in rule."""

    allowed: bool = True
    require_approval: bool = False


class Policy(BaseModel):
    """Loaded from YAML. Every field defaults to the closed position, so a
    `Policy()` built from an empty document denies every action."""

    name: str = "unnamed"
    mode: Literal["unattended", "attended"] = "unattended"
    allowed_origins: list[str] = Field(default_factory=list)
    allowed_routes: list[str] = Field(default_factory=list)
    allowed_action_types: list[ActionType] = Field(default_factory=list)
    risk_rules: dict[RiskClass, RiskRule] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"policy file {path} must contain a mapping")
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> "Policy":
        return cls.from_yaml(DEFAULT_POLICY_PATH)

    @classmethod
    def empty(cls) -> "Policy":
        """The deny-everything policy. Useful as an explicit test baseline."""
        return cls(name="empty")


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def _origin_of(url: str) -> str | None:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


def _route_of(url: str) -> str:
    parts = urlsplit(url)
    route = parts.path or "/"
    if parts.query:
        route = f"{route}?{parts.query}"
    return route


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(value, p) for p in patterns)


class PolicyGate:
    """The single chokepoint. `check` is total: it returns a Decision for every
    input, and every path that is not an explicit allow returns Deny."""

    def __init__(self, policy: Policy | None = None) -> None:
        # None is not "no restrictions" -- it is the empty, deny-all policy.
        self.policy = policy if policy is not None else Policy.empty()

    # -- public API --------------------------------------------------------

    def check(
        self, action: Action, *, current_url: str, risk: RiskClass
    ) -> Decision:
        p = self.policy

        if not p.allowed_origins or not p.allowed_action_types:
            return Deny(
                reason=(
                    "policy declares no allowed origins or no allowed action "
                    "types; deny-by-default applies"
                ),
                code=NO_POLICY,
            )

        # 1. Action type must be explicitly allowlisted.
        if action.type not in p.allowed_action_types:
            return Deny(
                reason=f"action type {action.type.value!r} is not in the allowlist",
                code=ACTION_TYPE_NOT_ALLOWED,
            )

        # 2. Every URL this action touches must be allowlisted. For a NAVIGATE
        #    that is BOTH where we are and where we are going -- an allowlisted
        #    page must not be usable as a springboard off the allowlist.
        urls = [current_url]
        if action.type is ActionType.NAVIGATE and action.url:
            urls.append(action.url)

        for url in urls:
            origin = _origin_of(url)
            if origin is None:
                return Deny(
                    reason=f"cannot determine an origin for {url!r}",
                    code=MALFORMED_URL,
                )
            if not _matches_any(origin, p.allowed_origins):
                return Deny(
                    reason=f"origin {origin!r} is not in the allowlist",
                    code=ORIGIN_NOT_ALLOWED,
                )
            route = _route_of(url)
            if not _matches_any(route, p.allowed_routes):
                return Deny(
                    reason=f"route {route!r} on {origin!r} is not in the allowlist",
                    code=ROUTE_NOT_ALLOWED,
                )

        # 3. Risk handling.
        return self._check_risk(risk)

    # -- internals ---------------------------------------------------------

    def _check_risk(self, risk: RiskClass) -> Decision:
        p = self.policy
        rule = p.risk_rules.get(risk)

        if rule is not None and not rule.allowed:
            return Deny(
                reason=f"risk class {risk.value!r} is disallowed by policy",
                code=RISK_NOT_ALLOWED,
            )

        if risk is RiskClass.IRREVERSIBLE:
            # An irreversible action is never taken by an unattended agent. In
            # attended mode it does not proceed either -- it leaves the machine
            # and enters the human escalation queue.
            if p.mode == "unattended":
                return Deny(
                    reason=(
                        "irreversible actions are denied outright in unattended "
                        "mode"
                    ),
                    code=IRREVERSIBLE_UNATTENDED,
                )
            return RequireApproval(
                reason="irreversible action requires human approval in attended mode"
            )

        # read_only / reversible proceed if allowlisted, unless policy asks for
        # approval on this class explicitly.
        if rule is not None and rule.require_approval:
            return RequireApproval(
                reason=f"policy requires approval for risk class {risk.value!r}"
            )
        return Allow(reason=f"allowlisted; risk class {risk.value!r} may proceed")


__all__ = [
    "Policy",
    "PolicyGate",
    "RiskRule",
    "Decision",
    "Allow",
    "Deny",
    "RequireApproval",
    "DEFAULT_POLICY_PATH",
    "NO_POLICY",
    "ORIGIN_NOT_ALLOWED",
    "ROUTE_NOT_ALLOWED",
    "ACTION_TYPE_NOT_ALLOWED",
    "IRREVERSIBLE_UNATTENDED",
    "RISK_NOT_ALLOWED",
    "MALFORMED_URL",
]
