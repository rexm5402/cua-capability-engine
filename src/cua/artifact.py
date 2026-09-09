"""The capability artifact: the IR emitted by discovery and consumed by replay.

This module is the contract for the entire system. Discovery writes it, replay
reads it, humans review it, and calling agents consume its JSON Schema. Nothing
here may import Playwright, an LLM client, or any surface-specific code.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------
# Locators: how we point at a control, five independent ways, best first.
# --------------------------------------------------------------------------


class LocatorTier(int, Enum):
    """Ranked strategies. Lower is more durable; resolving above SEMANTIC is a
    drift signal, not merely a fallback."""

    SEMANTIC = 1  # role + accessible name
    ANCHOR_RELATIVE = 2  # positional relation to a labelled anchor
    TEXT = 3  # normalized text content
    STRUCTURAL = 4  # frame path + dom path; recorded but distrusted
    GEOMETRY = 5  # normalized bounding box; last resort


class SemanticLocator(BaseModel):
    tier: Literal[LocatorTier.SEMANTIC] = LocatorTier.SEMANTIC
    role: str
    name: str | None = None
    name_match: Literal["exact", "normalized", "contains"] = "normalized"
    scope: list[str] = Field(
        default_factory=list,
        description="Containing frame/landmark path, outermost first.",
    )


class Relation(str, Enum):
    SAME_ROW = "same_row"
    SAME_CELL = "same_cell"
    RIGHT_OF = "right_of"
    BELOW = "below"
    WITHIN = "within"


class AnchorRelativeLocator(BaseModel):
    """The workhorse for table-based legacy screens, where the only stable thing
    on the page is the label text printed beside the field."""

    tier: Literal[LocatorTier.ANCHOR_RELATIVE] = LocatorTier.ANCHOR_RELATIVE
    anchor_text: str
    relation: Relation
    target_role: str
    nth: int = 0


class TextLocator(BaseModel):
    tier: Literal[LocatorTier.TEXT] = LocatorTier.TEXT
    text: str
    match: Literal["exact", "normalized", "contains"] = "normalized"
    role: str | None = None
    nth: int = 0


class StructuralLocator(BaseModel):
    tier: Literal[LocatorTier.STRUCTURAL] = LocatorTier.STRUCTURAL
    frame_path: list[str] = Field(default_factory=list)
    dom_path: str


class GeometryLocator(BaseModel):
    tier: Literal[LocatorTier.GEOMETRY] = LocatorTier.GEOMETRY
    x: float = Field(ge=0.0, le=1.0, description="Normalized viewport coordinate.")
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


LocatorStrategy = Annotated[
    SemanticLocator
    | AnchorRelativeLocator
    | TextLocator
    | StructuralLocator
    | GeometryLocator,
    Field(discriminator="tier"),
]


class LocatorBundle(BaseModel):
    """All the ways we know to find one control. Resolution tries them in tier
    order and requires a UNIQUE match; ambiguity is a hard failure, never a
    silent first()."""

    description: str = Field(description="Human-readable, e.g. 'the Search button'.")
    strategies: list[LocatorStrategy] = Field(min_length=1)

    def by_tier(self) -> list[LocatorStrategy]:
        return sorted(self.strategies, key=lambda s: int(s.tier))


# --------------------------------------------------------------------------
# Typed contract: what the calling agent supplies and receives.
# --------------------------------------------------------------------------


class Sensitivity(str, Enum):
    PUBLIC = "public"
    PII = "pii"
    SECRET = "secret"


class ParamSpec(BaseModel):
    name: str
    type: Literal["string", "integer", "number", "boolean"]
    description: str
    required: bool = True
    sensitivity: Sensitivity = Sensitivity.PUBLIC
    pattern: str | None = None
    example: str | None = Field(
        default=None,
        description="Never a real value; synthetic only.",
    )


class OutputSpec(BaseModel):
    name: str
    type: Literal["string", "integer", "number", "boolean"]
    description: str
    source_step_id: str
    sensitivity: Sensitivity = Sensitivity.PUBLIC


# --------------------------------------------------------------------------
# Steps, checkpoints, and declared runtime conditions.
# --------------------------------------------------------------------------


class ActionType(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    PRESS = "press"
    READ = "read"
    WAIT = "wait"


class RiskClass(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class ValueRef(BaseModel):
    """A step's value is either a literal or a reference to a typed input.
    Sensitive values are NEVER literals; they bind late, at replay time."""

    literal: str | None = None
    param: str | None = None


class Checkpoint(BaseModel):
    """A condition asserted to confirm we actually reached the expected state,
    rather than assuming a click worked."""

    description: str
    locator: LocatorBundle | None = None
    text_present: str | None = None
    text_absent: str | None = None
    url_matches: str | None = None
    timeout_ms: int = 10_000


class OutcomeClass(str, Enum):
    """The taxonomy. Conflating BUSINESS with HARD is the failure mode this
    schema exists to prevent."""

    BUSINESS = "business"  # a legitimate answer the caller needs
    RECOVERABLE = "recoverable"  # handle inline; caller never hears about it
    HARD = "hard"  # stop and surface a debuggable error


class SignalRule(BaseModel):
    """Declared, not hardcoded: how an observation maps to a classification.
    Living in data is what makes a differently-worded error message a one-line
    tenant overlay instead of a code change."""

    code: str
    classification: OutcomeClass
    text_present: str | None = None
    locator: LocatorBundle | None = None
    message: str
    handler: Literal["dismiss", "retry", "reauth", "abort", "return"] | None = None
    max_attempts: int = 2


class Step(BaseModel):
    id: str
    intent: str = Field(
        description="Why this step exists, in prose. This is what a human "
        "reviewer approving the capability actually reads."
    )
    action: ActionType
    target: LocatorBundle | None = None
    value: ValueRef | None = None
    output_name: str | None = Field(
        default=None, description="For READ steps: which declared output this fills."
    )
    post_condition: Checkpoint | None = None
    risk: RiskClass = RiskClass.READ_ONLY
    timeout_ms: int = 10_000
    retries: int = 0


# --------------------------------------------------------------------------
# The artifact itself.
# --------------------------------------------------------------------------


class AppProfileRef(BaseModel):
    app_id: str
    vendor_product: str
    product_version: str | None = None
    tenant_id: str | None = Field(
        default=None,
        description="None means this is a base artifact, reusable across tenants.",
    )
    base_artifact_id: str | None = Field(
        default=None, description="Set when this artifact is a per-tenant overlay."
    )


class Provenance(BaseModel):
    recorded_by_model: str
    discovery_run_id: str
    recorded_at: str
    verified_at: str | None = Field(
        default=None,
        description="When the artifact was proven by a real LLM-free replay. "
        "An artifact without this was never allowed to be saved.",
    )


class StabilityStats(BaseModel):
    replays: int = 0
    successes: int = 0
    tier_escalations: int = Field(
        default=0,
        description="Resolutions above SEMANTIC. The drift early-warning signal.",
    )


class ApprovalState(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"


class Capability(BaseModel):
    """A reusable, reviewable, parameterized capability an AI agent can invoke."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    id: str
    name: str
    version: str = "1.0.0"
    description: str

    app_profile: AppProfileRef
    entry_url: str

    inputs: list[ParamSpec] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)

    steps: list[Step]
    success: Checkpoint
    signals: list[SignalRule] = Field(default_factory=list)

    risk_class: RiskClass = RiskClass.READ_ONLY
    approval_state: ApprovalState = ApprovalState.DRAFT
    provenance: Provenance
    stability: StabilityStats = Field(default_factory=StabilityStats)

    def tool_schema(self) -> dict[str, Any]:
        """JSON Schema for a calling agent. The point of the whole exercise:
        an agent invokes by name with typed args and never reads the steps."""
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.inputs:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.pattern:
                props[p.name]["pattern"] = p.pattern
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }
