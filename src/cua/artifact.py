"""The capability artifact: the IR emitted by discovery and consumed by replay.

This module is the contract for the entire system. Discovery writes it, replay
reads it, humans review it, and calling agents consume its JSON Schema. Nothing
here may import Playwright, an LLM client, or any surface-specific code.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    scope_text: str | None = Field(
        default=None,
        description="Restrict candidates to the container region holding this "
        "text, THEN require a unique match within it. A results grid has eight "
        "identical 'View' buttons; a globally-unique rule would make every "
        "list interaction fail, while taking the first would be guessing. "
        "Scope-then-unique keeps the no-guessing rule and makes lists workable.",
    )

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
    type: Literal["string", "integer", "number", "boolean", "array", "object"]
    description: str
    source_step_id: str
    sensitivity: Sensitivity = Sensitivity.PUBLIC
    item_type: Literal["string", "integer", "number", "boolean", "object"] | None = (
        Field(default=None, description="Element type when type == 'array'.")
    )
    shape: dict[str, str] | None = Field(
        default=None,
        description="Field name -> type, when type or item_type is 'object'. "
        "A capability that returns a list of sub-accounts is ordinary; scalars "
        "alone cannot express it.",
    )


class OutcomeSpec(BaseModel):
    """A business outcome this capability can legitimately return.

    Declared in the contract so a calling agent knows 'record_not_found' is a
    possible ANSWER. Without this the caller sees only input_schema, treats an
    unexpected outcome as an error, and reintroduces exactly the confusion the
    three-type result union exists to prevent.
    """

    code: str
    description: str
    partial_outputs: list[str] = Field(default_factory=list)


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

    Late binding is not optional for sensitive data. The discovery agent emits
    only `param` references and never a raw value; the ENGINE substitutes the
    real value at the moment of action. Without this the model would be asked
    to reason about a placeholder and would type the literal string
    "<param:member_id>" into the field.
    """

    literal: str | None = None
    param: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "ValueRef":
        if (self.literal is None) == (self.param is None):
            raise ValueError("ValueRef needs exactly one of literal or param")
        return self


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
    baseline_tier: int | None = Field(
        default=None,
        description="The locator tier that resolved this step at record time. "
        "Drift is deviation from THIS, not from tier 1 -- a hostile surface may "
        "legitimately resolve at tier 2 from day one, which would make a "
        "fixed 'above tier 1 means drift' rule fire constantly and mean nothing.",
    )

    @model_validator(mode="after")
    def _writes_must_verify(self) -> "Step":
        """Determinism depends on every state-changing step proving it landed.
        Enforced by the schema rather than trusted to the compiler."""
        state_changing = {
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.SELECT,
            ActionType.PRESS,
            ActionType.NAVIGATE,
        }
        if self.action in state_changing and self.post_condition is None:
            raise ValueError(
                f"step {self.id!r}: {self.action.value} is state-changing and "
                "requires a post_condition"
            )
        return self


# --------------------------------------------------------------------------
# The artifact itself.
# --------------------------------------------------------------------------


class AuthRequirement(BaseModel):
    """Authentication is a PRECONDITION, not steps.

    Credentials are the one thing that must never be recorded, so artifacts
    begin post-authentication and a SessionProvider establishes the context
    beforehand. This also supplies the re-auth path when a session-expiry
    signal fires mid-replay.
    """

    required: bool = True
    provider: str = "session_provider"
    scope: str | None = None


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
    verification_mode: Literal["full_replay", "dry_run", "none"] = Field(
        default="none",
        description="full_replay: re-executed end to end with a DIFFERENT input "
        "than discovery used, so a literal baked in place of a parameter fails "
        "verification. dry_run: executed to the last safe step, then remaining "
        "locators resolved without acting -- used for write capabilities, "
        "because verifying 'open a sub-account' by replaying it would create a "
        "second real record.",
    )
    verified_with_inputs: dict[str, str] | None = Field(
        default=None, description="Non-sensitive inputs used for verification."
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

    possible_outcomes: list[OutcomeSpec] = Field(default_factory=list)
    auth: AuthRequirement = Field(default_factory=AuthRequirement)

    risk_class: RiskClass = RiskClass.READ_ONLY
    approval_state: ApprovalState = ApprovalState.DRAFT
    provenance: Provenance
    # NOTE: stability/telemetry deliberately does NOT live here. It mutates on
    # every replay; keeping it on the artifact would churn the hash of a
    # supposedly immutable, versioned, reviewable document without its
    # behaviour changing. See cua.telemetry.

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
            "outputs": {
                o.name: {"type": o.type, "description": o.description}
                for o in self.outputs
            },
            "possible_outcomes": [
                {"code": o.code, "description": o.description}
                for o in self.possible_outcomes
            ],
        }
