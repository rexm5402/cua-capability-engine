"""Policy gate + redaction. No browser, no network."""

from __future__ import annotations

import logging

import pytest

from cua.artifact import ActionType, ParamSpec, RiskClass, Sensitivity
from cua.policy.gate import (
    ACTION_TYPE_NOT_ALLOWED,
    IRREVERSIBLE_UNATTENDED,
    NO_POLICY,
    ORIGIN_NOT_ALLOWED,
    ROUTE_NOT_ALLOWED,
    Allow,
    Deny,
    Policy,
    PolicyGate,
    RequireApproval,
)
from cua.policy.redaction import (
    RedactingFilter,
    RedactionUnavailable,
    Redactor,
)
from cua.surface.base import Action

LOCAL = "http://localhost:5055"


@pytest.fixture
def default_policy() -> Policy:
    return Policy.default()


@pytest.fixture
def gate(default_policy: Policy) -> PolicyGate:
    return PolicyGate(default_policy)


# --------------------------------------------------------------------------
# Allowlist
# --------------------------------------------------------------------------


def test_default_policy_loads_and_is_scoped_to_the_fixture_app(default_policy):
    assert default_policy.allowed_origins == [LOCAL]
    assert default_policy.mode == "unattended"


def test_allowlisted_read_only_action_is_allowed(gate):
    d = gate.check(
        Action(type=ActionType.CLICK),
        current_url=f"{LOCAL}/search",
        risk=RiskClass.READ_ONLY,
    )
    assert isinstance(d, Allow)


def test_non_allowlisted_origin_is_denied(gate):
    d = gate.check(
        Action(type=ActionType.CLICK),
        current_url="https://evil.example.com/search",
        risk=RiskClass.READ_ONLY,
    )
    assert isinstance(d, Deny)
    assert d.code == ORIGIN_NOT_ALLOWED


def test_navigate_off_the_allowlist_from_an_allowed_page_is_denied(gate):
    """An allowlisted page must not be a springboard off the allowlist."""
    d = gate.check(
        Action(type=ActionType.NAVIGATE, url="https://evil.example.com/x"),
        current_url=f"{LOCAL}/search",
        risk=RiskClass.READ_ONLY,
    )
    assert isinstance(d, Deny)
    assert d.code == ORIGIN_NOT_ALLOWED


def test_non_allowlisted_route_on_an_allowed_origin_is_denied(gate):
    d = gate.check(
        Action(type=ActionType.CLICK),
        current_url=f"{LOCAL}/admin/delete-everything",
        risk=RiskClass.READ_ONLY,
    )
    assert isinstance(d, Deny)
    assert d.code == ROUTE_NOT_ALLOWED


def test_disallowed_action_type_is_denied(default_policy):
    policy = default_policy.model_copy(
        update={
            "allowed_action_types": [ActionType.READ, ActionType.WAIT],
        }
    )
    d = PolicyGate(policy).check(
        Action(type=ActionType.CLICK),
        current_url=f"{LOCAL}/search",
        risk=RiskClass.READ_ONLY,
    )
    assert isinstance(d, Deny)
    assert d.code == ACTION_TYPE_NOT_ALLOWED


def test_malformed_url_is_denied(gate):
    d = gate.check(
        Action(type=ActionType.CLICK),
        current_url="not-a-url",
        risk=RiskClass.READ_ONLY,
    )
    assert isinstance(d, Deny)


# --------------------------------------------------------------------------
# Deny by default
# --------------------------------------------------------------------------


@pytest.mark.parametrize("risk", list(RiskClass))
@pytest.mark.parametrize("action_type", list(ActionType))
def test_empty_policy_denies_everything(risk, action_type):
    d = PolicyGate(Policy.empty()).check(
        Action(type=action_type, url=f"{LOCAL}/"),
        current_url=f"{LOCAL}/",
        risk=risk,
    )
    assert isinstance(d, Deny)
    assert d.code == NO_POLICY


def test_gate_with_no_policy_argument_denies_everything():
    """`PolicyGate()` must not mean 'unrestricted'."""
    d = PolicyGate().check(
        Action(type=ActionType.READ),
        current_url=f"{LOCAL}/",
        risk=RiskClass.READ_ONLY,
    )
    assert isinstance(d, Deny)


# --------------------------------------------------------------------------
# Risk classes
# --------------------------------------------------------------------------


def test_reversible_action_is_allowed_when_allowlisted(gate):
    d = gate.check(
        Action(type=ActionType.TYPE, value="x"),
        current_url=f"{LOCAL}/members/42",
        risk=RiskClass.REVERSIBLE,
    )
    assert isinstance(d, Allow)


def test_irreversible_is_denied_in_unattended_mode(gate):
    d = gate.check(
        Action(type=ActionType.CLICK),
        current_url=f"{LOCAL}/claims/9",
        risk=RiskClass.IRREVERSIBLE,
    )
    assert isinstance(d, Deny)
    assert d.code == IRREVERSIBLE_UNATTENDED


def test_irreversible_requires_approval_in_attended_mode(default_policy):
    policy = default_policy.model_copy(update={"mode": "attended"})
    d = PolicyGate(policy).check(
        Action(type=ActionType.CLICK),
        current_url=f"{LOCAL}/claims/9",
        risk=RiskClass.IRREVERSIBLE,
    )
    assert isinstance(d, RequireApproval)


def test_irreversible_off_allowlist_is_denied_even_in_attended_mode(default_policy):
    """The allowlist is checked before risk: no approval prompt for evil.com."""
    policy = default_policy.model_copy(update={"mode": "attended"})
    d = PolicyGate(policy).check(
        Action(type=ActionType.CLICK),
        current_url="https://evil.example.com/",
        risk=RiskClass.IRREVERSIBLE,
    )
    assert isinstance(d, Deny)
    assert d.code == ORIGIN_NOT_ALLOWED


# --------------------------------------------------------------------------
# Redaction: bound values
# --------------------------------------------------------------------------


PARAMS = [
    ParamSpec(name="member_id", type="string", description="id"),
    ParamSpec(
        name="ssn",
        type="string",
        description="member ssn",
        sensitivity=Sensitivity.PII,
    ),
    ParamSpec(
        name="api_key",
        type="string",
        description="portal key",
        sensitivity=Sensitivity.SECRET,
    ),
]


@pytest.fixture
def redactor() -> Redactor:
    return Redactor.from_params(
        PARAMS,
        {"member_id": "M-1001", "ssn": "078-05-1120", "api_key": "sk-liveAAAABBBBCCCC"},
    )


def test_bound_pii_value_is_replaced_with_a_placeholder(redactor):
    out = redactor.redact("looking up member with ssn 078-05-1120 now")
    assert "078-05-1120" not in out
    assert "<param:ssn>" in out


def test_bound_secret_is_removed_from_a_would_be_artifact_string(redactor):
    artifact = '{"headers": {"x-key": "sk-liveAAAABBBBCCCC"}}'
    out = redactor.redact(artifact)
    assert "sk-liveAAAABBBBCCCC" not in out
    assert "<param:api_key>" in out


def test_non_sensitive_bound_value_is_preserved(redactor):
    """Redaction must not shred the evidence it exists to protect."""
    assert "M-1001" in redactor.redact("member M-1001 found")


def test_redact_obj_walks_nested_structures(redactor):
    out = redactor.redact_obj(
        {"a": ["ssn is 078-05-1120"], "password": "hunter2xyz"}
    )
    assert "078-05-1120" not in str(out)
    assert out["password"] == "<redacted:credential>"


# --------------------------------------------------------------------------
# Redaction: patterns (defence in depth, nothing bound)
# --------------------------------------------------------------------------


def test_pattern_catches_ssn_with_no_bound_param():
    out = Redactor().redact("screen showed 123-45-6789 in the SSN field")
    assert "123-45-6789" not in out
    assert "<redacted:ssn>" in out


def test_pattern_catches_card_number_with_no_bound_param():
    out = Redactor().redact("card 4111 1111 1111 1111 on file")
    assert "4111" not in out
    assert "<redacted:" in out


def test_pattern_catches_long_account_number():
    out = Redactor().redact("acct 000123456789")
    assert "000123456789" not in out


@pytest.mark.parametrize(
    "line",
    [
        "password=hunter2secret",
        'token: "abc123def456"',
        "api_key=AKIAEXAMPLE12345",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9",
    ],
)
def test_pattern_catches_credential_key_names(line):
    out = Redactor().redact(line)
    assert "<redacted:credential>" in out
    for token in ("hunter2secret", "abc123def456", "AKIAEXAMPLE12345", "eyJhbGciOiJIUzI1NiJ9"):
        assert token not in out


# --------------------------------------------------------------------------
# Logging filter
# --------------------------------------------------------------------------


def test_log_filter_scrubs_a_record_from_a_real_logger(redactor, caplog):
    stream_records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            stream_records.append(self.format(record))

    logger = logging.getLogger("cua.test.redaction")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = Capture()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(RedactingFilter(redactor))
    logger.handlers = [handler]

    logger.info("submitting ssn=%s with key %s", "078-05-1120", "sk-liveAAAABBBBCCCC")
    logger.warning("raw ssn on screen: 123-45-6789")

    joined = "\n".join(stream_records)
    assert "078-05-1120" not in joined
    assert "sk-liveAAAABBBBCCCC" not in joined
    assert "123-45-6789" not in joined
    assert "<param:ssn>" in joined


# --------------------------------------------------------------------------
# Image masking
# --------------------------------------------------------------------------


def _png_bytes(color=(255, 0, 0), size=(40, 40)) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_mask_bytes_blacks_out_the_requested_region(tmp_path):
    from io import BytesIO

    from PIL import Image

    masked = Redactor().mask_bytes(_png_bytes(), [(0.0, 0.0, 0.5, 0.5)])
    im = Image.open(BytesIO(masked))
    assert im.getpixel((5, 5)) == (0, 0, 0)
    assert im.getpixel((35, 35)) == (255, 0, 0)


def test_mask_fails_closed_when_pillow_is_missing(monkeypatch, tmp_path):
    """No Pillow means NO screenshot, never an unmasked one."""
    import builtins

    # Build the fixtures BEFORE Pillow is blocked.
    png = _png_bytes()
    path = tmp_path / "shot.png"
    path.write_bytes(png)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RedactionUnavailable):
        Redactor().mask_bytes(png, [(0.0, 0.0, 0.5, 0.5)])

    with pytest.raises(RedactionUnavailable):
        Redactor().mask_regions(str(path), [(0.0, 0.0, 0.5, 0.5)])
    assert not path.exists(), "unmasked file must be destroyed, not left behind"
