"""Redaction: sensitive values must not be able to reach disk.

Two layers, deliberately:

1. Value-based. The capability declares which of its inputs are PII/SECRET, so
   at bind time we know the exact strings that must never be written. This is
   exact and complete for the values we were handed.
2. Pattern-based defence in depth. The screen may show a SSN we never supplied,
   an error message may echo a card number, a log line may contain
   `password=hunter2`. Patterns catch what binding cannot.

Layer 2 exists because layer 1 only knows what it was told. Neither layer is
sufficient alone and the combination is still not a guarantee -- see the
limitations note at the bottom of this module.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Iterable, Sequence

from cua.artifact import Capability, ParamSpec, Sensitivity

SENSITIVE = (Sensitivity.PII, Sensitivity.SECRET)

# Minimum length of a bound value we will string-replace. Replacing a bound
# value of "1" or "NY" would shred every artifact into placeholder confetti and
# destroy the evidence we are trying to keep; short values are left to the
# pattern layer. This is a real, documented hole, not an oversight.
MIN_VALUE_LEN = 4

_CRED_KEYS = r"(?:password|passwd|pwd|token|api[_\-]?key|secret|authorization|auth|bearer|session[_\-]?id|cookie)"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # credential key = value  /  "token": "..."  /  Authorization: Bearer x
    (
        "credential",
        re.compile(
            rf'(?i)\b{_CRED_KEYS}\b\s*["\']?\s*[:=]\s*["\']?\s*(?:bearer\s+)?[^\s,;"\'&}}\]]+'
        ),
    ),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}")),
    # SSN: 123-45-6789 or 123 45 6789
    ("ssn", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")),
    # Card numbers: 13-19 digits, optionally grouped by spaces or dashes.
    (
        "card",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    ),
    # Long bare digit runs that look like account numbers.
    ("account", re.compile(r"\b\d{9,}\b")),
]

_CRED_REPLACEMENTS = {"credential", "bearer"}


class Redactor:
    """Built from a capability's declared inputs plus the bound argument values.

    Construct with `Redactor.for_capability(capability, args)` in the normal
    flow; the bare constructor is for tests and for callers that already know
    which values are sensitive.
    """

    def __init__(
        self,
        secrets: dict[str, str] | None = None,
        *,
        enable_patterns: bool = True,
    ) -> None:
        # name -> value, only for params marked PII/SECRET.
        self._secrets: dict[str, str] = {}
        self.enable_patterns = enable_patterns
        for name, value in (secrets or {}).items():
            self.add(name, value)

    # -- construction ------------------------------------------------------

    @classmethod
    def for_capability(
        cls, capability: Capability, args: dict[str, object]
    ) -> "Redactor":
        return cls.from_params(capability.inputs, args)

    @classmethod
    def from_params(
        cls, params: Iterable[ParamSpec], args: dict[str, object]
    ) -> "Redactor":
        r = cls()
        for p in params:
            if p.sensitivity in SENSITIVE and p.name in args:
                r.add(p.name, args[p.name])
        return r

    def add(self, name: str, value: object) -> None:
        """Register one more value to scrub. Idempotent."""
        if value is None:
            return
        text = str(value)
        if len(text) < MIN_VALUE_LEN:
            # Too short to replace safely; the pattern layer is the only
            # defence for these. See MIN_VALUE_LEN.
            return
        self._secrets[name] = text

    @property
    def bound_names(self) -> list[str]:
        return sorted(self._secrets)

    # -- text --------------------------------------------------------------

    def redact(self, text: str) -> str:
        if not text:
            return text
        out = text
        # Longest values first, so a value that contains another is replaced
        # whole rather than being partially clobbered.
        for name, value in sorted(
            self._secrets.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            if value in out:
                out = out.replace(value, f"<param:{name}>")
        if self.enable_patterns:
            out = self._redact_patterns(out)
        return out

    def _redact_patterns(self, text: str) -> str:
        out = text
        for label, pattern in _PATTERNS:
            if label in _CRED_REPLACEMENTS:
                out = pattern.sub(
                    lambda m: _mask_credential(m.group(0)), out
                )
            else:
                out = pattern.sub(f"<redacted:{label}>", out)
        return out

    def redact_obj(self, obj):
        """Recursively redact strings inside dicts/lists/tuples for JSON writes.

        Dict KEYS whose name looks like a credential have their values dropped
        entirely rather than pattern-matched, since a structured record makes
        the intent unambiguous.
        """
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, str) and re.fullmatch(
                    rf"(?i){_CRED_KEYS}", k.strip()
                ):
                    out[k] = "<redacted:credential>"
                else:
                    out[k] = self.redact_obj(v)
            return out
        if isinstance(obj, (list, tuple)):
            return [self.redact_obj(v) for v in obj]
        return obj

    # -- images ------------------------------------------------------------

    def mask_bytes(
        self, image_bytes: bytes, boxes: Sequence[Sequence[float]]
    ) -> bytes:
        """Black out normalized boxes and return NEW bytes.

        Bytes in, bytes out, on purpose: the caller can mask before anything
        touches disk, so an unmasked screenshot never exists as a file that a
        crash, a backup agent, or an `ls` could pick up.

        FAIL CLOSED: if Pillow is unavailable we raise rather than return the
        original bytes. Refusing to produce a screenshot is strictly better than
        producing an unmasked one -- a missing artifact is an inconvenience, a
        leaked one is an incident.
        """
        if not boxes:
            return image_bytes
        try:
            from PIL import Image, ImageDraw  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RedactionUnavailable(
                "Pillow is not installed, so screenshot regions cannot be "
                "masked; refusing to emit an unmasked screenshot."
            ) from exc

        with Image.open(io.BytesIO(image_bytes)) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            w, h = im.size
            for box in boxes:
                x, y, bw, bh = (float(v) for v in box)
                x0, y0 = int(x * w), int(y * h)
                x1, y1 = int((x + bw) * w), int((y + bh) * h)
                draw.rectangle(
                    [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                    fill=(0, 0, 0),
                )
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()

    def mask_regions(
        self,
        image_path: str,
        boxes: Sequence[Sequence[float]],
        *,
        out_path: str | None = None,
    ) -> str:
        """Path-based convenience wrapper over `mask_bytes`.

        Prefer `mask_bytes`: by the time a path exists, the unmasked pixels have
        already been on disk. This exists for surfaces whose capture API can
        only write a file.
        """
        from pathlib import Path  # noqa: PLC0415

        src = Path(image_path)
        data = src.read_bytes()
        try:
            masked = self.mask_bytes(data, boxes)
        except RedactionUnavailable:
            # Fail closed: destroy the unmasked file we were handed.
            src.unlink(missing_ok=True)
            raise
        dest = Path(out_path) if out_path else src
        dest.write_bytes(masked)
        return str(dest)


class RedactionUnavailable(RuntimeError):
    """Raised when redaction cannot be performed and the write must not happen."""


class RedactingFilter(logging.Filter):
    """Last line of defence on the logging path.

    Attach to every handler, not every logger: handlers are where bytes leave
    the process. Mutates the record's message and args in place because
    downstream formatters are not under our control.
    """

    def __init__(self, redactor: Redactor, name: str = "") -> None:
        super().__init__(name)
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - broken %-format in caller
            message = str(record.msg)
        record.msg = self.redactor.redact(message)
        record.args = ()
        if record.exc_text:
            record.exc_text = self.redactor.redact(record.exc_text)
        return True


def _mask_credential(fragment: str) -> str:
    """Keep the key name, drop the value: `password=x` -> `password=<redacted>`."""
    m = re.match(r"(?i)^(\s*bearer\s+)", fragment)
    if m:
        return "bearer <redacted:credential>"
    m = re.match(r'(?i)^(\s*["\']?\s*[\w\-]+\s*["\']?\s*[:=]\s*)', fragment)
    if m:
        return f"{m.group(1)}<redacted:credential>"
    return "<redacted:credential>"


__all__ = [
    "Redactor",
    "RedactingFilter",
    "RedactionUnavailable",
    "MIN_VALUE_LEN",
]
