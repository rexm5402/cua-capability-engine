"""Per-run evidence: what happened, in a form an auditor can read cold.

Layout:

    evidence/runs/<run_id>/
        manifest.json     run identity, timing, environment, final outcome
        steps.jsonl       one JSON object per step, append-only
        run.log           free-text notes and attached-file records
        screenshots/      captured frames; on failure, a richer snapshot

Every write goes through the attached `Redactor` first. There is one private
`_write` path and it always redacts, so "remembering to redact" is not a thing
a caller can get wrong.

Deliberately git-less: this records the environment (python, platform, host)
rather than shelling out to git, because the runtime that replays a capability
is not necessarily a checkout.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Sequence

from cua.policy.redaction import Redactor, RedactionUnavailable

RunKind = Literal["discovery", "replay"]

DEFAULT_ROOT = Path("evidence") / "runs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion; pydantic models and dataclasses included."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {
            f: _jsonable(getattr(obj, f)) for f in obj.__dataclass_fields__
        }
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum
        return _jsonable(obj.value)
    return str(obj)


class EvidenceRecorder:
    """Usable as a context manager; `finish` is idempotent."""

    def __init__(
        self,
        run_id: str,
        kind: RunKind,
        *,
        root: str | Path = DEFAULT_ROOT,
        capability_id: str | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.run_id = run_id
        self.kind = kind
        self.capability_id = capability_id
        self.redactor = redactor

        self.dir = Path(root) / run_id
        self.screenshots_dir = self.dir / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.dir / "manifest.json"
        self.steps_path = self.dir / "steps.jsonl"
        self.log_path = self.dir / "run.log"

        self.started_at = _now()
        self.ended_at: str | None = None
        self._step_count = 0
        self._finished = False
        self._attachments: list[str] = []

        self.steps_path.touch()
        self.log_path.touch()
        self._write_manifest(outcome=None)

    # -- redaction chokepoint ---------------------------------------------

    def _scrub(self, obj: Any) -> Any:
        if self.redactor is None:
            return obj
        return self.redactor.redact_obj(obj)

    def _scrub_text(self, text: str) -> str:
        if self.redactor is None:
            return text
        return self.redactor.redact(text)

    def attach_redactor(self, redactor: Redactor) -> None:
        self.redactor = redactor

    # -- writes ------------------------------------------------------------

    def _write_manifest(self, *, outcome: Any) -> None:
        manifest = {
            "run_id": self.run_id,
            "kind": self.kind,
            "capability_id": self.capability_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "steps": self._step_count,
            "final_outcome": _jsonable(outcome),
            "attachments": list(self._attachments),
            "environment": _environment(),
            "redaction": {
                "enabled": self.redactor is not None,
                "bound_params": (
                    self.redactor.bound_names if self.redactor else []
                ),
            },
        }
        payload = json.dumps(self._scrub(manifest), indent=2, sort_keys=False)
        self.manifest_path.write_text(payload + "\n")

    def step(
        self,
        *,
        step_id: str,
        intent: str = "",
        action: Any = None,
        resolved_tier: int | None = None,
        candidates: int | None = None,
        duration_ms: float | None = None,
        decision: Any = None,
        outcome: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        self._step_count += 1
        record = {
            "seq": self._step_count,
            "ts": _now(),
            "step_id": step_id,
            "intent": intent,
            "action": _jsonable(action),
            "resolved_tier": resolved_tier,
            "candidates": candidates,
            "duration_ms": duration_ms,
            "decision": _jsonable(decision),
            "outcome": _jsonable(outcome),
        }
        if extra:
            record["extra"] = _jsonable(extra)
        scrubbed = self._scrub(record)
        with self.steps_path.open("a") as fh:
            fh.write(json.dumps(scrubbed) + "\n")
        return scrubbed

    def note(self, message: str, level: str = "INFO") -> None:
        line = f"{_now()} {level:<7} {self._scrub_text(message)}"
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")

    def attach(self, path: str | Path, *, label: str | None = None) -> str:
        """Record an already-existing artifact file (screenshot, trace, HAR).

        Copies into the run directory so the evidence bundle is self-contained.
        The file's CONTENT is not redacted -- see `screenshot` for the path that
        masks pixels before they reach disk.
        """
        import shutil  # noqa: PLC0415

        src = Path(path)
        dest_dir = (
            self.screenshots_dir
            if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            else self.dir
        )
        dest = dest_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        rel = str(dest.relative_to(self.dir))
        self._attachments.append(rel)
        self.note(f"attached {rel}" + (f" ({label})" if label else ""))
        return str(dest)

    def screenshot(
        self,
        name: str,
        image_bytes: bytes,
        *,
        mask_boxes: Sequence[Sequence[float]] = (),
    ) -> str:
        """Mask THEN write. Unmasked pixels never become a file.

        If masking is requested and cannot be performed, nothing is written and
        the exception propagates -- failing closed.
        """
        if mask_boxes:
            if self.redactor is None:
                raise RedactionUnavailable(
                    "mask_boxes were requested but no Redactor is attached; "
                    "refusing to write a potentially unmasked screenshot."
                )
            image_bytes = self.redactor.mask_bytes(image_bytes, mask_boxes)
        dest = self.screenshots_dir / (name if name.endswith(".png") else f"{name}.png")
        dest.write_bytes(image_bytes)
        rel = str(dest.relative_to(self.dir))
        self._attachments.append(rel)
        self.note(f"screenshot {rel} (masked_regions={len(mask_boxes)})")
        return str(dest)

    def finish(self, outcome: Any = None) -> Path:
        if self._finished:
            return self.dir
        self.ended_at = _now()
        self._finished = True
        self._write_manifest(outcome=outcome)
        self.note(f"run finished: {type(outcome).__name__ if outcome else 'none'}")
        return self.dir

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "EvidenceRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        if exc is not None:
            self.note(f"uncaught {exc_type.__name__}: {exc}", level="ERROR")
            if not self._finished:
                self.finish({"kind": "failure", "code": "uncaught_exception"})
        elif not self._finished:
            self.finish(None)
        return False


__all__ = ["EvidenceRecorder", "RunKind", "DEFAULT_ROOT"]
