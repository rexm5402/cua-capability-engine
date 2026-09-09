"""Evidence recorder. No browser, no network."""

from __future__ import annotations

import json
from io import BytesIO

import pytest

from cua.artifact import ActionType, ParamSpec, RiskClass, Sensitivity
from cua.evidence.recorder import EvidenceRecorder
from cua.outcomes import Failure, Success
from cua.policy.gate import Policy, PolicyGate
from cua.policy.redaction import Redactor, RedactionUnavailable
from cua.surface.base import Action

SSN = "078-05-1120"
KEY = "sk-liveAAAABBBBCCCC"


@pytest.fixture
def redactor() -> Redactor:
    return Redactor.from_params(
        [
            ParamSpec(
                name="ssn",
                type="string",
                description="ssn",
                sensitivity=Sensitivity.PII,
            ),
            ParamSpec(
                name="api_key",
                type="string",
                description="key",
                sensitivity=Sensitivity.SECRET,
            ),
        ],
        {"ssn": SSN, "api_key": KEY},
    )


def _png(color=(255, 0, 0), size=(40, 40)) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------


def test_recorder_creates_the_expected_layout(tmp_path):
    with EvidenceRecorder("run-1", "replay", root=tmp_path) as rec:
        rec.note("hello")
    d = tmp_path / "run-1"
    assert (d / "manifest.json").is_file()
    assert (d / "steps.jsonl").is_file()
    assert (d / "run.log").is_file()
    assert (d / "screenshots").is_dir()


def test_manifest_is_well_formed(tmp_path):
    with EvidenceRecorder(
        "run-2", "discovery", root=tmp_path, capability_id="cap.lookup"
    ) as rec:
        rec.step(step_id="s1", intent="open the search page")
        rec.finish(Success(capability_id="cap.lookup", steps_executed=1))

    m = json.loads((tmp_path / "run-2" / "manifest.json").read_text())
    assert m["run_id"] == "run-2"
    assert m["kind"] == "discovery"
    assert m["capability_id"] == "cap.lookup"
    assert m["started_at"] and m["ended_at"]
    assert m["steps"] == 1
    assert m["final_outcome"]["kind"] == "success"
    assert m["environment"]["python"]
    assert "hostname" in m["environment"]


def test_steps_jsonl_is_one_valid_object_per_line(tmp_path):
    gate = PolicyGate(Policy.default())
    action = Action(type=ActionType.CLICK)
    decision = gate.check(
        action, current_url="http://localhost:5055/search", risk=RiskClass.READ_ONLY
    )
    with EvidenceRecorder("run-3", "replay", root=tmp_path) as rec:
        for i in range(3):
            rec.step(
                step_id=f"s{i}",
                intent="click search",
                action=action,
                resolved_tier=1,
                candidates=1,
                duration_ms=12.5,
                decision=decision,
                outcome="ok",
            )

    lines = (tmp_path / "run-3" / "steps.jsonl").read_text().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert records[0]["action"]["type"] == "click"
    assert records[0]["decision"]["kind"] == "allow"
    assert records[0]["resolved_tier"] == 1


def test_recorder_redacts_on_write(tmp_path, redactor):
    with EvidenceRecorder(
        "run-4", "replay", root=tmp_path, redactor=redactor
    ) as rec:
        rec.step(
            step_id="s1",
            intent=f"type the member ssn {SSN}",
            action=Action(type=ActionType.TYPE, value=SSN),
            outcome={"echoed": f"key={KEY}"},
        )
        rec.note(f"used api key {KEY}")
        rec.finish(
            Failure(
                capability_id="cap.x",
                code="hard_timeout",
                observed=f"page showed {SSN}",
            )
        )

    d = tmp_path / "run-4"
    blob = "\n".join(
        p.read_text() for p in (d / "manifest.json", d / "steps.jsonl", d / "run.log")
    )
    assert SSN not in blob
    assert KEY not in blob
    assert "<param:ssn>" in blob
    assert "<param:api_key>" in blob


def test_unbound_sensitive_pattern_is_still_scrubbed_on_write(tmp_path, redactor):
    with EvidenceRecorder(
        "run-5", "replay", root=tmp_path, redactor=redactor
    ) as rec:
        rec.note("screen text: SSN 123-45-6789")
    assert "123-45-6789" not in (tmp_path / "run-5" / "run.log").read_text()


def test_screenshot_is_masked_before_it_lands_on_disk(tmp_path, redactor):
    from PIL import Image

    with EvidenceRecorder(
        "run-6", "replay", root=tmp_path, redactor=redactor
    ) as rec:
        path = rec.screenshot("step1", _png(), mask_boxes=[(0.0, 0.0, 0.5, 0.5)])

    im = Image.open(path)
    assert im.getpixel((5, 5)) == (0, 0, 0)
    assert im.getpixel((35, 35)) == (255, 0, 0)


def test_screenshot_with_mask_but_no_redactor_refuses_to_write(tmp_path):
    with EvidenceRecorder("run-7", "replay", root=tmp_path) as rec:
        with pytest.raises(RedactionUnavailable):
            rec.screenshot("s", _png(), mask_boxes=[(0.0, 0.0, 0.5, 0.5)])
        assert not list((tmp_path / "run-7" / "screenshots").iterdir())


def test_attach_copies_the_file_into_the_run_dir(tmp_path):
    src = tmp_path / "trace.png"
    src.write_bytes(_png())
    with EvidenceRecorder("run-8", "replay", root=tmp_path) as rec:
        rec.attach(src, label="failure snapshot")
    d = tmp_path / "run-8"
    assert (d / "screenshots" / "trace.png").is_file()
    m = json.loads((d / "manifest.json").read_text())
    assert "screenshots/trace.png" in m["attachments"]


def test_context_manager_records_an_uncaught_exception(tmp_path):
    with pytest.raises(ValueError):
        with EvidenceRecorder("run-9", "replay", root=tmp_path):
            raise ValueError("boom")

    d = tmp_path / "run-9"
    m = json.loads((d / "manifest.json").read_text())
    assert m["ended_at"] is not None
    assert m["final_outcome"]["code"] == "uncaught_exception"
    assert "boom" in (d / "run.log").read_text()


def test_finish_is_idempotent(tmp_path):
    rec = EvidenceRecorder("run-10", "replay", root=tmp_path)
    rec.finish(Success(capability_id="c"))
    first = json.loads((tmp_path / "run-10" / "manifest.json").read_text())["ended_at"]
    rec.finish(Failure(capability_id="c", code="late"))
    second = json.loads((tmp_path / "run-10" / "manifest.json").read_text())
    assert second["ended_at"] == first
    assert second["final_outcome"]["kind"] == "success"
