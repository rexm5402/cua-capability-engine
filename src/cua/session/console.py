"""The operator console: deliberately bare.

This is a server-rendered FastAPI app with three routes and no JavaScript, no
build step, no auth, and no styling worth the name. That is a choice, not an
omission. The contribution of this subsystem is the LEASE and the RESUME RULE;
a prettier console would not make the control-transfer model any more correct,
and a console with its own state would make it less so. Everything here reads
and writes through `RequestStore`, so the console holds no session state at
all and can be killed and restarted mid-intervention.

(It has no authentication. In any real deployment this sits behind the same SSO
the rest of the ops tooling uses, and the operator id below comes from that,
not from a text box.)

HOW THE OPERATOR ACTUALLY TAKES OVER -- honestly
------------------------------------------------
In the headed-browser development setup, "handing over control" is literal: the
engine drives a visible Chrome window on the operator's own machine, the lease
flips to OPERATOR, automation stops touching the page, and the human uses the
same live browser context -- same cookies, same session, same tab. Nothing is
proxied. This works, and it is genuinely useful, but only when the operator is
sitting at the machine running the engine.

That is not production. In production the engine runs headless on a server and
the operator is somewhere else entirely. The real transport there is a CDP
screencast (`Page.startScreencast`) streaming frames to the browser-based
console, plus input forwarding (`Input.dispatchMouseEvent` /
`dispatchKeyEvent`) sending the operator's clicks and keystrokes back into the
remote page. Same idea as any remote-desktop product, scoped to one tab.

The thing to be clear about: that changes HOW pixels and clicks travel. It does
not change WHO HOLDS THE LEASE. The lease is a file (or a row) that both
processes compare-and-swap against; it knows nothing about screencasts. Swap
the transport and every line of `lease.py`, `escalation.py`, and `handoff.py`
is unchanged. That transport-independence is precisely why the lease -- and not
the console -- is the actual contribution here.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from cua.session.escalation import (
    Aborted,
    InterventionRequest,
    MarkedFailed,
    RequestStore,
    Resumed,
)
from cua.session.lease import SessionLease


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _row(req: InterventionRequest) -> str:
    return (
        f"<tr><td><a href='/requests/{_esc(req.request_id)}'>"
        f"{_esc(req.request_id)}</a></td>"
        f"<td>{_esc(req.reason_code.value)}</td>"
        f"<td>{_esc(req.capability_name)}</td>"
        f"<td>{_esc(req.step_id)}</td>"
        f"<td>{_esc(req.step_intent)}</td>"
        f"<td>{_esc(req.created_at)}</td></tr>"
    )


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title>"
        "<style>body{font:14px/1.5 ui-monospace,monospace;margin:2rem;max-width:60rem}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #999;padding:.35rem .5rem;text-align:left;"
        "vertical-align:top}"
        "dt{font-weight:700;margin-top:.6rem}"
        "form{display:inline}button{padding:.4rem .9rem;margin-right:.5rem}"
        "img{max-width:100%;border:1px solid #999}</style>"
        f"{body}"
    )


def create_app(
    store: RequestStore,
    lease: SessionLease | None = None,
    *,
    screenshot_root: str | Path | None = None,
):
    """Build the FastAPI app. Imported lazily so the rest of the session
    package -- and its tests -- do not require FastAPI to be installed."""
    from fastapi import Form, HTTPException  # noqa: PLC0415
    from fastapi import FastAPI  # noqa: PLC0415
    from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse  # noqa: PLC0415

    app = FastAPI(title="cua operator console")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        reqs = store.list_requests(open_only=True)
        lease_line = ""
        if lease is not None:
            rec = lease.read()
            lease_line = (
                f"<p><b>lease</b>: {_esc(rec.state.value)} "
                f"holder={_esc(rec.holder)} fence={rec.fence} "
                f"expired={rec.is_expired()}</p>"
            )
        if not reqs:
            return _page("operator console", lease_line + "<p>No open requests.</p>")
        rows = "".join(_row(r) for r in reqs)
        return _page(
            "operator console",
            lease_line
            + "<table><tr><th>id</th><th>reason</th><th>capability</th>"
            "<th>step</th><th>intent</th><th>created</th></tr>"
            + rows
            + "</table>",
        )

    @app.get("/requests/{request_id}", response_class=HTMLResponse)
    def detail(request_id: str) -> str:
        req = store.get_request(request_id)
        if req is None:
            raise HTTPException(404, "no such request")
        res = store.get_resolution(request_id)
        shot = ""
        if req.screenshot_path:
            shot = (
                f"<p><img src='/requests/{_esc(request_id)}/screenshot' "
                f"alt='latest screenshot'></p>"
            )
        fields = {
            "reason": req.reason_code.value,
            "why": req.why,
            "capability": f"{req.capability_name} ({req.capability_id})",
            "run": req.run_id,
            "step": req.step_id,
            "STEP INTENT": req.step_intent,
            "proposed action": req.proposed_action,
            "risk": req.risk_class,
            "url": req.current_url,
            "observation digest": req.observation_digest,
            "created": req.created_at,
            "deadline": req.deadline,
        }
        dl = "".join(
            f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>"
            for k, v in fields.items()
            if v not in (None, "")
        )
        if res is not None:
            controls = f"<p><b>resolved</b>: {_esc(res.kind)}</p>"
        else:
            allowed = {a.value for a in req.permitted_actions}
            buttons = "".join(
                f"<form method='post' action='/requests/{_esc(request_id)}/{a}'>"
                f"<input type='hidden' name='operator_id' value='operator'>"
                f"<button type='submit'>{a.capitalize()}</button></form>"
                for a in ("resume", "abort", "fail")
                if a in allowed
            )
            controls = f"<p>{buttons}</p>"
        return _page(
            f"request {request_id}",
            f"<p><a href='/'>&larr; open requests</a></p><dl>{dl}</dl>{shot}{controls}",
        )

    @app.get("/requests/{request_id}/screenshot")
    def screenshot(request_id: str):
        req = store.get_request(request_id)
        if req is None or not req.screenshot_path:
            raise HTTPException(404, "no screenshot")
        p = Path(req.screenshot_path)
        if not p.is_absolute() and screenshot_root is not None:
            p = Path(screenshot_root) / p
        if not p.exists():
            raise HTTPException(404, "screenshot file missing")
        return FileResponse(p)

    def _resolve(request_id: str, resolution) -> RedirectResponse:
        if store.get_request(request_id) is None:
            raise HTTPException(404, "no such request")
        if store.get_resolution(request_id) is not None:
            # First resolution wins. Two operators clicking at once must not
            # produce two different answers to the same question.
            raise HTTPException(409, "already resolved")
        store.put_resolution(resolution)
        return RedirectResponse(f"/requests/{request_id}", status_code=303)

    @app.post("/requests/{request_id}/resume")
    def resume(request_id: str, operator_id: str = Form("operator"), note: str = Form("")):
        return _resolve(
            request_id, Resumed(request_id=request_id, operator_id=operator_id, note=note)
        )

    @app.post("/requests/{request_id}/abort")
    def abort(request_id: str, operator_id: str = Form("operator"), note: str = Form("")):
        return _resolve(
            request_id, Aborted(request_id=request_id, operator_id=operator_id, note=note)
        )

    @app.post("/requests/{request_id}/fail")
    def fail(request_id: str, operator_id: str = Form("operator"), note: str = Form("")):
        return _resolve(
            request_id,
            MarkedFailed(request_id=request_id, operator_id=operator_id, note=note),
        )

    return app


__all__ = ["create_app"]
