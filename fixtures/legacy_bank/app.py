"""LEGACY BANK -- a deliberately hostile target application.

This is a stand-in for API-less legacy bank back-office software. It is ugly
ON PURPOSE. Every hostile trait below is a fixture requirement, not sloppiness:

  * <frameset> shell        -- every working page is a banner frame plus a work
                               frame, so automation must traverse frames to
                               reach any control.
  * nested <table> layout   -- positioning is done with tables inside tables;
                               there is no CSS grid/flex anywhere.
  * meaningless names       -- inputs are f_7 / ctl00_x3 / f_21; classes are
                               gridRow_alt. No data-testid, no semantic ids.
  * label adjacency         -- the ONLY reliable handle on a field is the text
                               in the <td> next to it, plus a real <label for>
                               pointing at a machine-generated id. A real (old)
                               enterprise app would plausibly have both, and an
                               accessibility-tree-driven agent can use them.
  * full page reloads       -- server-rendered HTML, zero JavaScript.

Run:
    python -m fixtures.legacy_bank.app
    flask --app fixtures/legacy_bank/app.py run -p 5055

Tenant selection: TENANT=a (default) or TENANT=b, or hit the /t/b/... prefix.
"""

from __future__ import annotations

import os
import random
import time
from urllib.parse import urlencode

from flask import (
    Blueprint, Flask, g, redirect, render_template, request, session, url_for,
)

from .config import (
    FUNDING_SOURCES, INJECT_MODES, MEMBERS, PRODUCT_CODES, SEARCH_FIELDS,
    SUBACCOUNT_FIELDS, TENANTS, UNIVERSAL_MODES,
)

SLOW_SECONDS = 6.0
DEFAULT_TENANT = os.environ.get("TENANT", "a").lower()

bp = Blueprint("bank", __name__)


# --------------------------------------------------------------------------
# tenant + session plumbing
# --------------------------------------------------------------------------

@bp.url_defaults
def _add_tenant_default(endpoint, values):
    """Keep the /t/<x>/ prefix sticky across url_for() calls on that mount."""
    if endpoint.startswith("bank_t.") and "t" not in values:
        values["t"] = getattr(g, "tenant_key", "a")


@bp.url_value_preprocessor
def _pull_tenant(endpoint, values):
    """Resolve the tenant, then REMOVE <t> so views keep clean signatures."""
    key = (values or {}).pop("t", None)
    g.tenant_key = key if key in TENANTS else (
        DEFAULT_TENANT if DEFAULT_TENANT in TENANTS else "a"
    )
    g.cfg = TENANTS[g.tenant_key]


def u(endpoint: str, **kw) -> str:
    """url_for that stays inside whichever mount (bare or /t/<x>) we are on."""
    name = request.blueprint or "bank"
    return url_for(f"{name}.{endpoint}", **kw)


def logged_in() -> bool:
    return bool(session.get("uid"))


def _login_redirect(msg: str | None = None):
    session.pop("uid", None)
    q = {"msg": msg} if msg else {}
    return redirect(u("login") + (("?" + urlencode(q)) if q else ""))


# --------------------------------------------------------------------------
# fault injection
# --------------------------------------------------------------------------
# A mode is armed via ?inject=<mode> on any request, or /admin/inject/<mode>.
# It is stored in the session and FIRES ONCE on the next request for which it
# is relevant, then disarms itself. That "arm once, fire once" behaviour is
# what makes the fixture useful for testing recovery paths deterministically.

@bp.before_request
def _arm_inject_from_query():
    mode = request.args.get("inject")
    if mode is None:
        return
    if mode in ("", "none", "clear", "off"):
        session.pop("inject", None)
    elif mode in INJECT_MODES:
        session["inject"] = mode


def take_inject(applicable: set[str] | None = None) -> str | None:
    """Pop the armed mode if it applies to this step."""
    mode = session.get("inject")
    if not mode:
        return None
    if mode in UNIVERSAL_MODES or (applicable and mode in applicable):
        session.pop("inject", None)
        return mode
    return None


def universal_response(mode: str | None):
    """Handle the modes that can fire on any page. Returns a response or None."""
    if mode == "timeout":
        return _login_redirect(g.cfg["messages"]["timeout"])
    if mode == "error":
        # Raw 500-style app error page, the kind a 1998 app leaks verbatim.
        return render_template("error500.html", cfg=g.cfg), 500
    if mode == "dialog":
        # Interstitial that must be dismissed before the real page renders.
        # Continue re-requests the same PATH (never the query string, or the
        # ?inject= that armed this would re-arm it and loop forever).
        return render_template("dialog.html", cfg=g.cfg, back=request.path)
    if mode == "slow":
        time.sleep(SLOW_SECONDS)
    return None


# --------------------------------------------------------------------------
# frameset shells
# --------------------------------------------------------------------------
# Each user-facing URL renders a <frameset>: a banner frame and a work frame.
# The work frame loads the /f/... twin of the same route. Nothing useful is in
# the top document -- automation has to descend into the work frame.

def shell(work_path: str, title: str):
    return render_template(
        "frameset.html", cfg=g.cfg, work=work_path,
        banner=u("f_banner"), title=title,
    )


# --------------------------------------------------------------------------
# 1. login
# --------------------------------------------------------------------------

@bp.route("/", methods=["GET", "POST"])
def login():
    cfg = g.cfg
    if request.method == "POST":
        # Any non-empty credentials work. Demo creds are demo/demo.
        user = (request.form.get("ctl00_u1") or "").strip()
        pwd = (request.form.get("ctl00_p1") or "").strip()
        if not user or not pwd:
            return render_template(
                "login.html", cfg=cfg,
                msg="Operator ID and Password are both required.",
            )
        session["uid"] = user
        return redirect(u("search"))
    return render_template("login.html", cfg=cfg, msg=request.args.get("msg"))


@bp.route("/logoff")
def logoff():
    session.pop("uid", None)
    return redirect(u("login"))


# --------------------------------------------------------------------------
# 2. member search
# --------------------------------------------------------------------------

@bp.route("/search", methods=["GET", "POST"])
def search():
    if not logged_in():
        return _login_redirect()
    cfg = g.cfg

    if request.method == "GET":
        mode = take_inject()
        resp = universal_response(mode)
        if resp:
            return resp
        return shell(u("f_search"), "Member Search")

    # POST: the search form (in the work frame) targets _top so the frameset
    # is rebuilt rather than nested inside itself.
    mode = take_inject({"not_found"})
    resp = universal_response(mode)
    if resp:
        return resp

    mid = (request.form.get(SEARCH_FIELDS["member_id"]) or "").strip()
    if mode == "not_found" or mid not in MEMBERS:
        session["last_query"] = mid
        return shell(u("f_search", r="none"), "Member Search")
    return redirect(u("member", mid=mid))


@bp.route("/f/search", methods=["GET"])
def f_search():
    if not logged_in():
        return _login_redirect()
    return render_template(
        "search.html", cfg=g.cfg, F=SEARCH_FIELDS,
        no_results=(request.args.get("r") == "none"),
        last_query=session.get("last_query", ""),
        post_to=u("search"),
    )


# --------------------------------------------------------------------------
# 3. member detail
# --------------------------------------------------------------------------

@bp.route("/member/<mid>")
def member(mid):
    if not logged_in():
        return _login_redirect()
    mode = take_inject({"perm_denied"})
    resp = universal_response(mode)
    if resp:
        return resp
    if mode == "perm_denied":
        return shell(u("f_member", mid=mid, denied="1"), "Member Detail")
    if mid not in MEMBERS:
        return shell(u("f_search", r="none"), "Member Search")
    return shell(u("f_member", mid=mid), "Member Detail")


@bp.route("/f/member/<mid>")
def f_member(mid):
    if not logged_in():
        return _login_redirect()
    if request.args.get("denied"):
        return render_template("denied.html", cfg=g.cfg, mid=mid,
                               back=u("search"))
    m = MEMBERS.get(mid)
    if not m:
        return render_template("denied.html", cfg=g.cfg, mid=mid,
                               back=u("search"))
    return render_template("member.html", cfg=g.cfg, m=m,
                           new_sub=u("subaccount_new", mid=mid),
                           back=u("search"))


# --------------------------------------------------------------------------
# 4. new sub-account form
# --------------------------------------------------------------------------

@bp.route("/member/<mid>/subaccount/new", methods=["GET", "POST"])
def subaccount_new(mid):
    if not logged_in():
        return _login_redirect()
    cfg = g.cfg

    if request.method == "GET":
        mode = take_inject()
        resp = universal_response(mode)
        if resp:
            return resp
        return shell(u("f_subaccount_new", mid=mid), "Open Sub-Account")

    mode = take_inject({"validation"})
    resp = universal_response(mode)
    if resp:
        return resp

    values = {k: (request.form.get(v) or "").strip()
              for k, v in SUBACCOUNT_FIELDS.items()}

    err_field, err_msg = _validate(values, cfg)
    if mode == "validation":
        # Forced field-level rejection on the deposit amount.
        err_field, err_msg = "initial_deposit", cfg["messages"]["validation"]

    if err_field:
        session["form_err"] = {"field": err_field, "msg": err_msg,
                               "vals": values}
        return shell(u("f_subaccount_new", mid=mid), "Open Sub-Account")

    ref = "%s-%s-%06d" % (cfg["ref_prefix"], time.strftime("%Y%m%d"),
                          random.randint(0, 999999))
    session["last_ref"] = ref
    session["last_sub"] = values
    return redirect(u("subaccount_confirm", mid=mid))


def _validate(values, cfg):
    """Ordinary, non-injected validation. Field-level, one error at a time."""
    if not values["product"]:
        return "product", "Field in error: a product code must be selected."
    amt = values["initial_deposit"].replace(",", "").replace("$", "")
    try:
        if float(amt) < 25.0:
            return "initial_deposit", cfg["messages"]["validation"]
    except ValueError:
        return "initial_deposit", cfg["messages"]["validation"]
    if not values["effective_date"]:
        return "effective_date", "Field in error: an effective date is required (YYYY-MM-DD)."
    return None, None


@bp.route("/f/member/<mid>/subaccount/new")
def f_subaccount_new(mid):
    if not logged_in():
        return _login_redirect()
    m = MEMBERS.get(mid)
    err = session.pop("form_err", None)
    return render_template(
        "subnew.html", cfg=g.cfg, m=m, mid=mid, F=SUBACCOUNT_FIELDS,
        products=PRODUCT_CODES, sources=FUNDING_SOURCES, err=err,
        post_to=u("subaccount_new", mid=mid),
        back=u("member", mid=mid),
    )


# --------------------------------------------------------------------------
# 5. confirmation
# --------------------------------------------------------------------------

@bp.route("/member/<mid>/subaccount/confirm")
def subaccount_confirm(mid):
    if not logged_in():
        return _login_redirect()
    mode = take_inject()
    resp = universal_response(mode)
    if resp:
        return resp
    return shell(u("f_subaccount_confirm", mid=mid), "Request Confirmation")


@bp.route("/f/member/<mid>/subaccount/confirm")
def f_subaccount_confirm(mid):
    if not logged_in():
        return _login_redirect()
    return render_template(
        "confirm.html", cfg=g.cfg, m=MEMBERS.get(mid), mid=mid,
        ref=session.get("last_ref", "(none)"),
        sub=session.get("last_sub", {}),
        products=dict(PRODUCT_CODES), sources=dict(FUNDING_SOURCES),
        back=u("member", mid=mid),
    )


# --------------------------------------------------------------------------
# banner frame + inject admin
# --------------------------------------------------------------------------

@bp.route("/f/banner")
def f_banner():
    return render_template(
        "banner.html", cfg=g.cfg, uid=session.get("uid", "(not signed on)"),
        armed=session.get("inject"), search=u("search"), logoff=u("logoff"),
    )


@bp.route("/admin/inject/<mode>")
def admin_inject(mode):
    if mode in ("clear", "none", "off"):
        session.pop("inject", None)
        state = "cleared"
    elif mode in INJECT_MODES:
        session["inject"] = mode
        state = "armed"
    else:
        return render_template("inject.html", cfg=g.cfg, state="unknown",
                               mode=mode, modes=INJECT_MODES,
                               home=u("search")), 400
    return render_template("inject.html", cfg=g.cfg, state=state, mode=mode,
                           modes=INJECT_MODES, home=u("search"))


# --------------------------------------------------------------------------
# app factory
# --------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    # Fixture-only secret: this app holds nothing but synthetic data.
    app.secret_key = os.environ.get("LEGACY_BANK_SECRET", "legacy-bank-fixture")
    app.jinja_env.trim_blocks = False
    # Same blueprint mounted three times: bare (env-selected tenant) plus an
    # explicit /t/a and /t/b prefix, so both tenants are reachable at once.
    app.register_blueprint(bp)
    app.register_blueprint(bp, name="bank_t", url_prefix="/t/<any(a,b):t>")
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5055")),
            debug=False, threaded=True)
