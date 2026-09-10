"""Establish an authenticated session BEFORE a capability runs.

Credentials are the one thing that must never be recorded, so artifacts begin
post-authentication and this module gets the surface there. That is why
`Capability.auth` is an `AuthRequirement` and not a list of steps: a recorded
login would put a password in a reviewable, version-controlled document.

Real deployments would source credentials from a vault and support SSO; this
reads them from the environment, which is the same seam with a smaller
implementation behind it.
"""

from __future__ import annotations

import os

from cua.artifact import ActionType, LocatorBundle, SemanticLocator
from cua.surface.base import Action, Surface


class AuthError(RuntimeError):
    pass


def _field(role: str, name: str) -> LocatorBundle:
    return LocatorBundle(
        description=f"the {name} field",
        strategies=[SemanticLocator(role=role, name=name)],
    )


def authenticate(
    surface: Surface,
    *,
    user_field: str = "Operator ID",
    pass_field: str = "Password",
    submit: str = "Sign On",
    ready_text: str = "Search Criteria",
    user_env: str = "CUA_APP_USER",
    pass_env: str = "CUA_APP_PASS",
) -> bool:
    """Log in if the surface is showing a login screen.

    Returns True if a session is established. Idempotent: if the surface is
    already past the login screen, this is a no-op, which is what makes it
    usable both up front and as the `reauth` recovery handler when a
    session-expiry signal fires mid-replay.
    """
    obs = surface.observe()
    if ready_text.casefold() in (obs.text_digest or "").casefold():
        return True

    user = os.environ.get(user_env)
    secret = os.environ.get(pass_env)
    if not user or not secret:
        raise AuthError(
            f"{user_env}/{pass_env} not set. Credentials are supplied at run "
            "time, never recorded into an artifact."
        )

    for bundle, value in ((_field("textbox", user_field), user),
                          (_field("textbox", pass_field), secret)):
        res = surface.act(Action(type=ActionType.TYPE, target=bundle, value=value))
        if not res.ok:
            raise AuthError(f"could not fill {bundle.description}: {res.detail}")

    res = surface.act(
        Action(
            type=ActionType.CLICK,
            target=LocatorBundle(
                description=f"the {submit} button",
                strategies=[SemanticLocator(role="button", name=submit)],
            ),
        )
    )
    if not res.ok:
        raise AuthError(f"could not submit the login form: {res.detail}")

    after = surface.observe()
    ok = ready_text.casefold() in (after.text_digest or "").casefold()
    if not ok:
        raise AuthError(
            f"login did not reach the expected post-auth screen "
            f"(looking for {ready_text!r}); observed url={after.url!r}"
        )
    return True
