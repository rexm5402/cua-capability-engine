"""`build_surface` -- the one place that knows concrete surfaces exist.

Upstream (discovery, replay, session) imports `Surface` from `base` and calls
this factory.  Nothing upstream imports Playwright, so the desktop story stays
a matter of adding a `kind` here rather than editing engines.
"""

from __future__ import annotations

from typing import Any

from cua.surface.base import Surface

_KINDS = ("web", "recorded")


def build_surface(kind: str, **cfg: Any) -> Surface:
    """Return a concrete `Surface`.

    web:
        ``build_surface("web", page=page, app_id=..., tenant_id=None,
                        config=WebSurfaceConfig(...))``
        Playwright is imported lazily, inside this branch only -- a hermetic
        test run never touches it.  To boot a browser yourself use
        ``WebSurface.launch(...)`` as a context manager.

    recorded:
        ``build_surface("recorded", frames=[...])`` or
        ``build_surface("recorded", directory="fixtures/run1")``
    """
    k = (kind or "").strip().lower()

    if k == "recorded":
        from cua.surface.recorded import RecordedSurface  # noqa: PLC0415

        directory = cfg.pop("directory", None)
        if directory is not None:
            return RecordedSurface.from_dir(directory, **cfg)
        return RecordedSurface(**cfg)

    if k == "web":
        from cua.surface.web import WebSurface, WebSurfaceConfig  # noqa: PLC0415

        if "config" in cfg and isinstance(cfg["config"], dict):
            cfg["config"] = WebSurfaceConfig(**cfg["config"])
        return WebSurface(**cfg)

    raise ValueError(f"unknown surface kind {kind!r}; expected one of {_KINDS}")
