"""Locator engine: record durable locators, resolve them unique-or-fail."""

from cua.locator.record import build_bundle, describe, find_anchor_candidates
from cua.locator.resolve import Resolution, normalize, resolve

__all__ = [
    "Resolution",
    "resolve",
    "normalize",
    "build_bundle",
    "describe",
    "find_anchor_candidates",
]
