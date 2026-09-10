"""Discovery: the one part of the system where a model drives a real app.

    discover(...)  -> Trajectory      (LLM in the loop, once per capability)
    compile_trajectory(...) -> Capability
    verify_and_save(...) -> Path | None   (no LLM; nothing unverified is saved)
"""

from cua.discovery.agent import (
    StopReason,
    Trajectory,
    TrajectoryStep,
    discover,
    parse_value_ref,
    risk_of,
    tool_definitions,
)
from cua.discovery.compile import (
    AppProfile,
    CompileError,
    canonicalize_route,
    compile_trajectory,
    route_regex,
)
from cua.discovery.verify import (
    VerificationReport,
    derive_different_inputs,
    dump_capability_yaml,
    verify,
    verify_and_save,
)

__all__ = [
    "discover",
    "Trajectory",
    "TrajectoryStep",
    "StopReason",
    "tool_definitions",
    "parse_value_ref",
    "risk_of",
    "compile_trajectory",
    "AppProfile",
    "CompileError",
    "canonicalize_route",
    "route_regex",
    "verify",
    "verify_and_save",
    "VerificationReport",
    "derive_different_inputs",
    "dump_capability_yaml",
]
