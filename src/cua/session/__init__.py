"""Human-in-the-loop escalation and control transfer.

Three pieces, in dependency order:

* `lease`      -- who holds the live session, externalised so a separate
                  console process can see and contest it.
* `escalation` -- the single queue by which a stuck run or an approval request
                  reaches a person.
* `handoff`    -- pause / hand over / block / reacquire / RE-VERIFY.

`console` (a FastAPI app) is not imported here, so importing this package does
not require FastAPI.
"""

from cua.session.escalation import (
    Aborted,
    EscalationTimeout,
    Escalator,
    HumanIntervention,
    InterventionRequest,
    MarkedFailed,
    PermittedAction,
    ReasonCode,
    RequestStore,
    Resolution,
    Resumed,
)
from cua.session.handoff import HandoffOutcome, diff_observations, handoff
from cua.session.lease import (
    LeaseError,
    LeaseExpired,
    LeaseGrant,
    LeaseHeld,
    LeaseRecord,
    LeaseState,
    LeaseStateMismatch,
    NotHolder,
    SessionLease,
)

__all__ = [
    "SessionLease",
    "LeaseState",
    "LeaseRecord",
    "LeaseGrant",
    "LeaseError",
    "LeaseHeld",
    "LeaseStateMismatch",
    "NotHolder",
    "LeaseExpired",
    "Escalator",
    "RequestStore",
    "InterventionRequest",
    "ReasonCode",
    "PermittedAction",
    "Resolution",
    "Resumed",
    "Aborted",
    "MarkedFailed",
    "HumanIntervention",
    "EscalationTimeout",
    "handoff",
    "HandoffOutcome",
    "diff_observations",
]
