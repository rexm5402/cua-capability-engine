"""The session lease: exactly one holder of the live browser session, ever.

Why this is a file and not an object
------------------------------------
The obvious implementation of "pause automation and let a human drive" is a
boolean on the engine object. That is decorative. The operator console runs in
a DIFFERENT PROCESS (and, in production, on a different machine) from the
replay engine, so an in-memory flag is invisible to the only party whose
exclusion actually matters. The invariant "exactly one holder" is only real if
it is externalised somewhere both processes can see and mutate atomically.

So the lease is a file, and every mutation is a read-modify-write performed
under an `O_EXCL` lock file and committed by writing a temp file and
`os.replace`-ing it over the lease path. `os.replace` is atomic on POSIX and on
Windows, so a reader never observes a half-written lease.

Two tokens make preemption safe:

* `holder_token` -- a fresh uuid minted on every successful acquisition. It
  proves "I am the specific acquisition that currently holds this", not merely
  "I am a process calling myself the engine".
* `fence` -- a monotonically increasing counter, bumped on every state
  transition. A holder that was preempted (TTL expiry, forced transfer) comes
  back with a stale fence and every mutating call it makes is rejected. This is
  the standard fencing-token construction: it does not prevent a zombie from
  *acting* on the surface, but it prevents it from ever re-entering the lease
  and it makes the zombie's staleness detectable at its next checkpoint.

TTL exists so a crashed holder does not deadlock the session forever, and
expiry is EXPLICIT: an expired lease is still reported as held, with
`is_expired() == True`. Somebody must call `expire_if_stale()` (or pass
`break_expired=True`) to reclaim it, and that reclamation bumps the fence and
is recorded in the lease file. Nothing silently evaporates.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional
from contextlib import contextmanager

DEFAULT_TTL_S = 300.0
DEFAULT_LOCK_TIMEOUT_S = 5.0
# A lock file older than this is assumed to belong to a process that died
# between creating it and releasing it. It bounds the damage from a crash
# inside the tiny critical section; see module notes on limits.
LOCK_STALE_S = 30.0


class LeaseState(str, Enum):
    """The three states. SUSPENDED is the unheld state: no holder, nobody
    driving. AUTOMATION and OPERATOR are both *held* states, which is the whole
    point -- handing control to a human is not "no lease", it is a lease held
    by a different party."""

    AUTOMATION = "automation"
    OPERATOR = "operator"
    SUSPENDED = "suspended"


HELD_STATES = frozenset({LeaseState.AUTOMATION, LeaseState.OPERATOR})

# Legal edges of the state machine. Anything not listed is rejected loudly.
_LEGAL_TRANSITIONS: frozenset[tuple[LeaseState, LeaseState]] = frozenset(
    {
        (LeaseState.SUSPENDED, LeaseState.AUTOMATION),
        (LeaseState.SUSPENDED, LeaseState.OPERATOR),
        (LeaseState.AUTOMATION, LeaseState.OPERATOR),
        (LeaseState.OPERATOR, LeaseState.AUTOMATION),
        (LeaseState.AUTOMATION, LeaseState.SUSPENDED),
        (LeaseState.OPERATOR, LeaseState.SUSPENDED),
    }
)


class LeaseError(RuntimeError):
    """Base class. Every failure mode below is a refusal, never a silent no-op."""


class LeaseHeld(LeaseError):
    """Someone else holds it and it has not expired."""


class LeaseStateMismatch(LeaseError):
    """The caller's `expected_state` did not match, or the transition is not a
    legal edge of the state machine."""


class NotHolder(LeaseError):
    """The caller is not the current holder, or presented a stale token/fence.
    This is what a preempted holder gets when it tries to resume."""


class LeaseExpired(LeaseError):
    """The caller held it, but its TTL lapsed and it must not act."""


class LockTimeout(LeaseError):
    """Could not take the lock file in time. Contention, or a wedged holder."""


@dataclass(frozen=True)
class LeaseRecord:
    """The externalised lease, exactly as it appears on disk."""

    state: LeaseState
    holder: Optional[str]
    holder_token: Optional[str]
    fence: int
    acquired_at: Optional[float]
    expires_at: Optional[float]
    ttl_s: float
    note: str = ""

    def is_held(self) -> bool:
        return self.state in HELD_STATES and self.holder is not None

    def is_expired(self, now: float | None = None) -> bool:
        if not self.is_held() or self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at

    def remaining_s(self, now: float | None = None) -> float:
        if self.expires_at is None:
            return float("inf")
        return self.expires_at - (now if now is not None else time.time())

    def to_json(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_json(cls, d: dict) -> "LeaseRecord":
        return cls(
            state=LeaseState(d["state"]),
            holder=d.get("holder"),
            holder_token=d.get("holder_token"),
            fence=int(d.get("fence", 0)),
            acquired_at=d.get("acquired_at"),
            expires_at=d.get("expires_at"),
            ttl_s=float(d.get("ttl_s", DEFAULT_TTL_S)),
            note=d.get("note", ""),
        )


@dataclass(frozen=True)
class LeaseGrant:
    """What a successful acquisition hands back. The holder must present the
    token (and implicitly the fence) on every later mutating call."""

    holder: str
    token: str
    fence: int
    state: LeaseState
    expires_at: float


class SessionLease:
    """File-backed, cross-process, compare-and-swap session lease."""

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.ttl_s = ttl_s
        self.lock_timeout_s = lock_timeout_s
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- reading -----------------------------------------------------------

    def read(self) -> LeaseRecord:
        """Lock-free read. Safe because writes land via `os.replace`, so the
        path always points at a complete file."""
        try:
            raw = self.path.read_text()
        except FileNotFoundError:
            return self._initial()
        if not raw.strip():
            return self._initial()
        try:
            return LeaseRecord.from_json(json.loads(raw))
        except (ValueError, KeyError):
            # A truncated or corrupt lease is NOT treated as "free". Refusing
            # is the fail-closed choice: guessing here could hand two parties
            # the session at once, which is the one thing this class exists to
            # prevent.
            raise LeaseError(f"lease file {self.path} is unreadable/corrupt")

    def _initial(self) -> LeaseRecord:
        return LeaseRecord(
            state=LeaseState.SUSPENDED,
            holder=None,
            holder_token=None,
            fence=0,
            acquired_at=None,
            expires_at=None,
            ttl_s=self.ttl_s,
            note="never acquired",
        )

    # -- critical section --------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Mutual exclusion via `O_CREAT|O_EXCL`, which is atomic on local
        POSIX filesystems and on Windows. Everything that mutates the lease
        goes through here, so read-modify-write cannot interleave."""
        deadline = time.time() + self.lock_timeout_s
        fd = None
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                # Break a lock left behind by a process that died mid-section.
                try:
                    age = time.time() - os.path.getmtime(self.lock_path)
                    if age > LOCK_STALE_S:
                        os.unlink(self.lock_path)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() >= deadline:
                    raise LockTimeout(
                        f"could not acquire {self.lock_path} within "
                        f"{self.lock_timeout_s}s"
                    )
                time.sleep(0.005)
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            fd = None
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass

    def _commit(self, record: LeaseRecord) -> LeaseRecord:
        """Write temp + `os.replace`: readers see old or new, never a splice."""
        tmp = self.path.with_suffix(self.path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(record.to_json(), indent=2) + "\n")
        os.replace(tmp, self.path)
        return record

    # -- mutations ---------------------------------------------------------

    def acquire(
        self,
        holder: str,
        *,
        target_state: LeaseState = LeaseState.AUTOMATION,
        expected_state: LeaseState = LeaseState.SUSPENDED,
        ttl_s: float | None = None,
        break_expired: bool = False,
        note: str = "",
    ) -> LeaseGrant:
        """Compare-and-swap acquisition.

        `expected_state` is the CAS comparand: the caller states what it
        believes the world looks like, and the swap only happens if it is
        right. Two concurrent acquires against the same `expected_state`
        therefore cannot both win -- the loser sees the winner's state.
        """
        if target_state is LeaseState.SUSPENDED:
            raise LeaseStateMismatch("cannot acquire INTO the suspended state")
        ttl = self.ttl_s if ttl_s is None else ttl_s
        with self._locked():
            cur = self.read()
            now = time.time()

            if cur.is_held() and cur.is_expired(now):
                if not break_expired:
                    raise LeaseExpired(
                        f"lease is held by {cur.holder!r} but expired at "
                        f"{cur.expires_at}; reclaim it explicitly "
                        f"(break_expired=True or expire_if_stale())"
                    )
                cur = self._commit(self._suspend(cur, note="expired; reclaimed"))

            if cur.state is not expected_state:
                raise LeaseStateMismatch(
                    f"expected state {expected_state.value!r} but lease is "
                    f"{cur.state.value!r} (holder={cur.holder!r})"
                )
            if cur.is_held():
                # expected_state matched a held state; only a transfer may do
                # that, and transfer() is the API for it.
                raise LeaseHeld(
                    f"lease is held by {cur.holder!r}; use transfer() to move it"
                )
            self._check_edge(cur.state, target_state)

            rec = LeaseRecord(
                state=target_state,
                holder=holder,
                holder_token=uuid.uuid4().hex,
                fence=cur.fence + 1,
                acquired_at=now,
                expires_at=now + ttl,
                ttl_s=ttl,
                note=note or f"acquired by {holder}",
            )
            self._commit(rec)
            return LeaseGrant(
                holder=holder,
                token=rec.holder_token,  # type: ignore[arg-type]
                fence=rec.fence,
                state=rec.state,
                expires_at=rec.expires_at,  # type: ignore[arg-type]
            )

    def release(self, holder: str, *, token: str | None = None, note: str = "") -> LeaseRecord:
        with self._locked():
            cur = self.read()
            self._assert_holder(cur, holder, token)
            rec = self._suspend(cur, note=note or f"released by {holder}")
            return self._commit(rec)

    def transfer(
        self,
        from_holder: str,
        to_holder: str,
        *,
        token: str | None = None,
        to_state: LeaseState = LeaseState.OPERATOR,
        ttl_s: float | None = None,
        note: str = "",
    ) -> LeaseGrant:
        """Hand the session to another party without ever passing through an
        unheld state. Automation -> operator handoff is exactly this: there is
        no instant at which nobody owns the live browser."""
        ttl = self.ttl_s if ttl_s is None else ttl_s
        with self._locked():
            cur = self.read()
            self._assert_holder(cur, from_holder, token)
            self._check_edge(cur.state, to_state)
            now = time.time()
            rec = LeaseRecord(
                state=to_state,
                holder=to_holder,
                holder_token=uuid.uuid4().hex,
                fence=cur.fence + 1,
                acquired_at=now,
                expires_at=now + ttl,
                ttl_s=ttl,
                note=note or f"transferred {from_holder} -> {to_holder}",
            )
            self._commit(rec)
            return LeaseGrant(
                holder=to_holder,
                token=rec.holder_token,  # type: ignore[arg-type]
                fence=rec.fence,
                state=rec.state,
                expires_at=rec.expires_at,  # type: ignore[arg-type]
            )

    def heartbeat(self, holder: str, token: str, *, ttl_s: float | None = None) -> LeaseRecord:
        """Extend the TTL. A holder that stops heartbeating loses the lease --
        that is the crash-recovery path."""
        with self._locked():
            cur = self.read()
            self._assert_holder(cur, holder, token)
            ttl = cur.ttl_s if ttl_s is None else ttl_s
            rec = LeaseRecord(
                state=cur.state,
                holder=cur.holder,
                holder_token=cur.holder_token,
                fence=cur.fence,  # heartbeat is not a transition: fence unchanged
                acquired_at=cur.acquired_at,
                expires_at=time.time() + ttl,
                ttl_s=ttl,
                note="heartbeat",
            )
            return self._commit(rec)

    def expire_if_stale(self, *, note: str = "") -> LeaseRecord:
        """Explicit reclamation of a lapsed lease. Bumps the fence, so the
        crashed-or-slow previous holder can never resume."""
        with self._locked():
            cur = self.read()
            if not (cur.is_held() and cur.is_expired()):
                return cur
            rec = self._suspend(
                cur, note=note or f"TTL expired; reclaimed from {cur.holder}"
            )
            return self._commit(rec)

    # -- validation --------------------------------------------------------

    def validate(self, holder: str, token: str, *, fence: int | None = None) -> LeaseRecord:
        """The call a long-running holder makes before touching the surface.

        This is where fencing earns its keep: a holder that was preempted while
        it was blocked somewhere gets `NotHolder` here instead of quietly
        driving a session someone else owns.
        """
        cur = self.read()
        self._assert_holder(cur, holder, token, fence=fence)
        if cur.is_expired():
            raise LeaseExpired(f"lease for {holder!r} expired at {cur.expires_at}")
        return cur

    def _assert_holder(
        self,
        cur: LeaseRecord,
        holder: str,
        token: str | None,
        *,
        fence: int | None = None,
    ) -> None:
        if not cur.is_held():
            raise NotHolder(f"lease is not held (state={cur.state.value})")
        if cur.holder != holder:
            raise NotHolder(f"lease is held by {cur.holder!r}, not {holder!r}")
        if token is not None and cur.holder_token != token:
            raise NotHolder(
                f"stale holder token for {holder!r}: this acquisition was "
                f"superseded (current fence={cur.fence})"
            )
        if fence is not None and fence != cur.fence:
            raise NotHolder(
                f"stale fence {fence} != current {cur.fence}; this holder was "
                f"preempted and may not resume"
            )

    def _check_edge(self, src: LeaseState, dst: LeaseState) -> None:
        if (src, dst) not in _LEGAL_TRANSITIONS:
            raise LeaseStateMismatch(
                f"illegal lease transition {src.value} -> {dst.value}"
            )

    def _suspend(self, cur: LeaseRecord, *, note: str) -> LeaseRecord:
        return LeaseRecord(
            state=LeaseState.SUSPENDED,
            holder=None,
            holder_token=None,
            fence=cur.fence + 1,
            acquired_at=None,
            expires_at=None,
            ttl_s=cur.ttl_s,
            note=note,
        )

    # -- waiting -----------------------------------------------------------

    def wait_for_state(
        self,
        state: LeaseState,
        timeout: float,
        *,
        poll_s: float = 0.02,
    ) -> LeaseRecord:
        """Poll until the lease reaches `state`. Polling, not inotify, because
        the mechanism has to work identically on any filesystem the console and
        engine can both see."""
        deadline = time.time() + timeout
        while True:
            cur = self.read()
            if cur.state is state:
                return cur
            if time.time() >= deadline:
                raise TimeoutError(
                    f"lease did not reach {state.value!r} within {timeout}s "
                    f"(still {cur.state.value!r}, holder={cur.holder!r})"
                )
            time.sleep(poll_s)


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
    "LockTimeout",
    "HELD_STATES",
    "DEFAULT_TTL_S",
]
