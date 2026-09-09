# Computer-Use Automation System — Design Spec

Date: 2026-09-09
Status: approved, in implementation

## 1. Problem framing

The deliverable is not browser automation. It is a **compiler**.

An LLM is a slow, expensive, non-deterministic frontend that runs once against a
live application surface and emits an intermediate representation. The replay
engine is a fast, deterministic backend that executes that IR forever, with no
model in the loop. Everything genuinely hard in this project lives in the IR
design and in what happens when the runtime does not match the recording.

The target environment is API-less back-office banking software: stable UIs,
hostile markup, real runtime errors, and hundreds of tenants running the same
vendor product configured differently.

## 2. Architecture

Single Python process, four layers, hard internal seams. No queues, no services,
no multi-tenant plumbing — the brief explicitly does not reward building scaling
infrastructure, only designing abstractions that could scale.

    surface/    Surface protocol + WebSurface (Playwright) + RecordedSurface
    locator/    record and resolve LocatorBundle
    discovery/  LLM observe -> decide -> act; compile trajectory to artifact
    replay/     artifact + inputs -> ReplayOutcome
    policy/     allowlist, risk class, redaction   (wraps every action)
    evidence/   structured run records             (wraps every run)
    session/    lease, escalation, operator surface

### The Surface seam

`Surface` is a Protocol of six methods: `observe`, `act`, `read`, `wait_for`,
`snapshot`, `describe`. Neither the discovery agent nor the replay engine
imports Playwright. A new surface — a native desktop app driven through the
platform accessibility API — is six methods, and nothing upstream changes.

We ship two implementations. `RecordedSurface` replays serialized observations,
which makes the entire test suite hermetic (no browser, no network, no API key)
and simultaneously proves the seam is real rather than asserted.

## 3. Artifact schema

Pydantic models serialized as YAML: diffable and reviewable by a human, with
JSON Schema generated automatically for a calling agent.

A `Capability` carries: identity and semver, an `AppProfileRef` (vendor product,
version, optional tenant, optional base artifact for overlays), typed `inputs`
and `outputs`, ordered `steps`, a terminal `success` checkpoint, declared
`signals`, a risk class, an approval state, provenance, and stability stats.

Three deliberate choices:

**Every step carries an `intent` string.** Prose, not just a click list. A human
approving a capability that touches a member account reads intent. It costs
nothing and it is what makes the artifact reviewable.

**Inputs and outputs are the contract; steps are the implementation.** A calling
agent consumes `Capability.tool_schema()` and invokes by name with typed args,
never reading the steps. That is what makes this a capability rather than a
macro.

**The artifact is decoupled from the model transcript and proven before it is
saved.** Discovery runs, compiles a trajectory into an artifact, then
immediately replays that artifact deterministically against a fresh session. If
the verification replay does not pass, the artifact is never written. A saved
artifact is therefore not what the model claims it did — it is a flow that has
demonstrably executed without an LLM at least once. `provenance.verified_at`
records this. Record, verify, commit.

## 4. Locator strategy

Each target records a ranked bundle of independent strategies:

| Tier | Strategy | Rationale |
|---|---|---|
| 1 | Semantic: role + accessible name + scope | Exists on legacy web and desktop AX alike |
| 2 | Anchor-relative: control in a stated relation to a labelled anchor | The workhorse for table-based legacy screens |
| 3 | Text: normalized content, optional role, nth | |
| 4 | Structural: frame path + DOM path | Recorded but distrusted |
| 5 | Geometry: normalized bounding box | Disambiguation and desktop last resort |

Two rules carry the design:

**Unique match required.** A strategy succeeds only if exactly one node matches.
Ambiguity falls through to the next tier and is reported; it never silently
takes the first candidate. Guessing is how automation quietly does the wrong
thing to a customer account.

**Tier escalation is a drift signal.** Resolution records which tier actually
resolved. Any resolution above tier 1 is emitted into evidence. Drift detection
therefore falls out of telemetry we were already collecting: when one tenant
begins resolving at tier 3, that tenant needs an overlay — and we know before
anything breaks.

Tier 2 deserves the emphasis. Legacy bank screens are table soup where the only
stable handle is the label printed beside the field. "The textbox in the same
row as the cell reading 'Member ID'" survives restyling, rebranding, and
reordering of unrelated rows — which is precisely the multi-tenant case.

At record time, every synthesized strategy is validated by resolving it against
the observation it came from; ambiguous strategies are dropped. We only record
locators proven unique.

## 5. Determinism and error handling

Replay returns a discriminated union of three types:

- `Success` — checkpoint verified, typed outputs returned.
- `BusinessOutcome` — `record_not_found`, `permission_denied`,
  `validation_rejected`. A legitimate answer with a code, message and any
  partial outputs. Not an exception.
- `Failure` — step id, step intent, expected, observed, resolved tier,
  candidate count, evidence paths. Debuggable by construction.

Recoverable conditions never reach the caller at all: known interstitials are
dismissed, transient loads are waited out, expired sessions trigger bounded
re-authentication.

The brief's glossary names conflating a business outcome with a failure as the
most common design mistake in this problem. Representing them as three distinct
types makes it structurally impossible rather than merely discouraged.

Detection mechanism: after every action, before proceeding, a signal scan runs.
Signals are declared data (`SignalRule` on the artifact and app profile) mapping
observed conditions to a classification and a handler — not hardcoded strings.
That is what lets a differently-worded error message at another institution be a
one-line overlay instead of a code change.

Determinism rests on four rules: no LLM in the replay path; no blind sleeps,
only explicit wait conditions; every step verifies its post-condition before the
next step runs; every locator requires a unique match.

## 6. Escalation and control transfer

A `SessionLease` with an explicit state machine: `AUTOMATION | OPERATOR |
SUSPENDED`. Exactly one holder, ever. That invariant is the control-transfer
model.

1. **Detect stuck** — three triggers: discovery makes no state progress for N
   steps; replay produces a `Failure`; or the policy gate encounters an
   irreversible action requiring approval.
2. **Route** — an `InterventionRequest` carrying capability, step, intent, why
   it stopped, current observation, screenshot, permitted actions, deadline.
3. **Cede** — automation releases the lease and blocks. The browser runs headed,
   so the human takes control of the same live context: same session, same
   cookies, same half-completed form. Not a fresh one; a fresh one would be
   useless.
4. **Operator surface** — a minimal FastAPI page: the request, a live screenshot
   stream, Resume / Abort / Fail. Deliberately bare and documented as such; the
   brief asks for a real mechanism, not a co-browsing console.
5. **Observe the human** — while the operator holds the lease, sample
   observations and record a `HumanIntervention` describing what changed.
6. **Resume** — automation reacquires the lease and re-observes and re-verifies
   the current step's post-condition before continuing. Never assume the human
   left the session where the automation expected it.

## 7. Safety

Every action passes `PolicyGate.check`. One chokepoint, so allowlist enforcement
is architectural rather than a matter of remembering.

- **Allowlist**: permitted origins, route patterns, action types. Deny by default.
- **Risk classes**: `read_only` / `reversible` / `irreversible`. Irreversible is
  blocked outright in unattended mode and routed to human approval in attended
  mode, reusing the escalation channel — the panic button and the permission
  button are the same button.
- **Redaction**: params carry a sensitivity tag. Secrets are referenced by vault
  key and never serialized. The discovery agent sees `<param:member_id>`
  placeholders, never real values, so sensitive data never enters the model
  transcript. Screenshots are masked at recorded bounding boxes before bytes
  reach disk, and fail closed if masking is unavailable. A redaction filter sits
  on the log handler as defence in depth.

Honest limits, to be stated in the write-up: an allowlist cannot stop a
semantically wrong but permitted action; screenshot masking is best-effort and
depends on correct sensitivity tagging at record time; and a compromised
application page could in principle prompt-inject the discovery agent — which is
exactly why the production path contains no model at all.

## 8. Heterogeneity and multi-tenant

**Surfaces.** The seam is the `Surface` protocol, and the artifact never names a
technology. A locator bundle is expressed in terms — role, accessible name,
label adjacency — that a desktop accessibility API supplies as readily as a
browser. Tiers 4 and 5 degrade rather than break: a desktop surface simply
records no DOM path.

**Tenants.** An artifact is a base plus optional per-tenant overlay, keyed by
vendor product and version. An overlay may override locator bundles, signal
wording, entry URL, and individual steps; it never forks the flow. Drift is
detected from tier-escalation telemetry and checkpoint failure rates, which
tells us which tenants need an overlay before a run fails. Concrete routes are
canonicalized into parameterized patterns so `/member/12345` records as
`/member/:id`.

## 9. Target application

`fixtures/legacy_bank/` — a Flask app, "CoreBanking 3.1": server-rendered,
frameset-based, nested table layout, machine-generated element names, no test
ids. Flow: login, member search, member detail, open sub-account, confirmation.

It carries a fault-injection switch for `not_found`, `perm_denied`,
`validation`, `dialog`, `timeout`, `slow`, and `error`, which turns "show how
you handle exceptional states" from a hope into a make target.

A tenant B variant serves the same product rebranded, with reordered fields and
differently worded conditions, so one artifact plus a small overlay demonstrates
cross-tenant reuse instead of describing it.

## 10. Build order and cut lines

1. Fixture app and fault injection
2. Surface protocol, WebSurface, RecordedSurface
3. Locator record and resolve
4. Artifact models and validation
5. Replay engine, checkpoints, outcome union
6. Discovery agent, trajectory compiler, verify-before-save
7. Policy gate and redaction
8. Evidence recorder
9. Escalation, lease, operator page
10. Tenant B overlay and capability catalog export (only once 1-9 are solid)

Deliberate cuts, to be documented: a real desktop surface (protocol plus the
`RecordedSurface` proof instead), a co-browsing operator console (minimal page
instead), queues and multi-tenant plumbing (explicitly not rewarded), a
credential vault (interface only).

Tests target what matters: locator resolution against mutated fixtures, the
error taxonomy against the fault injector, redaction, and policy enforcement —
all hermetic via `RecordedSurface`, requiring no browser and no API key. That is
also the answer to running the project without live services.
