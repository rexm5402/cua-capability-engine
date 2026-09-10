# Design Write-Up

## 1. Architecture

This is not a browser-automation project; it is a **compiler**. The LLM is a
slow, expensive, non-deterministic frontend that runs once against a live
surface and emits an IR. The replay engine is a fast, deterministic backend that
executes that IR forever with no model in the loop. Everything genuinely hard
lives in the IR design and in what happens when runtime disagrees with the
recording.

One Python process, four layers, hard internal seams:

    surface/    Surface protocol; WebSurface (Playwright) + RecordedSurface
    locator/    record and resolve LocatorBundle
    discovery/  LLM observe -> decide -> act; compile trajectory to artifact
    replay/     artifact + inputs -> ReplayOutcome
    policy/     allowlist, risk class, redaction   (wraps every action)
    evidence/   structured run records             (wraps every run)
    session/    lease, escalation, operator console
    telemetry/  replay stability and drift, kept OFF the artifact

No queues, no services, no multi-tenant plumbing. The brief rewards abstractions
that *could* scale, not infrastructure built before it is needed.

**The load-bearing decision is the `Surface` protocol**: six methods — `observe`,
`act`, `read`, `wait_for`, `snapshot`, `describe`. Neither the discovery agent
nor the replay engine imports Playwright. A native desktop surface is six
methods, and nothing upstream changes.

We ship two implementations. `RecordedSurface` replays serialized observations,
which makes the whole test suite hermetic — no browser, no network, no API key —
and simultaneously *proves* the seam rather than asserting it. It is also the
answer to "how do I run this without live services."

**Trade-off accepted:** a single process cannot survive a crash mid-run, and the
lease (below) is file-backed rather than a real coordination service. Both are
correct at this scale and both are stated as limits rather than hidden.

## 2. Artifact schema

Pydantic models serialized as YAML: diffable and reviewable by a human, with
JSON Schema generated for a calling agent. A `Capability` carries identity and
semver, an `AppProfileRef`, typed `inputs`/`outputs`, `possible_outcomes`,
ordered `steps`, a terminal `success` checkpoint, declared `signals`, a risk
class, an approval state, and provenance.

Four choices worth defending:

**Steps carry an `intent` string.** Prose, not just a click list. Whoever
approves a capability that touches a member account reads intent, and so does an
operator handed an intervention request. It costs nothing.

**Inputs, outputs and possible outcomes are the contract; steps are the
implementation.** A calling agent consumes `tool_schema()` and invokes by name
with typed args, never reading steps. Declaring the business outcomes matters as
much as declaring the inputs: a caller that does not know `record_not_found` is
a possible *answer* will treat it as an error, reintroducing exactly the
confusion §3 exists to prevent.

**Authentication is a precondition, not steps.** Credentials are the one thing
that must never be recorded, so artifacts begin post-auth and a `SessionProvider`
establishes context. This also supplies the re-auth path when a session-expiry
signal fires mid-replay.

**The artifact is decoupled from the transcript and proven before it is saved.**
Discovery compiles a trajectory, then immediately replays it deterministically
against a fresh session; if that fails, the artifact is never written. Two
refinements make this a real proof rather than a nice story:

- Verification replays with a **different input** than discovery used. If the
  compiler baked a literal in place of a parameter, verification now fails —
  otherwise we would only be proving the flow runs, not that it is parameterized.
- Write capabilities verify in **dry-run** mode: execute to the last safe step,
  then resolve every remaining locator without acting. Verifying "open a
  sub-account" by replaying it would create a second real record — in a banking
  context, precisely the sin the safety model exists to prevent.

`provenance.verification_mode` records which proof an artifact carries.

This is not theoretical. During the reference run the model found a flow that
began by pressing Retrieve on an empty form -- which renders "No records
located." before the real search. On verification, replay correctly classified
that page as `record_not_found` and stopped, so **the artifact was never
written**. Verify-before-save caught a flow that worked for the model and would
have misreported in production.

This is not theoretical. During the reference run the model found a flow that
began by pressing Retrieve on an empty form -- which renders "No records
located." before the real search. On verification, replay correctly classified
that page as `record_not_found` and stopped, so **the artifact was never
written**. Verify-before-save caught a flow that worked for the model and would
have misreported in production.

Telemetry deliberately does **not** live on the artifact: it mutates every
replay and would churn the hash of a supposedly immutable, reviewable document
without its behaviour changing.

## 3. Determinism & error handling

Replay returns a discriminated union of three types:

- **`Success`** — terminal checkpoint verified, typed outputs returned.
- **`BusinessOutcome`** — `record_not_found`, `permission_denied`,
  `validation_rejected`: a legitimate answer with a code, message and any
  partial outputs. Not an exception.
- **`Failure`** — step id, step intent, expected, observed, resolved tier,
  candidate count, evidence paths. Debuggable by construction.

Recoverable conditions never reach the caller: known interstitials are
dismissed, transient loads waited out, expired sessions re-authenticated, each
under an attempt budget so a condition that never clears becomes a hard failure
rather than an infinite loop. Whether the original action is re-performed after
a dismissal depends on whether it landed: an action that succeeded before the
interstitial appeared must not be repeated, while one that failed *because* the
interstitial covered the control has not happened yet and is retried. Whether the original action is re-performed after
a dismissal depends on whether it landed: an action that succeeded before the
interstitial appeared must not be repeated, while one that failed *because* the
interstitial covered the control has not happened yet and is retried.

The glossary names conflating a business outcome with a failure as the most
common design mistake here. Three distinct types make it structurally impossible
rather than merely discouraged.

**Detection.** After every action, before proceeding, a signal scan evaluates
declared `SignalRule`s mapping observed conditions to a classification and a
handler. Signals are *data*, not code — which is what lets a differently-worded
error message at another institution be a one-line overlay.

**Determinism** rests on four rules: no LLM in the replay path; no blind sleeps,
only explicit wait conditions; every state-changing step verifies a
post-condition before the next runs (enforced by a schema validator, not
trusted to the compiler); and every locator requires a unique match.

**Locators** are a ranked bundle of five independent strategies: semantic
(role + accessible name), anchor-relative (the control in a stated relation to a
labelled anchor), text, structural (frame + DOM path, recorded but distrusted),
geometry. Two rules carry it:

- **Unique match required.** Ambiguity falls through to the next tier and is
  reported; it never takes the first candidate. A tie within 2px is ambiguity,
  not a coin flip. Guessing is how automation quietly does the wrong thing to a
  customer account. For lists — eight identical "View" buttons — a `scope_text`
  narrows to the row region first, then requires uniqueness *within* it, so
  lists work without weakening the rule.
- **Tier escalation is drift.** Each step records the tier that resolved it at
  record time; drift is deviation from *that baseline*, not from tier 1, since a
  hostile surface may legitimately baseline at tier 2 and a fixed rule would
  fire constantly and mean nothing. Drift detection therefore falls out of
  telemetry we already collect.

Anchor-relative is the workhorse. On table-soup legacy screens the only stable
handle is the label printed beside the field; "the textbox in the same row as
the cell reading 'Member ID'" survives restyling, rebranding, and reordering of
unrelated rows. Our tests record a locator against one tenant's markup and
resolve it against a second tenant's — different order, different styling,
different DOM paths — and it still lands on the right control.

At record time every synthesized strategy is validated by resolving it against
its own observation; ambiguous ones are dropped. We only record locators proven
unique.

Two bugs here were found only by running against a real page, and both shape the
design. First, row tolerances were absolute pixel constants while the surface
reports normalized `[0,1]` geometry -- a tolerance of six full viewports, under
which every node shared a row with every other and the direction test was
vacuous. All 35 synthetic locator tests passed through it, because their
fixtures used pixel-like numbers. Tolerances are now derived from the boxes
themselves and are unit-free. Second, `_same_row` used `max(heights)`, letting a
tall ancestor widen its own band until it captured rows far away; in a
table-soup accessibility tree those ancestors are everywhere, and they inherit
their children's concatenated accessible name and bounding box. Matching now
requires mutual centre containment, and containment chains collapse to the
innermost node before ambiguity is judged -- an ancestor and its descendant are
one visual control at two depths, not two matches.

Two bugs here were found only by running against a real page, and both are worth
stating because they shape the design. First, row tolerances were absolute pixel
constants while the surface reports normalized `[0,1]` geometry -- a tolerance of
six full viewports, under which every node shared a row with every other and the
direction test was vacuous. All 35 synthetic locator tests passed through it,
because their fixtures used pixel-like numbers. Tolerances are now derived from
the boxes themselves and are unit-free. Second, `_same_row` used `max(heights)`,
letting a tall ancestor widen its own band until it captured rows far away; in a
table-soup accessibility tree those ancestors are everywhere, and they inherit
their children's concatenated accessible name and bounding box. Matching now
requires mutual centre containment, and containment chains collapse to the
innermost node before ambiguity is judged -- an ancestor and its descendant are
one visual control at two depths, not two matches.

## 4. Heterogeneity & multi-tenant

**Surfaces.** The seam is the `Surface` protocol, and the artifact never names a
technology. Locators are expressed in terms — role, accessible name, label
adjacency — that a desktop accessibility API supplies as readily as a browser.
Tiers 4 and 5 degrade rather than break: a desktop surface records no DOM path,
and anchor-relative falls back to tree-order adjacency when no geometry exists,
which is exactly the desktop case.

**Tenants.** An artifact is a base plus an optional per-tenant overlay keyed by
vendor product and version. An overlay may override locator bundles, signal
wording, entry URL and individual steps; it never forks the flow. Concrete
routes canonicalize to parameterized patterns (`/member/12345` -> `/member/:id`).
Drift is detected from baseline-relative tier escalation and checkpoint failure
rates, which identifies which tenants need an overlay *before* a run fails —
rather than re-recording per tenant.

## 5. Escalation & handoff

A `SessionLease` with states `AUTOMATION | OPERATOR | SUSPENDED` and exactly one
holder, ever. That invariant *is* the control-transfer model.

The operator console runs in a different process, so an in-memory lease would
make the invariant decorative. It is externalized: a lease file with atomic
compare-and-swap, a holder token, a fencing counter so a preempted holder cannot
resume, and a TTL so a crashed holder does not deadlock the session.

1. **Detect stuck** — discovery makes no state progress for N steps; replay
   produces a `Failure`; or the gate returns `RequireApproval` for an
   irreversible action. One channel serves both "stuck" and "please approve" —
   the panic button and the permission button are the same button.
2. **Route** — an `InterventionRequest` carrying capability, step, intent, why it
   stopped, URL, screenshot, permitted actions and a deadline: enough for a
   person to act without reading code.
3. **Cede** — automation releases the lease and blocks. The browser is headed, so
   the operator takes over the *same live context*: same session, same cookies,
   same half-filled form.
4. **Observe** — sample observations while the operator holds the lease and
   record a `HumanIntervention` describing what changed.
5. **Resume** — automation reacquires and **re-observes and re-verifies the
   current step's post-condition before continuing.** Never assume the human
   left the session where the automation expected it; a mismatch is a fresh
   failure, not a silent continue.

**Stated limit.** Headed handoff works when the operator is at the machine. In
production the engine runs on a server and the operator is elsewhere, so the
real transport is CDP screencast plus input forwarding. That changes how pixels
and clicks travel, not who holds the lease — the lease is transport-independent,
which is why it, not the console, is the contribution.

## 6. Safety

Every action passes `PolicyGate.check`. One chokepoint, so allowlist enforcement
is architectural rather than a matter of remembering.

- **Deny by default.** A gate with no policy denies everything; that is tested.
  Order is action type -> URL -> risk, so an irreversible action on a
  non-allowlisted origin is flatly denied and never reaches a human as an
  approval prompt. `NAVIGATE` checks both current and destination URL, so an
  allowlisted page cannot be a springboard off the allowlist.
- **Risk classes.** `irreversible` is denied outright in unattended mode and
  routed to human approval in attended mode.
- **Redaction.** Sensitivity tags on parameters; secrets referenced by vault key,
  never serialized. Late binding is enforced by a schema validator: the
  discovery agent emits only parameter *references* and the engine substitutes
  the real value at the moment of action, so sensitive data never enters the
  model transcript. Screenshots are masked from raw bytes before a file exists,
  and fail closed — deleting rather than writing an unmasked image. A logging
  filter scrubs records as defence in depth.

**Honest limits.** The gate binds only callers who call it: it is an object in
the same process as the agent, not a proxy or sandbox, so code that constructs a
`Surface` directly bypasses it. Making the claim structural rather than
conventional needs enforcement *below* the agent — a filtering proxy or a
privileged process holding the only surface handle. URL allowlisting is a weak
proxy for what an action *does*: risk is a label applied at discovery time, and
a step mislabelled `reversible` that actually submits a payment is allowed.
Redaction is best-effort in both directions — it will miss a name, an address,
or a reformatted identifier, and over-redact legitimate long numbers; screenshot
masking only covers regions someone thought to declare. The over-redaction risk
is real and we hit it: a 13-digit epoch-millisecond screenshot filename was
scrubbed as a card number, corrupting the evidence reference it was meant to
protect. Filenames now carry a UTC timestamp, but the general hazard -- a
heuristic that cannot tell an account number from a claim id -- remains. Finally, a compromised
application page could prompt-inject the discovery agent — which is exactly why
the production path contains no model at all.

## 7. Cuts

**Deliberately not built:**

- *A real desktop surface.* The protocol plus a second implementation
  (`RecordedSurface`) proves the seam; a third against a platform accessibility
  API is mechanical, not novel.
- *A co-browsing operator console.* Minimal server-rendered page instead. The
  brief asks for a real mechanism, not a console; the lease is the mechanism.
- *Queues, clusters, multi-tenant plumbing.* Explicitly not rewarded.
- *A credential vault.* Interface only.
- *LLM-assisted recovery on replay failure.* Tempting, and deliberately omitted:
  a bounded single-step recovery reintroduces a model into the production path,
  and its policy story deserves more care than time allowed.

**Known gaps, disclosed rather than hidden:**

- *`ValueRef` cannot express a templated URL.* A `NAVIGATE` step whose URL
  embeds an input can only compile as a literal, since `ValueRef` is
  literal-or-param with no interpolation. It fails closed -- verification
  rejects such an artifact -- but the fix is a schema change (`ValueRef` needs a
  `template` form). The reference flow uses the search form, so this is latent.
- *Canonical route patterns have nowhere to live on `Capability`.* They end up
  inside `Checkpoint.url_matches` regexes rather than in a field a reviewer
  would look for.
- *The cross-tenant claim is designed and fixtured but not demonstrated.* Tenant
  B exists in the fixture app with reordered fields and different wording; no
  artifact has been run against it with an overlay.
- *Discovery is prompt-sensitive.* The model sometimes records a redundant first
  action. Verification catches the consequences, but the artifact quality varies
  run to run in a way a production system would want to normalize.

**Known gaps, disclosed rather than hidden:**

- *`ValueRef` cannot express a templated URL.* A `NAVIGATE` step whose URL
  embeds an input can only compile as a literal, since `ValueRef` is
  literal-or-param with no interpolation. It fails closed -- verification
  rejects such an artifact -- but the fix is a schema change (`ValueRef` needs a
  `template` form). The reference flow uses the search form, so this is latent.
- *Canonical route patterns have nowhere to live on `Capability`.* They end up
  inside `Checkpoint.url_matches` regexes rather than in a field a reviewer
  would look for.
- *The cross-tenant claim is designed and fixtured but not demonstrated.*
  Tenant B exists in the fixture app with reordered fields and different
  wording; no artifact has been run against it with an overlay.
- *Discovery is prompt-sensitive.* The model sometimes records a redundant first
  action; verification catches the consequences, but artifact quality varies run
  to run in a way a production system would want to normalize.

**Next, in order:** the tenant-B overlay applied end to end; the capability
catalog exposed as a callable tool surface; multi-run stability scoring gating
unattended replay on a draft -> approved transition; then CDP-based remote
handoff, which is the only cut that changes an architectural claim rather than
just deferring work.
