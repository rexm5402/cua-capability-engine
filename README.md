# Computer-Use Automation System

An LLM discovers how to drive an API-less legacy application once. The run is
compiled into a typed, versioned **capability artifact**. From then on the flow
replays **deterministically, with no model in the loop** — with an explicit error
taxonomy, safety guardrails, and a real human-takeover path when it gets stuck.

Design rationale, trade-offs and cut lines: **[REPORT.md](REPORT.md)**.

    goal ──▶ LLM discovery run ──▶ artifact ──▶ verified by LLM-free replay ──▶ saved
                                                          │
                          AI agent ──▶ replay(capability, inputs) ──▶ Success
                                                                   │ BusinessOutcome
                                                                   │ Failure ──▶ human

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium        # only needed for live runs
```

### Configuration

Only the discovery run needs a model. Replay never calls one.

```bash
export XAI_API_KEY=...             # provider-agnostic; see src/cua/llm/client.py
export CUA_LLM_PROVIDER=xai        # xai (default) | openai
export CUA_LLM_MODEL=grok-4
```

No key is needed to run the test suite, the fixture app, or a replay against a
recorded surface.

## Demo path

**1. Start the target application** — a deliberately hostile stand-in for legacy
bank software: frameset-based, nested-table layout, machine-generated element
names, no test ids. See [fixtures/legacy_bank/README.md](fixtures/legacy_bank/README.md).

```bash
python -m fixtures.legacy_bank.app          # http://localhost:5055
```

**2. Discovery — the real LLM-driven run.**

```bash
python -m cua discover \
  --goal "look up member 10001 and read their current savings balance" \
  --url http://localhost:5055/search \
  --out artifacts/lookup_member_balance.yaml
```

The agent observes the accessibility tree, decides, and acts. On success it
compiles the trajectory into an artifact, then **immediately replays that
artifact with the LLM switched off and a different member id**. If the
verification replay fails, the artifact is not written. A saved artifact is
therefore a flow already proven to run without a model.

**3. Replay — the production path an AI agent triggers.**

```bash
python -m cua replay artifacts/lookup_member_balance.yaml --input member_id=10002
```

**4. Replay against an exceptional state.** The fixture app can be told to
misbehave on demand:

```bash
python -m cua replay artifacts/lookup_member_balance.yaml \
  --input member_id=10001 --inject not_found
```

Returns a **`BusinessOutcome`**, not a failure — "no such member" is a legitimate
answer the caller needs. Try `--inject dialog` (recovered silently),
`--inject slow`, `--inject perm_denied`, or `--inject error` (hard failure with a
debuggable payload).

**5. Human escalation.** In a second terminal:

```bash
python -m cua console                       # http://localhost:5056
```

When a run gets stuck it raises an intervention request, releases the session
lease, and blocks. The operator takes over the *same live browser session*,
fixes things by hand, and hits Resume — at which point automation reacquires the
lease and **re-verifies the current step's post-condition before continuing.**

## Tests

```bash
pytest -q
```

The suite is **hermetic**: no browser, no network, no API key. This works because
`RecordedSurface` is a second real implementation of the `Surface` protocol,
which is also the proof that the surface seam is genuine rather than asserted.
Live-browser tests skip cleanly when Chromium is absent.

## Layout

| Path | What |
|---|---|
| `src/cua/artifact.py` | The capability schema — the contract for the whole system |
| `src/cua/surface/` | Six-method `Surface` protocol; Playwright and recorded implementations |
| `src/cua/locator/` | Five-tier locator bundles: record, and resolve with fallbacks |
| `src/cua/replay/` | Deterministic execution, signal scan, outcome union |
| `src/cua/discovery/` | LLM loop and trajectory compiler |
| `src/cua/policy/` | Allowlist gate, risk classes, redaction |
| `src/cua/session/` | Session lease, escalation, operator console |
| `src/cua/telemetry.py` | Replay stability and drift, kept off the artifact |
| `fixtures/legacy_bank/` | The hostile target app, two tenants, fault injection |
| `evidence/` | Committed runs: discovery, clean replay, failing replay |

## What is mocked, and why

Documented in full in [REPORT.md §7](REPORT.md). In short: the desktop surface is
the protocol plus a second implementation rather than a third against a platform
accessibility API; the operator console is deliberately bare, because the brief
asks for a real control-transfer *mechanism* and that mechanism is the lease, not
the UI; and no scaling infrastructure was built, which the brief explicitly does
not reward.
