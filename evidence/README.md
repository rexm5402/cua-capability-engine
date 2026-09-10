# Evidence

Runs produced by the committed code against the local fixture app
(`fixtures/legacy_bank/`, "CoreTeller 4.2"). Discovery used
**gpt-oss:120b-cloud via Ollama**; every replay ran with **no model in the
loop**.

| Run | What it shows |
|---|---|
| `01-discovery` | The genuine LLM-driven run: observe → decide → act against the live frameset app, then compiled to an artifact and **verified by an LLM-free replay before being written**. |
| `02-replay-success` | Deterministic replay with `member_id=10002` — a **different member than discovery used**, which is what proves the flow is parameterized rather than baked. Returns `Success` with the typed output. |
| `03-replay-business-outcome-not-found` | `member_id=99999` returns **`BusinessOutcome(record_not_found)`**, not a failure. "No such member" is a legitimate answer the caller needs. |
| `04-replay-recovered-interstitial` | A maintenance interstitial is injected; replay **dismisses it and completes successfully**. The caller sees `Success` and never learns it happened. |
| `05-replay-hard-failure` | An application error is injected; replay stops with **`Failure(application_error)`** carrying step id, intent, expected, observed, and a screenshot. |

Each run directory holds `manifest.json`, `steps.jsonl` (one record per step,
with the resolved locator tier), `run.log`, and screenshots on failure.

## Reproducing

```bash
python -m fixtures.legacy_bank.app                 # http://localhost:5055
export CUA_APP_USER=demo CUA_APP_PASS=demo         # fixture credentials, never recorded
export CUA_LLM_PROVIDER=ollama CUA_LLM_MODEL=gpt-oss:120b-cloud

python -m cua discover \
  --goal "look up member 10001 and read their current savings balance" \
  --url http://localhost:5055/search \
  --input member_id=10001 --verify-input member_id=10002 \
  --profile profiles/coreteller.yaml --name lookup_member_balance \
  --out artifacts/lookup_member_balance.yaml

python -m cua replay artifacts/lookup_member_balance.yaml --input member_id=10002
python -m cua replay artifacts/lookup_member_balance.yaml --input member_id=99999
python -m cua replay artifacts/lookup_member_balance.yaml --input member_id=10001 --inject dialog
python -m cua replay artifacts/lookup_member_balance.yaml --input member_id=10001 --inject error
```

## Notes worth reading

**Redaction is visible in the logs.** `01-discovery/steps.jsonl` records the
typed value as `<param:member_id>` — the member id never entered the model
transcript or the evidence, because the agent emits parameter *references* and
the engine binds the real value at the moment of action.

**Verification rejected an earlier artifact.** During development the model
found a flow that began by pressing Retrieve on an empty form, which renders
"No records located." before the real search. Replay correctly classified that
page as `record_not_found` and stopped, so the artifact was **not written**.
Verify-before-save caught a flow that would have misreported in production.
