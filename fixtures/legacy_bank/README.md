# `legacy_bank` — hostile target application

A Flask fixture standing in for API-less legacy bank back-office software:
server-rendered HTML 4.01, `<frameset>`, nested-table layout, machine-generated
element names, zero JavaScript. It is the target that the capability engine
records against and replays against.

Everything here is **synthetic**. The members, balances and reference numbers
are invented; the only credentials are `demo` / `demo` (and in fact any
non-empty pair is accepted).

## Running

```bash
python -m fixtures.legacy_bank.app                       # port 5055
flask --app fixtures/legacy_bank/app.py run -p 5055
TENANT=b python -m fixtures.legacy_bank.app              # default to tenant B
PORT=5060 python -m fixtures.legacy_bank.app
```

Requires `flask>=3.0` (already in the project's dependencies).

Both tenants are always reachable regardless of `TENANT`, via the URL prefix:

* `http://127.0.0.1:5055/` — tenant selected by `TENANT` (default `a`)
* `http://127.0.0.1:5055/t/a/` — tenant A explicitly
* `http://127.0.0.1:5055/t/b/` — tenant B explicitly

## Why it is ugly (on purpose)

| Trait | Why |
| --- | --- |
| `<frameset>` shell on every working page | Automation must descend into the `workArea` frame; the top document has no controls at all. |
| Nested `<table>` for positioning | No CSS layout system, no containers to anchor on. |
| Names like `f_7`, `ctl00_x3`, `gridRow_alt` | Look machine-generated and carry no meaning. There are **no** `data-testid`s and no semantic ids. |
| `<label for>` + label-adjacent `<td>` | These are the *only* reliable handles. A real (old) enterprise app plausibly has both, and they are what an accessibility-tree-driven locator should key on. |
| Forms `target="_top"` | Posting rebuilds the frameset instead of nesting it, which is how these apps behave. |

## Routes

Every user-facing route renders a frameset; its `/f/...` twin renders the actual
content inside the `workArea` frame. Prefix any of these with `/t/a` or `/t/b`.

| Route | Method | What it is |
| --- | --- | --- |
| `/` | GET, POST | Sign-on. Fields `ctl00_u1` (Operator ID) and `ctl00_p1` (Password). Any non-empty pair works. Redirects to `/search`. |
| `/logoff` | GET | Clears the session. |
| `/search` | GET, POST | Member search shell. POST performs the lookup; found → redirect to `/member/<id>`, not found → frameset with the no-results page. |
| `/f/search` | GET | Search form (in frame). `?r=none` renders the no-results message. |
| `/member/<id>` | GET | Member detail shell. |
| `/f/member/<id>` | GET | Name, branch, status, savings balance, checking balance. `?denied=1` renders the not-authorized page. |
| `/member/<id>/subaccount/new` | GET, POST | Sub-account form shell. POST validates; failure re-renders the form with a field-level error, success redirects to the confirmation. |
| `/f/member/<id>/subaccount/new` | GET | The six-field form. |
| `/member/<id>/subaccount/confirm` | GET | Confirmation shell. |
| `/f/member/<id>/subaccount/confirm` | GET | Reference number and a summary of the request. |
| `/f/banner` | GET | Banner frame: institution, operator, currently armed injection. |
| `/admin/inject/<mode>` | GET | Arms an injection. `clear` / `none` / `off` disarms. |

### Form fields

Field **names are identical in both tenants** — an automation keying off the
name attribute would pass tenant A and tenant B and still be wrong, because the
labels and the field order differ.

Search: `f_7` member id, `f_3` branch, `f_9` include-closed checkbox.

Sub-account: `ctl00_x3` product (select), `ctl00_x4` nickname, `f_21` initial
deposit, `f_22` funding source (select), `ctl00_x9` statement delivery (select),
`f_24` effective date (`YYYY-MM-DD`).

Natural (non-injected) validation: a product must be selected, the deposit must
parse as a number ≥ 25.00, and the effective date must be non-empty.

## Seed members

| Member ID | Name | Branch | Savings | Checking | Status |
| --- | --- | --- | --- | --- | --- |
| 10001 | Dana Whitfield | 004 | $4,182.55 | $912.40 | ACTIVE |
| 10002 | Marcus Enfield | 011 | $217.03 | $3,540.18 | ACTIVE |
| 10003 | Priya Raghunath | 004 | $26,904.12 | $1,180.77 | ACTIVE |
| 10004 | Oscar Delacroix | 027 | $0.00 | $58.19 | DORMANT |
| 10005 | Ingrid Solheim | 011 | $8,775.00 | $0.00 | ACTIVE |

## Fault injection

Arm a mode in either of two ways:

```bash
curl -b jar -c jar 'http://127.0.0.1:5055/search?inject=perm_denied'   # query param
curl -b jar -c jar  http://127.0.0.1:5055/admin/inject/perm_denied     # console
curl -b jar -c jar  http://127.0.0.1:5055/admin/inject/clear           # disarm
```

The mode is stored in the session and **fires once on the next request for
which it is relevant, then disarms itself**. That arm-once/fire-once behaviour
is what makes recovery paths deterministic to test. The banner frame shows the
currently armed mode.

| Mode | Fires on | What the operator sees | Simulates |
| --- | --- | --- | --- |
| `not_found` | member search | A results page carrying the tenant's no-records message (HTTP 200). | A legitimate business outcome, not an error — the ID simply isn't on file. |
| `perm_denied` | member detail | The tenant's not-authorized message in place of the record. | Entitlement/permission boundaries that vary per operator. |
| `validation` | sub-account POST | Field-level rejection on the deposit amount, form re-rendered with values preserved. | Server-side business rules the client can't predict. |
| `dialog` | any page | A full-page "System Notice" interstitial with a **Continue** button that must be pressed before the requested page appears. | Unexpected modal advisories, maintenance banners, MOTD screens. |
| `timeout` | any page | Session cleared, redirect to sign-on with "Your session has expired." | Idle-timeout mid-task; automation must re-authenticate and resume. |
| `slow` | any page | The response sleeps ~6 seconds, then renders normally. | A loaded mainframe back end; tests timeouts and waiting strategy. |
| `error` | any page | A raw HTTP 500 vendor error page with a fake stack trace. | An unrecoverable server fault. |

`dialog`'s Continue button re-requests the current **path** only — never the
query string — so an injection armed via `?inject=dialog` cannot re-arm itself
into a loop.

## Tenant A vs tenant B

Same vendor product, two institutions. Every difference lives in
`TENANTS` in `config.py`, so the delta is legible in one place. The underlying
flow, routes and field names are identical.

| | Tenant A | Tenant B |
| --- | --- | --- |
| Institution | First Meridian Credit Union | Harborline Savings Bank |
| Chrome colour | navy `#000080` on white text | green `#004000` on yellow text |
| Sign-on heading | Sign on to CoreTeller | Harborline Teller Sign-On |
| **Search field order** | Member ID, Branch Code, Include Closed | **Branch, Member Number, Show Closed** |
| Member id label | Member ID | Member Number |
| Branch label | Branch Code | Branch |
| Search button | Retrieve | Find |
| Balance labels | Savings Balance / Checking Balance | Savings / Checking |
| Sub-account labels | Product Code, Account Nickname, Initial Deposit, Funding Source, Statement Delivery, Effective Date | Account Product, Nickname, Opening Deposit, Source of Funds, Statements, Open Date |
| Submit button | Submit Request | Save |
| No results | `No records located.` | `0 records found.` |
| Not authorized | `You are not authorized to view this record.` | `Access to this account is restricted.` |
| Validation | `Field in error: Initial Deposit must be a numeric amount of 25.00 or greater.` | `Invalid entry: Opening Deposit requires a numeric value of at least 25.00.` |
| Dialog body | scheduled maintenance window | operator advisory queued |
| Confirmation label | Reference Number | Confirmation No. |
| Reference prefix | `FM-YYYYMMDD-NNNNNN` | `HL-YYYYMMDD-NNNNNN` |

Deliberately **identical**: route shapes, frame structure, field `name`
attributes, validation thresholds, seed data, and the `timeout` message.

## Files

* `app.py` — routes, frameset shells, fault injection, app factory.
* `config.py` — the two tenant dicts, seed members, field names, mode list.
* `templates/` — `frameset.html` (the shell), `_chrome.html` (shared window
  chrome), plus one template per page.
* `static/legacy.css` — beveled 1998 house style.
