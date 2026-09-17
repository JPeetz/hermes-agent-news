# News relevance shadow experiment

This package compares the original news relevance decision, a fresh incumbent
control, and pinned TypeSafe `jev-1.13.0`. A separate, no-tools
`deepseek-v4.1-flash` judge uses the RDSec endpoint. It never promotes a candidate
or publishes report output. Ordinary production runs retain the incumbent route.

## Current qualification

The September 8–17 historical artifacts were reconstructed without inference.
All ten bundles support **filter replay**. None contain the dependencies needed
for a full pipeline replay. The metadata-only inventory is in
`docs/plans/evidence/typesafe-news-shadow-cohort-2026-09-17.json`; local source
bundles are ignored under `data/shadow-review/cohort/<run-id>-1/`.

Healthy development dates: September 9 and 12 (111 inputs). Initial holdout:
September 13, 14, and 16 (111 inputs). September 17 is the engineering sample
(78 inputs). September 8, 10 and 11 have degraded collection; September 15 has a
verified retry followed by a publication revert. These belong to recovery cases.
September 9 is superseded, with its original publication independently verified.

Only syntax/static review and read-only reconstruction have been performed.
The repository owner runs the tests and authorizes live pipeline/model execution.
No schedule, capture variable, credential, workflow dispatch or production route
was changed while implementing this package.

## Commands

Run from the repository root with the existing Python dependencies installed.
Secrets come from the existing environment or protected CI secrets; never place
them in command arguments, policy JSON, artifacts or logs. Production `.env`
and `config/providers.yaml` are not loaded by the isolated evaluator.

```bash
# Read-only import through existing gh access, with an explicit run attempt.
python3 scripts/shadow/import_legacy_bundle.py \
  --run-id 35192960377 --attempt 1 \
  --out data/shadow-review/bundles/35192960377-1-new

# Check configuration and sealed input integrity; this makes no model calls.
python3 scripts/shadow/preflight.py \
  --bundle data/shadow-review/cohort/35192960377-1

# Owner-run offline test suite (not executed during implementation).
python3 -m unittest discover -s tests -p 'shadow_*_test.py' -v

# Explicit paid access probe using synthetic records only.
python3 scripts/shadow/preflight.py --require-credentials --probe-models

# Explicit paid filter experiment after credentials and tests are qualified.
python3 scripts/shadow/run.py \
  --bundle data/shadow-review/cohort/35192960377-1 \
  --out data/shadow-review/experiments \
  --policy config/shadow/news-relevance-v1-dev.json \
  --mode filter --cohort engineering --repeat-control
```

Required configuration:

| Name | Purpose |
| --- | --- |
| `TYPESAFE_API_KEY` | Candidate only |
| `RDSEC_API_KEY` | Dedicated pinned judge |
| `SHADOW_INCUMBENT_MODEL` | Explicit control model, matching the captured incumbent |
| `SHADOW_INCUMBENT_API_KEY` | Control credential; falls back to `RDSEC_API_KEY` |
| `SHADOW_INCUMBENT_BASE_URL` | Defaults to `https://api.rdsec.trendmicro.com/prod/aiendpoint/v1`; the other reviewed route is `https://openrouter.ai/api/v1` |

Model ID prefixes may differ between RDSec and OpenRouter; the model revision
must match. The selected route is part of experiment identity and the actual
returned model is checked by the dedicated adapters. Full downstream replay
uses the production client, whose configured model is not independent proof of
the server's returned model. Judge/model-family overlap is disclosed.

`preflight.py` without `--probe-models` proves integrity and the presence of named
configuration only. It explicitly reports that model access has not been verified.

## Policy and evidence

The checked-in policy is deliberately **not frozen**. Development uses
conservative reject/keep/sufficiency thresholds, with abstention and errors
retaining the item. Noul probabilities are preserved directly; there is no
invented confidence score. Holdout and prospective execution require a versioned
policy with `frozen: true`. Prospective runs also require `prospective_start_date`
set to a report date after policy freeze and after the historical cohort. Change
the version and policy hash when changing any
threshold, rubric, model revision, chunking or cohort definition.

Each experiment gets a new immutable identity derived from source run/attempt,
bundle hash, evaluator commit and actual source-file hashes, policy, model route,
mode and repeat settings. Existing experiment directories cannot be overwritten.
Every retry consumes a request reservation; unreported usage/cost stays unknown.
Terminal failures are not retried automatically. Use a new `--retry-version`
(or the matching manual CI input) for an explicitly requested fresh attempt.
The quoted TypeSafe input price is an estimate, not an invoice. Historical
aggregate cost estimates and unknown failed-attempt costs remain separate.

Reports distinguish original-to-control drift from candidate changes, report
abstentions and missing evidence, and label quality as judge-estimated. Human
review is required for any critical losses, judge disagreements or promotion
proposal. A successful experiment never changes production routing.

## Prospective capture and downstream replay

Production capture is off unless the publishing repository variable
`NEWS_SHADOW_CAPTURE_ENABLED` is `true`. The optional observer records the exact
filter request/decision, gathered items, pre-continuity reports, pre-run history,
grounding, prompt/configuration versions and freshness responses already fetched
by production. It adds no source fetch, candidate call or judge call.

After publication, `publish_capture.py` compares all five report files against
the actual pushed Git commit before sealing a receipt. Capture/finalization and
artifact upload are best effort and cannot block publication. Bundles and existing
diagnostics have 90-day retention. An incomplete capture cannot become eligible
merely because the run is green or a same-date report exists.

Use `--mode pipeline` only with a verified `pipeline_replay` bundle. Each branch
runs in its own process and scratch directories, reuses unaffected original
pre-continuity reports and the original hero, and reruns joint continuity and
downstream text stages. Frozen history and HTTP evidence are mandatory. If a
candidate retains a previously rejected item whose freshness URL was never
captured, the downstream comparison becomes unavailable while its completed
filter comparison remains useful. It never fetches a present-day page or turns
off freshness checks to fill the gap.

The evaluator's Python transport guard permits HTTPS POST only to explicitly
configured model endpoints, rejects redirects/proxies and guards DNS/socket
destinations. It is a process-level guard, **not an operating-system sandbox**.
Workers receive only explicit model/runtime environment variables, and downloaded
artifacts are data, never imported source code.

## Internal automation

`.github/workflows/news-shadow.yml` runs only on `main` in
`trend-ai-acceleration-task-force/ai-news-aggregator`. It is tracked upstream so
the existing mirror deployment does not delete it; an exact repository guard
keeps it inactive in the public publishing repository. Durable coordinator state
lives on the separate internal `shadow-results` branch.

`NEWS_SHADOW_ENABLED=true` enables discovery. `NEWS_SHADOW_EVALUATE_ENABLED=true`
separately enables paid scheduled evaluation with `NEWS_SHADOW_POLICY_PATH`
pointing to a frozen policy. `NEWS_SHADOW_MODE` selects `filter` (default) or
`pipeline`; a pipeline job admits only bundles with that declared capability.
The pilot stops at seven completed healthy prospective dates for the same
policy/cohort/retry version. Each invocation runs at most two acquired dates
serially. A deterministic 20% date sample repeats the incumbent control; manual
dispatch also offers `repeat_control`. Failed/incomplete results require an
explicit `NEWS_SHADOW_RETRY_VERSION` change to run again.

Source acquisition uses read-only access to the publishing repository and seals
the data before passing it to the evaluator. Model credentials are scoped to the
evaluation job. Only the result writer can write the internal results branch;
it does not receive model or source credentials. Existing App installation access
or a protected source token must be verified; an internal repository token alone
does not establish cross-repository access.

Keep discovery/evaluation gates off until offline checks, explicit access probes,
the engineering filter experiment and policy freeze are complete. CI dispatch,
successful inference, a captured production day, and seven healthy prospective
evaluation days are separate qualification steps, not implied by the code landing.
