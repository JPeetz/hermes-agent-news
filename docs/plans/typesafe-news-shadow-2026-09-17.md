# TypeSafe news relevance shadow: implementation plan

Reviewed 17 September 2026. The implementation is now on `feat/typesafe-news-shadow`;
the experiment remains disabled and has not run inference. See the
[implementation guide](../../shadow/README.md) for current commands and qualification status.

Implement the handoff as an internal, bounded comparison service with two entry points: historical selection and discovery of newly published production runs. Start by recovering the filter evidence already preserved in CI. Add explicit capture for future runs, then support full downstream comparisons only when their dependencies are complete. Production continues to use the incumbent filter; no evaluator can promote Jev.

## 1. Verified baseline and changes to the handoff

The checkout was fast-forwarded, at the owner's request, from `4b18b833` to public `origin/main` at `374ad911955411d1bd2cd7238f26f1f5068f2d98`. The pre-existing untracked `tests/hero_image_cost_test.py` is unchanged. The internal mirror is at `2e78f07675e0664bc0ad2fb5ba36990889a4e6ff`; its only tree differences are the internal README and removal of `daily-pipeline.yml`.

The [handoff](../../../typesafe-assessment-2026-09-17/AI-NEWS-SHADOW-TEST-HANDOFF.md) is sound on scope and experimental design. The implementation should incorporate these current findings:

| Finding | Implementation consequence |
| --- | --- |
| Workflow text logs report filter counts, timings, model identity and a reasoning excerpt, but do not systematically log the selected IDs. The diagnostic checkpoint's `_replay.spans` preserves the filter prompt and response. All ten inspected days have a complete successful filter response. | Recover decisions before assuming that historical filter labels are unavailable. Use text logs to corroborate counts and publication identity; use validated captured response data for exact decisions. |
| The public replay stream is size-bounded. In the inspected current files, only September 12 and 13 retain the complete filter text; the corresponding diagnostic checkpoints retain it for all ten days. | Prefer diagnostic checkpoints. Public replay and survivor-only reconstruction are fallback evidence with explicitly narrower capabilities. |
| The filter sees normalized/clipped title and content, source, and IDs. Title and content are each clipped at 300 characters with the existing ellipsis behavior. It does **not** receive the ecosystem grounding added to downstream analysis. | Jev gets the same article evidence. Do not add URL, publication time, full text, release grounding or article-kind routing to this first experiment. |
| `NewsAnalyzer._filter_with_llm()` accepts shortened/prefix IDs; most current IDs are 12 characters. | Preserve incumbent behavior during comparison, but record raw returned IDs and mapped full IDs. Reject ambiguous historical reconstruction. Candidate question-to-article mapping uses exact full IDs in code. Do not combine an incumbent parser fix with this trial. |
| Freshness runs inside `BaseAnalyzer._reduce_phase()` and again after continuity. Its old-anchor path can call an LLM; its article/date path can fetch websites. | Freezing only the orchestrator's Phase 2.7 is insufficient. Inject frozen dependencies into every freshness instance. |
| `analysis.json` is written after continuity and staleness. Continuity includes cross-category curation. | Capture a new pre-continuity Phase 2 snapshot if unaffected categories are to be reused. Do not feed the existing post-continuity checkpoint through that stage again. |
| Current production dispatch explicitly pins configured routes when `anthropic_model` is supplied. Scheduled runs leave provider configuration unchanged. | The handoff's description of overrides is stale. The judge needs an independent client/configuration, never a production-model environment override. |
| September 14–17 inspected replacement/latest runs use DeepSeek V4.1 Flash; earlier inspected runs use GLM 5.3 Flash. | Report results by incumbent model. The mandated DeepSeek judge shares a model family with part of the incumbent cohort; human calibration matters. |
| `scripts/deploy.sh` force-updates internal `main`, replaces README and removes only `daily-pipeline.yml`. | Commit the internally guarded shadow workflow in the upstream source so normal mirroring preserves it. An internal-main-only implementation would be overwritten. Keep result state on a separate internal branch. |
| The existing publication assertion checks that a date's summary exists on `origin/main`. | That is insufficient to bind a new experiment to this run's bytes. Capture the post-rebase pushed commit and compare output hashes. A published Git commit and verified live deployment remain separate facts. |

No tests, pipeline, model inference or workflow dispatch were run in this review. Live checks used GitHub metadata, downloaded artifacts, logs and local data inspection. The complete read-only inventory is [recorded here](evidence/typesafe-news-shadow-inventory-2026-09-17.json).

## 2. Historical availability and cohorts

The inventory covers the latest 50 Daily Pipeline runs through September 17, not the entire retention window. Successful 08:00 UTC schedule-gated runs have no diagnostics; they must not count as published days. Ten artifact archives were downloaded and inspected. All run attempts below are `1`; all retained filter answers map to full IDs present in their input/gathering sets, with no prompt truncation or dropped deltas on the successful call.

| Report date | Source run | Gathered news | Keyword input | Kept / rejected | Classification and proposed use |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-09-08 | [34197560505](https://github.com/flyryan/ai-news-aggregator/actions/runs/34197560505) | 24 | 24 | 21 / 3 | Filter recoverable; research collection partial. Degraded-case coverage only. |
| 2026-09-09 | [34322038673](https://github.com/flyryan/ai-news-aggregator/actions/runs/34322038673) | 71 | 69 | 50 / 19 | Development. |
| 2026-09-10 | [34448162124](https://github.com/flyryan/ai-news-aggregator/actions/runs/34448162124) | 82 | 76 | 60 / 16 | Degraded-case coverage: 12 Reddit subreddit steps partial despite top-level success. |
| 2026-09-11 | [34572717043](https://github.com/flyryan/ai-news-aggregator/actions/runs/34572717043) | 78 | 71 | 60 / 11 | Degraded-case coverage: Reddit collection was empty despite top-level success. |
| 2026-09-12 | [34702681392](https://github.com/flyryan/ai-news-aggregator/actions/runs/34702681392) | 48 | 42 | 32 / 10 | Development; use the healthy replacement, not the earlier run. |
| 2026-09-13 | [34744303131](https://github.com/flyryan/ai-news-aggregator/actions/runs/34744303131) | 30 | 29 | 26 / 3 | Initial historical holdout. |
| 2026-09-14 | [34848516532](https://github.com/flyryan/ai-news-aggregator/actions/runs/34848516532) | 19 | 19 | 15 / 4 | Initial historical holdout; successful replacement run. |
| 2026-09-15 | [34949019524](https://github.com/flyryan/ai-news-aggregator/actions/runs/34949019524) | 75 | 66 | 49 / 17 | Recoverable retry case, but publication later reverted. Separate operational cohort. |
| 2026-09-16 | [35066823382](https://github.com/flyryan/ai-news-aggregator/actions/runs/35066823382) | 66 | 63 | 48 / 15 | Initial historical holdout. |
| 2026-09-17 | [35192960377](https://github.com/flyryan/ai-news-aggregator/actions/runs/35192960377) | 86 | 78 | 51 / 27 | First engineering smoke day; exclude from quality holdout. |

September 15 illustrates why run identity matters: the later run's first filter call failed; `c026` contains its successful 49-ID answer. Its outputs landed in `6a9bc2e6`, then `8f72adf8` reverted them. Current September 15 data represents the earlier run. Never join the later run's decisions to the currently served day's output. The importer may separately qualify the earlier run `34939998488`; that archive was listed but not downloaded in this review.

Implementation-time reconstruction inspected nested source steps and corrected the original cohort classification. September 10 had partial Reddit collection; September 11 had none. The two healthy development dates (September 9 and 12) supply 111 relevance inputs. September 9 was subsequently superseded, but its original publication is verified against the exact pushed commit. The sealed cohort inventory is [recorded here](evidence/typesafe-news-shadow-cohort-2026-09-17.json). The initial three-date holdout supplies 111. This is enough for a pilot, not a production reliability claim. Freeze rubric, thresholds, model revision and cohort membership before scoring holdout quality. Follow with the first seven healthy, complete daily bundles after the freeze; record their actual dates then. A version change starts a new cohort/version rather than re-tuning on the same holdout.

Historical capabilities must be computed, not assigned from file presence alone:

- `filter_replay`: complete original filter request/evidence and final decision, exact IDs, source health, prompt version and successful-call lineage verified. This is supported by the inspected evidence, subject to the importer's canonical-rendering and publication checks.
- `pipeline_replay`: all of the above plus frozen grounding, pre-enrichment release state, correct historical files, effective configuration and every external dependency reached by either replay branch. **No inspected day is yet qualified for this capability.**
- `output_review`: only published output, survivor-only reconstruction or incomplete request/response capture. No rejection-recall claims.
- `unavailable`: required evidence expired, missing, corrupt or ambiguous. Preserve a reason; do not produce zero-valued quality metrics.

Build `import_legacy_bundle.py` to join gathering JSON, successful `news_analyzer.filter` spans, cost records, publication commit and source health. Decode only text deltas, require a normal terminal outcome and complete text length, validate schemas/ID sets and recompute the historical renderer/keyword selection using reviewed versioned adapters. Do not execute historical source code downloaded with an artifact. Preserve failed attempts and their spend without interpreting partial JSON. Recover full prompts where present; never parse reasoning prose into authoritative keep/reject decisions. Truncation, missing input blocks or ambiguous prefix mapping downgrade capability.

## 3. Capture and replay boundaries

Introduce `shadow/capture.py` and `shadow/replay_context.py`. Production capture is an optional observer with bounded writes; its failure marks the experiment unavailable and must not prevent a valid production publication. It does not call Jev, the judge or additional source APIs. Replay, by contrast, fails closed on absent dependencies.

| Boundary / existing file | Smallest intended change |
| --- | --- |
| Workflow, immediately after checkout/config resolution | Record `git rev-parse HEAD` as execution SHA separately from event SHA; emit only allowlisted effective settings. Store pre-run `model_releases.yaml`, original context-cache input and selected historical files before any writes. |
| `EcosystemContextManager.initialize()` / Phase 0 | Capture exact resulting grounding text and resolved model catalog state. In replay, restore these bytes and skip the live catalog refresh. Preserve originally known facts; do not retrofit today's catalog into the historical day. |
| Phase 1 gathering checkpoint | Capture every original `CollectedItem`, including future rejections, collection health, item order, report date, coverage window and timezone. Do not rebuild from `CategoryReport.all_items`. |
| `NewsAnalyzer.analyze()` after keyword filtering | Capture exact ordered semantic-filter input, the full-ID map, renderer/rubric hashes and keyword rule hash. |
| `_filter_with_llm()` after mapping or on exception | Capture raw selected IDs, actual mapped keep/reject set, anomalies, final provider/model, attempts and declared superset fallback. This small decision record is independent of replay-stream caps. |
| `BaseAnalyzer._reduce_phase()` and both orchestrator freshness paths | Supply an evidence-store dependency. Production records exact data needed by the existing date/article lookup path, including observed failures/absence. Replay resolves only captured entries. Preserve existing scoring and old-anchor LLM logic. |
| `agents/continuity/coordinator.py` and freshness history readers | Freeze the two-day continuity history and up to 45 days of old-anchor history, including whether `search-documents.json` or category files were actually used. Record missing historical files explicitly. Reuse the exact historical snapshot even if today's branch contains later edits to old dates. |
| End of Phase 2, before Phase 2.5 | Add `analysis_pre_continuity.json`, with independent serializations of all category reports. Enables reuse of unaffected categories without replaying post-continuity decisions. |
| Phase 4.6 enrichment and Phase 4.7 image | Enrichment writes only to each branch's scratch config. Record its results/cost. Reuse the recorded hero in both branches, or one fixed placeholder if absent; make no image request. |
| Output assembly / `run_pipeline.py` | Reuse JSON, feed and search generators with scratch paths if those artifacts are produced. Export a shared output helper rather than duplicate the pipeline. Deployment, commits and notifications are not runner capabilities. |
| After production push/rebase and verification | Seal a publication receipt binding input-manifest hash, source run/attempt, generated/pushed commit and hashes of this run's report files. Upload a finalized bundle only after the receipt is known. Capture failures remain a separate status from publication success. |

`ReplayContext` carries read-only bundle paths, frozen report/coverage time, history/evidence readers, required capability, per-branch writable roots and an explicit execution policy. `MainOrchestrator` accepts this optional dependency; absent means existing production behavior. Its shadow constructor must not initialize gatherers or an image client. Add an optional relevance strategy only to `NewsAnalyzer`; production has no config route that can select Jev in this delivery.

Do not implement full replay as a bare `--resume-from 2`. That currently refreshes Phase 0 and permits live source access. Add a dedicated `scripts/shadow/run.py` that requires a verified bundle and explicit branch output directory before building the orchestrator. Run control and candidate in separate processes/checkouts with separate checkpoints, caches, global trackers and model-release files. Read-only mount input/config/history; use fresh writable copies for branch state. Prevent symlink escapes and paths outside the scratch root.

Keep all current source-health gates, especially healthy/nonempty Reddit. A missing frozen dependency raises `ReplayIntegrityError`; explicitly propagate it through existing broad exception handlers rather than turning it into a nonfatal freshness/context failure. Allow only configured model endpoints through the runner's network boundary. Source HTTP, OpenRouter catalog discovery, GitHub writes, alert endpoints and deployment access are absent/blocked inside model execution.

A candidate may keep an article that production rejected and consequently request previously uncaptured freshness evidence. This is a real capability gap. Stop/downgrade the downstream comparison; preserve its valid filter comparison. Do not fetch the current page, disable staleness, invent a negative lookup or reuse the incumbent's final freshness label. Additional prospective evidence collection would be a separately specified capture expansion, not an implicit replay fallback.

Reuse unaffected categories from the **pre-continuity** capture when available, with deep copies for each branch. Re-run the joint continuity curator and all dependent cross-category text stages; changed news can legitimately change other categories' curation. With only an old post-continuity snapshot, rerun all categories from original inputs if full replay prerequisites hold. Otherwise remain filter-only.

Run control and candidate on the same pinned replay code and effective downstream settings. Save original production output as a third view. Compare candidate–control for intervention effects and original–control for drift; neither guarantees deterministic prose. Include a second fresh control on September 17 and at least two later pilot days, without sharing generated caches. Record route selection/model identity per call; unexpected route/model changes invalidate the paired attribution or form a separately labelled experiment.

## 4. Versioned contracts and candidate policy

Use JSON Schema/Pydantic contracts with `additionalProperties: false` where appropriate. Keep generated source content inert. A proposed layout is:

```text
run-bundle/
  input-manifest.json                 # immutable input hashes and provenance
  gathered/{news,research,social,reddit}.json
  relevance/{input,incumbent-decision}.json
  context/{grounding.txt,model-releases-before.yaml,effective-config.json}
  context/{resolved-ecosystem.json,history-manifest.json}
  history/...                        # only files read for dates before report date
  evidence/{index.json,objects/...}   # content-addressed lookup evidence/failures
  checkpoints/analysis_pre_continuity.json
  original/{summary,news,research,social,reddit}.json
  diagnostics/{usage.jsonl,timings.json,health.json}
  publication.json                    # final receipt referencing input-manifest hash
  manifest.json                       # sealed bundle inventory and capability result
```

Required manifest fields:

```jsonc
{
  "schema_version": "news-shadow-bundle/v1",
  "source": {
    "repository": "flyryan/ai-news-aggregator",
    "workflow_path": ".github/workflows/daily-pipeline.yml",
    "run_id": 35192960377,
    "run_attempt": 1,
    "event_sha": "<sha>",
    "execution_sha": "<actual checkout sha>",
    "output_commit": "<post-rebase pushed sha>"
  },
  "report_date": "2026-09-17",
  "coverage": {"timezone": "America/New_York", "start": "<recorded ISO>", "end": "<recorded ISO>"},
  "capabilities": {"filter_replay": true, "pipeline_replay": false},
  "missing_dependencies": ["<typed reason and evidence key>"],
  "versions": {"keyword": "<hash>", "renderer": "<hash>", "rubric": "<hash>"},
  "files": [{"path": "<relative path>", "sha256": "<hash>", "bytes": 123}],
  "publication": {"status": "published", "receipt_sha256": "<hash>"}
}
```

Compute `bundle_sha256` over canonical manifest bytes, excluding any self-hash, with every referenced file hashed. Enforce date consistency, ordered unique IDs, complete decision coverage and matching effective-input hashes before paying for any inference. `null`/unknown costs and missing evidence are distinct from zero.

The effective-config allowlist includes resolved provider/route IDs, model IDs, mode, effort/output limits, timeouts, concurrency/retry policy, analysis batch sizes, prompt hashes and freshness switches. Store approved endpoint identifiers without credentials or URL query secrets. Never serialize provider Pydantic objects wholesale: `api_key` is a normal field. Exclude `providers.yaml`, environment dumps, auth stores and raw CI logs from new bundles.

### Jev adapter

Add `agents/relevance/{contracts.py,incumbent.py,typesafe.py}`. Prefer a small `httpx.AsyncClient` adapter with Pydantic response validation and an explicit shared retry budget; `httpx` is already used. This avoids introducing an SDK retry layer underneath the experiment budget. If the SDK is chosen during implementation, pin its version and configure its retries explicitly.

The live [HTTP API](https://docs.typesafe.ai/api.md) accepts `POST https://api.typesafe.ai/v1/systemone`, bearer authentication, `state`, `model` and a `questions` map. Noul answers are `{ "type": "noul", "noul": 0.92 }`; there is no separate Noul confidence field. The current [model documentation](https://docs.typesafe.ai/models.md) identifies `jev-1.13.0` and recommends revision pinning for calibrated policies. Pin that ID for the initial experiment, verify it in preflight and record the returned model on every response. An alias change must not silently change the cohort.

Use shared state containing only the frozen bounded article records. Each article gets an independently worded frontier-relevance Noul and, if retained in the locked development design, an evidence-sufficiency Noul. Instructions explicitly point to the article's array position and define the rubric; question keys such as `r_0007` are only code identifiers. Batch 16 articles initially, reuse connections and allow at most two in-flight Jev calls. Freeze chunking/order with the policy because changing request context is an experimental change.

The relevance proposition uses the current rubric's model/company/product/research/safety/policy/infrastructure scope, with explicit inclusion of controversy and negative AI news. Evidence sufficiency asks whether this bounded evidence supports a relevance judgment, not whether the story is true. No novelty, importance or article-kind decision changes downstream behavior.

Code maps results to `keep`, `reject` or `abstain`; effective selection retains abstentions. Tune `reject_max`, `keep_min` and any `sufficiency_min` on development dates, then freeze them. Require `0 <= reject_max < keep_min <= 1`. A possible conservative development starting point is `reject_max=0.10`, `keep_min=0.80`, `sufficiency_min=0.90`; these are trial settings, not established acceptance thresholds. Missing/invalid/nonfinite/out-of-range answers or exhausted transport retries retain the affected inputs and record `abstain_error` plus degradation. Unexpected question IDs or model revisions invalidate the response. Initial fallback is the superset, avoiding a second semantic decision with hidden cost; an incumbent fallback is a distinct later policy version.

Persist one decision per input: full article ID, evidence hash, raw probabilities, semantic decision, effective keep flag, fallback reason, request ID/hash, requested/returned model, policy/rubric version, attempt references, latency and usage. Do not label retained abstentions as successful Jev classifications. Attribute usage at request level; do not invent precise per-article token charges for a batch.

### Dedicated DeepSeek judge

Add `scripts/shadow/judge.py` with no tools, browsing or shell execution in the model interface. Use exactly `deepseek-v4.1-flash` through RDSec. A dedicated plain OpenAI-compatible HTTP client can use base `https://api.rdsec.trendmicro.com/prod/aiendpoint/v1` plus `/chat/completions`; if reusing `AsyncAnthropicClient`, its base must omit `/v1`. Do not route through production `llm.routes` or override `ANTHROPIC_MODEL`.

CI preflight must verify approved authentication, response/usage schema, actual returned model identity, finish-reason handling and supported structured-output parameters with a small synthetic request. The handoff verified catalog presence, not CI inference access. Do not assume `response_format`, seeds or reasoning controls work until the preflight proves them. Missing access/model, malformed JSON or truncated/error responses produce explicit assessment failure with bounded retries; never substitute another model. Capture the provider-reported model, not merely the requested ID.

Judge contracts:

- `input-adjudication/v1`: `article_id`, `evidence_ids`, `relevance` (`relevant|irrelevant|insufficient_evidence`), evidence sufficiency, rubric category, critical-story flag and concise supported reason. No branch labels, decisions, probabilities or scores in this request.
- `output-comparison/v1`: blinded pair ID; per-dimension `A|B|tie|insufficient_evidence`; supported gained/lost-story and claim findings; severity; article/evidence references; concise reasons; overall `A|B|tie|inconclusive`. Criteria cover important-story coverage, irrelevant inclusions, safety/policy omissions, supported claims, duplicates, ranking usefulness and summary quality. Code checks every reference against the supplied evidence, and validates any quoted span against its source.

For the observed daily volume, adjudicate **all** filter inputs rather than sample. If a future volume/budget forces sampling, include all disagreements and a seeded stratified sample of both kept and rejected agreements; record stratum sizes and inclusion probability per row. Label weighted estimates and uncertainty; never report sample counts as population precision/recall. Blind ordering deterministically from a saved seed, keep the mapping outside judge input, remove provider/branch metadata, and reverse A/B on 10% of output comparisons. Source text has no instruction authority. Full-output assessment may use additional frozen source evidence for claims, but relevance adjudication uses the same bounded filter evidence and remains separately reported.

Have a human calibrate at least 50 development items, balanced across kept/rejected, safety/policy and ambiguous evidence. Expand that set if strata are missing; retain disagreements without overwriting labels. Human review of holdout critical losses and judge disagreements is required before any production proposal. Report judge-estimated quality separately from labelled accuracy/Brier score. The earlier synthetic assessment and its thresholds supply no production error-rate guarantee.

## 5. CI orchestration, storage and budgets

Create `.github/workflows/news-shadow.yml` in the upstream source with a job-level repository guard for `trend-ai-acceleration-task-force/ai-news-aggregator`. Preserve the public-only publishing workflow and mirror removal rule. Confirm the new workflow survives one normal mirror update before enabling its schedule.

Use manual dispatch first. Inputs: source run IDs plus optional explicit attempts, or an inclusive date range; capability mode (`filter|pipeline`); frozen policy/cohort version; and explicit retry/version selection. Reject mixed selectors, invalid dates, arbitrary repositories/URLs and untrusted evaluator refs. The job resolves one trusted evaluator commit and records it. The production source SHA is provenance, not permission to execute downloaded code.

Later enable a 15-minute internal poll, gated by `NEWS_SHADOW_ENABLED`. Both entry points call the same discovery/import/run/compare path. Cross-repository production completion does not directly trigger an internal `workflow_run`. GitHub describes the available [workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows); polling avoids adding a production dispatch dependency.

Discovery paginates completed runs, validates actual pipeline execution, success, source health and publication receipt, and processes source run **and attempt**. Do not advance a single creation-time cursor past incomplete/retried runs: maintain a pending set, overlapping rescan and periodic reconciliation of the retention window. A later revert updates publication status in the index without rewriting the original experiment. Failed/degraded/dry-run/reverted cohorts require explicit selection and remain separate. Healthy transport recovery is allowed and measured; degraded source collection is not silently admitted.

Define identity as a hash of repository + source run/attempt + bundle hash + replay SHA + dependency lock/container digest + replay mode + candidate policy/rubric/model + judge version/model + sampling/repeat plan. Persist step checkpoints with request hashes and typed statuses: `discovered`, `running`, `completed`, `failed`, `incomplete`, `unavailable`; separate comparison capability and source/publication eligibility. Identical discovery deduplicates. Explicit version changes create new identities. Retries resume validated completed steps; repeats have distinct replicate IDs and cannot reuse their inference outputs. Do not claim exactly-once provider billing after an ambiguous timeout.

Store the cumulative compact index and immutable experiment manifests on an internal `shadow-results` branch, leaving internal `main` under the existing mirror. Limit `contents: write` to a separate index-writer job that runs trusted code and validates manifests; evaluator jobs have no Git write/deploy/notification credentials. Use append-only event records, optimistic SHA updates and expired-lease recovery. The mirror currently pushes only `main`, so it should not overwrite this branch; include that in the rollout checks. Full source/response payloads stay in artifacts, not this branch.

Cross-repository authentication is partly known: the existing local approved GitHub identity successfully downloaded the public run artifacts. That does **not** prove internal CI access. The internal repository exposes `APP_ID` and `APP_PRIVATE_KEY` secret names; no shadow environment exists yet. Verify the app's actual installation/permissions before reuse. Artifact access needs authorized source-repository Actions read access. Do not assume the internal default token covers another repository. GitHub documents [token scoping](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token) and [cross-run artifact download requirements](https://docs.github.com/en/actions/tutorials/store-and-share-data).

Use approved protected CI secrets for TypeSafe, replay providers and RDSec. Keep GitHub acquisition credentials in the acquisition job; model credentials enter only the bounded execution jobs. Use `persist-credentials: false`, allowlisted archive extraction with size limits, and no execution/import of bundle files. Existing app permission reuse is a preflight task, not evidence that new grants are required. If no existing approved identity covers the source, resolve the credential with the owner before enabling CI.

Retention choice: explicitly set `retention-days: 90` for source bundles and internal results. The internal repository currently reports both retention and maximum allowed retention as 90 days. The ten inspected production artifacts expire December 7–16, 2026. Preserve archive/artifact digests and canonical bundle hashes when importing internally. This is adequate for the initial cohort, but not a permanent benchmark: choose an approved durable store before any frozen benchmark approaches expiry; retain manifests/index after payload expiry and mark payload availability accurately.

Proposed initial operating limits, to settle before paid CI is enabled:

| Control | Initial limit |
| --- | --- |
| Automatic work | One new date at a time; maximum two historical dates per coordinator invocation; remaining work queued |
| Date-range dispatch | At most 14 dates; one active date initially |
| Concurrency | Separate `news-shadow-coordinator` group; `cancel-in-progress: false`; distinct from production; state recovery tolerates superseded pending scheduler invocations |
| Jev | 16 articles/request, two concurrent requests, 30 s/request, three total attempts, 64 total HTTP attempts/date, 2M input tokens/date |
| Judge | At most two concurrent requests; 120 s/request, three total attempts; 64 requests and 64k aggregate output tokens/date; full adjudication only within this budget |
| Text replay | One branch process at a time initially, paired order alternated by seed; per-route concurrency at most four with matched limits in both branches; 300 calls total including retries/repeat controls; 120 min total execution deadline |
| Proposed monetary envelope | Candidate $0.25, judge $2, text replay/repeats $10 per date; total pilot ceiling $100. These are ceilings to agree, not projected charges or an authorization recorded by this plan. |

Count retries and partial spend against the same enclosing budgets; honor retry headers within the wall-clock deadline. Reserve estimated worst-case tokens/cost before issuing concurrent requests, then reconcile reported usage. A provider with unverified pricing has `cost_usd: null` plus token/request caps until a trusted price schedule is supplied; never apply the incumbent OpenRouter price to RDSec billing. On budget exhaustion, save `incomplete` and the observed costs instead of silently reducing evidence coverage.

On September 17 the recorded incumbent filter itself cost about **$0.006032** and took **156.4 seconds**, with 7,862 input and 8,087 output tokens. Those are this call's recorded measurements/price estimate, not an end-to-end saving forecast. The other categories run concurrently, so filter latency reduction may not shorten the whole pipeline. Report filter-stage savings, changed downstream workload, critical-path duration, judge spend and total evaluation overhead separately.

Each experiment emits:

```text
experiments/<experiment-id>/
  manifest.json
  original/...
  control/{decisions.jsonl,outputs/...,usage.jsonl}
  candidate/{decisions.jsonl,outputs/...,usage.jsonl}
  repeats/...                       # only scheduled fresh controls
  comparison/{metrics.json,sample.json,blind-map.json}
  judge/{requests.jsonl,results.jsonl,usage.jsonl}
  assessment.md
```

Calculate counts, set differences, retention/abstention/failure rates, top-k overlap, paired rank changes, duration and cost in code. Use null for undefined denominators. Mark unavailable output comparisons when only filter replay completed. The report answers which supported important stories changed, whether critical omissions occurred, how reliable Jev was, what the incremental/full-path costs were, and whether to continue the pilot. Add an internal Actions summary linking the manifest, assessment and detailed artifact; no external notification or automatic promotion.

## 6. Delivery sequence and acceptance

| Stage | Deliverables | Exit condition |
| --- | --- | --- |
| 1. Historical importer and contracts | Schemas, `inventory.py`, `import_legacy_bundle.py`, validated recovered decisions, cohort manifest | Reproduce the table above, recover the September 15 final retry correctly, distinguish its reverted publication, and identify incomplete evidence without inference. |
| 2. Capture and dependency seams | Optional capture observer; explicit input/decision/pre-continuity records; frozen context/history/evidence readers; publication receipt | An authorized fresh production day supplies a verified bundle without changing selected IDs or published behavior. Capture failure remains separate from publication. |
| 3. Candidate and judge | Exact-ID strategy contract, Jev adapter, bounded dedicated RDSec judge, schemas, metrics/report renderer | Offline contract checks pass; owner-authorized CI preflight proves both model routes and artifact access. Manual filter comparison completes on September 17. |
| 4. Isolated full runner | Replay execution policy, paired scratch workspaces, unaffected-category reuse, image reuse and dependency enforcement | One fully captured day runs control/candidate through downstream text output; missing counterfactual dependencies correctly stop full replay. Original–control drift and a repeated-control comparison are reported. |
| 5. Coordinator and retention | Internally guarded workflow, index branch, dedup/resume, artifact retention, budget enforcement | Manual run IDs and date ranges use the same path; overlapping discovery/retries do not duplicate completed experiments; workflow and index survive mirror sync. |
| 6. Pilot | Development calibration, locked policy, historical holdout, then seven prospective days | Evidence supports continuing or stopping. Schedule is enabled only after preceding gates. Any production-routing proposal is a separate reviewed change with an owner-agreed critical-story error budget. |

Pipeline-wide replay work and filter-only evaluation can progress independently after contracts are settled. Ship useful historical relevance comparisons first, while capture accumulates future full-replay inputs. Do not block all evaluation on reconstructing dependencies that no longer exist.

Plan focused offline tests in `tests/shadow_*_test.py`; run relevant tests during development. Production pipeline execution requires explicit user authorization. Add only mocked/offline shadow tests to `tests.yml`, never live model evaluation. Required cases:

1. Exact canonical evidence parity, original keyword selection/order, full IDs, duplicate/unknown/ambiguous IDs, empty input, normalized title/snippet boundaries and secret-safe config serialization.
2. Complete/partial stream reconstruction, last successful retry lineage, per-attempt cost retention, truncated prompts, failed calls with partial JSON and survivor-only checkpoints.
3. Historical date/timezone including DST, no history on/after report date, original pre-enrichment release state, unavailable dependencies, and propagation through broad exception handlers.
4. Identical input hashes across branches, immutable source bundles, separate caches/state, joint continuity rerun, unchanged source health, no source HTTP, no hero request, no Git/deploy/alert access or shared-path writes.
5. Jev response schema/model/probability validation, question coverage, abstention/error superset behavior, bounded 429/529/timeouts and usage accounted once across retries.
6. Judge route isolation, single `/v1`, actual response model, malformed/truncated output, fabricated evidence references, prompt injection, blinding, deterministic sampling and order reversal.
7. Schedule no-ops, unsuccessful/degraded source runs, same-date multiple publications, reverts, new run attempts, missing artifacts, resumable budget stops, expired leases and version-aware deduplication.
8. Existing Reddit/publication, analysis identity, routing/transport, freshness/SSRF and replay-secret guards remain intact. Include a meaningful offline production-parity fixture for the default strategy; no broad behavior refactor.

Original planned command surface (see `shadow/README.md` for the implemented command names and current validation status):

```bash
# Read-only discovery and data reconstruction.
python3 scripts/shadow/inventory.py --source-repo flyryan/ai-news-aggregator \
  --from-date 2026-09-08 --to-date 2026-09-17 --out /tmp/news-shadow/inventory.json
python3 scripts/shadow/import_legacy_bundle.py \
  --source-repo flyryan/ai-news-aggregator --run-id 35192960377 --attempt 1 \
  --out /tmp/news-shadow/bundles/35192960377-1

# Focused offline checks.
python3 -m unittest discover -s tests -p 'shadow_*_test.py' -v

# Owner-authorized CI access/model preflight, without source articles.
gh workflow run news-shadow.yml \
  -R trend-ai-acceleration-task-force/ai-news-aggregator --ref main \
  -f operation=preflight

# Manual model evaluation before any schedule is enabled.
gh workflow run news-shadow.yml \
  -R trend-ai-acceleration-task-force/ai-news-aggregator --ref main \
  -f operation=evaluate -f source_runs=35192960377:1 \
  -f mode=filter -f policy_version=news-relevance-v1-dev

# Development dates; pipeline mode requires a qualified full bundle.
gh workflow run news-shadow.yml \
  -R trend-ai-acceleration-task-force/ai-news-aggregator --ref main \
  -f operation=evaluate -f from_date=2026-09-09 -f to_date=2026-09-12 \
  -f mode=filter -f policy_version=news-relevance-v1-dev
```

The full-run gate requires one concrete successful paired experiment, not merely a green workflow. Capture any failed, partial or unpriced work in the report. For pilot continuation, propose zero human-confirmed critical-story losses, no unresolved unsupported judge findings, acceptable measured abstention/fallback volume and a measured cost, latency or editorial benefit beyond control variation. The owner must settle the production error budget before any promotion discussion; three historical holdout dates cannot establish that budget empirically.

## 7. Remaining owner decisions and preflight checks

These do not prevent implementing offline capture, contracts, recovery and mocked clients:

- **CI access:** which existing approved noninteractive credentials supply source artifact access, replay providers, TypeSafe and RDSec? Secret names/app presence are known; installation scope, inherited organization secrets and model entitlement remain to be checked. No credentials should be entered in chat.
- **Spending:** accept or adjust the proposed pilot envelope and confirm how RDSec usage is priced. Token/time/request caps remain mandatory even where dollar pricing is unknown.
- **Editorial review:** name the human reviewer and agree the critical-story definition/error budget. Development thresholds are learned and frozen; synthetic thresholds are not carried forward as acceptance criteria.
- **Long-term retention:** only needed before promoting this into a benchmark that must outlive 90-day artifacts. Select an existing approved durable store rather than silently creating new infrastructure.

Historical recovery and the implementation through Stage 5 are present locally.
The next qualification step is an offline test run, followed by an
explicit synthetic access probe and engineering filter experiment. Full replay
requires a new dependency-complete production capture; the ten historical bundles
qualify only for filter replay. Capture, CI gates and production routing remain unchanged.
