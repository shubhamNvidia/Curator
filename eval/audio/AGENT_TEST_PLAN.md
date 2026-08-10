# Agent End-to-End Test Plan — Audio Curation Agent

Status: living document. Companion to the runnable harness in this directory
(`run_eval.py`, `queries.yaml`, `scenarios/`, `taxonomy.yaml`, `trace_check.py`,
`judge.py`, `fixtures/`, `run_all.sh`). This document is the human-facing test
strategy; the harness is the machine-facing enforcement of it.

> Do not evaluate only "did the pipeline run." Evaluate whether the agent chose
> the **appropriate, complete, compatible, correctly-configured, efficient,
> recoverable, and explainable** pipeline for the user's requirement.

---

## 1. Test strategy

### 1.1 The agent is two planes

| Plane | What it is | What it owns | How we test it |
| --- | --- | --- | --- |
| **LLM plane** | The host model (Claude / Cursor) following `nemo_curator/audio_agent/skills/audio-curation/SKILL.md` | Intent understanding, clarification, capability & module selection, ordering rationale, outcome→parameter decisions, explanations, recovery loop | Golden tool-call **trace assertions** (`trace_check.py`), **LLM-as-judge** (`judge.py`), and **human review** |
| **Deterministic core** | `nemo_curator.audio_agent` verbs + `nemo_curator.stages.audio` contracts | Validation codes, `resolve`, `verify`/honesty, `plan_continuation`, planner feasibility, guardrails, card gate | Fully automated in `run_eval.py` (`queries.yaml`) — GPU-free |

The LLM proposes; the deterministic core disposes. Most "agent got it wrong"
bugs surface as either (a) a wrong *decision* by the LLM plane that the core
would have caught but the LLM ignored, or (b) a correct LLM decision that the
core mis-graded. The plan tests both directions.

```
user prompt ──▶ LLM plane (SKILL.md loop) ──tool calls──▶ deterministic core verbs
                     │                                          │
                     └────────────── JSON verdicts ◀────────────┘
                     ▼
              final Recipe IR ──▶ smoke / run on real data ──▶ report / verify
```

### 1.2 North-star scorecard (7 dimensions)

Every scenario is graded on these, not just execution:

1. **Appropriate** — selected the best-fitting module(s) among alternatives.
2. **Complete** — no required processing stage missing.
3. **Compatible** — every stage's I/O contract is satisfied by upstream.
4. **Configured** — right params & metadata keys; thresholds *resolved* (never hand-picked).
5. **Efficient** — reuses prior work; picks the right execution mode; no redundant stages.
6. **Recoverable** — converges to a runnable pipeline after a fault within the loop budget.
7. **Explainable** — rationale is grounded in card facts and verdicts, not invented.

Execution success is **necessary but not sufficient**.

---

## 2. Test layers

| Layer | Scope | Mechanism | GPU |
| --- | --- | --- | --- |
| L-Unit | one agent decision | single verb (`resolve`, `find_producers`/`unproducible`, one-issue `validate`, `honesty_review`, `plan_continuation`, `_safety.*`) | no |
| L-Component | planning / selection / config / validation | multi-stage `validate` → `Verdict.status` + issue codes; `card_conformance.audit`; planner mode/feasibility | no |
| L-Integration | real modules, structure only | `build_stages` + `validate` + `planner.plan` on real contracts / `EnvProfile` | no |
| L-E2E | real datasets | `smoke`→`run`→`report`→`verify` on FLEURS (Ray :6457, RTX 4090) | yes (opt-in) |
| L-Regression | existing behavior | deterministic queries/card gate/snapshots plus paired host semantic-intent traces | no* |
| L-Stress/scale | large / many-stage | big synthetic manifests; many-GPU-stage recipes forcing `batch`/infeasible; throughput via `calibrate` | mixed |
| L-Adversarial/edge | hostile / boundary | conflicting/impossible prompts, goalpost-moving, path traversal, secret/transcript leakage, `tensor_into_sink`, composite blind spots | no |
| L-Human | reasoning & quality | rubric review on a sampled subset | n/a |

The default harness run (`python -m eval.audio.run_eval`) covers L-Unit,
L-Component, L-Integration, L-Regression, and the deterministic parts of
L-Adversarial — all GPU-free. L-E2E is opt-in via `AUDIO_AGENT_EVAL_EXECUTE=1`.

---

## 3. Failure classification

Machine-readable form: `taxonomy.yaml`. Each class maps to a **detection stage**
(`intent | clarify | plan | select | order | param | validate | resolve |
execute | verify | continue`), a **real code or "LLM-plane"** signal, and a
**default severity** (P0–P3, see §8).

| # | Category | Detection stage | Real signal (code / verb result) | Sev |
| --- | --- | --- | --- | --- |
| 1 | User-intent misunderstanding | intent | LLM-plane (trace/judge) | P1 |
| 2 | Missing clarification | clarify | LLM-plane (asked vs should-have) | P1 |
| 3 | Incorrect capability selection | select | LLM-plane + `unproducible_role` | P1 |
| 4 | Missing required capability | plan | `unsatisfied_reads`, `unproducible_roles`, `missing_output_producer` | P1 |
| 5 | Incorrect module selection | select | LLM-plane + card mismatch | P1 |
| 6 | Incorrect module ordering | order | `unsatisfied_reads`, `key_removed_upstream`, `task_type_mismatch` | P1 |
| 7 | Input-output contract mismatch | validate | `task_type_mismatch`, `dangling_key`, `tensor_into_sink` | P1 |
| 8 | Invalid parameter value | param | `bad_params`, `card_batch_size`, `card_max_speakers` | P1 |
| 9 | Missing required parameter | param | `bad_params` / unresolved `"REQUIRED"` placeholder | P1 |
| 10 | Metadata-key mismatch | param | `dangling_key` | P2 |
| 11 | Pipeline validation failure | validate | `Verdict.status=="fail"` | P1 |
| 12 | Runtime execution failure | execute | `run.status=="failed"`, smoke `errors[]` | P1 |
| 13 | Partial pipeline failure | execute | `report.rejected>0` / smoke `goals_met=false` | P2 |
| 14 | Incorrect output | verify | `verify` `not_met` / `missing_output_producer` | P1 |
| 15 | Unnecessary module execution | plan | LLM-plane extra-stage delta vs golden | P2 |
| 16 | Re-execution of completed work | continue | `plan_continuation` should be `incremental`/`already_done` | P2 |
| 17 | Poor error handling | any | missing `fix`/`escalate_to` hint; harness-error wrapper | P2 |
| 18 | Failed recovery | plan | validate loop (≤3) never reaches `runnable` | P1 |
| 19 | Unsupported requirement | intent | refuse/redirect (`no_stage_for`) | P1 |
| 20 | Infrastructure/environment failure | execute | `gpu_unavailable`, `ffmpeg_missing`, planner `feasible=false`/`escalations`, Ray down | P2 |
| 21 | Dataset-related failure | execute | `DataProfile.unreadable`, empty manifest, sample-rate mismatch | P2 |
| 22 | Non-deterministic behavior | any | `config_hash`/verdict differs across repeats | P0 |
| 23 | Performance/scalability failure | execute | planner `mode`/throughput regression | P3 |

Guardrail breaches (path traversal, secret/transcript leak, silent full-scale
run, accepted goalpost-moving) are always **P0** regardless of category (§8).

---

## 4. Evaluation metrics

| Metric | Definition | Source |
| --- | --- | --- |
| Intent-understanding accuracy | matches(parsed task/domain/outputs vs golden) / scenarios | trace + judge |
| Capability precision / recall | selected vs golden capability roles (stage `writes` roles) | trace |
| Module-selection accuracy | final stage-set vs golden, with equivalence classes (UTMOS~SIGMOS~SQUIM) | trace |
| Pipeline completeness | 1 − (missing_output_producer or unsatisfied output roles) | validate |
| Pipeline validity | share with `runnable==true` / `status=="pass"` | validate |
| Parameter accuracy | resolved params == golden **and** every threshold has a `config_strategy` entry | resolve + trace |
| Execution success rate | `run.status=="completed"` | E2E |
| Unnecessary-stage rate | extra stages beyond golden minimal / total | trace |
| Clarification quality | precision/recall of "asked exactly when needed" + rubric score | trace + judge |
| Recovery success rate | injected-fault scenarios reaching `runnable` within ≤3 iters | trace |
| Cost / efficiency | planner `estimate` + continuation reuse rate (`reuse_stages`/total) | planner + continuation |
| Determinism | identical `config_hash`/`resolve`/verdict across N repeats | regression |
| Generalization | pass rate on held-out prompts + injected novel card (level 14) | scenarios + fixtures |
| Failure-detection accuracy | seeded faults the harness flags / seeded | harness |
| Failure-classification accuracy | assigned category == ground truth | taxonomy |
| Root-cause ID accuracy | correct root cause / failures | report review |
| Diagnostic completeness | % failures with a complete report (§7.2) | report |
| MTTI | mean wall-time to identify the failure stage | process |
| MTTR | mean wall-time to reproduce + resolve | process |

---

## 5. Test matrix — 14 complexity levels

Deterministic anchors live in `queries.yaml`; full behavioral scenarios (with
clarification, selection, ordering, explanation rubric) live in `scenarios/`.

| Lvl | Theme | Representative prompt | Expected shape (real stages) | Primary signal |
| --- | --- | --- | --- | --- |
| 1 | Single-stage | "How long is each clip?" | `GetAudioDurationStage` (+ reader/sink) | `status=pass` |
| 2 | Standard multi-stage | "Curate clean read speech by quality" | Reader → `MonoConversionStage` → `UTMOSFilterStage` → `TimestampMapperStage` | `runnable`, `resolve` used |
| 3 | Branch / fan-out / segment / aggregate | "Per-segment quality + speakers on long-form" | Reader → `VADSegmentationStage`(fan-out) → `InferenceSortformerStage` → `UTMOSFilterStage`; ALM: `ALMDataBuilderStage`→`ALMDataOverlapStage` | fan-out cardinality |
| 4 | Ambiguous → clarify | "Give me good audio" | ask quality level before `resolve` | clarification asked |
| 5 | Incomplete | "Filter by WER" (no transcripts) | `unproducible_role: reference_text`; ask/insert ASR | unproducible role |
| 6 | Conflicting / impossible | "Keep 100% but only studio quality" | escalate / refuse | no valid recipe |
| 7 | Multiple valid designs | "quality filter" | `UTMOS` vs `SIGMOS` vs `TorchSquim` — recommend + allow override | both validate |
| 8 | Incremental | "also add transcripts" (after mono+UTMOS run) | `plan_continuation` `incremental`, run only `InferenceAsrNemoStage` | reuse |
| 9 | File vs in-memory handoff | "…and export JSONL" | `ManifestReader`/`ManifestWriterStage` vs waveform-resident chain | residency, no false `tensor_into_sink` |
| 10 | Configurable metadata keys | "score is under `mos`, filter on that" | correct `*_key` wiring; mismatch → `dangling_key` | key alignment |
| 11 | Incompatible modules | "convert to a document then measure audio duration" | `AudioToDocumentStage`→AudioTask stage → `task_type_mismatch`; keep-waveform→writer → `tensor_into_sink` | hard error |
| 12 | Capability unavailable | "label emotion / accent" | refuse (`no_stage_for`) | refusal |
| 13 | Failure scenarios | invalid params, missing deps, runtime, partial | `bad_params`, `card_max_speakers`, `missing_secret`, smoke `errors`, `rejected>0` | seeded fault |
| 14 | New unseen module/card | "use the NewFooStage" (card added post-hoc) | discover via `discover`/`cards`; conform via `check_card` | generalization |
| 15 | Mechanically valid, semantically wrong | paired parent/child, segment/recording, aggregate/row, transform-use, and key-selection intents | same valid topology/config family; host critic must derive the intent-specific order/params | deterministic evidence packet + structured host critique + model judge |

\* Level 15's checked-in scenario definitions and trace-grader unit tests are
CPU-only. Capturing fresh host traces requires an LLM endpoint.

---

## 6. Harness architecture (this directory)

| File | New/extend | Role |
| --- | --- | --- |
| `AGENT_TEST_PLAN.md` | new | this document |
| `taxonomy.yaml` | new | code → {category, stage, severity, plane} for auto-classification |
| `queries.yaml` | extend | deterministic-core cases (every code, guardrails, planner); existing 32 preserved |
| `run_eval.py` | extend | `safety`/`plan`/`execute` branches, `branch`/`category` tagging, per-category aggregation, `--report` JSON emitter, skip support |
| `scenarios/` | new | LLM-plane scenarios `L01..L14_*.yaml` with the full per-case schema |
| `trace_check.py` | new | assert a captured agent trace (ordered tool calls + final recipe) against a scenario |
| `judge.py` | new | LLM-as-judge rubric scorer (+ deterministic fallback) |
| `agent_runner.py` | new (optional) | capture traces via Cursor SDK; manual paste fallback |
| `fixtures/` | new | manifest variants + held-out novel stage/card + FLEURS pointer |
| `reports/` | new | emitted failure reports + dashboard JSON |
| `run_all.sh` | new | orchestrate deterministic → optional trace/judge → optional E2E → report |

### 6.1 How trace capture works (LLM plane)

Assertion is decoupled from capture, so the harness is usable with or without
model access:

- **Automated**: `agent_runner.py` drives the host agent via the Cursor SDK on a
  scenario prompt, recording every `nemo_curator.audio_agent` tool call and the
  final recipe into a trace JSON.
- **Manual**: run the agent in-IDE on the prompt, then paste the ordered tool
  calls + final recipe into the scenario's `observed:` block (or a sidecar
  `traces/<id>.json`). `trace_check.py` grades either source identically.

A trace is:

```json
{
  "scenario_id": "L04_ambiguous_quality",
  "tool_calls": [
    {"verb": "context", "args": {...}},
    {"verb": "cards", "args": {"category": "quality"}},
    {"clarification": "studio, general, or lenient?"},
    {"verb": "resolve", "args": {"stage": "UTMOSFilterStage", "label": "studio"}},
    {"verb": "validate", "args": {...}, "result": {"runnable": true, "status": "pass"}}
  ],
  "final_recipe": {"stages": [ ... ]}
}
```

Level-15 semantic regressions additionally carry:

```json
{
  "semantic_critique": {
    "mechanically_runnable": true,
    "intent_status": "pass",
    "stage_reviews": [
      {
        "stage": "MetricStage",
        "finding": "This stage implements the requested metric operation.",
        "evidence": ["goal:user-request", "card:MetricStage", "recipe:MetricStage"]
      }
    ],
    "field_reviews": [],
    "behavior_checks": [],
    "transform_checks": [],
    "model_checks": [],
    "assumptions_or_questions": []
  }
}
```

The validate tool result must separately contain the deterministic
`semantic_review` evidence packet. `trace_check.py` requires card inspection, a
green validation of the exact final recipe, required critique sections,
substantive findings, resolvable citation tokens, and the intent-specific recipe
consequence declared in scenario YAML. It contains no field- or module-specific
semantic rules. `judge.py --model ...`/human review remains responsible for
whether the critique's meaning is correct.

---

## 7. Standard templates

### 7.1 Per-scenario definition (`scenarios/*.yaml`)

```yaml
- id: L02_readspeech_quality
  level: 2
  label: standard
  objective: "Curate clean read speech by quality with a resolved threshold."
  prompt: "I have a folder of WAVs, give me cleaner speech data."
  dataset: "folder of 48kHz WAVs, no transcripts"
  prerequisites: []
  expect_clarification: ["which quality level (studio/general/lenient)?"]
  expected_capabilities: [ingest, preprocess, quality, filter, export]
  expected_modules:
    required: [MonoConversionStage, UTMOSFilterStage]
    equivalence: {quality: [UTMOSFilterStage, SIGMOSFilterStage, TorchSquimQualityMetricsStage]}
    forbidden: [InferenceAsrNemoStage]   # no transcripts requested
  expected_order: [ingest, MonoConversionStage, quality, TimestampMapperStage]
  expected_params:
    must_resolve: [mos_threshold]        # never hand-picked; must come from `resolve`
  expected_structure: linear
  expected_validation: {status: pass, runnable: true}
  expected_execution: {mode: streaming, accepted_gt: 0}
  expected_output_roles: [score, duration]
  explanation_rubric:
    - "names the quality metric and what it does/doesn't capture"
    - "states the resolved threshold came from a named outcome (studio/general/lenient)"
    - "explains why no ASR stage is present"
  pass_criteria: "final recipe validates; quality module present; mos_threshold resolved; no ASR"
  fail_criteria: "hand-picked threshold; missing quality; unrequested ASR; validation fails"
```

### 7.2 Failure report (auto-emitted per failing case → `reports/`)

```yaml
case_id: ...
prompt: ...
generated_recipe: {...}
plan_and_capabilities: {...}
selected_modules_and_params: {...}
expected_behavior: ...
actual_behavior: ...
failure_stage: validate            # intent|clarify|plan|select|order|param|validate|resolve|execute|verify|continue
failure_category: "Input-output contract mismatch"   # from taxonomy.yaml
plane: deterministic               # deterministic | llm
error_message: ...
stack_trace: ...                   # when available
root_cause: ...
deterministic: true                # true|false (intermittent)
severity: P1
impact: ...
repro_steps: [...]
artifacts: {recipe: ..., logs: ..., report: ...}
recovery_attempted: false
recovery_succeeded: null
corrective_action: ...
owner_component: nemo_curator.audio_agent.checks
status: open                       # open|under_investigation|fixed|blocked|accepted_limitation
```

### 7.3 Dashboard (`reports/dashboard.json`)

```json
{
  "generated_at": "...",
  "totals": {"executed": 0, "passed": 0, "failed": 0, "partial": 0, "blocked": 0, "skipped": 0},
  "by_category": {},
  "by_severity": {"P0": 0, "P1": 0, "P2": 0, "P3": 0},
  "by_stage": {},
  "by_module_failure_rate": {},
  "top_root_causes": [],
  "recovery": {"attempts": 0, "successes": 0},
  "regression_failures": [],
  "perf_bottlenecks": [],
  "open_defects": [],
  "accepted_limitations": []
}
```

---

## 8. Defect prioritization & tracking

**Priority = severity × frequency × blast-radius.**

- **P0** — guardrail breach (path traversal, secret/transcript leak, silent
  full-scale run, accepted goalpost-moving) **or** a `must` criterion falsely
  reported `met` **or** non-determinism in `config_hash`.
- **P1** — a wrong or invalid pipeline that nonetheless validates, or a valid
  pipeline the core wrongly rejects; missing capability; failed recovery.
- **P2** — suboptimal-but-valid (extra stage, missed reuse, minor key mismatch).
- **P3** — cosmetic, explanation-only, or performance-only regressions.

**Lifecycle:** `open → under_investigation → (fixed | blocked | accepted_limitation)`.
Each transition is recorded with timestamps to compute MTTI / MTTR. The dashboard
groups open defects by severity and category.

---

## 9. Coverage goals & acceptance thresholds

Coverage goals:

- Every documented issue code has ≥1 positive and ≥1 negative case.
- Every one of the 14 levels has ≥3 scenarios in `scenarios/`.
- Every failure category (§3) has ≥1 seeded case.

Acceptance thresholds ("ready to ship a change"):

- Deterministic harness pass-rate **= 1.0**; card gate **44/44** (or current N/N).
- LLM-plane structural trace-assertion pass **≥ 0.90** (proposed).
- Recovery success rate **≥ 0.80** (proposed).
- Determinism **= 1.0** on repeated `config_hash` / `resolve` / verdict.
- **Zero P0** open defects.

Thresholds marked "proposed" are tunable once a baseline exists.

---

## 10. Regression strategy

- The existing `queries.yaml` cases + card gate are the frozen regression floor;
  `run_all.sh` fails if the deterministic pass-rate drops below `--min-pass-rate`.
- Snapshots (`reports/snapshots/`) pin `config_hash` for library recipes,
  `resolve` outputs for the metrics cards, and `Verdict.status` for a canonical
  set of recipes. A diff against a snapshot is a regression candidate.
- Any newly fixed bug earns a permanent case (positive + negative) so it can
  never silently regress.

---

## 11. Phased execution plan

1. **P1 Foundation** — this doc + `taxonomy.yaml`; extend `queries.yaml` /
   `run_eval.py` (branches + reporter). Runnable, GPU-free.
2. **P2 LLM plane** — `scenarios/`, `trace_check.py`, `judge.py`, optional
   `agent_runner.py`; capture a few real traces to validate the assertions.
3. **P3 Integration + E2E** — `fixtures/` + reuse FLEURS/GPU; run a small E2E
   subset as a harness smoke (full matrix deferred).
4. **P4 Regression + report** — snapshots, `run_all.sh`, a sample populated
   final report + dashboard.

---

## 12. Final report shape (produced by P4 / `run_all.sh --report`)

- Totals: passed / failed / partially-passed / blocked / skipped.
- Failures grouped by category and severity.
- Most common root causes.
- Modules and agent components with the highest failure rates.
- Recovery attempts and their success rate.
- Regression failures vs the frozen floor.
- Performance bottlenecks (planner mode changes, throughput).
- Open defects and accepted limitations.
- Recommended fixes and next testing priorities.
- **Recurring-pattern log** → architectural-improvement recommendations.

---

## 13. Scope & assumptions

- Target = the **audio** curation agent only (`nemo_curator/audio_agent`,
  `eval/audio`, the audio cards).
- L-E2E reuses the existing FLEURS / RTX 4090 / Ray :6457 setup.
- The harness is built and runnable, but the full GPU campaign is out of scope
  for this pass; E2E fixtures + runbook are prepared and smoke-tested on a small
  subset.
- LLM-plane automation via the Cursor SDK is optional; trace assertions run on
  captured traces regardless of how they were captured.

---

## 14. How to run

```bash
# deterministic suite (GPU-free) + card gate, with a JSON report
python -m eval.audio.run_eval --report eval/audio/reports/latest.json

# LLM plane: fresh isolated traces + complete authoritative Level-15 gate
export CURSOR_API_KEY=cursor_...
bash eval/audio/run_all.sh --llm
# Existing-trace fallback is deliberately diagnostic and cannot certify semantics:
python -m eval.audio.aggregate_traces --non-llm \
  --report eval/audio/reports/llm_plane_diagnostic.json
# (grade a single captured trace)
python -m eval.audio.trace_check --id L04_good_audio --trace eval/audio/traces/L04_good_audio.json
python -m eval.audio.judge --id L04_good_audio --trace eval/audio/traces/L04_good_audio.json

# GPU/real-data E2E (reuses /tmp/aa_real; Ray on :6457)
export RAY_ADDRESS=127.0.0.1:6457 AUDIO_AGENT_WORKSPACE=/tmp/aa_real
python -m eval.audio.run_e2e --sample 4 --report eval/audio/reports/e2e.json

# end-to-end user-simulation: a 2nd LLM plays the user (multi-turn), the agent builds
# a recipe, and that recipe is run on real FLEURS; then the inside-out debug report
export MODEL_SUT=claude-opus-4-8 MODEL_USER=composer-2.5
python -m eval.audio.simulate_user                      # all ~26 personas (see personas.yaml)
python -m eval.audio.simulate_user --group pattern      # just the composition-pattern personas
python -m eval.audio.debug_report                       # reports/debug_report.md (inside-out)

# merge all planes into one report
python -m eval.audio.final_report

# everything at once (toggles): deterministic -> generalization -> selftests ->
# snapshot -> (opt-in) LLM plane -> (opt-in) E2E -> (opt-in) user-sim -> debug + final report
bash eval/audio/run_all.sh --llm --e2e --sim
```

## 15. Layer: end-to-end user-simulation (personas)

A second LLM plays a realistic user (natural, first-person prompts in
[personas.yaml](personas.yaml); no stage names), answering the agent's clarifying
questions from `persona_facts`/`preferences`. The curation agent (Opus 4.8) builds a
recipe over a multi-turn conversation; that recipe is then executed on real FLEURS
(smoke -> run -> report -> verify). Each persona is graded on: success kind
(executes/refuses/clarifies), recipe validity, module selection, and composition
ordering (`order_constraints` sourced from `patterns/composition.yaml` -
enforced rules fail, advisory deviations become findings). Gaps/bugs are recorded in
`reports/findings.json` and the "Findings (fix later)" section of
`reports/debug_report.md`; nothing is fixed during the run. Driver:
[simulate_user.py](simulate_user.py); report: [debug_report.py](debug_report.py).
