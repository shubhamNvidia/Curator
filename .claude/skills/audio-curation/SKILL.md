---
description: Build and run a NeMo Curator audio curation pipeline from a natural-language goal (host-driven planner over the audio_agent tool core)
---

# Audio Curation Agent (P1)

Turn a user's audio-curation goal into a validated, runnable NeMo Curator recipe,
with pre-flight checks, a bounded smoke test, a confirmation gate, and an
evidence-backed report. **You (the host model) are the planner and critic.** The
`nemo_curator.audio_agent` tool core is deterministic and grounds every decision:
it tells you which stages exist, whether a recipe composes, and what happened.

All commands print JSON. Run them with the repo virtualenv interpreter (base
`python` may lack Curator's deps) from the Curator repo root:

```bash
/localhome/local-shbhawsar/ADV/Curator/.venv/bin/python -m nemo_curator.audio_agent <verb> [args]
# or: source .venv/bin/activate  &&  python -m nemo_curator.audio_agent <verb> [args]
```

## Golden rules (non-negotiable)

- **Never invent** stage or parameter names. Only use stages from `discover` /
  `catalog-tree` / `cards`, and only params the cards/contracts list.
- **0 silent full-scale runs.** Never call `run` with `--confirm` until the user
  has explicitly approved, after seeing a smoke result and the scale/cost estimate.
- **Evidence only.** Never claim quality/throughput improved without before/after
  numbers from a `report`.
- **Define success up front, verify it after.** Derive `acceptance_criteria` (the
  success contract) from the request, confirm them at the gate, and `verify` the
  result against them. Never declare success without an AcceptanceReport whose
  `overall` is `met`; never silently relax a `must` criterion.
- **Refuse / redirect**: human labeling of speaker gender/accent/emotion or other
  subjective traits; model training/eval/deployment; large runs without confirmation.
  Offer a safe alternative (e.g. a manifest + quality report).
- Separate **local-ready** from **GPU/endpoint/approval-needed** (see gate flags).
- **User-facing questions only.** Never ask about internal parameters (thresholds,
  `*_key`s, residency, batch size, model IDs, task-types). Ask at the **outcome layer** —
  quality *level* ("studio / broadcast / general?"), language, output format — and
  resolve to params via the card's `anchors`/`presets`. Asking a **module choice** is
  fine, framed as plain capability trade-offs from the cards. Ask only decisions that
  are **material + preference-dependent + not inferable**; otherwise use a safe
  default/inference and note it.

## The loop

### 1. Interpret + build a capability plan

Turn the request into a small goal **and a capability plan**: `task` (validate /
quality-filter / VAD / transcribe / WER-filter / diarize / ALM-windows / convert),
`domain` (conversational / read / long-form / multilingual), plus `expected_outputs`
(what the user wants to end up with), `capability_areas` (which categories likely
apply), `constraints` (quality/hardware/deps), and `open_questions` (what's still
missing). The capability plan is your **coverage checklist** for later steps.
Ask 1-2 short **user-facing** questions only if something material is ambiguous
(e.g. no quality *level*). If the request hits the refuse list, stop and redirect.

Also derive `acceptance_criteria` — the **success contract** (what "done" means):
`output_completeness` (required outputs, e.g. transcripts), `quality_standard`
(a metric target, `absolute` like "studio" or `relative` like "best 20%"), and
`yield` (how much to keep). Classify each `absolute`/`relative`. These drive both
`validate` (output-completeness + request-type sanity) and the final `verify`.

### 2. Inspect (always before planning)

```bash
python -m nemo_curator.audio_agent context --goal '{"task":"quality_filter","domain":"read"}' --data /path/to/data
```

`context` returns the L0 `category_tree`, the profiler's `data_profile`
(sample rate, channels, transcripts present?, file count) and `env_profile`
(GPU? ffmpeg? installed extras?), matched blueprints/recipes, and patterns.
Tell the user what you found (it is often news to them).

### 3. Route coarse-to-fine (do NOT read every card)

- **L0**: from `catalog-tree`, pick only the categories the goal needs; prune the
  rest (no filtering intent -> skip `quality`/`filter`; no transcripts -> `transcribe`/WER
  is usually unproducible).
- **L1**: `cards --category <cat>` for the chosen categories -> shortlist stages.
- **L2**: `cards --names A B C` for the finalists -> read full cards (model
  constraints, presets, ordering hints) and select stages.

**When two stages overlap** (e.g. two diarizers/VADs), do NOT pick arbitrarily.
Compare them on card facts (supported I/O, accuracy, language, hardware, latency,
resource, config complexity, known limitations, compatibility, goal-suitability),
then apply the decision policy: **auto** if one clearly fits best or the choice is
low-impact; **recommend** if there's a trade-off (state it, allow override); **ask**
only if the choice is material *and* preference-dependent *and* not inferable — with a
one-line plain-language difference + your recommendation (never expose internal params).

Prefer adapting a `matched_blueprint` (it encodes idiomatic ordering with
`enforced`/`advisory` tags and `topology_selection`) over composing from scratch.
Adapt, do not blindly copy.

### 3b. Resolve outcomes to parameters (never expose internal numbers)

Do not hand-pick thresholds. Map the user's **outcome** to a concrete param with
`resolve`, which reads the card's `metrics` anchors/presets:

```bash
python -m nemo_curator.audio_agent resolve --stage UTMOSFilterStage --label studio
# -> params {mos_threshold: 4.0}   (+ an auditable strategy trail)
```

- `--label <outcome>` (e.g. `studio`, `wideband`, `transcription_grade`) maps via
  the card anchors to the stage's own threshold param, or — for an annotator like
  WER — to a `PreserveByValueStage` filter (`filter_stage` in the result).
- `--use-case <preset>` applies a named card preset; `--explicit '{"p": v}'` uses a
  value the user gave.
- If it returns `asks` (unknown label, or a **relative** objective like "best 20%"
  which needs data / Path B), ask the user a plain-language question or get an
  explicit bar — never invent the number.

Apply the returned `params` to the stage (and insert any `filter_stage`), and keep
the `strategy` trail for the plan/report (it records *why* each value was chosen).

### 4. Plan -> validate -> critique (static loop, <= 3 iterations)

Emit a Recipe (YAML): `{stages: [{ref, params}], inputs, preset}`. Save it and:

```bash
python -m nemo_curator.audio_agent validate --recipe recipe.yaml --data /path/to/data \
  --acceptance-criteria criteria.yaml --request-type quality_filter
```

Passing `--acceptance-criteria` + `--request-type` compiles each criterion's
output/metric into a producible-role check (so "success needs transcripts, no ASR
stage" fails here as `missing_output_producer`) and runs **request-type sanity**
(a filtering request with no `yield` criterion is flagged `missing_implied_criterion`).

Read the `Verdict`. If not `runnable`, fix from the issues and re-validate. **A gap
here means your candidate-card set was incomplete — you are NOT limited to the first
set.** When `validate` names a missing role, do a **targeted re-retrieval** (it tells
you *which* role, so query only that: the producing `cards --category`, or the role
graph in `context`), add the producer, and re-plan. This is the retrieve↔plan loop.

- `unsatisfied_reads` / `unproducible_roles`: insert an upstream producer via targeted
  re-retrieval (role graph in `context`); only if a role is truly unproducible across
  the whole catalog is the goal impossible with these stages -- tell the user.
- `dangling_key`: align the producer's `*_key` value with what the consumer reads.
- `tensor_into_sink`: insert `AudioToDocumentStage` before a JSON writer.
- `card_*`: honor the model constraint (e.g. `batch_size` fixed, `<= max_speakers`).
- `ffmpeg_missing` / `missing_secret` / `gpu_unavailable`: surface as setup steps.

### 5. Smoke (empirical loop, <= 2 iterations)

```bash
python -m nemo_curator.audio_agent smoke --recipe recipe.yaml --sample 10 --data /path/to/data --bootstrap-ray
```

`--bootstrap-ray` lets the agent start a correctly-configured local Ray head
itself (free port, plasma on /tmp, API limit) so no manual Ray setup is needed.
If a cluster already exists, set `RAY_ADDRESS` and omit the flag.

Show the user `retained` / `rejected` + examples. If `goals_met` is false (0
retained, errors), read the `notes` (they include a failure classification),
adjust thresholds, and re-plan.

### 6. Confirm gate -> run

Present the plan, the smoke evidence, the scale/time estimate, **and the
acceptance-criteria contract** — stating for each criterion **what its metric
captures and what it does NOT** (e.g. "UTMOS measures naturalness/overall quality,
not background-noise level — add a noise/SIGMOS criterion?"). Never silently decide
which metric stands for a fuzzy word ("clean", "good"); surface it here. Then ask
the user to confirm. Only then:

```bash
python -m nemo_curator.audio_agent run --recipe recipe.yaml --confirm <config_hash> --data /path/to/data --bootstrap-ray
```

Passing the `config_hash` (from the refusal output) enforces plan-execution
integrity: what was approved is exactly what runs. `--bootstrap-ray` starts the
Ray head if needed (same as smoke).

Guardrails enforced in the tool: paths are restricted to `AUDIO_AGENT_WORKSPACE`
(when set); secrets/transcripts are stripped from tool output; and if
`AUDIO_AGENT_REQUIRE_SMOKE` is set, also pass `--smoke-token <token>` (from the
`smoke` output) or `run` refuses. The resource planner auto-picks streaming/batch
and refuses if the recipe can't fit the machine.

### 7. Report + verify acceptance

Summarize the returned `report` (retained/rejected, per-filter counts, failure
reasons, output paths) in plain language. Then **verify the success contract**:
assemble the evidence (from `validate`: `produced_roles`/`produced_keys`; from the
`report`/smoke: `metrics`, `retained`, `input_count`) and run:

```bash
python -m nemo_curator.audio_agent verify --criteria criteria.yaml --evidence evidence.json \
  --recipe recipe.yaml   # frozen contract -> runs the honesty guard
```

Report the `AcceptanceReport`: `overall` (`met` iff every `must` criterion is met)
plus each criterion's state — `met` / `not_met` / `unverifiable` (no evidence, e.g.
WER with no references) / `unachievable` (the data cannot reach an absolute target).
Only declare success when `overall` is `met`. `unachievable`/`not_met` are honest
outcomes: offer options (adjust thresholds, provide references, relabel an absolute
bar with the user's consent) — never silently relax a `must`.

**Reviewer charter (you, the host, are the reviewer).** After the deterministic
verify:
1. Resolve any `semantic_fit` criteria (they come back `unverifiable` — that's your
   job): judge, grounded in the evidence/examples, whether the result coheres with
   intent.
2. Read the `honesty` section — the guard flags goalpost-moving (a confirmed `must`
   dropped/downgraded/relaxed vs the frozen contract). If non-empty, `overall` is
   forced `not_met`: do **not** present success. The contract is frozen into the
   recipe and covered by `config_hash`, so relaxing a bar means re-confirming a new
   contract with the user, never editing it silently.
3. You may surface semantic concerns but **may not override the deterministic
   verdict** — if unresolved, escalate to the user.

### 8. Follow-up requests (reuse prior work)

Every `run` writes a local **Run Record** (provenance for tracing + continuation —
local history, not shared memory). For a follow-up on the **same data** (e.g. "also
add transcripts to what we just curated"), build the new recipe, set its
`parent_run_id` to the prior run, and ask for the incremental plan:

```bash
python -m nemo_curator.audio_agent runs                       # list prior runs
python -m nemo_curator.audio_agent continue --recipe new.yaml --parent-run-id <id> --data /path/to/data
```

- **incremental** — the new recipe extends the parent: reuse the parent output
  (`reuse_from`) and run only `run_stages` (point the appended stage(s) at the reused
  manifest instead of re-curating).
- **already_done** — identical recipe/data: reuse the parent output, run nothing.
- **full_rerun** — an upstream param/stage changed, or the dataset differs
  (fingerprint mismatch): there is no safe reusable intermediate, so rerun. The
  `reason` / `diverged_at` explains why. Never silently reuse across a change.

## Control conditions

- **Success**: `runnable` + smoke goals met -> confirm -> run -> report.
- **Escalate**: static/empirical loops exhausted, or an unproducible role -> hand
  the user the best candidate + the blocking issues.
- **Refuse**: intent on the refuse list -> stop + safe alternative.

## When the catalog cannot help (skill fallback)

If the goal matches no recipe and you cannot ground a plan from the cards
(e.g. a brand-new stage without a card), say so explicitly and reason from the
stage contracts (`describe`) rather than guessing parameters.
