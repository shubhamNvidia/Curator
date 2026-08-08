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
.venv/bin/python -m nemo_curator.audio_agent <verb> [args]   # repo virtualenv, from the repo root
# or: source .venv/bin/activate  &&  python -m nemo_curator.audio_agent <verb> [args]
```

## Golden rules (non-negotiable)

- **Never invent** stage or parameter names. Only use stages from `discover` /
  `catalog-tree` / `cards`, and only params the cards/contracts list.
- **0 silent full-scale runs.** Never call `run` with `--confirm` until the user
  has explicitly approved, after seeing a smoke result and the scale/cost estimate.
- **Nothing is written before approval.** Do not create, delete, move or truncate any file or
  directory ahead of the confirm gate — above all not the user's output. A gate the agent has
  already prepared the ground for is not a gate. `validate` returns `output_targets`, stating
  what already exists at every path the recipe writes to (with row and file counts): put that
  in the plan and let the user decide. Never pre-clean an output "so the rerun is clean" — the
  pipeline replaces its own manifest output, so a rerun does **not** accumulate rows. Reading
  `ManifestWriterStage.process` alone suggests it does (it opens in append mode), but `setup`
  truncates first and the stage is pinned to a single worker; two consecutive runs over 4 files
  yield 4 rows, not 8. This is not hypothetical: an agent deleted a user's file before the gate
  on exactly this misreading.
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
- **An unchecked remedy is a guess, not an option.** When a result comes back thin or failed,
  a parameter change is a *fix* only if the data you just observed can actually produce the
  outcome it promises. Check that before offering it; if you cannot, present it as unverified
  and name what would settle it. A knob whose whole range is ruled out by the input's shape —
  a grouping that must contain two of something the upstream stage emitted once, a threshold
  below every value present — is not a choice worth putting to the user, at any setting. And
  once you expect a choice to fail, say so **before** executing it, not in the caveats
  afterwards: running it spends the user's full compute to produce data you already doubt.
- **Report the unit the request was about.** A row count is not a deliverable count — one row
  can carry a list of segments, windows or snippets, and a row can be blank. `run` reports
  `output_rows_written` (read back from the file rather than counted in memory) and
  `sparse_fields`, naming each written field left blank in some rows — surface both. Never
  call an output complete or ready while a field the request depends on is empty in most rows.

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
For `quality_standard` / `distribution` with `scope: aggregate`, normal
run/report/reuse evidence is the arithmetic mean of that finite numeric field
across every valid terminal-manifest row. A partial scan or a missing/non-numeric
value on any retained row remains unverifiable.

### 2. Inspect (always before planning)

```bash
python -m nemo_curator.audio_agent context --goal '{"task":"quality_filter","domain":"read"}' --data /path/to/data
```

`context` returns the L0 `category_tree`, the profiler's `data_profile`
(sample rate, channels, transcripts present?, file count) and `env_profile`
(GPU? ffmpeg? installed extras?), matched blueprints/recipes, and patterns.
Tell the user what you found (it is often news to them).

For environment health specifically, run `doctor` (the single source of truth for env):

```bash
.venv/bin/python -m nemo_curator.audio_agent doctor --json
```

Run it before a heavy GPU run, and first whenever anything env-related looks wrong (import /
CUDA / model-load errors). Do NOT diagnose env from stage cards — cards defer to `doctor`
(details in ENVIRONMENT.md).

**A "no GPU" reading is NOT a hardware fact when the run may be sandboxed.** The tool
sandbox (and containers) can block GPU device access (`/dev/nvidia*`), so `doctor`/`context`
report `has_gpu=false` even when a GPU is physically present and usable. If the env shows
`gpu_possibly_masked=true` — or `nvidia_smi`/`nvidia_device_nodes` indicate hardware, or torch
is a CUDA build (`torch_cuda_built=true`) — do **NOT** tell the user "no GPU". Say "GPU not
reachable from this (sandboxed) run" and **re-verify with full device access**: rerun `doctor`,
and any `smoke`/`run`, with full permissions (outside the sandbox). Only conclude "no GPU"
after a full-access probe still finds none. GPU‑touching verbs (`smoke`/`run`, and the GPU
probe) should run with full device access, not sandboxed.

The deterministic core supplies the facts and grounded options; **you are the
environment analyst**. `context` includes machine-wide `env_health`, and
`validate` / `smoke` / `run` return a recipe-aware `environment_decision`.
When `decision_required` is true:

1. Stop before execution. State the detected fact and confidence separately from
   any inference, then explain how it affects the selected recipe/stages.
2. Recommend the best **available** option using the user's stated constraints
   (for example: host changes allowed but environment changes forbidden), explain
   the material trade-offs of the other viable choices, and ask one outcome-level
   question.
3. Never silently install/upgrade/downgrade, change the launch command, request a
   secret value in chat, switch to CPU, change a model/decoder/stage, or retry the
   same action. The core's choices are proposals, not authorization.
4. Offer CPU only as a conditional candidate when the packet proves every affected
   flattened execution leaf has a supported CPU path; do not call it executable
   until the candidate builds and smokes. A CPU/CTC/model/stage alternative is a **new recipe**
   and config hash: validate -> smoke -> present -> confirm again.
5. After a host/environment/credential fix, rerun `doctor --json` and recipe
   preflight. Never assume the change worked.
6. Scope evidence to the execution target. Do not treat driver GPU, ffmpeg,
   credential, disk, Python, or uv-launch facts as external-worker facts. A
   driver/toolkit mismatch blocks a known runtime-PTX/JIT path; other GPU stages
   require bounded smoke evidence before proposing an invasive host change.

For an execution error, call:

```bash
.venv/bin/python -m nemo_curator.audio_agent diagnose --error '...' --recipe recipe.yaml
```

Use its sanitized classification, live preflight, `attempted_actions`, and
grounded choices. If status is `unknown`, say so and collect only the packet's
minimal diagnostics; do not invent a root cause or fix.

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

**Prune to the request.** A stage earns its place only if it serves a stated goal. Every
filter DROPS data, so each filter must trace to a criterion the user actually asked for
(e.g. "clean" -> a noise gate; "high-quality" -> a MOS gate) — do NOT add a filter for a
dimension the user never mentioned (bandwidth, VAD, an extra quality gate). A blueprint is
a menu, not a mandate: keep the `enforced` stages plus only the `advisory` ones that match
the goal, and drop the rest — never carry a template's (or composite's) full stage set
wholesale. Rule of thumb: if a filter has no matching acceptance criterion, it should not
be in the recipe.

This applies to **preprocess**, not just filters: add mono/resample only if a downstream
stage actually needs that form — UTMOS/SIGMOS/SQUIM accept any channel count and resample
internally, so they do NOT require an upstream mono/resample. And avoid **no-op** stages: a
mono/resample with `keep_waveform_in_task=false` AND `write_to_disk=false` (or `write_to_disk`
without `update_audio_filepath`) while the next stage reads from file **converts the audio and
then discards it** — downstream still scores the ORIGINAL files. Either make it effective
(`write_to_disk=true` + `update_audio_filepath`, or keep the waveform and have the next stage
read it via `input_residency`) or drop the stage.

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

**`resources` is a knob too** — set it from the card's `resource` facts, don't leave the
default. If a stage's card is `bound: gpu` / `gpu_optional: false`, set
`resources=Resources(gpus=1)` (size VRAM from the card's `gpu_mem_gb`); a GPU-*optional*
stage may stay on CPU or opt into GPU for throughput. A GPU-required stage left at its CPU
default runs on CPU (very slow) and can over-parallelize into many model-loading actors.

### 4. Plan -> validate -> critique (static loop, <= 3 iterations)

Emit a Recipe (YAML): `{stages: [{ref, params}], inputs, preset,
acceptance_criteria}`. The complete success contract **must live inside this
recipe before validation, smoke, confirmation, and run**. That makes it part of
`config_hash`; a separate criteria file alone is not executable intent and must
never be the only copy.

For every recipe-driven verb, the first supported source stage's configured
parameter is execution truth. `Recipe.inputs` and `--data` are optional
consistency assertions: they never populate or rewrite that stage. Omit
`--data`, or pass the same canonical source. A missing, mismatched, unsupported,
or unsafe ambiguous source makes validation fail and execution refuse. A
multi-manifest `ManifestReader` may run as authored only with singular `--data`
omitted; it remains unkeyed and unreusable until aggregate identity is
supported. (`context --data` is the pre-recipe exception: it profiles that path
directly for planning.)

Save the recipe **under the scratch directory**, not in the current directory. A recipe written
for one request is working material, and the working directory is usually a git checkout, where
it shows up as an untracked file that looks like unfinished work:

```bash
python -c 'import nemo_curator.audio_agent as aa; print(aa.scratch_dir())'
# -> <workspace>/.audio_agent_runs/recipes   (git-ignored, moves with AUDIO_AGENT_RUNS_DIR)
```

Put it somewhere else only when the user asks for the recipe itself as a deliverable. Then:

```bash
python -m nemo_curator.audio_agent validate --recipe "$(python -c 'import nemo_curator.audio_agent as aa; print(aa.scratch_dir())')/recipe.yaml" \
  --data /path/to/data --acceptance-criteria criteria.yaml --request-type quality_filter
```

The three templates in `nemo_curator/audio_agent/recipes/` each ship a filled-in
`acceptance_criteria` block; start from one rather than writing the list from scratch.

The optional `--acceptance-criteria` file is a cross-check and must match the
recipe's embedded `acceptance_criteria`; validation fails if they differ.
Together with `--request-type`, the criteria compile each criterion's
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
- `tensor_into_sink`: insert `AudioToDocumentStage`, then write its
  `DocumentBatch` with `DocumentBatchJsonlWriterStage`. Keep
  `ManifestWriterStage` for already-serializable `AudioTask` flows only.
- `card_*`: honor the model constraint (e.g. `batch_size` fixed, `<= max_speakers`).
- `ffmpeg_missing` / `missing_secret` / `gpu_unavailable`: surface as setup steps.

#### Mandatory semantic critique (mechanical pass != intent approval)

A `runnable: true` / `status: pass` Verdict proves that the recipe is mechanically
composable under the checks the core can enforce. It does **not** prove that a
valid field means what the user meant, that a filter is applied at the right
entity/granularity, or that the chosen model/metric is a good proxy for the
request. Treat green validation as necessary plumbing evidence, never as an
intent verdict.

**Semantic verification checklist — run it on every field you filter and every
stage you pick, grounding each answer in the packet's `semantic_facts`/notes (or
the source), NOT the key name.** For each such field/stage, answer five questions:

1. **Meaning + unit** — what does this field actually represent, in what unit?
   Read the producer's `semantic_facts[field].meaning`/`unit`; if absent, inspect
   the stage source. A plausible key name is NOT meaning — `num_speakers`,
   `num_segments`, `duration`, `sample_rate` all read one way and behave another.
2. **Scope / entity / granularity** — WHOSE value is it at THIS point in the
   pipeline: the original file, a fan-out child (per speech-segment / per speaker
   turn), or a recombined/aggregate output? Scope is set by where the field is
   produced. A per-segment field *after a fan-out* is not a per-file property, and a
   count/aggregate over children (`num_segments`, a distinct-speaker count) is not a
   per-child attribute. A whole-clip property (e.g. "single speaker") is not something
   a per-segment field can filter.
3. **Provenance** — is it measured, a configured TARGET, or a relative label? A
   target (`output_sample_rate`, resample `sample_rate`) verifies/produces a value;
   it does not describe the input. A diarizer `speaker` is a per-recording cluster id
   (speaker_0 ≠ a person, ≠ comparable across files), not a global identity or a count.
4. **Stage effect vs intent** — does the stage TRANSFORM what the user said stays
   fixed, or DROP rows they expected kept? Separation/resample/mono change the audio;
   strict-rate mono DROPS rate-mismatched rows (it does not convert them); every
   filter DROPS. If intent implies "unchanged / just measured", a transform/mixer is
   the wrong tool.
5. **Direction** — for a metric filter, is lower or higher "better"
   (`metrics.scale.direction`)? Keep the correct side of the threshold (WER/CER are
   error rates → drop ABOVE, not below).

If the honest answer to "does filtering/using this field at this point achieve the
user's stated intent?" is no — or you cannot ground meaning/scope from cards or
source — the critique is `revise` (choose the right field/stage) or `ask`, never a
silent green. **Trigger points that demand this check before you proceed:** a
fan-out or nesting seam, a join/aggregate/concatenation, any transform/materialization
whose output a later stage may or may not read, a diarizer speaker label, and any
count/`num_*`/rate/duration field.

**Worked trap (why this checklist exists).** "Keep only single-speaker clips" does
NOT mean `SpeakerSeparationStage` + filter `num_speakers == 1`: `num_speakers` is the
count of speakers the diarizer found in the ORIGINAL clip (scope = whole recording, an
aggregate), and Separation TRANSFORMS the audio into per-speaker streams — so that
recipe both mis-scopes the field and mangles the audio. Correct: diarize, then keep
rows whose distinct-speaker count is 1 — no separation, no per-segment speaker filter.
The plumbing (a filterable `num_speakers` key) is green either way; only the
meaning + scope + effect check catches it.

After validation is mechanically runnable, **do not smoke yet**. Inspect the
Verdict's `semantic_review` packet. It is built from configured dynamic contracts
and automatically co-locates exact-key lineage, latest/prior producers,
fan-out/nesting/aggregation/filter seams, and the producer/consumer cards'
`semantic_facts`, notes, metrics, domain and limitations. For a generic consumer
such as a value filter, use the packet's exact upstream producer rather than
inferring from the key name. If the packet reports unresolved lineage, missing
card semantics, or opaque visibility, do targeted retrieval/source inspection;
if meaning still is not grounded, mark the critique `ask` rather than inventing
it. A matching key/role proves connectivity, not meaning.

Emit one compact `semantic_critique` with:

- `mechanically_runnable`: copied from the deterministic Verdict;
- `recipe_config_hash`: copied exactly from
  `semantic_review.recipe.config_hash`; it binds the critique to this canonical
  recipe, and any recipe change requires another validation and critique;
- `intent_status`: exactly `pass`, `revise`, or `ask`;
- `stage_reviews`: for **every** stage, the user goal/acceptance criterion it
  serves (a stage with no justification means `revise`);
- `field_reviews`: for every filter/generic field consumer, its exact producer,
  the field's meaning and unit, its entity/granularity at that point in the
  pipeline, and how fan-out, nesting, aggregation or copying changes—or does not
  change—that meaning;
- `behavior_checks`: representative values/rows on both sides of every
  filter—including the boundary—stating what would be retained and dropped;
- `transform_checks`: for every conversion/materialization, which downstream
  stage consumes the transformed waveform/path/key; flag transforms whose result
  is discarded while downstream reads the original;
- `model_checks`: language/domain, model limits, metric coverage and known
  caveats against the profiled data and the user's wording;
- `assumptions_or_questions`: every unresolved semantic assumption, phrased as an
  outcome-level question if user input is required.

The result controls the loop:

- `pass` — intent and plumbing both look coherent; proceed to smoke.
- `revise` — change the recipe, then re-run **validate -> semantic critique**.
  Any recipe change invalidates the prior critique and config hash.
- `ask` — stop before smoke and ask the minimum user-facing question. After the
  answer, revise as needed and repeat validation + critique.

This is an LLM judgment over grounded card/data evidence, not a request for a new
deterministic rule per module. Crashes, corruption, impossible composition and
other universal hard invariants remain core gates; intent-dependent meanings and
trade-offs remain here.

### 5. Smoke (empirical loop, <= 2 iterations)

```bash
python -m nemo_curator.audio_agent smoke --recipe recipe.yaml --sample 10 --data /path/to/data --bootstrap-ray
```

`--bootstrap-ray` lets the agent start a correctly-configured local Ray head
itself (free port, plasma on /tmp, API limit) so no manual Ray setup is needed.
If a cluster already exists, set `RAY_ADDRESS` and omit the flag.

Show the user `retained` / `rejected` + examples. If `goals_met` is false (0
retained, errors), read the structured `diagnosis` (when present). Adjust a
threshold only for a grounded data/filter failure; environment/action-required
failures go through the decision policy above.

On a GPU smoke the result also carries a `calibration` block (measured per-stage
VRAM/throughput). Pass it to the full run so the resource planner can raise a
card/default estimate when the smoke observed a larger peak; because a bounded
smoke cannot prove the full-run maximum, calibration never lowers that baseline:
`run ... --calibration calib.json` (extract it with `calibrate --smoke
smoke.json`). On a CPU smoke there's no VRAM to measure, so the planner keeps
using the card facts.

### 6. Confirm gate -> run

Present the plan, the semantic critique (`intent_status: pass`), the smoke
evidence, the scale/time estimate, **and the acceptance-criteria contract** —
stating for each criterion **what its metric captures and what it does NOT**
(e.g. "UTMOS measures naturalness/overall quality, not background-noise level —
add a noise/SIGMOS criterion?"). Never silently decide which metric stands for a
fuzzy word ("clean", "good"); surface it here. Then ask the user to confirm. Only
then:

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
reasons, output paths) in plain language. `run` also returns `acceptance`, verified
against the recipe's embedded contract and terminal-output evidence; treat that as
the primary post-run verdict. Use standalone `verify` only for an explicitly
post-hoc evidence set or a newly proposed contract:

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

### 8. Don't redo finished work (reuse prior runs)

Every completed step publishes a content-addressed **artifact**, so a later request can
reuse it instead of recomputing it (design: `nemo_curator/audio_agent/REUSE_ARCHITECTURE.md`).
Once you have a candidate recipe — **before** smoking or running — scan for prior work:

```bash
python -m nemo_curator.audio_agent reuse-scan --recipe recipe.yaml --data /path/to/data
```

The `decision` is `already_done` (this pipeline matches a prior computation and
dataset key at the reported trust tier), `incremental` (the first *N* stages are
already done — e.g. resample + VAD + quality-filter exist and only ASR is new),
or `fresh`. Reuse survives things that do NOT change output bytes: a different
batch size, different `resources`, a different output path, or a **stricter
success bar** (the data is reused and the contract re-verified). A detected
dataset-key change, a missing completion marker, or a Curator version change
prevents reuse. Shape-tier matches are low trust and default fresh because
metadata gaps may hide changes. A stage declared non-deterministic is not
refused—it is offered with that said and `fresh` pre-selected, because the result
is real, just not one a rerun is promised to match.

**Two rules for the conversation:**

- **Never reuse silently.** If `prompt_user` is true, show the candidate card — objective,
  pipeline, input/output, key params, date, metrics, estimated time saved — and offer the
  three choices. Use the scan's `recommended` as your default; it is `fresh` whenever
  `trust` is `low` (say why: the `weaknesses` list is written for a human).
- **Never nag.** `prompt_user: false` means don't ask: either there is nothing to reuse
  (just run) or the saving was *measured* and is trivial (take it, and mention it in your
  summary). When `unpriced_stages` is non-empty the question is not about the size of the
  saving — nobody timed those stages and the cards call them expensive — so say that rather
  than quoting the `estimated_saving_sec`, which is a floor and will look absurdly small.

Then act on the choice — don't hand-edit the recipe:

```bash
python -m nemo_curator.audio_agent continue --recipe new.yaml --data /path/to/data \
  --execute --choice extend --confirm <config_hash>
```

- **as_is** — serve the completed output and re-check it against today's criteria.
- **extend** — rewrite the recipe to start from the reused artifact, re-validate the
  remaining stages against what that artifact actually carries, and run only those.
- **fresh** — ignore prior work and recompute.

`--parent-run-id <id>` is optional and additive: it diffs against that specific run, and
whichever engine reuses more wins. `runs --data /path/to/data` shows everything already
done to a corpus; `reindex` rebuilds the lookup index from the JSON records if it is lost.

Record what a run was FOR with `run --goal "..."` — that objective is what makes the
candidate legible to a human months later.

## Control conditions

- **Success**: `runnable` + smoke goals met -> confirm -> run -> report.
- **Escalate**: static/empirical loops exhausted, or an unproducible role -> hand
  the user the best candidate + the blocking issues.
- **Refuse**: intent on the refuse list -> stop + safe alternative.

## When the catalog cannot help (skill fallback)

If the goal matches no recipe and you cannot ground a plan from the cards
(e.g. a brand-new stage without a card), say so explicitly and reason from the
stage contracts (`describe`) rather than guessing parameters.
