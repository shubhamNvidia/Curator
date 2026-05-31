# Agentic Flow — Architecture & Ownership Guide

> **Audience:** the project owner. Read this top-to-bottom once and you'll
> know which file does what, where the LLM gets to think, where it
> *doesn't*, and how to diagnose a bad pipeline from the run artefacts.
>
> Companion document:
> - `INTENT_V2.md` — the intent schema design (what fields exist, why
>   `FilterMode` has four states, the stage-application matrix).
> - This doc — the *flow* (who calls whom, what the LLM sees, what's
>   deterministic, how critics work, how to debug).

---

## 1. TL;DR

A user types a prompt in the browser. The system runs:

1. **Extract** intent from the prompt with an LLM (one structured-JSON call).
2. **Profile** the dataset with deterministic decoders, then **prefill**
   any obvious fields the extractor missed (heuristics).
3. **Analyze gaps** deterministically, and **phrase questions** with an
   LLM (falls back to a static template if no LLM is configured).
4. User answers the form. Every answer is stamped as a
   `clarifier_answer:<path>=<value>` audit note so the rest of the
   pipeline knows it's locked.
5. **Select stages** deterministically from the answered intent (no LLM).
6. **Tune** workers / GPUs / execution mode deterministically.
7. **Validate** the IR (deterministic: phase order, ports, license).
8. **Critique** the IR with a SanityCritic (deterministic) and a
   PlanCritic (LLM). Apply patches that don't touch user-locked paths.
   Re-plan once if patches change anything.
9. **Compile** the IR to a YAML pipeline (deterministic).

The intent is the *only* thing the LLM is allowed to write. Everything
downstream — stage selection, ordering, params, resource sizing, YAML
emission — is pure Python.

---

## 2. Bird's-eye flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser (single page app served by web.py)                         │
│                                                                     │
│  prompt + source URI ─► /api/plan                                   │
│                            │                                        │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ web.py :: AgenticWebApp.plan                                 │  │
│  │   1. profile dataset       (profiler.py)         DETERMIN.   │  │
│  │   2. extract_intent        (planner_dag + LLM)    LLM        │  │
│  │   3. infer_intent_from_prompt (clarifier.py)     DETERMIN.   │  │
│  │   4. apply_profile_prefills (clarifier.py)       DETERMIN.   │  │
│  │   5. analyze_gaps          (smart_clarifier.py)  DETERMIN.   │  │
│  │   6. _llm_phrase_questions (smart_clarifier+LLM) LLM (opt.)  │  │
│  │   → SmartForm with inferred-chips + questions                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  user answers form ─► /api/build                                    │
│                            │                                        │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ web.py :: AgenticWebApp.build                                │  │
│  │   1. apply_answers       (clarifier.py)          DETERMIN.   │  │
│  │      ├ stamps clarifier_answer: notes                        │  │
│  │      └ triggers IntentCategories._consistency                │  │
│  │   2. plan_from_intent    (deterministic_planner.py)          │  │
│  │      ├ select_stages         (stage_selector.py) DETERMIN.   │  │
│  │      ├ tune                  (tuner.py)          DETERMIN.   │  │
│  │      ├ validate              (validator.py)      DETERMIN.   │  │
│  │      └ dry_run               (dryrun.py)         DETERMIN.   │  │
│  │   3. review_and_replan   (plan_critic/orchestrator.py)       │  │
│  │      ├ SanityCritic.review     (sanity.py)       DETERMIN.   │  │
│  │      ├ PlanCritic.review       (plan.py + LLM)   LLM (opt.)  │  │
│  │      └ apply_findings          (base.py)         DETERMIN.   │  │
│  │            ├ skips user-locked patches                       │  │
│  │            ├ surfaces validator-coerced patches              │  │
│  │            └ demotes invalid enum patches                    │  │
│  │   4. compile             (compiler.py)           DETERMIN.   │  │
│  │   → compiled.yaml + artefacts                                │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module map (one row per file)

### `nemo_curator/agentic/`

| File | What it owns | LLM? |
|---|---|---|
| `web.py` | HTTP server, session state, `/api/plan` and `/api/build` handlers, LLM client construction, UI HTML, run artefact bookkeeping. | indirect (constructs the client) |
| `intent.py` | `IntentCategories` Pydantic schema + `_consistency` model validator (the canonical coercions, e.g. `single_speaker_clips ↔ speakers=split`). | no |
| `cards.py` | `StageCard` schema. Every catalog stage carries one of these (load via YAML). Owns the `Phase` enum (`source < preprocess < load < segment < analyze < filter < write`) the validator's phase-reorder uses. | no |
| `registry.py` | Loads all `stage_card.yaml` files, resolves class targets, builds the `CapabilityRegistry`. Also `lint()` for drift between cards and classes. | no |
| `profiler.py` | `DatasetCard` builder. Decodes a handful of files to compute sample-rate / channel / duration histograms used by the prefill heuristics and the dataset summary chip in the UI. | no |
| `cards.py` ↔ `cards/*.yaml` | per-stage metadata: `phase`, `produces`/`consumes` keys, `est_vram_inference_gb`, `params_total`, `gpu_class`. | no |

### Extract / clarify (steps 1 – 6 in §2)

| File | What it owns | LLM? |
|---|---|---|
| `planner_dag.py :: extract_intent` | The one place where the prompt → schema LLM call happens. Uses `nat/prompts/extractor.md`. Retries once at temp=0 on JSON parse error. | YES |
| `clarifier.py :: infer_intent_from_prompt` | Regex heuristics ("TTS" → sr=24k, "phone" → 16k, "between 2 and 60 s" → duration). Pure code. | no |
| `clarifier.py :: apply_profile_prefills` | If the dataset has one dominant sample rate, prefill it. If all files are mono, prefill `channels=mono`. | no |
| `clarifier.py :: apply_answers` | Merges form answers into an intent. Stamps every answer as a `clarifier_answer:<path>=<value>` note. **This is the lock mechanism.** | no |
| `clarifier.py :: build_clarification_form` | The "advanced" ingredient-picker form (every field exposed). Rendered in the `Show all options` expander. | no |
| `smart_clarifier.py :: analyze_gaps` | Decides which questions to ask. Returns a list of `Gap` records. | no |
| `smart_clarifier.py :: template_questions` | Deterministic phrasing fallback when no LLM is configured (or LLM errors). | no |
| `smart_clarifier.py :: _llm_phrase_questions` | Asks LLM to re-phrase the template questions in the user's own words. Uses `nat/prompts/clarifier.md`. Validates the LLM output against the template anchor — option `value`/`apply` always come from the template, only the *strings* come from the LLM. | YES |
| `nat/prompts/extractor.md` | System prompt for `extract_intent`. | LLM prompt |
| `nat/prompts/clarifier.md` | System prompt for `_llm_phrase_questions`. | LLM prompt |

### Plan / select (steps 7 – 9)

| File | What it owns | LLM? |
|---|---|---|
| `deterministic_planner.py :: plan_from_intent` | Re-validates intent, builds SourceSpec/SinkSpec, calls `select_stages → tune → validate → dry_run`. Re-runs the tuner after validator auto-inserts. | no |
| `stage_selector.py :: select_stages` | The brain. Walks the intent and emits an ordered list of `StageRef`. ~12 helpers (`_output_normalizers`, `_file_level_speaker_stages`, `_segmentation_stages`, `_quality_stages`, `_cleaning_concat_stages`, `_speaker_stages`, `_post_segmenter_vad_trim`, `_text_stages`, `_alm_stages`, `_whole_file_duration_filter_stages`, `_segment_extraction_stages`). | no |
| `ir.py` | `PipelineIR` + `StageRef` schemas. `StageRef.phase_override` lets the selector pin a stage's rank (Sortformer → preprocess instead of analyze). | no |
| `tuner.py :: tune` | VRAM-packing math → `gpus_per_worker`. Weighted fair-share by `cost_weight` → `num_workers`. Streaming vs batch from the concurrent-demand floor. | no |
| `validator.py :: validate` | Phase-order reorder (respecting `phase_override`), auto-insert (manifest writer, mono conversion, resample), port-compatibility check, license gate, terminal-stage placement. | no |
| `dryrun.py :: dry_run_pipeline` | Simulates the pipeline on a tiny synthetic `AudioTask` to catch missing-key errors at compile time. | no |
| `compiler.py :: compile_ir_to_yaml` | Walks the IR, asks each `StageCard.target` class to YAML-serialize its params, writes `compiled.yaml`. | no |

### Critic (between validate and compile)

| File | What it owns | LLM? |
|---|---|---|
| `plan_critic/base.py` | `CriticFinding` / `CriticReport` / `CriticSeverity` shapes. `apply_findings` — the patch applicator with three safety nets: user-locked paths, invalid enum patches, validator-coerced patches. `user_locked_paths` parses `clarifier_answer:` notes. | no |
| `plan_critic/sanity.py :: SanityCritic` | Deterministic checks: missing writer, output unit ↔ stage mismatch, quality intent ↔ scoring stages, speaker filter without bounds, `single_speaker_clips` without SpeakerSeparation, etc. | no |
| `plan_critic/plan.py :: PlanCritic` | LLM-driven critique using `nat/prompts/plan_critic.md`. Sees the compiled IR, the intent, the dataset profile, and the list of `user_locked_paths`. Emits at most 5 findings. | YES |
| `plan_critic/orchestrator.py :: review_and_replan` | Runs critics in order, applies patches with `apply_findings`, re-plans once with the patched intent if anything actionable changed. Bounded by `max_iterations` (default 2). | no |
| `nat/prompts/plan_critic.md` | System prompt for `PlanCritic`. Explicitly tells the LLM about `user_locked_paths` and the `single_speaker_clips` ↔ `speakers=split` cross-field constraint. | LLM prompt |

### Misc

| File | What it owns |
|---|---|
| `llm.py` | `LLMClient`, message types, OpenAI-compatible HTTP wrapper, sync + async, retry logic, token accounting. |
| `llm_transcript.py` | Per-request transcript recorder. Every LLM call inside a `bind_transcript(...)` context lands in `llm_calls.jsonl` for the run. The hook is plumbed through `LLMClient.chat`. |
| `cli.py` | CLI entrypoint mirroring `/api/build` for non-web usage. |
| `runner.py` | Wraps the executor invocation that actually runs the compiled YAML on the cluster. |
| `tools.py`, `adapters.py`, `cache.py`, `multipipeline.py` | Helpers for source-spec adapters, manifest-cache, multi-pipeline experiments. Not in the hot path of a single build. |

---

## 4. Deterministic ↔ LLM responsibility split

| Decision | Who decides | Where | Why this split |
|---|---|---|---|
| Map prompt → `IntentCategories` fields | **LLM** | `extract_intent` | Language is fuzzy; the LLM is good at "studio quality" → `quality.mos=filter, mos_threshold=4.0`. |
| "TTS → 24 kHz" / "phone → 8 kHz" prefill | Code | `clarifier.infer_intent_from_prompt` | Deterministic catch for the obvious cases the LLM might miss. |
| Dataset-driven prefill (dominant SR / mono) | Code | `clarifier.apply_profile_prefills` | Pure histogram math. |
| Which questions to ask | Code | `smart_clarifier.analyze_gaps` | Policy — the LLM must not silently widen or skip questions. |
| How to phrase a question | **LLM** | `_llm_phrase_questions` | Cosmetic; the LLM only rewrites titles + option labels. |
| The set of intent paths a question writes to | Code | template anchor | If the LLM invents a path, we drop it (`_merge_llm_into_template`). |
| Pre-select an option from a heuristic prefill | Code | `web.py :: prepopulatePrefills` (front-end) | One click ≠ no choice. The user always confirms. |
| Cross-field consistency (e.g. `single_speaker_clips` ↔ `speakers=split`) | Code | `IntentCategories._consistency` | The validator is the final arbiter; coercions are explicit + logged. |
| Which stages go in the pipeline | Code | `stage_selector.select_stages` | The compiler-style matrix from `INTENT_V2.md` §4. |
| Stage ordering | Code | `cards.Phase` enum + `validator._phase_reorder` (+ `StageRef.phase_override`) | Phase rank: `source < preprocess < load < segment < analyze < filter < write`. |
| Param values bound to a stage | Code | `stage_selector` helpers + `validator._check_params` | We never let the LLM tune a stage param directly; everything maps from an intent field. |
| Resource sizing (`num_workers`, `gpus_per_worker`, streaming vs batch) | Code | `tuner.tune` | Math driven by `StageCard.est_vram_inference_gb` and `cost_weight`. |
| Validator coercions / auto-inserts | Code | `validator.validate` + `_apply_autoinsert` | If the user forgot a `ManifestWriterStage`, we insert one. If the sample rate needs a `ResampleAudioStage`, we insert one. |
| "Does the pipeline match the intent?" | Code + **LLM** | `SanityCritic` (code) + `PlanCritic` (LLM) | Cheap structural rules in code; "does this *look* right for a TTS dataset?" in the LLM. |
| Whether to apply a critic patch | Code | `apply_findings` | The LLM proposes; code decides. Three guards: user-locked paths, schema-invalid enums, validator-coerced reverts. |
| YAML emission | Code | `compiler.compile_ir_to_yaml` | Pure mechanical serialisation. |

The rule of thumb: **the LLM produces words, code produces structure.** When
the two disagree, the structure wins and the disagreement is logged.

---

## 5. The flow, step by step

### 5.1 `/api/plan` — produce a form

`web.py :: AgenticWebApp.plan` runs five things in order:

```
prompt + source URI
        │
        ▼
1. _extract_intent_and_profile           (web.py, helper)
   │  ├─ profile = build_dataset_card(source_uri)        (profiler.py)
   │  └─ rough_intent = extract_intent(prompt, profile)  (planner_dag.py)
   │
2. compose_smart_form(prompt, rough_intent, profile, llm)
   │  ├─ infer_intent_from_prompt        (clarifier.py)  ← regex prefills
   │  ├─ apply_profile_prefills          (clarifier.py)  ← dataset prefills
   │  ├─ analyze_gaps                    (smart_clarifier) ← what to ask
   │  ├─ _llm_phrase_questions OR template_questions
   │  ├─ build_inferred_chips            (smart_clarifier) ← chips for the UI
   │  └─ build_clarification_form        (clarifier.py)  ← advanced form
   │
   ▼
SmartForm { inferred, questions, assumptions, advanced_form, intent }
```

Artefacts written under `<run_dir>/`:
- `prompt.txt`
- `request.json`
- `intent_initial.json` (the *enriched* intent — extractor + heuristics + profile)
- `profile.json` (if a profile was built)
- `smart_form.json` (the actual JSON shown to the browser)
- `llm_calls.jsonl` (every LLM call with model, tier, latency, tokens)

### 5.2 The clarifier-form policy

The smart form has three layers, all visible at once:

1. **Inferred chips** (collapsed by default) — what we already filled in.
2. **Quick questions** — the actual gaps. Hierarchical (yes/no with
   follow-ups). At most 5 top-level questions.
3. **Show all options** — the legacy ingredient-picker form, every
   intent path exposed.

Three rules govern what surfaces as a question (`analyze_gaps`):

| Rule | When it fires |
|---|---|
| **Essential, always-asked** | `output.sample_rate` (always — even if prefilled. The prefill becomes the *suggested* radio.) |
| **Prompt-mentioned** | The user said "duration" / "speaker" / "quality" / "transcript" — we ask the matching question even if the extractor filled it in. The prefill is the default radio. |
| **Vague-prompt safety net** | The prompt is ≤ 12 words. Always emit pivots for `speakers.mode` and `quality.mos` even without keywords, so a one-line prompt doesn't silently inherit defaults. |
| **TTS pivots** | The word "TTS" appears. Force the speakers and quality questions because TTS users always care. |

The LLM never gets to remove a question, only re-phrase it. The
template anchor pins the `intent_path`, the option `value`s, and the
`apply` side-effect dicts. The LLM owns titles + option labels +
`description`.

### 5.3 `/api/build` — produce a YAML

`web.py :: AgenticWebApp.build`:

```
form answers
    │
    ▼
1. apply_answers(intent, answers)                (clarifier.py)
   ├─ for path, value in answers:
   │    ├─ set intent.<path> = value
   │    └─ append "clarifier_answer:<path>=<value>" to intent.notes
   └─ IntentCategories.model_validate(...)        (intent.py)
        └─ _consistency coercions trigger here    (e.g. single_speaker_clips coercions)
    │
    ▼
2. plan_from_intent(intent, source, sink, registry, cluster)
   ├─ select_stages(intent, registry, sink)       (stage_selector.py)
   ├─ tune(stages, registry, cluster)             (tuner.py)
   ├─ build IR + validate + dry_run               (validator.py + dryrun.py)
   └─ if validator auto-inserted, re-tune
    │
    ▼
3. review_and_replan(intent, ir, replan, critics) (plan_critic/orchestrator.py)
   ├─ round 1
   │   ├─ SanityCritic.review                     (sanity.py)
   │   ├─ PlanCritic.review (LLM)                 (plan.py)
   │   ├─ apply_findings                          (base.py)
   │   │   ├─ skip user-locked paths
   │   │   ├─ demote invalid enum patches
   │   │   └─ surface validator-coerced patches
   │   └─ if patches applied → replan
   ├─ round 2 (only runs if round 1 changed things)
   └─ never more than max_iterations
    │
    ▼
4. compile_ir_to_yaml(ir)                         (compiler.py)
```

Artefacts written:
- `intent.json` (final, after critics)
- `ir.json` + `ir.validated.json`
- `findings.json` (validator)
- `critic.json` (critic findings, after patches applied / demoted)
- `dry_run.json`
- `compiled.yaml` (the runnable pipeline)
- `llm_calls.jsonl` (extended with the critic and clarifier calls)

### 5.4 `select_stages` — the deterministic matrix

`stage_selector.select_stages` walks the intent in **this** order:

```python
stages.extend(_output_normalizers(intent, sink))           # Resample + Mono
stages.extend(_file_level_speaker_stages(intent))          # Sortformer + PBV (file-level)
stages.extend(_segmentation_stages(intent, sink, cleaning))# VAD or SplitLongAudio
stages.extend(_quality_stages(intent))                     # UTMOS + SIGMOS + Band + PBV gates
stages.extend(_cleaning_concat_stages(cleaning_flow=...))  # SegmentConcatenation (cleaning flow only)
stages.extend(_speaker_stages(intent))                     # SpeakerSeparation (SPLIT mode)
stages.extend(_post_segmenter_vad_trim(intent))            # VAD again, after a non-VAD segmenter
stages.extend(_text_stages(intent))                        # ASR / Aligner / WER
stages.extend(_alm_stages(intent))                         # ALMDataBuilder
stages.extend(_whole_file_duration_filter_stages(intent))  # PBV(duration ge/le)
stages.extend(_segment_extraction_stages(intent, sink))    # SegmentExtraction + Writer
```

The order above is the *emission* order. The validator then reorders by
`Phase` rank, with `StageRef.phase_override` letting the selector pin a
stage out of its card's default phase (e.g. Sortformer is normally
`analyze`, but `_file_level_speaker_stages` pins it to `preprocess` so
it lands *before* VAD).

Two key conventions in the helpers:

- **Score-and-gate split.** `UTMOSFilterStage` and `SIGMOSFilterStage`
  always run with `threshold=0.0`. The drop logic lives in a separate
  `PreserveByValueStage` row. This is how `quality.gates` gets the full
  operator matrix (`lt`/`le`/`eq`/`ne`/`ge`/`gt`) — the score stage
  computes, the PBV gate filters.
- **`StageRef.phase_override`** is the only way the selector explicitly
  controls position. Used twice today:
  - `_file_level_speaker_stages` → Sortformer + its PBV gates pinned to
    `preprocess`.
  - Future stages that need an unusual position should use this too,
    *not* hack the card's phase.

### 5.5 `validator.validate` — the safety net

After selection + tuning, the validator does six checks:

1. **`_check_stage_resolution`** — every `StageRef.stage` resolves to a card.
2. **`_check_resource_sanity`** — no negative workers / GPUs.
3. **`_check_params`** — every stage's bound params type-check against
   its card's param schema.
4. **`_check_license_gate`** — if `policy.commercial_only=True`, no
   non-commercial models.
5. **`_check_preconditions_only` + `_check_key_flow`** — the
   `produces` of an upstream card cover the `consumes` of every
   downstream card. Catches "ASR runs before VAD" type bugs.
6. **`_check_ordering_hints` + `_phase_reorder`** — the stage list is
   sorted by phase rank, respecting `phase_override`.
7. **`_apply_autoinsert` + `_apply_shape_autoinsert`** — auto-insert
   `ManifestWriterStage`, `MonoConversionStage`, `ResampleAudioStage`
   when the intent demands them and the selector somehow didn't emit.

`dry_run_pipeline` then feeds a synthetic `AudioTask` through the IR to
catch any remaining missing-key errors.

### 5.6 Critics — code first, LLM second

Three layers:

```
SanityCritic  (deterministic)
   │   walks the IR + intent; raises ERROR/WARN/INFO findings against
   │   structural rules (single_speaker_clips_missing_separation,
   │   speaker_filter_no_bounds, quality_filter_without_score, ...)
   │
PlanCritic    (LLM)
   │   sees the IR, intent, profile, and user_locked_paths
   │   emits at most 5 findings, each with optional suggested_change
   │
apply_findings(intent, report)
   ├─ if patch.path in user_locked_paths
   │    → demote to INFO, code = "<orig>__user_locked"
   │      no suggested_change, rationale: "your form pick wins"
   ├─ if intent.model_validate(...) raises
   │    → demote to INFO, code = "<orig>__invalid_patch"
   │      rationale: schema rejected this value
   └─ if validator coerced the value back (e.g. single_speaker_clips → speech_segments)
        → demote to INFO, code = "<orig>__validator_coerced"
          rationale: "Validator coerced ... → ... see intent notes"
```

The orchestrator (`review_and_replan`) calls critics, applies findings,
re-plans once if anything actionable changed, and stops at
`max_iterations` (default 2). If a re-plan crashes, the original IR is
kept and a `replan_failed` finding is added.

---

## 6. End-to-end example

**Prompt:** *"make tts dataset"*

**User picks in the form:**
- `output.sample_rate` = 24000 (the default we suggested)
- `speakers.mode` = `filter`
- `speakers.target_count` = 1 (the new follow-up the `filter` option reveals)
- `quality.mos` = `filter`, `mos_threshold` = 4.0
- `text.transcript_source` = `generate`

**Trace through the files:**

| Step | File / function | What happens |
|---|---|---|
| 1 | `web.py :: plan` | Builds `SessionState`, calls extractor. |
| 2 | `planner_dag.extract_intent` | LLM call → `IntentCategories` with `raw_prompt="make tts dataset"`, sample_rate guessed from "tts". |
| 3 | `clarifier.infer_intent_from_prompt` | Regex sees "tts" → sets `output.sample_rate=24000`. |
| 4 | `clarifier.apply_profile_prefills` | No source given → no-op. |
| 5 | `smart_clarifier.analyze_gaps` | Emits gaps: `output.sample_rate` (always-asked), `speakers.mode` (TTS pivot), `quality.mos` (TTS pivot). |
| 6 | `smart_clarifier._llm_phrase_questions` | LLM phrases the three questions. `q_speakers.follow_ups = [q_speaker_count]`. |
| 7 | `web.py :: plan_response` | `smart_form.json` returned to the browser. |
| 8 | UI | User picks `filter`, the `q_speaker_count` follow-up appears, user picks 1. |
| 9 | `web.py :: build` → `clarifier.apply_answers` | Stamps `clarifier_answer:speakers.mode=filter`, `clarifier_answer:speakers.target_count=1`, etc. |
| 10 | `IntentCategories._consistency` | `speakers.mode=filter` + `output_unit=original_files` → fine. |
| 11 | `deterministic_planner.plan_from_intent` → `select_stages` | |
|  | `_output_normalizers` | sample_rate=24000 + resample → `ResampleAudioStage`, `MonoConversionStage`. |
|  | `_file_level_speaker_stages` | speakers.mode=filter → `InferenceSortformerStage(phase_override=preprocess)` + `PreserveByValueStage(num_speakers eq 1, phase_override=preprocess)`. |
|  | `_segmentation_stages` | output_unit=original_files, no speech_policy → no segmenter. |
|  | `_quality_stages` | quality.mos=filter, threshold=4.0 → `UTMOSFilterStage(mos_threshold=0.0)` + `PreserveByValueStage(utmos_mos ge 4.0)`. |
|  | `_text_stages` | transcript_source=generate → `InferenceAsrNemoStage`. |
|  | `_segment_extraction_stages` | output_unit=original_files → no SegmentExtraction. |
| 12 | `validator.validate` | Auto-inserts `ManifestWriterStage` at the end. Phase-reorders. |
| 13 | `dry_run_pipeline` | Synthetic `AudioTask` → no missing-key errors. |
| 14 | `SanityCritic.review` | Speaker filter has bounds (target_count=1) → no warning. UTMOS+PBV pair present → no warning. |
| 15 | `PlanCritic.review` (LLM) | Maybe raises `[plan] tts_no_word_timing` (info, no patch); maybe nothing. |
| 16 | `compiler.compile_ir_to_yaml` | Writes `compiled.yaml`. |

**Final stage list:**

```
ManifestReader
ResampleAudioStage
InferenceSortformerStage          ← phase_override=preprocess
PreserveByValueStage              ← num_speakers eq 1, phase_override=preprocess
MonoConversionStage
UTMOSFilterStage                  ← mos_threshold=0.0 (score-only)
PreserveByValueStage              ← utmos_mos ge 4.0  (the actual gate)
InferenceAsrNemoStage
ManifestWriterStage
```

This is the same shape you'll see in `<run_dir>/compiled.yaml`.

---

## 7. Run artefacts

Every run lives under
`$CURATOR_ADV_WEBSITE_DIR/<run_id>/`. The naming is stable, so you can
read any of these to diagnose a build.

| File | Written by | Read it when… |
|---|---|---|
| `prompt.txt` | `web.py` | You want to remember the exact wording. |
| `request.json` | `web.py` | You need the source URI / model override / cluster config the user submitted. |
| `intent_initial.json` | `web.py` (after extract + prefills) | The clarifier asked weird questions and you want to know what the LLM extractor wrote vs. what the heuristics added. |
| `profile.json` | `web.py` (if source was profileable) | The clarifier prefilled something unexpected. Check the dominant SR / channel histogram here. |
| `smart_form.json` | `web.py` | The questions look wrong. Compare the `template` shape against `smart_clarifier.template_questions`. |
| `intent.json` | `web.py` (post-build) | The pipeline is wrong. This is the *final* intent after answers + critic patches + validator coercions. Look at `notes` for the audit trail (`clarifier_answer:` and coercion notes). |
| `ir.json`, `ir.validated.json` | `deterministic_planner` | You want the raw stage list before vs. after validator auto-inserts. |
| `findings.json` | `validator` | A validator error blocked the build. Each finding has `code`, `detail`, `severity`. |
| `critic.json` | `plan_critic/orchestrator` | The critic UI shows something unexpected. Each finding has `severity`, `code`, `detail`, `suggested_change`, `rationale`. Codes ending in `__user_locked` / `__invalid_patch` / `__validator_coerced` were demoted. |
| `dry_run.json` | `dryrun` | Pipeline compiled but the dry-run flagged a missing key. |
| `compiled.yaml` | `compiler` | This is what would actually run on the cluster. |
| `llm_calls.jsonl` | `llm_transcript` | You want to read the actual LLM input/output for any call this run made. One JSON per line: `{purpose, model, tier, latency_ms, prompt_tokens, completion_tokens, messages, response}`. |

### The notes audit trail

`intent.json :: notes` is the single most useful file when something
goes wrong. Three kinds of notes:

| Prefix | Source | Meaning |
|---|---|---|
| `prompt_inference: ...` | `clarifier.infer_intent_from_prompt` | Heuristic explanation, e.g. *"TTS hint: defaulting output sample rate to 24 kHz."* |
| `clarifier_answer:<path>=<value>` | `clarifier.apply_answers` | The user picked this value in the form. **This is what makes the path user-locked.** |
| Free-form sentence ending in a `.` | `IntentCategories._consistency` or selector helpers | A coercion or degradation, e.g. *"speakers.mode=FILTER without target_count / min_count / max_count; selector will degrade to ANNOTATE."* |

When the pipeline looks wrong, read `notes` first. 90 % of bugs leave a
breadcrumb here.

---

## 8. Where the LLM is allowed to lie, and the safety nets

The LLM is wrong sometimes. Every LLM call has a structural guard that
catches the common failure modes:

| Failure mode | Guard | Where |
|---|---|---|
| Returns malformed JSON | Retry once at `temperature=0`; second failure raises `ValueError`. | `planner_dag._call_json` |
| Hallucinates a field outside the schema | `IntentCategories.model_validate` rejects; `extract_intent` raises. | `planner_dag.extract_intent` |
| Drops a question or changes an `intent_path` | We anchor on the template; LLM additions get matched by `id` / `intent_path` and dropped if neither matches. | `smart_clarifier._merge_llm_into_template` |
| Phrases the right question with a typo'd option value | Option `value` and `apply` come from the template, the LLM only owns the human-readable `label` / `description`. | `smart_clarifier._merge_llm_into_template` |
| Overrides a user-locked path | Demoted to INFO, code suffixed `__user_locked`. | `plan_critic.apply_findings` |
| Proposes an invalid enum value (`quality.mos = "preserve"`) | Patch skipped, demoted to INFO, code suffixed `__invalid_patch`. | `plan_critic.apply_findings` |
| Patches a value the validator will revert (`single_speaker_clips` with locked `speakers.mode=filter`) | Validator coerces back; finding demoted to INFO, code suffixed `__validator_coerced`. | `plan_critic.apply_findings` + `IntentCategories._consistency` |
| Suggests a stage / order / param | The LLM can't. `select_stages` and `validator` are the only writers; the critic only ever proposes intent patches. | by construction |

If a new failure mode shows up, the right place to add a guard is in
`apply_findings` (per-patch) or `_consistency` (per-intent-shape). Do
not let the critic LLM start writing stage params or order.

---

## 9. Debugging recipe

| Symptom | First file to open | What to look for |
|---|---|---|
| The form asked something dumb / didn't ask something important | `smart_form.json` | The `questions` array; compare against `analyze_gaps` policy. |
| The form didn't pre-select the value I expected | `smart_form.json :: questions[i].prefill` and the front-end `prepopulatePrefills` in `web.py`. |
| The compiled pipeline is missing a stage I wanted | `intent.json :: notes` (look for `clarifier_answer:` and degradation lines) → trace the relevant selector helper in `stage_selector.py`. |
| The stages are in the wrong order | `ir.json` (pre-reorder) vs `ir.validated.json` (post-reorder). Check stage cards' `phase` and `StageRef.phase_override`. |
| The critic complained about a thing that's actually fine | `critic.json :: findings` — look at the `code` suffix (`__user_locked` / `__invalid_patch` / `__validator_coerced`). |
| The critic patched something the user explicitly chose | This is the bug we fixed — should never happen. If it does, `intent.json :: notes` is missing the matching `clarifier_answer:` line — check `apply_answers`. |
| The LLM produced garbage | `llm_calls.jsonl` — find the call by `purpose` field (`extractor`, `smart_clarifier`, `plan_critic`), read the response. |
| The dry-run failed | `dry_run.json :: all_issues` — each issue has a stage name + index; the offending stage's `consumes` doesn't match the upstream `produces`. |
| Pipeline ran but produced wrong output | Not covered by this doc — check the executor / runtime logs (`runner.py` / cluster). |

---

## 10. Mental model — three sentences

**Intent is the contract.** Everything below the clarifier is a pure
function from `IntentCategories` → `compiled.yaml`. If the intent is
right, the YAML is right; if the YAML is wrong, the intent is wrong.

**The LLM only writes words.** Extract → fields. Clarifier → question
phrasing. Plan Critic → patch *proposals*. No LLM ever picks a stage,
chooses an order, or tunes a param.

**Audit notes are the truth.** Every heuristic, every form pick, every
coercion lands as a sentence in `intent.notes`. When something goes
wrong, that's the first place to read.
