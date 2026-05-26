# Intent V2 — Ingredient-Picker Design

Status: **Active design** (replaces the goal-anchored clarifier in `clarifier.py`
and the flat `IntentCategories` in `intent.py`).

This document is the source of truth for:

- the new `IntentCategoriesV2` Pydantic schema,
- the user-facing question DAG (no goal taxonomy; the user picks ingredients),
- the stage-application matrix the compiler reads to decide which catalog
  stages to add and in which mode.

The redesign keeps the existing 28-stage audio catalog intact. No new stages
are required.

---

## 1. Core idea — `FilterMode`

Every analytical module today does an implicit job. The redesign makes the
job explicit: each module has at most four states.

```
FilterMode = OFF | ANNOTATE | FILTER | SPLIT
```

| Mode | What it means | Catalog mapping |
|---|---|---|
| `OFF` | Stage not added. | – |
| `ANNOTATE` | Stage runs; its output keys go into `task.data`; nothing is dropped. | UTMOS/SIGMOS with `threshold=0.0`; VAD with `nested=True`; Sortformer/PyAnnote always behaves this way today. |
| `FILTER` | Stage runs; rows that fail the threshold are dropped. Row count goes down, but each surviving row is still one task per original file. | UTMOS/SIGMOS with real threshold; `PreserveByValueStage` chained after `InferenceSortformerStage`. |
| `SPLIT` | Stage runs and produces N tasks per input (per-segment or per-speaker). Row schema *changes* — one row per segment, or per speaker. | VAD with `nested=False` (+ `SegmentExtractionStage`); `SpeakerSeparationStage`. |

This is the only way the compiler is allowed to decide whether to run an
analytical stage and how. Stage selection is **never** triggered by an
implicit "a capability is required", it is always triggered by a specific
mode value on a specific intent field.

---

## 2. The new schema (`IntentCategoriesV2`)

```python
class FilterMode(str, Enum):
    OFF = "off"
    ANNOTATE = "annotate"
    FILTER = "filter"
    SPLIT = "split"


class OutputFormat(BaseModel):
    sample_rate: int | Literal["any"] | None = None        # Hz, "any", or unset
    channels: Literal["mono", "stereo", "any"] | None = None
    audio_format: Literal["wav", "flac", "ogg"] | None = "wav"
    resample_input: bool | None = None                      # gated follow-up to sample_rate


class Segmentation(BaseModel):
    output_unit: Literal[
        "original_files",
        "speech_segments",
        "long_windows",
        "single_speaker_clips",
    ] = "original_files"
    duration_min_sec: float | None = None
    duration_max_sec: float | None = None
    speech_policy: FilterMode = FilterMode.OFF              # OFF / ANNOTATE / FILTER
    long_window_sec: float | None = None                    # required when output_unit=long_windows
    vad_threshold: float | None = None                      # raw VAD model knob (advanced)
    speech_pad_ms: int | None = None                        # advanced; ms padding


class Quality(BaseModel):
    mos: FilterMode = FilterMode.OFF                        # UTMOS; compiles to PBV(utmos_mos, ge, mos_threshold)
    mos_threshold: float | None = None                      # used only when mos == FILTER
    sigmos: FilterMode = FilterMode.OFF                     # compiles to one PBV(sigmos_<axis>, ge, threshold) per axis
    sigmos_axes: list[Literal[
        "ovrl", "noise", "sig", "col", "disc", "loud", "reverb",
    ]] = []
    sigmos_thresholds: dict[str, float] = {}                # axis → threshold (ge floor)
    band: FilterMode = FilterMode.OFF                       # OFF or FILTER (no ANNOTATE today)
    band_value: Literal["narrow_band", "full_band"] | None = None
    band_min_hz: float | None = None                        # informational only
    gates: list[QualityGate] = []                           # free-form PBV rows; only path to lt/gt/ne/non-ge


class QualityGate(BaseModel):
    """A single PreserveByValueStage row against a quality-score key.

    Use this whenever the user's prompt implies an operator OTHER than
    the default 'keep above this floor' (ge). The selector always emits
    UTMOS/SIGMOS with threshold=0.0, so the operator lives entirely in
    the gate.
    """

    key: Literal[
        "utmos_mos", "sigmos_ovrl", "sigmos_noise", "sigmos_sig",
        "sigmos_col", "sigmos_disc", "sigmos_loud", "sigmos_reverb",
        "band_prediction",
    ]
    operator: Literal["lt", "le", "eq", "ne", "ge", "gt"]
    value: float | str                                      # str only valid for band_prediction


class Speakers(BaseModel):
    mode: FilterMode = FilterMode.OFF                       # OFF/ANNOTATE/FILTER/SPLIT
    target_count: int | None = None                         # FILTER: keep == target   (PreserveByValue eq)
    min_count: int | None = None                            # FILTER: keep >= min      (PreserveByValue ge)
    max_count: int | None = None                            # FILTER: keep <= max      (PreserveByValue le)
    exclude_overlaps: bool = True                           # SPLIT param


class TextPolicy(BaseModel):
    transcript_source: Literal["off", "generate", "existing"] = "off"
    asr_backend: Literal["nemo", "whisper", "auto"] = "auto"
    word_timing: bool = False
    wer_mode: FilterMode = FilterMode.OFF                   # off / annotate / filter
    wer_max: float | None = None


class Policy(BaseModel):
    commercial_only: bool = False
    privacy_mode: bool = False
    budget_gpu_hours: float | None = None
    budget_disk_gb: float | None = None


class IntentCategoriesV2(BaseModel):
    raw_prompt: str | None = None
    output: OutputFormat = OutputFormat()
    segmentation: Segmentation = Segmentation()
    quality: Quality = Quality()
    speakers: Speakers = Speakers()
    text: TextPolicy = TextPolicy()
    policy: Policy = Policy()
    notes: list[str] = []
```

There is **no `goal` / `task_profile` field** by design. The user picks
ingredients; defaults come from the prompt + dataset profile + sensible
per-question fallbacks.

---

## 3. Question DAG (user-facing)

Sections are independent; only follow-ups inside a section are gated.
The clarifier returns the whole form at once (one round-trip after the
prompt), with profile-suppressed questions hidden and inferred answers
pre-filled with a "source" tag (`prompt`, `profile`, `default`).

### A. Output format

| ID | Prompt | Options |
|---|---|---|
| `output.sample_rate` | "What output sample rate?" | `8` kHz · `16` · `22.05` · `24` · `32` · `44.1` · `48` · `keep input rates` · **freeform kHz** |
| `output.resample_input` *(visible when `sample_rate` is a concrete number)* | "Resample input files to this rate?" | `Yes — convert all files` · `No — keep matching, drop mismatched` |
| `output.channels` | "Channel layout?" | `mono` · `stereo` · `match input` · `skip` |
| `output.audio_format` | "Output audio file format?" | `wav` · `flac` · `ogg` · `skip (manifest only)` |

Profile suppression rules:
- Skip `sample_rate` entirely if `DatasetCard.profile.sample_rates_hz` has one
  key and `prompt` doesn't mention a rate (we pre-fill `sample_rate=keep`).
- Skip `channels` if `channel_distribution` is a single bucket and prompt
  doesn't mention channels.

### B. Segmentation & speech

| ID | Prompt | Options |
|---|---|---|
| `segmentation.output_unit` | "What does one output row represent?" | `original file` · `VAD speech segment` · `long-audio window` · `single-speaker clip` |
| `segmentation.duration_min_sec` & `duration_max_sec` *(visible when output_unit ≠ original_files)* | "Clip duration limits?" | preset `2–60` (TTS) · `2–30` (ASR) · `5–120` (long) · `no limit` · **freeform min,max** |
| `segmentation.speech_policy` *(visible when output_unit = original_files)* | "Drop files with no speech?" | `off` · `annotate only` · `filter (drop empty)` |
| `segmentation.long_window_sec` *(visible when output_unit = long_windows)* | "ALM window length?" | `30 s` · `60 s` · `120 s` · **freeform sec** |

Backend rule: see §4.

### C. Quality (annotation vs. filter)

| ID | Prompt | Options |
|---|---|---|
| `quality.mos` | "Naturalness MOS gate?" | `off` · `annotate only` · `filter — strict (≥4.0)` · `filter — balanced (≥3.4)` · `filter — loose (≥3.0)` · **freeform threshold** |
| `quality.sigmos` *(visible when `mos` ≠ off)* | "Multi-axis SIGMOS quality gate?" | `off` · `annotate only` · `filter — overall + noise` · `filter — broadcast (overall+noise+col+loud)` · `filter — studio (all axes strict)` · **freeform per-axis** |
| `quality.band` | "Bandwidth filter?" | `off` · `filter — narrow-band only (phone)` · `filter — full-band only (studio)` |

Profile suppression: skip the question block if `prompt` doesn't mention
any quality keywords and `DatasetCard.profile` doesn't suggest noise/decode
issues. Default everything to `OFF`.

### D. Annotations

| ID | Prompt | Options |
|---|---|---|
| `text.transcript_source` | "Transcripts?" | `off` · `generate via ASR` · `use existing manifest column` |
| `text.word_timing` *(visible when transcript_source ≠ off)* | "Word-level timestamps?" | `yes` · `no` |
| `text.asr_backend` *(visible when transcript_source = generate)* | "ASR backend?" | `auto` · `nemo` · `whisper` |
| `text.wer_mode` *(visible when transcript_source ≠ off and existing transcript present)* | "WER quality gate vs. existing transcript?" | `off` · `annotate WER` · `filter — drop WER > X` |
| `speakers.mode` | "Speaker handling?" | `off` · `annotate (diarize)` · `filter — exactly N` · `filter — at most N` · `split — fan out per speaker` |
| `speakers.target_count` / `max_count` *(visible when mode = filter)* | "Speaker count?" | `1` · `2` · `3` · **freeform N** |

### E. Policy

| ID | Prompt | Options |
|---|---|---|
| `policy.commercial_only` | "Commercial use?" | `yes` · `no` · `skip` |
| `policy.privacy_mode` | "Privacy mode?" | `yes` · `no` |

### F. Augmentation (not currently executed)

Recorded as a note. Always returns `notes += [...]`. Listed as the last
section so it's clearly opt-in.

---

## 4. Stage-application matrix (compiler contract)

This table replaces `nemo_curator.agentic.intent.required_capabilities`.
The compiler walks the intent and emits **at most one** matching row from
each section. Order in the IR follows the `Phase` enum from `cards.py`.

### Sources / sinks (always added when not already present)

| Always | Reason |
|---|---|
| `ManifestReader` (or appropriate dataset-create) | source from `SourceSpec` |
| `ManifestWriterStage` | sink; auto-inserted by validator if missing |

### Output normalization

| Intent condition | Stages |
|---|---|
| `output.sample_rate is int` AND `resample_input == True` | `ResampleAudioStage(target=sr)` → `MonoConversionStage(output_sample_rate=sr)` |
| `output.sample_rate is int` AND `resample_input == False` | `MonoConversionStage(strict_sample_rate=True, output_sample_rate=sr)` |
| `output.channels == "mono"` only (no sample rate) | `MonoConversionStage(output_sample_rate=48000)` |
| `output.channels == "stereo"` | not implemented today — record as a note |

### Segmentation / speech

| Intent condition | Stages | Resulting row shape |
|---|---|---|
| `segmentation.output_unit == "speech_segments"` | `VADSegmentationStage(nested=False, min/max)` → `SegmentExtractionStage` | per-segment |
| `segmentation.output_unit == "original_files"` AND `speech_policy == ANNOTATE` | `VADSegmentationStage(nested=True, min_duration_sec=0.5)` | per-file (with `segments` key) |
| `segmentation.output_unit == "original_files"` AND `speech_policy == FILTER` | same as ANNOTATE; **TODO:** add `PreserveByValueStage(num_segments ge 1)` once the catalog has a list-length filter | per-file |
| `segmentation.output_unit == "long_windows"` | `SplitLongAudioStage(window_sec=long_window_sec)` → optionally `ALMDataBuilderStage` | per-window |
| `segmentation.output_unit == "single_speaker_clips"` | `VADSegmentationStage(nested=False)` → `SpeakerSeparationStage` → `SegmentExtractionStage` | per-speaker-clip |

### Quality — score → gate split

The selector now treats `UTMOSFilterStage` and `SIGMOSFilterStage` as
**pure annotators** (all thresholds pinned to `0.0`) and pushes every
drop decision into a downstream `PreserveByValueStage`. This gives the
user the full operator matrix on every score key — `ge` covers the
"keep clean" base case, but `lt` / `gt` / `eq` / `ne` are first-class
now too. The implementation lives in `_quality_stages` /
`_quality_gate_to_pbv` in `stage_selector.py`.

| Intent condition | Stages emitted |
|---|---|
| `quality.mos == ANNOTATE` or `mos == FILTER` or any `gates` entry on `utmos_mos` | `UTMOSFilterStage(mos_threshold=0.0)` |
| `quality.sigmos == ANNOTATE` / `FILTER` or any `gates` entry on `sigmos_*` | `SIGMOSFilterStage` with each relevant axis threshold pinned to `0.0` |
| `quality.band == FILTER` AND `band_value` set | `BandFilterStage(band_value=...)` *(no annotate mode in the catalog)* |
| `quality.mos == FILTER` (legacy floor) | + `PreserveByValueStage(utmos_mos, ge, mos_threshold or 3.4)` |
| `quality.sigmos == FILTER` (legacy floors) | + one `PreserveByValueStage(sigmos_<axis>, ge, threshold or 3.5)` per requested axis |
| `quality.band == FILTER` (legacy class) | + `PreserveByValueStage(band_prediction, eq, band_value)` |
| Every `quality.gates[i]` | + `PreserveByValueStage(gate.key, gate.operator, gate.value)` (appended LAST, so user-specified operators trump the legacy `ge` defaults) |

Notes:
* `BandFilterStage` is still emitted alongside its PBV mirror because
  the stage has no `score_only` knob; the PBV row exists for IR
  uniformity (drop logic is always visible at the same node type).
* `quality.gates` is the only path to non-`ge` operators (`lt`, `le`,
  `gt`, `ne`). Use it for "drop high quality", "noise below X", etc.

### Annotations

| Intent condition | Stages |
|---|---|
| `text.transcript_source == "generate"` AND `text.word_timing == True` | `NeMoASRAlignerStage` (replaces standalone ASR) |
| `text.transcript_source == "generate"` AND `text.word_timing == False` | `InferenceAsrNemoStage` (or WhisperX VAD chain if `asr_backend == whisper`) |
| `text.transcript_source == "existing"` | nothing added; downstream stages may consume the existing `text` column |
| `text.wer_mode == ANNOTATE` | `GetPairwiseWerStage` |
| `text.wer_mode == FILTER` | `GetPairwiseWerStage` → `PreserveByValueStage(pairwise_wer le wer_max)` |
| `speakers.mode == ANNOTATE` | `InferenceSortformerStage` |
| `speakers.mode == FILTER` AND `target_count` set | `InferenceSortformerStage` → `PreserveByValueStage(num_speakers eq target_count)` |
| `speakers.mode == FILTER` AND `min_count` set (and `target_count` unset) | `InferenceSortformerStage` → `PreserveByValueStage(num_speakers ge min_count)` |
| `speakers.mode == FILTER` AND `max_count` set (and `target_count` unset) | `InferenceSortformerStage` → `PreserveByValueStage(num_speakers le max_count)` |
| `speakers.mode == FILTER` AND both `min_count` + `max_count` set | `InferenceSortformerStage` → `PreserveByValueStage(ge)` → `PreserveByValueStage(le)` (closed range) |
| `output_unit == original_files` AND `duration_min_sec` set (no cleaning flow) | `PreserveByValueStage(duration ge duration_min_sec)` |
| `output_unit == original_files` AND `duration_max_sec` set (no cleaning flow) | `PreserveByValueStage(duration le duration_max_sec)` |
| `speakers.mode == SPLIT` | `SpeakerSeparationStage(exclude_overlaps=exclude_overlaps)` |

### Policy / housekeeping

| Intent condition | Effect |
|---|---|
| `policy.commercial_only == True` | License gate in validator stays as today. |
| `policy.privacy_mode == True` | Future: passed to model loaders to disable downloads. |
| `policy.budget_gpu_hours` / `budget_disk_gb` | Logged as findings; runner enforces. |

---

## 5. Prompt-language → ingredient inference

The clarifier uses the *full prompt* (and `DatasetCard`) to pre-fill
ingredients before showing the form. There is no "task profile" anymore;
each ingredient is inferred independently.

| Heuristic | Pre-fills |
|---|---|
| Prompt mentions `tts`, `text-to-speech`, `voice clone` | `speakers.mode=SPLIT`, `output.sample_rate=16000`, `quality.mos=FILTER (3.4)` |
| Prompt mentions `phone`, `call`, `telephone`, `narrowband` | `quality.band=FILTER`, `band_value=narrow_band`, `output.sample_rate=16000` |
| Prompt mentions `transcript`, `transcribe`, `subtitles` | `text.transcript_source=generate` |
| Prompt mentions `speaker labels`, `who said`, `meeting`, `diarize` | `speakers.mode=ANNOTATE` |
| Prompt mentions `chunks`, `audio-language`, `alm` | `output_unit=long_windows`, `text.transcript_source=generate`, `text.word_timing=True` |
| Prompt mentions `speech is there`, `drop empty`, `at least speech`, `no silence` | `speech_policy=FILTER` (or `ANNOTATE` if `output_unit ≠ original_files`) |
| Prompt mentions `clean`, `studio`, `broadcast`, `high quality` | `quality.mos=FILTER (matching band)` |
| Profile says all files share the same SR | `sample_rate=keep`, `resample_input=False` |
| Profile says all files are mono | `channels=mono`, no MonoConversionStage if SR also matches |
| Profile has `text` column non-empty in samples | `text.transcript_source=existing` |
| Profile has `num_speakers` column non-empty | `speakers.mode=annotate` is pre-checked but still asked |

Each pre-fill is tagged with `{source: "prompt"|"profile"|"default", confidence: 0..1, reason: "..."}` so the UI can render the chip
"inferred from your prompt" next to the option.

---

## 6. Web flow

One round-trip:

1. `POST /api/plan` with `{prompt, dataset, kind, out, tier}`.
2. Server returns `{session_id, form, dataset_profile_summary}`. `form` is
   a list of sections with questions, options, freeform schemas, and
   pre-filled values.
3. User adjusts (or hits "Build with these defaults").
4. `POST /api/build` with `{session_id, intent}` — server compiles and returns YAML.

No clarification loop, no "still pending" trap. The whole form is shown
once; pre-filled and profile-suppressed values stay visible (with the
"why" chip) so the user can still flip them.

---

## 7. Migration plan

1. Land `IntentCategoriesV2` alongside the legacy `IntentCategories`.
2. Add `select_stages(intent_v2, registry, source)` in a new module
   `nemo_curator/agentic/stage_selector.py` per §4.
3. Rewrite `clarifier.py` around the question DAG in §3; profile-suppress
   per §5. The new clarifier returns one form, no follow-up rounds.
4. Migrate `deterministic_planner.plan_from_intent` to call
   `select_stages` instead of `required_capabilities + _select_stage_names`.
5. Migrate `planner_dag.extract_intent`'s prompt to return the new
   namespaced JSON; keep the legacy schema for `team/` planners as a
   deprecation island until they migrate.
6. Rewrite `web.py` UI to render the namespaced form (§6).
7. Migrate `tests/agentic/*` per the new schema; add filter↔annotate
   tests, freeform tests, profile-suppression tests.
8. Remove `IntentCategories`, `infer_task_profile`, `_defaults_for_profile`,
   `_forced_keys_for_profile`, and `required_capabilities` once nothing
   imports them.

---

## 8. Heaviness-aware resource allocation

The tuner (`nemo_curator/agentic/tuner.py`) decides three things per
compile, all driven by stage-card metadata plus the user-supplied
`ClusterProfile`:

1. **`gpus_per_worker` (the fraction)** — VRAM packing:

   ```
   vram_needed     = est_vram_inference_gb × 1.30
   pack            = clamp(floor(gpu_memory_gb / vram_needed), 1, 8)
   gpus_per_worker = 1 / pack
   ```

   When the card lacks `est_vram_inference_gb` (no inspector run yet),
   the fallback is the gpu_class default: 1.0 for `full` stages, 0.5
   for `frac` stages.

2. **`num_workers` (the worker count)** — weighted fair-share by
   `cost_weight`:

   ```
   slice_i      = cluster.gpus × cost_weight_i / Σ cost_weight
   num_workers_i = max(1, floor(slice_i / gpus_per_worker_i))
   ```

   `cost_weight` is derived from `params_total` via a piecewise
   `log10` map. Fallbacks: 5 (full), 2 (frac), 1 (cpu).

3. **`execution_mode`** — *not* a user input. The tuner auto-picks:

   ```
   floor = Σ gpus_per_worker_i  (min concurrent demand, 1 worker each)
   streaming  if floor ≤ cluster.gpus
   batch      otherwise
   ```

### Refreshing card heaviness numbers

The card fields are populated by an offline CLI:

```
python -m nemo_curator.agentic.inspect_models --all
python -m nemo_curator.agentic.inspect_models --stage InferenceAsrNemoStage
python -m nemo_curator.agentic.inspect_models --dry-run --all
```

The inspector handles HuggingFace-hosted models (`.nemo` archives,
`safetensors`, `pytorch_model.bin`). torch_hub / ONNX / private model
hosts are skipped — those entries must be filled manually if their
heaviness matters. Updates land in `stage_card.yaml` files in-place and
are reviewable in PR.

### Audit invariant

After every allocation, the tuner sums the concurrent demand and
attaches a warning chip to `executor_config.tuner_reasons` if:

```
Σ over concurrent stages (gpus_per_worker_i × num_workers_i) > cluster.gpus
```

This is informational — the executor's own scheduler will throttle —
but it tells the user the cluster is the bottleneck before they hit
runtime back-pressure.

## 9. Open items (out of this iteration)

- Adding `score_only` to `BandFilterStage` so band ANNOTATE mode becomes
  real. Track as a separate stage-card change.
- Adding `PreserveByValueStage`-style list-length filter so
  `speech_policy=FILTER` truly drops empty files on whole-file rows. Today
  we run VAD nested and document the limitation in a note.
- Augmentation stages — still GAP. Notes recorded only.
- Language ID / multilingual filtering — still GAP.
- SNR / loudness normalization — still GAP.
- torch_hub / ONNX model inspector — currently manual fallback.
- Throughput / latency profiling pass to refine `cost_weight` from
  measured items-per-second instead of parameter counts.
