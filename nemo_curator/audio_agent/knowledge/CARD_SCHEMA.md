<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Capability Card Schema (v2)

A capability **card** is the agent's factual, code-independent view of one stage. It lets
the host LLM select, configure, and compose a stage **without reading source**. Cards live
in `knowledge/cards/<name>.yaml`, one per stage, keyed by `stage_id`.

The mechanical facts in a card are enforced against the real stage by the **conformance
gate** (`card_conformance.py`), so a card can never drift from the code.

## Golden rule: never fabricate

Fill only what you can verify from the stage's code/contract or an authoritative model card.
If you don't know a value, **leave it empty (null / omit) with a `# TODO(fill): <why>` comment**
so a human fills it later. A wrong fact is worse than a missing one — the agent trusts cards.

## Honesty tiers (`verified`, required)

Every card declares how each fact group was established:

| tier         | meaning                                                              | examples                          |
|--------------|---------------------------------------------------------------------|-----------------------------------|
| `mechanical` | derived from code/contract; the gate re-checks it                   | `params`, `tags`, `category`      |
| `measured`   | from a real run/benchmark on known hardware/data                    | `resource.gpu_mem_gb`, throughput |
| `best_guess` | author judgment, not yet verified                                   | `use_cases`, `domain`, sweetspots |

```yaml
verified: {params: mechanical, resource: best_guess, model_version: measured, use_cases: best_guess}
```

## Fields

| field | required | notes |
|-------|----------|-------|
| `stage_id` | yes | exact stage class name (must resolve). |
| `category` | yes | one taxonomy category (`ingest`/`preprocess`/`segment`/`diarize`/`transcribe`/`quality`/`filter`/`text_norm`/`export`/`alm`). |
| `summary` | yes | one line: what it does. |
| `tags` | rec. | capability flags: `needs_gpu`, `needs_ffmpeg`, `needs_internet_first_run`, `needs_hf_token`, `writes_disk`, `sink`, `sanitizes_output`, `produces_score`, `is_filter`, `fanout`, `batch_only`. A tag states **default** behavior — a capability reachable only by setting a non-default param is a knob, not a tag (`needs_gpu` is wrong for a stage whose gate is `requires_gpu=self.resources.gpus > 0`; that is `resource.gpu_optional: true`). `fanout` marks a **mid-pipeline granularity change** (one row in, many rows out, so downstream rows describe a different entity) — a *source* that reads a folder or manifest into N rows does not take it, since there is no upstream entity to be confused with. Only `writes_disk` and `needs_ffmpeg` are gate-checked, and only forward (tag present → gate must be true); a **missing** tag is not caught by anything. |
| `model_id` | if model | model identifier, else `null`. |
| `model_version` | if `model_id` | pinned revision/entrypoint so a silent model change is detectable. `TODO(fill)` allowed if unpinned in code. |
| `deterministic` | opt. | `false` when the same inputs can legitimately give different output (unseeded randomness, order-dependent decode). Omitted means reproducible. Execution reuse offers a stored result from such a stage with the caveat shown and *fresh* pre-selected, rather than serving it silently — so declaring it costs a prompt, not the feature. |
| `domain` | rec. | `{language, style}` — usually `best_guess`. |
| `constraints` | rec. | only **real** facts: `supported_sample_rates`, `max_speakers`, `batch_size:{fixed,reason}`, `input_duration_sweetspot_sec:{min,max}`. |
| `resource` | rec. | `{cpus, gpu_mem_gb, host_mem_gb, gpu_optional, bound: cpu\|gpu\|io, throughput_hint, disk_expansion}`. Feeds the resource planner. |
| `use_cases` | rec. | `{good_for: [...], avoid_for: [...]}` — `best_guess`. |
| `composition` | rec. | `{typical_upstream: [...], typical_downstream: [...]}` — idiomatic ordering. Every stage id must resolve (gate-checked). Optional `incompatible_upstream: {stage_id: why}` records a pairing that is known to *break*, not merely be unidiomatic — a stage id may not appear in both, and the reason is required, because "don't use X" without "because X writes a list where this reads an int" is a rule the next reader will discount. |
| `params_of_note` | rec. | `{param: description}` — **keys must be real constructor params** (gate-checked). |
| `presets` | opt. | `{name: {param: value}}` — **keys must be real params** (gate-checked). |
| `metrics` | opt. | `{metric: {scale:{min,max,direction}, threshold_param, valid_range:[lo,hi], presets}}` — the deterministic source of score directions/targets (1A.2) that keeps a filter from being inverted. `scale.direction` must be `higher_better`/`lower_better`; `threshold_param` (if set) must be a real param; `valid_range` must be `[lo, hi]` (all gate-checked). Omit `threshold_param` for annotate-only stages that are filtered downstream (e.g. `ComputeWERStage` → `PreserveByValueStage`). |
| `semantic_facts` | opt. | Advisory mapping from an externally consumed output/concept to prose facts such as `{meaning, unit, provenance, scope, propagation, counterexamples}`. It helps the host reason across filters, fan-out and aggregation; it is deliberately not a deterministic ontology or runtime gate. |
| `versions` | opt. | model-backed stages only: `{model_id: "when-to-use"}` for checkpoints verified interchangeable via `model_name`/`model_path` (same output structure, no module code change). Any version-selecting `preset` (one that sets `model_name`/`model_path`) **must list its model id here** (gate-checked). Mark `verified.versions` `measured` when empirically tested. |
| `conflicts_with` | opt. | stage_ids that are alternatives / shouldn't co-occur. |
| `param_dependencies` | opt. | notes on params that depend on each other or on upstream data. |
| `comparison` | opt. | disambiguation fields for overlapping modules: `{language_support, accuracy_hint, latency_hint, config_complexity, known_limitations}`. |
| `notes` / `caveats` | opt. | free text. |
| `provenance` | rec. | `{model_card_url, card_version, last_validated}`. |

## Semantic output facts (LLM reasoning layer)

Constructor params and semantic roles make a stage mechanically connectable;
they do not tell the host what an output means at a particular pipeline
position. Document that meaning for outputs that a downstream stage can filter,
compare, aggregate or otherwise interpret. This is especially important when a
stage changes cardinality (`1:N`, nested output, `N:1`) or preserves a value
whose entity is different from the emitted row.

`semantic_facts` is an optional organization aid, not a fixed semantic ontology.
Each value may be one compact prose string or the richer mapping below. The gate
checks only that the YAML is readable prose; `scope` is not an enum and the
validator does not turn it into a pipeline pass/fail rule.

```yaml
semantic_facts:
  some_output_key:
    meaning: "What the value represents."
    unit: "seconds (float)"
    provenance: "Computed from the original input clip before fan-out."
    scope: "original input clip; parent-level aggregate"
    propagation: "Copied unchanged onto every emitted child row."
    counterexamples:
      - "Filtering this on a child does not measure a property recomputed for that child."
```

For every externally interpreted output, cover:

1. **Meaning** — describe the concept, not just its storage key.
2. **Unit/range/vocabulary** — include boundaries and direction when relevant.
3. **Provenance** — name the input entity and computation/model that produced it.
4. **Scope/granularity** — file, original parent, child, segment, speaker, batch
   or corpus, expressed as precise prose.
5. **Propagation** — explain copying, recomputation, nesting, fan-out and
   aggregation behavior.
6. **Counterexamples** — give at least one tempting wrong interpretation and
   what a downstream filter/consumer would actually do.

Use `notes`/`caveats` instead when a fact spans multiple fields. Do not fabricate
missing semantics and do not grow a per-module Python rule table to compensate:
leave a TODO and make the host ask. Mark code-grounded facts
`verified.semantic_facts: mechanical`; use `measured` or `best_guess` honestly
when that is how the claim was established.

## `resource.gpu_mem_gb` is a reference, not a per-GPU constant

A stage's VRAM tracks the *workload* (model weights + activations for a given precision,
batch size, and input length), **not** the GPU model — a 5 GB stage needs ~5 GB on any
card; the GPU's total VRAM only decides whether it *fits*. So `gpu_mem_gb` is a **reference
estimate**, and each measured value records its conditions in the comment
(e.g. `# measured on RTX 4090, fp32, batch 4, 6 s clip`). It shifts with **precision**
(fp16/bf16 ≈ half), **batch size / audio length**, and (minor) framework/arch. The planner
probes the *actual* machine's VRAM per run and checks each need against it; the **1C.2
calibration** path (`smoke` → `calibrate` → `run --calibration`) re-measures on the real
box and **overrides** the card per machine (measured > card), so the card value is a sane
default, not a hard truth.

## Gate (what is enforced)

`python -m nemo_curator.audio_agent.card_conformance` fails if any card:
- has a `stage_id` that doesn't resolve, or a missing required field;
- lists a `params_of_note` / `presets` key that isn't a real constructor param;
- uses an unknown `resource` key / non-numeric numeric / bad `bound`;
- sets `model_id` without a `model_version`;
- declares a `versions` block that isn't a `{model_id: string}` map, sets it without a `model_id`, or has a model-selecting `preset` (sets `model_name`/`model_path`) whose model id isn't documented in `versions`;
- declares a `metrics` block with a bad `scale.direction`, a `threshold_param` that isn't a real param, or a `valid_range` that isn't `[lo, hi]`;
- declares `semantic_facts` with a non-mapping top level, non-prose values, or malformed `counterexamples`;
- declares `semantic_facts` without a corresponding
  `verified.semantic_facts` evidence tier, or uses a `verified` tier outside
  `{mechanical, measured, best_guess}`.

`semantic_facts`, `notes`, `use_cases` and other intent-facing prose are
advisory reasoning material, not deterministic validation rules. Review them
against code/model evidence when authoring the card; green conformance means the
mechanical surface is honest, not that every intended use is semantically sound.

Coverage gaps (stages with no card) are **reported, not failed** — they name the authoring backlog.
