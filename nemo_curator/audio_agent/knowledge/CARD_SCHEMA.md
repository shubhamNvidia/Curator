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
| `tags` | rec. | capability flags: `needs_gpu`, `needs_ffmpeg`, `needs_internet_first_run`, `needs_hf_token`, `writes_disk`, `sink`, `sanitizes_output`, `produces_score`, `is_filter`, `fanout`, `batch_only`. |
| `model_id` | if model | model identifier, else `null`. |
| `model_version` | if `model_id` | pinned revision/entrypoint so a silent model change is detectable. `TODO(fill)` allowed if unpinned in code. |
| `domain` | rec. | `{language, style}` — usually `best_guess`. |
| `constraints` | rec. | only **real** facts: `supported_sample_rates`, `max_speakers`, `batch_size:{fixed,reason}`, `input_duration_sweetspot_sec:{min,max}`. |
| `resource` | rec. | `{cpus, gpu_mem_gb, host_mem_gb, gpu_optional, bound: cpu\|gpu\|io, throughput_hint, disk_expansion}`. Feeds the resource planner. |
| `use_cases` | rec. | `{good_for: [...], avoid_for: [...]}` — `best_guess`. |
| `composition` | rec. | `{typical_upstream: [...], typical_downstream: [...]}` — idiomatic ordering. |
| `params_of_note` | rec. | `{param: description}` — **keys must be real constructor params** (gate-checked). |
| `presets` | opt. | `{name: {param: value}}` — **keys must be real params** (gate-checked). |
| `conflicts_with` | opt. | stage_ids that are alternatives / shouldn't co-occur. |
| `param_dependencies` | opt. | notes on params that depend on each other or on upstream data. |
| `comparison` | opt. | disambiguation fields for overlapping modules: `{language_support, accuracy_hint, latency_hint, config_complexity, known_limitations}`. |
| `notes` / `caveats` | opt. | free text. |
| `provenance` | rec. | `{model_card_url, card_version, last_validated}`. |

## Gate (what is enforced)

`python -m nemo_curator.audio_agent.card_conformance` fails if any card:
- has a `stage_id` that doesn't resolve, or a missing required field;
- lists a `params_of_note` / `presets` key that isn't a real constructor param;
- uses an unknown `resource` key / non-numeric numeric / bad `bound`;
- sets `model_id` without a `model_version`;
- uses a `verified` tier outside `{mechanical, measured, best_guess}`.

Coverage gaps (stages with no card) are **reported, not failed** — they name the authoring backlog.
