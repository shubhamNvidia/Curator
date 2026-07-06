# Making an Audio Stage Agent-Ready

This is the **short** checklist for stage owners. "Agent-ready" means an LLM agent can
**discover** your stage, **configure** it, and **chain** it with others to build a pipeline —
without reading your source. The design goal is **minimal burden**: you declare a small core,
the framework auto-derives the rest, and one test tells you if anything is missing.

> **Golden rule:** every new knob defaults to today's behavior. Agent-readiness must not change
> how your stage runs in existing pipelines.

---

## TL;DR — the whole job is 3 things

1. Inherit `AgentReady` and implement **`describe()`** returning a `StageContract` with
   **`reads`, `writes`, `cardinality`** (+ honest **`gates`**).
2. Make every `task.data` key you read/write a **`*_key` constructor field** (no bare key
   literals in `process()`).
3. Add one test: **`assert_agent_ready(MyStage(...), fixture_factory=...)`**.

Everything else is auto-derived or optional. If `assert_agent_ready` passes, you're done.

---

## What you MUST declare (only you know these)

```python
from dataclasses import dataclass
from nemo_curator.stages.audio._agent_ready import AgentReady, StageContract, IOSpec, Gates
from nemo_curator.stages.base import ProcessingStage

@dataclass
class MyStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """One-line summary becomes the agent-facing description.

    Args:
        audio_filepath_key: Path to the input audio.   # docstring Args -> param descriptions
        score_key: Where the score is written.
    """
    audio_filepath_key: str = "audio_filepath"
    score_key: str = "my_score"
    name: str = "MyStage"

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
            writes=IOSpec(data_keys=[self.score_key]),
            cardinality="1:1",                       # see "Cardinality" below
            gates=Gates(requires_gpu=self.resources.gpus > 0),
        )
```

- **`reads` / `writes`** — the `task.data` keys consumed/produced. Use `segment_data_keys` for keys
  written *inside* `segments[]` items. Use `reads_one_of=[IOSpec(...), ...]` if input can take more
  than one shape (e.g. waveform **or** file).
- **`cardinality`** — one of `"1:1"`, `"1:1 nested-list"`, `"1:N fan-out"`, `"N:1"`, `"filter"`.
  (`"filter"` = `process` may return `None`/`[]` to drop items.)
- **`gates`** — be honest about side effects: `requires_gpu`, `writes_to_disk`,
  `requires_internet_first_run`, `requires_ffmpeg`. Serializability (both exist on `Gates`): a sink
  that `json.dumps` `task.data` as-is must set `requires_serializable_input=True`; a converter that
  strips tensors/audio blobs sets `sanitizes_output=True`.

## What is AUTO-DERIVED — do NOT hand-write these

| Field | Derived from |
|---|---|
| `params` (names, types, defaults, `choices` from `Literal[...]`) | your dataclass fields / `__init__` |
| param `description`s | your class docstring `Args:` section |
| key `role`s | your `*_key` field names (shared `_roles.KEY_ROLES`) |
| `dispatch` | whether you override `process_batch` |
| `description`, `stage_id` | class docstring / class name |

You never put `params` in `describe()`.

## What is OPTIONAL — set only if it's obvious

Declared via one class attribute, `AGENT_STATIC = StaticHints(...)`, or on the contract:

- **StaticHints-settable** (instance-free): `cardinality_options` (e.g. `["fan_out", "nested"]`),
  `gates`, `dispatch`, `error_policy` (`"skip" | "fail" | "annotate"` — default `"unknown"`; set
  only if your stage has a clear, uniform policy), `description`, `stage_id`.
- **Contract-only** (return them from `describe()`; StaticHints has no such fields):
  `iteration_key`, `size_envelope`.
- `BATCH_ONLY = True` — only if your `process()` raises and just `process_batch` works.

If you're unsure, leave them. The agent treats missing optionals safely.

---

## Naming rule: config-knobs-only

Keep **today's default key names**. Do **not** rename keys to a global vocabulary. Just expose a
`*_key` field for each so an agent can remap when wiring two stages. Compatibility comes from
**semantic roles** (below), not from everyone using the same strings.

## Semantic roles — the compatibility contract

An agent chains a producer's output to a consumer's input by **role**, not key string. Roles are
resolved automatically from your `*_key` **field name** via `nemo_curator/stages/audio/_roles.py`.

- If your key fields use existing names (`audio_filepath_key`, `waveform_key`, `score_key`,
  `text_key`, `segments_key`, …) you get the right role for free.
- If you add a **brand-new** `*_key` concept, add one line to `KEY_ROLES` in `_roles.py` (or, for a
  truly stage-internal key, list it in `INTERNAL_KEY_FIELDS`). The conformance test fails if you
  forget — it won't let a key silently fall through.

## Discovery — how the agent finds your stage

Nothing to do: your stage auto-registers (via `StageMeta`) and appears in the catalog. Consumers
go through the public entry point — `agent.py` is the sanctioned public surface; don't import the
private `_catalog` module directly:
```python
from nemo_curator.stages.audio import agent

agent.list_agent_ready_stages()  # -> [... "MyStage" ...]
agent.describe_stage("MyStage")  # -> StageContract (static, instance-free)
agent.catalog_as_json()          # -> JSON the agent/UI consumes
```

---

## The safety net: `assert_agent_ready`

Add one test. It runs the static checks (contract shape, valid roles, JSON-serializable, reads
satisfiable by role) and — with a fixture — runs your stage and verifies declared writes appear,
no undeclared top-level keys leak, and cardinality matches runtime:

```python
from nemo_curator.stages.audio._conformance import assert_agent_ready

def test_my_stage_is_agent_ready(tmp_path):
    def fixture():
        return AudioTask(data={"audio_filepath": str(_write_wav(tmp_path / "a.wav"))})
    assert_agent_ready(MyStage(), fixture, expected_cardinality="1:1", available_keys={"audio_filepath"})
```

For GPU/model stages, reuse the existing fake-model/stub setup (see
`tests/stages/audio/test_agent_simulation_pipelines.py`) so the test needs no GPU. You don't need to
memorize the rules — if the test passes, the contract is honest.

---

## Checklist (copy into your PR)

- [ ] `AgentReady` + `describe()` with `reads`, `writes`, `cardinality`, honest `gates`
- [ ] every read/written `task.data` key is a `*_key` constructor field (no bare literals)
- [ ] new `*_key` concepts have a `_roles.KEY_ROLES` entry (or `INTERNAL_KEY_FIELDS`)
- [ ] new `AudioTask`s preserve `_metadata` and `list(_stage_perf)` (manual — not covered by `assert_agent_ready`)
- [ ] `assert_agent_ready(...)` test added and green
- [ ] defaults unchanged → existing pipelines behave exactly as before

That's it. Auto-derivation handles params/roles/dispatch/description; the test enforces the rest.
