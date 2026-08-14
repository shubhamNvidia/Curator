# Audio Agent — Implementation Guide for Maintainers

**Who this is for.** Someone who did not write this code and needs to maintain it, debug it,
extend it, and explain it confidently.

**How to read it.** Sections 1–2 give you the map. Section 3 explains each file. Section 4 is
the single most valuable section — one realistic request traced through every file. Sections
5–10 are reference. Section 12 tells you what to study before presenting.

**Running example used throughout:**

> *"Transcribe the audio data, keep only single-speaker audio, and resample the output to 48 kHz."*

This example is chosen deliberately: it contains a **hidden conflict** (the ASR and diarization
models only accept 16 kHz, but the user wants 48 kHz output) and a **classic semantic trap**
("single-speaker" tempts you toward the wrong stage). Watching the system handle both is the
fastest way to understand why it is built this way.

---

## Vocabulary (read this first)

| Term | Plain meaning |
| --- | --- |
| **NeMo Curator** | The data-curation framework this agent drives. Provides *stages* and a *pipeline runner*. |
| **Stage** | One unit of audio work — resample, transcribe, score quality, write a file. A Python class. |
| **Pipeline** | An ordered list of configured stages that Curator executes. |
| **AudioTask** | The record that flows between stages. Think of it as one row: a dictionary (`task.data`) holding the file path, transcripts, scores, etc. |
| **Key** | A named field inside `task.data` — e.g. `audio_filepath`, `pred_text`, `num_speakers`. |
| **Role** | A *stable semantic name* for a key. Key *values* are renameable per pipeline; roles are not. Stages are chained by role, not by string. |
| **Contract** | A stage's machine-readable declaration: which keys it reads/writes, whether it changes row count, what it needs (GPU, ffmpeg…). |
| **Capability card** | A YAML "instruction manual" for one stage: what it is good for, its limits, its metric scale, its resource cost, what its outputs *mean*. |
| **Recipe** | The declarative document the LLM writes: a list of `{stage, parameters}` plus success criteria. The only thing the LLM emits. |
| **Verb** | One deterministic tool function (`validate`, `smoke`, `run`, …). All return JSON. |
| **Host / host LLM** | The model driving the tools (Claude, Cursor). It is *outside* this package. |
| **Deterministic core** | `nemo_curator.audio_agent` — plain Python, no LLM inside. |
| **Cardinality** | Whether a stage keeps row count (`1:1`), multiplies it (`1:N fan-out`), reduces it (`N:1`), or drops rows (`filter`). |
| **Fan-out** | One input row becomes many output rows (e.g. one file → many speech segments). Changes what a field *describes*. |
| **Artifact** | A completed step's output on disk, content-addressed so it can be reused. |
| **Smoke** | A bounded real run on ~10 items, for evidence before paying for the full run. |

**Four analogies that carry most of the design:**

- **Capability cards** = instruction manuals for each audio operation. The code says *how* a
  stage works; the card says *when to use it and what its numbers mean*.
- **Deterministic validation** = a rule-based safety gate at the airport. It doesn't judge
  where you're going; it proves you're not carrying something that will explode.
- **Semantic review** = a second reviewer who asks *"is this actually the trip you meant to
  book?"* — a question no rule can answer.
- **Reuse** = continuing from saved work rather than starting over, but only when you can
  *prove* the saved work is identical.

---

## 1. Project Map

Only files that matter. `→` marks the one-line purpose.

```
Curator/
├── .agents/skills/                    → Cross-host skill discovery (Codex, Cursor); links into the package
├── .claude/skills/audio-curation      → Same links again; Claude Code does not read .agents/skills
│                                        (each shim is a REAL directory of symlinked entries, because a
│                                         walk that does not follow symlinks sees nothing inside a linked
│                                         directory. On Windows, use `install-skill --copy` instead.)
│
├── nemo_curator/audio_agent/          ★ THE DETERMINISTIC CORE (no LLM inside)
│   ├── skills/audio-curation/
│   │   ├── SKILL.md                   → ★ THE AGENT'S BRAIN. The written procedure the host LLM follows
│   │   └── references/                → Long sections, loaded only when their step is reached
│   ├── skills/audio-stage-authoring/
│   │   └── SKILL.md                   → Procedure for making a stage agent-ready
│   ├── AGENTS.md                      → Auto-attached guardrails + loop order for this directory
│   ├── CLAUDE.md                      → One-line `@AGENTS.md` import so Claude Code sees the same text
│   ├── ENVIRONMENT.md                 → Human guide to environment health; `doctor` is its runnable form
│   ├── REUSE_ARCHITECTURE.md          → Design doc for the memoization subsystem
│   │
│   ├── __init__.py                    → Public SDK surface (re-exports verbs + contracts)
│   ├── __main__.py                    → `python -m nemo_curator.audio_agent` entry
│   ├── cli.py                         → CLI adapter: one subcommand per verb, prints JSON
│   ├── mcp_server.py                  → MCP adapter: same verbs as typed tools for MCP hosts
│   ├── verbs.py                       → ★ The verb implementations (4099 lines; the orchestrator)
│   │
│   ├── contracts.py                   → Typed data objects: Verdict, DataProfile, EnvProfile, RunRecord…
│   ├── recipe.py                      → ★ Recipe IR: the anti-hallucination boundary + the 3 hashes
│   ├── _resolve.py                    → Stage name → real class (rejects invented names)
│   │
│   ├── index.py                       → Knowledge Index: tiered retrieval L0→L1→L2, role graph
│   ├── context.py                     → Packages index + profiles into one PlanningContext
│   │
│   ├── profiler.py                    → Reads the data and the machine (the agent's "eyes")
│   ├── env_health.py                  → `doctor`: environment check registry + fix steps
│   ├── diagnostics.py                 → Recipe-aware env decisions + failure diagnosis
│   ├── failures.py                    → Error text → classified failure (regex over the taxonomy)
│   │
│   ├── config_strategy.py             → `resolve`: outcome word → concrete parameter, via cards
│   ├── checks.py                      → ★ Pluggable validation check registry (9 checks)
│   ├── semantic_review.py             → Builds the evidence packet for the LLM critic
│   ├── acceptance.py                  → Success contract: compile, verify, honesty guard
│   │
│   ├── planner.py                     → Resource planner: streaming vs batch + feasibility
│   ├── calibration.py                 → Extract measured resources from a smoke report
│   ├── _ray.py                        → Opt-in Ray head bootstrap
│   ├── report.py                      → RunReport: the evidence artifact from a run
│   │
│   ├── input_identity.py              → Which dataset will this recipe actually read? (closed adapter table)
│   ├── artifacts.py                   → ★ Content-addressed step keys + atomic publish
│   ├── reuse.py                       → Reuse scan + the approval card
│   ├── continuation.py                → Recipe rewriting to resume from an artifact
│   ├── run_store.py                   → JSON run records (source of truth)
│   ├── run_index.py                   → Rebuildable SQLite cache over records/artifacts
│   ├── _safety.py                     → Workspace lock, secret redaction, smoke token
│   ├── card_conformance.py            → Gate: cards must match the real stages
│   │
│   ├── knowledge/                     ★ ALL DOMAIN KNOWLEDGE (static, versioned YAML)
│   │   ├── CARD_SCHEMA.md             → How to author a card + what the gate enforces
│   │   ├── taxonomy.yaml              → The 10 L0 categories
│   │   ├── failures.yaml              → 18 classified error signatures → cause → guidance
│   │   ├── cards/*.yaml               → 47 capability cards, one per stage
│   │   ├── blueprints/*.yaml          → 3 idiomatic end-to-end shapes (enforced/advisory tags)
│   │   └── patterns/composition.yaml  → Abstract ordering rules (teaching layer)
│   └── recipes/*.yaml                 → 3 working reference recipes
│
├── nemo_curator/stages/audio/         ★ THE STAGE-SIDE FOUNDATION
│   ├── AGENT_READY.md                 → Checklist for stage owners
│   ├── agent.py                       → Public facade re-exporting the foundation
│   ├── _agent_ready.py                → StageContract / IOSpec / Gates dataclasses + AgentReady base
│   ├── _agent_registry.py             → Auto-derives params from constructors; builds contracts
│   ├── _catalog.py                    → Stage discovery + role index (producers/consumers)
│   ├── _roles.py                      → key-field-name → semantic role table
│   ├── _planning.py                   → ★ validate_pipeline(): the role/key composition walk
│   ├── _conformance.py                → assert_agent_ready(): the stage-owner's test
│   ├── _residency.py                  → file-vs-waveform input handling (shared helper)
│   └── <stage modules…>               → The 47 actual stages
│
├── eval/audio/                        ★ TEST HARNESS (two planes)
│   ├── AGENT_TEST_PLAN.md             → Human-facing test strategy; 23-class failure taxonomy
│   ├── run_eval.py                    → Deterministic core eval (GPU-free, CI)
│   ├── queries.yaml                   → Gold recipes + expected verdicts
│   ├── taxonomy.yaml                  → Failure signal → category/stage/severity mapping
│   ├── scenarios/L01..L15*.yaml       → LLM-plane scenarios (L15 = semantic intent)
│   ├── agent_runner.py                → Captures real host-LLM tool-call traces
│   ├── trace_check.py                 → Deterministic grading of those traces
│   ├── judge.py                       → LLM-as-judge over the critique quality
│   ├── aggregate_traces.py            → The authoritative semantic regression gate
│   ├── simulate_user.py               → Persona-driven end-to-end simulation (opt-in)
│   ├── run_all.sh                     → Orchestrates every plane
│   └── fixtures/                      → Tiny WAVs + manifests + a "novel stage" generalization test
│
└── tests/
    ├── audio_agent/                   → 24 test modules, ~11.3k lines (core behavior)
    └── stages/audio/test_agent_*.py   → Stage-side agent-readiness conformance
```

---

## 2. Architecture Overview

### 2.1 The layers (corrected against the actual implementation)

Your proposed list was close. Here is what the code actually has, in execution order.
**Bold = a layer your list did not name but the implementation clearly has.**

| # | Layer | Lives in | Owner |
| --- | --- | --- | --- |
| 0 | **Stage agent-readiness** (contracts on stages) | `stages/audio/_agent_ready.py`, `_agent_registry.py` | Deterministic |
| 1 | User interaction / tool surface | `cli.py`, `mcp_server.py`, `__init__.py` | Deterministic |
| 2 | Intent understanding + **success contract** | `SKILL.md` (procedure), `acceptance.py` (compilation) | **LLM** |
| 3 | Knowledge & capability cards | `knowledge/`, `index.py` | Deterministic |
| 4 | **Situational awareness** (data profile + env probe) | `profiler.py`, `context.py` | Deterministic |
| 5 | Health checking | `env_health.py`, `diagnostics.py` | Deterministic facts, **LLM** explains |
| 6 | Routing & module selection | `index.py` serves; `SKILL.md` decides | **LLM** |
| 7 | **Outcome→parameter resolution** | `config_strategy.py` | Deterministic (card-driven) |
| 8 | Recipe generation | `recipe.py` (the IR), LLM writes it | **LLM** writes, deterministic validates |
| 9 | **Source binding / dataset identity** | `input_identity.py` | Deterministic |
| 10 | Deterministic validation | `checks.py` + `stages/audio/_planning.py` | Deterministic |
| 11 | **Semantic review** (intent correctness) | `semantic_review.py` builds evidence | **LLM** judges |
| 12 | **Resource planning** (streaming vs batch) | `planner.py` | Deterministic |
| 13 | Previous-work reuse | `artifacts.py`, `reuse.py`, `continuation.py`, `run_index.py` | Deterministic, **user** approves |
| 14 | **Bounded smoke + calibration** | `verbs.smoke`, `calibration.py` | Deterministic |
| 15 | **Safety gates** (confirm, workspace, redaction) | `_safety.py`, `verbs.run` | Deterministic |
| 16 | Execution | `verbs.run`, `_ray.py` | Deterministic |
| 17 | Monitoring & failure handling | `report.py`, `failures.py`, `diagnostics.py` | Deterministic detects, **LLM** explains |
| 18 | Result validation & reporting | `acceptance.py`, `report.py` | Deterministic verdict, **LLM** narrates |
| 19 | **Provenance** | `run_store.py`, `run_index.py` | Deterministic |
| 20 | **Evaluation** | `eval/audio/` | Both planes |

### 2.2 The one sentence that explains the whole architecture

> **The LLM proposes. The deterministic core disposes.**

The LLM never emits code — only a Recipe naming registered stages. The gates live **inside the
tool functions**, not in the prompt, so a weaker model or a direct CLI call hits identical
refusals.

### 2.3 Control flow at a glance

```
   host LLM  ──JSON tool calls──▶  verbs.py  ──▶  {index, profiler, checks, planner, artifacts, …}
       ▲                              │
       └───── JSON verdicts ──────────┘                        verbs.run ──▶ Curator Pipeline ──▶ Ray/Xenna
```

`verbs.py` is the **only** module the outside world calls. Everything else is called by it.

---

### Presentation questions — Architecture

**Q: Why is this not one big script?**
Each layer answers a different question and fails differently. Keeping "what exists" (index)
separate from "does it compose" (checks) separate from "may it run" (safety) means a bug in one
cannot silently weaken another, and each can be tested alone. `run_eval.py` tests all of them
without a GPU or an LLM precisely because they are separable.

**Q: Where does the LLM actually live?**
Outside this repo. `nemo_curator.audio_agent` contains no model and makes no model calls. The
"agent" is the combination of *any* capable host model plus `SKILL.md` plus this core. That is
why the core is testable in CI and why we are not locked to one provider.

**Q: What is the single most important file?**
`verbs.py` is the biggest, but `recipe.py` is the most important: it defines the boundary that
makes hallucination structurally impossible. Second is `SKILL.md`, because it is the only place
the LLM's procedure is written down.

---

## 3. File-by-File Explanation

Files are grouped by role. The most important ones get the full 8-point treatment; the rest get
a compact but complete entry.

### 3.0 Code classification

Before the detail — **know what kind of code you are looking at**:

| Category | Files |
| --- | --- |
| **Core implementation** | `verbs.py`, `recipe.py`, `checks.py`, `index.py`, `contracts.py`, `profiler.py`, `env_health.py`, `diagnostics.py`, `acceptance.py`, `semantic_review.py`, `config_strategy.py`, `planner.py`, `artifacts.py`, `reuse.py`, `input_identity.py`, `_safety.py`, `stages/audio/_agent_ready.py`, `_agent_registry.py`, `_catalog.py`, `_planning.py`, `_roles.py` |
| **Supporting utilities** | `cli.py`, `mcp_server.py`, `context.py`, `report.py`, `run_store.py`, `run_index.py`, `calibration.py`, `failures.py`, `_ray.py`, `_resolve.py`, `continuation.py`, `card_conformance.py`, `_residency.py`, `_conformance.py` |
| **Knowledge / documentation** | `knowledge/**`, `recipes/**`, `SKILL.md`, `AGENTS.md` ×2, `ENVIRONMENT.md`, `REUSE_ARCHITECTURE.md`, `AGENT_READY.md`, `CARD_SCHEMA.md`, `.cursor/rules/*.mdc` |
| **Test-only** | `tests/audio_agent/**`, `tests/stages/audio/test_agent_*.py` |
| **Evaluation harness** (not shipped, not tests) | `eval/audio/**` |
| **Experimental / opt-in** | `simulate_user.py` (persona simulation, needs an API key + GPU), the `data_driven` flag in `config_strategy.py` (Path B, deliberately not implemented) |
| **Possibly redundant — see §3.9** | `continuation.plan_continuation` (older reuse engine, kept alongside the newer `reuse.scan`), `PlanResult` in `contracts.py` |
| **Local runtime output (gitignored)** | `.audio_agent_runs/` |
| **Untracked scratch** | root-level `curate_demo_recipe.yaml`, `single_speaker_16k_asr*.yaml` — real recipes from manual sessions, not part of the shipped package |

---

### 3.1 `nemo_curator/audio_agent/verbs.py` — the orchestrator

**1. What it is.** The deterministic tool surface. Every verb the LLM can call is a function
here. 4,099 lines, and legitimately the largest file in the package.

**2. Why it exists.** Something has to sequence "profile the input → build the stages → run the
checks → plan resources → enforce the gate → execute → publish artifacts → record the run". If
that logic were spread across the callers (CLI, MCP, SDK), the three surfaces would drift and
the safety gates would exist in three versions. Concentrating it here means **one gate
implementation, three doors**.

*Without it:* the CLI and MCP would each reimplement the confirm gate. History says one of them
would get it wrong (and one did — see the `confirm: null` bug in §9).

**3. How it works.** Every verb follows the same skeleton:

```
 1. Coerce the input to a Recipe and freeze it (compute the 3 hashes)
 2. Safety: path_violations() on every path the recipe touches
 3. Bind the source: resolve_dataset_binding() → what data will REALLY be read
 4. Profile that source (read-only)
 5. build_stages() — instantiate real stage classes (catches invented names/params)
 6. Verb-specific work (run_checks / environment_preflight / plan / execute)
 7. Redact secrets + transcripts from the return value
 8. Return a JSON-safe dict — never raise, never print
```

Key verbs and what is distinctive about each:

| Verb | Distinctive logic |
| --- | --- |
| `discover` / `catalog_tree` / `cards` / `describe` | Thin pass-through to `index.py`. Tiered so the LLM reads little. |
| `context` | Assembles the planning bundle (delegates to `context.py`). |
| `resolve` | Delegates to `config_strategy.py`. |
| `validate` | Builds a `CheckContext`, runs the check registry, attaches `semantic_review`, `environment_decision`, and `output_targets`. |
| `smoke` | Bounds the input, **isolates every output into a temp tree** (`_isolate_smoke_outputs`, `_smoke_write_issues` — fails closed if isolation cannot be *proven*), runs, extracts calibration. |
| `run` | The confirm gate, then execute, then publish artifacts, verify acceptance, and record the run. |
| `report` / `verify` | Post-hoc evidence and acceptance verdict. |
| `reuse_scan` / `plan_continuation` | Reuse lookup and the three-way choice; `_merge_plans` picks whichever of the two reuse engines found more. |
| `runs` / `reindex` | Provenance queries. |

**4. When it is used.** Continuously — it is the boundary the LLM talks to at every step.

**5. Simple example.** For our request, `validate(recipe, data=...)` receives the recipe below
and returns a `Verdict` telling the LLM that `InferenceAsrNemoStage` will receive 48 kHz audio
when its card says 16 kHz — before a single GPU second is spent.

**6. Connections.** Depends on nearly everything. Called by `cli.py`, `mcp_server.py`,
`__init__.py`, `eval/audio/run_eval.py`, and tests.

**7. Key design decisions.**
- *Verbs return dicts, never raise.* The LLM is a caller that cannot catch exceptions. A
  traceback where JSON was promised is an unrecoverable state for the host.
- *Everything goes through `_safety.redact()` on the way out.* Redaction at the boundary means
  no verb can forget it.
- *Reuse keys are computed BEFORE redaction.* Otherwise redaction would silently change
  identity — a subtle and very bad bug class.

**8. Failure scenarios.**
| Symptom | Cause | Check |
| --- | --- | --- |
| `status: refused`, `reason: path(s) resolve outside…` | `AUDIO_AGENT_WORKSPACE` is set and a path is outside it | `echo $AUDIO_AGENT_WORKSPACE`; `_safety.path_violations` |
| `status: refused`, mentions `data_binding` | `--data` disagrees with the recipe's source stage | Compare `verdict.data_binding.primary_path` with your `--data` |
| `unknown_stage` in issues | Stage name is wrong, or its optional dependency is not installed | `python -m nemo_curator.audio_agent discover \| grep -i <name>` |
| Verb returns `{"status":"error", "diagnosis": …}` | Stage construction failed | Read `diagnosis.classification` |

---

### 3.2 `recipe.py` — the anti-hallucination boundary

**1. What it is.** The `Recipe` and `StageRef` dataclasses: the single artifact the LLM emits
and the core consumes.

**2. Why it exists.** This is *the* safety decision of the project. The alternative — letting
the model write Python — is unreviewable and unbounded. A Recipe is a short list of names that
a human reads in ten seconds and a machine verifies completely.

*Without it:* generated code in the execution path. It has been verified across the repo that
there is **zero** `exec`/`eval`/`compile`/temp-`.py`-writing anywhere in execution.

**3. How it works.**

`build_stages(recipe)` is the enforcement point:
```python
cls = resolve_stage_class(s.ref)      # KeyError → "unknown_stage" issue (invented name dies here)
params = dict(s.params)
with_kwargs = {k: params.pop(k) for k in params if k in EXECUTION_KNOB_PARAMS}
inst = cls(**params)                  # TypeError → "bad_params" issue, listing accepted params
inst = inst.with_(**with_kwargs)      # resources / batch_size applied the framework way
```

**The three hashes** (the part people find confusing — learn this):

| Hash | Covers | Question it answers |
| --- | --- | --- |
| `config_hash` | everything: stages, params, output paths, batch sizes, acceptance criteria | *"Is this exactly what the user approved?"* → the confirm gate |
| `semantic_hash` | stages + inputs, **minus** execution knobs and output *locations* | *"Would this produce the same bytes?"* → reuse identity |
| `contract_hash` | acceptance criteria alone | *"Is the success bar the same?"* → re-verify, don't recompute |

Why split them: changing a batch size does not change a single output byte, but it *does* change
what the user approved. One hash for both jobs gives you either unsafe approval or reuse that
never fires. (It gave us the latter — see §8.)

**Layered save.** `machine_plan`, `data_derived`, `config_strategy`, `knowledge_version` are
attached to the recipe but **excluded from every hash**, each stamped with the machine or dataset
fingerprint it was computed for. `stale_layers()` tells a caller which must be recomputed. This
keeps the recipe portable: the same intent on a different machine hashes identically.

**4. When it is used.** The LLM writes it in step 6; every recipe-driven verb consumes it.

**5. Simple example.**

```yaml
stages:
  - ref: CreateInitialManifestAudioFolderStage
    params: {data_dir: /data/raw}
  - ref: ResampleAudioStage
    params: {target_sample_rate: 16000, target_nchannels: 1,
             resampled_audio_dir: /out/16k, update_audio_filepath: true}
  - ref: InferenceSortformerStage
    params: {resources: {gpus: 1}}
  - ref: PreserveByValueStage
    params: {input_value_key: num_speakers, operator: eq, target_value: 1}
  - ref: InferenceAsrNemoStage
    params: {model_name: nvidia/parakeet-tdt-0.6b-v2, resources: {gpus: 1}}
  - ref: ResampleAudioStage
    params: {target_sample_rate: 48000, resampled_audio_dir: /out/48k, update_audio_filepath: true}
  - ref: ManifestWriterStage
    params: {output_path: /out/curated.jsonl}
acceptance_criteria: [...]
```

**6. Connections.** Depends on `_resolve.py` and `acceptance.py` (criteria shape checking).
Used by `verbs.py`, `artifacts.py` (step keys), `continuation.py`, `reuse.py`, `_safety.py`.

**7. Key design decisions.**
- *No DAG engine.* Audio recipes are linear chains. A general graph IR is cost without benefit
  today.
- *Criteria are shape-checked at the door* (`_criteria`). A mapping like
  `acceptance_criteria: {must: [...]}` used to silently become `["must"]` (because `list()` over
  a dict yields keys) — an unverifiable contract that `run` then skipped **while reporting
  success**. Now it raises with the expected shape.
- *`from_dict` rejects near-miss field names* (`acceptance_criterion`, `acceptance_check`, …) so
  a typo cannot silently produce an empty contract.

**8. Failure scenarios.**
| Symptom | Cause | Check |
| --- | --- | --- |
| `ValueError: acceptance_criteria must be a LIST…` | Criteria written as a mapping | Fix the YAML shape |
| Confirm hash mismatch | Recipe edited after approval | Re-validate; the hash is *meant* to change |
| Reuse never fires | Something semantic changed | Compare `semantic_hash` between the two recipes |

---

### 3.3 `stages/audio/_agent_ready.py` + `_agent_registry.py` + `_roles.py` — the foundation

**1. What they are.** The layer that makes a stage machine-describable.
- `_agent_ready.py` — the dataclasses: `StageContract`, `IOSpec`, `Gates`, `ParamSpec`,
  `ConditionalWrite`, `SizeEnvelope`, and the `AgentReady` base class.
- `_agent_registry.py` — auto-derives parameters from the constructor and merges them with the
  stage's hand-written `describe()`.
- `_roles.py` — the table mapping `*_key` field names to stable semantic roles.

**2. Why they exist.** Two stages can only be chained reliably if you know what each reads and
writes. That knowledge used to live in source code and people's heads.

The **role indirection** is the clever part. Key *values* are configurable (an agent may rename
`score` to `my_score`), but the *field name* (`score_key`) is invariant because the stage's code
is written around it. So roles are derived from field names, and composition survives renaming.

*Without it:* no compatibility checking, no semantic review, no reuse identity. Everything above
this layer collapses.

**3. How it works.**

A stage owner declares only three things:
```python
def describe(self) -> StageContract:
    return StageContract(
        reads=IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
        writes=IOSpec(data_keys=[self.score_key]),
        cardinality="1:1",
        gates=Gates(requires_gpu=self.resources.gpus > 0),
    )
```

Everything else is **auto-derived** by `_agent_registry.build_contract(stage)`:

| Derived | From |
| --- | --- |
| `params` (names, types, defaults, `choices` from `Literal[...]`) | dataclass fields / `__init__` signature |
| parameter descriptions | the class docstring's `Args:` section |
| `key_roles` | `*_key` field names via `_roles.KEY_ROLES` |
| `dispatch` | whether the stage overrides `process_batch` |
| `accepts_task_type` / `produces_task_type` | the `ProcessingStage[X, Y]` generic |
| `description`, `stage_id` | first docstring line / class name |

Two contract flavours, and confusing them is a real bug source:

- **`static_contract(cls)`** — instance-free, used by `describe`/L2 cards. It is explicitly
  marked `contract_resolution: "static_params_and_hints"`: real parameters and class hints, but
  its placeholder reads/writes are **not configured runtime facts**.
- **`build_contract(stage)`** — from a *configured instance*, so key values are resolved. This
  is what validation and semantic review use. **Always the authoritative one.**

**4. When used.** `build_contract` is called by `_planning.validate_pipeline`, by three of the
checks in `checks.py`, by `semantic_review.py`, and by `diagnostics._execution_requirements`.

**5. Simple example.** For `InferenceSortformerStage`, `build_contract` returns
`writes.data_keys = ["diar_segments", "num_speakers"]`, `cardinality = "1:1"`,
`gates.requires_gpu = True`. That is how the validator knows `PreserveByValueStage` reading
`num_speakers` is satisfied.

**6. Connections.** `_planning.py`, `_catalog.py`, `_conformance.py`, and everything in
`audio_agent/` that reasons about stage I/O.

**7. Key design decisions.**
- *Minimal burden on stage owners.* Three declarations plus one test. Adoption across 47 stages
  would not have happened otherwise.
- *Every new knob defaults to today's behavior.* Agent-readiness never changes how a stage runs
  in an existing pipeline.
- *`ConditionalWrite` describes possibility, not behavior.* It labels branch-dependent writes and
  value provenance (`upstream_same_key` = "this stage copied it, it is not a new measurement") for
  the semantic reviewer, without changing what the mechanical validator does.

**8. Failure scenarios.**
| Symptom | Cause | Check |
| --- | --- | --- |
| `contract_error: describe() failed` | `describe()` raises for this configuration | Call `build_contract(stage)` in a REPL |
| `unsatisfied_reads` for an obviously-present field | New `*_key` field missing a `KEY_ROLES` entry | Add it to `_roles.py` (or `INTERNAL_KEY_FIELDS`); `assert_agent_ready` should have caught this |
| Contract looks wrong in a card | You read the **static** contract | Use `build_contract(instance)` instead |

---

### 3.4 `stages/audio/_planning.py` — `validate_pipeline()`

**1. What it is.** The mechanical composition walk. Given an ordered list of *configured* stages,
does the data actually flow?

**2. Why it exists.** Curator will happily build a pipeline whose stage 4 reads a field nobody
produces. You find out at runtime, on the GPU, after paying for stages 1–3.

**3. How it works.** A single forward pass maintaining five pieces of state:

```
available        : roles produced so far      (starts from the data profile)
available_keys   : literal key values so far
tensor_resident  : is a non-serializable tensor sitting in task.data?
past_composite   : has a composite hidden its writes?
removed_roles    : roles whose carrier key was deleted upstream
```

For each stage: build its contract → check reads are satisfied **by role** → if satisfied, check
the literal **key values** match → run gate checks → merge its writes into `available`.

**Two confidence levels, deliberately separated:**
- `report.ok` — every required *role* is available. Rename-tolerant. This is the **error** gate.
- `report.keys_ok` — every role-satisfied read's actual *key value* is produced upstream. A
  `True`/`False` combination means "the roles line up but a producer key was renamed" — the
  pipeline validates and yields **zero rows at runtime**. Surfaced as a *warning*, not an error,
  so legitimate reads of source-manifest columns are not false-rejected.

**Three subtleties worth knowing:**
- **Composites** hide their inner writes. Reads past one are downgraded to
  `unsatisfied_reads_after_composite` (a warning that escalates to `smoke`) rather than a false
  hard error. But serialization/GPU/key-flow checks still run for every concrete stage — this
  fixed a real blind spot where starting with `ManifestReader` (a composite) skipped them all.
- **`ambiguous_default_key`** — when two upstream stages wrote same-kind keys and a consumer is
  left at its default, the default is silently choosing between them. The documented case: a
  diarization merge left at `segments_key="segments"` merges into the *VAD* segments and produces
  a plausible, wrong answer with **no error**.
- **`removes_keys`** tracking — a role whose carrier key was deleted upstream produces
  `key_removed_upstream`, which is a different fix from `unsatisfied_reads`.

**4. When used.** From `checks._check_data_flow`, the first registered check.

**5. Simple example.** In our recipe, when the walk reaches `PreserveByValueStage`, `available`
contains `num_speakers` (written by Sortformer two stages earlier) → read satisfied. Had the LLM
put the filter *before* the diarizer, this walk returns `unsatisfied_reads` with
`available so far: [audio_filepath, sample_rate, ...]`.

**6. Connections.** Depends on `_agent_registry.build_contract`, `_conformance`, `_roles`.
Called only by `checks.py`.

**7. Key design decisions.**
- *Role matching over string matching.* Keys are configurable; roles are not.
- *`ok` vs `keys_ok` split.* Collapsing them either false-rejects legitimate manifest reads or
  misses renamed producers.
- *Advisory and read-only.* It never executes a stage. `ok` is a **necessary, not sufficient**
  condition — the module docstring says so explicitly.

**8. Failure scenarios.** See §10 rows for `unsatisfied_reads`, `dangling_key`,
`ambiguous_default_key`, `tensor_into_sink`.

---

### 3.5 `checks.py` — the validation registry

**1. What it is.** Nine registered checks, each `fn(ctx) -> CheckResult`, merged into one
`Verdict`.

**2. Why it exists.** Validation must be *extensible without touching the verb surface*. Adding
a check is one decorated function.

**3. How it works.** `validate` builds a `CheckContext` (recipe, built stages, data profile,
env, initial roles/keys, expected outputs, criteria, request type, execution target), then
`run_checks` runs each in registration order and merges.

**Critically: each check is isolated.** If one raises, it becomes a `check_error` issue and the
recipe is marked not-ok — *the grounding layer must never emit a traceback where it promised a
JSON Verdict.*

| Check | What it catches |
| --- | --- |
| `data_flow` | Delegates to `validate_pipeline` (roles, keys, residency, serialization) |
| `card_constraints` | Fixed batch size, unsupported sample rates, duration sweet spot, max speakers |
| `gpu_reservation` | A `bound: gpu`, non-optional stage that reserves no GPU (runs on CPU, very slowly) |
| `gates` | ffmpeg missing, GPU unavailable, first-run download, missing secrets |
| `unproducible` | A required role **no stage in the entire catalog** can produce |
| `output_completeness` | "You asked for transcripts; no stage produces them" |
| `request_type_sanity` | A filtering request with no yield criterion |
| `task_type` | `DocumentBatch` producer feeding an `AudioTask`-only stage |
| `diarization_continuity` | A diarizer after a VAD with no re-join (would see torn audio) |

**Two pieces of subtle logic worth understanding:**

**Effective sample rate tracking** (`card_constraints`). Naively comparing a model's supported
rates against the *source* profile warns about 48 kHz input to a 16 kHz model **even when a
resample sits immediately upstream** — correct pipelines told they are broken, which is how a
validator loses its authority. So the check walks the recipe maintaining `effective_srs`, judging
each stage on what it *receives*. It also consults the **built** stage so an omitted parameter
resolves to its real default. And it deliberately does *not* treat `MonoConversionStage`'s
`output_sample_rate` as a conversion — that card is explicit that the parameter *verifies* a rate
(dropping non-matching rows) rather than converting.

**Card-derived role detection** (`diarization_continuity`). "Is this a diarizer?" is answered by
card category, unioned with an explicit fallback set, so a *new* diarizer is covered with no code
change.

**4. When used.** Inside `verbs.validate`, on every call.

**5. Simple example.** Our example's first draft — resample to 48 kHz *first*, then ASR — trips
`card_sample_rate` on both `InferenceSortformerStage` and `InferenceAsrNemoStage`
("input sample rates [48000] not all in supported [16000]"), with the fix
"insert a resample stage upstream to the supported rate".

**6. Connections.** Depends on `index.py`, `stages/audio/agent`, `acceptance.py`. Called only by
`verbs.validate`.

**7. Key design decisions.**
- *A registry, not a monolith.* Extension is one function.
- *Universal invariants only.* Module-specific *intent* rules are deliberately excluded — that is
  the semantic reviewer's job (§3.7). Encoding meaning as rules produces an ever-growing,
  always-incomplete table.
- *`gpu_reservation` is card-driven*, so a new GPU stage is covered automatically.

**8. Failure scenarios.**
| Symptom | Cause | Check |
| --- | --- | --- |
| `check_error` in issues | A check raised, usually on a malformed card | The message names the check; inspect that card |
| Spurious `card_sample_rate` | A conversion the tracker doesn't recognize | `_rate_converted_to` only reads `target_sample_rate` |
| `unproducible` empty when you expect a hit | The role resolved to `"unknown"` | Missing `KEY_ROLES` entry in `_roles.py` |

---

### 3.6 `index.py` + `knowledge/` — the knowledge layer

**1. What it is.** `KnowledgeIndex` loads all static YAML (47 cards, taxonomy, 3 blueprints,
patterns, 3 recipes) and serves it in tiers.

**2. Why it exists.** Handing 47 full cards to a model for every request is expensive and,
counterintuitively, **less accurate** — the model reasons over noise. Tiered retrieval is cheaper
*and* better.

**3. How it works.**

```
L0  category_tree()        → 10 categories + descriptions + member stages   (prune here)
L1  card_oneliners(cat)    → {stage, summary, tags} for one category        (shortlist)
L2  full_cards([names])    → full card + static contract for finalists      (decide)
```

Plus: `match_blueprints/​match_recipes` (token-overlap ranking against the goal),
`patterns()`, `role_neighborhood(roles)` (producers/consumers per role — used for **targeted
re-retrieval** when validation names a missing role), and `unproducible(roles)`.

`category_of(stage)` reads the card's `category`; a stage with no card falls back to a keyword
rule over its module path — so **a new stage joins the tree automatically**.

Cached process-wide via `@lru_cache` because the knowledge is static.

**The knowledge files themselves:**

| File | Contains |
| --- | --- |
| `taxonomy.yaml` | The 10 L0 categories |
| `cards/*.yaml` | Per stage: summary, tags, model id + pinned version, constraints, resource facts, use cases, composition, `metrics` (scale/direction/anchors/presets), `semantic_facts`, `comparison`, `notes`/`caveats`, `verified` honesty tiers, `provenance` |
| `blueprints/*.yaml` | Idiomatic shapes with per-stage `enforced`/`advisory` tags, a `topology_selection` rule, and explicit `pitfalls` |
| `patterns/composition.yaml` | Abstract ordering rules. **Note: not parsed by the validator** — a teaching layer kept in sync with `checks.py` by hand |
| `failures.yaml` | 18 error signatures → cause → layer → guidance |
| `recipes/*.yaml` | 3 fully working reference recipes |

**Card honesty tiers** are the schema's best idea. Every fact group declares how it was
established: `mechanical` (from code; the gate re-checks it), `measured` (a real run on named
hardware), `best_guess` (author judgment). The golden rule is **never fabricate** — an unknown
value is left empty with a `TODO(fill)` comment, because the agent trusts cards absolutely and a
wrong fact is worse than a missing one.

**4. When used.** Steps 2, 4, 5 of the loop — routing, comparison, parameter resolution.

**5. Simple example.** For our request, L0 prunes to `preprocess`, `diarize`, `transcribe`,
`export`. L1 on `diarize` returns three stages. L2 on `InferenceSortformerStage` +
`SpeakerSeparationStage` reveals the decisive fact in the summary line:

> *"Annotates only — does NOT separate audio"* vs *"…TRANSFORMS the audio into per-speaker streams"*

That one line is what steers the LLM to the right stage.

**6. Connections.** Depends on `stages/audio/_catalog.py` and `_resolve.py`. Used by `checks.py`,
`config_strategy.py`, `context.py`, `planner.py`, `artifacts.py`, `semantic_review.py`, `verbs.py`.

**7. Key design decisions.**
- *Knowledge as versioned YAML, not prompt text.* Reviewable, diffable, testable, gate-enforced.
  Prompt text drifts silently; files do not.
- *A malformed card is skipped, not fatal* (`_load_dir` swallows parse errors) — one bad file
  must not break discovery. **Trade-off:** a typo makes a card silently vanish. Run
  `card_conformance` to catch it.
- *The index serves material; it never makes the final relevance choice.* That is the LLM's job.

**8. Failure scenarios.**
| Symptom | Cause | Check |
| --- | --- | --- |
| A stage has no card facts | Card missing or YAML invalid | `python -c "import yaml;yaml.safe_load(open('...yaml'))"` |
| Stage lands in category `other` | No card and no keyword rule match | Add `category:` to the card |
| `resolve` says "no anchor for label" | Card has no `metrics.anchors` | Add anchors, or ask the user |

---

### 3.7 `semantic_review.py` — evidence for the LLM critic

**1. What it is.** Builds a read-only evidence packet so the host LLM can judge whether a
mechanically valid recipe **means** what the user asked.

**2. Why it exists — the trap this layer exists to catch.**

*"Keep only single-speaker clips."* A tempting recipe: `SpeakerSeparationStage`, then filter
`num_speakers == 1`. It validates **perfectly green**. It is also wrong twice:

- `num_speakers` is the count found in the **original whole clip** — a parent-level aggregate.
  After separation each row is one speaker's stream, so the field is applied at the wrong
  granularity.
- Separation **transforms the audio** into per-speaker streams. The user asked to *select* clips,
  not to mangle them.

The correct recipe — diarize (annotate only), then filter `num_speakers == 1` — is
**indistinguishable from the wrong one by any mechanical check**. The plumbing is green either
way.

*Without this layer:* confidently wrong pipelines that pass every gate and answer a different
question.

**3. How it works.** `build_semantic_review(stages, initial_keys, recipe, data_profile)` uses
**configured dynamic contracts** (not static ones) to co-locate:

- exact field lineage — which stage *really* produced the value each consumer reads (the "latest
  producer", so a generic value filter can be traced instead of guessed from the key name);
- cardinality seams — fan-out, nesting, aggregation, filter boundaries;
- the producer's and consumer's card prose — `semantic_facts` (meaning / unit / provenance /
  scope / propagation / counterexamples), notes, metrics, domain, limitations;
- composite expansion (up to depth 8, 512 leaves) so hidden inner stages are visible;
- a **checklist** of 12 required review items, and explicit gap flags
  (`unresolved_lineage`, `missing_card_semantics`, `contract_visibility`).

`semantic_response_contract()` returns the exact shape the host must answer with.

**The module deliberately does not judge.** It is evidence. If it fails, `validate` still
succeeds and the packet says `status: unavailable, review_required: true` — validation must never
go down because an advisory packet broke.

**The host's obligation** (from `SKILL.md`): five questions for every filtered field and every
selected stage —

1. **Meaning + unit** — what does this represent? *A plausible key name is not meaning.*
2. **Scope / entity / granularity** — whose value is this **at this point**: the original file, a
   fan-out child, or an aggregate?
3. **Provenance** — measured, a configured *target*, or a relative label? (A resample target
   verifies a value; it does not describe the input. A diarizer's `speaker_0` is a per-recording
   cluster id, not a person.)
4. **Stage effect vs intent** — does this stage **transform** what the user said stays fixed, or
   **drop** rows they expected kept?
5. **Direction** — lower or higher better? (WER/CER drop *above* the bar; quality keeps above.)

Answer: `pass`, `revise`, or `ask` — plus `recipe_config_hash` copied exactly, which **binds the
critique to that exact recipe**. Any recipe change invalidates it.

**4. When used.** After every clean `validate`, **before** smoke. Mandatory, not optional.

**5. Simple example.** For our recipe the packet shows, for `PreserveByValueStage`:

```
field: num_speakers
  latest_producer: InferenceSortformerStage (stage 2)
  semantic_facts.meaning: "Total distinct speakers detected in the recording (passthrough mode only)."
  semantic_facts.scope:   "the whole recording"
  cardinality at this point: 1:1 (no fan-out crossed)
```

Scope is "whole recording", the filter runs at recording granularity, no fan-out was crossed →
`pass`. Had the LLM used `SpeakerSeparationStage` (cardinality `1:N fan-out`), the packet would
show a fan-out seam between producer and consumer, and the honest answer is `revise`.

**6. Connections.** Depends on `index.py` and `_agent_registry.build_contract`. Called from
`verbs.validate`; consumed by the host and graded by `eval/audio/trace_check.py` +
`scenarios/L15_semantic_intent.yaml`.

**7. Key design decisions.**
- *Facts in cards, judgment in the LLM — never module-specific rules in the core.* The temptation
  is to encode "`num_speakers` must not be filtered after separation". That is one special case of
  infinitely many. Putting the *fact* in the card and letting the critic reason generalizes.
- *Configured contracts, not static ones.* Static contracts carry placeholder key values; only a
  configured instance tells you what a stage actually reads here.
- *Failure is non-fatal but loud.* An unavailable packet says "do not infer intent from a green
  verdict; retrieve cards manually".

**8. Failure scenarios.**
| Symptom | Cause | Check |
| --- | --- | --- |
| `semantic_review.status: unavailable` | The packet builder raised | Read `contract_issues[0].message` |
| Packet says `missing_card_semantics` | The producer card lacks `semantic_facts` | Add them (and the `verified.semantic_facts` tier) |
| Lineage says "unresolved" | Producer hidden behind a composite | Check composite expansion depth/leaf caps |

---

### 3.8 `profiler.py`, `env_health.py`, `diagnostics.py`, `failures.py` — situational awareness

**1. What they are.**
- `profiler.py` — `profile_data(path)` (the data) and `probe_env()` (the machine). The agent's eyes.
- `env_health.py` — `doctor()`: a registry of 7 environment checks with concrete fixes.
- `diagnostics.py` — `environment_preflight(stages, env)` (recipe-aware) and `diagnose_failure(error)`.
- `failures.py` — regex matching of error text against `knowledge/failures.yaml`.

**2. Why they exist.** Curator assumes you know your data and that your machine is set up. Both
assumptions fail constantly, and the failures *look like code bugs*.

**3. How they work.**

`profile_data` samples up to 256 files for shape (rates, channels, durations, transcripts
present, manifest columns) and stats up to 100,000 files for the **dataset identity**:

| Tier | Basis | Trust |
| --- | --- | --- |
| `stat` | manifest bytes + sorted `(relpath, size, mtime_ns)` of referenced local files | high — catches ordinary edits |
| `shape` | incomplete identity digest, or the sampled shape hash | low — metadata gaps can hide a mutation |

`dataset_key()` returns `"<tier>:<hash>"`, and the tier travels with **every** reuse decision.

`probe_env` builds `EnvProfile`. Its most important derived property is **`gpu_status`** — the
single source of truth shared by the gate check, the preflight, and the resource planner so they
can never disagree:

| Value | Meaning |
| --- | --- |
| `available` | torch can use ≥1 GPU right now |
| `possibly_masked` | torch cannot, but hardware/driver signals say a GPU is likely **present and merely unreachable** from this (sandboxed) process. **Not a hardware fact** — re-verify, never hard-fail, never say "no GPU" |
| `absent` | definitively no usable GPU (only a CPU-only torch build yields this) |
| `unknown` | no visibility facts at all — treat as re-verify |

`env_health` checks: `python` · `gpu` · **`cuda_driver_toolkit`** · `ffmpeg` · `audio_extras` ·
**`worker_env`** · `disk`. Overall status is the worst individual status. Two deserve attention:

- **`cuda_driver_toolkit`** — torch bundles a CUDA *runtime*; the driver supports up to some max
  CUDA. Basic ops work under minor-version compatibility, but anything that **JIT-compiles PTX at
  runtime** (NeMo's RNNT/TDT ASR decoders) fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
  (error 222). Three grounded fixes: upgrade the driver, install a matching `+cuXXX` torch, or set
  `decoder_type='ctc'` — and the recipe-aware packet only offers the third when the recipe
  actually uses that path.
- **`worker_env`** — pipelines execute in Ray **workers**, not the driver. Launching via
  `uv run` without carrying the extra makes Ray rebuild the worker env from that command line,
  resolving only the *base* dependency set. Result: a driver that imports `soundfile` happily
  beside workers that die on `ModuleNotFoundError`. **This reads like a broken install; it is a
  launch-flag problem.**

`environment_preflight` filters machine facts to *this recipe's* flattened execution leaves and
returns an `EnvironmentDecision`: `status`, `can_execute`, `decision_required`, ranked `choices`
(each with `kind` ∈ host/environment/launch/recipe-variant/credential/diagnostic and
`availability` ∈ available/conditional/unavailable/unknown), `recommended`, one `question`, and a
`host_directive` restating the rules.

**The recovery discipline — the part that matters most.** The core supplies facts and options;
**it never applies them**. The agent must stop, state the fact separately from any inference,
explain the recipe-specific impact, recommend the best *available* option, and ask. It must
never silently install/upgrade, switch to CPU, change a launch command, swap a model or decoder,
request a credential in chat, or retry the same failure. CPU is a **conditional candidate** only
when every affected leaf proves a CPU path — and any such alternative is a **new recipe and a new
hash**: validate → smoke → confirm again. Unknown stays unknown: minimal diagnostics only, never
an invented cause.

**4. When used.** `context`/`doctor` during inspection; `environment_preflight` inside
`validate`/`smoke`/`run`; `diagnose` after any failure.

**5. Simple example.** For our request on a sandboxed run: `doctor` reports
`gpu: warn — not reachable from this process; nvidia_device_nodes=1, torch_cuda_built=true,
gpu_possibly_masked=true`. The agent must say *"GPU not reachable from this (sandboxed) run"* and
re-verify with full device access — **not** "you have no GPU". The planner likewise assumes a GPU
is present and defers to smoke.

**6. Connections.** `profiler` → everything. `env_health` → `diagnostics` → `verbs`. `failures`
→ `diagnostics`.

**7. Key design decisions.**
- *One `gpu_status` property.* Three modules used to answer "is there a GPU?" three ways.
- *Never state an absence as a fact when the observation could be blocked.*
- *A check registry.* New environment concerns are one decorated function, and the design is
  generic — not audio-specific.
- *Diagnosis proposes; it never applies.* Slower recovery, but a machine and a result you can
  still trust.

**8. Failure scenarios.**
| Symptom | Cause | Check |
| --- | --- | --- |
| `has_gpu: false` on a GPU box | Sandbox/container masking | `gpu_possibly_masked`, `nvidia_device_nodes`, `torch_cuda_built`; re-run with full device access |
| `ModuleNotFoundError` in a Ray worker only | Launch-flag problem | `doctor` → `worker_env`; use `.venv/bin/python -m …` |
| `CUDA error 222` / unsupported PTX | Driver older than torch's CUDA | `doctor` → `cuda_driver_toolkit` |
| `diagnose` returns `unknown_failure` | No signature matches | Collect the packet's minimal diagnostics; consider adding a signature to `failures.yaml` |

---

### 3.9 `artifacts.py`, `reuse.py`, `continuation.py`, `run_store.py`, `run_index.py` — reuse

Covered in depth in **§8**. Summary here:

| File | Responsibility |
| --- | --- |
| `artifacts.py` | Step-key Merkle chain, `Artifact` record, atomic `_COMPLETE` publish, `invalid_reasons` vs `caution_reasons`, `lookup` |
| `reuse.py` | `scan()` — longest safely reusable prefix + the human-readable approval card + anti-nag rules |
| `continuation.py` | `materialize()` — rewrite the recipe to start from an artifact; the disk-boundary guard |
| `run_store.py` | JSON run records — **the source of truth** |
| `run_index.py` | Rebuildable SQLite cache; every read tolerates a missing/corrupt DB |

> ⚠️ **Call-out — two reuse engines coexist.** `continuation.plan_continuation` is the *older*
> parent-diff engine (compare against one previous run). `reuse.scan` is the *newer*
> content-addressed engine. `verbs._merge_plans` runs both and takes whichever verified more.
> This is defensible (the parent-diff engine still handles `--parent-run-id`), but it is the
> highest-complexity area of the codebase and a reasonable simplification target. If you touch
> reuse, read `REUSE_ARCHITECTURE.md` first — it documents exactly why the first design failed.

---

### 3.10 Remaining files — compact reference

| File | Purpose | Why separate | Called by | Debug check |
| --- | --- | --- | --- | --- |
| `contracts.py` | All typed JSON-safe data objects (`DataProfile`, `EnvProfile`, `Verdict`, `SmokeReport`, `AcceptanceCriterion`, `AcceptanceReport`, `RunRecord`, `ConfigStrategyEntry`, `Issue`) | One place defines the wire format for every verb | Everything | `_validated_acceptance_mapping` raises on malformed criteria — read the message |
| `config_strategy.py` | `resolve()`: outcome → parameter via card anchors/presets | Keeps thresholds out of the LLM's head *and* out of user conversations | `verbs.resolve` | Card has `metrics.anchors`? Is `direction` right? |
| `acceptance.py` | Compile criteria → roles; request-type sanity; `verify()`; **honesty guard** | Success must be defined before and checked after, immune to optimism | `verbs.validate/run/verify`, `checks` | `overall` forced `not_met`? Read `honesty[]` |
| `planner.py` | Streaming vs batch + feasibility; composite flattening | Mode selection must match the scheduler's real constraints | `verbs.smoke/run` | `escalations[]` names the exact dimension |
| `calibration.py` | Extract measured resources from a smoke | Card estimates are guesses; measurements refine them **upward only** | `verbs.smoke/run` | Machine fingerprint must match or it is ignored |
| `report.py` | `RunReport`; `stage_duration_sec` reading convention | Fixes fan-out double-counting; one reader beside the writer | `verbs.run/smoke/report` | `cardinality_proven`? `source_items` vs `output_rows` |
| `context.py` | Packages index + profiles into `PlanningContext` | Pure assembly; selection stays with the host | `verbs.context` | Is `data_profile` null? (no `--data` passed) |
| `input_identity.py` | Which dataset will actually be read (closed adapter table for 5 source stages) | Guessing from parameter names would serve wrong bytes | `verbs.*` | `binding.status` ∈ resolved/missing/mismatch/ambiguous/unsupported |
| `_safety.py` | Workspace lock, secret+transcript redaction, smoke token | Guardrails a weak host cannot bypass | `verbs.*` | `AUDIO_AGENT_WORKSPACE`, `AUDIO_AGENT_REQUIRE_SMOKE` |
| `_ray.py` | Opt-in local Ray head (free port, writable plasma dir, API cap) | Ray setup fails in unhelpful ways; opt-in leaves existing clusters alone | `verbs.smoke/run` | Respects `RAY_ADDRESS`; never clobbers it |
| `failures.py` | Error text → classified failure | Deterministic, versioned, additive | `diagnostics` | Signatures tried in file order, specific-before-generic |
| `card_conformance.py` | Gate: cards must match real stages | Documentation rots; gates do not | CI / manual | `python -m nemo_curator.audio_agent.card_conformance` |
| `_resolve.py` | Name → class / Hydra target | The anti-hallucination lookup | `recipe.py` | `KeyError` = unknown stage |
| `cli.py` | One subcommand per verb, prints JSON | Shell/notebook surface | Humans, agents | `_criteria_list` fails **loud** on a malformed criteria file |
| `mcp_server.py` | Same verbs as MCP tools | Native tools for MCP hosts | MCP clients | `mcp` is an optional dep |
| `run_store.py` / `run_index.py` | JSON records (truth) + SQLite cache | Nothing lives only in the DB | `verbs` | `reindex` rebuilds from JSON |
| `stages/audio/_catalog.py` | Discovery + role index over the framework registry | No new registry introduced | `index.py`, `_resolve.py` | `_ensure_audio_stages_imported` must run first |
| `stages/audio/_conformance.py` | `assert_agent_ready()` for stage owners | One test tells an owner what is missing | Stage tests | Run it when adding a stage |
| `stages/audio/_residency.py` | file-vs-waveform handling | Single source of truth so `accepts` cannot lie | Stages | A `file`-mode instance must not advertise `waveform` |

---

### Presentation questions — Files

**Q: Why is `verbs.py` 4,000 lines? Isn't that a smell?**
It is the orchestrator, and the alternative is worse: splitting it means the confirm gate, the
redaction boundary, and the source binding exist in several places. Each verb is a flat,
readable sequence; the *logic* lives in the small modules it calls. That said, `run` and `smoke`
are long and are the honest refactor candidates.

**Q: How easy is it to add a new audio module?**
For a non-source stage: inherit `AgentReady`, implement `describe()` with reads/writes/
cardinality/gates, make every key a `*_key` constructor field, add `assert_agent_ready` to your
test, and write a capability card. **No change to the agent core.** A new *source* stage
additionally needs an entry in `input_identity.py`'s adapter table — deliberately a closed table,
because getting dataset identity wrong means serving the wrong bytes.

**Q: What stops a card from going stale?**
`card_conformance.py`. It fails on a `stage_id` that doesn't resolve, a `params_of_note`/`presets`
key that isn't a real constructor parameter, an unknown resource key, a `model_id` without a
pinned `model_version`, an invalid metric direction or `threshold_param`, or a capability tag
that contradicts the stage's default behavior.

---

## 4. End-to-End Execution Flow

> **Request:** *"Transcribe the audio data, keep only single-speaker audio, and resample the
> output to 48 kHz."*
> **Data:** `/data/raw/` — a folder of 48 kHz stereo WAVs, no transcripts.

---

**Step 1 — The request arrives.**
The host matches the request against the `audio-curation` skill's `description` and loads
`nemo_curator/audio_agent/skills/audio-curation/SKILL.md`. *No core code has run yet.*

**Step 2 — Interpret intent + define success.** **[LLM]** · `SKILL.md` §1

The model produces a goal and a **capability plan**:
```json
{"task": "transcribe+filter+convert", "domain": "unknown",
 "expected_outputs": ["pred_text", "48kHz audio", "manifest"],
 "capability_areas": ["ingest","preprocess","diarize","transcribe","export"],
 "open_questions": ["is 48 kHz needed for the audio files or just recorded in the manifest?"]}
```
And a **success contract** (this is what "done" means, and it goes *inside* the recipe):
```yaml
- {id: transcripts_present, type: output_completeness, severity: must,
   check: {field: pred_text, op: non_empty, scope: per_retained_item}}
- {id: single_speaker_only, type: yield, severity: must,
   check: {field: retained, op: ">", value: 0}}
```
The refuse list is checked (nothing subjective here → proceed). One outcome-level clarification
may be asked; internal parameters never are.

**Step 3 — Inspect the data and the machine.** **[Core]**
`verbs.context` → `context.assemble` → `profiler.profile_data` + `profiler.probe_env` +
`index.category_tree/match_blueprints` + `env_health.env_report`

Returns: `num_files: 1,240`, `sample_rates: {48000: 1240}`, `channels: {2: 1240}`,
`has_transcripts: false`, `dataset_key: "stat:9f3c…"`, plus GPU/ffmpeg/extras facts.
**The agent tells the user this** — "your audio is 48 kHz stereo with no transcripts" is
frequently news.

**Step 4 — Health check.** **[Core facts, LLM explains]**
`verbs.doctor` → `env_health.doctor()` → 7 checks. Run before heavy GPU work. If
`gpu_possibly_masked`, the agent says "not reachable from this sandboxed run", not "no GPU".

**Step 5 — Route coarse-to-fine.** **[LLM over core data]** · `index.py`

- **L0** `catalog_tree()` → prune to `ingest`, `preprocess`, `diarize`, `transcribe`, `export`.
  (No quality bar was requested → **skip `quality`/`filter` entirely**. `SKILL.md` is explicit:
  a stage earns its place only if it serves a stated goal.)
- **L1** `cards --category diarize` → 3 candidates.
- **L2** `cards --names InferenceSortformerStage SpeakerSeparationStage` → the decisive fact:

  > Sortformer: *"Annotates only — does **NOT** separate audio"*, writes `num_speakers`.
  > SpeakerSeparation: transforms audio into per-speaker streams.

  **The user wants to *select* clips, not modify them → Sortformer + a value filter.**

**Step 6 — Handle the hidden conflict.** **[LLM reading card constraints]**

L2 cards reveal:
- `InferenceSortformerStage.constraints.supported_sample_rates: [16000]` — "stage does not resample"
- `InferenceAsrNemoStage.constraints.supported_sample_rates: [16000]` — same
- `UTMOSFilterStage` would *not* need this (it resamples internally) — but we aren't using it

So **the models need 16 kHz and the user wants 48 kHz output.** The resolution: resample to
16 kHz for the model stages, then resample the *output* to 48 kHz at the end. This is exactly the
kind of thing a naive generator gets wrong and this system catches (see step 9).

**Step 7 — Resolve configuration.** **[Core]** · `config_strategy.resolve`
Here there is no quality threshold to resolve. The equivalent decision is **`resources`**:
Sortformer and ASR cards are `bound: gpu`, `gpu_optional: false` → both get
`resources: {gpus: 1}`. Leaving the default would run them on CPU (very slowly) and could
over-parallelize into many model-loading actors — `checks._check_gpu_reservation` warns if you
forget.

**Step 8 — Emit the Recipe.** **[LLM]** · consumed by `recipe.py`
The recipe shown in §3.2. Note the acceptance criteria are **embedded** — a separate file alone
is not executable intent.

**Step 9 — Validate.** **[Core]** · `verbs.validate` → `checks.run_checks`

*Suppose the first draft resampled to 48 kHz up front.* The verdict:
```json
{"runnable": false,
 "card_violations": [
   {"code":"card_sample_rate","severity":"warning","stage":"InferenceSortformerStage",
    "message":"input sample rates [48000] not all in supported [16000]",
    "fix":"insert a resample stage upstream to the supported rate"}],
 "issues":[{"code":"card_batch_size","severity":"error","stage":"InferenceSortformerStage",
    "message":"batch_size must be 1 (streaming variant keeps per-clip FIFO/cache state)"}]}
```
The LLM fixes and re-validates (≤3 iterations). The corrected recipe returns
`runnable: true, status: pass`, plus `output_targets` (what already exists at each write path),
`environment_decision`, and `data_binding`.

Inside, in order: path safety → source binding (`input_identity`) → profile → `build_stages`
(real classes, real params) → the 9 checks → semantic packet → environment preflight →
output targets.

**Step 10 — Semantic critique.** **[LLM over the core's packet]** · `semantic_review.py`

The packet shows `num_speakers` produced by `InferenceSortformerStage`, scope
*"the whole recording"*, no fan-out seam between producer and consumer. Both resamples are
checked as **transform_checks** — each sets `update_audio_filepath: true`, so downstream really
does read the converted file rather than silently scoring the original. `intent_status: pass`,
with `recipe_config_hash` copied exactly.

*(Had the LLM chosen `SpeakerSeparationStage`, the packet would show a `1:N fan-out` seam between
producer and filter, and the honest answer is `revise`.)*

**Step 11 — Reuse scan.** **[Core proposes, user decides]** · `verbs.reuse_scan` → `reuse.scan`
`artifacts.plan_steps` computes the Merkle chain from `dataset_key`; each key is probed. First
run → nothing found → `decision: fresh`, `prompt_user: false`. **No candidate means no question.**

**Step 12 — Smoke.** **[Core]** · `verbs.smoke`
Bounds the input to 10 items, **isolates every output into a temp tree** (and refuses if isolation
cannot be *proven*), runs the real models, returns retained/rejected/examples/errors plus a
`calibration` block (measured VRAM/throughput) and a `smoke_token`.

Say it returns `retained: 6, rejected: 4` — 4 clips had 2+ speakers. That is real evidence about
this corpus, obtained for ~1% of the full cost.

**Step 13 — Confirm gate.** **[LLM presents, user decides, core enforces]**

The agent presents: the plan and why each stage is there; the semantic critique; smoke evidence;
the scale estimate (1,240 files); **and the acceptance contract, stating what each metric captures
and what it does not**. Nothing has been written yet.

`run` without confirmation returns:
```json
{"status":"refused","reason":"full run requires explicit confirmation (0 silent full-scale runs)",
 "config_hash":"a7f3…","confirm_with":"pass confirm='a7f3…' to proceed"}
```

**Step 14 — Run.** **[Core]** · `verbs.run`
Confirm hash matched → optional smoke-token check → source binding re-verified → `build_stages` →
`planner.plan` (streaming vs batch; **GPU reservations gate, VRAM only warns**) → optional Ray
bootstrap → `Pipeline.build()` → execute with streaming→batch auto-fallback.

**Step 15 — Failure handling (if it happens).** **[Core classifies, LLM explains]**
An error goes to `failures.classify` (18 signatures) and `diagnostics.diagnose_failure`, which
adds a fresh environment probe, the affected stages, and ranked options. The agent explains and
**asks**. It never installs, switches device, swaps a model, or retries silently.

**Step 16 — Validate the result.** **[Core]** · `verbs._acceptance_result` → `acceptance.verify`
Reads back the terminal manifest exhaustively (up to 2,000 rows of evidence) and evaluates the
frozen contract:

| Criterion | Result |
| --- | --- |
| `transcripts_present` | `met` — every retained row has non-empty `pred_text` |
| `single_speaker_only` | `met` — 742 rows retained of 1,240 |

Then `honesty_review` compares what was verified against the frozen contract. Empty → fine. Any
dropped/downgraded/relaxed `must` **forces `overall: not_met`**.

**Step 17 — Publish artifacts + record the run.** **[Core]**
`verbs._publish_artifacts` writes an `Artifact` per persisted step, each with an atomic
`_COMPLETE` marker containing the step key, row count, byte count, and content digest.
`run_store.save` writes the JSON `RunRecord` (with `goal`, step chain, per-stage metrics,
acceptance outcome, versions); `run_index.index_run` caches it in SQLite.

**Step 18 — Report.** **[Core evidence, LLM narrates]**
`report.build_run_report`: retained/rejected, per-filter counts, failure reasons + examples,
output paths, elapsed time, and explicit `source_items` vs `output_rows` (so fan-out rows are
never mistaken for retained inputs). The agent explains it in plain language and resolves any
judgment-based criteria.

---

### The same request, one week later

> *"Now also add quality scores to those transcripts."*

Steps 1–10 repeat. At **step 11** the picture changes: `plan_steps` recomputes the chain; the
first *N* keys are unchanged (same data, same stages, same semantic params) and the terminal
writer's artifact matches. The scan returns `incremental` or `already_done` with a card showing
what the earlier run was *for*, when, and the measured time saved. See **§8** for the full trace.

---

### Presentation questions — End-to-end flow

**Q: What is the single point where a hallucination would be caught?**
`recipe.build_stages` → `_resolve.resolve_stage_class`. An invented name raises `KeyError` →
`unknown_stage`. An invented parameter raises `TypeError` → `bad_params`, listing what *is*
accepted. Neither can reach execution.

**Q: How many chances does the agent get to be wrong before it costs real money?**
Four gates before any full-scale GPU time: mechanical validation, semantic critique, the bounded
smoke, and the human confirm gate. Plus reuse, which may make the cost zero.

**Q: What if the user's request is impossible?**
The `unproducible` check names the role that **no stage in the catalog** can produce, and the
agent tells the user the goal is not achievable with the available stages — rather than shipping
a plausible pipeline that quietly does something else.

---

## 5. Deterministic vs. LLM-Controlled Logic

| Decision | Owner | Why assigned there | Risk if flipped |
| --- | --- | --- | --- |
| Understand an ambiguous request | **LLM** | Language is the LLM's home turf | Deterministic parsing = a rigid command language; the usability win disappears |
| Decide what "clean"/"single-speaker" means here | **LLM** | Needs context and a conversation | Hard-coded meanings break on the next phrasing |
| Choose between overlapping modules | **LLM**, from card facts | A trade-off, not a rule | A fixed preference table picks wrong whenever the goal shifts |
| Explain *why* a choice was made | **LLM** | Explanation is reasoning | Templated rationales that don't match the actual decision |
| Judge whether a valid recipe means the right thing | **LLM**, over the evidence packet | Open-ended; does not reduce to rules | An ever-growing, always-incomplete rule table — and green pipelines answering the wrong question |
| Decide what is material enough to ask about | **LLM** | Requires judgment | Either an interrogation or silent wrong assumptions |
| **What stages exist** | **Core** (`index`, `_catalog`) | A fact. Facts must not be generated | Hallucinated stages |
| **What a parameter is called** | **Core** (`_agent_registry`) | A fact | Hallucinated parameters; runtime `TypeError` |
| **Whether a pipeline composes** | **Core** (`_planning`, `checks`) | Provable by graph walk | Plausible-looking pipelines that crash mid-GPU-run |
| **What threshold "studio" means** | **Core** (`config_strategy` + card anchors) | Must be repeatable and auditable | Invented numbers; inverted filters; different answer each run |
| **Filter direction (higher/lower better)** | **Core** (card `metrics.scale.direction`) | A fact about the metric | Silently keeping exactly the data the user wanted dropped |
| **What the machine can do** | **Core** (`profiler`, `env_health`) | Measurable | Confidently wrong hardware claims; plans that cannot schedule |
| **Whether execution may start at scale** | **Core** gate + **user** decision | Must be un-skippable | Silent multi-hour GPU runs |
| **Whether the plan changed after approval** | **Core** (`config_hash`) | Byte-level integrity | Approving one thing and running another |
| **What actually happened** | **Core** (`report`) | Evidence, not narration | Confident unverified claims |
| **Whether success criteria were met** | **Core** (`acceptance.verify`) | Must be immune to optimism | An agent that always says "success" |
| **Whether a `must` bar was weakened** | **Core** (`honesty_review`) | Must be immune to the LLM's own judgment | Silent goalpost-moving — the previous row becomes decoration |
| **Whether prior work can be reused** | **Core** (`artifacts`, `reuse`) | Must be provable, not plausible | Serving a stale or wrong result silently |
| **Whether to apply an environment fix** | **User** (core proposes, LLM explains) | Mutates the user's machine | An agent that "fixes" CUDA by installing packages and switching devices |

**The governing rule:**

> Deterministic checks own **universal invariants**: real stages, impossible key flow,
> serialization, environment gates, side effects, confirmation integrity.
> The LLM owns **intent-dependent meaning**: what a field means here, whether this metric is a
> good proxy, which trade-off fits.
> **Never add module-specific intent rules to the core.**

---

### Presentation questions — Determinism

**Q: Why not build a fully LLM-driven agent?**
Five concrete reasons, all observable in this codebase: (1) hallucinated names cannot execute
here *by construction*, not by prompting; (2) the same request must produce the same approved
plan, or approval is meaningless; (3) tiered retrieval is both cheaper and *more* accurate than
dumping 47 cards; (4) an LLM cannot prove a graph composes or that a machine can schedule a
pipeline; (5) an LLM asked to grade its own success will grade generously — `honesty_review`
exists because that is not hypothetical.

**Q: Then why use an LLM at all?**
Because everything the deterministic layers *cannot* do is exactly where the value is:
interpreting a vague request, weighing two reasonable modules, noticing that a green pipeline
answers the wrong question, and explaining a trade-off in plain language.

**Q: Which part is most important for reliability?**
The confirm gate bound to `config_hash` — it is the only thing standing between a plan and hours
of irreversible compute. Close second: the semantic review layer, because it catches the failures
that are *silent*.

---

## 6. Important Data Structures

**1. Goal / capability plan** — LLM-produced, not a core type.
```json
{"task":"transcribe+filter+convert","domain":"unknown",
 "expected_outputs":["pred_text","48kHz audio"],
 "capability_areas":["diarize","transcribe","preprocess"],
 "open_questions":["48 kHz for the files, or just recorded?"]}
```

**2. `AcceptanceCriterion`** (`contracts.py`) — one checkable condition of success.
```yaml
id: transcripts_present
type: output_completeness      # output_completeness | quality_standard | yield | distribution | semantic_fit
kind: absolute
check: {field: pred_text, op: non_empty, scope: per_retained_item}
severity: must                 # must | nice
on_unachievable: escalate
```
`check.field` is **any** key — the verifier contains no metric names.

**3. `StageContract`** (`_agent_ready.py`) — the mechanical truth about a stage.
```json
{"stage_id":"InferenceSortformerStage",
 "reads":{"data_keys":["audio_filepath"],"accepts":["file"]},
 "writes":{"data_keys":["diar_segments","num_speakers"]},
 "cardinality":"1:1",
 "gates":{"requires_gpu":true,"requires_internet_first_run":true},
 "key_roles":{"diar_segments":"diar_segments","num_speakers":"num_speakers"},
 "contract_resolution":"configured"}
```

**4. Capability card** (`knowledge/cards/*.yaml`) — the *meaning* the code cannot express.
```yaml
stage_id: UTMOSFilterStage
metrics:
  utmos_mos:
    scale: {min: 0.0, max: 5.0, direction: higher_better}
    threshold_param: mos_threshold
    anchors: {"4.0-5.0": studio, "3.4-4.0": general, "2.5-3.4": lenient}
semantic_facts:
  utmos_mos:
    meaning: "Predicted overall perceptual naturalness for the item scored."
    scope: "mode='task' → one score per task; mode='segments' → one per nested segment"
    counterexamples: ["A high score does not prove low background noise."]
verified: {params: mechanical, resource: best_guess, metrics: best_guess}
```

**5. `Recipe`** — see §3.2.

**6. `Verdict`** (`contracts.py`) — the output of `validate`.
```json
{"status":"pass","runnable":true,"ok":true,"keys_ok":true,
 "produced_roles":["audio_filepath","diar_segments","num_speakers","pred_text"],
 "issues":[],"card_violations":[],"gate_flags":[{"code":"internet_first_run","severity":"info"}],
 "unproducible_roles":[],
 "output_targets":[{"path":"/out/curated.jsonl","exists":false}],
 "data_binding":{"status":"resolved","primary_path":"/data/raw"},
 "environment_decision":{"status":"ready","can_execute":true,"decision_required":false},
 "semantic_review":{"status":"ok","review_required":true,"recipe":{"config_hash":"a7f3…"},"checklist":[…]}}
```
> ⚠️ **Gate on `runnable` or `status == "pass"`, never on `ok`.** `ok` is data-flow only; a
> recipe can be `ok: true` while a card constraint or environment gate makes it unrunnable. The
> docstring says this explicitly because it has been misread.

**7. `StepPlan` / `Artifact`** (`artifacts.py`) — reuse identity.
```json
{"index":4,"stage_ref":"InferenceAsrNemoStage",
 "step_key":"8d47b03e7846db1ca787e8c4","input_key":"3e5dd5611a273aad18b513c2",
 "uri":"/out/curated.jsonl","kind":"manifest","deterministic":true}
```
```json
{"step_key":"8d47…","stage_ref":"ManifestWriterStage","uri":"/out/curated.jsonl",
 "rows_in":1240,"rows_out":742,"content_digest":"c9e1…",
 "duration_sec":41.2,"cumulative_sec":5122.7,"gpu_seconds":4870.1,
 "dataset_key":"stat:9f3c…","fingerprint_tier":"stat","status":"complete"}
```

**8. `RunRecord`** (`contracts.py`) — provenance, one JSON per run.
```json
{"run_id":"run-20260803T091500.123456Z-a7f31b2c-9d4e",
 "config_hash":"a7f3…","semantic_hash":"5b21…","contract_hash":"77aa…",
 "goal":{"task":"transcribe+filter+convert"},
 "dataset_key":"stat:9f3c…","fingerprint_tier":"stat",
 "steps":["4917…","cf1b…","298b…","38e9…","8d47…"],
 "status":"completed","accepted":742,"input_count":1240,"elapsed_sec":5164.0,
 "acceptance_result":{"overall":"met"},"curator_version":"…","reuse":{}}
```

**9. `AcceptanceReport`** — the post-run verdict.
```json
{"overall":"met",
 "criteria":[{"id":"transcripts_present","status":"met","severity":"must",
              "evidence":"742/742 retained rows have non-empty pred_text"}],
 "honesty":[]}
```
Four honest states: `met` · `not_met` · `unverifiable` (no evidence exists) · `unachievable`
(the data provably cannot reach an absolute target).

**10. `EnvironmentDecision`** — recipe-aware environment packet.
```json
{"status":"action_required","can_execute":false,"decision_required":true,
 "issues":[{"code":"cuda_driver_toolkit","blocking":true,
            "affected":["InferenceAsrNemoStage"],"confidence":"high"}],
 "choices":[{"id":"upgrade_driver","kind":"host_change","availability":"available"},
            {"id":"ctc_decoder","kind":"recipe_variant","availability":"conditional"}],
 "recommended":"ctc_decoder","question":"…"}
```

**11. Final manifest** — the actual output.
```json
{"audio_filepath":"/out/48k/clip_001.wav","duration":6.2,
 "num_speakers":1,"pred_text":"the quick brown fox","sample_rate":48000}
```

---

## 7. Configuration and Control Flow

### 7.1 Where every value comes from, in priority order

**For a stage parameter:**

| Priority | Source | Mechanism |
| --- | --- | --- |
| 1 | **User-explicit value** | `resolve --explicit '{"p": v}'` → used as-is |
| 2 | **Card preset** | `resolve --use-case tts_reference` → the named bundle |
| 3 | **Card anchor** (outcome label) | `resolve --label studio` → via `metrics.anchors` + `direction` |
| 4 | **LLM choice grounded in card facts** | Module selection, ordering, `resources` sizing |
| 5 | **Stage constructor default** | Whatever the dataclass says |

**Relative goals ("the best 20%") are refused, not guessed** — `resolve` returns an `ask`.

**For the dataset that will be read:**

> **The recipe's first supported source stage's configured parameters are execution truth.**
> `Recipe.inputs` and the CLI `--data` value are *assertions only*. Neither injects nor rewrites
> a stage. Omit `--data`, or pass the same canonical source. A mismatch is **refused before
> execution**.

The exception: `context --data` profiles the path directly, because it runs *before* a recipe
exists.

**For execution mode and resources:**

| Priority | Source |
| --- | --- |
| 1 | Ray reservations read from the **built stage** (`stage.resources`) — scheduling truth |
| 2 | **Measured calibration** from a smoke, on a matching machine fingerprint — **may only raise, never lower** a card estimate |
| 3 | **Card `resource` facts** |
| 4 | Conservative defaults (2.0 GB VRAM for a GPU stage, 1.0 GB host RAM) |

**For environment facts:** always a *fresh* probe. Nothing is cached across calls; `probe_env()`
re-reads current reality every time.

### 7.2 Environment variables

| Variable | Effect |
| --- | --- |
| `AUDIO_AGENT_WORKSPACE` | Path lock — every path must resolve under this root. Also relocates the runs directory |
| `AUDIO_AGENT_RUNS_DIR` | Explicit runs directory (highest priority) |
| `AUDIO_AGENT_REQUIRE_SMOKE` | `run` refuses without a valid `--smoke-token` for this exact recipe |
| `AUDIO_AGENT_SMOKE_SECRET` | HMAC secret for smoke tokens (per-process random if unset) |
| `RAY_ADDRESS` | Existing cluster — **always respected, never clobbered** |
| `AUDIO_AGENT_EVAL_EXECUTE` | Opt into the eval harness's real-smoke branch |

Runs directory resolution: `AUDIO_AGENT_RUNS_DIR` → `<AUDIO_AGENT_WORKSPACE>/.audio_agent_runs`
→ `<cwd>/.audio_agent_runs`.

### 7.3 Which layer decides what

```
LLM decides    : which stages, what order, which module among alternatives,
                 which outcome label, what to ask the user, intent correctness
Cards decide   : what a threshold means, filter direction, resource estimates,
                 model limits, what an output field means
Core decides   : whether it composes, whether it can schedule, whether it may run,
                 whether prior work is reusable, whether success was met
User decides   : full-scale execution, environment changes, reuse choices,
                 any relaxation of a confirmed success bar
```

---

## 8. Reuse of Previous Work

> **Read `REUSE_ARCHITECTURE.md` before changing anything here.** It documents why the first
> design failed, which is the fastest way to avoid rebuilding it.

### 8.1 Why the first design failed (instructive)

The original engine diffed a new recipe against one parent `RunRecord`. Correct, and it almost
never fired:

1. **Identity conflated three things.** `config_hash` covered execution knobs and output paths,
   so changing a batch size, a GPU reservation, an output directory, or *tightening the success
   bar* changed identity **without changing a single output byte**. Every one a false negative.
2. **The dataset key was a shape hash** — simultaneously *unsafe* (a file edited in place was
   invisible) and *too coarse* (adding one file invalidated everything).
3. **Reuse was all-or-nothing and could not execute** — it printed a plan a human had to
   hand-implement.
4. **Intermediates were invisible** — output discovery only looked at a few literal parameter names.
5. **No cost data, no index** — "estimated time saved" was not computable.

### 8.2 The fix: change the question

Stop asking *"was this whole recipe run before?"* Start asking
**"has this step, with these semantics and this dataset identity, already produced an artifact?"**

```
dataset_key = tiered_fingerprint(resolved source)
step_key(0) = H(dataset_key,   ref, semantic_params, code_version, model_version)
step_key(i) = H(step_key(i-1), ref, semantic_params, code_version, model_version)
```

One mechanism gives whole-pipeline, prefix, and single-stage reuse — and **invalidation falls out
of the chain** instead of needing hand-written rules:

| Change | Effect |
| --- | --- |
| Semantic param at step *i* | keys *i…n* change → reuse *0…i-1* |
| Batch size / GPU reservation | no key change → full reuse |
| Output location | no key change → full reuse, published to the new place |
| Acceptance criteria | data reused, **contract re-verified** |
| Detectable data change | `dataset_key` changes → everything reruns |

### 8.3 What is stored

- `.audio_agent_runs/<run_id>.json` — the `RunRecord` (**source of truth**)
- `.audio_agent_runs/artifacts/<step_key>.json` — one `Artifact` per persisted step
- `<uri>._COMPLETE` / `<uri>/_COMPLETE` — the atomic marker: step key, rows, bytes, **content
  digest**, timestamp
- `.audio_agent_runs/index.db` — a **rebuildable** SQLite cache (`reindex` restores it)

### 8.4 When reuse is safe — and when it is not

`artifacts.invalid_reasons()` — **any one disqualifies** (an unknown fails closed):
marker present and matching · URI still exists · marker + record + **current bytes** share one
content digest · inside the workspace lock · `dataset_key` matches · `code_version`/`model_version`
compatible · not TTL-expired · `status == "complete"`.

`artifacts.caution_reasons()` — **does not disqualify; requires an explicit yes:**
the stage declared `deterministic: false`; the dataset identity is only `shape` tier.

> The split matters. These cautions used to sit in `invalid_reasons`, which meant a
> non-deterministic stage's artifact did not become a warned-about candidate — it **vanished**,
> and the user was never told prior work existed. Now it downgrades trust and asks.

**Content digests are recomputed at lookup.** Editing a manifest after publication invalidates
reuse even if row count and byte size are unchanged.

**The disk boundary.** Reuse resumes from a *persisted* artifact, which cannot carry an in-memory
waveform. `continuation._resume_breaks_on_disk_boundary` re-validates the remaining stages with
and without the waveform role and reports only reads that break *specifically* because it was
dropped.

### 8.5 The approval UX — never silent, never nagging

In priority order (`reuse.scan`):

1. **No candidate → no prompt.** Run fresh, say nothing.
2. **Measured saving < 30 s → just take it**, and disclose it in the report.
3. **Low trust → default to fresh**, with the weakness written for a human.
4. **Otherwise → the three-way choice**: `as_is` / `extend` / `fresh`.

> **"Measured" carries weight.** Silence about a stage's cost is not evidence it was cheap. An
> unmeasured stage counted as zero seconds, so an unmeasured hour of transcription once qualified
> as "trivial". Now `_unpriced()` lists unmeasured stages the cards call expensive
> (`bound: gpu`, a declared model, a network fetch) and **forces the question**.

`cumulative_sec` — time from the *source* to this step — is what reuse actually saves, and it is
what the 30-second threshold is measured against. Charging reuse only the writer's milliseconds
would auto-serve an hour-old ASR result without ever asking.

**Explaining a miss.** When the data changed, every key changes and the probe finds *nothing*.
Reporting "never ran this before" would be true and useless. So the scan re-keys against datasets
already in the registry and reports `prior_on_other_data` naming the earlier dataset and date.
Separately, `prior_unsaved` reports a prefix that matches an earlier completed run but persisted
nothing — *"this ran before and is being recomputed"* — plus an offer to add a writer next time.
**Recomputation is never silently presented as new work.**

### 8.6 Worked example: transcription is done, now add quality scores

**Week 1.** Our recipe runs. Artifacts are published; the terminal writer's artifact holds
`step_key = 8d47…`, `cumulative_sec = 5122.7`, `dataset_key = stat:9f3c…`.

**Week 2.** *"Now also add quality scores."* The LLM appends `UTMOSFilterStage(action=annotate)`
before the writer and re-validates.

`reuse_scan` →
```json
{"decision":"incremental",
 "reuse_stages":["CreateInitialManifestAudioFolderStage","ResampleAudioStage",
                 "InferenceSortformerStage","PreserveByValueStage","InferenceAsrNemoStage"],
 "run_stages":["UTMOSFilterStage","ResampleAudioStage","ManifestWriterStage"],
 "reuse_point":{"stage_index":4,"uri":"/out/asr_manifest.jsonl","kind":"manifest"},
 "estimated_saving_sec":5122.7,"saving_is_lower_bound":false,
 "prompt_user":true,"recommended":"extend",
 "candidates":[{"objective":"transcribe + keep single-speaker + 48 kHz",
                "date":"2026-07-27","rows_out":742,"trust":"high"}]}
```

The agent shows the card and asks. On `extend`:

`continuation.materialize` drops the first 5 stages and prepends
`ManifestReader(manifest_path="/out/asr_manifest.jsonl")` → `validate` re-checks the remainder
**seeded from what the artifact actually carries** (plus the disk-boundary guard) → the same
confirm gate → `run` executes only the tail.

**~85 minutes of GPU time skipped. The user was asked. The lineage is in the report.**

**One subtlety worth knowing.** The tail is published under the step keys of the recipe the
**user asked for**, not the rewritten one (`verbs._logical_identity`). The rewritten recipe
describes a pipeline nobody requested — a reader over an intermediate file — so registering the
tail under it would leave the same follow-up finding nothing and recomputing forever. Repeating
the request now returns `already_done`.

### 8.7 Honest limits

Only a stage that **writes something** can be a resume point. For the common shape (in-memory
transforms feeding one writer), the realistic outcomes are `already_done` and `fresh` —
`incremental` needs an intermediate that persists. This is a property of resuming from disk, not
a lookup gap, and it is **disclosed** via `prior_unsaved` rather than hidden.

Not built (designed, in `REUSE_ARCHITECTURE.md` §11): content-digest identity tiers above `stat`,
per-file dataset deltas, compatible-superset (T2) reuse, artifact GC/retention, a `why-rerun`
explain verb, row-level lineage across fan-out.

---

### Presentation questions — Reuse

**Q: How does it avoid repeating expensive processing?**
Content-addressed step keys. Each step's identity is a hash of the dataset identity plus every
prior step plus this step's semantic parameters, code version, and model version. If a key
matches a published artifact that passes all validity checks, the work is skipped.

**Q: What stops it serving me a stale result?**
Eight validity conditions, all of which must hold, plus a content digest recomputed at lookup
time. Editing the output after publication invalidates reuse even if size and row count match.
And low-trust matches default to **fresh**.

**Q: Do I always get asked?**
No — deliberately. Nothing to reuse means no question. A *measured* saving under 30 seconds is
taken and disclosed. Everything else asks. The two failure modes we designed against are "you
silently got yesterday's answer" and "you stopped reading the prompts".

---

## 9. Health Checks and Safety Mechanisms

### 9.1 Before execution

| Check | Where | Catches | Why it must exist |
| --- | --- | --- | --- |
| Workspace path lock | `_safety.path_violations` | Traversal, writes outside the allowed root | An agent with filesystem reach and no boundary |
| Source binding | `input_identity.resolve_dataset_binding` | `--data` disagreeing with the recipe; ambiguous or unsupported sources | Processing the wrong dataset silently |
| Data readability | `profiler.profile_data` → `source_errors` | Corrupt manifest, bad JSON | Execution cannot be trusted to consume the whole dataset |
| Stage existence & params | `recipe.build_stages` | Invented names/parameters, missing extras | **The anti-hallucination gate** |
| Role/key composition | `_planning.validate_pipeline` | `unsatisfied_reads`, `dangling_key`, `key_removed_upstream`, `ambiguous_default_key` | A pipeline that crashes at stage 4 after paying for 1–3 |
| Serialization | `_gate_issues` → `tensor_into_sink` | A tensor reaching a JSON writer | A crash at the very last stage |
| Card constraints | `checks._check_card_constraints` | Fixed batch size, unsupported rates, duration, max speakers | Model-level failures that look like bugs |
| GPU reservation | `checks._check_gpu_reservation` | A GPU stage left at its CPU default | Silent 50× slowdown + actor over-parallelization |
| Environment gates | `checks._check_gates` | ffmpeg, GPU, secrets, first-run download | Setup failures mid-run |
| Environment preflight | `diagnostics.environment_preflight` | Recipe-relevant blockers only | An irrelevant CUDA warning must not block a CPU recipe |
| Unproducible roles | `checks._check_unproducible` | A capability **nothing** in the catalog provides | Telling the user honestly instead of shipping a wrong pipeline |
| Output completeness | `checks._check_output_completeness` | "Asked for transcripts, no ASR stage" | Discovering it after the run |
| Request-type sanity | `checks._check_request_type_sanity` | A filtering request with no yield criterion | Declaring success while ignoring the point |
| Resource feasibility | `planner.plan` | Reservations that cannot schedule | An executor abort after setup |
| Output occupancy | `verbs._output_targets` | What already exists at each write path | **Facts, so nobody guesses or pre-cleans** |
| Semantic review | `semantic_review` + host critique | Wrong field, wrong granularity, wrong stage effect, inverted direction | Green pipelines answering the wrong question |
| Reuse validity | `artifacts.invalid_reasons` | Stale/partial/moved/changed artifacts | Serving a wrong result |
| Bounded smoke | `verbs.smoke` | Everything the static checks cannot see | Discovering it at full scale |
| Smoke output isolation | `_isolate_smoke_outputs`, `_smoke_write_issues` | A smoke writing into real outputs | **Fails closed** if isolation cannot be *proven* |
| Confirm gate | `verbs.run` | Any unapproved full-scale run | Zero silent full-scale runs |
| Hash integrity | `confirm == rec.config_hash` | A recipe edited after approval | Approving one thing, running another |
| Optional smoke evidence | `_safety.verify_smoke_token` | Running without ever smoking | Stricter deployments |

### 9.2 During execution

- **Streaming → batch auto-fallback** when the runtime reports streaming does not fit.
- **Per-stage metrics** (timings, throughput, GPU seconds, resource peaks).
- **Fan-out-aware counting** — aggregate one `StagePerfStats` per `(stage, source)`, and keep
  `source_items` distinct from `output_rows`.
- **Optional checkpointing** for partial-run recovery when the stages are resumability-safe.
- **Ray bootstrap** on a free port, with the plasma store on a writable dir (avoiding the
  `/dev/shm` permission trap) and the state-API cap set.

### 9.3 After execution

- **Acceptance verification** against the frozen contract, using an exhaustive terminal-manifest
  read-back (up to 2,000 evidence rows) when the contract needs it.
- **Honesty guard** — `must_dropped` / `must_downgraded` / `must_relaxed`. Non-empty **forces
  `overall: not_met`**. Deliberately conservative: a changed type, scope, method, target, or
  failure policy is not treated as comparable even if the number looks stricter. Only a
  same-operator numeric strengthening is auto-accepted.
- **Atomic publish** — an artifact counts only once its `_COMPLETE` marker exists. This also
  closes a crash bug: `ManifestWriterStage` appends, so a crashed run left a partial-but-valid
  JSONL, and re-running into the same path silently duplicated rows.
- **Secret + transcript redaction** on every return value.

### 9.4 Three real incidents encoded as structure

**(a) The deleted file.** An agent read `ManifestWriterStage.process`'s append-mode open,
concluded reruns would accumulate rows, and **deleted the user's output before the confirm gate**.
(They do not accumulate — `setup` truncates first and the stage is pinned to one worker.)
*Fix:* `output_targets` so the agent can *see* what is there, plus the explicit rule that nothing
is written before approval, plus the reasoning recorded in `SKILL.md` so nobody re-derives the
wrong conclusion.

**(b) The `null` confirm.** The gate tested `confirm is False`, so any other falsy value —
including a JSON-RPC `confirm: null` forwarded by the MCP adapter — slipped past **both** the
refusal and the hash check into a silent full-scale run.
*Fix:* only a literal `True` or a matching hash string counts.

**(c) The silently-empty contract.** `acceptance_criteria: {must: [...]}` became `["must"]`
because `list()` over a mapping yields keys. `run` then skipped verification **while reporting
success**.
*Fix:* shape-checked at the door in `recipe._criteria`, in `acceptance.parse_criteria`, and in
`cli._criteria_list` — all failing loud with the expected shape.

### 9.5 What is deliberately *not* a deterministic guardrail

Semantic misuse refusal ("isolate this named person's voice") needs judgment a deterministic tool
cannot make. It stays a skill/policy concern in `SKILL.md`.

---

### Presentation questions — Safety

**Q: What happens when a pipeline fails?**
The error is classified (18 signatures), a fresh environment probe runs, the affected stages are
identified, and ranked recovery options come back — each labelled by kind and availability. The
agent explains and **asks**. It never installs, upgrades, switches to CPU, changes a model, or
retries silently.

**Q: Could the agent run something huge without asking?**
No. `run` refuses unless handed a literal `True` or the recipe's exact `config_hash`. The gate is
inside the tool, so the CLI, the MCP adapter, and a scripted caller all hit it.

**Q: How do you know these gates work?**
`eval/audio/run_eval.py` asserts them GPU-free in CI, including negative cases (unknown stage
rejected, unproducible role, subjective-trait labelling mapping to no stage). Guardrail breaches
are P0 in the evaluation taxonomy regardless of category.

---

## 10. Debugging Guide

### 10.1 First moves, always

```bash
.venv/bin/python -m nemo_curator.audio_agent doctor --json
```
```bash
.venv/bin/python -m nemo_curator.audio_agent validate --recipe R.yaml --data /path/to/data
```
```bash
.venv/bin/python -m nemo_curator.audio_agent diagnose --error '<paste the error>' --recipe R.yaml
```

> ⚠️ Always use `.venv/bin/python`. The base interpreter lacks Curator's dependencies, and — more
> subtly — launching via `uv run` **without** the audio extra makes Ray rebuild the worker
> environment from the base dependency set, giving you a working driver and broken workers.

### 10.2 Failure catalog

**`unknown_stage` / `stage_import_error`**
*Symptom:* validation fails immediately. *Cause:* wrong name, or a missing optional extra.
*Inspect:* `discover`, then `pyproject.toml` extras. *Verify:* `python -c "from nemo_curator.stages.audio._catalog import get_agent_ready_stage_class as g; g('YourStage')"`.
*Fix:* correct the name or `uv sync --extra audio_cuda12`.

**`unsatisfied_reads`**
*Symptom:* "requires role X not produced upstream; available so far: […]".
*Cause:* missing producer, or wrong order. *Inspect:* `context --roles X` for the role graph.
*Verify:* the `available so far` list tells you exactly what the walk had at that point.
*Fix:* insert the producer via **targeted re-retrieval** — the verdict names the role, so query
only that. A gap here means your candidate-card set was incomplete, not that you are stuck.

**`dangling_key` (warning — but it means zero rows)**
*Symptom:* validates green, run produces nothing. *Cause:* the producer's `*_key` value was
renamed away from what the consumer reads. *Inspect:* `verdict.produced_keys` vs the consumer's
configured key. *Fix:* align them.

**`ambiguous_default_key`**
*Symptom:* plausible but wrong results, no error. *Cause:* two upstream stages wrote same-kind
keys and a consumer is at its default. *Fix:* set the `*_key` explicitly. The documented case:
a diarization merge at `segments_key="segments"` merges into the **VAD** segments.

**`tensor_into_sink`**
*Symptom:* crash at the final writer. *Cause:* a resident waveform reaching a JSON serializer.
*Fix:* strip it in a way that **preserves the sink's input task type** — read from file
(`input_residency=file`), or `keep_waveform_in_task=false`. Note `AudioToDocumentStage` emits a
`DocumentBatch`, so it fits `DocumentBatchJsonlWriterStage`, **not** `ManifestWriterStage`.

**`task_type_mismatch`**
*Symptom:* `AudioToDocumentStage → ManifestWriterStage` fails. *Fix:* use
`DocumentBatchJsonlWriterStage`, or move every `AudioTask`-only stage before the converter.

**`card_sample_rate`**
*Symptom:* warning about rates. *Verify:* is a resample really upstream, and does it set
`update_audio_filepath`? The tracker only recognizes `target_sample_rate` as a conversion —
`MonoConversionStage.output_sample_rate` *verifies* a rate (dropping mismatches), it does not
convert.

**`gpu_reservation_missing`**
*Symptom:* runs but 50× too slow; many actors loading models. *Fix:* `resources: {gpus: 1}`,
sized from the card's `gpu_mem_gb`.

**`worker_env_mismatch` — `ModuleNotFoundError` in a Ray worker only**
*Symptom:* driver imports fine, `Node setup failed for stage …` in a worker.
*Cause:* **a launch-flag problem, not a broken install.**
*Fix:* `.venv/bin/python -m nemo_curator.audio_agent …` or `uv run --extra audio_cuda12 python -m …`.

**`asr_decoder_cuda_graph` / CUDA error 222 / unsupported PTX**
*Cause:* the GPU driver is older than the CUDA toolkit torch was built with; the RNNT/TDT decoder
JIT-compiles PTX at runtime. *Inspect:* `doctor` → `cuda_driver_toolkit`.
*Fix (any one):* upgrade the driver; install a matching `+cuXXX` torch; or set
`decoder_type='ctc'` on `NeMoASRAlignerStage`/`SplitASRAlignJoinStage`. **Note:** plain
`InferenceAsrNemoStage` with a pure-TDT checkpoint has no CTC head — there the environment fix is
the only option.

**"No GPU" on a machine that has one**
*Cause:* sandbox/container masking `/dev/nvidia*`. *Inspect:* `gpu_possibly_masked`,
`nvidia_device_nodes`, `torch_cuda_built`. *Fix:* re-run `doctor` and any smoke/run with full
device access. **Do not report "no GPU" until a full-access probe still finds none.**

**`zero_rows_retained` / `empty_vad_output`**
*Cause:* filters too aggressive, or a `dangling_key` silently breaking the flow.
*Inspect:* smoke `per_filter_counts`; check `keys_ok`. *Fix:* run the filter with
`action='annotate'` first to audit the score distribution on **your** data, then set a threshold.

**Reuse not firing when you expect it**
*Inspect:* `reuse-scan` → `steps[]` gives per-step `found` / `reusable` / `blocked_by`.
*Common causes:* `dataset_key` changed (a file edited/added); no `_COMPLETE` marker (crashed run);
content digest changed (output edited after publication); `code_version` changed; the prefix is
all in-memory so nothing persisted (look for `prior_unsaved`).

**Reuse firing when you do NOT expect it**
*Cause:* you changed only an execution knob or an output path — by design those do not change
`semantic_hash`. *Verify:* compare `semantic_hash` between the two recipes.

**`overall: not_met` with an empty-looking reason**
*Inspect:* `acceptance.honesty[]`. A dropped/downgraded/relaxed `must` **forces** `not_met`.
*Fix:* do not edit the contract — re-confirm a new one with the user.

**`overall: unverifiable`**
*Cause:* the contract was empty, or no evidence exists for a criterion (e.g. WER with no
references). This is honest, not a bug.

**`check_error` in the verdict**
*Cause:* a validation check raised, usually on a malformed card. *Inspect:* the message names the
check. *Verify:* `python -m nemo_curator.audio_agent.card_conformance`.

### 10.3 Where the logs are

| What | Where |
| --- | --- |
| Run records (source of truth) | `.audio_agent_runs/<run_id>.json` |
| Artifacts | `.audio_agent_runs/artifacts/<step_key>.json` |
| Lookup cache (rebuildable) | `.audio_agent_runs/index.db` — `reindex` restores it |
| Everything already done to a corpus | `runs --data /path/to/data` |
| Pipeline/Ray logs | `report.logs_pointer`; Ray's own session directory |
| Eval reports | `eval/audio/reports/latest.json` |

---

## 11. Things to Call Out (unused, duplicated, or over-complex)

Stated plainly, because you asked me not to assume every file is necessary.

| Item | Assessment |
| --- | --- |
| **Two reuse engines** — `continuation.plan_continuation` (parent-diff) alongside `reuse.scan` (content-addressed), merged by `verbs._merge_plans` | **Real redundancy with a defensible reason.** The parent-diff engine still serves `--parent-run-id`. But this is the highest-complexity area and the strongest simplification candidate. If the newer engine covers the parent case, retiring the older one would remove a whole class of "which engine decided this?" confusion. |
| **`PlanResult` in `contracts.py`** | **Defined, exported, never constructed anywhere.** Dead type from an earlier design where the core produced the plan. Safe to delete; keeping it costs a reader's attention. |
| **`patterns/composition.yaml`** | **Deliberately not parsed.** Its `enforced` rules are also implemented in `checks.py` and kept in sync **by hand**. Editing the YAML changes nothing at runtime. The header says so, but it is a genuine drift risk — worth a test asserting each `enforced` pattern has a corresponding check. |
| **`manifest_reader.yaml` + `manifest_reader_stage.yaml`** | **Not duplication.** `ManifestReader` is the composite; `ManifestReaderStage` is its inner reader. Both are separately selectable, so both need cards. |
| **`verbs.py` at 4,099 lines** | Justified as one gate implementation, but `run` and `smoke` are each several hundred lines with many early returns. The honest refactor is extracting the shared preamble (safety → binding → profile → build → preflight). |
| **`_load_dir` swallows YAML errors** | A malformed card **silently vanishes** rather than failing discovery. The right trade-off for robustness, but it means a typo produces "this stage has no card facts" with no error. Run `card_conformance` — that is what it is for. |
| **`config_strategy` Path B (`data_driven`)** | **Deliberately unimplemented.** A relative goal returns an `ask` rather than a guess. Not dead code — an explicit, documented boundary. |
| **`eval/audio/simulate_user.py` (576 lines)** | Persona-driven end-to-end simulation, opt-in behind `--sim`, needs an API key and a GPU. Legitimate but rarely exercised; treat as experimental. |
| **Root-level `curate_demo_recipe.yaml`, `single_speaker_16k_asr*.yaml`** | Untracked working files from manual sessions. Useful as real examples; not part of the package. Move to `tutorials/` or delete. |
| **Composite depth/leaf caps (8 / 512) in `semantic_review`** | Magic numbers with no named constant explanation. Fine in practice; worth a comment saying why those values. |

**Not a problem, despite appearances:** every script in `eval/audio/` is wired into `run_all.sh`.
There is no dead code there.

---

## 12. Learning Checklist

### Must understand (you cannot present without these)

- [ ] **The two-plane split.** LLM proposes, deterministic core disposes. Be able to say which side
      owns any given decision.
- [ ] **Why a Recipe, not code.** Names resolve through a registry; there is zero codegen in the
      execution path. This is *the* anti-hallucination mechanism.
- [ ] **Capability cards.** What is in them, why honesty tiers exist, why `card_conformance` matters.
- [ ] **Tiered retrieval (L0→L1→L2).** Cheaper *and* more accurate than dumping everything.
- [ ] **Mechanical validation vs. semantic review.** Green ≠ correct. Be able to tell the
      single-speaker trap from memory.
- [ ] **The confirm gate + `config_hash`.** What was approved is what runs.
- [ ] **The four honest acceptance states**, and the honesty guard that forces `not_met`.
- [ ] **Reuse in one sentence:** content-addressed step keys, never silent, never nagging,
      low trust defaults to fresh.
- [ ] **The end-to-end flow** (§4) well enough to narrate it without notes.

### Good to understand (expect these questions)

- [ ] **Roles vs. keys** — why composition matches roles, and what `dangling_key` means.
- [ ] **`ok` vs `keys_ok` vs `runnable`** — and why you gate on `runnable`.
- [ ] **The three hashes** and why splitting them was necessary.
- [ ] **Why VRAM warns but GPU reservations gate** — never let a guess block the measurement that
      would replace it.
- [ ] **`gpu_status`** and why "no GPU" from a sandbox is not a hardware fact.
- [ ] **The `worker_env` check** — a launch-flag problem that looks like a broken install.
- [ ] **`resolve`** — outcome labels to parameters via card anchors, with direction from the metric.
- [ ] **Smoke output isolation** and calibration's raise-only rule.
- [ ] **The two-plane eval harness** and the 23-class failure taxonomy.
- [ ] **How to add a stage** (three declarations + a card; a *source* stage also needs an
      `input_identity` adapter).

### Deep technical details (for follow-up questions)

- [ ] The **check registry** — all nine checks and what each catches.
- [ ] **Effective sample-rate tracking** in `card_constraints`, and why `MonoConversionStage`'s
      `output_sample_rate` is not a conversion.
- [ ] **Composite flattening** in `planner._execution_needs` and `semantic_review`.
- [ ] **`invalid_reasons` vs `caution_reasons`** — and why moving non-determinism between them
      mattered.
- [ ] **`cumulative_sec`** vs `duration_sec` and the 30-second auto-take threshold.
- [ ] **`_logical_identity`** — why a continued run publishes under the *requested* recipe's keys.
- [ ] **`_weakened_reason`** — why a changed scope/method/target is not safely comparable.
- [ ] **The disk-boundary guard** in `continuation`.
- [ ] **Layered save** — which annotations are excluded from the hash and why.
- [ ] **The tiered dataset key** and what `stat` tier still cannot catch (a restored size+mtime).
- [ ] The three incidents in §9.4, and what each became structurally.

---

### Final presentation questions (mixed)

**Q: How is this different from a chatbot that generates YAML?**
A chatbot generates YAML and hopes. This system: (1) can only name stages that exist; (2) proves
the YAML composes before running it; (3) requires a grounded semantic critique that green
validation is not enough; (4) proves it on 10 items first; (5) refuses to run at scale without a
hash-bound approval; (6) verifies the result against a contract frozen before the run; (7) blocks
itself from quietly lowering that bar; and (8) skips work it can prove was already done. A YAML
generator has none of those. Every one exists because its absence produced a real failure.

**Q: How does it know which Curator module to use?**
Coarse-to-fine over the knowledge index — prune categories, read one-liners, then read full cards
for the finalists. When two overlap, it compares them on card facts (`comparison` block: language
support, accuracy, latency, config complexity, known limitations, "choose when") and applies a
decision policy: **auto** if one clearly fits, **recommend** if there is a trade-off, **ask** only
if the choice is material *and* preference-dependent *and* not inferable.

**Q: What is the maintenance burden?**
The 47 capability cards are the ongoing cost — they are hand-authored and must stay accurate. The
conformance gate keeps their *mechanical* facts honest automatically; the judgment facts
(`use_cases`, `semantic_facts`) need human review. Everything else is generic: adding a stage
needs no core change.

**Q: If you rebuilt this, what would you do differently?**
Split the three hashes from day one, and design reuse around per-step content addressing rather
than whole-run comparison — those two mistakes cost a full rewrite of the reuse subsystem. I would
also consolidate to one reuse engine rather than merging two.
