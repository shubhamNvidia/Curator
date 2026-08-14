# The Audio Curation Agent — Knowledge Document

**Source of truth for the agentification presentation.**
Audience: engineers and stakeholders with **no prior knowledge of NeMo Curator**.
Scope: what we built, why each piece exists, what we learned, and what changed for users.

---

## Table of contents

1. [The problem we started from](#1-the-problem-we-started-from)
2. [Why a simple "LLM wrapper" is not enough](#2-why-a-simple-llm-wrapper-is-not-enough)
3. [The core design idea: two planes](#3-the-core-design-idea-two-planes)
4. [Overall architecture](#4-overall-architecture)
5. [How a user request moves through the system](#5-how-a-user-request-moves-through-the-system)
6. [The agentification layers, one by one](#6-the-agentification-layers-one-by-one)
7. [Deterministic core vs. fully LLM-driven design](#7-deterministic-core-vs-fully-llm-driven-design)
8. [Key design decisions](#8-key-design-decisions)
9. [Important learnings](#9-important-learnings)
10. [Impact of the agentification work](#10-impact-of-the-agentification-work)
11. [Limits, non-goals, and future possibilities](#11-limits-non-goals-and-future-possibilities)
12. [Presentation storyline](#12-presentation-storyline)
13. [Appendix: quick reference](#13-appendix-quick-reference)

---

## 1. The problem we started from

### 1.1 What NeMo Curator is (the 60-second version)

NeMo Curator is a data-curation framework. For audio, it ships a library of **processing
stages** — small, independent units of work such as "resample audio to 16 kHz", "detect
speech regions", "score perceptual quality", "transcribe", "identify who spoke when",
"write a manifest file".

A user builds a **pipeline** by putting stages in an order and configuring each one. The
pipeline runs on a distributed execution backend (Ray/Xenna), usually on GPUs, over
thousands of audio files.

Today the catalog has **49 agent-ready audio stages** across **10 functional categories**
(ingest, preprocess, segment, diarize, transcribe, text normalization, quality, filter,
audio-language-model data building, export).

### 1.2 What "audio curation" actually means

Raw audio is not training data. Turning one into the other means answering questions like:

- Which clips are clean enough to train on?
- Which have exactly one speaker?
- Which are in the right language, the right sample rate, the right channel count?
- Where is the speech, and where is silence?
- What was said, and how confident are we?
- What do I keep, what do I drop, and how much did I lose?

Each of those is a stage or a combination of stages.

### 1.3 Why doing this by hand is hard for a new user

The difficulty is **not** the individual stages. It is everything around them:

| Challenge | What it looks like in practice |
| --- | --- |
| **Discovery** | 49 stages across 10 categories. Which ones apply to "give me clean single-speaker data"? Three different quality scorers exist (UTMOS, SIGMOS, SQUIM) and two different diarizers. Nothing tells a newcomer which to pick. |
| **Compatibility** | Stages communicate through named fields in a shared record. If stage B reads `pred_text` and nothing upstream writes it, the pipeline crashes — after you have already paid for the GPU time. |
| **Ordering** | Diarization needs continuous audio, so it must not come after a segmenter that chopped the file up. Quality filters are cheaper before expensive model stages. None of this is discoverable from the API. |
| **Configuration** | "Clean" is not a number. Somewhere it becomes `mos_threshold: 3.4`. A newcomer has no idea whether 3.4 is strict or lenient, or whether higher is better. |
| **Direction traps** | Quality scores are "higher is better". Error rates (WER) are "lower is better". Filtering the wrong side silently keeps exactly the data you wanted to drop. |
| **Meaning traps** | A field named `num_speakers` sounds per-clip, but after a fan-out stage the row is a *segment*, not a clip. Filtering it then answers a different question than the user asked. |
| **Environment** | GPU drivers vs. CUDA toolkits, ffmpeg, optional dependency extras, Ray worker environments that differ from the driver's. Failures here look like code bugs but are setup problems. |
| **Cost** | A full run over a real corpus is hours of GPU time. Discovering a mistake at the end is expensive. |
| **Repetition** | "Now also add transcripts" often means re-running everything from scratch, including the hours already spent. |

**In short:** the stages were usable, but only by someone who already understood the
whole system. The gap was not capability — it was **access to capability**.

---

## 2. Why a simple "LLM wrapper" is not enough

The obvious approach is "give an LLM the docs and let it write the pipeline." We
deliberately did not do that. Here is why, in plain terms.

| Failure mode of a naive LLM wrapper | Consequence |
| --- | --- |
| **Invents stages and parameters** | A confidently generated `NoiseReductionStage(strength=0.8)` that does not exist. Crashes at build time — or worse, a real stage with a made-up parameter. |
| **Writes ad-hoc Python** | Unreviewable, unrepeatable, unbounded. Any generated code can delete files, leak data, or run for days. |
| **Cannot check compatibility** | Produces a plausible-looking pipeline whose stage 4 reads a field nobody writes. |
| **Guesses thresholds** | "I'll use 0.7" — with no idea of the metric's scale or direction. |
| **Cannot see the machine** | Plans a 4-GPU pipeline on a 1-GPU box; blames the code when it fails. |
| **Not repeatable** | The same request produces a different pipeline each time. Nothing can be audited or approved. |
| **Token-expensive** | Dumping 49 stage sources and 49 cards into context for every request is slow, costly, and *less* accurate (the model drowns in irrelevant detail). |
| **Declares success without evidence** | "Done! Your data is now high quality." Based on nothing. |
| **Silently moves the goalposts** | Asked for 4.0, could only reach 3.2, quietly reports success at 3.2. |
| **Runs first, thinks later** | Launches a 6-hour GPU job to discover the recipe was wrong. |

The last two matter most. An agent that is *usually* right but *occasionally* confident
and wrong is worse than no agent, because a human stops checking it.

**Conclusion:** the LLM is genuinely good at interpretation, selection, trade-off
explanation, and judgment. It is genuinely bad at being an oracle for facts, a validator,
or an executor. So we built a system where it only does the first set of things.

---

## 3. The core design idea: two planes

Everything in the architecture follows from one sentence:

> **The LLM proposes. The deterministic core disposes.**

| Plane | Who | What it owns |
| --- | --- | --- |
| **LLM plane** (the "host") | Claude / Cursor / Codex, driven by a written skill | Understanding intent, asking clarifying questions, choosing between overlapping modules, explaining trade-offs, judging whether a mechanically valid recipe *means* what the user asked, presenting choices |
| **Deterministic core** | `nemo_curator.audio_agent` — a plain Python package, **no embedded LLM** | Knowing what exists, what composes, what the machine can do, what already ran, what actually happened, and refusing anything unproven |

Two structural properties make this safe:

1. **The LLM never emits code.** It emits a **Recipe** — a small declarative document
   listing stage names and parameters. There is zero `exec`/`eval`/`compile`/generated-`.py`
   anywhere in the execution path (verified across the repo). Stage names are resolved
   through a registry; an invented name simply does not resolve.
2. **The LLM cannot skip a gate.** The gates live inside the tool functions, not in the
   prompt. A weaker model, a direct CLI call, or a scripted caller hits the same refusals.

This is what we mean by **"grounding"**: every LLM decision is either selected from a real,
indexed list, or checked against a deterministic verdict before it can have consequences.

---

## 4. Overall architecture

```
                            ┌──────────────────────────────────────────┐
   User's words  ─────────▶ │  LLM PLANE (host model + SKILL.md)       │
   "clean single-speaker    │  interpret · route · select · configure  │
    16 kHz data with        │  critique · explain · ask · present      │
    transcripts"            └───────────────┬──────────────────────────┘
                                            │  JSON tool calls
                                            ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  DETERMINISTIC CORE   (nemo_curator.audio_agent — no LLM inside)            │
   │                                                                            │
   │  KNOWLEDGE          discover · catalog-tree · cards · describe · context    │
   │  SITUATION          data profiler · doctor (environment health)             │
   │  CONFIGURATION      resolve  (outcome label → concrete parameter)           │
   │  VALIDATION         validate (composition, contracts, gates, criteria)      │
   │                     + semantic_review evidence packet (for the LLM critic)  │
   │  PLANNING           resource planner (streaming vs batch, feasibility)      │
   │  EVIDENCE           smoke  (bounded run) · calibrate (measured resources)   │
   │  SAFETY             confirm gate · workspace lock · redaction               │
   │  EXECUTION          run (confirm-gated) · Ray bootstrap · auto-fallback     │
   │  RECOVERY           diagnose · failure taxonomy · environment decisions     │
   │  RESULTS            report · verify (acceptance + honesty guard)            │
   │  MEMOIZATION        reuse-scan · continue · runs · reindex                  │
   └────────────────────────────────┬───────────────────────────────────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  NeMo Curator                                                              │
   │  49 agent-ready audio stages  ·  Pipeline builder  ·  Ray/Xenna executor    │
   └────────────────────────────────────────────────────────────────────────────┘

   Persistent, versioned, hand-authored knowledge (read-only YAML):
   49 capability cards · category taxonomy · 3 blueprints · 3 library recipes
   · composition patterns · failure taxonomy
```

The same core is exposed three ways — a **CLI**, an **MCP server**, and a **Python SDK** —
all returning JSON. One implementation, three doors.

---

## 5. How a user request moves through the system

A single worked example: *"I have a folder of recordings. Give me clean, single-speaker,
16 kHz audio with transcripts."*

| # | Step | Plane | What happens |
| --- | --- | --- | --- |
| 1 | **Interpret** | LLM | Turns words into a goal (task, domain), a capability plan (which categories are likely needed), and a **success contract** — the acceptance criteria that define "done". Asks at most 1–2 plain-language questions (e.g. "studio, broadcast, or general quality?"). Refuses out-of-scope requests. |
| 2 | **Inspect** | Core | `context` profiles the actual data (file count, sample rates, channels, transcripts present?) and the machine (GPU? ffmpeg? which extras?), and returns the category tree plus any matching blueprint. Often this alone is news to the user. |
| 3 | **Health check** | Core | `doctor` reports environment status with per-issue fixes. Run before any heavy GPU work. |
| 4 | **Route** | LLM | Coarse-to-fine: prune categories → read one-line summaries for the survivors → read full cards only for the finalists. Compares overlapping modules on card facts, not on vibes. |
| 5 | **Resolve config** | Core | `resolve --label general` turns the *outcome word* into a concrete parameter (`mos_threshold: 3.4`) using the card's anchors, plus an audit trail of why. |
| 6 | **Plan** | LLM | Emits a **Recipe**: an ordered list of `{stage, parameters}` plus the embedded success contract. |
| 7 | **Validate** | Core | Checks that every stage is real, every parameter is accepted, every field a stage reads is produced upstream, card constraints hold, environment gates pass, and the success contract is producible at all. Returns a verdict with fixes. Loop until runnable (≤3 iterations). |
| 8 | **Semantic critique** | LLM | Mandatory. Using a deterministic evidence packet (field lineage, meaning, granularity, fan-out seams), the model must answer: does this pipeline *mean* what the user asked? Returns `pass`, `revise`, or `ask`. |
| 9 | **Reuse scan** | Core | Before spending anything: has this exact computation already been done on this exact data? Offers `already_done` / `incremental` / `fresh`. |
| 10 | **Smoke** | Core | Runs the real pipeline on ~10 items into a throwaway directory. Returns retained/rejected counts, examples, errors, and measured resource usage. |
| 11 | **Confirm gate** | Both | The LLM presents the plan, the critique, the smoke evidence, the scale/cost estimate, and the success contract — **stating what each metric does and does not capture**. Nothing has been written yet. The user says yes. |
| 12 | **Run** | Core | `run` refuses unless handed the recipe's `config_hash` — so what was approved is byte-for-byte what runs. Resource planner picks streaming or batch. Ray starts if asked. Streaming auto-falls back to batch under pressure. |
| 13 | **Verify** | Core | Evaluates the frozen success contract against real output evidence. Four honest states: met / not met / unverifiable / unachievable. An **honesty guard** blocks any silent weakening of a confirmed requirement. |
| 14 | **Report** | Both | Counts, per-filter drop reasons, failure examples, output paths, elapsed time. The LLM explains it in plain language and resolves any judgment-based criteria. |
| 15 | **Remember** | Core | Every persisted step is published as a content-addressed artifact, so the next request can skip it. |

Any failure at any point routes to `diagnose`, which classifies the error and returns
grounded recovery options — **which the agent may explain but never apply on its own**.

---

## 6. The agentification layers, one by one

Each layer below follows the same shape: **the Curator reality that created the need →
what the layer does → what would go wrong without it → how it connects to the rest.**

---

### Layer 0 — Stage agent-readiness (the foundation)

**Curator context.** A Curator stage is a Python class. It reads and writes named fields in
a shared record. Which fields, and whether a stage produces one row per input or many,
was knowledge that lived in the source code and in people's heads.

**What we added.** A small, mandatory declaration on each stage — its **contract**:

- **reads / writes** — exactly which named fields it consumes and produces
- **cardinality** — `1:1`, `1:N fan-out`, `N:1`, `filter`, `1:1 nested-list`
- **gates** — honest side-effect flags: needs GPU, writes to disk, needs ffmpeg, needs
  internet on first run, requires serializable input, sanitizes output
- **conditional writes** — fields written only on a runtime branch, and where their value
  came from (computed here vs. copied from upstream)

Everything else is auto-derived: parameter names/types/defaults from the dataclass,
parameter descriptions from the docstring, field *roles* from the `*_key` naming
convention. Stage owners declare three things and get a test that tells them if anything
is missing.

**Without it.** Nothing above this layer can exist. Compatibility checking, semantic
review, and reuse identity all read this contract.

**Design principle that made adoption possible:** every new knob defaults to today's
behavior. Agent-readiness never changes how a stage runs in an existing pipeline.

---

### Layer 1 — Capability cards (the knowledge layer)

**Curator context.** A contract says a stage is *mechanically connectable*. It does not say
whether a stage is *appropriate*. Nothing in the code tells you that UTMOS is a single
overall quality score while SIGMOS breaks quality into seven dimensions, that a diarizer's
speaker labels are per-recording cluster IDs rather than real identities, or that a score
of 4.0 means "studio grade".

**What we added.** One YAML **capability card** per stage (**49 today**), holding the facts
source code cannot express:

- **summary, category, tags** — what it is, where it belongs, what it needs
- **model identity and pinned version** — so a silent model swap is detectable
- **constraints** — supported sample rates, max speakers, fixed batch sizes
- **resource facts** — CPU, VRAM, host RAM, whether GPU is optional, cost profile
- **use cases** — good for / avoid for
- **composition** — typical upstream and downstream neighbours
- **metrics** — the scale, the **direction** (higher-better vs lower-better), the threshold
  parameter, valid range, **outcome anchors** ("4.0–5.0 = studio"), and presets
- **comparison** — the disambiguation block used when two stages overlap: language
  support, accuracy, latency, config complexity, known limitations, "choose when"
- **semantic facts** — for every externally consumed output: its **meaning**, **unit**,
  **provenance**, **scope/granularity**, **propagation** across fan-out and aggregation,
  and at least one **counterexample** (a tempting wrong interpretation)
- **honesty tiers** — every fact group declares how it was established:
  `mechanical` (derived from code, re-checked automatically), `measured` (from a real run
  on known hardware), or `best_guess` (author judgment)

**Golden rule of the card schema: never fabricate.** An unknown value is left empty with a
`TODO(fill)` comment. A wrong fact is worse than a missing one, because the agent trusts
cards.

**A conformance gate keeps cards honest.** A card claiming a parameter the stage does not
have, an unknown resource key, a model without a pinned version, a metric with an invalid
direction, or a capability tag that contradicts the stage's real default behaviour — all
fail an automated check. Cards cannot drift away from the code.

**Without it.** The LLM would choose between overlapping modules by name similarity,
invert filters, expose raw thresholds to users, and misread field meanings after a fan-out.

**Connects to.** Routing (Layer 2), configuration resolution (Layer 5), validation
constraints (Layer 7), semantic review (Layer 8), resource planning (Layer 9).

---

### Layer 2 — Knowledge index and coarse-to-fine routing

**Curator context.** 49 stages, each with a rich card. Handing all of that to a model for
every request is expensive and, counterintuitively, *less* accurate.

**What we added.** A retrieval backbone that serves knowledge in **three tiers**:

- **L0 — the category tree.** Ten functional groups with one-line descriptions. The model
  prunes here first: no filtering intent → skip `quality`/`filter`; no transcripts in the
  data → transcription-dependent metrics are unproducible.
- **L1 — one-line summaries** for the surviving categories → shortlist.
- **L2 — full cards** for the two or three finalists only.

Plus three supporting structures:

- **Blueprints** (3) — idiomatic end-to-end shapes for known scenarios, with each stage
  tagged `enforced` (structurally required) or `advisory` (a judgment call), plus explicit
  pitfalls and a topology-selection rule. A blueprint is **a menu, not a mandate**.
- **Library recipes** (3) — fully working reference pipelines.
- **Composition patterns** — abstracted ordering wisdom ("mono first", "cheap CPU filters
  before expensive GPU filters", "re-join segments before diarization", "put the filter
  after the metric that produces its field"), each labelled `enforced` (also checked in
  code) or `advisory`.
- **Role graph** — which stages produce a given semantic role, used for targeted
  re-retrieval when validation reports a missing producer.

**Without it.** Either enormous token cost with degraded accuracy, or arbitrary stage
selection.

**Important property:** all of this is **static, versioned, hand-authored YAML**. The index
serves material; it never makes the final relevance choice, and it never dumps source code.

---

### Layer 3 — Intent understanding and the success contract

**Curator context.** Curator has no notion of "what the user wanted". It has stages and a
runner. Success has historically meant "the process exited zero".

**What we added.** Two things, both owned by the LLM plane and both written down.

**(a) A capability plan.** The request becomes a small structured goal: task, domain,
expected outputs, likely capability areas, constraints, and open questions. This doubles as
a **coverage checklist** for later steps — a way to notice at the end that something the
user asked for was never planned.

**(b) A success contract (acceptance criteria).** Before anything runs, "done" is defined:

- **output completeness** — which outputs must exist (e.g. every retained clip has a transcript)
- **quality standard** — an absolute bar ("studio") or a relative one ("best 20%")
- **yield** — how much data should survive

Each criterion is classified `must` or lower, and marked absolute or relative.

**Two rules make this more than paperwork:**

1. The contract **lives inside the recipe**, so it is covered by the approval hash. A
   separate criteria file alone is not executable intent.
2. **Request-type sanity** is checked deterministically: a *filtering* request with no
   yield criterion is flagged; a *transcription* request that never declares its transcript
   output is flagged. You cannot declare success while ignoring the point of the request.

**Question discipline.** The agent asks only at the **outcome layer** — quality *level*,
language, output format — never about internal parameters (thresholds, field names, batch
sizes, model IDs). It asks only when a decision is *material* **and** *preference-dependent*
**and** *not inferable*; otherwise it picks a safe default and says so. Asking which of two
modules to use is fine, framed as a plain capability trade-off.

**Refusal boundary.** Human labelling of subjective traits (speaker gender, accent,
emotion), and model training/evaluation/deployment, are out of scope. The agent stops and
offers a safe alternative rather than improvising.

**Without it.** "Success" becomes "it ran", every clarification turns into an interrogation
about internals, and there is nothing to verify against at the end.

---

### Layer 4 — Situational awareness: data profile and environment health

**Curator context.** Curator assumes you know your data and that your machine is set up.
Both assumptions fail constantly. A GPU-driver-vs-CUDA-toolkit mismatch, a missing ffmpeg,
or an optional dependency extra that the Ray *workers* lack while the *driver* has it —
these surface as cryptic runtime errors deep inside a model load.

**What we added — two probes.**

**(a) The data profiler.** Read-only, cheap, sampling-based: file count, sample rates,
channel counts, durations, whether transcripts are present, manifest field names. It also
computes a **dataset identity** used later for reuse (see Layer 15). It feeds the scale
estimate at the confirm gate and the pre-flight in validation.

**(b) `doctor` — environment health.** One place that checks the machine and says how to
fix what is wrong. A small **check registry**: each check reads the probed environment and
returns status + finding + impact + concrete fix steps. Adding a new environment concern is
one function.

Checks today: Python interpreter range · GPU presence and VRAM · **GPU driver vs. the CUDA
toolkit torch was built with** · ffmpeg · importable audio extras · **Ray worker
environment** · free disk.

Two checks are worth calling out because they encode hard-won knowledge:

- **Driver/toolkit mismatch.** Basic GPU operations work under minor-version compatibility,
  but anything that JIT-compiles at runtime (certain ASR decoders) fails with an obscure
  PTX error. `doctor` names this precisely and offers three grounded fixes — upgrade the
  driver, install a matching torch build, or switch to a decoder that does not JIT — and
  the recipe-aware version only offers the third when the recipe actually uses that path.
- **Worker environment.** Pipelines execute in Ray *workers*, not the driver. Launching via
  a package manager without carrying the dependency extra through makes Ray rebuild the
  worker environment from the base dependency set — producing a driver that imports
  everything happily next to workers that cannot. This reads like a broken install; it is a
  **launch-flag problem**, and the check says so.

**GPU masking.** Sandboxes and containers can block GPU device access, making a real GPU
report as absent. The agent detects the *signals* of masking (visible NVIDIA devices, a
CUDA-built torch) and refuses to state "no GPU" as a hardware fact — it says "not reachable
from this run" and asks for a re-check with full device access. Planning treats a masked GPU
as probably present and defers to the bounded smoke rather than blocking.

**Without it.** Hours lost to failures that look like code bugs but are setup problems, and
confident wrong statements about the hardware.

---

### Layer 5 — Outcome-to-parameter resolution (configuration strategy)

**Curator context.** Every filter stage has a numeric threshold. The number is meaningless
without knowing the metric's scale and direction.

**What we added.** A `resolve` verb that maps a **user-facing outcome** to a **concrete
parameter**, using only the card:

```
resolve --stage UTMOSFilterStage --label studio   →   { mos_threshold: 4.0 }
```

Three input modes, in priority order: an **explicit** value the user gave; a named **preset**
from the card; or an **outcome label** mapped through the card's metric anchors.

Two things make it generic rather than hard-coded:

- The metric, the parameter, the labels, and the direction all come from the card. There is
  **no metric name anywhere in the logic**. A new metric ships a card block and works
  unchanged.
- For a stage that only *annotates* (e.g. computes word error rate without filtering), the
  resolver emits a separate generic filter stage and **derives the comparison operator from
  the metric's direction** — error rates drop values *above* the bar, quality scores keep
  values *above* it. This is where inverted filters used to come from.

Every resolution produces an **audit trail** (what was chosen, from which source, why, and
when it must be recomputed) that travels with the recipe into the plan and the report.

**Relative goals are refused, not guessed.** "The best 20%" needs the data's distribution.
With the data-informed path not enabled, the resolver returns a **question**, not a number.

**"Resources is a knob too."** The same discipline applies to GPU reservations: a
GPU-required stage left at its CPU default silently runs on CPU (very slowly) and can
over-parallelise into many model-loading workers. The card's resource facts drive that
setting.

**Without it.** Hallucinated thresholds, inverted filters, and internal numbers leaking into
user conversations.

---

### Layer 6 — The Recipe IR (pipeline generation)

**Curator context.** Curator pipelines are normally built in Python.

**What we added.** The LLM emits exactly one artifact — a **Recipe**:

```yaml
recipe_id: clean-16k-quality
name: clean_16k
intent: "clean 16 kHz audio at general quality"
inputs:
  raw_data_dir:   "REQUIRED: folder of WAV files"
  resampled_dir:  "REQUIRED: where resampled audio is written"
  output_manifest: "REQUIRED: output .jsonl path"
stages:
  - ref: ResampleAudioStage
    params: {resampled_audio_dir: "REQUIRED_resampled_dir", target_sample_rate: 16000, target_nchannels: 1}
  - ref: UTMOSFilterStage
    params: {mos_threshold: 3.4}
  - ref: ManifestWriterStage
    params: {output_path: "REQUIRED_output_manifest"}
acceptance_criteria: [...]
```

Two conventions the example is showing on purpose. **Every required parameter must be present** —
`resampled_audio_dir` has no default, so omitting it fails at construction, not at runtime;
`describe --stage <name>` lists what a stage requires. And **a value that comes from `inputs` is
bound by the `REQUIRED_<input-name>` placeholder**, which is what connects the `inputs` block to
the stages that consume it.

Typed, hashable, serializable, and round-tripping to the pipeline configuration format
Curator already understands. It is deliberately **not a general DAG language** — audio
recipes are linear chains, and a DAG engine would be cost without benefit today.

**This is the anti-hallucination boundary.** Each `ref` is resolved through the stage
registry; an invented name does not resolve. Parameters are passed to a real constructor;
a wrong one produces an actionable error naming the accepted parameters. No code is
generated, ever.

**Three hashes, three jobs** — a deliberate separation that took a rewrite to get right:

| Hash | Covers | Answers |
| --- | --- | --- |
| `config_hash` | everything, including execution knobs, output paths, and the success contract | **"Is this exactly what the user approved?"** — the confirm gate |
| `semantic_hash` | stages + inputs, with execution knobs and output *locations* stripped | **"Would this produce the same bytes?"** — reuse identity |
| `contract_hash` | the success criteria alone | **"Is the success bar the same?"** — re-verify, don't recompute |

Why this matters: changing a batch size, a GPU reservation, or an output directory does not
change a single output byte — but it *does* change what the user approved. Using one hash
for both jobs meant either unsafe approval or reuse that never fired. Splitting them fixed
both.

**Layered save.** Machine-specific and data-specific annotations (the resource plan,
data-derived values, the configuration audit trail) ride *alongside* the recipe and are
excluded from the hash, each stamped with the machine or dataset it was computed for. The
recipe stays portable: the same intent on a different machine hashes identically and
recomputes the machine-specific parts rather than reusing stale numbers.

---

### Layer 7 — Deterministic validation (before anything runs)

**Curator context.** Curator will happily build a pipeline whose stage 4 reads a field
nobody produces. You find out at runtime, on the GPU, after paying for stages 1–3.

**What we added.** A `validate` verb backed by a **pluggable check registry** — adding a
check is one decorated function, with no change to the tool surface. Checks today:

| Check | Catches |
| --- | --- |
| **Well-formedness** | unknown stage names; unimportable stages (missing extras); wrong/missing parameters |
| **Data flow** | a stage reading a field nothing upstream writes (`unsatisfied_reads`); a producer whose field name does not match what the consumer reads (`dangling_key`); a field removed upstream; audio residency (in-memory waveform vs. on-disk file) |
| **Serialization** | a non-serializable tensor flowing into a JSON writer (`tensor_into_sink`) — with the exact fix |
| **Card constraints** | a fixed batch size violated; more speakers than the model supports; unsupported sample rates |
| **GPU reservation** | reservations that cannot be scheduled on this machine |
| **Environment gates** | missing ffmpeg, missing credentials, unavailable GPU — surfaced as setup steps |
| **Unproducible roles** | a required capability that **no stage in the whole catalog** can produce → the goal is impossible with these stages, and the user is told |
| **Output completeness** | "success needs transcripts, but no stage produces transcripts" — caught here, not after the run |
| **Request-type sanity** | a filtering request with no yield criterion |
| **Task type** | incompatible record types between stages |
| **Domain continuity** | e.g. diarization following a segmenter that destroyed the continuous waveform it needs |

Two additional outputs matter:

- **`output_targets`** — what already exists at every path the recipe will write to, with
  row and file counts. Facts only, no action. This exists because an agent once *deleted a
  user's file before the confirm gate*, having misread the writer's append-mode file open
  and concluded reruns would accumulate rows. (They do not — the writer truncates on setup.)
  Reporting an occupied path is the point; deciding what to do about it is the user's call;
  clearing it is nobody's.
- **`environment_decision`** — the machine facts filtered to *this* recipe's actual
  execution stages, so an irrelevant CUDA warning does not block a CPU-only pipeline.

**The retrieve↔plan loop.** A validation gap is not a dead end. When validation names a
*missing role*, the LLM does a **targeted re-retrieval** for exactly that role — the `producers`
verb answers "which stages write this role or key?" — adds the producer, and re-validates. The
first candidate set is a starting point, not a ceiling.

**Without it.** Every composition mistake becomes a runtime failure that costs GPU hours.

---

### Layer 8 — Semantic review (the LLM critic)

**This is the layer people find least obvious and it is arguably the most important.**

**Curator context.** Validation proves a pipeline *composes*. It cannot prove the pipeline
*means* what the user asked. Those are genuinely different questions, and the second one
does not reduce to rules.

**The canonical trap.** *"Keep only single-speaker clips."*

A tempting recipe: run speaker separation, then filter `num_speakers == 1`. It validates
perfectly green — the field exists, the filter is valid, everything composes.

It is also wrong, twice over:

- `num_speakers` is the count the diarizer found in the **original whole clip** — a
  parent-level aggregate. After separation, each row is one speaker's stream. The field is
  being applied at the wrong granularity.
- Speaker separation **transforms the audio** into per-speaker streams. The user asked to
  *select* clips, not to *mangle* them.

The correct recipe — diarize, then keep rows whose distinct-speaker count is 1, with no
separation — is not distinguishable from the wrong one by any mechanical check. The
plumbing is green either way.

**What we added.**

**(a) A deterministic evidence packet.** After every clean validation, `validate` returns a
`semantic_review` packet built from the *configured, instantiated* stages. It co-locates:
exact field lineage (which stage really produced the value this filter reads), cardinality
seams (fan-out, nesting, aggregation, filters), and the producer's and consumer's card
prose — meaning, unit, provenance, scope, propagation, limitations, model caveats. It also
includes a checklist of what must be reviewed and flags gaps (unresolved lineage, missing
card semantics, opaque visibility).

**The packet deliberately does not judge.** It is read-only evidence.

**(b) A mandatory LLM critique.** Before smoke, the host must answer five questions for
every field it filters and every stage it picked:

1. **Meaning and unit** — what does this field represent? *A plausible key name is not
   meaning.*
2. **Scope / entity / granularity** — whose value is this **at this point in the pipeline**:
   the original file, a fan-out child, or an aggregate?
3. **Provenance** — is it measured, a configured *target*, or a relative label? (A resample
   target verifies a value; it does not describe the input. A diarizer's `speaker_0` is a
   per-recording cluster ID, not a person.)
4. **Stage effect vs. intent** — does this stage **transform** what the user said stays
   fixed, or **drop** rows they expected kept?
5. **Direction** — for a metric filter, is lower or higher better? Keep the correct side.

The critique returns `pass`, `revise`, or `ask`, and must copy the recipe's `config_hash` —
binding the judgment to that exact recipe. **Any recipe change invalidates the critique.**

**A deliberate architectural rule:** *do not add module-specific intent rules to the
deterministic core.* Universal invariants (real stages, impossible key flow, serialization,
environment gates, confirmation integrity) stay deterministic. Open-ended meaning stays with
the LLM, grounded in card evidence. Trying to encode meaning as rules produces an
ever-growing, always-incomplete table of special cases.

**Without it.** Confidently wrong pipelines that pass every check and answer a different
question than the one asked.

---

### Layer 9 — Resource planning and feasibility

**Curator context.** Curator can run stages in **streaming** mode (all stages concurrently,
fast) or **batch** mode (one stage at a time, sequential, lower footprint). Picking wrong
means either an aborted run or wasted time. The scheduler enforces its own resource
reservations and aborts if the concurrent sum does not fit.

**What we added.** A deterministic planner that, from each stage's declared needs and the
probed machine, picks streaming or batch and reports feasibility.

The subtle and important part is **what it gates on and what it merely warns about**:

| Gated (hard, exact facts the scheduler itself enforces) | Advisory only |
| --- | --- |
| Concurrent GPU **reservations** must fit the GPU count | Estimated **VRAM** |
| A GPU-only stage needs a GPU on the box | |
| CPU demand and reservation vs. the machine's allocatable CPUs | |
| Host RAM vs. the machine's RAM | |

**Why VRAM is deliberately not a gate.** Real VRAM depends on weights × activations × batch
size × precision, and machine VRAM is frequently unknown on cloud hosts. The card value is a
best guess. The scheduler does not reserve VRAM anyway — it reserves GPU *fractions*. And
critically: **gating on VRAM would block the very measurement step (smoke) that would
resolve the uncertainty.** So the planner warns, the bounded smoke measures on the real GPU,
and the runtime auto-falls-back from streaming to batch if streaming turns out not to fit.

*Principle: never let a guess block the measurement that would replace the guess.*

**Composite flattening.** Some stages are composites that expand into several inner stages
at runtime, advertising only their own footprint. Summing the wrapper undercounts GPU
reservations and wrongly picks streaming — which the executor then aborts. The planner
expands composites before doing its arithmetic. This is generic, not one recipe's fix.

---

### Layer 10 — Bounded smoke and calibration

**Curator context.** A full run is hours of GPU time. There is no built-in "try it small
first".

**What we added.** `smoke` runs the **real pipeline with the real models** on a bounded
sample (~10 items), writing only into an ephemeral directory. It returns retained/rejected
counts, concrete examples, errors, whether the goals were met, and a structured diagnosis
when they were not.

This turns three otherwise-invisible questions into evidence:

- Does the pipeline actually run end-to-end on *this* data?
- Does the threshold retain a sensible fraction, or did it drop everything?
- What does it really cost per item?

**Calibration.** A GPU smoke measures per-stage VRAM and throughput on the actual machine.
Those measurements feed the full run's resource plan — but **only upward**. A bounded smoke
cannot prove the full run's maximum, so a measurement may *raise* a conservative card
estimate and never lower it. Calibration is stamped with a machine fingerprint and is
discarded on a different machine.

**Without it.** Every mistake is discovered at full scale, and every resource estimate stays
a guess.

---

### Layer 11 — Safety guardrails and the confirmation gate

**Curator context.** Curator is a library. It does what it is told, at whatever scale it is
given. There is no notion of "ask first".

**What we added — enforced inside the tools, not in the prompt.**

**(a) The confirmation gate — zero silent full-scale runs.** `run` refuses without explicit
confirmation and returns a scale estimate instead. Confirmation carries the recipe's
`config_hash`, so **what was approved is byte-for-byte what runs**. A changed threshold after
approval produces a different hash and a fresh refusal.

*(A real bug worth mentioning: gating on "not exactly False" once let a JSON `null` forwarded
by the MCP adapter slip through into a silent full-scale run. The gate now accepts only a
literal `true` or a matching hash string.)*

**(b) Nothing is written before approval.** No file or directory is created, deleted, moved,
or truncated ahead of the gate — above all not the user's output. *A gate the agent has
already prepared the ground for is not a gate.*

**(c) Workspace path lock.** File paths must resolve under an allowed root. Blocks traversal
and any read/write outside it.

**(d) Secret and transcript redaction.** Secret-looking keys and transcript text are stripped
from tool output before it reaches the model. Reuse keys are computed *before* redaction, so
redaction never corrupts identity — only the persisted and returned copies are redacted.

**(e) Optional require-smoke.** In stricter deployments, `run` refuses unless handed a valid
token proving a smoke ran for *this exact recipe*.

**What is deliberately *not* here.** Semantic misuse refusal ("isolate this named person's
voice") needs judgment a deterministic tool cannot make. It stays a skill/policy concern.

---

### Layer 12 — Execution and monitoring

**Curator context.** Curator executes on Ray/Xenna. Getting a correctly configured cluster
up is a setup task, and it fails in unhelpful ways (occupied ports, unwritable shared-memory
locations, API limits).

**What we added.**

- **Opt-in Ray bootstrap.** With one flag, the agent starts a correctly-configured local head
  on a free port, with the object store on a writable directory and the state-API cap set. An
  externally configured cluster address is always respected and never clobbered. Users with
  their own cluster are unaffected.
- **Streaming → batch auto-fallback.** If the runtime reports that streaming does not fit, the
  run drops to batch rather than failing.
- **Per-stage metrics** collected during execution: timings, throughput, GPU seconds, resource
  peaks. These feed the report, the calibration, and the reuse cost model.
- **Fan-out-aware counting.** A stage that turns one file into many segments used to be
  double-counted, inflating the numbers a user reads. Counting now aggregates per stage and
  source, and distinguishes *source items* from *output rows* explicitly.
- **Optional checkpointing** for partial-run recovery when the stages are resumability-safe.

---

### Layer 13 — Failure analysis and recovery

**Curator context.** Curator errors are raw stack traces from Python, CUDA, Ray, or a model
loader. Many look identical and mean completely different things.

**What we added.**

**(a) A versioned failure taxonomy.** 18 classified failure signatures mapping symptom →
likely cause → layer (data / recipe / environment / execution / model) → guidance:
unreadable file · missing stage input · sample-rate mismatch · disk full · permission
denied · native library / ABI mismatch · CUDA driver-runtime · TLS certificate · unknown
target capacity · **worker environment mismatch** · missing dependency · **ASR decoder
CUDA-graph / PTX** · out of memory · executor error · model auth or download · bad path ·
empty VAD output · zero rows retained. Signatures are matched specific-before-generic, and
new ones are purely additive.

**(b) `diagnose` — recipe-aware.** Given a captured error, it returns a sanitized
classification, a *fresh* environment probe, which of the recipe's stages are affected, and
**ranked, grounded recovery options** — each labelled by kind (host change, environment
change, launch change, recipe variant, credential, diagnostic) and by availability
(available / conditional / unavailable / unknown).

**(c) The recovery discipline — this is the part that matters.** The core supplies facts and
options. **It never applies them.** When a decision is required, the agent must:

- stop before execution, and state the detected fact and its confidence **separately from any
  inference**;
- explain the impact on *this* recipe;
- recommend the best *available* option given the user's stated constraints, with the
  trade-offs of the alternatives;
- ask one outcome-level question — and wait.

And it must **never**: silently install, upgrade, or downgrade anything; switch to CPU;
change the launch command; change a model, decoder, or stage; request or expose a credential
in chat; or retry the same failed action.

Three refinements worth highlighting:

- **CPU fallback is a conditional candidate, not a fix.** It is offered only when the evidence
  proves *every* affected execution stage supports CPU — and it is not called executable until
  the new recipe builds and smokes. Any such alternative is a **new recipe and a new hash**:
  validate → smoke → confirm again.
- **Scope facts to the execution target.** For an external cluster, the local machine's GPU,
  ffmpeg, disk, and launch facts are *not* claimed as remote facts. The packet says the target
  is unverified and asks for a bounded target-side smoke.
- **Unknown stays unknown.** An unrecognized failure permits only the packet's minimal
  diagnostic steps — never an invented root cause or command.
- **Re-verify after any fix.** Never assume an external change worked; re-run the pre-flight.

**Without it.** The classic agent anti-pattern: an agent that "fixes" a CUDA error by
installing packages, switching devices, and retrying — mutating the user's machine while
producing results nobody can trust.

---

### Layer 14 — Result validation and honest reporting

**Curator context.** Curator produces output files. It does not evaluate them against
anything.

**What we added.**

**(a) Deterministic acceptance verification.** The success contract frozen into the recipe is
evaluated against real output evidence — an exhaustive read-back of the terminal manifest,
not a sample, when the contract requires it. Four **honest states**:

| State | Meaning |
| --- | --- |
| `met` | proven by evidence |
| `not_met` | proven false |
| `unverifiable` | no evidence exists (e.g. word error rate with no reference transcripts) |
| `unachievable` | the data provably cannot reach an absolute target |

`overall` is `met` **only if every `must` criterion is met**. Two edge cases the agent must not
misreport: an **empty** contract returns `unverifiable`, not `met` — nothing was verified, so
success cannot be claimed on zero evidence; a non-empty contract carrying only `nice` criteria
*does* stay `met`, because `nice` is non-blocking by design. The last two states exist so the
agent can be honest rather than optimistic — "unachievable" is a legitimate, useful outcome.

**(b) The honesty guard (anti-goalpost-moving).** This compares the criteria actually verified
against the **frozen, confirmed** contract and flags:

- `must_dropped` — a confirmed requirement is missing from the verified set
- `must_downgraded` — a `must` quietly became a `should`
- `must_relaxed` — the bar moved in the easier direction

If the guard fires, `overall` is **forced** to `not_met`. Success cannot be presented.
Relaxing a bar means re-confirming a *new* contract with the user — the contract is inside the
recipe and covered by the approval hash, so it cannot be edited silently.

The guard is deliberately conservative: a changed type, scope, method, target, or failure
policy is **not** treated as safely comparable even if the number looks stricter. Only a
same-operator numeric strengthening is auto-accepted.

**(c) The reviewer charter.** After the deterministic verdict, the LLM resolves judgment-based
criteria (which correctly come back `unverifiable` — that is the reviewer's job), reads the
honesty section, and may **surface** semantic concerns but **may not override** the
deterministic verdict. Unresolved concerns escalate to the user.

**(d) Evidence-only claims.** No claim that quality or throughput improved without before/after
numbers from a report.

**(e) Report the metric's limits at the gate.** When presenting the contract, the agent must
state what each metric captures **and what it does not** — e.g. "UTMOS measures overall
perceptual naturalness, not background-noise level specifically; add a noise criterion?" It
must never silently decide which metric stands in for a fuzzy word like "clean".

---

### Layer 15 — Reuse of previous work (deterministic memoization)

**Curator context.** Re-running a pipeline re-runs everything. Transcribing a corpus is hours
of GPU time. "Now also add speaker labels" traditionally means paying for the transcription
again.

**The first attempt failed, instructively.** The original mechanism compared a new recipe
against one previous run. It was correct and almost never fired, because:

- identity conflated three different things — changing a batch size, a GPU reservation, an
  output directory, or *tightening the success bar* all changed the identity hash without
  changing a single output byte;
- the dataset key was a shape hash — simultaneously **unsafe** (a file edited in place was
  invisible) and **too coarse** (adding one file invalidated everything);
- reuse was all-or-nothing and could not actually execute — it printed a plan a human had to
  hand-implement;
- intermediate outputs existed on disk but were invisible, because output discovery only
  looked at a few literal parameter names.

**The rewrite: ask a better question.** Stop asking *"was this whole recipe run before?"* and
start asking **"has this step, with these semantics and the same resolved dataset identity,
already produced an artifact?"**

Identity became a chain over the pipeline: the dataset identity seeds step 1's key, and each
step's key is derived from the previous key plus its stage, its semantic parameters, the code
version, and the model version. One mechanism then gives whole-pipeline, prefix, and
single-stage reuse — and invalidation **falls out of the chain** instead of needing
hand-written rules.

| Change | Effect |
| --- | --- |
| A meaningful parameter at step *i* | steps *i…n* change → reuse *0…i-1*, rerun the rest |
| A batch size or GPU reservation | no key change → full reuse |
| An output location | no key change → full reuse, published to the new place |
| The success criteria | data reused, **contract re-verified** (not recomputed) |
| The input data itself | every key changes → full rerun, **unless `delta-run` applies** (below) |

**Safety mechanisms that make this trustworthy:**

- **Atomic completion markers.** An artifact is reusable only once a marker exists naming its
  step key, row count, byte count, and a content digest. Lookup **recomputes that digest**, so
  editing a file after publication invalidates reuse even if the size and row count match. This
  also closes a crash bug: a crashed run left a partial-but-valid-looking output file.
- **A tiered dataset key.** The strong tier is metadata-backed (relative paths, sizes,
  modification times) and catches ordinary in-place edits; a weaker shape tier is the fallback
  for remote or unreadable sources. **A weak-tier match is low trust and defaults to a fresh
  run**, and the tier is always reported.
- **Declared non-determinism.** A stage whose output can legitimately differ between runs says
  so on its card. Its stored artifact is still *offered* — with the caveat shown and *fresh*
  pre-selected — rather than hidden. (Hiding it meant users were never told prior work existed.)
- **A rebuildable index.** JSON records remain the human-readable source of truth; a SQLite
  index is a pure cache for fast lookups, and `reindex` rebuilds it from disk. Nothing lives
  only in the database.

**The approval UX — never silent, never nagging:**

1. No candidate → no prompt. Run fresh, say nothing.
2. A **measured** saving under 30 seconds → just take it, and disclose it in the report.
3. Low trust → show the weakness in plain language and pre-select **fresh**.
4. Otherwise → present a card a human can judge (what the earlier run was *for*, what it ran,
   on what data, where the output is, when, how it scored, estimated time saved) and offer
   **as-is / extend / fresh**.

**"Measured" carries weight there.** Silence about a stage's cost is not evidence it was
cheap — that is how an unmeasured hour of transcription once qualified as "trivial". Unpriced
stages that the cards call expensive force the question rather than being auto-taken.

**Executable continuation.** Choosing "extend" no longer produces advice. A deterministic
materializer rewrites the recipe to start from the reused artifact, re-validates the remainder
against what that artifact actually carries (including a guard for reads that break because a
persisted file cannot carry an in-memory waveform), and runs only the remaining stages — through
the **same** validate → confirm → run path. Reuse buys no shortcut past the safety gates.

**Honest about what it cannot do.** Only a stage that *writes something* can be a resume point —
you can only resume from disk. For the common shape (several in-memory transforms feeding one
writer), the realistic outcomes are `already_done` and `fresh`. Rather than hide this, the scan
reports `prior_unsaved` — "these stages ran before and are being recomputed; here is what they
cost last time" — plus an offer to persist them next time: **`add-checkpoint`** names where a
mid-pipeline manifest would make the expensive stages reusable. **Recomputation is never silently
presented as new work.**

**Explaining a miss.** When the data changed, every key changes and the probe finds *nothing*.
Reporting "never ran this before" would be technically true and useless. So on a miss the scan
re-keys against datasets already in the registry and reports "you ran this pipeline on *that*
dataset on *that* date". The three rejection reasons stay distinct on purpose — *your data moved
on*, *the output vanished*, *the output was never marked complete* — because each implies a
different fix.

**Per-file deltas (`delta-run`) — when the corpus changed, not the recipe.** "Your data moved on"
is the most common miss, and it does *not* have to mean a full rerun. `delta-run` processes only
the files that changed since a prior run and merges them into that run's result. It is the reuse
path for a growing corpus, and it lives on the miss path by design.

Three preconditions are checked before anything runs, and each one **refuses by name** rather than
degrading into a guess:

- a **per-file inventory** must exist for the input — without one, changed files cannot be
  identified at all;
- a prior run over an **overlapping** corpus must be found — the newest run is not automatically
  the right one, because running the same recipe over several corpora is ordinary;
- every stage up to the resume depth must be **provably per-row independent**. A stage that has
  not declared `gates.per_row_independent` and can reach the corpus is a **refusal, not an
  assumption** — it may well be independent, but silence is not evidence.

It also refuses when the prior run persisted nothing resumable (→ `add-checkpoint`), and when the
stages after the resume point need in-memory state a manifest cannot carry.

**A `no_delta` answer means "run normally" — never "nothing changed."** The delta path buys no
shortcut past the safety gates: the rewritten recipe goes through the same validate → confirm →
run path as any other.

**An explicitly stated boundary.** This is **deterministic memoization**, not learning. There is
no cross-session learning, no learned priors, nothing that changes *what* the agent plans. Records
let it skip work it can *prove* is identical. The original "no memory" non-goal was **amended in
writing** to say this, rather than being silently contradicted.

---

### Layer 16 — Provenance and run records

Every run writes a JSON record: the recipe and its hashes, the goal it was **for** (recorded in
the user's words — this is what makes a reuse candidate legible to a human months later), the
dataset identity and tier, per-stage metrics, elapsed time, the acceptance *outcome*, an
environment summary, the Curator and knowledge versions, and the step-key chain. `runs --data
<path>` answers "what has already been done to this corpus?"

---

### Layer 17 — The tool surface

The same deterministic core, three doors:

- **CLI** — every verb prints JSON; drivable from a shell or notebook by a human or an agent.
- **MCP server** — the verbs as typed tools for any MCP host (Claude, Cursor, …).
- **Python SDK** — `from nemo_curator import audio_agent`.

Plus the **written driving instructions** — a skill file, a Cursor rule, and `AGENTS.md` files —
that turn the golden rules into something a host model actually follows. These documents are part
of the system, not commentary on it.

---

### Layer 18 — Evaluation

**Curator context.** "Did the pipeline run" is a weak test for an agent.

> Do not evaluate only whether the pipeline ran. Evaluate whether the agent chose the
> **appropriate, complete, compatible, correctly-configured, efficient, recoverable, and
> explainable** pipeline for the user's requirement.

**Two planes, two harnesses:**

- **The deterministic core** is fully automated and GPU-free in CI: composition, parameters,
  gates, acceptance, reuse, safety, planner feasibility, and the card conformance gate — driven
  by a query set with gold recipes and expected outcomes, including negative cases (unknown stage
  rejected; word error rate unproducible without transcripts; subjective-trait labelling maps to
  no stage → refuse).
- **The LLM plane** is checked by golden tool-call **trace assertions**, an **LLM-as-judge**, and
  human review. The trace grader is deliberately domain-agnostic: it requires card inspection, a
  green validation of the *exact* final recipe, review coverage for every configured stage, and
  resolvable citations — and it verifies citations appear in **returned results**, not merely in
  tool arguments. It contains no metric-, module-, or speaker-specific rules.

**Two details worth presenting:**

- Every semantic-intent scenario ships a **mechanically valid counterexample** — a pipeline that
  composes perfectly and answers the intent *wrongly*. These exist to prove, concretely, why
  validation cannot be the semantic oracle.
- Live captures are bound to real checked-in audio and **never mock GPU availability or rewrite a
  failed verdict to green**. On a host whose GPU pre-flight genuinely fails, those scenarios are
  reported as blocked and the gate **fails honestly**.

A 23-class failure taxonomy maps each failure kind to the pipeline stage that should detect it,
the real signal that detects it, and a severity — with guardrail breaches (path traversal, secret
leakage, a silent full-scale run, accepted goalpost-moving) always at the highest severity
regardless of category.

---

## 7. Deterministic core vs. fully LLM-driven design

### 7.1 The split, in one table

| Concern | Owner | Why |
| --- | --- | --- |
| Understanding an ambiguous request | **LLM** | Language is the LLM's home turf |
| Deciding what "clean" means for this user | **LLM** | Requires context and a conversation |
| Choosing between two overlapping modules | **LLM**, from card facts | A trade-off, not a rule |
| Explaining *why* a choice was made | **LLM** | Explanation is reasoning |
| Judging whether a valid recipe means the right thing | **LLM**, from the evidence packet | Open-ended; does not reduce to rules |
| Presenting options and asking | **LLM** | Requires judgment about what is material |
| **What stages exist** | **Core** | A fact. Facts must not be generated |
| **What a parameter is called** | **Core** | A fact |
| **Whether a pipeline composes** | **Core** | Provable |
| **What a threshold should be for "studio"** | **Core**, from card anchors | Must be repeatable and auditable |
| **What the machine can do** | **Core** | Measurable |
| **Whether to run at full scale** | **Core** (gate) + **user** (decision) | Must be un-skippable |
| **What actually happened** | **Core** | Evidence, not narration |
| **Whether success criteria were met** | **Core** | Must be immune to optimism |
| **Whether work can be skipped** | **Core** | Must be provable, not plausible |

### 7.2 Why each property required this split

**Preventing hallucinations.** A hallucinated stage name cannot execute, because names resolve
through a registry. A hallucinated parameter cannot execute, because it is passed to a real
constructor. A hallucinated threshold cannot be entered, because thresholds come from card
anchors. The LLM is not *asked* to avoid hallucinating; it is *unable* to hallucinate anything
consequential.

**Staying within supported capabilities.** The catalog is the boundary. If nothing can produce a
required capability, the core says so explicitly and the user is told the goal is impossible with
these stages — rather than receiving a plausible pipeline that quietly does something else.

**Producing valid, compatible pipelines.** Compatibility is a graph property over declared
contracts. That is a computation, and computations should be computed.

**Reducing token usage.** Tiered retrieval means a typical request reads a category tree, a
handful of one-liners, and two or three full cards — not 49 cards and certainly not 49 source
files. Deterministic verbs also return *compact JSON verdicts* rather than logs to be
interpreted. Fewer tokens here is not only cheaper — it is **more accurate**, because the model
is not reasoning over noise.

**Repeatability and reliability.** The same recipe always produces the same hash, the same
validation verdict, the same resource plan, and the same acceptance verdict. That is what makes
approval meaningful: what a user approved is provably what runs.

**Using the LLM where it adds value.** Everything the deterministic layers *cannot* do —
interpreting a vague request, weighing two reasonable modules, noticing that a green pipeline
answers the wrong question, explaining a trade-off in plain language — is exactly where the LLM
is placed, and it is given real evidence to work from rather than being asked to recall facts.

### 7.3 The rule we kept coming back to

> Deterministic checks own **universal invariants**: real stages, impossible key flow,
> serialization, environment gates, side effects, confirmation integrity.
> The LLM owns **intent-dependent meaning**: what a field means here, whether this metric is a
> good proxy for what the user said, which trade-off fits.
> **Do not add module-specific intent rules to the core.**

Every time we were tempted to encode a semantic rule ("`num_speakers` must not be filtered after
separation"), the right move was instead to put the *fact* in a card (meaning, scope,
counterexample) and let the critic reason. Rules would have produced an ever-growing,
always-incomplete table of special cases.

---

## 8. Key design decisions

Each stated as a decision, its reasoning, and what it bought.

**1. Host-driven, not embedded-LLM.** The core contains no model and makes no calls. It works
with whatever host the user already has, is testable without a model, is deterministic and
cheap, and does not lock the design to one provider.

**2. A declarative Recipe, never generated code.** The single most important safety decision.
Generated code is unreviewable and unbounded; a recipe is a short list of real names that a
human can read in ten seconds and a machine can verify completely.

**3. Knowledge as versioned YAML, not prompt text.** Cards, taxonomy, blueprints, patterns, and
the failure taxonomy are files — reviewable, diffable, testable, and gate-enforced against the
code. Prompt text drifts silently; files do not.

**4. Cards are gate-checked against the code.** A card that claims a parameter the stage does not
have fails an automated check. Without this, cards become documentation, and documentation rots.

**5. Honesty tiers on every fact.** Each fact group declares whether it is derived from code,
measured on real hardware, or author judgment. The agent (and the reader) can tell a measurement
from an opinion. The golden rule — *a missing fact is better than a wrong one* — follows from
this.

**6. Tiered retrieval instead of full context.** Cheaper *and* more accurate.

**7. Validation is mechanical only, and says so out loud.** Rather than pretending validation
proves correctness, we drew the line explicitly and built the semantic-review layer beside it.
Overclaiming here would have been the most dangerous possible design choice, because green means
"stop worrying".

**8. Semantic review is mandatory, not optional.** It is not a nice-to-have step the model may
skip when confident. It runs after every clean validation, and it is bound to the exact recipe by
hash.

**9. Three hashes for three questions.** Approval integrity, reuse identity, and the success bar
are genuinely different questions. Collapsing them meant either unsafe approval or reuse that
never fired.

**10. A confirm gate with hash integrity, and nothing written before it.** "Ask before you run" is
worthless if the plan can change after approval, or if the agent has already touched the
filesystem. Both holes are closed.

**11. Guardrails in the tools, not the prompt.** A weaker model, a direct CLI call, or a scripted
caller hits the same refusals. A prompt-level rule protects only well-behaved callers.

**12. Never let a guess block the measurement that would replace it.** Concretely: VRAM estimates
warn, they do not gate; masked GPUs defer to a real probe. This one principle prevented a whole
class of "the agent refuses to try" failures.

**13. Measurements may raise an estimate, never lower it.** A sample of ten cannot prove the
maximum of ten thousand.

**14. Reuse identity is content-addressed per step, not per run.** One mechanism yields
whole-pipeline, prefix, and single-stage reuse; invalidation falls out of the chain rather than
needing rules.

**15. Never reuse silently; never nag.** Both halves matter. Silent reuse means the user does not
know they got yesterday's answer. Nagging means they stop reading the prompts.

**16. Low trust defaults to fresh, and says why.** Between a slow answer and a wrong one, this
system fails towards slow — explicitly and by design.

**17. Diagnosis proposes; it never applies.** No silent installs, device switches, model swaps, or
retries. The trade-off is a slower recovery loop; the gain is a machine and a result the user can
still trust.

**18. Success is defined before the run and verified after — with an honesty guard.** Without the
guard, the previous decision is decoration: an agent that can quietly lower the bar always passes.

**19. Every stage carries its own contract; adding a stage needs no core change.** A new
non-source stage becomes plannable by shipping a contract and a card. *(A new **source** stage
additionally needs an explicit input-identity adapter — a deliberate closed table rather than
guessing from parameter names, because getting dataset identity wrong means serving the wrong
bytes.)*

**20. Linear recipes, not a DAG engine.** Audio recipes are chains. A general DAG intermediate
representation would be cost without benefit today. Revisit when a real branching case appears.

**21. Needing ad-hoc Python is a capability gap to report, not a workaround to normalise.** When a
real session reached for hand-written Python to reshape output, the answer was to add a proper
export stage — not to make script generation an accepted escape hatch.

**22. Amend a stated non-goal in writing rather than quietly contradicting it.** The "no memory"
non-goal was rewritten to state precisely what deterministic memoization is and is not.

---

## 9. Important learnings

**1. LLM reasoning is valuable exactly where rules run out — and dangerous exactly where rules
exist.** Interpreting "clean single-speaker data", weighing two diarizers, noticing that a green
pipeline answers the wrong question: excellent. Recalling a parameter name, judging whether a
graph composes, deciding a threshold: replace with a lookup or a computation.

**2. Mechanically valid ≠ semantically correct, and the gap is where the real bugs live.** The
single-speaker trap validates perfectly. Every bug of this class was invisible to every
mechanical check we had, and every one was catchable by asking *"what does this field mean, at
this point, for this entity?"*

**3. Structured knowledge beats prose, and gate-checked knowledge beats structured knowledge.** A
card that lists a metric's direction and anchors is worth more than a paragraph explaining it,
because it can be resolved and enforced. A card that is automatically checked against the code is
worth more still, because it cannot rot.

**4. A missing fact is safer than a wrong one.** The agent trusts cards absolutely. A `TODO` makes
the agent ask; a wrong value makes it confidently wrong. This is why honesty tiers exist.

**5. Validation must come before execution — and there must be more than one kind.** Mechanical
validation before smoke; semantic critique before smoke; bounded smoke before scale; acceptance
verification after. Each catches a class the others cannot.

**6. Agents must see the environment, and must not fix it.** Environment problems are a large
share of real failures and masquerade as code bugs. But an agent that silently installs packages
and switches devices to make an error go away leaves a mutated machine and an untrustworthy
result. Detect, explain, propose, ask.

**7. Never state an absence as a fact when the observation could be blocked.** "No GPU" from
inside a sandbox is not a hardware fact. The distinction between *"is not"* and *"cannot be seen
from here"* prevented a whole class of confidently wrong environment claims.

**8. Reuse is worth the complexity, but only if it is provable.** Hours of GPU time saved matters.
It is worth nothing if a stale result can be served silently. Every safety mechanism — completion
markers, content digests, identity tiers, declared non-determinism, low-trust-defaults-fresh —
exists because the naive version would be worse than no reuse at all.

**9. Silence is not evidence.** An unmeasured stage is not a cheap stage. Reading absence as zero
is how an unmeasured hour of transcription qualified as a "trivial" saving.

**10. Launching a pipeline is not the job — analysing it is.** Counts, per-filter drop reasons,
failure examples, and a verdict against a pre-agreed contract. Otherwise the agent is a fancy
`subprocess.run`.

**11. Define success before you start, or you will discover it afterwards.** Retro-fitting "what
did we want?" onto results is how goalposts move. Freezing the contract into the approved artifact
makes moving them detectable.

**12. Design for the honest failure.** `unverifiable` and `unachievable` are first-class outcomes.
An agent that can only say "success" or "error" will say "success".

**13. Guardrails belong in the tools.** Every rule that lives only in a prompt is a rule that a
weaker model, a direct call, or a future refactor will skip.

**14. Real incidents should become structural fixes, not warnings.** An agent deleted a user's
file before the confirm gate, having misread a stage's append-mode file open. The fix was not a
warning — it was `output_targets` (the agent can now *see* what is there) plus an explicit rule
that nothing is written before approval, plus the reasoning recorded so nobody re-derives the
wrong conclusion.

**15. Flexibility comes from generic mechanisms, not from more cases.** The tests: is there any
metric name in the resolver's logic? Any module name in the semantic layer? Any per-recipe rule in
the planner? The answer must stay *no*. Every layer is a registry, a table, or a chain, and every
extension is one card, one YAML entry, or one decorated function.

**16. Write the boundaries down, in the same place as the capability.** Every document in this
system states what its layer does *not* prove. `validate` says it does not certify intent.
`semantic_review` says it does not judge. Cards say which facts are guesses. Reuse says what it
cannot resume from. Overclaiming is the failure mode that makes people stop checking.

**17. The first design of the hardest subsystem was wrong, and rewriting it was correct.** Reuse
was built, shipped, and almost never fired. The rewrite came from changing the *question*, not
from tuning the answer.

---

## 10. Impact of the agentification work

### 10.1 Before and after

| Dimension | Manually using Curator | With the Audio Agent |
| --- | --- | --- |
| **Getting started** | Read docs, learn 49 stages and 10 categories, understand field-passing conventions | Describe the outcome in plain language |
| **Finding capability** | Browse source and docs | Coarse-to-fine routing over an indexed catalog |
| **Choosing between similar stages** | Guess, or ask an expert | Compared on card facts, with the trade-off explained |
| **Setting thresholds** | Guess a number and hope the direction is right | Say "studio" or "general"; the value is resolved from anchors with an audit trail |
| **Compatibility** | Discovered at runtime, on the GPU | Proven before anything runs, with the fix named |
| **Intent correctness** | Nothing checks it | A mandatory grounded critique, with a five-question checklist |
| **Environment problems** | Cryptic runtime errors | Named, explained, with grounded fix options |
| **Cost of a mistake** | Hours of GPU time | Caught at validation, or at a ~10-item smoke |
| **Running at scale** | Whatever you typed, immediately | Refused until explicitly approved, hash-bound to the plan |
| **Repeating work** | Full re-run | Provably identical work is skipped, with disclosure |
| **Knowing if it worked** | Read the output and judge | A verdict against a pre-agreed contract, with an anti-optimism guard |
| **Explaining the pipeline** | Reconstruct from code | Every stage justified against a stated goal, every value traced to its source |

### 10.2 Concrete outcomes

- **Reduced engineering effort.** Building a working audio-curation pipeline goes from "learn the
  framework" to "describe the outcome and answer one or two questions".
- **Faster pipeline creation.** The expensive iteration loop — write, run, crash, fix, re-run — is
  replaced with validate, critique, and a bounded smoke.
- **Fewer configuration mistakes.** Whole classes are structurally eliminated: invented stage or
  parameter names, incompatible wiring, inverted filters, wrong sample rates, unsatisfiable
  success criteria.
- **Better consistency.** The same request produces the same validated recipe with the same
  resolved values — a plan that can be reviewed, approved, versioned, and compared.
- **Reuse of existing results.** Hours of GPU work are skipped when it is *provable* that the
  computation and data are identical, never silently.
- **More explainable decisions.** Every stage traces to a stated goal, every value to a card
  anchor, every claim to measured evidence. The audit trail is a produced artifact, not a
  reconstruction.
- **Access for non-Curator users.** Someone who has never heard of Curator can produce a curated
  dataset — and, importantly, be told honestly when their goal is not achievable with the
  available stages.
- **A safer default posture.** Nothing runs at scale unapproved. Nothing is written before
  approval. No secret reaches the model. No environment is mutated silently. No success is
  declared without evidence.

### 10.3 A benefit that is easy to miss

The work made the **stages themselves better documented and better behaved**. Contracts, cards,
the conformance gate, and the failure taxonomy are valuable to *human* users too. Agentification
forced the codebase to state things it had only ever implied — and much of that knowledge existed
nowhere before, not even in someone's head.

---

## 11. Limits, non-goals, and future possibilities

**Stated honestly — this is a deliberate part of the story.**

**Current limits.**
- Knowledge is hand-authored. A new stage needs a card; coverage gaps are *reported*, not failed.
- Semantic correctness depends on card quality and on the host model's judgment. Structured
  critique sections prove review *coverage*, not that the prose is *true* — hence the model judge
  and periodic human review.
- Reuse resumes only from persisted output. An all-in-memory pipeline has nothing to resume from;
  this is disclosed rather than hidden.
- The strong dataset identity tier is metadata-backed, not a full content hash. A mutation that
  deliberately restores both size and modification time would evade it — which is why the tier is
  always reported and low-trust matches default to fresh.
- Relative goals ("the best 20%") need the data-informed path and currently return a question.
- Recipes are linear chains, not general graphs.

**Explicit non-goals (today).** No cross-session learning, no learned priors, no best-of-N
scoring, no embedded planner model, no automatic environment remediation, no agent-authored
Python.

**Designed but not built.** Content-digest identity tiers above the metadata tier;
compatible-superset reuse (re-filtering a looser result to satisfy a stricter request, always
approval-gated); artifact retention and garbage collection; a `why-rerun` verb that names exactly
which key changed; row-level lineage across fan-out boundaries.

**Natural extensions.** The same architecture — contracts, cards, an index, deterministic
validation, a semantic-evidence packet, gates, and memoization — is **modality-agnostic**. Nothing
in the core is audio-specific except the knowledge files. The environment health registry is
already generic.

---

## 12. Presentation storyline

A suggested flow, with the one idea each section must land.

| # | Section | The one idea | Suggested material |
| --- | --- | --- | --- |
| 1 | **The audio-curation challenge** | Raw audio is not training data; turning one into the other is many decisions | §1.2 |
| 2 | **Curator in 60 seconds** | Powerful stages, assembled by hand | §1.1 |
| 3 | **Why this is hard for a newcomer** | The stages were never the problem — everything around them was | §1.3 table |
| 4 | **Why not just wrap an LLM?** | An agent that is usually right and occasionally confidently wrong is worse than none | §2 table |
| 5 | **The core idea** | The LLM proposes; the deterministic core disposes | §3 |
| 6 | **The architecture** | Two planes, one JSON boundary, three doors | §4 diagram |
| 7 | **Following one request end to end** | Fifteen steps, and no step can be skipped | §5 table |
| 8 | **The layers** | Each layer removes one specific way to be wrong | §6, grouped into: know · understand · see · plan · prove · run · check · remember |
| 9 | **The layer people underestimate** | *Green validation is not correctness* — the single-speaker trap | Layer 8 |
| 10 | **Who decides what** | Facts computed, judgment reasoned | §7 tables |
| 11 | **Design decisions** | Pick 6–8: no codegen · gate-checked knowledge · three hashes · confirm gate · guardrails in tools · guesses never block measurements · diagnosis proposes but never applies · honesty guard | §8 |
| 12 | **What we learned** | Pick 6–8, lead with #2 and #6 | §9 |
| 13 | **Impact** | Before/after, then the outcomes | §10 |
| 14 | **Limits and what is next** | Stating boundaries is the credibility | §11 |

**Three moments that carry the talk.** If time is short, keep these:

1. **The single-speaker trap** (Layer 8) — makes "valid ≠ correct" visceral in 60 seconds.
2. **The deleted file** (Layer 7 / decision 10) — makes the safety architecture concrete: a real
   incident that became structure, not a warning.
3. **The reuse rewrite** (Layer 15) — a subsystem that was correct, shipped, and useless, fixed by
   changing the question rather than tuning the answer.

**Suggested closing line.** *We did not make an LLM that runs Curator. We made Curator's knowledge
explicit enough that an LLM can be trusted with the parts of the job that actually need judgment —
and prevented from touching the parts that do not.*

---

## 13. Appendix: quick reference

### The verbs (all return JSON)

| Group | Verbs | Purpose |
| --- | --- | --- |
| Knowledge | `discover` · `catalog-tree` · `describe` · `cards` · `context` · `producers` | What exists; route coarse-to-fine; assemble planning context; **`producers` answers "which stages write this role or key?"** — the targeted re-retrieval for a missing producer |
| Environment | `doctor` · `diagnose` | Machine health with fixes; recipe-aware failure analysis |
| Configuration | `resolve` | Outcome label → concrete parameter, with an audit trail |
| Validation | `validate` | Composition, contracts, gates, criteria + the semantic-review packet |
| Evidence | `smoke` · `calibrate` | Bounded real run; measured per-stage resources |
| Execution | `run` | Confirm-gated full run + report + acceptance |
| Results | `report` · `verify` | Post-hoc evidence; acceptance verdict + honesty guard |
| Reuse | `reuse-scan` · `continue` · `runs` · `reindex` | Find, execute, inspect, and rebuild prior work |
| Reuse (corpus changed) | `delta-run` · `add-checkpoint` | `delta-run` processes only the changed files and merges them into a prior run; `add-checkpoint` names where a mid-pipeline manifest would make the expensive stages reusable |
| Setup | `install-skill` | Install the packaged skills where Codex/Cursor/Claude Code find them |

### The loop

```
interpret + define success  →  inspect (data + environment)  →  route (L0→L1→L2)
   →  resolve outcomes to parameters  →  plan (Recipe)  →  validate
   →  semantic critique (pass / revise / ask)  →  reuse-scan  →  smoke
   →  present + confirm  →  run  →  verify  →  report
```

### The golden rules

1. Never invent a stage or parameter name.
2. Zero silent full-scale runs — the confirm gate carries the recipe's hash.
3. Nothing is written before approval.
4. Evidence-only claims — no before/after numbers, no claim.
5. Define success up front; verify it after; never silently relax a `must`.
6. Ask at the outcome layer, never about internal parameters.
7. Refuse subjective labelling and training/deployment requests; offer a safe alternative.
8. Diagnose and explain environment problems; never fix them silently.
9. Never reuse silently; never nag.
10. Green validation is not intent approval.

### By the numbers

49 agent-ready stages · 49 capability cards · 10 categories · 3 blueprints · 3 library recipes ·
22 verbs · 18 classified failure signatures · 23-class evaluation failure taxonomy · 7-dimension
agent scorecard.

*Stage and card counts come from the code and the knowledge files — read them from
`discover` and `knowledge/cards/`, do not trust a number typed into prose. The conformance
audit fails on an **orphan card** (a card whose stage no longer exists) but only **reports** an
**uncarded stage**, so the two counts can legitimately diverge: a new stage is plannable before
its card lands. Treat any count here as of its writing.*
