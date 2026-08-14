# Audio Curation Agent — Design Source Document

**Purpose:** knowledge source for Claude Design to generate the presentation.
**Audience of the final deck:** technical + semi-technical, **no prior NeMo Curator knowledge**.
**Tone:** simple, explainable, confident. Explain *why* before *how*.

Structure below maps 1:1 to the intended slide sections.

---

# SECTION 1 — The Audio Data Curation Problem

### The one-line problem

> **Raw audio is not training data. Someone has to turn one into the other.**

### What "raw audio" actually looks like

Teams collect audio from many places — call recordings, podcasts, field recordings, public
datasets, scraped media. What arrives is inconsistent in almost every dimension:

- **Mixed quality** — studio-clean next to noisy phone recordings
- **Mixed formats** — different sample rates, mono vs stereo, different codecs
- **Mixed content** — speech, silence, music, background noise, overlapping speakers
- **Mixed length** — 3-second clips next to 2-hour recordings
- **Mixed labelling** — some files have transcripts, most do not
- **Unknown composition** — nobody knows what is actually in the folder

### Why this blocks model training

Speech and audio models (ASR, TTS, audio-language models) are **extremely sensitive to data
quality**. Feeding raw audio in directly produces models that are worse, not better.

So before training, every team has to answer the same questions:

| Question | The curation task it implies |
| --- | --- |
| Which clips are clean enough? | Quality scoring and filtering |
| Which are one speaker vs many? | Speaker diarization |
| Where is the speech, where is silence? | Voice activity detection, segmentation |
| Are they all the same format? | Resampling, channel conversion |
| What was said? | Transcription |
| Are the transcripts trustworthy? | Error-rate scoring and filtering |
| How much data survived, and what did I lose? | Reporting and verification |

### Why this is genuinely hard

- **It is not one step.** It is a *pipeline* of many dependent steps.
- **Order matters.** Some steps destroy what later steps need.
- **It is expensive.** Real corpora mean hours of GPU time per pass.
- **Mistakes are found late.** A wrong decision at step 2 surfaces after step 6 has been paid for.
- **It is repeated constantly.** Every new dataset, every new model, every new quality bar.

> **Design note — Slide framing:** open on the *mess* (a visual of inconsistent audio), then the
> *questions*, then land on: "every team rebuilds this same pipeline, by hand, every time."

---

# SECTION 2 — Audio DataVerse

> ⚠️ **CONTENT SLOT — animation already exists. Insert here.**

**Role of this section in the story:** this is the transition from *problem* to *platform*.
Section 1 establishes the pain. Section 2 introduces Audio DataVerse as the answer at the
product level. Section 3 then zooms into the engine underneath it.

**Design instruction for this slide:**

- Full-bleed animation, minimal text overlay
- One headline + at most one supporting line
- No bullet lists competing with the animation
- This is a **breath** slide — visual, not dense

**Narrative bridge to write on/after the animation:**

> "Audio DataVerse is how we deliver audio curation. Underneath it sits NeMo Curator — and that
> is where the agentification work happened."

---

# SECTION 3 — NeMo Curator: Context

### What NeMo Curator is (in one paragraph)

NeMo Curator is NVIDIA's open-source **data curation framework**. It handles the "turn raw data
into training data" problem across modalities — text, image, video, and **audio**. It is built
to run at scale on distributed GPU infrastructure.

### The audio stack

NeMo Curator's audio stack provides the building blocks for every curation task listed in
Section 1 — plus the runtime to execute them over large datasets on GPUs.

### What a "module" is

> **A module (stage) is one self-contained unit of audio processing.**

Each module:

- Takes audio records in, does **one job**, passes records out
- Is independently configurable (thresholds, models, output locations)
- Can be chained with other modules to form a **pipeline**

Examples in plain language:

| Module | What it does |
| --- | --- |
| Resample | Convert audio to a target sample rate and channel count |
| Voice Activity Detection | Find where speech actually is |
| Quality scorer | Predict how clean/natural a clip sounds |
| Diarization | Work out who spoke when |
| ASR | Transcribe speech to text |
| Error-rate scorer | Compare a transcript against a reference |
| Manifest writer | Write the final curated dataset out |

### The scale of the stack

> ## **NeMo Curator currently has 46 audio modules**
> organised into **10 functional categories**

Categories: **ingest · preprocess · segment · diarize · transcribe · text normalization ·
quality · filter · audio-language-model data building · export**

### How a pipeline is built today

A user writes Python: picks modules, puts them in an order, configures each one, and runs the
pipeline on a Ray-based distributed executor.

> **Design note:** show the 46 modules as a **grid or category cloud**, not a list. The visual
> point is *abundance* — which sets up the next section's point: abundance is also the problem.

---

# SECTION 4 — Problems in the Current Audio Stack

> **Framing line:** *"The modules were never the problem. Everything around them was."*

### Problem 1 — Discovery

- 46 modules across 10 categories
- Multiple modules do overlapping jobs — **three different quality scorers**, **two different
  diarizers**
- Nothing tells a new user which one fits their goal
- Choosing correctly requires already understanding the whole stack

### Problem 2 — Compatibility

- Modules communicate through **named fields** in a shared record
- If module 4 reads a field that nothing upstream produces → the pipeline **crashes**
- It crashes at **runtime**, on the GPU, *after* modules 1–3 have already been paid for
- Nothing checks this in advance

### Problem 3 — Ordering

- Some orderings are structurally required, and undocumented:
  - Diarization needs *continuous* audio → it must not follow a module that chopped the file up
  - A filter must come *after* the module that produces the field it filters on
  - Cheap CPU filters should precede expensive GPU ones, or you pay for data you were going to drop
- This knowledge lived in people's heads, not in the API

### Problem 4 — Configuration

- Every filter has a numeric threshold
- The number is meaningless without knowing the metric's **scale** and **direction**
- "Clean" has to become something like `mos_threshold: 3.4` — and a new user has no way to know
  whether 3.4 is strict or lenient
- **Direction traps:** quality scores are *higher is better*; error rates are *lower is better*.
  Filtering the wrong side **silently keeps exactly the data you wanted to drop**

### Problem 5 — Meaning traps

- Field names are misleading at pipeline boundaries
- Example: a field called `num_speakers` sounds per-clip. But after a module that splits one
  record into many, each row is a **segment**, not a clip
- Filtering it there answers a **different question than the user asked** — and nothing errors

### Problem 6 — Environment fragility

- GPU driver vs CUDA toolkit mismatches
- Missing system dependencies (ffmpeg)
- Distributed **workers** having a different environment than the **driver**
- These fail with cryptic errors that look like code bugs but are setup problems

### Problem 7 — Cost of getting it wrong

- A full run is **hours of GPU time**
- There is no "try it small first"
- Mistakes are discovered at the end, after the money is spent

### Problem 8 — Repeated work

- "Now also add transcripts" traditionally means **re-running everything**
- Hours of already-completed transcription or scoring get paid for twice

### The summary slide

> **NeMo Curator gave users powerful capability.**
> **It did not give them access to that capability.**
>
> Using it well required already being an expert in it.

> **Design note:** this section is the emotional core of the "why". Consider one icon per
> problem, 8 tiles. Each tile = 1 short title + 1 consequence line. Keep it scannable.

---

# SECTION 5 — Agentifying the NeMo Curator Audio Stack

> **Framing line:** *"Before an agent can use the stack, the stack has to be usable by an agent."*

The agentification work happened in two foundational steps **before** any agent existed.

---

## Step 1 — Making modules agent-ready and discoverable

### The problem being solved

An agent cannot use a module it cannot **find**, and cannot chain a module whose inputs and
outputs it cannot **see**. Previously both facts lived only in Python source code.

### What we added

**(a) A machine-readable contract on every module.** Each module now declares:

| Declaration | Meaning |
| --- | --- |
| **Reads** | Exactly which named fields it consumes |
| **Writes** | Exactly which named fields it produces |
| **Cardinality** | Does it produce one record out per record in, many (fan-out), fewer (filter), or aggregate? |
| **Gates** | Honest side-effect flags: needs GPU · writes to disk · needs ffmpeg · needs internet on first run |

**(b) A discovery mechanism.** A registry that can answer, without reading any source code:

- What modules exist?
- What category is each one in?
- What does each one do, in one line?
- What is this specific module's full contract?

### Why this matters

- **Compatibility becomes checkable.** "Does module 4's input exist?" is now a computation, not a
  runtime surprise.
- **The catalog becomes the boundary.** The agent can only ever reference a module that actually
  exists and is registered.
- **Adding a module is enough.** A new module ships its contract and becomes usable by the agent
  with **no changes to the agent itself**.

### The design principle that made adoption possible

> **Every new declaration defaults to today's behaviour.**
> Agent-readiness never changes how a module runs in an existing pipeline.

---

## Step 2 — Capability cards

### The gap that remained

A contract makes a module **mechanically connectable**. It does **not** say whether the module is
**appropriate**.

Nothing in the code tells you:

- that one quality scorer gives a single overall score while another breaks quality into seven
  separate dimensions
- that a score of 4.0 means "studio grade" and 2.5 means "poor"
- that a diarizer's speaker labels are *per-recording cluster IDs*, not real identities
- that a particular module downloads a model on first run

### What a capability card is

> **One file per module holding the facts source code cannot express.**
> There are **46 cards — one for every module.**

### What is inside a capability card

| Field group | What it holds |
| --- | --- |
| **Identity** | Module name, category, one-line summary, capability tags |
| **Model info** | Which model it runs, and a **pinned version** so a silent model swap is detectable |
| **Constraints** | Real hard limits — supported sample rates, maximum speakers, fixed batch sizes |
| **Resources** | CPU, GPU memory, host memory, whether GPU is optional, cost profile |
| **Use cases** | Explicit **good for** / **avoid for** lists |
| **Composition** | Typical upstream and downstream neighbours — idiomatic ordering |
| **Metrics** | The metric's scale, its **direction** (higher-better vs lower-better), valid range, and **outcome anchors** — e.g. "4.0–5.0 = studio" |
| **Presets** | Named bundles, e.g. "TTS reference", "ASR fine-tune" |
| **Comparison** | The tie-breaker block used when two modules overlap: language support, accuracy, latency, config complexity, known limitations, and an explicit **"choose when"** |
| **Semantic facts** | For each externally used output: its **meaning**, **unit**, **provenance**, **scope/granularity**, how it **propagates** across splits and merges, and at least one **counterexample** (a tempting wrong interpretation) |
| **Honesty tiers** | Every fact group declares how it was established: `mechanical` (from code), `measured` (from a real run), or `best_guess` (author judgment) |
| **Caveats & notes** | Known limitations in plain language |

### Two rules that make cards trustworthy

**Rule 1 — Never fabricate.**
An unknown value is left blank with a `TODO`. **A wrong fact is worse than a missing one**,
because the agent trusts cards absolutely. A blank makes the agent ask; a wrong value makes it
confidently wrong.

**Rule 2 — Cards are automatically checked against the code.**
A card that claims a parameter the module does not have, an invalid metric direction, or a model
without a pinned version **fails an automated gate**. Cards cannot silently drift away from
reality.

> **Design note:** show a capability card as an annotated visual — a card graphic with 4–5
> callouts pointing at the most storytelling-friendly fields: *metrics/anchors*, *comparison /
> choose-when*, *semantic facts*, *honesty tiers*.

---

# SECTION 6 — The Four Pillars

> **Framing line:** *"With modules discoverable and described, the agent itself rests on four
> pillars."*

## The four pillars

| # | Pillar | One-line role |
| --- | --- | --- |
| 1 | **LLM** | Understands intent and decides **direction** |
| 2 | **Deterministic Core** | Proves, gates, and executes — **facts, not opinions** |
| 3 | **Knowledge Base** | The ground truth the LLM reasons over |
| 4 | **Reuse Mechanism** | Never pay twice for the same computation |

---

## 6.1 — Flow diagram: how the four pillars answer a user query

> **DIAGRAM 1 — "The Four Pillars"**
> Design intent: show that the LLM sits *on top*, never touches execution directly, and is fed
> by the Knowledge Base at every reasoning step.

```
                          ┌──────────────────────────────┐
        User query  ─────▶│         1 · LLM              │
   "clean single-speaker  │                              │
    16 kHz + transcripts" │  understand intent           │
                          │  choose direction            │
                          │  select modules              │
                          │  judge meaning               │
                          │  explain & ask               │
                          └──────┬────────────▲──────────┘
                                 │            │
                    proposes     │            │  facts, verdicts,
                    a Recipe     │            │  evidence, refusals
                    (never code) │            │
                                 ▼            │
   ┌─────────────────────┐  ┌────────────────────────────┐
   │  3 · KNOWLEDGE BASE │─▶│   2 · DETERMINISTIC CORE   │
   │                     │  │                            │
   │  46 capability cards│  │  discover  · describe      │
   │  category taxonomy  │  │  profile   · doctor        │
   │  module contracts   │  │  resolve   · validate      │
   │  blueprints         │  │  plan      · smoke         │
   │  ordering patterns  │  │  gate      · run           │
   │  failure taxonomy   │  │  verify    · report        │
   └─────────────────────┘  └──────┬─────────────▲───────┘
            ▲                      │             │
            │                      │ checks      │ publishes
            │ grounds every        │ before      │ completed
            │ LLM decision         │ running     │ work
            │                      ▼             │
            │              ┌────────────────────────────┐
            └──────────────│   4 · REUSE MECHANISM      │
                           │                            │
                           │  content-addressed record  │
                           │  of every completed step   │
                           │  "already done?" → skip    │
                           └──────────────┬─────────────┘
                                          ▼
                                  NeMo Curator
                             46 modules · Ray executor
                                          │
                                          ▼
                              ✅ Curated audio dataset
                                 + evidence report
```

### The single sentence that explains the whole diagram

> ## 🔑 **The LLM proposes. The deterministic core disposes.**

**Two structural guarantees:**

1. **The LLM never writes code.** It proposes a **Recipe** — a list of real module names and
   parameters. An invented name simply does not resolve.
2. **The LLM cannot skip a gate.** The gates live *inside* the core's functions, not in the
   prompt — so a weaker model, a direct script, or a command-line call hits the same refusals.

> **Design note:** this is the **hero diagram** of the deck. Give it a full slide. Consider
> animating it in 4 builds — one pillar at a time, in order 1 → 3 → 2 → 4.

---

# SECTION 7 — Inside Each Pillar

---

## Pillar 1 · LLM

### Purpose

> **Decide the *direction* of the workflow. Nothing else.**

### What is inside

| Responsibility | What it means |
| --- | --- |
| **Intent understanding** | Turn "clean single-speaker data" into a structured goal |
| **Clarifying questions** | Ask 1–2 questions — at the **outcome layer** only ("studio, broadcast, or general?"), never about internal parameters |
| **Success contract** | Define what "done" means *before* anything runs |
| **Routing** | Prune 10 categories → shortlist → read full cards for 2–3 finalists only |
| **Module selection** | Choose between overlapping modules using card facts, not name similarity |
| **Semantic judgment** | Decide whether a technically valid pipeline actually *means* what the user asked |
| **Explanation** | State trade-offs and rationale in plain language |
| **Presentation** | Present the plan and the evidence for human approval |

### What the LLM is explicitly **not** allowed to do

- ❌ Write or execute code
- ❌ Invent a module or parameter name
- ❌ Pick a threshold number itself
- ❌ Decide whether a pipeline is valid
- ❌ Decide whether to run at full scale
- ❌ Declare success
- ❌ Change the environment or install anything

> **Speaker note:** *"The LLM is a navigator, not a driver. It chooses the route. It never touches
> the steering wheel, the engine, or the brakes."*

---

## Pillar 2 · Deterministic Core

### Purpose

> **Own everything that must be provable, repeatable, and un-skippable.**
> There is **no LLM inside this pillar.** It is plain code.

### What is inside

| Capability | What it does |
| --- | --- |
| **Discovery** | What modules exist, what each one's contract is |
| **Data profiling** | Read the actual dataset — file count, sample rates, channels, transcripts present? |
| **Environment doctor** | Check the machine — GPU, driver/toolkit match, ffmpeg, dependencies, worker environment — and return **concrete fixes** |
| **Configuration resolution** | Turn an outcome word ("studio") into a concrete parameter using the card's anchors, with an audit trail |
| **Validation** | Prove the pipeline composes: real modules, valid parameters, every field produced before it is read, constraints honoured, required outputs producible |
| **Semantic evidence packet** | Assemble field lineage, meaning, and split/merge boundaries **for the LLM to review** (it assembles evidence; it does not judge) |
| **Resource planning** | Pick execution mode and check the pipeline fits the machine |
| **Bounded smoke test** | Run the real pipeline on ~10 items into a throwaway location |
| **Confirmation gate** | Refuse a full run without explicit approval, bound by a hash so **what was approved is exactly what runs** |
| **Execution** | Run the pipeline, monitor it, collect per-module metrics |
| **Failure diagnosis** | Classify errors against a taxonomy and return **grounded recovery options** |
| **Acceptance verification** | Check the result against the pre-agreed success contract, with an anti-optimism guard |
| **Reporting** | Counts, per-filter drop reasons, failure examples, output paths |
| **Safety guardrails** | Workspace path lock · secret and transcript redaction · nothing written before approval |

### Why this pillar exists

Every item above is either a **fact** or a **proof**. Facts must not be generated by a language
model — they must be looked up or computed. That is the entire justification for this pillar.

> **Speaker note:** *"If the answer can be computed, we compute it. Generating it would only add
> a chance of being wrong."*

---

## Pillar 3 · Knowledge Base

### Purpose

> **Give the LLM real facts to reason over, so it never has to recall or invent them.**

### What is inside

| Component | Count | What it provides |
| --- | --- | --- |
| **Capability cards** | 46 | Everything about each module that source code cannot express (Section 5) |
| **Module contracts** | 46 | Machine-readable reads / writes / cardinality / gates |
| **Category taxonomy** | 10 categories | The coarse routing tree the LLM prunes over first |
| **Blueprints** | 3 | Proven end-to-end pipeline shapes for known scenarios, with each module tagged *required* or *optional* |
| **Library recipes** | 3 | Fully working reference pipelines |
| **Composition patterns** | — | Ordering wisdom made explicit: "mono first", "cheap filters before expensive ones", "re-join segments before diarization" |
| **Failure taxonomy** | 18 classes | Symptom → likely cause → which layer → what to do |

### Key properties

- **Versioned, read-only YAML files** — not prompt text. Diffable, reviewable, testable.
- **Automatically gate-checked against the code** — knowledge cannot silently rot.
- **Served in three tiers** so the LLM reads only what it needs:
  **L0** category tree → **L1** one-line summaries → **L2** full cards for finalists only.

### Why the tiering matters

Dumping all 46 cards into context for every query would be slow, expensive, **and less
accurate** — the model would be reasoning over mostly irrelevant detail. Tiered retrieval is
cheaper *and* better.

> **Speaker note:** *"This is the difference between an agent that knows things and an agent that
> guesses things."*

---

## Pillar 4 · Reuse Mechanism

### Purpose

> **Never pay twice for the same computation — and never silently serve a stale result.**

### The problem it solves

Curation work is expensive and repetitive. "Now also add transcripts to this dataset" should not
mean re-paying for the hours of quality scoring and resampling already completed.

### What is inside

| Component | What it does |
| --- | --- |
| **Content-addressed step identity** | Each completed step gets a key derived from the data identity + the module + its meaningful parameters + code and model versions |
| **Chained identity** | Each step's key builds on the previous one — so changing something in the middle automatically invalidates only what comes after |
| **Artifact registry** | Every step that wrote output is recorded with what it produced, what it cost, and how long it took |
| **Completion markers** | Output is only reusable once atomically marked complete, with a content digest — a crashed half-run can never be mistaken for a finished one |
| **Trust tiers** | Data identity is graded; a weak-confidence match is marked **low trust** and defaults to a fresh run |
| **Reuse scan** | Before spending anything: is this exact work already done? |
| **Executable continuation** | Choosing "reuse" actually rewrites and runs the remaining pipeline — it does not just print advice |
| **Run records** | Provenance: what ran, on what, for what goal, when, and what it produced |

### The two rules of the reuse conversation

**Never silent.** Reuse is always disclosed. "You got yesterday's answer" is not a detail to bury
in a log line.

**Never nagging.** Nothing to reuse → no prompt at all. A *measured* trivial saving → just take
it and say so in the report. Low trust → show the weakness in plain language and pre-select
"fresh".

### An important boundary

> This is **memoization, not learning.**
> It skips work it can *prove* is identical. It never changes *what* the agent plans.
> There are no learned priors and no cross-session memory.

> **Speaker note:** *"Between a slow answer and a wrong one, this system fails towards slow — on
> purpose."*

---

# SECTION 8 — The Complete Agent Flow

> **DIAGRAM 2 — "User Query → Curated Dataset"**
> Design intent: a **vertical walkthrough** showing the query passing through every stage, with
> the owner of each step colour-coded (🟦 LLM · 🟩 CORE · 🟨 USER).

```
┌──────────────────────────────────────────────────────────────────────────┐
│  USER: "Give me clean, single-speaker, 16 kHz audio with transcripts."    │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  ▼
🟦 1 · UNDERSTAND
   Turn words into a goal + define the SUCCESS CONTRACT (what "done" means)
   Ask at most 1–2 outcome-level questions   →  "Studio, broadcast, or general?"
                                  ▼
🟩 2 · INSPECT
   Profile the DATA (files, sample rates, channels, transcripts present?)
   Check the MACHINE (GPU, drivers, dependencies)  →  report findings to the user
                                  ▼
🟦 3 · ROUTE                                        ┌─────────────────┐
   Prune 10 categories → shortlist → read full  ◀───│ KNOWLEDGE BASE  │
   cards for the 2–3 finalists only                 │ 46 cards        │
   Compare overlapping modules on card facts        └─────────────────┘
                                  ▼
🟩 4 · RESOLVE CONFIG
   "general quality"  ─────────────▶  mos_threshold: 3.4
   (from the card's anchors, with an audit trail — never a guessed number)
                                  ▼
🟦 5 · PLAN
   Emit a RECIPE  =  ordered list of real module names + parameters
                     + the embedded success contract
   ⚠️ No code is ever generated.
                                  ▼
🟩 6 · VALIDATE  ────────────────────────────────┐
   Real modules? Valid parameters?               │  ❌ fail
   Every field produced before it is read?       │  → returns the exact fix
   Constraints honoured? Outputs producible?     │  → loop back to step 5
   ✅ pass                                       │     (max 3 iterations)
                                  ▼              │
🟦 7 · SEMANTIC CRITIQUE  ◀── evidence packet ───┘
   "It composes — but does it MEAN what was asked?"
   For every filtered field: meaning · scope · provenance · effect · direction
   →  pass  /  revise  /  ask
                                  ▼
🟩 8 · REUSE SCAN
   Has this exact work already been done on this exact data?
   →  already done  /  partially done  /  fresh
                                  ▼
🟩 9 · SMOKE TEST
   Run the REAL pipeline on ~10 items, into a throwaway location
   →  retained / rejected counts · examples · errors · measured resource use
                                  ▼
🟨 10 · CONFIRM GATE
   Present: the plan · the critique · the smoke evidence · the scale & cost
            · the success contract, stating what each metric does NOT capture
   ⚠️ NOTHING has been written to disk yet.
   →  USER APPROVES
                                  ▼
🟩 11 · RUN
   Refuses unless handed the approved recipe's hash
   →  what was approved is byte-for-byte what runs
   Picks execution mode · starts the cluster if needed · monitors every module
                                  ▼
🟩 12 · VERIFY
   Check the frozen success contract against real output evidence
   →  met  /  not met  /  unverifiable  /  unachievable
   🛡️ Honesty guard blocks any silent weakening of an approved requirement
                                  ▼
🟩 13 · REPORT     +     🟩 14 · PUBLISH FOR REUSE
   Counts · per-filter drop reasons     Every completed step recorded
   · failure examples · output paths     so the next request can skip it
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ✅ Curated dataset  +  evidence report  +  reusable artifacts           │
└──────────────────────────────────────────────────────────────────────────┘

   ↺  Any failure at any step → DIAGNOSE: classify the error, return grounded
      recovery options — which the agent EXPLAINS but never applies on its own.
```

### The three gates worth calling out on this slide

> **Before anything runs at scale, three independent gates must pass:**
> **① Validate** (does it compose?) → **② Critique** (does it mean the right thing?) →
> **③ Smoke** (does it actually work on real data?) → then **④ the user confirms.**

> **Design note:** consider a simplified 6-step version for the main slide, with this full
> 14-step version as an appendix/backup slide. The colour coding (LLM / CORE / USER) is the most
> important visual signal — it makes the division of labour obvious at a glance.

---

# SECTION 9 — Why This Architecture?

> **Framing question for the slide:** *"Why not just give an LLM the docs and let it build the
> pipeline?"*

## 9.1 — What a purely LLM-driven architecture costs you

### ❌ Higher token usage

- A fully LLM-driven agent has to load **everything** into context to reason about anything:
  all 46 module descriptions, all parameter lists, all documentation
- It has to re-read that context on **every turn**
- Raw execution logs go back into the model for interpretation
- **The cost is not just money — it is accuracy.** A model reasoning over mostly irrelevant
  detail makes *worse* decisions, not better ones

> **Our approach:** tiered retrieval (10 categories → shortlist → 2–3 full cards) and **compact
> JSON verdicts** instead of raw logs. The model reads a small, relevant, structured slice.

### ❌ Inconsistent behaviour

- The same request produces a **different pipeline every time**
- Thresholds vary run to run
- Module choices vary run to run
- **Nothing can be approved, versioned, audited, or compared** — because there is no stable
  artifact to approve

> **Our approach:** the same request produces the same validated Recipe with the same resolved
> values and the same hash. A plan that can be reviewed, approved, and reproduced.

### ❌ Decision drift

This is the most dangerous one. Over a long interaction, a purely LLM-driven agent gradually
slides away from what was agreed:

- It quietly **lowers the success bar** when the target proves hard to hit
- It **adds modules** nobody asked for, because they seem helpful
- It **changes a threshold** after approval, then reports success against the new one
- It "**fixes**" an environment error by installing packages and switching devices
- It **declares success** based on the narrative it has built, not on evidence

Each individual step looks reasonable. The **accumulation** is a result nobody agreed to.

> **Our approach:** four structural anti-drift mechanisms —
> **① the approved plan is hash-bound** (change anything and approval is void)
> **② the success contract is frozen inside the approved artifact**
> **③ an honesty guard** blocks any dropped, downgraded, or relaxed requirement
> **④ diagnosis proposes but never applies** — the environment cannot be silently mutated

---

## 9.2 — So what did we constrain the LLM to?

> ## 🔑 **The LLM decides the *direction* of the workflow — not the code, not the configuration, not the execution.**

| The LLM **does** decide | The LLM **does not** decide |
| --- | --- |
| What the user actually wants | What modules exist |
| Which capabilities are needed | What a parameter is called |
| Which module fits best among alternatives | Whether the pipeline composes |
| Whether the plan *means* the right thing | What threshold "studio" is |
| What trade-offs to explain | What the machine can do |
| What to ask the user | Whether to run at full scale |
| How to present the result | Whether the criteria were met |
|  | Whether work can be skipped |

### 9.3 — Why each constraint exists

| Constraint | What it buys |
| --- | --- |
| **No code generation** | Nothing unreviewable or unbounded can ever execute. The Recipe is a short list of real names a human can read in 10 seconds and a machine can verify completely |
| **No invented names** | Module names resolve through a registry; parameters go to a real constructor. Hallucination becomes structurally unable to have consequences |
| **No self-chosen thresholds** | Values come from card anchors with an audit trail — repeatable and explainable |
| **No self-approved execution** | The confirm gate is inside the tool, not the prompt. It cannot be reasoned around |
| **No self-declared success** | Verification is computed against a frozen contract, immune to optimism |
| **No silent environment changes** | The user's machine is never mutated to make an error go away |
| **No silent reuse** | Reused work is always disclosed |

### 9.4 — The principle in one line

> ### **Facts get computed. Judgment gets reasoned.**
>
> The LLM is placed exactly where deterministic logic runs out — interpreting a vague request,
> weighing two reasonable modules, noticing that a valid pipeline answers the wrong question.
> Everywhere else, it is a lookup or a computation.

### 9.5 — The closing statement

> ## **We did not build an LLM that runs NeMo Curator.**
> ## **We made NeMo Curator's knowledge explicit enough that an LLM can be trusted with the parts of the job that need judgment — and structurally prevented from touching the parts that don't.**

> **Design note:** Section 9 should feel like the *payoff*. Suggested build: show the three costs
> (tokens / inconsistency / drift) as a red column, then the constraint table as a green column,
> then land the closing statement on its own full slide with no other content.

---

# APPENDIX — Facts & Figures for the Designer

| Fact | Value |
| --- | --- |
| Audio modules in NeMo Curator | **46** |
| Functional categories | **10** |
| Capability cards | **46** (one per module) |
| Blueprints (proven pipeline shapes) | **3** |
| Reference recipes | **3** |
| Classified failure signatures | **18** |
| Pillars | **4** |
| Steps in the full agent flow | **14** |
| Gates before a full-scale run | **3** + user confirmation |

### Category names (for a visual grid)

`ingest` · `preprocess` · `segment` · `diarize` · `transcribe` · `text normalization` ·
`quality` · `filter` · `audio-language-model data` · `export`

### Recurring visual motifs to keep consistent across slides

- 🟦 **LLM** — one colour, used everywhere the model decides
- 🟩 **Deterministic Core** — a second colour, used everywhere code decides
- 🟨 **User** — a third colour, used only at the confirmation gate
- 🔒 **Gate icon** — reused at validate, critique, smoke, and confirm
- ♻️ **Reuse icon** — reused wherever prior work is skipped

### The three phrases that should repeat in the deck

1. **"The LLM proposes. The deterministic core disposes."**
2. **"Facts get computed. Judgment gets reasoned."**
3. **"The LLM decides direction — not code, configuration, or execution."**
