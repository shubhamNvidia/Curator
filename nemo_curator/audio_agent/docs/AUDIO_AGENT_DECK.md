# Audio Curation Agent — Agentification

**Presentation outline** · 14 slides · ~15 min
Reference/source of truth: `AUDIO_AGENT_KNOWLEDGE.md`

> 🎤 = speaker key note (say this out loud)
> Legend: **LLM** = model decides · **CORE** = deterministic code decides

---

## 1 · The Problem

- Raw audio is **not** training data.
- Turning one into the other = many decisions: clean enough? one speaker? right rate? what was said? what do I drop?
- NeMo Curator ships the tools: **47 audio stages** across **10 categories**.
- Users assemble them **by hand**, in Python.

> 🎤 **"Curator was never missing capability. It was missing access to capability."**

---

## 2 · Why It's Hard for a Newcomer

| Problem | Reality |
|---|---|
| **Discovery** | 47 stages. 3 quality scorers. 2 diarizers. Which one? |
| **Compatibility** | Stage 4 reads a field nobody writes → crash **after** paying for stages 1–3 |
| **Ordering** | Diarization needs continuous audio. Cheap filters before expensive ones. Undocumented. |
| **Configuration** | "Clean" → `mos_threshold: 3.4`. Is 3.4 strict or lenient? |
| **Direction traps** | Quality: higher is better. Error rate: **lower** is better. Wrong side = keep exactly what you wanted to drop. |
| **Environment** | GPU driver vs CUDA toolkit, ffmpeg, Ray workers ≠ driver env |
| **Cost** | Full run = hours of GPU. Mistakes found at the end. |
| **Repetition** | "Also add transcripts" → re-run everything |

> 🎤 **"Every one of these is a way to be confidently wrong. The agent is 8 layers that each remove one."**

---

## 3 · Why Not Just Wrap an LLM?

- ❌ Invents stages and parameters that don't exist
- ❌ Writes ad-hoc Python — unreviewable, unbounded
- ❌ Can't prove compatibility → plausible pipeline, runtime crash
- ❌ Guesses thresholds with no idea of scale or direction
- ❌ Can't see the machine → plans 4 GPUs on a 1-GPU box
- ❌ Not repeatable → nothing to approve or audit
- ❌ Declares success with **no evidence**
- ❌ Silently moves the goalposts (asked 4.0, delivered 3.2, reported "done")

> 🎤 **"An agent that is usually right and occasionally confidently wrong is worse than no agent — because the human stops checking."**

---

## 4 · The Core Idea

# 🔑 The LLM proposes. The deterministic core disposes.

Two structural guarantees:

1. **The LLM never emits code.** It emits a **Recipe** — a list of real stage names + parameters.
   → Zero `exec`/`eval`/generated `.py` anywhere in the execution path.
   → An invented name simply **does not resolve**.
2. **The LLM cannot skip a gate.** Gates live **inside the tools**, not in the prompt.
   → A weaker model, a CLI call, or a script hits the same refusals.

> 🎤 **"We don't ask the model not to hallucinate. We make hallucination structurally unable to have consequences."**

---

## 5 · Architecture — Two Planes

```
   User's words
        │
        ▼
┌───────────────────────────────────────────────┐
│  LLM PLANE  (host model + written skill)      │
│  interpret · route · select · critique · ask  │
└──────────────────┬────────────────────────────┘
                   │  JSON tool calls
                   ▼
┌───────────────────────────────────────────────┐
│  DETERMINISTIC CORE   (no LLM inside)         │
│  KNOWLEDGE  · discover / cards / context      │
│  SITUATION  · data profiler / doctor          │
│  CONFIG     · resolve                         │
│  VALIDATE   · validate + semantic packet      │
│  EVIDENCE   · smoke / calibrate               │
│  SAFETY     · confirm gate / workspace lock   │
│  EXECUTE    · run / diagnose                  │
│  RESULTS    · report / verify                 │
│  REUSE      · reuse-scan / continue           │
└──────────────────┬────────────────────────────┘
                   ▼
      NeMo Curator · 47 stages · Ray executor

  Knowledge = versioned, read-only YAML
  47 cards · taxonomy · blueprints · patterns · failure taxonomy
```

- Same core exposed 3 ways: **CLI · MCP server · Python SDK**. All JSON.

> 🎤 **"There is no model inside the core. It's testable, cheap, deterministic, and works with whatever host you already have."**

---

## 6 · The Flow

```
interpret + define success
   → inspect (data + environment)
   → route  L0 categories → L1 one-liners → L2 full cards
   → resolve outcomes to parameters
   → plan (Recipe)
   → VALIDATE          ← mechanical
   → SEMANTIC CRITIQUE ← intent      (pass / revise / ask)
   → reuse-scan
   → SMOKE             ← ~10 items, real models
   → present + CONFIRM ← user says yes
   → run
   → verify + report
```

**Three gates before anything runs at scale:** validate → critique → smoke → confirm.

> 🎤 **"Nothing is written to disk before the confirm gate. A gate you've already prepared the ground for is not a gate."**

---

## 7 · The Agentification Layers

| # | Layer | Removes this failure |
|---|---|---|
| 0 | **Stage contracts** | "Nobody knows what a stage reads or writes" |
| 1 | **Capability cards** (47) | "Which of these 3 scorers? What does 4.0 mean?" |
| 2 | **Knowledge index** (L0→L1→L2) | Token blowout + random stage picks |
| 3 | **Intent + success contract** | "Success" meaning "it exited zero" |
| 4 | **Data profile + `doctor`** | Setup problems that look like code bugs |
| 5 | **Outcome → parameter** (`resolve`) | Invented thresholds, inverted filters |
| 6 | **Recipe IR** | Generated code |
| 7 | **Deterministic validation** | Runtime crashes that cost GPU hours |
| 8 | **Semantic review** ⭐ | **Valid pipelines that answer the wrong question** |
| 9 | **Resource planner** | Aborted runs, wrong execution mode |
| 10 | **Bounded smoke + calibration** | Discovering mistakes at full scale |
| 11 | **Confirm gate + guardrails** | Silent full-scale runs, leaked secrets |
| 12 | **Execution + monitoring** | Ray setup pain, miscounted results |
| 13 | **Diagnose + recovery** | Agents that "fix" things by mutating your machine |
| 14 | **Acceptance + honesty guard** | Declaring success by lowering the bar |
| 15 | **Reuse (memoization)** | Paying twice for the same GPU hours |
| 16 | **Provenance / run records** | "What did we already do to this corpus?" |
| 17 | **CLI · MCP · SDK** | One core, three doors |
| 18 | **Two-plane evaluation** | "Did it run?" as the only test |

> 🎤 **"Read this as a list of failure modes, not features. Each row is a specific way to be wrong that is now structurally impossible."**

---

## 8 · ⭐ The Layer People Underestimate

### Validation proves a pipeline **composes**. It cannot prove it **means** the right thing.

**The trap — "Keep only single-speaker clips":**

```yaml
SpeakerSeparationStage          # ✅ real stage
PreserveByValueStage:
  input_value_key: num_speakers # ✅ real field
  operator: eq, target: 1       # ✅ valid filter
```

**Validates 100% green. Wrong twice:**

- ❌ `num_speakers` = speakers found in the **original whole clip** — a parent-level aggregate.
  After separation each row is **one speaker's stream**. Wrong granularity.
- ❌ Separation **transforms the audio**. The user asked to *select* clips, not *mangle* them.

**Correct:** diarize → keep rows with distinct-speaker count 1. No separation.

**The fix — two parts:**

1. **CORE** builds a deterministic evidence packet: real field lineage, fan-out seams, card prose (meaning · unit · provenance · **scope** · propagation · counterexamples). *It does not judge.*
2. **LLM** must answer 5 questions per filtered field before smoke:
   **meaning** · **scope/granularity at this point** · **provenance** · **transform vs select** · **direction**
   → returns `pass` / `revise` / `ask`, bound to the recipe hash.

> 🎤 **"The plumbing is green either way. Only 'what does this field mean, for which entity, at this point?' catches it. That's not a rule you can write — it's judgment. So it's the LLM's job, with the facts handed to it."**

> 🎤 **Architectural rule:** *"We never add module-specific intent rules to the core. Put the fact in a card; let the critic reason. Rules produce an always-incomplete table of special cases."*

---

## 9 · Who Decides What

| Concern | Owner |
|---|---|
| Understanding an ambiguous request | **LLM** |
| Choosing between 2 overlapping modules | **LLM** — from card facts |
| Does this valid recipe mean the right thing? | **LLM** — from the evidence packet |
| Explaining trade-offs, asking the user | **LLM** |
| What stages exist / what params are called | **CORE** — facts aren't generated |
| Does this pipeline compose? | **CORE** — provable |
| What threshold is "studio"? | **CORE** — from card anchors |
| What can this machine do? | **CORE** — measurable |
| Run at full scale? | **CORE** gate + **USER** decision |
| What actually happened? | **CORE** — evidence, not narration |
| Were the criteria met? | **CORE** — immune to optimism |
| Can we skip this work? | **CORE** — provable, not plausible |

> 🎤 **"Facts get computed. Judgment gets reasoned. Every time we blurred that line, we got a bug."**

---

## 10 · Key Design Decisions

1. **Host-driven, no embedded LLM** → testable, cheap, provider-agnostic
2. **Declarative Recipe, never generated code** → the single biggest safety decision
3. **Knowledge as versioned YAML, not prompt text** → diffable, testable, doesn't drift
4. **Cards are gate-checked against the code** → a card claiming a fake param **fails CI**
5. **Honesty tiers on every fact** → `mechanical` / `measured` / `best_guess`; a missing fact beats a wrong one
6. **Three hashes, three jobs** → approval integrity · reuse identity · success bar
7. **Guardrails in the tools, not the prompt** → a prompt rule only protects well-behaved callers
8. **A guess never blocks the measurement that would replace it** → VRAM warns, never gates
9. **Diagnosis proposes; it never applies** → no silent installs, device switches, or retries
10. **Success defined before, verified after, with an honesty guard** → the bar cannot be quietly lowered

> 🎤 **On #6:** *"Changing a batch size changes nothing about the output bytes — but it does change what the user approved. One hash for both jobs meant either unsafe approval, or reuse that never fired."*

> 🎤 **On #8:** *"Gating on a VRAM guess would have blocked the smoke test that measures real VRAM. That one principle killed a whole class of 'the agent refuses to try'."*

---

## 11 · What We Learned

1. **LLM reasoning is valuable where rules run out — dangerous where rules exist.**
   Interpreting "clean single-speaker data" ✅ · recalling a parameter name ❌
2. **Mechanically valid ≠ semantically correct.** That gap is where the real bugs live.
3. **Structured knowledge > prose. Gate-checked knowledge > structured knowledge.**
4. **A missing fact is safer than a wrong one.** The agent trusts cards absolutely.
5. **Validation must precede execution — and one kind isn't enough.** Mechanical, semantic, empirical, acceptance.
6. **Agents must see the environment and must NOT fix it.** Detect → explain → propose → ask.
7. **Never state an absence as a fact when the observation could be blocked.** "No GPU" from a sandbox is not a hardware fact.
8. **Silence is not evidence.** An unmeasured stage is not a cheap stage.
9. **Launching a pipeline isn't the job — analysing it is.** Otherwise it's a fancy `subprocess.run`.
10. **Design for the honest failure.** `unverifiable` and `unachievable` are first-class outcomes — an agent that can only say "success" or "error" will say "success".
11. **Flexibility comes from generic mechanisms, not more cases.** No metric name in the resolver. No module name in the semantic layer. Every extension = one card or one decorated function.
12. **Write the boundaries down next to the capability.** Overclaiming is what makes people stop checking.

> 🎤 **"Two real incidents shaped this deck. An agent deleted a user's file before the confirm gate. And our first reuse system was correct, shipped, and fired almost never. Both became structure, not warnings."**

---

## 12 · Impact

| | Before (manual Curator) | After (Audio Agent) |
|---|---|---|
| **Getting started** | Learn 47 stages + field conventions | Describe the outcome |
| **Choosing a module** | Guess, or ask an expert | Compared on card facts, trade-off explained |
| **Thresholds** | Guess a number, hope the direction is right | Say "studio"; resolved from anchors + audit trail |
| **Compatibility** | Found at runtime, on the GPU | Proven before anything runs, fix named |
| **Intent correctness** | Nothing checks it | Mandatory grounded critique |
| **Cost of a mistake** | Hours of GPU | Caught at validation, or a 10-item smoke |
| **Running at scale** | Whatever you typed, immediately | Refused until approved, hash-bound to the plan |
| **Repeating work** | Full re-run | Provably identical work skipped, with disclosure |
| **Did it work?** | Read output and judge | Verdict vs a pre-agreed contract |
| **Explainability** | Reconstruct from code | Every stage justified, every value traced |

**Outcomes:** less engineering effort · faster pipelines · fewer config mistakes · consistent, reviewable plans · reused GPU hours · explainable decisions · **usable by people who've never heard of Curator**.

> 🎤 **"An easy one to miss: this made the stages better for humans too. Contracts, cards, and the failure taxonomy forced the codebase to state things it had only ever implied — knowledge that existed nowhere before, not even in someone's head."**

---

## 13 · Limits & What's Next

**Honest limits**
- Knowledge is hand-authored; a new stage needs a card
- Semantic correctness depends on card quality + host judgment
- Reuse resumes only from **persisted** output (disclosed, never hidden)
- Dataset identity is metadata-backed, not a full content hash
- Recipes are linear chains, not general graphs

**Non-goals today:** no cross-session learning · no learned priors · no embedded planner model · no automatic environment remediation · no agent-authored Python

**Designed, not built:** content-digest identity · per-file deltas (process only new files) · superset reuse · a `why-rerun` verb · row-level lineage across fan-out

**Natural extension:** the architecture is **modality-agnostic** — nothing in the core is audio-specific except the knowledge files.

> 🎤 **"Stating the boundaries is the credibility. Every doc in this system says what its layer does NOT prove."**

---

## 14 · Close

**Three moments to remember:**

1. 🪤 **The single-speaker trap** — valid ≠ correct
2. 🗑️ **The deleted file** — a real incident became architecture, not a warning
3. ♻️ **The reuse rewrite** — fixed by changing the *question*, not tuning the answer

> 🎤 **Closing line:**
> **"We didn't build an LLM that runs Curator. We made Curator's knowledge explicit enough that an LLM can be trusted with the parts of the job that need judgment — and structurally prevented from touching the parts that don't."**

---

## Appendix · One-Slide Cheat Sheet

**The verbs**
`discover` `catalog-tree` `cards` `describe` `context` · `doctor` `diagnose` · `resolve` · `validate` · `smoke` `calibrate` · `run` · `report` `verify` · `reuse-scan` `continue` `runs`

**The 10 golden rules**
1. Never invent a stage or parameter name
2. Zero silent full-scale runs
3. Nothing written before approval
4. Evidence-only claims
5. Define success up front, verify after, never relax a `must`
6. Ask at the outcome layer, never about internals
7. Refuse subjective labelling + training/deploy requests
8. Diagnose the environment; never fix it silently
9. Never reuse silently; never nag
10. **Green validation is not intent approval**

**By the numbers**
47 stages · 47 cards · 10 categories · 3 blueprints · 18 classified failures · ~17.4k lines core · ~11.3k lines tests · 23-class eval taxonomy
