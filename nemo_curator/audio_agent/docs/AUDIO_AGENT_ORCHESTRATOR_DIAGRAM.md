# Orchestrator Diagram — Design Specification

**Purpose:** knowledge source for Claude Design to build the **main architecture diagram**.
**Diagram title:** *"The Host LLM as Orchestrator"*
**Where it goes in the deck:** the hero slide, replacing the old four-pillar block diagram.

---

## 1 · The idea the diagram must communicate

> ## 🔑 The Host LLM is the **orchestrator**, not the worker.
> It **operates inside a Skill**. It **queries** three specialist blocks.
> It is the **only** thing the user ever talks to.

Four things a viewer should understand within 5 seconds of seeing it:

1. **The orchestrator sits at the centre.** Everything routes through it.
2. **It runs inside the Skill.** The Skill is the operating procedure that governs how it
   behaves — the model is powerful, but it is not improvising.
3. **It talks to three blocks** — each answers a different kind of question.
4. **The user channel is exclusive.** The user speaks only to the orchestrator, and the
   orchestrator is the only thing that speaks back. No block reaches the user directly.

---

## 2 · Layout: hub and spoke, with the orchestrator inside the Skill

**Shape:** the Skill is a container. The orchestrator sits **inside** it. Three spokes run down
to the specialist blocks, and a dedicated two-way channel runs up to the user.

**Why this shape:** it proves two points at once — the orchestrator is structurally in the middle
of every path, *and* it is operating within a defined procedure rather than freestyling.

```
                    ┌─────────────────────────────────┐
                    │            👤 USER              │
                    └───────┬─────────────────▲───────┘
                            │                 │
              request ·     │                 │   questions · plans ·
              answers ·     │                 │   evidence · results ·
              approval ·    │                 │   explanations
              feedback      │                 │
                            ▼                 │
   ╔════ 📜 SKILL — the operating procedure ══════════════════════════╗
   ║                                                                  ║
   ║   the loop · golden rules · routing discipline ·                 ║
   ║   question discipline · refuse list · review checklists          ║
   ║                                                                  ║
   ║   ┌──────────────────────────────────────────────────────────┐   ║
   ║   │                                                          │   ║
   ║   │        🧠  ORCHESTRATOR  —  HOST LLM                     │   ║
   ║   │                                                          │   ║
   ║   │  • Understands what the user actually wants              │   ║
   ║   │  • Decides the DIRECTION of the workflow                 │   ║
   ║   │  • Chooses which modules fit                             │   ║
   ║   │  • Judges whether the plan MEANS the right thing         │   ║
   ║   │  • Explains trade-offs, asks, presents                   │   ║
   ║   │  • Reacts to every verdict and re-plans                  │   ║
   ║   │                                                          │   ║
   ║   │  ✗ never writes code    ✗ never picks thresholds         │   ║
   ║   │  ✗ never approves its own run                            │   ║
   ║   └──────────────────────────────────────────────────────────┘   ║
   ╚═══════╦═══════════════════╦═══════════════════════╦══════════════╝
           ║                   ║                       ║
     asks  ║             asks  ║                 asks  ║
           ▼                   ▼                       ▼
  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────┐
  │ 📚 KNOWLEDGE   │  │ ⚙️ DETERMINISTIC │  │ ♻️ REUSE           │
  │    BASE        │  │    CORE          │  │    MECHANISM       │
  │                │  │                  │  │                    │
  │ "What exists   │  │ "Is this valid,  │  │ "Have we already   │
  │  and what is   │  │  and what        │  │  done this?"       │
  │  it good for?" │  │  happened?"      │  │                    │
  │                │  │                  │  │                    │
  │ 46 capability  │  │ validate · smoke │  │ step fingerprints  │
  │ cards          │  │ profile · doctor │  │ artifact registry  │
  │ module         │  │ resolve · plan   │  │ completion markers │
  │ contracts      │  │ gate · run       │  │ trust tiers        │
  │ taxonomy       │  │ verify · report  │  │ run history        │
  │ blueprints     │  │ diagnose         │  │                    │
  │ patterns       │  │ guardrails       │  │                    │
  │ failure taxo.  │  │                  │  │                    │
  └───────┬────────┘  └────────┬─────────┘  └─────────┬──────────┘
          │                    │                      │
          └──── facts ─────────┴──── verdicts ────────┘
                    evidence · refusals · savings
                               │
                               ▲
                   all answers return to the orchestrator
                         (never to the user)
                               │
                               ▼
                   ┌───────────────────────────┐
                   │   NeMo Curator            │
                   │   46 modules · Ray        │
                   └───────────────────────────┘
                         ▲ only the CORE
                           executes
```

---

## 3 · The Skill — the outer container

**Label:** `SKILL — the operating procedure`
**Visual role:** a container/frame that visibly **encloses** the orchestrator.

### What the Skill is, in one line

> **The written procedure that turns a general-purpose model into *this* agent.**

### What is inside the Skill

| Content | What it defines |
| --- | --- |
| **The loop** | The fixed sequence the orchestrator must follow: interpret → inspect → route → plan → validate → critique → reuse-scan → smoke → confirm → run → verify → report |
| **Golden rules** | The non-negotiables: never invent a name · zero silent full-scale runs · nothing written before approval · evidence-only claims · never relax an approved requirement |
| **Routing discipline** | Read coarse-to-fine — prune categories, then one-liners, then full cards for finalists only. Never read everything. |
| **Question discipline** | Ask only at the outcome layer ("studio or general?"), never about internal parameters. Ask only when it is material, preference-dependent, and not inferable. |
| **Refuse list** | What to decline and redirect — subjective human labelling, model training/deployment, unapproved large runs |
| **Review checklists** | The 5 questions to answer for every filtered field: meaning · scope · provenance · effect · direction |
| **Decision policy** | When to decide alone, when to recommend, when to ask |
| **Recovery policy** | Explain environment failures and offer options — never silently install, switch, or retry |
| **Reviewer charter** | How to judge the final result honestly, and what it may not override |

### Why the Skill is a separate element and not just "prompt text"

- It is **version-controlled and reviewable** — it lives in the repository beside the code
- It is **portable across hosts** — the same procedure drives Claude, Cursor, or Codex
- It is **testable** — the evaluation harness grades whether the orchestrator actually followed it
- It is **the behavioural layer**, distinct from the *factual* layer (Knowledge Base) and the
  *enforcement* layer (Deterministic Core)

### The three layers of control — worth saying out loud

| Layer | What it does | If it fails |
| --- | --- | --- |
| 📜 **Skill** | Tells the orchestrator *how to behave* | The agent takes a bad route |
| 📚 **Knowledge Base** | Tells it *what is true* | The agent reasons from wrong facts |
| ⚙️ **Deterministic Core** | *Enforces* what must hold | Nothing — this is the backstop that cannot be talked around |

> **Speaker note:** *"The Skill is guidance. The core is enforcement. That distinction matters —
> if the model ignores the Skill, the core still refuses. Good behaviour is encouraged; bad
> outcomes are made impossible."*

> **Design note:** the Skill container should look like a **frame or a boundary**, not a box that
> competes with the orchestrator for attention. Suggested treatment: a thin border with the label
> on the top edge, a subtle tint, and its contents as a small caption line — the orchestrator
> node inside it stays the visual hero.

---

## 4 · The centre node — Orchestrator (Host LLM)

**Label:** `ORCHESTRATOR — Host LLM`
**Visual weight:** largest node, strongest colour, centre of gravity.

### What it does (use these six lines verbatim in the node)

- Understands what the user actually wants
- Decides the **direction** of the workflow
- Chooses which modules fit the goal
- Judges whether the plan **means** the right thing
- Explains trade-offs, asks questions, presents results
- Reacts to every verdict and re-plans

### What it never does (small "constraints" strip inside or under the node)

- ✗ Never writes or executes code
- ✗ Never invents a module or parameter name
- ✗ Never picks a threshold itself
- ✗ Never approves its own full-scale run
- ✗ Never declares success

> **Design note:** the ✗ list is what makes the diagram interesting. A centre node that only says
> "the LLM does everything" is boring and wrong. The tension — *powerful at the centre, tightly
> constrained* — is the whole story. The Skill frame around it reinforces exactly that.

---

## 5 · The three blocks

Each block answers **one kind of question**. Put the question in the block, in quotes — it is the
fastest way for a viewer to understand what the block is for.

| Block | The question it answers | What is inside |
| --- | --- | --- |
| 📚 **Knowledge Base** | *"What exists, and what is it good for?"* | 46 capability cards · module contracts · category taxonomy · blueprints · composition patterns · failure taxonomy |
| ⚙️ **Deterministic Core** | *"Is this valid — and what actually happened?"* | validate · profile · doctor · resolve · plan · smoke · gate · run · verify · report · diagnose · guardrails |
| ♻️ **Reuse Mechanism** | *"Have we already done this?"* | step fingerprints · artifact registry · completion markers · trust tiers · run history |

### Suggested block ordering (left → right)

**Knowledge → Core → Reuse.**
This reads as the natural sequence of a request: *learn what exists → prove the plan → avoid
redoing work.*

---

## 6 · The connections — what flows on each arrow

Label the arrows. Unlabelled arrows make a diagram look busy without adding meaning.

### Skill → Orchestrator (containment, not a query)

The Skill does **not** get an arrow in the normal sense. It **contains** the orchestrator.

If a connector is preferred over containment, use a single one-way line labelled **"governs"** —
never a two-way arrow. The orchestrator does not negotiate with its own procedure.

### User ↔ Orchestrator (the exclusive channel)

| Direction | Label |
| --- | --- |
| User → Orchestrator | request · answers to questions · **approval** · feedback · corrections |
| Orchestrator → User | clarifying questions · the plan · evidence · results · explanations |

> **Emphasis:** this channel should be visually **thicker or highlighted** — it is the one the
> user experiences. Consider making it the only two-way arrow drawn vertically.

### Orchestrator ↔ Knowledge Base

| Direction | Label |
| --- | --- |
| → | *"which modules exist? what is this one good for? how do these two compare?"* |
| ← | module facts · metrics and their direction · presets · ordering guidance · caveats |

### Orchestrator ↔ Deterministic Core

| Direction | Label |
| --- | --- |
| → | the proposed **Recipe** · "check this" · "test this small" · "run this" |
| ← | verdicts · evidence · **refusals** · measured results · grounded fix options |

### Orchestrator ↔ Reuse Mechanism

| Direction | Label |
| --- | --- |
| → | *"has this exact work already been done on this data?"* |
| ← | already done / partly done / fresh · what it would save · trust level |

### Core → NeMo Curator

| Direction | Label |
| --- | --- |
| → | **executes the approved pipeline** |

> **Critical visual rule:** the execution arrow starts at the **Core**, never at the
> Orchestrator. The LLM never touches execution. If a viewer can draw a line from the LLM to
> NeMo Curator, the diagram has failed.

---

## 7 · The user-feedback rule (the point of the diagram)

> ## Everything the user says goes to the orchestrator.
> ## Everything the user sees comes from the orchestrator.
> ## No block talks to the user. Ever.

**Why this matters — say this out loud when presenting:**

The three blocks produce raw material: JSON verdicts, error codes, refusals, measurements. None
of that is a conversation. The orchestrator's job is to turn machine output into something a
human can act on — and to turn a human's answer back into the next machine call.

**Two consequences worth showing:**

1. **Feedback re-enters at the top.** When a user says "actually, make it stricter" or "no, use
   the other model", that goes to the orchestrator, which **re-plans and re-queries the blocks**.
   The loop restarts — it does not patch the pipeline sideways.
2. **Refusals are translated, not forwarded.** When the core refuses (invalid recipe, blocked
   environment, unapproved run), the user does not receive an error code. They receive an
   explanation and a choice.

> **Design note:** draw the feedback path as a **returning curve** from the user back into the
> orchestrator, visually distinct from the initial request. It should read as a *loop*, not a
> one-way pipe.

---

## 8 · Visual specification

### Colour system

| Element | Role | Suggested treatment |
| --- | --- | --- |
| 👤 **User** | Human | Neutral / light — clearly outside the system |
| 📜 **Skill** | The procedure | Subtle tinted **frame**, label on the top edge — a boundary, not a box |
| 🧠 **Orchestrator** | The model | Strongest accent colour, largest node, inside the frame |
| 📚 **Knowledge Base** | Static facts | Cool tone — implies "reference material" |
| ⚙️ **Deterministic Core** | Code and proof | A distinctly different tone — implies "machinery" |
| ♻️ **Reuse Mechanism** | Memory of past work | A third tone — implies "history" |
| **NeMo Curator** | The engine | Muted / grey — it is infrastructure, not the story |

**Keep the three blocks visually equal in size.** They are peers. Making one bigger implies a
hierarchy that does not exist.

**The Skill must not out-weigh the orchestrator.** It frames; it does not compete.

### Typography hierarchy inside each node

1. Icon + block name (largest)
2. The question in quotes (medium, italic)
3. Contents list (smallest)

### Arrow styling

- **User channel** — thickest, two-way, highlighted
- **Orchestrator ↔ blocks** — medium, two-way
- **Skill → Orchestrator** — containment (no arrow), or a single thin "governs" line
- **Core → Curator** — single direction, distinct style (it is the only arrow that *does* something)

---

## 9 · Suggested animation build (6 steps)

If the diagram is animated, this order tells the story correctly:

| Build | Reveal | Narration beat |
| --- | --- | --- |
| 1 | User + Orchestrator + the two-way channel | *"The user talks to one thing."* |
| 2 | The Skill frame closing around the orchestrator | *"And that one thing follows a written procedure — it isn't improvising."* |
| 3 | Knowledge Base + its arrows | *"To know what exists, it asks the knowledge base."* |
| 4 | Deterministic Core + its arrows | *"To know whether a plan is valid, it asks the core."* |
| 5 | Reuse Mechanism + its arrows | *"Before spending anything, it asks what's already been done."* |
| 6 | NeMo Curator + the execution arrow **from the Core** | *"And only the core executes."* |

**Final state emphasis:** briefly pulse the user channel to reinforce that all feedback returns
to the orchestrator.

---

## 10 · Caption options for the slide

Pick one:

- **"The LLM proposes. The deterministic core disposes."**
- **"One orchestrator. Three specialists. One procedure. One conversation."**
- **"Facts get computed. Judgment gets reasoned."**
- **"The LLM decides direction — not code, configuration, or execution."**

---

## 11 · Technical accuracy notes

**Read this before Q&A.**

### On the three blocks

In the real implementation, the orchestrator reaches the Knowledge Base and the Reuse Mechanism
**through the deterministic core's functions** — the core is what reads the knowledge files and
what probes the artifact registry. Strictly, the core is the single interface.

The diagram draws all three as peer blocks because they answer three genuinely different kinds
of question, and collapsing them into one box hides the architecture rather than explaining it.

**This is a presentation abstraction, and it is a fair one.** If someone asks "does the LLM read
the YAML directly?", the honest answer is:

> "No. It calls the core, and the core serves knowledge in tiers so the model only reads what it
> needs. We've drawn them separately because they're different responsibilities — but there's one
> interface, and everything goes through it."

### On the Skill

The Skill is a **document**, not running code. It is loaded into the model's context and shapes
its behaviour. It is guidance the model follows, not a mechanism that forces compliance.

If someone asks "what if the model ignores the Skill?", the honest — and strong — answer is:

> "Then the deterministic core refuses. The Skill makes good behaviour the default; the core
> makes bad outcomes impossible. We deliberately did not rely on the Skill alone, which is why
> every guardrail lives inside the tools rather than in the instructions."

**Optional alternative layout** for a strictly technical audience: nest the Knowledge Base and
Reuse Mechanism *inside* the Deterministic Core's boundary as sub-blocks, with the orchestrator's
arrows passing through the core's edge. More accurate, slightly less readable.

---

## 12 · What this diagram replaces and why

The earlier four-pillar diagram showed the LLM as *one of four* equal pillars. That was
misleading in three ways:

- It implied the four parts are peers. They are not — **one directs, three serve**.
- It hid the user relationship entirely.
- It left out the Skill, so the LLM appeared to be improvising rather than following a defined,
  version-controlled procedure.

This version fixes all three: the orchestrator is visibly central, the blocks are visibly
subordinate, the user channel is visibly exclusive, and the Skill is visibly the boundary the
orchestrator operates within.
