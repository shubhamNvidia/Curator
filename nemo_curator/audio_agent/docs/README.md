# Audio agent — human documentation

**These are human-facing documents. Nothing here is loaded as agent context.**

An agent working with the audio curation tooling reads none of these files. It reads:

| What the agent actually loads | Where |
| --- | --- |
| The skill and its references | `../skills/audio-curation/SKILL.md`, `../skills/audio-curation/references/` |
| Host working instructions | `../AGENTS.md` (and `../CLAUDE.md`, which just points at it) |
| The knowledge itself | `../knowledge/` — cards, taxonomy, blueprints, patterns, failures, served **through the verbs**, never pasted as prose |

Keep it that way. If a fact in this folder matters to how the agent behaves, it belongs in
`../knowledge/` (where the conformance audit can check it against the code) or in the skill files
(where the golden rules live) — not here, where nothing enforces it and nothing reads it.

## Contents

| Document | Purpose |
| --- | --- |
| [AUDIO_AGENT_KNOWLEDGE.md](AUDIO_AGENT_KNOWLEDGE.md) | Source of truth for the agentification narrative: what was built, why, what was learned. Written for readers with no prior NeMo Curator knowledge. |
| [AUDIO_AGENT_DECK.md](AUDIO_AGENT_DECK.md) | Presentation deck, drawn from the knowledge document. |
| [AUDIO_AGENT_DESIGN_SOURCE.md](AUDIO_AGENT_DESIGN_SOURCE.md) | Design rationale and decision record. |
| [AUDIO_AGENT_IMPLEMENTATION_GUIDE.md](AUDIO_AGENT_IMPLEMENTATION_GUIDE.md) | Implementation detail for contributors. |
| [AUDIO_AGENT_ORCHESTRATOR_DIAGRAM.md](AUDIO_AGENT_ORCHESTRATOR_DIAGRAM.md) | Orchestration diagrams. |

## A caveat worth reading first

These documents are prose, hand-maintained, and **not gate-checked against the code** — unlike the
capability cards, which are. They drift. A 2026-08-14 review of the knowledge document found a
shipped subsystem (`delta-run`) documented as unbuilt, a stage count two behind reality, and a
recipe example missing a required parameter. All were corrected, but the class of error recurs.

**Verify any specific claim against the code or the verbs before relying on it.** Counts come from
`discover` and `knowledge/cards/`; the verb surface comes from `--help`; stage parameters come from
`describe --stage <name>`.
