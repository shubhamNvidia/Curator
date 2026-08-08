# Audio Agent (P1) — instructions for host agents

This package (`nemo_curator.audio_agent`) is a **host-driven** audio pipeline
builder. The host LLM (Claude/Cursor/Codex) is the planner and critic; this
package is the deterministic tool core that grounds every decision. For the full
step-by-step driving instructions see
[`.claude/skills/audio-curation/SKILL.md`](../../.claude/skills/audio-curation/SKILL.md).

## Tool surface (all print JSON)

```bash
python -m nemo_curator.audio_agent doctor --json            # machine health + grounded options -- run first on any env symptom
python -m nemo_curator.audio_agent diagnose --error '...' [--recipe R.yaml]  # failure + recipe-aware recovery choices
python -m nemo_curator.audio_agent discover                 # list stages (name, category, one-liner)
python -m nemo_curator.audio_agent catalog-tree             # L0 category tree (route over this)
python -m nemo_curator.audio_agent describe NAME            # static contract (+ card) for one stage
python -m nemo_curator.audio_agent cards --category quality # L1 one-liners
python -m nemo_curator.audio_agent cards --names UTMOSFilterStage SIGMOSFilterStage   # L2 full cards
python -m nemo_curator.audio_agent context --goal '{...}' --data DATA   # PlanningContext (tree + profile + env + blueprints)
python -m nemo_curator.audio_agent resolve --stage NAME --label studio  # outcome/label/use-case -> concrete params (1A.2)
python -m nemo_curator.audio_agent validate --recipe R.yaml --data DATA # Verdict (roles/keys/cards/gates)
python -m nemo_curator.audio_agent smoke --recipe R.yaml --sample 10 --data DATA --bootstrap-ray
python -m nemo_curator.audio_agent run --recipe R.yaml --confirm <hash> --data DATA --bootstrap-ray   # confirm-gated
python -m nemo_curator.audio_agent report --output OUT --data DATA
python -m nemo_curator.audio_agent verify --criteria C.yaml --evidence E.yaml   # acceptance criteria vs evidence (1A.1/1A.3)
python -m nemo_curator.audio_agent calibrate --smoke SMOKE.json         # measured per-stage resources from a smoke (1C.2)
python -m nemo_curator.audio_agent runs [--run-id ID] [--data DATA]     # local run records + artifacts (provenance)
python -m nemo_curator.audio_agent reuse-scan --recipe R.yaml --data DATA   # prior work this recipe could reuse (read-only)
python -m nemo_curator.audio_agent continue --recipe R.yaml --data DATA [--parent-run-id ID] \
    [--execute --choice as_is|extend|fresh --confirm <hash>]            # plan, then carry out the reuse choice
python -m nemo_curator.audio_agent reindex                              # rebuild the run/artifact index from the JSON records
```

For recipe-driven verbs, the first supported source stage's configured parameters
bind execution. `Recipe.inputs` and `--data` are optional assertions about that
source; neither injects or rewrites parameters. Omit `--data`, or pass the same
canonical source. A mismatch is refused. A multi-manifest `ManifestReader` can
run as authored only with singular `--data` omitted, but remains unkeyed until
aggregate source identity is supported. `context --data` is different: before a
recipe exists, it profiles that path directly for planning.

`--bootstrap-ray` (opt-in) makes the agent start a correctly-configured local Ray
head itself (free port, plasma on /tmp, API limit) when none is reachable, so no
manual Ray setup is needed. An externally set `RAY_ADDRESS` is always respected.

Or import the same verbs: `from nemo_curator import audio_agent`.

Contracts returned by `describe`/L2 cards are explicitly marked
`contract_resolution: static_params_and_hints`: they expose real params and
class hints, but their placeholder reads/writes/cardinality are not configured
runtime facts. `validate.semantic_review` uses the instantiated dynamic
contracts and is the authoritative recipe-lineage view.

## Environment diagnosis and user decisions

The core does not embed an LLM and never applies remediation. It returns an
`environment_decision` containing detected facts, affected execution leaves,
blocking/uncertain issues, ranked grounded choices, CPU feasibility, and a
`decision_required` flag. The host LLM must correlate that packet with the
user's constraints, explain the recipe-specific impact, recommend an available
choice, and ask before any host, environment, launch, credential, device,
model/decoder, or recipe change.

Never silently install/upgrade/downgrade, switch to CPU, expose or request a
credential value in chat, or retry the same failed action. CPU is a conditional
candidate only when every affected flattened execution stage explicitly supports
it; do not call it executable until its new recipe builds and smokes. Any CPU,
decoder, model, or stage alternative is a new recipe/config hash and must
validate, smoke, and pass the confirmation gate again. After an external fix,
re-run the preflight; do not assume it worked. An unknown diagnosis permits only
the packet's minimal diagnostic steps, never an invented cause or command.
Scope facts to the actual execution target: never project driver GPU, ffmpeg,
credential, disk, Python, or launch facts onto external Ray/custom workers.
A driver/toolkit mismatch hard-blocks only a selected path known to require
runtime PTX/JIT; other GPU paths get a bounded-smoke warning until evidence says
they fail.

## Semantic review boundary

`validate` certifies mechanically provable runnability only. A clean `status`
and `runnable=true` are not claims that the recipe expresses the user's intent.
The returned `semantic_review` packet deterministically assembles configured
producer/consumer field lineage, cardinality transitions, and relevant card
facts. The host LLM must use it after validation and before smoke to produce one
of `pass`, `revise`, or `ask`.

The host response must copy `semantic_review.recipe.config_hash` into
`recipe_config_hash`. That binds the critique to the exact canonical recipe;
changing any recipe content requires validation and critique again.

The core must not decide open-ended meaning with module-specific rules. Cards
own field meaning, units, provenance, granularity, propagation across fan-out or
aggregation, and common misinterpretations; the host critic maps those facts to
the request. Deterministic checks remain responsible for universal facts such
as real stages/params, impossible task/key flow, serialization, environment
gates, side effects, and confirmation integrity.

## Recipe IR

A recipe is `{stages: [{ref, params}], inputs, preset}`. `ref` must be a real
stage name from `discover`. The first supported source stage's parameters name
the data execution reads; `inputs` only asserts those values. The core resolves
the recipe to a runnable Curator pipeline, validates it, and freezes it to a
`recipe_id` + `config_hash`.

## Guarantees (why this is safe and generic)

- **Grounded**: retrieval/validation/execution/reporting is deterministic; the
  host only selects from real, index-served entries and can never run an invented
  stage (the validator rejects unknown refs).
- **Generic**: all task knowledge lives in the catalog, cards, blueprints and
  recipes (YAML). Adding a non-source stage (contract + card) makes it plannable
  with no core change; a new source stage also needs an explicit input-identity
  adapter before it can be profiled or reused.
- **Safe**: `run` refuses without explicit confirmation (0 silent full-scale
  runs) and checks plan-execution integrity via the config hash.

## Not in P1

Local run records (`runs`), content-addressed artifacts and reuse (`reuse-scan` /
`continue`) exist for provenance and for skipping work whose computation key and
resolved dataset identity match at the recorded trust tier. Low-trust matches
default to fresh. This is **deterministic memoization**, described in
[`REUSE_ARCHITECTURE.md`](REUSE_ARCHITECTURE.md).
There is still **no cross-session learning** (no learned priors, nothing that changes *what*
the agent plans), no agent decision tracing, no best-of-N scorer, and no embedded planner
LLM. The knowledge (recipes/cards/blueprints) is read-only and hand-authored.
