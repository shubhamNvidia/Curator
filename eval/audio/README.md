# Audio Agent eval harness + GPU-free sandbox

Regression evals for both planes of `nemo_curator.audio_agent`:

- `run_eval.py` checks the deterministic oracle the host relies on (composition,
  parameters, gates, acceptance, reuse, and safety). It does **not** certify that a
  mechanically valid recipe matches the user's intended field/metric meaning.
- `agent_runner.py` + `trace_check.py` + `judge.py` check the host LLM's planning
  and semantic critique over captured traces.

The deterministic suite loads no models and runs no pipelines, so it is CPU-safe
in CI. LLM trace capture is opt-in.

## Run

```bash
python -m eval.audio.run_eval                 # exit 0 iff every query passes
python eval/audio/run_eval.py --min-pass-rate 0.9
bash eval/audio/run_all.sh --llm               # fresh traces + authoritative gate
python -m eval.audio.aggregate_traces --non-llm # explicit diagnostics only
```

Add queries in [`queries.yaml`](queries.yaml). Each carries a gold recipe (a
library `recipe_ref` or an inline `recipe`) and an `expect` block
(`validate_ok`, `runnable`, `has_code`, `unproducible_role`, `no_stage_for`).

## What it covers (P1 seed set)

- core: the three library recipes (readspeech quality, FLEURS ASR+WER, ALM windowing)
- extended: inline recipes (duration, VAD+quality, mono/resample, diarization)
- negative/guarded: unknown stage rejected; WER unproducible without transcripts;
  emotion/accent labeling maps to no stage (refuse/redirect)
- paired semantic-intent traces in
  [`scenarios/L15_semantic_intent.yaml`](scenarios/L15_semantic_intent.yaml):
  parent-vs-child field meaning after fan-out, segment-vs-recording quality,
  aggregate-vs-row metrics, intentionally persisted-vs-consumed transforms, and
  explicit selection among multiple valid producer keys

## Semantic-critique trace contract

`validate.semantic_review` is the core's deterministic evidence packet. Level-15
traces separately capture the existing host-LLM judgement required by the
audio-curation skill:

```yaml
semantic_critique:
  mechanically_runnable: true
  recipe_config_hash: "<copy validate.semantic_review.recipe.config_hash>"
  intent_status: pass
  stage_reviews:
    - stage: ConsumerStage
      finding: "This stage implements the requested field-level decision."
      evidence: ["goal:user-request", "card:ConsumerStage", "recipe:ConsumerStage"]
  field_reviews:
    - field: producer_value
      producer: ProducerStage
      finding: "The value still has the requested entity and granularity here."
      evidence: ["contract:ProducerStage", "recipe:ConsumerStage"]
  behavior_checks: []
  transform_checks: []
  model_checks: []
  assumptions_or_questions: []
```

The deterministic trace grader is deliberately domain-agnostic. Scenario YAML
declares the required non-empty critique sections and resulting stage
order/params. The grader requires card inspection, a green validation of the
exact final recipe, the returned deterministic evidence packet, review coverage
for every configured stage, and resolvable `source:locator` citations. Card and
contract citations must occur in successful returned result payloads; merely
requesting a card or naming a stage in tool arguments is not evidence. It does
not contain speaker-, metric-, or module-specific meaning rules.

`judge.py` evaluates whether the critique is actually correct. Its deterministic
fallback is diagnostic only; `aggregate_traces --judge-model ...` is the
authoritative semantic regression gate used by `run_all.sh --llm`. That gate
requires every Level-15 scenario in a fresh trace directory, a passing
deterministic trace score, a schema-valid authoritative judge response, and a
passing judge score. Missing, blocked, stale, non-authoritative, or failed rows
make the command fail. Judge JSON is type-checked (`bool` is not accepted as a
number), range-checked to `[0,1]`, and rejected unless every required field and
item has the declared type.

Every Level-15 side also contains a `mechanically_valid_counterexample`. These
counterexamples prove why validation cannot be the semantic oracle: the topology
composes, but it answers the paired intent incorrectly.

Each Level-15 live capture is bound to a checked-in WAV manifest through its
`live_capture.fixture`. The runner passes that exact absolute path to the host,
requires `doctor` before validation, and still uses the real core/environment
verdict. It never mocks GPU availability or rewrites a failed verdict to green.
Consequently the speaker-separation and ASR pairs require a host whose actual
GPU/CUDA preflight passes; on another host they are reported as blocked and the
authoritative gate fails honestly.

Limitations:

- Structured critique sections and citations prove review coverage, not that the
  prose is true. Use the model judge and periodic human review for semantic
  correctness.
- Scenario gold consequences are hand-authored and must evolve with card/stage
  semantics.
- The suite does not execute GPU models; runtime quality remains covered by the
  separate smoke/E2E layers.

## GPU-free sandbox

Two layers of GPU-free testing:

1. **This harness** exercises `validate` (structural composition + card
   constraints + environment/safety gates) with no model loading — the fast CI
   gate. A green verdict is necessary, not semantic approval.
2. **Full-pipeline simulation** (running `smoke`/`run` without GPUs) reuses the
   model-boundary stubs in
   [`tests/stages/audio/test_agent_simulation_pipelines.py`](../../tests/stages/audio/test_agent_simulation_pipelines.py):
   install those stubs, then call `audio_agent.smoke(recipe, sample=N)` on a tiny
   fixture (e.g. `tests/fixtures/audio/alm/sample_input.jsonl`) to get real
   retained/rejected evidence without a GPU.

## A/B (contract vs skill)

The pass rate here is the deterministic-oracle score. To A/B the contract-based
planner against a source-reading skill on the same prompts, run both planners
over `queries.yaml` and compare tokens/latency/validity-before-execution — the
harness already reports the validity rate side of that comparison.
