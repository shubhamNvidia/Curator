# Audio Agent eval harness + GPU-free sandbox

Regression evals for the `nemo_curator.audio_agent` deterministic core. Because
the planner is host-driven (the chat LLM), these evals assert on the deterministic
oracle the host relies on — gold-recipe validity, unproducible roles, and
capability coverage — not on an LLM. They load no models and run no pipelines, so
they run on CPU in CI.

## Run

```bash
python -m eval.audio.run_eval                 # exit 0 iff every query passes
python eval/audio/run_eval.py --min-pass-rate 0.9
```

Add queries in [`queries.yaml`](queries.yaml). Each carries a gold recipe (a
library `recipe_ref` or an inline `recipe`) and an `expect` block
(`validate_ok`, `runnable`, `has_code`, `unproducible_role`, `no_stage_for`).

## What it covers (P1 seed set)

- core: the three library recipes (readspeech quality, FLEURS ASR+WER, ALM windowing)
- extended: inline recipes (duration, VAD+quality, mono/resample, diarization)
- negative/guarded: unknown stage rejected; WER unproducible without transcripts;
  emotion/accent labeling maps to no stage (refuse/redirect)

## GPU-free sandbox

Two layers of GPU-free testing:

1. **This harness** exercises `validate` (structural + semantic + card + gate)
   with no model loading — the fast CI gate.
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
