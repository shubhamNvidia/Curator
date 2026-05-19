# `curator-adv` examples

Phase 2 example IRs you can `plan` and `run` end-to-end against the existing
28-stage catalog.

## `examples/tts_clean_2_to_60.json`

The headline Phase-3 demo prompt as a hand-authored IR.

```bash
# Validate + compile only
curator-adv plan tutorials/audio/agentic/examples/tts_clean_2_to_60.json

# Dry-run (instantiates every stage but does not feed data)
curator-adv run tutorials/audio/agentic/examples/tts_clean_2_to_60.json --dry-run

# Full execution — point --target-dir somewhere with enough free space
curator-adv run tutorials/audio/agentic/examples/tts_clean_2_to_60.json \
  --target-dir /tmp/curator-adv-out
```

The runner emits `<target_dir>/.adv/`:

- `ir.json`            — the original IR
- `compiled.yaml`      — canonical `stages:` YAML (Hydra-compatible)
- `findings.yaml`      — validator findings (if any)
- `run_card.yaml`      — complete run record for replay / audit

Use `curator-adv replay <target_dir>` to re-run a saved run with the same
IR and (optionally) the same `run_id`.

## Phase 3: prompt → pipeline via NAT

The agentic layer uses NVIDIA NeMo Agent Toolkit. The 12 ADV tools register
automatically at import (entry-point `nat.components.curator_adv`).

```bash
# Drive the React agent with a natural-language prompt
curator-adv plan-from-prompt \
  "Build a clean, single-speaker dataset, 48 kHz mono, clips 2-60 s. \
   Commercial-safe only." \
  --dataset /data/manifest.jsonl --kind manifest \
  --out /tmp/adv-run-1

# Run NAT directly using the bundled workflow
nat run --config_file nemo_curator/agentic/nat/workflow.yml \
        --input "Build a clean, single-speaker dataset from /data/audio."
```

### LLM model selection

`workflow.yml` configures a two-tier setup:

- `llms.planner` — cheap, low-temperature model used by the React loop for
  intent extraction + tool selection (defaults to a Nemotron 3 Nano).
- `llms.synth` — premium model used by the Layer-5 critic for intent-output
  alignment scoring (defaults to a Nemotron 3 Super v3).

Override via environment variables:

```bash
export NVIDIA_API_KEY=nvapi-...
export ADV_PLANNER_MODEL=nvidia/your-cheap-model
export ADV_SYNTH_MODEL=nvidia/your-premium-model
```

### Refusing impossible prompts

The agent uses `gap_report` before planning. Any prompt asking for emotion,
language ID, accent, gender, speaker embedding, AED, noise classification,
or any augmentation will be refused with an explicit list of missing
capability tags — those modules ship in Phase 5 via the wizard onboarding
flow.

