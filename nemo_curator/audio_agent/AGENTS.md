# Audio Agent — agent guardrails and workflow

These instructions apply to any AI coding agent working in `nemo_curator/audio_agent/`.
Every host reads this file: Codex and Cursor load a nested `AGENTS.md` automatically for
work in this directory, and Claude Code reaches it through the sibling `CLAUDE.md` import.

The full procedure lives in `skills/audio-curation/SKILL.md`, with its long sections under
`skills/audio-curation/references/`. This file carries only what an agent must know
*before* it decides to read that skill.

## Treat the Curator repo as read-only for dataset use cases

While using the audio agent to build/curate a dataset (ALM windows-type datasets,
quality/duration filtering, resampling, etc.), you are a **consumer** of Curator,
never an editor of it.

- NEVER create, edit, or delete any file inside this repository (`nemo_curator/`,
  pipeline/stage source, the `audio_agent/` source, repo configs, tests, etc.) in
  order to satisfy a particular user's dataset or use case.
- NEVER tweak stage/pipeline source, thresholds, filters, windowing logic, or agent
  code because a single dataset produced empty or unexpected output.
- Treat everything under the Curator repo as READ-ONLY reference you may read and run,
  but not change, while serving a use case.

A dataset that comes out empty or small is a data/config problem, not a reason to patch
shared library source. Editing the repo to force a single use case corrupts the library
for everyone and hides the real, data-level explanation the user needs.

Scope note: this bans repo edits made to force a specific dataset/run to "work". It
does not ban deliberate, separately-requested development of Curator or the audio
agent itself — that is a different task, done explicitly.

### What to do instead when output is empty/unexpected

1. Diagnose first — do not "fix" code. Empty output is usually correct for the input.
   Example: ALM speaker-turn `windows` are only produced from multi-speaker
   turn-taking audio; single-speaker clips legitimately yield `"windows": []` (seen as
   `lost_win` in the stats). Expected, not a bug.
2. Explain the cause to the user from the data (`duration`, `num_speakers`,
   `stats`/`lost_*` fields, thresholds actually applied).
3. Adjust inputs and configuration, not the repo. Point the pipeline at appropriate
   data, or edit recipe/criteria/config files OUTSIDE the repo (e.g. under the user's
   working directory such as `adv_test/`); write outputs to user-owned directories.
4. If the goal genuinely requires a Curator code change, STOP and ask for explicit
   confirmation, describing the change and impact, before touching any repo file.
   Never do it silently as part of "getting the dataset to work".

## Route audio work through the tool core

For ANY task involving audio files or audio datasets — file durations, quality filtering,
VAD/segmentation, resampling, mono conversion, transcription, WER, diarization, building a
TTS/ASR training set, exporting a manifest — drive the `nemo_curator.audio_agent` verbs.
**Do NOT write ad-hoc shell/Python** (ffprobe, soundfile, librosa, torchaudio loops) to
inspect or process the audio, even for a "quick" one-off like reading durations — use
`GetAudioDurationStage` through the agent.

All verbs print JSON. Run them with the repo virtualenv interpreter from the repo root
(base `python` lacks Curator's deps), or as `nemo-curator-audio <verb>` from an install:

```bash
.venv/bin/python -m nemo_curator.audio_agent discover        # stages (name, category, one-liner)
.venv/bin/python -m nemo_curator.audio_agent catalog-tree    # L0 category tree (route over this)
.venv/bin/python -m nemo_curator.audio_agent cards --category quality        # L1 one-liners
.venv/bin/python -m nemo_curator.audio_agent cards --names UTMOSFilterStage  # L2 full cards
.venv/bin/python -m nemo_curator.audio_agent describe UTMOSFilterStage --params '{...}'
.venv/bin/python -m nemo_curator.audio_agent producers duration             # who writes a key
.venv/bin/python -m nemo_curator.audio_agent context --goal '{...}' --data DATA
.venv/bin/python -m nemo_curator.audio_agent doctor --json
.venv/bin/python -m nemo_curator.audio_agent diagnose --error '...' --recipe R.yaml
.venv/bin/python -m nemo_curator.audio_agent resolve --stage UTMOSFilterStage --label studio
.venv/bin/python -m nemo_curator.audio_agent validate --recipe R.yaml --data DATA
.venv/bin/python -m nemo_curator.audio_agent reuse-scan --recipe R.yaml --data DATA
.venv/bin/python -m nemo_curator.audio_agent smoke --recipe R.yaml --sample 10 --data DATA --bootstrap-ray
.venv/bin/python -m nemo_curator.audio_agent run --recipe R.yaml --confirm <hash> --data DATA --bootstrap-ray
.venv/bin/python -m nemo_curator.audio_agent report --output OUT --data DATA
.venv/bin/python -m nemo_curator.audio_agent verify --criteria C.yaml --evidence E.json --recipe R.yaml
.venv/bin/python -m nemo_curator.audio_agent continue --recipe R.yaml --data DATA --execute --choice extend --confirm <hash>
.venv/bin/python -m nemo_curator.audio_agent add-checkpoint --recipe R.yaml --output-path CK.jsonl  # make GPU work resumable
.venv/bin/python -m nemo_curator.audio_agent delta-run --recipe R.yaml --data DATA --confirm <hash>  # only the files that changed
```

`smoke` and `run` need a Ray cluster. `--bootstrap-ray` starts a correctly-configured
local head (free port, plasma on /tmp, API limit) so no manual setup is needed; if a
cluster already exists, set `RAY_ADDRESS` and omit the flag (it is respected, never
clobbered).

## The loop

1. Interpret + clarify the goal (task, domain, quality bar, output) and derive
   `acceptance_criteria`. Refuse if out of scope.
2. Inspect: `context --data DATA` for the data profile, environment and matched
   blueprints. Report the findings — they are often news to the user.
3. Route coarse-to-fine: `catalog-tree` -> prune categories -> `cards --category` ->
   `cards --names` -> `describe` with the params you intend to use. Prefer adapting a
   matched blueprint over composing from scratch, and prune to the request.
4. Plan -> `validate` -> fix from the issues -> re-validate (at most 3 rounds). A missing
   role means the card set was incomplete, so re-retrieve for that role rather than
   stopping at the first set.
5. Mandatory semantic critique -> `pass` / `revise` / `ask`. Only `pass` may continue.
6. `reuse-scan` before spending compute, then `smoke --sample N` and show
   retained/rejected plus examples (at most 2 rounds). When the scan reports
   `delta.status: ready`, a few files changed since a prior run: offer `delta-run`
   instead of recurating the whole corpus, and relay its `reason` when it refuses.
7. Present the plan, the semantic pass, the smoke evidence, the scale estimate and the
   acceptance contract; get explicit user approval; then `run --confirm <hash>`.
8. Summarize the `report`, verify acceptance, and propose a next action.

## Non-negotiables

- **Never invent** stage or parameter names. Only what `discover` / `cards` / `describe`
  return exists.
- **Ask the tools what a stage reads and writes; never grep the stage source for it.**
  Pass the params you will actually use — the contract is resolved FROM them.
- **0 silent full-scale runs.** Never `run --confirm` before the user approves, having
  seen a smoke result and the scale estimate. The `config_hash` binds approval to plan.
- **Nothing touches the filesystem before approval.** No creating, deleting, moving or
  truncating files ahead of the gate, least of all the user's output. A gate the agent has
  already prepared the ground for is not a gate. Never pre-clean an output "so the rerun is
  clean": the pipeline replaces its own manifest output, so a rerun does not accumulate
  rows. An agent once deleted a user's file over exactly that misreading.
- **A green Verdict is mechanically runnable, not intent-approved.** The semantic critique
  between `validate` and `smoke` is mandatory.
- **Evidence only.** No quality or throughput claim without before/after numbers from a
  `report`.
- **Environment questions have one home:** `doctor`. Do not diagnose the environment from
  stage cards, and never silently install, upgrade, switch CPU/GPU, or retry a failure.
- **Write scratch recipes to `scratch_dir()`**, not the working directory — that is a git
  checkout, where a one-off recipe reads as unfinished work someone forgot to remove.
- **User-facing questions only.** Ask at the outcome layer, never about thresholds, keys,
  residency, batch size or model IDs.

Everything above is the short form. `skills/audio-curation/SKILL.md` is authoritative and
explains why each rule exists; read it before planning a pipeline.
