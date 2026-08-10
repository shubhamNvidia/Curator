# Getting started with the audio agent

For someone who has cloned this repo and wants to curate their own audio data.

## 1. Set up the environment

```bash
uv sync --extra audio_cuda12 && source .venv/bin/activate   # use --extra audio_cpu if you have no GPU
```

## 2. Check the environment once

```bash
python -m nemo_curator.audio_agent doctor
```

This tells you what is working: GPU, ffmpeg, installed packages, disk. Do this before you
plan anything, so you find out here rather than halfway through a run.

## 3. Just ask

Open this repo in Cursor, Codex, or Claude Code and say what you want in plain language,
and point at your data. For example:

> Filter the WAV files in /data/raw down to studio-quality clips and give me a manifest.

**You do not need to load or install a skill.** The instructions are already in the clone,
in the place each of the three tools looks. The tool finds them from your request.

## 4. Answer two kinds of question

The agent does the rest, and asks you only about things it cannot decide for you:

- **What you want**, in plain words — for example "studio, broadcast, or general quality?".
  You will never be asked about thresholds, keys, or batch sizes.
- **Permission to run**, once. Before the full run you get the plan, a small test result,
  a time estimate, and what "done" will mean. Nothing runs at full scale, and nothing is
  written, until you say yes.

You do not write a recipe or pick numbers. Your data stays where it is, and results go to
a directory you own. The agent does not change this repo to make your data work.

## If your tool does not find the skill

Two cases need one command.

**On Windows.** Git cannot create the links this repo uses, so it writes plain text files
instead and every tool quietly finds nothing. Fix it with:

```bash
python -m nemo_curator.audio_agent install-skill --copy
```

**If you installed with pip instead of cloning.** The instructions are inside the package,
where no tool looks. Copy them out:

```bash
nemo-curator-audio install-skill                 # into the current project
nemo-curator-audio install-skill --scope user    # or once, for every project
```

Add `--dry-run` to either command to see where the files would go before writing anything.

## Where to read more

- `skills/audio-curation/SKILL.md` — the full procedure the agent follows.
- `AGENTS.md` — the short version, plus the rules the agent must not break.
- `ENVIRONMENT.md` — environment problems in detail.
- `../stages/audio/AGENT_READY.md` — only if you are writing a new stage, not curating data.
