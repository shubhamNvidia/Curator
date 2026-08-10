# Don't redo finished work: reusing prior runs

Loaded on demand from `SKILL.md` step 8. Read this once you have a candidate recipe,
**before** smoking or running it.

Every completed step publishes a content-addressed **artifact**, so a later request can
reuse it instead of recomputing it (design: `nemo_curator/audio_agent/REUSE_ARCHITECTURE.md`).
Once you have a candidate recipe — **before** smoking or running — scan for prior work:

```bash
python -m nemo_curator.audio_agent reuse-scan --recipe recipe.yaml --data /path/to/data
```

The `decision` is `already_done` (this pipeline matches a prior computation and
dataset key at the reported trust tier), `incremental` (the first *N* stages are
already done — e.g. resample + VAD + quality-filter exist and only ASR is new),
or `fresh`. Reuse survives things that do NOT change output bytes: a different
batch size, different `resources`, a different output path, or a **stricter
success bar** (the data is reused and the contract re-verified). A detected
dataset-key change, a missing completion marker, or a Curator version change
prevents reuse. Shape-tier matches are low trust and default fresh because
metadata gaps may hide changes. A stage declared non-deterministic is not
refused—it is offered with that said and `fresh` pre-selected, because the result
is real, just not one a rerun is promised to match.

**Three rules for the conversation:**

- **Never read prior artifact content.** `runs` returns artifact URIs so humans can find
  their outputs — do NOT open, `cat`, or inspect those files. A prior manifest records what a
  *different* pipeline did to the data, under criteria the current request may not share, so
  reading it anchors your plan to the old design before the current recipe exists — including
  its empty or filtered-out fields, which say nothing about what this request can achieve.
  Reuse decisions go through `reuse-scan` exclusively — it works on config hashes, not
  content. Artifact URIs are for human reference only.

- **Never reuse silently.** If `prompt_user` is true, show the candidate card — objective,
  pipeline, input/output, key params, date, metrics, estimated time saved — and offer the
  three choices. Use the scan's `recommended` as your default; it is `fresh` whenever
  `trust` is `low` (say why: the `weaknesses` list is written for a human).
- **Never nag.** `prompt_user: false` means don't ask: either there is nothing to reuse
  (just run) or the saving was *measured* and is trivial (take it, and mention it in your
  summary). When `unpriced_stages` is non-empty the question is not about the size of the
  saving — nobody timed those stages and the cards call them expensive — so say that rather
  than quoting the `estimated_saving_sec`, which is a floor and will look absurdly small.

Then act on the choice — don't hand-edit the recipe:

```bash
python -m nemo_curator.audio_agent continue --recipe new.yaml --data /path/to/data \
  --execute --choice extend --confirm <config_hash>
```

- **as_is** — serve the completed output and re-check it against today's criteria.
- **extend** — rewrite the recipe to start from the reused artifact, re-validate the
  remaining stages against what that artifact actually carries, and run only those.
- **fresh** — ignore prior work and recompute.

`--parent-run-id <id>` is optional and additive: it diffs against that specific run, and
whichever engine reuses more wins. `runs --data /path/to/data` shows everything already
done to a corpus; `reindex` rebuilds the lookup index from the JSON records if it is lost.

Record what a run was FOR with `run --goal "..."` — that objective is what makes the
candidate legible to a human months later.
