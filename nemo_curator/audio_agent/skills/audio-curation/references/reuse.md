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
success bar** (the data is reused and the contract re-verified), and a Curator
build change that did not touch the stages in play. A detected dataset-key
change, a missing completion marker, or an edit to a stage's own implementation
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

## When only a few files changed

A dataset key names the whole corpus, so adding one file to a curated folder misses every step
key and the plain reading of that miss is "recompute all thousand files". When the scan can do
better it says so on the same card: `delta.status: ready`, `recommended: delta`, and a `choices`
list whose first entry is running the changed files only. The `delta` block names them
(`change.added_files`, `modified_files`, `removed_files`), says which stages would run
(`run_stages`), how many prior rows survive (`rows_kept`) and how many are dropped and
recomputed (`rows_dropped`).

```bash
python -m nemo_curator.audio_agent delta-run --recipe recipe.yaml --data /path/to/data \
  --confirm <config_hash>
```

This runs the user's own stages over the changed files, merges the rows into the existing
manifest, and republishes it under the key the full pipeline has for the enlarged corpus — so
the next `reuse-scan` answers `already_done` by an ordinary probe. It rewrites the manifest in
place after merging, which is why it is confirm-gated like `run`. Relay its `next`: when the
delta covers only a prefix of the recipe, the remaining stages still run over every row, via
`continue --choice extend`.

`status: no_delta` is an answer, not an error, and its `reason` is worth relaying because it
says what would have to change. Common ones: nothing is persisted early enough to merge into
(`add-checkpoint` fixes it), a stage in the prefix has not declared that it computes each row
from that row alone, the prior rows cannot be traced back to the files that produced them, or
the two corpora share no file at all — a different dataset rather than a changed one. A refusal
means a full run; it never means a partial result presented as a whole one.

## When the scan says the work was done but nothing was saved

A stage only leaves something to resume from if it was configured to write somewhere. A
pipeline whose GPU stages hand their rows to the next stage in memory has nothing on disk, so
`decision` is `fresh` even though the transcription ran last week — the scan discloses this as
`prior_unsaved` and attaches an `offer`.

When the offer's `action` is `add_checkpoint`, relay it: one `ManifestWriterStage` in their
recipe makes the expensive stages resumable from then on. Get the recipe rather than editing
by hand — the position is not always where the expensive work ends, because a manifest cannot
hold a waveform that is still in memory, nor state some stages pass to each other outside the
row:

```bash
python -m nemo_curator.audio_agent add-checkpoint --recipe recipe.yaml \
  --output-path /path/they/choose/checkpoint.jsonl
```

This returns the recipe with the writer in place and changes nothing on disk. Save it, then
`validate` → smoke → `run` as usual. `action: no_checkpoint` means don't raise it: the `why`
says whether the work is too cheap to be worth a file, a writer is already there, or the
pipeline holds audio in memory to the end. Never propose a checkpoint the offer did not.

Record what a run was FOR with `run --goal "..."` — that objective is what makes the
candidate legible to a human months later.
