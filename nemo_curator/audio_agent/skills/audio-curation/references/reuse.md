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
`delta` (the corpus key missed, but only a few files changed and the rest of the
prior result still stands — see [When only a few files changed](#when-only-a-few-files-changed)),
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

- **Never reuse silently.** If `prompt_user` is true, disclose before running. Usually that
  means a reuse candidate: show its card — objective, pipeline, input/output, key params, date,
  metrics, estimated time saved — and offer the three choices, defaulting to the scan's
  `recommended` (which is `fresh` whenever `trust` is `low`; say why — the `weaknesses` list is
  written for a human). But `prompt_user` is *also* set when there is no reuse candidate and only
  a `prior_on_same_path` notice — a `fresh`/`delta` result over a folder curated before. Then
  there is no as_is/extend card to show; surface the notice instead (see
  [When you've curated this folder before](#when-youve-curated-this-folder-before-with-a-different-pipeline))
  and let the user choose to align or proceed. Read `prior_on_same_path` whenever it is present —
  do not summarise a scan from `decision`/`prompt_user` alone, which is how a correct notice went
  unspoken and the user was told "nothing to reuse" over a folder they had just curated.
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
better the `decision` itself is **`delta`**, not `fresh`, with `recommended: delta` and a
`choices` list whose first entry is running the changed files only. `key_matched: false` records
the miss the decision rests on — the key did miss; a full rerun is still the wrong response to
it. **A `decision: delta` card must never be answered with a full `run`** without putting the
delta to the user first, with its `estimated_saving_sec`. Recurating files that are already
done is the failure this exists to prevent, and it has happened: a host read `fresh`, recurated
the whole corpus, and told the user a checkpoint was missing when none was needed. The `delta`
block names the changed files
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
place after merging, which is why it is confirm-gated like `run`.

`status: tail_required` means the merge is done but the curation is not: the delta owned only a
prefix of the recipe, so the stages after the checkpoint still have to see every row. The files
listed in `tail.stale_outputs` are on disk describing the corpus as it was *before* the change,
so do not report the work as finished — run the `continue --execute --choice extend` from its
`next` first, then report. Only `status: completed` means the deliverable is current.

`status: no_delta` is an answer, not an error, and its `reason` is worth relaying because it
says what would have to change. Common ones: nothing is persisted early enough to merge into
(`add-checkpoint` fixes it), a stage in the prefix has not declared that it computes each row
from that row alone, the prior rows cannot be traced back to the files that produced them, or
the two corpora share no file at all — a different dataset rather than a changed one. A refusal
means a full run; it never means a partial result presented as a whole one.

## When you've curated this folder before with a different pipeline

`already_done`, `incremental` and `delta` all match on the step-key chain, which a changed
source stage or changed corpus moves wholesale — so a folder you curated an hour ago becomes
invisible to them the moment the recipe drifts, even slightly. The scan closes that blind spot
with `prior_on_same_path`, present on a `fresh` (or `delta`) result whenever a prior **completed
run read the same source folder**, matched by path rather than by recipe. It is advisory: it
never changes `decision` and reuses nothing. It carries:

- `created_at` and `run_id` — when, and which run.
- `recipe_diff` — `added_stages`, `removed_stages`, `changed_params` (`{stage, param, from, to}`),
  and a human `phrase`. This is how you see that last time used a different source stage, or a
  `mos_threshold` of 3.4 where you now have 2.5.
- `data_delta` — added / modified / removed / unchanged file counts and names since that run
  (`basis: inventory`), or a labelled count comparison when no per-file record was kept.
- `recommendation` — `delta` (same pipeline, only the corpus moved → the changed-file path is the
  cheap answer), `align` (a different pipeline → matching the prior stages is what would make its
  work reusable), or `fresh`.

**Surface it; do not silently run fresh over it.** When `prior_on_same_path` is present, tell the
user before smoking or running: this folder was curated on <date>, here is how the current plan
differs (`recipe_diff.phrase`), and here is what changed in the folder since (`data_delta.phrase`).
Then let them choose — align the differing stages so the prior work reuses (or a `delta` becomes
possible), or proceed fresh as an informed decision. The whole point is that "I've done this here
before" is a fact the user should hear, not one a step-key miss is allowed to hide. It is a notice,
never an action: you still reuse only through `continue`/`delta-run`, never by editing bytes.

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
