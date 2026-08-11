# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run only the files that changed, and keep the rest of the prior result.

Reuse (``reuse.py``) is all-or-nothing per step: the dataset key is one digest over every
source file, so adding one file to a thousand-file corpus changes it and every step key
below it, and the honest answer becomes "nothing matches, run it all again". That answer is
correct -- the prior artifact really does not cover the new file -- but it is far more
expensive than the truth requires, because 1000 of those 1001 files were already done.

This module turns "the dataset changed" into "these files changed", using the per-file
inventory the profiler already computes for the dataset key and this module persists as
artifact coverage (``artifacts.save_coverage``). It only ever *decides*; ``verbs.delta_run``
executes. Nothing here tells a stage it is running incrementally: a delta run is an ordinary
run over a filtered input, followed by a merge.

Four questions have to be answered yes before that is sound, and each one refuses by name
rather than assuming:

1. **Which files changed?** ``classify`` compares two inventories and reports added, modified
   and removed by name. An unrecorded inventory is not an empty one, so it cannot be compared.
2. **How deep is per-file work independent?** ``region`` reads the ``Cardinality`` every
   contract declares: ``1:1``, ``1:N fan-out``, ``filter`` and nested-list all keep a row
   traceable to the one input file it came from, while ``N:1`` mixes files and ends the
   region. For the case no contract or row count can reveal -- one row out per row in, each
   value computed against a corpus statistic -- stages declare
   ``Gates.per_row_independent``, and undeclared refuses.
3. **Is the declaration true?** ``region`` cross-checks it against what the reused run
   actually recorded (artifact row counts, filter accounting) and refuses on disagreement,
   so a wrong declaration is caught by the evidence instead of trusted.
4. **Which prior rows belong to the changed files?** ``provenance`` finds a row column whose
   values land on the inventory's files and verifies that mapping against the artifact's real
   rows. Derived and checked, never assumed: a pipeline whose rows no longer name their origin
   simply gets no delta.

See REUSE_ARCHITECTURE.md §7b.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nemo_curator.audio_agent.artifacts import Artifact, StepPlan
    from nemo_curator.audio_agent.contracts import Recipe

# Cardinalities that keep every output row descended from exactly one input file. A fan-out
# splits one row into many, which is still one origin per child; ``N:1`` is the only shape
# that mixes files, and it is where the incremental region ends.
_TRACEABLE = frozenset({"1:1", "1:1 nested-list", "1:N fan-out", "filter"})
# How many rows of a prior artifact to read when deriving provenance. Enough to be confident
# the column really is the origin, small enough that a million-row manifest is not re-read.
_PROVENANCE_SAMPLE = 2000
Kind = Literal["identical", "added_only", "changed", "removed", "unrelated"]


@dataclass(frozen=True)
class Change:
    """What moved between two inventories of the same corpus."""

    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()

    @property
    def kind(self) -> Kind:
        """Named so a caller can branch on the case rather than re-deriving it.

        ``unrelated`` is the guard against a coincidence: two corpora that share no file at
        all are not one corpus that changed, they are two datasets, and subtracting one from
        the other would delete every prior row and call the result incremental.
        """
        if not self.added and not self.modified and not self.removed:
            return "identical"
        if not self.unchanged:
            return "unrelated"
        if self.removed:
            return "removed"
        if self.modified:
            return "changed"
        return "added_only"

    @property
    def touched(self) -> tuple[str, ...]:
        """The files a delta run has to process: new ones and edited ones."""
        return (*self.added, *self.modified)

    @property
    def stale(self) -> tuple[str, ...]:
        """Files whose prior rows are no longer valid and must not survive the merge."""
        return (*self.modified, *self.removed)

    def phrase(self) -> str:
        """The change as a clause, for a card whose reader wants the shape before the counts."""
        parts = [
            f"{len(group)} file(s) {word}"
            for group, word in ((self.added, "were added"), (self.modified, "changed"), (self.removed, "were removed"))
            if group
        ]
        if not parts:
            return "no file changed"
        return f"{', '.join(parts[:-1])} and {parts[-1]}" if len(parts) > 1 else parts[0]

    def summary(self) -> dict[str, Any]:
        """Counts first, then names, capped -- a caller shows this to a human."""
        return {
            "kind": self.kind,
            "added": len(self.added),
            "modified": len(self.modified),
            "removed": len(self.removed),
            "unchanged": len(self.unchanged),
            "added_files": list(self.added[:20]),
            "modified_files": list(self.modified[:20]),
            "removed_files": list(self.removed[:20]),
        }


@dataclass(frozen=True)
class Delta:
    """A decision about processing only the changed files, ready to execute or refuse."""

    status: Literal["ready", "refused", "none"] = "none"
    reason: str = ""
    change: Change | None = None
    # The artifact the surviving rows come from, and where its output lives.
    prior_step_key: str = ""
    prior_dataset_key: str = ""
    uri: str = ""
    kind: str = ""
    stage_ref: str = ""
    # Stages 0..prefix-1 are what a delta run executes over the changed files.
    prefix: int = 0
    # The row column that names each row's originating input file.
    provenance_key: str = ""
    # Absolute paths of the files to process, and how many prior rows the merge drops.
    files: tuple[str, ...] = ()
    drops: int = 0
    keeps: int = 0
    # Every manifest inside the delta's stages, each of which is rewritten and merged.
    sinks: tuple[Sink, ...] = ()
    estimated_saving_sec: float = 0.0
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "reason": self.reason}
        if self.change is not None:
            out["change"] = self.change.summary()
        if self.status != "ready":
            return out
        out.update(
            {
                "resume_from": {
                    "stage": self.stage_ref,
                    "stage_index": self.prefix - 1,
                    "uri": self.uri,
                    "step_key": self.prior_step_key,
                    "dataset_key": self.prior_dataset_key,
                },
                "run_stages": self.prefix,
                "files": list(self.files[:20]),
                "file_count": len(self.files),
                "rows_kept": self.keeps,
                "rows_dropped": self.drops,
                "provenance_key": self.provenance_key,
                "merges": [s.summary() for s in self.sinks],
                "estimated_saving_sec": round(self.estimated_saving_sec, 1),
            }
        )
        if self.notes:
            out["notes"] = list(self.notes)
        return out


def classify(prior: dict[str, str] | None, current: dict[str, str] | None) -> Change | None:
    """Compare two inventories. ``None`` when either was never recorded.

    ``None`` is not ``{}``: an unrecorded inventory means the comparison cannot be made, while
    an empty one is the claim that the corpus holds nothing. Treating the first as the second
    would report every file as added.
    """
    if prior is None or current is None:
        return None
    added = tuple(sorted(set(current) - set(prior)))
    removed = tuple(sorted(set(prior) - set(current)))
    both = set(current) & set(prior)
    return Change(
        added=added,
        modified=tuple(sorted(f for f in both if current[f] != prior[f])),
        removed=removed,
        unchanged=tuple(sorted(f for f in both if current[f] == prior[f])),
    )


def region(recipe: Recipe, *, upto: int) -> tuple[int, str]:
    """How many leading stages keep each row's work independent of the other files.

    Returns ``(depth, reason)`` where ``depth`` counts stages from the source and ``reason``
    says what ended the region -- empty when the whole ``upto`` prefix qualifies. A depth
    below the checkpoint is not a failure: it means the checkpoint has to move up, and the
    caller reports which stage decided that.
    """
    from nemo_curator.audio_agent.recipe import build_stages
    from nemo_curator.stages.audio import agent as foundation

    built, _ = build_stages(recipe)
    if not built:
        return 0, "the recipe has no runnable stages"
    for index, stage in enumerate(built[: max(0, upto)]):
        name = type(stage).__name__
        try:
            contract = foundation.build_contract(stage)
        except Exception as exc:  # noqa: BLE001 - an unreadable contract is a refusal, not a crash
            return index, f"{name}'s contract could not be read ({type(exc).__name__})"
        if contract.cardinality not in _TRACEABLE:
            return index, f"{name} combines rows from several files ({contract.cardinality})"
        independent = contract.gates.per_row_independent
        if independent is False:
            return index, f"{name} computes each row against a statistic over the whole corpus"
        if independent is None:
            return index, (
                f"{name} has not declared gates.per_row_independent, so whether its output for "
                f"one file depends on the others is unknown"
            )
    return max(0, upto), ""


def contradictions(
    recipe: Recipe,
    *,
    upto: int,
    published: dict[int, Artifact],
    per_stage: dict[str, Any],
) -> list[str]:
    """Declarations the reused run's own numbers disprove.

    A contract is a claim about a stage, and a claim that its last run contradicts is worth
    less than nothing -- it is a claim a delta would rely on. Two records can disagree with
    one: an artifact's ``rows_in``/``rows_out`` (published per persisted step) and the
    accept/reject counts filters log into ``per_stage_metrics``. Row-preserving cardinalities
    are the only ones with a number to check; a fan-out or a filter is free to change counts.
    """
    from nemo_curator.audio_agent.recipe import build_stages
    from nemo_curator.stages.audio import agent as foundation

    preserving = {"1:1", "1:1 nested-list"}
    built, _ = build_stages(recipe)
    found: list[str] = []
    for index, stage in enumerate(built[: max(0, upto)]):
        name = type(stage).__name__
        try:
            cardinality = foundation.build_contract(stage).cardinality
        except Exception:  # noqa: BLE001, S112 - region() already refuses on an unreadable contract
            continue
        if cardinality not in preserving:
            continue
        art = published.get(index)
        if art is not None and art.rows_in and art.rows_out and art.rows_in != art.rows_out:
            found.append(
                f"{name} declares {cardinality} but its prior run took {art.rows_in} row(s) "
                f"and produced {art.rows_out}"
            )
        counts = _counts_for(per_stage, name)
        if counts and counts[0] != counts[1]:
            found.append(
                f"{name} declares {cardinality} but its prior run recorded {counts[0]} row(s) in and {counts[1]} out"
            )
    return found


def _counts_for(per_stage: dict[str, Any], name: str) -> tuple[int, int] | None:
    """``(rows in, rows out)`` a stage logged, or ``None`` when it logs no accounting.

    Stage names in ``per_stage_metrics`` come from the runtime and may drop the ``Stage``
    suffix, which is the same mismatch ``planner`` handles when reading measured resources.
    """
    metrics = per_stage.get(name) or per_stage.get(name.removesuffix("Stage")) or {}
    got = {}
    for key in ("custom.input_count", "custom.output_count"):
        entry = metrics.get(key)
        if isinstance(entry, dict) and entry.get("sum") is not None:
            got[key] = int(entry["sum"])
    if len(got) != 2:  # noqa: PLR2004 - both halves of the pair or nothing to compare
        return None
    return got["custom.input_count"], got["custom.output_count"]


def provenance(uri: str, *, inventory: dict[str, str], root: str) -> tuple[str, str]:
    """Find the column in a prior artifact's rows that names each row's input file.

    Returns ``(key, reason)``: a column name, or ``""`` and why none qualifies. Qualifying
    means every sampled row's value resolves to a file the inventory knows -- checked against
    the artifact's real rows, so a pipeline that rewrote its paths to derived chunks (or built
    fresh rows that no longer mention their origin) yields no key and no delta.

    Reading a prior artifact here is bookkeeping, not planning: the merge has to know which
    rows belong to which file. It is the same file whose digest ``artifacts`` already verifies.
    """
    rows = list(_read_rows(uri, limit=_PROVENANCE_SAMPLE))
    if not rows:
        return "", f"no rows could be read from the prior output at {uri}"
    known = set(inventory)
    candidates = [k for k, v in rows[0].items() if isinstance(v, str) and v]
    for key in candidates:
        hits = 0
        for row in rows:
            value = row.get(key)
            if not isinstance(value, str) or _relpath(value, root) not in known:
                hits = -1
                break
            hits += 1
        if hits > 0:
            return key, ""
    return "", (
        "no column in the prior output names a file from the input inventory "
        f"(columns: {sorted(rows[0])[:12]}), so rows cannot be traced back to the files they came from"
    )


def _relpath(value: str, root: str) -> str:
    """A row's path value as the inventory spells it, or the value unchanged when it is not one."""
    try:
        return os.path.relpath(os.path.abspath(os.path.expanduser(value)), root)
    except (OSError, ValueError):
        return value


def _read_rows(uri: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Rows of a JSONL artifact, skipping anything unparseable (a partial line is not a row)."""
    path = os.path.expanduser(uri)
    if not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    with contextlib.suppress(OSError, UnicodeError), open(path, encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(raw)
                if isinstance(row, dict):
                    out.append(row)
            if limit is not None and len(out) >= limit:
                break
    return out


def plan(
    recipe: Recipe,
    *,
    dataset_key: str,
    inventory: dict[str, str] | None,
    inventory_root: str,
) -> Delta:
    """Decide whether the changed files alone can be run, and refuse with a reason if not.

    Called on the reuse miss path, where ``dataset_key`` matched nothing: a changed corpus
    changes the key, so a miss is exactly the situation a delta is for.
    """
    if not dataset_key or inventory is None:
        return Delta(reason="no per-file inventory for this input, so a changed-file delta cannot be computed")

    prior_key, artifacts_seen = _prior_run_of(recipe, dataset_key)
    if not prior_key:
        return Delta(reason="this pipeline has no prior run on any dataset to compare against")

    change = classify(_coverage_of(artifacts_seen), inventory)
    if change is None:
        return Delta(
            reason=(
                "the prior run recorded no per-file inventory (it predates coverage, or its corpus "
                "was too large to record), so which files changed cannot be established"
            )
        )
    if change.kind == "identical":
        return Delta(
            change=change,
            reason=(
                "the files are identical yet the dataset key differs; something outside the file "
                "list changed (a manifest's own bytes, or the path the corpus is read from)"
            ),
        )
    if change.kind == "unrelated":
        return Delta(
            change=change,
            reason=(
                f"the prior run shares no file with this input ({len(change.removed)} gone, "
                f"{len(change.added)} new), so this is a different dataset rather than a changed one"
            ),
        )

    return _plan_from(recipe, change, prior_key, artifacts_seen, inventory_root=inventory_root)


def _plan_from(  # noqa: PLR0911 - see plan()
    recipe: Recipe,
    change: Change,
    prior_key: str,
    prior: list[Artifact],
    *,
    inventory_root: str,
) -> Delta:
    """The half of :func:`plan` that runs once the change itself is understood."""
    from nemo_curator.audio_agent import artifacts as art_mod
    from nemo_curator.audio_agent.continuation import _resume_breaks_on_disk_boundary

    plans = art_mod.plan_steps(recipe, prior_key)
    by_key = {a.step_key: a for a in prior}
    resumable = [p for p in plans if p.persists() and p.step_key in by_key]
    if not resumable:
        return Delta(
            change=change,
            reason=(
                "the prior run persisted nothing that could be resumed from; add a checkpoint "
                "(add-checkpoint) so the expensive stages have somewhere to leave their result"
            ),
        )

    depth, ended = region(recipe, upto=len(plans))
    step = _deepest_within(resumable, depth)
    if step is None:
        return Delta(
            change=change,
            reason=(
                f"per-file work stays independent only through the first {depth} stage(s) "
                f"({ended or 'end of the recipe'}), and nothing is persisted that early; "
                "the whole pipeline has to see the corpus together"
            ),
        )
    artifact = by_key[step.step_key]
    prefix = step.index + 1

    disproved = contradictions(
        recipe,
        upto=prefix,
        published={p.index: by_key[p.step_key] for p in resumable},
        per_stage=_metrics_of(artifact),
    )
    if disproved:
        return Delta(change=change, reason="a stage's own prior run contradicts its contract: " + "; ".join(disproved))

    lost = _resume_breaks_on_disk_boundary(recipe, prefix)
    if lost and prefix < len(plans):
        return Delta(
            change=change,
            reason=(
                f"the stages after {step.stage_ref} need in-memory state a manifest cannot carry "
                f"({lost}), so the merged rows could not be handed on"
            ),
        )

    owned, why = sinks(recipe, prefix=prefix, plans=plans, published={p.index: by_key[p.step_key] for p in resumable})
    if why:
        return Delta(change=change, reason=why)
    coverage = _coverage_of(prior) or {}
    owned, why = traced(owned, inventory=coverage, root=inventory_root, stale=set(change.stale))
    if why:
        return Delta(change=change, reason=why)

    resume = next((s for s in owned if s.index == step.index), None)
    if resume is None:  # the resume point is a manifest by construction; belt for a future kind
        return Delta(change=change, reason=f"{step.stage_ref} does not own a manifest the merge can rewrite")
    return Delta(
        status="ready",
        change=change,
        prior_step_key=step.step_key,
        prior_dataset_key=prior_key,
        uri=artifact.uri,
        kind=artifact.kind,
        stage_ref=step.stage_ref,
        prefix=prefix,
        provenance_key=resume.key,
        files=tuple(os.path.join(inventory_root, f) for f in change.touched),
        drops=resume.drops,
        keeps=resume.keeps,
        sinks=tuple(owned),
        estimated_saving_sec=_saving(artifact, change),
        notes=_notes(change, ended, depth, len(plans)),
    )


# --------------------------------------------------------------------------- execution
# The param every source stage that can be narrowed to a subset accepts. A source without it
# cannot run a delta, and says so by name rather than being fed a filtered copy of the input.
_INCLUDE_PARAM = "include_files"


@dataclass(frozen=True)
class Sink:
    """A manifest a prefix stage truncates and rewrites, and therefore one the merge must own."""

    index: int
    param: str
    uri: str
    step_key: str
    stage_ref: str = ""
    # Derived per sink, not once for the pipeline: the column that names a row's origin at the
    # top of the recipe may have been rewritten by the time a deeper manifest is written (a
    # resampled copy replaces the source path), and a sink whose rows no longer trace back
    # cannot be merged even though a shallower one can.
    key: str = ""
    keeps: int = 0
    drops: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage_ref,
            "uri": self.uri,
            "traced_by": self.key,
            "rows_kept": self.keeps,
            "rows_dropped": self.drops,
        }


def sinks(
    recipe: Recipe, *, prefix: int, plans: list[StepPlan], published: dict[int, Artifact]
) -> tuple[list[Sink], str]:
    """Every manifest the delta's stages would rewrite, or why the delta cannot be run.

    Only manifests: ``ManifestWriterStage.setup()`` truncates its file, so a run over three
    files would leave a three-row manifest where a thousand rows used to be, and every one of
    them has to be merged back. Directory outputs are the opposite -- a resampled copy or a
    split chunk is written per file, so the new files simply add theirs beside the existing
    ones and nothing needs merging. Hence the kind decides, not the parameter name.

    A manifest with no prior artifact is a refusal: there is nothing trustworthy to merge the
    new rows into, and writing the delta alone would silently shrink the corpus.
    """
    found: list[Sink] = []
    for step in plans[:prefix]:
        if not step.persists():
            continue
        from nemo_curator.audio_agent import artifacts as art_mod

        kind = art_mod.classify_output(step.uri) or step.kind
        if kind != "manifest":
            continue
        art = published.get(step.index)
        if art is None:
            return [], (
                f"{step.stage_ref} rewrites {step.uri} from scratch on every run, and no prior "
                f"result for it was published, so the rows it already holds cannot be preserved"
            )
        param = _uri_param(recipe, step.index)
        if not param:
            return [], f"the output location of {step.stage_ref} is not a parameter that can be redirected"
        found.append(
            Sink(index=step.index, param=param, uri=step.uri, step_key=art.step_key, stage_ref=step.stage_ref)
        )
    return found, ""


def traced(sinks_: list[Sink], *, inventory: dict[str, str], root: str, stale: set[str]) -> tuple[list[Sink], str]:
    """Attach each sink's provenance column and row accounting, or say which one has none."""
    out: list[Sink] = []
    for sink in sinks_:
        key, why = provenance(sink.uri, inventory=inventory, root=root)
        if not key:
            return [], f"{sink.stage_ref} -> {sink.uri}: {why}"
        rows = _read_rows(sink.uri)
        drops = sum(1 for r in rows if _relpath(str(r.get(key) or ""), root) in stale)
        if stale and not drops:
            return [], (
                f"{len(stale)} file(s) changed or were removed, but no row in {sink.uri} traces back "
                f"to any of them through {key!r}; dropping nothing would leave stale results in place"
            )
        out.append(
            Sink(
                index=sink.index,
                param=sink.param,
                uri=sink.uri,
                step_key=sink.step_key,
                stage_ref=sink.stage_ref,
                key=key,
                keeps=len(rows) - drops,
                drops=drops,
            )
        )
    return out, ""


def _uri_param(recipe: Recipe, index: int) -> str:
    """Which param of stage ``index`` holds the output path ``output_uri`` picked."""
    from nemo_curator.audio_agent.artifacts import _URI_PREFERENCE

    params = recipe.stages[index].params
    return next((k for k in _URI_PREFERENCE if isinstance(params.get(k), str) and params[k]), "")


def prefix_recipe(
    recipe: Recipe,
    *,
    prefix: int,
    files: tuple[str, ...],
    sandbox: str,
    sinks_: list[Sink],
) -> tuple[Recipe | None, dict[str, str], str]:
    """The recipe that runs the changed files through the delta's stages.

    Returns ``(recipe, {real manifest: sandbox manifest}, error)``. It is the user's own
    recipe, truncated at the resume point, with the source narrowed to the changed files and
    each manifest writer pointed into ``sandbox`` -- so no stage is told anything about running
    incrementally, and the user's manifests are never truncated by a partial run.
    """
    from nemo_curator.audio_agent.recipe import Recipe, StageRef

    if not 0 < prefix <= len(recipe.stages):
        return None, {}, f"delta prefix {prefix} is outside the recipe ({len(recipe.stages)} stages)"
    source = recipe.stages[0]
    if not _accepts_include(source.ref):
        return (
            None,
            {},
            (
                f"{source.ref} cannot be narrowed to a subset of the input (no {_INCLUDE_PARAM!r} "
                f"parameter), so the changed files cannot be run on their own"
            ),
        )
    redirect = {s.uri: os.path.join(sandbox, f"{s.index:02d}-{os.path.basename(s.uri)}") for s in sinks_}
    by_index = {s.index: s for s in sinks_}
    stages: list[StageRef] = []
    for index, stage in enumerate(recipe.stages[:prefix]):
        params = dict(stage.params)
        if index == 0:
            params[_INCLUDE_PARAM] = list(files)
        sink = by_index.get(index)
        if sink is not None:
            params[sink.param] = redirect[sink.uri]
        stages.append(StageRef(ref=stage.ref, params=params))
    built = Recipe(
        stages=stages,
        inputs=dict(recipe.inputs),
        preset=recipe.preset,
        acceptance_criteria=list(recipe.acceptance_criteria),
        rationale=recipe.rationale,
        name=f"{recipe.name}_delta",
        knowledge_version=recipe.knowledge_version,
        parent_run_id=recipe.parent_run_id,
    )
    return built.freeze(), redirect, ""


def _accepts_include(ref: str) -> bool:
    """Whether a source stage can be restricted to a named list of files."""
    import dataclasses

    from nemo_curator.audio_agent._resolve import resolve_stage_class

    try:
        cls = resolve_stage_class(ref)
    except Exception:  # noqa: BLE001 - an unknown ref is reported by validation, not here
        return False
    if not dataclasses.is_dataclass(cls):
        return False
    return any(f.name == _INCLUDE_PARAM for f in dataclasses.fields(cls))


def merge(sink: Sink, *, produced: str, stale: set[str], key: str, root: str) -> tuple[int, int, str]:
    """Replace a manifest with (its surviving rows + the delta's rows). ``(kept, added, error)``.

    Written to a temporary file beside the target and moved into place, so a reader either sees
    the whole previous manifest or the whole merged one. Prior rows keep their order and the new
    ones follow; row order carries no meaning to preserve, since parallel writers append in
    completion order and no two full runs need agree on it either.
    """
    surviving = [r for r in _read_rows(sink.uri) if _relpath(str(r.get(key) or ""), root) not in stale]
    fresh = _read_rows(produced)
    if fresh and surviving:
        mismatch = set(fresh[0]) ^ set(surviving[0])
        if mismatch:
            return (
                0,
                0,
                (
                    "the delta's rows do not have the same columns as the rows already there "
                    f"({sorted(mismatch)[:8]}), so merging them would produce a manifest no run could"
                ),
            )
    tmp = f"{sink.uri}.delta-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in (*surviving, *fresh))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, sink.uri)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return 0, 0, f"the merged manifest could not be written to {sink.uri}: {exc}"
    return len(surviving), len(fresh), ""


def republish(  # noqa: PLR0913 - an artifact record's fields, none of them derivable from the others
    recipe: Recipe,
    decision: Delta,
    *,
    dataset_key: str,
    fingerprint_tier: str,
    inventory: dict[str, str],
    run_id: str,
    added_sec: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Register the merged manifests as artifacts of the enlarged corpus.

    This is the step that makes a delta worth running twice: the merged manifest holds what a
    full run over every file would have produced, so it is published under the step key the
    full pipeline has for the CURRENT dataset -- and the next run finds it by an ordinary probe,
    with no delta-specific lookup anywhere in the reuse path. Coverage becomes the union, which
    is what a later comparison has to be made against.

    Cost is carried forward and added to, not reset: the merged result really did take the
    prior run's hours plus this delta's minutes, and a later decision about whether reusing it
    is worth a question depends on that number being the truth.
    """
    from nemo_curator.audio_agent import artifacts as art_mod

    plans = {p.index: p for p in art_mod.plan_steps(recipe, dataset_key)}
    out: list[dict[str, Any]] = []
    problems: list[str] = []
    for sink in decision.sinks:
        step = plans.get(sink.index)
        if step is None:
            problems.append(f"no step plan for stage {sink.index}; {sink.uri} was not republished")
            continue
        prior = art_mod.load(sink.step_key)
        art = art_mod.Artifact(
            step_key=step.step_key,
            input_key=step.input_key,
            stage_ref=step.stage_ref,
            stage_index=step.index,
            semantic_params=step.semantic_params,
            contract_hash=recipe.contract_hash,
            uri=sink.uri,
            kind=art_mod.classify_output(sink.uri) or step.kind,
            rows_in=getattr(prior, "rows_in", 0),
            produced_roles=list(getattr(prior, "produced_roles", []) or []),
            produced_keys=list(getattr(prior, "produced_keys", []) or []),
            cumulative_sec=round(float(getattr(prior, "cumulative_sec", 0.0)) + added_sec, 3),
            gpu_seconds=float(getattr(prior, "gpu_seconds", 0.0)),
            device=str(getattr(prior, "device", "") or ""),
            dataset_key=dataset_key,
            fingerprint_tier=fingerprint_tier,
            covers_files=len(inventory),
            impl_version=step.impl_version,
            code_version=art_mod.code_version(),
            model_version=step.model_version,
            deterministic=step.deterministic,
            ttl_sec=step.ttl_sec,
            run_id=run_id,
        )
        try:
            art_mod.publish(art)
            art_mod.save_coverage(art.step_key, inventory)
        except Exception as exc:  # noqa: BLE001 - the merge already succeeded; publication is bookkeeping
            problems.append(f"{sink.uri} was merged but not republished: {type(exc).__name__}: {exc}")
            continue
        out.append({"step_key": art.step_key, "stage": art.stage_ref, "uri": art.uri, "rows": art.rows_out})
    return out, problems


def _notes(change: Change, ended: str, depth: int, total: int) -> tuple[str, ...]:
    """What the caller should say out loud even though the delta is sound."""
    out: list[str] = []
    if ended and depth < total:
        out.append(f"stages after the checkpoint rerun over every row, because {ended}")
    if change.modified:
        out.append(f"{len(change.modified)} file(s) changed in place; their prior rows are dropped and recomputed")
    if change.removed:
        out.append(f"{len(change.removed)} file(s) are gone; their prior rows are dropped and not replaced")
    return tuple(out)


def _saving(artifact: Artifact, change: Change) -> float:
    """Seconds the delta is expected to save: the prior cost, less the changed files' share.

    Per-file cost from the artifact's own measured ``cumulative_sec`` over the files it
    covered, which is the only rate this pipeline has actually demonstrated.
    """
    covered = artifact.covers_files or 0
    if not covered or artifact.cumulative_sec <= 0:
        return 0.0
    per_file = artifact.cumulative_sec / covered
    return max(0.0, per_file * len(change.unchanged))


def _deepest_within(resumable: list[StepPlan], depth: int) -> StepPlan | None:
    """The last persisted step at or before ``depth`` -- the deepest legal resume point."""
    inside = [p for p in resumable if p.index < depth]
    return inside[-1] if inside else None


def _prior_run_of(recipe: Recipe, dataset_key: str) -> tuple[str, list[Artifact]]:
    """The most recent other dataset this exact pipeline ran on, and its artifacts.

    Recomputing the step-key chain against a known dataset key is how ``reuse`` already turns
    a miss into "you ran this, but your data moved on"; a delta needs the artifacts themselves.
    """
    from nemo_curator.audio_agent import artifacts as art_mod
    from nemo_curator.audio_agent.reuse import _known_dataset_keys

    best: tuple[str, list[Artifact]] = ("", [])
    for other in _known_dataset_keys():
        if not other or other == dataset_key:
            continue
        found = [a for a in (art_mod.load(p.step_key) for p in art_mod.plan_steps(recipe, other)) if a is not None]
        if found and (not best[1] or _newest(found) > _newest(best[1])):
            best = (other, found)
    return best


def _newest(artifacts: list[Artifact]) -> str:
    return max((a.created_at or "") for a in artifacts)


def _coverage_of(artifacts: list[Artifact]) -> dict[str, str] | None:
    """The inventory behind a prior run: one coverage file, shared by all of its steps."""
    from nemo_curator.audio_agent import artifacts as art_mod

    for art in artifacts:
        found = art_mod.load_coverage(art.step_key)
        if found is not None:
            return found
    return None


def _metrics_of(artifact: Artifact) -> dict[str, Any]:
    """``per_stage_metrics`` of the run that produced an artifact (empty when unavailable)."""
    from nemo_curator.audio_agent import run_store

    with contextlib.suppress(Exception):
        record = run_store.load(artifact.run_id) if artifact.run_id else None
        if record is not None:
            return dict(getattr(record, "per_stage_metrics", {}) or {})
    return {}
