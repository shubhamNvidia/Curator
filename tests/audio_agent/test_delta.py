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

"""Delta reuse: which files changed, how deep per-file work is independent, and the merge.

The property under test throughout is that a delta either produces what a full run would
have produced, or refuses and says which file or stage stopped it. See
``nemo_curator/audio_agent/REUSE_ARCHITECTURE.md`` §7.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from nemo_curator.audio_agent import artifacts, delta, profiler
from nemo_curator.audio_agent.recipe import Recipe

if TYPE_CHECKING:
    from pathlib import Path

_KEY = "dataset-key-now"
_PRIOR = "dataset-key-before"


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated artifact/run store so tests never touch the developer's real history."""
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _corpus(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """A folder of files and the inventory the profiler computes for it."""
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"\0" * 16)
    return {name: f"16|{i}" for i, name in enumerate(names)}


def _pipeline(tmp_path: Path, source: Path) -> tuple[Recipe, str]:
    """reader -> duration -> writer: every stage declares per-row independence."""
    out = str(tmp_path / "out.jsonl")
    rec = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
                {"ref": "GetAudioDurationStage", "params": {}},
                {"ref": "ManifestWriterStage", "params": {"output_path": out}},
            ]
        }
    ).freeze()
    return rec, out


def _publish(
    rec: Recipe,
    index: int,
    *,
    dataset_key: str,
    rows: list[dict[str, Any]],
    coverage: dict[str, str] | None,
    **kw: Any,  # noqa: ANN401 - forwards arbitrary Artifact fields on purpose
) -> artifacts.Artifact:
    """Publish one step of ``rec`` with real output rows and (optionally) its coverage."""
    plan = artifacts.plan_steps(rec, dataset_key)[index]
    with open(plan.uri, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    art = artifacts.publish(
        artifacts.Artifact(
            step_key=plan.step_key,
            input_key=plan.input_key,
            stage_ref=plan.stage_ref,
            stage_index=plan.index,
            semantic_params=plan.semantic_params,
            uri=plan.uri,
            kind=plan.kind,
            dataset_key=dataset_key,
            fingerprint_tier="stat",
            impl_version=plan.impl_version,
            code_version=artifacts.code_version(),
            deterministic=plan.deterministic,
            cumulative_sec=kw.pop("cumulative_sec", 600.0),
            covers_files=len(coverage or {}),
            **kw,
        )
    )
    if coverage is not None:
        artifacts.save_coverage(art.step_key, coverage)
    return art


class TestNarrowingIsNotInvisible:
    """``include_files`` must reach the step key, or a partial manifest passes as a whole one.

    Nothing else stops it: a narrowed run reads the same folder, so it carries the same dataset
    key, and it writes the user's manifest path. Were the param filtered out of the semantic
    identity -- it reads like an execution knob, and the two frozensets that do that filtering
    are a plausible place to put it -- a one-file run would publish under the key the full
    pipeline probes, and the next scan would answer ``already_done`` from a manifest holding one
    row of a thousand. The delta relies on this, which is why it is asserted here rather than
    left to whoever next tidies those lists.
    """

    def test_a_narrowed_source_does_not_share_the_full_runs_step_keys(self, tmp_path: Path) -> None:
        source = tmp_path / "m.jsonl"
        source.write_text(json.dumps({"audio_filepath": str(tmp_path / "a.wav")}) + "\n")
        whole, _ = _pipeline(tmp_path, source)
        narrowed = Recipe.from_dict(
            {
                "stages": [
                    {
                        "ref": "ManifestReader",
                        "params": {"manifest_path": str(source), "include_files": [str(tmp_path / "a.wav")]},
                    },
                    *[s.to_dict() for s in whole.stages[1:]],
                ]
            }
        ).freeze()

        mine = artifacts.step_keys(narrowed, _KEY)
        theirs = artifacts.step_keys(whole, _KEY)
        assert mine[0] != theirs[0], "include_files must be part of the source stage's semantic identity"
        assert not set(mine) & set(theirs), "every key below the narrowed source must differ too"

    def test_a_narrowable_source_declares_that_narrowing_it_is_safe(self) -> None:
        """The conformance rule, on the two sources that carry the param today."""
        from nemo_curator.stages.audio._conformance import assert_contract_wellformed
        from nemo_curator.stages.audio.common import CreateInitialManifestAudioFolderStage, ManifestReaderStage

        for cls in (CreateInitialManifestAudioFolderStage, ManifestReaderStage):
            contract = assert_contract_wellformed(cls)
            assert contract.gates.per_row_independent is True, cls.__name__


class TestInventory:
    """The profiler records WHICH files back the dataset key, not just their digest."""

    def test_a_folder_scan_remembers_every_file(self, tmp_path: Path) -> None:
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.fingerprint_tier == "stat"
        assert sorted(prof.inventory) == ["a.wav", "b.wav"]
        assert prof.inventory_root == str(tmp_path / "audio")

    def test_the_inventory_stays_out_of_the_serialized_profile(self, tmp_path: Path) -> None:
        """It rides on every record and report that carries a profile; a corpus would bury them."""
        _corpus(tmp_path / "audio", ("a.wav",))
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.inventory
        assert "inventory" not in prof.to_dict()

    def test_an_edited_manifest_row_moves_that_files_token(self, tmp_path: Path) -> None:
        """A corrected transcript with the wav untouched is a change to that file's work."""
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        manifest = tmp_path / "m.jsonl"
        rows = [{"audio_filepath": str(tmp_path / "audio" / n), "text": "hi"} for n in ("a.wav", "b.wav")]
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        before = profiler.profile_data(str(manifest)).inventory

        rows[0]["text"] = "bye"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        after = profiler.profile_data(str(manifest)).inventory

        # Keys are relative to the manifest's directory, which is what holds the audio folder.
        assert before["audio/a.wav"] != after["audio/a.wav"]
        assert before["audio/b.wav"] == after["audio/b.wav"]

    def test_an_incomplete_scan_records_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A partial inventory would report unstattable files as removed and delete their rows."""
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        real = profiler.os.stat

        def flaky(path: str, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
            if str(path).endswith("b.wav"):
                msg = "no"
                raise OSError(msg)
            return real(path, *a, **kw)

        monkeypatch.setattr(profiler.os, "stat", flaky)
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.fingerprint_tier == "shape"
        assert prof.inventory == {}

    def test_a_corpus_past_the_cap_keeps_the_key_and_declines_the_inventory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(profiler, "_MAX_INVENTORY", 1)
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.fingerprint_tier == "stat"  # reuse is unaffected
        assert prof.inventory == {}
        assert any("delta" in note for note in prof.notes)


class TestClassify:
    """Comparing two inventories names the files, and refuses to guess when one is missing."""

    def test_a_new_file_is_added_only(self) -> None:
        change = delta.classify({"a": "1", "b": "1"}, {"a": "1", "b": "1", "c": "1"})
        assert change is not None
        assert change.kind == "added_only"
        assert change.added == ("c",)
        assert change.touched == ("c",)
        assert change.stale == ()

    def test_an_edited_file_is_stale_and_must_be_rerun(self) -> None:
        change = delta.classify({"a": "1", "b": "1"}, {"a": "2", "b": "1"})
        assert change is not None
        assert change.kind == "changed"
        assert change.modified == ("a",)
        assert change.touched == ("a",)  # rerun it
        assert change.stale == ("a",)  # and drop what it produced before

    def test_a_deleted_file_is_stale_but_not_rerun(self) -> None:
        change = delta.classify({"a": "1", "b": "1"}, {"a": "1"})
        assert change is not None
        assert change.kind == "removed"
        assert change.touched == ()
        assert change.stale == ("b",)

    def test_the_same_files_are_identical(self) -> None:
        change = delta.classify({"a": "1"}, {"a": "1"})
        assert change is not None
        assert change.kind == "identical"

    def test_two_corpora_sharing_no_file_are_not_one_that_changed(self) -> None:
        """Subtracting one dataset from another would drop every row and call it incremental."""
        change = delta.classify({"a": "1"}, {"z": "1"})
        assert change is not None
        assert change.kind == "unrelated"

    def test_an_unrecorded_inventory_is_not_an_empty_one(self) -> None:
        assert delta.classify(None, {"a": "1"}) is None
        assert delta.classify({"a": "1"}, None) is None


class TestRegion:
    """How deep per-file work stays independent, read from what stages declare."""

    def test_a_per_row_pipeline_is_traceable_end_to_end(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        depth, reason = delta.region(rec, upto=3)
        assert (depth, reason) == (3, "")

    def test_a_corpus_statistic_stage_ends_the_region_and_is_named(self, tmp_path: Path) -> None:
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": str(tmp_path / "s.json")}},
                ]
            }
        ).freeze()
        depth, reason = delta.region(rec, upto=2)
        assert depth == 1
        assert "PretrainMetricsAggregator" in reason
        assert "whole corpus" in reason

    def test_an_undeclared_stage_ends_the_region_rather_than_being_assumed_safe(self, tmp_path: Path) -> None:
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "ComputeWERStage", "params": {}},
                ]
            }
        ).freeze()
        depth, reason = delta.region(rec, upto=2)
        assert depth == 1
        assert "ComputeWER" in reason
        assert "per_row_independent" in reason


class TestContradictions:
    """A declaration the stage's own prior run disproves is worse than no declaration."""

    def test_a_row_preserving_stage_that_changed_the_row_count_is_caught(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        art = _publish(rec, 2, dataset_key=_PRIOR, rows=[], coverage={}, rows_in=10, rows_out=3)
        found = delta.contradictions(rec, upto=3, published={2: art}, per_stage={})
        assert found
        assert "ManifestWriterStage" in found[0]
        assert "10" in found[0]
        assert "3" in found[0]

    def test_recorded_filter_counts_can_disprove_it_too(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        metrics = {
            "GetAudioDuration": {
                "custom.input_count": {"sum": 12.0},
                "custom.output_count": {"sum": 4.0},
            }
        }
        found = delta.contradictions(rec, upto=3, published={}, per_stage=metrics)
        assert found
        assert "GetAudioDurationStage" in found[0]

    def test_agreeing_numbers_are_no_contradiction(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        art = _publish(rec, 2, dataset_key=_PRIOR, rows=[], coverage={}, rows_in=7, rows_out=7)
        assert delta.contradictions(rec, upto=3, published={2: art}, per_stage={}) == []


class TestProvenance:
    """Which prior row came from which input file is derived from the rows, never assumed."""

    def test_the_column_naming_the_input_file_is_found(self, tmp_path: Path) -> None:
        inventory = _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        out = tmp_path / "prior.jsonl"
        out.write_text(
            "".join(
                json.dumps({"audio_filepath": str(tmp_path / "audio" / n), "duration": 1.0}) + "\n"
                for n in ("a.wav", "b.wav")
            )
        )
        key, why = delta.provenance(str(out), inventory=inventory, root=str(tmp_path / "audio"))
        assert (key, why) == ("audio_filepath", "")

    def test_rows_that_no_longer_name_their_origin_get_no_delta(self, tmp_path: Path) -> None:
        """A pipeline that rewrote paths to derived chunks cannot be subtracted per file."""
        inventory = _corpus(tmp_path / "audio", ("a.wav",))
        out = tmp_path / "prior.jsonl"
        out.write_text(json.dumps({"audio_filepath": "/somewhere/else/chunk-000.wav", "duration": 1.0}) + "\n")
        key, why = delta.provenance(str(out), inventory=inventory, root=str(tmp_path / "audio"))
        assert key == ""
        assert "cannot be traced" in why

    def test_an_unreadable_prior_output_is_reported_rather_than_guessed(self, tmp_path: Path) -> None:
        key, why = delta.provenance(str(tmp_path / "missing.jsonl"), inventory={"a.wav": "1"}, root=str(tmp_path))
        assert key == ""
        assert "no rows" in why


class TestPlan:
    """The whole decision, on a real store: ready with names, or refused with a reason."""

    def _prior_run(self, tmp_path: Path, names: tuple[str, ...]) -> tuple[Recipe, dict[str, str]]:
        """Publish a prior run of the pipeline over ``names``, with coverage and real rows."""
        audio = tmp_path / "audio"
        inventory = _corpus(audio, names)
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("".join(json.dumps({"audio_filepath": str(audio / n)}) + "\n" for n in names))
        rec, _ = _pipeline(tmp_path, manifest)
        rows = [{"audio_filepath": str(audio / n), "duration": 1.0} for n in names]
        _publish(rec, 2, dataset_key=_PRIOR, rows=rows, coverage=inventory)
        return rec, inventory

    def test_one_added_file_is_all_that_needs_running(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        now = {**inventory, "c.wav": "16|99"}

        decision = delta.plan(rec, dataset_key=_KEY, inventory=now, inventory_root=str(tmp_path / "audio"))

        assert decision.status == "ready"
        assert decision.change is not None
        assert decision.change.kind == "added_only"
        assert decision.files == (str(tmp_path / "audio" / "c.wav"),)
        assert decision.keeps == 2  # both prior rows survive
        assert decision.drops == 0
        assert decision.provenance_key == "audio_filepath"
        assert decision.estimated_saving_sec > 0

    def test_an_edited_file_drops_its_prior_row_and_reruns_it(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        now = {**inventory, "a.wav": "16|changed"}

        decision = delta.plan(rec, dataset_key=_KEY, inventory=now, inventory_root=str(tmp_path / "audio"))

        assert decision.status == "ready"
        assert decision.files == (str(tmp_path / "audio" / "a.wav"),)
        assert (decision.drops, decision.keeps) == (1, 1)

    def test_a_removed_file_drops_its_rows_and_runs_nothing(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        now = {k: v for k, v in inventory.items() if k != "b.wav"}

        decision = delta.plan(rec, dataset_key=_KEY, inventory=now, inventory_root=str(tmp_path / "audio"))

        assert decision.status == "ready"
        assert decision.files == ()
        assert (decision.drops, decision.keeps) == (1, 1)
        assert any("gone" in note for note in decision.notes)

    def test_without_an_inventory_there_is_no_delta(self, tmp_path: Path) -> None:
        rec, _ = self._prior_run(tmp_path, ("a.wav",))
        decision = delta.plan(rec, dataset_key=_KEY, inventory=None, inventory_root="")
        assert decision.status == "none"
        assert "no per-file inventory" in decision.reason

    def test_a_prior_run_without_coverage_cannot_be_compared(self, tmp_path: Path) -> None:
        """Runs published before coverage existed are not wrong, they are unusable for a delta."""
        audio = tmp_path / "audio"
        inventory = _corpus(audio, ("a.wav",))
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": str(audio / "a.wav")}) + "\n")
        rec, _ = _pipeline(tmp_path, manifest)
        _publish(rec, 2, dataset_key=_PRIOR, rows=[{"audio_filepath": str(audio / "a.wav")}], coverage=None)

        decision = delta.plan(
            rec,
            dataset_key=_KEY,
            inventory={**inventory, "b.wav": "16|1"},
            inventory_root=str(audio),
        )
        assert decision.status == "none"
        assert "no per-file inventory" in decision.reason

    def test_nothing_to_compare_against_is_said_plainly(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        decision = delta.plan(rec, dataset_key=_KEY, inventory={"a.wav": "1"}, inventory_root=str(tmp_path))
        assert decision.status == "none"
        assert "no prior run" in decision.reason

    def test_a_different_dataset_is_refused_rather_than_subtracted(self, tmp_path: Path) -> None:
        rec, _ = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        decision = delta.plan(
            rec,
            dataset_key=_KEY,
            inventory={"x.wav": "16|1", "y.wav": "16|2"},
            inventory_root=str(tmp_path / "audio"),
        )
        assert decision.status == "none"
        assert "different dataset" in decision.reason

    def test_identical_files_under_a_new_key_say_what_else_moved(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav",))
        decision = delta.plan(rec, dataset_key=_KEY, inventory=inventory, inventory_root=str(tmp_path / "audio"))
        assert decision.status == "none"
        assert "identical" in decision.reason

    def test_nothing_persisted_inside_the_region_means_the_corpus_must_be_seen_together(self, tmp_path: Path) -> None:
        """The only saved output sits behind a stage that totals the corpus, so it cannot be split."""
        audio = tmp_path / "audio"
        inventory = _corpus(audio, ("a.wav",))
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": str(tmp_path / "s.json")}},
                ]
            }
        ).freeze()
        _publish(rec, 1, dataset_key=_PRIOR, rows=[{"audio_filepath": str(audio / "a.wav")}], coverage=inventory)

        decision = delta.plan(
            rec,
            dataset_key=_KEY,
            inventory={**inventory, "b.wav": "16|9"},
            inventory_root=str(audio),
        )
        assert decision.status == "none"
        assert "PretrainMetricsAggregator" in decision.reason
        assert "corpus together" in decision.reason
