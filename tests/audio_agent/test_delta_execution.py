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

"""Executing a delta: run the changed files, merge, republish -- and prove the equivalence.

The claim these pin down is the one that makes the feature safe to trust: a full run over N
files and a run over N-1 files followed by a delta over the last one produce the same manifest.
"""

from __future__ import annotations

import json
import math
import os
import struct
import wave
from typing import TYPE_CHECKING, Any

import pytest

from nemo_curator.audio_agent import artifacts, delta, verbs
from nemo_curator.audio_agent.recipe import Recipe

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _wav(path: Path, *, seconds: float = 0.25, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = int(seconds * rate)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(frames)))


def _recipe(folder: Path, out: Path) -> Recipe:
    """folder source -> duration -> manifest: CPU only, and every stage declares independence."""
    return Recipe.from_dict(
        {
            "stages": [
                {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                {"ref": "GetAudioDurationStage", "params": {}},
                {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}},
            ]
        }
    ).freeze()


def _rows(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(rec: Recipe, folder: Path) -> dict[str, Any]:
    return verbs.run(rec, confirm=True, data=str(folder))


@pytest.mark.usefixtures("store")
class TestDeltaExecution:
    """The verb, against a pipeline that really executes."""

    def test_adding_a_file_runs_only_that_file_and_completes_the_manifest(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        assert len(_rows(out)) == 2

        _wav(folder / "c.wav")
        # Unconfirmed, this is a card and a refusal -- the same shape run() returns.
        card = verbs.delta_run(rec, data=str(folder))
        assert card["status"] == "refused", card
        assert card["delta"]["change"]["added"] == 1
        assert card["delta"]["file_count"] == 1
        assert card["delta"]["change"]["added_files"] == ["c.wav"]

        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        assert done["ran_files"] == [str(folder / "c.wav")]
        # As a set: parallel writers append in completion order, so no run has a fixed row order.
        assert {os.path.basename(r["audio_filepath"]) for r in _rows(out)} == {"a.wav", "b.wav", "c.wav"}
        assert done["merged"][0]["rows_kept"] == 2
        assert done["merged"][0]["rows_added"] == 1
        assert done["published"], "the merged manifest must be findable by an ordinary reuse probe"

    def test_the_merged_manifest_is_what_a_full_run_would_have_produced(self, tmp_path: Path) -> None:
        """The equivalence the whole design rests on: N == (N-1) + delta."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        _run(rec, folder)
        _wav(folder / "c.wav")
        verbs.delta_run(rec, data=str(folder), confirm=True)
        incremental = _rows(out)

        whole = tmp_path / "out" / "whole.jsonl"
        assert _run(_recipe(folder, whole), folder)["status"] == "completed"

        by_file = {os.path.basename(r["audio_filepath"]): r for r in incremental}
        assert set(by_file) == {os.path.basename(r["audio_filepath"]) for r in _rows(whole)}
        for row in _rows(whole):
            mine = by_file[os.path.basename(row["audio_filepath"])]
            assert mine == row, "a delta row differs from the row a full run produced"

    def test_the_next_run_reuses_the_merged_result_instead_of_recomputing(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        _run(rec, folder)
        _wav(folder / "c.wav")
        verbs.delta_run(rec, data=str(folder), confirm=True)

        scan = verbs.reuse_scan(rec, data=str(folder))
        assert scan["decision"] == "already_done", scan["rationale"]

    def test_a_second_delta_resumes_from_what_the_first_one_merged(self, tmp_path: Path) -> None:
        """A corpus grows more than once, so a delta has to be able to follow a delta.

        The merged manifest is published under the full pipeline's own key, which means the next
        delta reads its record like any other -- and refuses if that record describes an
        execution that never happened. Pairing the merged row count with either run's input
        makes a row-preserving writer look like it took 2 rows and produced 3, which
        ``contradictions`` correctly treats as the stage disproving its contract. Without this,
        the first delta is the last one a pipeline can ever run.
        """
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"

        _wav(folder / "c.wav")
        assert verbs.delta_run(rec, data=str(folder), confirm=True)["status"] == "completed"

        _wav(folder / "d.wav")
        offered = verbs.reuse_scan(rec, data=str(folder))["delta"]
        assert offered["status"] == "ready", offered["reason"]

        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        assert done["ran_files"] == [str(folder / "d.wav")]
        assert done["merged"][0]["rows_kept"] == 3
        assert done["merged"][0]["rows_added"] == 1
        assert {os.path.basename(r["audio_filepath"]) for r in _rows(out)} == {"a.wav", "b.wav", "c.wav", "d.wav"}

    def test_a_removed_file_loses_its_rows_without_running_anything(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        _run(rec, folder)

        (folder / "b.wav").unlink()
        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        assert done["ran_files"] == []
        assert [os.path.basename(r["audio_filepath"]) for r in _rows(out)] == ["a.wav"]

    def test_an_edited_file_replaces_its_row_rather_than_duplicating_it(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav", seconds=0.25)
        _wav(folder / "b.wav", seconds=0.25)
        rec = _recipe(folder, out)
        _run(rec, folder)
        assert {round(r["duration"], 2) for r in _rows(out)} == {0.25}

        _wav(folder / "b.wav", seconds=0.5)  # same name, different audio
        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        rows = {os.path.basename(r["audio_filepath"]): round(r["duration"], 2) for r in _rows(out)}
        assert rows == {"a.wav": 0.25, "b.wav": 0.5}

    def test_a_delta_needs_confirmation_like_any_other_run(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        _run(rec, folder)
        _wav(folder / "b.wav")

        refused = verbs.delta_run(rec, data=str(folder), confirm="not-the-hash")
        assert refused["status"] == "refused"
        assert "confirm" in refused["reason"]
        assert len(_rows(out)) == 1, "a refused delta must not touch the prior manifest"

    def test_an_unchanged_corpus_is_told_there_is_no_delta(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        _run(rec, folder)

        card = verbs.delta_run(rec, data=str(folder))
        assert card["status"] == "no_delta"
        # Nothing changed, so the ordinary probe already serves it; that is not a delta's job.
        assert card["delta"]["reason"]

    def test_a_narrowed_run_records_only_the_files_it_read(self, tmp_path: Path) -> None:
        """Coverage is a claim about work done; a subset run must not claim the whole corpus."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = Recipe.from_dict(
            {
                "stages": [
                    {
                        "ref": "CreateInitialManifestAudioFolderStage",
                        "params": {"data_dir": str(folder), "include_files": [str(folder / "a.wav")]},
                    },
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}},
                ]
            }
        ).freeze()
        assert _run(rec, folder)["status"] == "completed"
        assert [os.path.basename(r["audio_filepath"]) for r in _rows(out)] == ["a.wav"]

        published = [a for a in artifacts.list_artifacts() if a.uri == str(out)]
        assert published
        assert list(artifacts.load_coverage(published[0].step_key) or {}) == ["a.wav"]
        assert published[0].covers_files == 1


@pytest.mark.usefixtures("store")
class TestTheCardOffersIt:
    """A delta nobody is told about saves nothing: the miss card is where it has to appear."""

    def test_the_reuse_card_offers_the_delta_on_the_miss_it_belongs_to(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        _run(rec, folder)
        _wav(folder / "c.wav")

        scan = verbs.reuse_scan(rec, data=str(folder))

        # The key really did miss -- the delta is an addition to that answer, not a change to it.
        assert scan["decision"] == "fresh"
        assert scan["delta"]["status"] == "ready"
        assert scan["recommended"] == "delta"
        assert scan["prompt_user"] is True, "an available delta is always worth asking about"
        assert [c["id"] for c in scan["choices"]] == ["delta", "fresh"]
        assert "1 file(s) were added" in scan["rationale"]

    def test_a_first_ever_run_is_not_told_about_deltas(self, tmp_path: Path) -> None:
        """Nothing to compare against is the ordinary case, and it must stay quiet."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")

        scan = verbs.reuse_scan(_recipe(folder, out), data=str(folder))

        assert scan["decision"] == "fresh"
        assert "delta" not in scan

    def test_an_unusable_delta_says_why_on_the_card(self, tmp_path: Path) -> None:
        """The reason the full run is unavoidable is the useful part of a refusal."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        _run(rec, folder)

        # A fresh drop of data sharing no file with the last one: a different dataset, not a
        # changed one, and subtracting the two would delete every prior row.
        (folder / "a.wav").unlink()
        for name in ("x.wav", "y.wav"):
            _wav(folder / name)

        scan = verbs.reuse_scan(rec, data=str(folder))

        assert scan["delta"]["status"] != "ready"
        assert scan["delta"]["change"]["kind"] == "unrelated"
        assert "different dataset rather than a changed one" in scan["rationale"]
        assert scan["recommended"] == "fresh"

    def test_a_pipeline_that_saved_nothing_is_told_so_instead_of_never_ran(self, tmp_path: Path) -> None:
        """Coverage lives on artifacts, so a pipeline that persists nothing has no delta to offer.

        What it must not get is the plain miss wording: this pipeline ran yesterday, and "no
        prior artifact matches" reads as "this is new" to anyone who did not write the scanner.
        """
        folder = tmp_path / "audio"
        _wav(folder / "a.wav")
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                ]
            }
        ).freeze()
        _run(rec, folder)
        _wav(folder / "b.wav")

        scan = verbs.reuse_scan(rec, data=str(folder))

        assert scan["decision"] == "fresh"
        assert scan["prior_on_other_data"]["saved"] is False
        assert "persisted nothing" in scan["rationale"]
        assert "add-checkpoint" in scan["rationale"]


@pytest.mark.usefixtures("store")
class TestMergeSafety:
    """The merge refuses rather than producing a manifest no run could have produced."""

    def test_rows_with_different_columns_are_not_merged(self, tmp_path: Path) -> None:
        prior, produced = tmp_path / "prior.jsonl", tmp_path / "new.jsonl"
        prior.write_text(json.dumps({"audio_filepath": "/x/a.wav", "duration": 1.0}) + "\n")
        produced.write_text(json.dumps({"audio_filepath": "/x/b.wav", "duration": 1.0, "extra": 1}) + "\n")
        sink = delta.Sink(index=0, param="output_path", uri=str(prior), step_key="k", key="audio_filepath")

        kept, added, why = delta.merge(sink, produced=str(produced), stale=set(), key="audio_filepath", root="/x")

        assert (kept, added) == (0, 0)
        assert "same columns" in why
        assert len(_rows(prior)) == 1, "the prior manifest must survive a refused merge intact"

    def test_a_manifest_that_grew_since_publication_stops_being_reusable(self, tmp_path: Path) -> None:
        """What protects a delta that dies between two merges.

        The merged manifest is right and its artifact record is not yet, so the record must not
        be honoured. Reuse is bound to the bytes, which is why the half-finished case degrades to
        recomputing rather than to serving a manifest whose record understates it.
        """
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        published = [a for a in artifacts.list_artifacts() if a.uri == str(out)]
        assert published
        assert not artifacts.invalid_reasons(published[0])

        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"audio_filepath": str(folder / "b.wav"), "duration": 1.0}) + "\n")

        assert any("changed after" in r for r in artifacts.invalid_reasons(published[0]))

    def test_the_swap_is_atomic(self, tmp_path: Path) -> None:
        """A reader sees the old manifest or the new one, never a half-written file."""
        prior, produced = tmp_path / "prior.jsonl", tmp_path / "new.jsonl"
        prior.write_text(json.dumps({"audio_filepath": "/x/a.wav"}) + "\n")
        produced.write_text(json.dumps({"audio_filepath": "/x/b.wav"}) + "\n")
        sink = delta.Sink(index=0, param="output_path", uri=str(prior), step_key="k", key="audio_filepath")
        before = os.stat(prior).st_ino

        kept, added, why = delta.merge(sink, produced=str(produced), stale=set(), key="audio_filepath", root="/x")

        assert (kept, added, why) == (1, 1, "")
        assert os.stat(prior).st_ino != before  # replaced wholesale, not appended in place
        assert [r["audio_filepath"] for r in _rows(prior)] == ["/x/a.wav", "/x/b.wav"]
