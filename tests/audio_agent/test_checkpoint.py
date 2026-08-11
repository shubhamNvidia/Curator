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

"""Placing a mid-pipeline manifest so the expensive stages become reusable.

The recipes here are miniatures of the ALM pipeline, which is where the problem was found:
GPU work that persists nothing, and a waveform still resident for several stages after it.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.audio_agent import checkpoint, reuse, verbs
from nemo_curator.audio_agent.recipe import Recipe

_READER = {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}}
_DUR = {"ref": "GetAudioDurationStage", "params": {}}
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}
# Expensive per its card, and persists nothing of its own -- the work worth not repeating.
_ASR = {"ref": "InferenceAsrNemoStage", "params": {"model_name": "nvidia/parakeet-tdt-0.6b-v2"}}
# Holds its waveform in the task, so everything downstream of it carries a tensor no manifest
# can serialize, until a sanitizing stage drops it.
_KEEPS_WAVEFORM = {
    "ref": "ResampleAudioStage",
    "params": {"resampled_audio_dir": "/tmp/rs", "keep_waveform_in_task": True, "write_to_disk": False},
}
_SANITIZER = {"ref": "AudioToDocumentStage", "params": {}}


def _recipe(*stages: dict[str, Any]) -> Recipe:
    return Recipe.from_dict({"stages": list(stages)}).freeze()


class TestWhereItGoes:
    def test_it_lands_just_past_the_expensive_stage(self) -> None:
        """Not as deep as it will go: the shallowest position clearing the GPU work leaves the
        most of the tail editable, and a tweak below the checkpoint is what reuses it."""
        spot, why = checkpoint.advise(_recipe(_READER, _ASR, _DUR, _WRITER))
        assert spot is not None, why
        assert (spot.index, spot.after_stage) == (2, "InferenceAsrNemoStage")
        assert spot.skips == ["InferenceAsrNemoStage"]

    def test_a_resident_waveform_pushes_it_to_the_stage_that_drops_one(self) -> None:
        """The ALM finding in miniature. Writing right after the ASR stage would crash on the
        tensor, which no list of stage names would have predicted -- the validator is asked."""
        spot, why = checkpoint.advise(_recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER))
        assert spot is not None, why
        assert spot.after_stage == "AudioToDocumentStage"
        assert "carrying audio in memory" in spot.not_earlier
        assert "InferenceAsrNemoStage" in spot.not_earlier

    def test_a_pipeline_that_never_drops_the_waveform_gets_no_position(self) -> None:
        """Refusing beats advising a writer that would fail at ``json.dumps``."""
        spot, why = checkpoint.advise(_recipe(_READER, _KEEPS_WAVEFORM, _ASR, _DUR, _WRITER))
        assert spot is None
        assert "nowhere after InferenceAsrNemoStage" in why

    def test_it_will_not_split_a_pair_that_passes_state_outside_the_row(self) -> None:
        """``OverlapFilterStage`` parks counters in ``task._metadata`` and the aggregator reads
        them. A manifest carries only ``task.data``, and nothing raises when the counters go
        missing -- the aggregator gets an empty dict and reports wrong numbers successfully."""
        overlap = {"ref": "OverlapFilterStage", "params": {}}
        metrics = {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": "/tmp/metrics.json"}}
        recipe = _recipe(_READER, _ASR, overlap, metrics, _WRITER)
        spot, why = checkpoint.at(recipe, index=3)
        assert spot is None
        assert "pretrain_long_form" in why
        assert "PretrainMetricsAggregatorStage" in why
        # Above the pair is fine: the counters are made and read entirely below the checkpoint.
        allowed, why_not = checkpoint.at(recipe, index=2)
        assert allowed is not None, why_not

    def test_cheap_work_is_not_worth_a_checkpoint(self) -> None:
        spot, why = checkpoint.advise(_recipe(_READER, _DUR, _WRITER))
        assert spot is None
        assert "expensive" in why

    def test_a_writer_already_past_the_expensive_work_is_the_checkpoint(self) -> None:
        """Advising a second writer beside one the recipe already has is noise, and would
        repeat itself every scan."""
        mid = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/mid.jsonl"}}
        spot, why = checkpoint.advise(_recipe(_READER, _ASR, mid, _DUR, _WRITER))
        assert spot is None
        assert "already writes a manifest" in why

    def test_an_output_nothing_can_re_read_is_not_a_checkpoint(self, tmp_path: Any) -> None:
        """Sortformer fills an RTTM directory and no source stage can start a pipeline from
        one, so counting it would answer "you are covered" to a user whose diarization is
        recomputed on every request."""
        rttm_dir = tmp_path / "rttm"
        rttm_dir.mkdir()
        (rttm_dir / "clip.rttm").write_text("SPEAKER clip 1 0.0 1.0 <NA> <NA> spk0 <NA> <NA>\n")
        diarize = {"ref": "InferenceSortformerStage", "params": {"rttm_out_dir": str(rttm_dir)}}
        spot, why = checkpoint.advise(_recipe(_READER, diarize, _DUR, _WRITER))
        assert spot is not None, why
        assert spot.after_stage == "InferenceSortformerStage"

    def test_the_pipelines_own_final_writer_does_not_count_as_one(self) -> None:
        """Resuming from the final sink only serves a request that was already finished. The
        case this exists for is a changed tail, where that artifact no longer matches and the
        GPU stages are recomputed to rebuild it."""
        spot, _why = checkpoint.advise(_recipe(_READER, _ASR, _DUR, _WRITER))
        assert spot is not None
        assert spot.index < len(_recipe(_READER, _ASR, _DUR, _WRITER).stages)


class TestTheRecipeItHandsBack:
    def test_the_writer_appears_at_the_advised_position_and_nothing_else_moves(self) -> None:
        original = _recipe(_READER, _ASR, _DUR, _WRITER)
        spot, _ = checkpoint.advise(original)
        assert spot is not None
        out, err = checkpoint.insert(original, index=spot.index, output_path="/tmp/ck.jsonl")
        assert err == ""
        assert out is not None
        assert [s.ref for s in out.stages] == [
            "ManifestReader",
            "InferenceAsrNemoStage",
            "ManifestWriterStage",
            "GetAudioDurationStage",
            "ManifestWriterStage",
        ]
        assert out.stages[spot.index].params == {"output_path": "/tmp/ck.jsonl"}
        assert [s.params for s in out.stages if s.ref != "ManifestWriterStage"] == [
            s.params for s in original.stages if s.ref != "ManifestWriterStage"
        ]

    def test_the_checkpointed_recipe_is_not_advised_again(self) -> None:
        original = _recipe(_READER, _ASR, _DUR, _WRITER)
        spot, _ = checkpoint.advise(original)
        assert spot is not None
        out, _ = checkpoint.insert(original, index=spot.index, output_path="/tmp/ck.jsonl")
        assert out is not None
        again, why = checkpoint.advise(out)
        assert again is None
        assert "already writes a manifest" in why

    def test_a_checkpoint_needs_somewhere_to_write(self) -> None:
        out, err = checkpoint.insert(_recipe(_READER, _ASR, _WRITER), index=2, output_path="")
        assert out is None
        assert "output path" in err

    def test_a_position_outside_the_recipe_is_refused(self) -> None:
        out, err = checkpoint.insert(_recipe(_READER, _ASR, _WRITER), index=9, output_path="/tmp/ck.jsonl")
        assert out is None
        assert "outside the recipe" in err


class TestACallersOwnPosition:
    def test_a_position_that_would_crash_is_refused(self) -> None:
        recipe = _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER)
        spot, why = checkpoint.at(recipe, index=3)
        assert spot is None
        assert "carrying audio in memory" in why

    def test_a_position_above_the_expensive_work_is_allowed_and_says_it_saves_nothing(self) -> None:
        """A preference, not an error: a user tuning a stage may want the checkpoint above it."""
        spot, why = checkpoint.at(_recipe(_READER, _ASR, _DUR, _WRITER), index=1)
        assert spot is not None, why
        assert spot.skips == []
        assert "saves little" in spot.as_dict()["effect"]

    def test_an_edge_of_the_recipe_is_not_a_position(self) -> None:
        spot, why = checkpoint.at(_recipe(_READER, _ASR, _WRITER), index=0)
        assert spot is None
        assert "no stage on one side" in why


class TestTheOfferOnAMiss:
    """``reuse`` recommends the simulated position, not the end of the recomputed prefix."""

    def test_the_offer_names_a_position_a_manifest_can_actually_hold(self) -> None:
        recipe = _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER)
        unsaved = {"stages": ["ManifestReader", "ResampleAudioStage", "InferenceAsrNemoStage"]}
        offer = reuse._persist_offer(recipe, unsaved)
        assert offer is not None
        # The prefix ends at the ASR stage, and a writer there is exactly what crashes.
        assert offer["after_stage"] == "AudioToDocumentStage"
        assert "InferenceAsrNemoStage" in offer["effect"]

    def test_a_prefix_that_already_persisted_is_offered_nothing(self) -> None:
        offer = reuse._persist_offer(
            _recipe(_READER, _ASR, _WRITER),
            {"stages": ["ManifestWriterStage"], "resume_point_persists": True},
        )
        assert offer is None

    def test_a_pipeline_with_no_legal_position_says_so_rather_than_advising_one(self) -> None:
        recipe = _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _DUR, _WRITER)
        offer = reuse._persist_offer(recipe, {"stages": ["ManifestReader", "InferenceAsrNemoStage"]})
        assert offer is not None
        assert offer["action"] == "no_checkpoint"
        assert "InferenceAsrNemoStage" in offer["why"]


class TestTheVerb:
    def test_without_a_path_it_only_advises(self) -> None:
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _DUR, _WRITER))
        assert out["status"] == "advice"
        assert out["advice"]["after_stage"] == "InferenceAsrNemoStage"
        assert "recipe" not in out

    def test_with_a_path_it_returns_the_recipe_and_runs_nothing(self) -> None:
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _DUR, _WRITER), output_path="/tmp/ck.jsonl")
        assert out["status"] == "ok"
        assert [s["ref"] for s in out["recipe"]["stages"]].count("ManifestWriterStage") == 2
        assert "validate" in out["next"]

    def test_a_named_stage_is_still_checked(self) -> None:
        out = verbs.add_checkpoint(
            _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER),
            after="InferenceAsrNemoStage",
        )
        assert out["status"] == "no_checkpoint"
        assert "carrying audio in memory" in out["reason"]

    def test_a_stage_that_is_not_in_the_recipe_is_an_error(self) -> None:
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _WRITER), after="NoSuchStage")
        assert out["status"] == "error"
        assert "NoSuchStage" in out["reason"]

    def test_a_path_outside_a_locked_workspace_is_refused(self, monkeypatch: Any, tmp_path: Any) -> None:
        """The checkpoint is a file the agent told the user to write, so its path answers to the
        same lock as everything else the agent proposes writing."""
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _WRITER), output_path="/etc/ck.jsonl")
        assert out["status"] == "refused"
        assert "workspace" in out["reason"].lower()
