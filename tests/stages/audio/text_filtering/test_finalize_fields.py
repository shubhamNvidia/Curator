# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.stages.audio.text_filtering.finalize_fields import FinalizeFieldsStage
from nemo_curator.tasks import AudioTask


def test_happy_path() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(
        data={
            "text": "original text",
            "cleaned_text": "cleaned version",
            "pnc": "pnc",
            "itn": "noitn",
            "timestamp": "notimestamp",
            "audio_filepath": "/a.wav",
            "duration": 3.5,
        }
    )
    result = stage.process(task)
    assert result.data["v1_text"] == "original text"
    assert result.data["text"] == "cleaned version"
    assert "cleaned_text" not in result.data
    assert "pnc" not in result.data
    assert "itn" not in result.data
    assert "timestamp" not in result.data
    assert result.data["audio_filepath"] == "/a.wav"
    assert result.data["duration"] == 3.5


def test_missing_source_text_key_is_ignored() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(data={"cleaned_text": "cleaned"})
    result = stage.process(task)
    assert result.data["text"] == "cleaned"
    assert "v1_text" not in result.data


def test_missing_drop_keys_are_ignored() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(data={"text": "t", "cleaned_text": "c"})
    result = stage.process(task)  # no pnc/itn/timestamp — should not raise
    assert result.data["text"] == "c"


def test_custom_drop_keys() -> None:
    stage = FinalizeFieldsStage(drop_keys=["custom_field", "another"])
    task = AudioTask(data={"text": "t", "cleaned_text": "c", "custom_field": "drop_me", "another": "also_drop"})
    result = stage.process(task)
    assert "custom_field" not in result.data
    assert "another" not in result.data


def test_other_fields_preserved() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(
        data={
            "text": "t",
            "cleaned_text": "c",
            "pred_text": "raw",
            "skip_me": 0,
            "shard_id": 42,
        }
    )
    result = stage.process(task)
    assert result.data["pred_text"] == "raw"
    assert result.data["skip_me"] == 0
    assert result.data["shard_id"] == 42


def test_cleaned_text_removed_after_rename() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(data={"text": "original", "cleaned_text": "clean"})
    result = stage.process(task)
    assert "cleaned_text" not in result.data
    assert result.data["text"] == "clean"
