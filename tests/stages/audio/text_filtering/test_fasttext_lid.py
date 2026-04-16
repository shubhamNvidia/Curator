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

from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.audio.text_filtering.fasttext_lid import FastTextLIDStage
from nemo_curator.tasks import AudioTask


def _make_stage(label: str, prob: float) -> FastTextLIDStage:
    """Create a stage with a mocked FastTextLangId that returns the given label/prob."""
    stage = FastTextLIDStage(model_path="/fake/model.bin")
    mock_lid = MagicMock()
    mock_lid.score_document.return_value = str([prob, label])
    stage._lid = mock_lid
    return stage


def test_correct_lang_and_high_prob_passes() -> None:
    stage = _make_stage("EN", 0.95)
    task = AudioTask(data={"cleaned_text": "hello world", "skip_me": 0})
    result = stage.process(task)
    assert result.data["skip_me"] == 0


def test_wrong_lang_sets_skip_me() -> None:
    stage = _make_stage("FR", 0.95)
    task = AudioTask(data={"cleaned_text": "bonjour monde", "skip_me": 0})
    result = stage.process(task)
    assert result.data["skip_me"] == 1


def test_low_prob_sets_skip_me() -> None:
    stage = _make_stage("EN", 0.1)
    task = AudioTask(data={"cleaned_text": "hello", "skip_me": 0})
    result = stage.process(task)
    assert result.data["skip_me"] == 1


def test_correct_lang_exactly_at_threshold_passes() -> None:
    stage = FastTextLIDStage(model_path="/fake/model.bin", min_lang_prob=0.3)
    mock_lid = MagicMock()
    mock_lid.score_document.return_value = str([0.3, "EN"])
    stage._lid = mock_lid
    task = AudioTask(data={"cleaned_text": "hello", "skip_me": 0})
    result = stage.process(task)
    assert result.data["skip_me"] == 0


def test_empty_text_sets_skip_me_without_calling_model() -> None:
    stage = FastTextLIDStage(model_path="/fake/model.bin")
    mock_lid = MagicMock()
    stage._lid = mock_lid
    task = AudioTask(data={"cleaned_text": "   ", "skip_me": 0})
    result = stage.process(task)
    assert result.data["skip_me"] == 1
    mock_lid.score_document.assert_not_called()


def test_preserves_existing_skip_me_one() -> None:
    stage = _make_stage("EN", 0.95)
    task = AudioTask(data={"cleaned_text": "hello", "skip_me": 1})
    result = stage.process(task)
    assert result.data["skip_me"] == 1


def test_invalid_model_path_raises() -> None:
    stage = FastTextLIDStage(model_path="/does/not/exist.bin")
    with pytest.raises(ValueError, match="not a valid file path"):
        stage._resolve_model_path()


def test_non_string_text_returns_task_unchanged() -> None:
    stage = _make_stage("EN", 0.95)
    task = AudioTask(data={"cleaned_text": None, "skip_me": 0})
    result = stage.process(task)
    assert result.data["skip_me"] == 0


def test_requires_model_path() -> None:
    with pytest.raises(ValueError, match="model_path is required"):
        FastTextLIDStage(model_path="")


def test_known_model_name_checks_cache(tmp_path: object) -> None:
    stage = FastTextLIDStage(model_path="lid.176.ftz")
    # Patch urlretrieve to avoid real download; patch cache dir to tmp_path
    with (
        patch("nemo_curator.stages.audio.text_filtering.fasttext_lid._DEFAULT_CACHE_DIR", str(tmp_path)),
        patch("urllib.request.urlretrieve") as mock_dl,
    ):
        mock_dl.side_effect = lambda url, path: open(path, "w").close()  # noqa: SIM115
        resolved = stage._resolve_model_path()
    assert resolved.endswith("lid.176.ftz")
