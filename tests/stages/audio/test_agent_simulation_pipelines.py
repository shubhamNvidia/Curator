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

"""Agent-simulation tests for audio stage contracts and pipeline composition.

These tests intentionally exercise the layer an agent would rely on: the
``describe()`` contract, custom keys, residency flags, process/process_batch
dispatch, and cross-stage data shapes.  Model-heavy stages are represented by
small fakes at the model boundary so CI never needs GPUs, HF tokens, internet,
or model downloads.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import sys
import types
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

sf = pytest.importorskip("soundfile")
torch = pytest.importorskip("torch")


def _install_agent_simulation_stubs() -> None:  # noqa: C901, PLR0915 (complexity accepted: one linear stub block per heavy dependency)
    """Install lightweight stand-ins for heavy model/audio dependencies."""

    class _IdentityResample:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def to(self, _device: Any) -> _IdentityResample:
            return self

        def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
            return waveform

    torchaudio = types.ModuleType("torchaudio")

    def _load_audio(path: str) -> tuple[torch.Tensor, int]:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), int(sr)

    def _save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
        array = waveform.detach().cpu().numpy()
        sf.write(path, array.T if array.ndim == 2 else array, sample_rate)

    torchaudio.load = _load_audio
    torchaudio.save = _save_audio
    torchaudio.transforms = SimpleNamespace(Resample=_IdentityResample)
    torchaudio_functional = types.ModuleType("torchaudio.functional")
    torchaudio_functional.resample = lambda waveform, _src, _dst: waveform
    torchaudio_pipelines = types.ModuleType("torchaudio.pipelines")
    torchaudio_pipelines.SQUIM_OBJECTIVE = SimpleNamespace(get_model=lambda: lambda batch: (batch, batch, batch))
    # Same never-shadow-real rule (see librosa below): a stub torchaudio is not
    # a package, so any REAL dependent (pyannote.audio, whisperx) importing an
    # unstubbed torchaudio submodule at runtime would break.
    if importlib.util.find_spec("torchaudio") is None:
        sys.modules["torchaudio"] = torchaudio
        sys.modules["torchaudio.functional"] = torchaudio_functional
        sys.modules["torchaudio.pipelines"] = torchaudio_pipelines

    # Never shadow a REAL librosa: it lazy-imports its own submodules
    # (librosa.core) through sys.modules, so replacing the entry breaks every
    # other test module in the session that already bound the real package.
    if importlib.util.find_spec("librosa") is None:
        librosa = types.ModuleType("librosa")
        librosa.load = lambda path, sr=None: (
            sf.read(path, dtype="float32")[0],
            sf.info(path).samplerate if sr is None else sr,
        )
        librosa.stft = lambda y, n_fft, hop_length, window: np.zeros((n_fft // 2 + 1, 1), dtype=np.complex64)
        librosa.power_to_db = lambda power, ref, top_db: np.asarray(power, dtype=np.float32)
        sys.modules["librosa"] = librosa

    class _FakeAudioSegment:
        sample_width = 2
        channels = 1
        frame_rate = 16000

        def get_array_of_samples(self) -> list[int]:
            return [0, 1000, -1000, 0] * 100

    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = _FakeAudioSegment
    sys.modules["pydub"] = pydub

    silero_vad = types.ModuleType("silero_vad")
    silero_vad.load_silero_vad = lambda: object()
    silero_vad.get_speech_timestamps = lambda *_args, **_kwargs: []
    sys.modules["silero_vad"] = silero_vad

    # Same rule as librosa: never shadow a REAL whisperx — the tests attach
    # stage-level fakes (stage._vad_model), so the stubs are import-satisfiers
    # for dep-less CI only; shadowing breaks the gpu-marked whisperx tests.
    if importlib.util.find_spec("whisperx") is None:
        whisperx = types.ModuleType("whisperx")
        whisperx_audio = types.ModuleType("whisperx.audio")
        whisperx_audio.SAMPLE_RATE = 16000
        whisperx_vads = types.ModuleType("whisperx.vads")
        whisperx_pyannote = types.ModuleType("whisperx.vads.pyannote")
        whisperx_pyannote.Pyannote = SimpleNamespace(merge_chunks=lambda segments, *_args, **_kwargs: segments)
        whisperx_pyannote.load_vad_model = lambda *_args, **_kwargs: lambda _payload: []
        sys.modules["whisperx"] = whisperx
        sys.modules["whisperx.audio"] = whisperx_audio
        sys.modules["whisperx.vads"] = whisperx_vads
        sys.modules["whisperx.vads.pyannote"] = whisperx_pyannote

    class _FakePyAnnotePipeline:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> _FakePyAnnotePipeline:
            return cls()

        def to(self, _device: Any) -> None:
            return None

    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = _FakePyAnnotePipeline
    pyannote_hook = types.ModuleType("pyannote.audio.pipelines.utils.hook")

    class _ProgressHook:
        def __enter__(self) -> _ProgressHook:  # noqa: PYI034 - minimal test stub; `Self` typing is unnecessary here
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    pyannote_hook.ProgressHook = _ProgressHook
    pyannote_core = types.ModuleType("pyannote.core")

    class _Segment:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    pyannote_core.Segment = _Segment
    # Same rule: a non-package stand-in for a REAL pyannote.audio breaks its
    # lazy submodule imports (pyannote.audio.models) in the gpu-marked tests;
    # the tests here fake at the stage level (pyannote._pipeline) instead.
    if importlib.util.find_spec("pyannote") is None:
        sys.modules["pyannote"] = types.ModuleType("pyannote")
        sys.modules["pyannote.audio"] = pyannote_audio
        sys.modules["pyannote.audio.pipelines"] = types.ModuleType("pyannote.audio.pipelines")
        sys.modules["pyannote.audio.pipelines.utils"] = types.ModuleType("pyannote.audio.pipelines.utils")
        sys.modules["pyannote.audio.pipelines.utils.hook"] = pyannote_hook
        sys.modules["pyannote.core"] = pyannote_core

    class _FakeASRModel:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> _FakeASRModel:
            return cls()

        @classmethod
        def restore_from(cls, *_args: Any, **_kwargs: Any) -> _FakeASRModel:
            return cls()

        def transcribe(self, files: list[str], **_kwargs: Any) -> list[Any]:
            return [SimpleNamespace(text="hello", timestamp={"word": []}, word_confidence=None) for _ in files]

    class _FakeSortformerEncLabelModel(_FakeASRModel):
        pass

    nemo = types.ModuleType("nemo")
    nemo_collections = types.ModuleType("nemo.collections")
    nemo_asr = types.ModuleType("nemo.collections.asr")
    nemo_asr_models = types.ModuleType("nemo.collections.asr.models")
    nemo_asr_models.ASRModel = _FakeASRModel
    nemo_asr_models.SortformerEncLabelModel = _FakeSortformerEncLabelModel
    nemo_asr.models = nemo_asr_models
    nemo_collections.asr = nemo_asr
    nemo.collections = nemo_collections
    sys.modules["nemo"] = nemo
    sys.modules["nemo.collections"] = nemo_collections
    sys.modules["nemo.collections.asr"] = nemo_asr
    sys.modules["nemo.collections.asr.models"] = nemo_asr_models

    metrics = types.ModuleType("nemo.collections.asr.metrics")
    wer_mod = types.ModuleType("nemo.collections.asr.metrics.wer")

    def _word_error_rate_detail(hypotheses: list[str], references: list[str], use_cer: bool = False) -> tuple:  # noqa: ARG001 - stub mirrors nemo's word_error_rate_detail signature
        return (0.0 if hypotheses == references else 1.0, 1, 0.0, 0.0, 0.0)

    wer_mod.word_error_rate_detail = _word_error_rate_detail
    sys.modules["nemo.collections.asr.metrics"] = metrics
    sys.modules["nemo.collections.asr.metrics.wer"] = wer_mod

    ctc_mod = types.ModuleType("nemo.collections.asr.parts.submodules.ctc_decoding")
    rnnt_mod = types.ModuleType("nemo.collections.asr.parts.submodules.rnnt_decoding")

    class _DecodingConfig:
        def __init__(self) -> None:
            self.confidence_cfg = SimpleNamespace(preserve_word_confidence=True)
            self.greedy = SimpleNamespace(compute_timestamps=True)

    ctc_mod.CTCDecodingConfig = _DecodingConfig
    rnnt_mod.RNNTDecodingConfig = _DecodingConfig
    sys.modules["nemo.collections.asr.parts"] = types.ModuleType("nemo.collections.asr.parts")
    sys.modules["nemo.collections.asr.parts.submodules"] = types.ModuleType("nemo.collections.asr.parts.submodules")
    sys.modules["nemo.collections.asr.parts.submodules.ctc_decoding"] = ctc_mod
    sys.modules["nemo.collections.asr.parts.submodules.rnnt_decoding"] = rnnt_mod

    # Same rule as librosa: an empty stand-in for a REAL nemo_text_processing
    # breaks tagging/text/test_itn.py's direct submodule import in-session.
    if importlib.util.find_spec("nemo_text_processing") is None:
        nmtp = types.ModuleType("nemo_text_processing")
        tn = types.ModuleType("nemo_text_processing.text_normalization")
        inv = types.ModuleType("nemo_text_processing.inverse_text_normalization.inverse_normalize")
        tn.Normalizer = lambda *_args, **_kwargs: _FakeNormalizer()
        inv.InverseNormalizer = lambda *_args, **_kwargs: _FakeNormalizer()
        sys.modules["nemo_text_processing"] = nmtp
        sys.modules["nemo_text_processing.text_normalization"] = tn
        sys.modules["nemo_text_processing.inverse_text_normalization"] = types.ModuleType(
            "nemo_text_processing.inverse_text_normalization"
        )
        sys.modules["nemo_text_processing.inverse_text_normalization.inverse_normalize"] = inv

    opencc = types.ModuleType("opencc")
    opencc.OpenCC = lambda *_args, **_kwargs: _FakeConverter()
    sys.modules["opencc"] = opencc

    # setdefault is not enough: in a fresh interpreter the stub occupies the
    # slot BEFORE real dependents (pyannote.audio -> torchmetrics) import the
    # real thing. Only stub when the real package is absent.
    if importlib.util.find_spec("transformers") is None:
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())
        sys.modules.setdefault("transformers", transformers)

    if importlib.util.find_spec("huggingface_hub") is None:
        hf_hub = types.ModuleType("huggingface_hub")
        hf_hub.hf_hub_download = lambda *_args, **_kwargs: ""
        hf_hub.snapshot_download = lambda *_args, **_kwargs: os.getcwd()
        sys.modules.setdefault("huggingface_hub", hf_hub)

    sigmos_mod = types.ModuleType("nemo_curator.stages.audio.filtering.sigmos_filter_module.third_party.sigmos.sigmos")
    sigmos_mod.build_sigmos_model = lambda *_args, **_kwargs: _FakeSIGMOSModel()
    sys.modules["nemo_curator.stages.audio.filtering.sigmos_filter_module.third_party.sigmos.sigmos"] = sigmos_mod

    band_predict_mod = types.ModuleType("nemo_curator.stages.audio.filtering.band_filter_module.predict")
    band_predict_mod.BandPredictor = lambda *_args, **_kwargs: _FakeBandPredictor()
    sys.modules["nemo_curator.stages.audio.filtering.band_filter_module.predict"] = band_predict_mod


class _FakeNormalizer:
    def split_text_into_sentences(self, text: str) -> list[str]:
        return [text]

    def normalize_list(self, sentences: list[str]) -> list[str]:
        return sentences

    def normalize(self, text: str, **_: Any) -> str:
        return text


class _FakeConverter:
    def convert(self, text: str) -> str:
        return text


_install_agent_simulation_stubs()

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.stages.audio._agent_ready import AgentReady, StageContract
from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter.audio_data_filter import AudioDataFilterStage
from nemo_curator.stages.audio.alm.alm_data_builder import ALMDataBuilderStage
from nemo_curator.stages.audio.alm.alm_data_overlap import ALMDataOverlapStage
from nemo_curator.stages.audio.alm.pretrain.extraction import SnippetExtractionStage
from nemo_curator.stages.audio.alm.pretrain.io import (
    PretrainMetricsAggregatorStage,
    ReadLongFormManifestStage,
    SnippetManifestWriterStage,
)
from nemo_curator.stages.audio.alm.pretrain.planning import (
    OverlapFilterStage,
    SnippetCutPlannerStage,
    SnippetRepetitionFilterStage,
)
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueStage,
)
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.datasets.readspeech.create_initial_manifest import (
    CreateInitialManifestReadSpeechStage,
)
from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.stages.audio.inference.asr.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage
from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.io.extract_segments import SegmentExtractionStage
from nemo_curator.stages.audio.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.audio.metrics.wer import ComputeWERStage, GetPairwiseWerStage
from nemo_curator.stages.audio.postprocessing.timestamp_mapper import TimestampMapperStage
from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
from nemo_curator.stages.audio.segmentation.speaker_separation import SpeakerSeparationStage
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import (
    BaseASRProcessorStage,
    NeMoASRAlignerStage,
)
from nemo_curator.stages.audio.tagging.merge_alignment_diarization import MergeAlignmentDiarizationStage
from nemo_curator.stages.audio.tagging.prepare_module_segments import PrepareModuleSegmentsStage
from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.stages.audio.tagging.split import (
    JoinSplitAudioMetadataStage,
    SplitASRAlignJoinStage,
    SplitLongAudioStage,
)
from nemo_curator.stages.audio.tagging.text.chinese_conversion import ChineseConversionStage
from nemo_curator.stages.audio.tagging.text.itn import InverseTextNormalizationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, DocumentBatch, EmptyTask, FileGroupTask


@dataclass(frozen=True)
class StageCase:
    cls: type
    factory: Any
    scenario: str


EXPECTED_AUDIO_AGENT_READY_STAGE_COUNT = 44
VALID_AGENT_SCENARIOS = {
    "alm",
    "alm_pretrain",
    "composite",
    "dataset_contract",
    "ingress_transform",
    "segmentation_quality",
    "speech_tagging",
    "split_join_extract",
}
VALID_ACCEPTS = {"file", "waveform"}
VALID_PRODUCES = {"tensor", "disk"}


def _waveform(sample_rate: int = 16000, duration_sec: float = 1.0, channels: int = 1) -> torch.Tensor:
    samples = max(1, int(sample_rate * duration_sec))
    base = torch.linspace(-0.25, 0.25, samples)
    if channels == 1:
        return base.unsqueeze(0)
    return torch.stack([base, -base])


def _write_wav(path: Path, sample_rate: int = 16000, duration_sec: float = 1.0, channels: int = 1) -> Path:
    waveform = _waveform(sample_rate, duration_sec, channels).numpy()
    sf.write(path, waveform.T if channels > 1 else waveform[0], sample_rate)
    return path


def _audio_task(path: Path, *, key: str = "audio_filepath") -> AudioTask:
    return AudioTask(
        dataset_name="agent",
        data={
            key: str(path),
            "text": "hello world",
            "duration": 1.0,
            "audio_sample_rate": 16000,
            "audio_item_id": "agent_item",
        },
        _metadata={"source": "agent"},
        _stage_perf=["upstream"],
    )


def _base_segments() -> list[dict[str, Any]]:
    return [
        {
            "start": 0.0,
            "end": 0.45,
            "speaker": "spk0",
            "text": "hello",
            "text_ref": "hello",
            "words": [{"word": "hello", "start": 0.0, "end": 0.45}],
            "metrics": {"bandwidth": 9000},
        },
        {
            "start": 0.45,
            "end": 0.9,
            "speaker": "spk1",
            "text": "world",
            "text_ref": "world",
            "words": [{"word": "world", "start": 0.45, "end": 0.9}],
            "metrics": {"bandwidth": 9000},
        },
    ]


def _copy_file_for_fake_ffmpeg(cmd: list[str], **_: Any) -> SimpleNamespace:
    src = cmd[cmd.index("-i") + 1]
    dst = cmd[-1]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return SimpleNamespace(returncode=0)


class _FakeUTMOSModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def parameters(self) -> Iterator[torch.Tensor]:
        yield torch.tensor([0.0])

    def __call__(self, _waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        return torch.tensor([self.score])


class _SequentialUTMOSModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)

    def parameters(self) -> Iterator[torch.Tensor]:
        yield torch.tensor([0.0])

    def __call__(self, _waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        return torch.tensor([self.scores.pop(0)])


class _FakeSIGMOSModel:
    def __init__(self, score: float = 2.0) -> None:
        self.score = score

    def run(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        return {
            "MOS_NOISE": self.score,
            "MOS_OVRL": self.score,
            "MOS_SIG": self.score,
            "MOS_COL": self.score,
            "MOS_DISC": self.score,
            "MOS_LOUD": self.score,
            "MOS_REVERB": self.score,
        }


class _FakeBandPredictor:
    def __init__(self, prediction: str = "narrow_band") -> None:
        self.prediction = prediction

    def predict_audio(self, waveform: torch.Tensor, sample_rate: int) -> str:
        return self.prediction


class _FakeVADModel:
    def get_vad_segments(self, audio: np.ndarray, merge_max_length: float, sample_rate: int = 16000) -> list[dict]:
        return [{"start": 0.0, "end": min(merge_max_length, 0.4)}]

    def to(self, device: str) -> None:
        return None


class _FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakePyAnnoteDiarization:
    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns
        self._tracks = turns

    def get_overlap(self) -> SimpleNamespace:
        return SimpleNamespace(segments_list_=[])

    def crop(self, _segment: Any) -> _FakePyAnnoteDiarization:
        return self

    def itertracks(self, yield_label: bool = True) -> Iterator[tuple[_FakeTurn, None, str]]:
        for start, end, speaker in self._turns:
            yield _FakeTurn(start, end), None, speaker


class _TinyAudioSegment:
    sample_width = 2
    channels = 1
    frame_rate = 16000

    def get_array_of_samples(self) -> list[int]:
        return [0, 500, -500, 0] * 100


def _pipeline_from_agent_yaml(spec: dict[str, Any]) -> Any:
    """Create a Pipeline from agent-authored YAML-like config.

    The public config runner expects ``stages`` while the ReadSpeech tutorial
    names the same list ``processors``.  The harness accepts either so the tests
    mirror the shape an agent sees in tutorial YAML.
    """
    normalized = dict(spec)
    if "processors" in normalized and "stages" not in normalized:
        normalized["stages"] = normalized.pop("processors")
    return create_pipeline_from_yaml(OmegaConf.create(normalized), log_config=False)


def _flatten_stage_output(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return [item for item in result if item is not None]
    return [result]


def _run_inline_agent_pipeline(stages: list[Any], tasks: list[Any]) -> list[Any]:
    """Run stages in-process while preserving fan-out and batch-only stages."""
    current: list[Any] = list(tasks)
    for stage in stages:
        if not current:
            return []
        if isinstance(stage, AudioToDocumentStage):
            audio_tasks = [task for task in current if isinstance(task, AudioTask)]
            current = stage.process_batch(audio_tasks)
            continue
        if isinstance(stage, TorchSquimQualityMetricsStage):
            audio_tasks = [task for task in current if isinstance(task, AudioTask)]
            current = stage.process_batch(audio_tasks)
            continue

        next_items: list[Any] = []
        for item in current:
            next_items.extend(_flatten_stage_output(stage.process(item)))
        current = next_items
    return current


def _patch_common_agent_pipeline_fakes(stages: list[Any]) -> None:  # noqa: C901 (complexity accepted: one fake-attachment branch per heavy stage type)
    """Attach deterministic fake model outputs to heavy stages in a built pipeline."""

    def fake_vad_segments(self: VADSegmentationStage, _waveform: torch.Tensor, _sample_rate: int) -> list[dict[str, float]]:
        if self.nested:
            return [{"start": 0.0, "end": 0.2}, {"start": 0.2, "end": 0.4}]
        return [{"start": 0.0, "end": 0.1}]

    def fake_speaker_audio_data(*_args: Any, **_kwargs: Any) -> dict[str, SimpleNamespace]:
        return {
            "speaker_0": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.4, diar_segments=[[0.0, 0.2]]),
            "speaker_1": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.4, diar_segments=[[0.2, 0.4]]),
        }

    for stage in stages:
        if isinstance(stage, VADSegmentationStage):
            stage._vad_model = object()
            stage._get_vad_segments = MethodType(fake_vad_segments, stage)
        elif isinstance(stage, WhisperXVADStage):
            stage._vad_model = SimpleNamespace(
                get_vad_segments=lambda _audio, _max_length, sample_rate=16000: [
                    {"start": 0.0, "end": 0.35, "text": "speech"},
                    {"start": 0.35, "end": 0.7, "text": "speech"},
                ]
            )
        elif isinstance(stage, UTMOSFilterStage):
            stage._model = _FakeUTMOSModel(4.6)
        elif isinstance(stage, SIGMOSFilterStage):
            stage._model = _FakeSIGMOSModel(4.6)
        elif isinstance(stage, BandFilterStage):
            stage._predictor = _FakeBandPredictor("full_band")
        elif isinstance(stage, TorchSquimQualityMetricsStage):
            stage._compute_metrics_batched = MethodType(
                lambda self, waveforms: [(3.2, 0.95, 14.0)] * len(waveforms),
                stage,
            )
        elif isinstance(stage, SpeakerSeparationStage):
            stage._separator = SimpleNamespace(get_speaker_audio_data=fake_speaker_audio_data)


def _coverage_cases(tmp_path: Path) -> list[StageCase]:
    manifest = tmp_path / "manifest.jsonl"
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    manifest.write_text("", encoding="utf-8")
    return [
        StageCase(AudioDataFilterStage, lambda: AudioDataFilterStage(config={"vad": {"enable": False}}), "composite"),
        StageCase(ALMDataBuilderStage, ALMDataBuilderStage, "alm"),
        StageCase(ALMDataOverlapStage, ALMDataOverlapStage, "alm"),
        StageCase(AudioToDocumentStage, AudioToDocumentStage, "ingress_transform"),
        StageCase(BandFilterStage, BandFilterStage, "segmentation_quality"),
        StageCase(BandwidthEstimationStage, BandwidthEstimationStage, "segmentation_quality"),
        StageCase(ChineseConversionStage, ChineseConversionStage, "speech_tagging"),
        StageCase(
            CreateInitialManifestFleursStage,
            lambda: CreateInitialManifestFleursStage(lang="en_us", split="dev", raw_data_dir=str(tmp_path)),
            "dataset_contract",
        ),
        StageCase(
            CreateInitialManifestReadSpeechStage,
            lambda: CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path), auto_download=False),
            "dataset_contract",
        ),
        StageCase(ComputeWERStage, ComputeWERStage, "speech_tagging"),
        StageCase(GetAudioDurationStage, GetAudioDurationStage, "ingress_transform"),
        StageCase(GetPairwiseWerStage, GetPairwiseWerStage, "speech_tagging"),
        StageCase(
            InferenceAsrNemoStage,
            lambda: InferenceAsrNemoStage(asr_model=object()),
            "speech_tagging",
        ),
        StageCase(InferenceSortformerStage, lambda: InferenceSortformerStage(diar_model=object()), "speech_tagging"),
        StageCase(InverseTextNormalizationStage, InverseTextNormalizationStage, "speech_tagging"),
        StageCase(JoinSplitAudioMetadataStage, JoinSplitAudioMetadataStage, "split_join_extract"),
        StageCase(ManifestReader, lambda: ManifestReader(manifest_path=str(manifest)), "composite"),
        StageCase(ManifestReaderStage, ManifestReaderStage, "ingress_transform"),
        StageCase(ManifestWriterStage, lambda: ManifestWriterStage(output_path=str(output_dir / "out.jsonl")), "ingress_transform"),
        StageCase(MergeAlignmentDiarizationStage, MergeAlignmentDiarizationStage, "speech_tagging"),
        StageCase(MonoConversionStage, MonoConversionStage, "ingress_transform"),
        StageCase(NeMoASRAlignerStage, NeMoASRAlignerStage, "speech_tagging"),
        StageCase(OverlapFilterStage, OverlapFilterStage, "alm_pretrain"),
        StageCase(PretrainMetricsAggregatorStage, lambda: PretrainMetricsAggregatorStage(str(output_dir / "metrics.json")), "alm_pretrain"),
        StageCase(PrepareModuleSegmentsStage, PrepareModuleSegmentsStage, "speech_tagging"),
        StageCase(PreserveByValueStage, lambda: PreserveByValueStage("keep", True), "segmentation_quality"),
        StageCase(PyAnnoteDiarizationStage, lambda: PyAnnoteDiarizationStage(hf_token="fake"), "speech_tagging"),  # noqa: S106 - not a real credential
        StageCase(ReadLongFormManifestStage, lambda: ReadLongFormManifestStage(str(manifest), str(audio_dir)), "alm_pretrain"),
        StageCase(ResampleAudioStage, lambda: ResampleAudioStage(resampled_audio_dir=str(output_dir)), "ingress_transform"),
        StageCase(SIGMOSFilterStage, SIGMOSFilterStage, "segmentation_quality"),
        StageCase(SegmentConcatenationStage, SegmentConcatenationStage, "split_join_extract"),
        StageCase(SegmentExtractionStage, lambda: SegmentExtractionStage(output_dir=str(output_dir)), "split_join_extract"),
        StageCase(SnippetCutPlannerStage, SnippetCutPlannerStage, "alm_pretrain"),
        StageCase(SnippetExtractionStage, lambda: SnippetExtractionStage(str(output_dir), str(output_dir / "audio.tar"), dry_run=True), "alm_pretrain"),
        StageCase(SnippetManifestWriterStage, lambda: SnippetManifestWriterStage(str(output_dir / "snippets.jsonl")), "alm_pretrain"),
        StageCase(SnippetRepetitionFilterStage, lambda: SnippetRepetitionFilterStage(tokenizer_path=str(tmp_path)), "alm_pretrain"),
        StageCase(SpeakerSeparationStage, SpeakerSeparationStage, "segmentation_quality"),
        StageCase(SplitASRAlignJoinStage, SplitASRAlignJoinStage, "composite"),
        StageCase(SplitLongAudioStage, SplitLongAudioStage, "split_join_extract"),
        StageCase(TimestampMapperStage, TimestampMapperStage, "split_join_extract"),
        StageCase(TorchSquimQualityMetricsStage, TorchSquimQualityMetricsStage, "segmentation_quality"),
        StageCase(UTMOSFilterStage, UTMOSFilterStage, "segmentation_quality"),
        StageCase(VADSegmentationStage, VADSegmentationStage, "segmentation_quality"),
        StageCase(WhisperXVADStage, WhisperXVADStage, "speech_tagging"),
    ]


def _discover_agent_ready_classes() -> set[type]:
    modules = [
        AudioDataFilterStage,
        ALMDataBuilderStage,
        ALMDataOverlapStage,
        AudioToDocumentStage,
        BandFilterStage,
        BandwidthEstimationStage,
        ChineseConversionStage,
        CreateInitialManifestFleursStage,
        CreateInitialManifestReadSpeechStage,
        ComputeWERStage,
        GetAudioDurationStage,
        GetPairwiseWerStage,
        InferenceAsrNemoStage,
        InferenceSortformerStage,
        InverseTextNormalizationStage,
        JoinSplitAudioMetadataStage,
        ManifestReader,
        ManifestReaderStage,
        ManifestWriterStage,
        MergeAlignmentDiarizationStage,
        MonoConversionStage,
        NeMoASRAlignerStage,
        OverlapFilterStage,
        PretrainMetricsAggregatorStage,
        PrepareModuleSegmentsStage,
        PyAnnoteDiarizationStage,
        ReadLongFormManifestStage,
        ResampleAudioStage,
        SIGMOSFilterStage,
        SegmentConcatenationStage,
        SegmentExtractionStage,
        SnippetCutPlannerStage,
        SnippetExtractionStage,
        SnippetManifestWriterStage,
        SnippetRepetitionFilterStage,
        SpeakerSeparationStage,
        SplitASRAlignJoinStage,
        SplitLongAudioStage,
        TimestampMapperStage,
        TorchSquimQualityMetricsStage,
        UTMOSFilterStage,
        VADSegmentationStage,
        WhisperXVADStage,
    ]
    package_modules = {inspect.getmodule(cls) for cls in modules}
    discovered: set[type] = set()
    for module in package_modules:
        assert module is not None
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in (AgentReady, BaseASRProcessorStage):
                continue
            if cls.__module__.startswith("nemo_curator.stages.audio") and issubclass(cls, AgentReady):
                discovered.add(cls)
    return discovered


def _assert_data_writes_present(stage: AgentReady, data: dict[str, Any]) -> None:
    contract = stage.describe()
    for key in contract.writes.data_keys:
        assert key in data, f"{type(stage).__name__} declared top-level write {key!r} but did not produce it"
    if contract.writes.segment_data_keys:
        segments = data.get(getattr(stage, "segments_key", "segments"), [])
        assert segments, f"{type(stage).__name__} declared segment writes but no segments were present"
        for key in contract.writes.segment_data_keys:
            assert any(key in segment for segment in segments), (
                f"{type(stage).__name__} declared segment write {key!r} but no segment produced it"
            )


def _contract_reads_satisfied(stage: AgentReady, available_keys: set[str]) -> bool:
    contract = stage.describe()
    if contract.reads.data_keys and not set(contract.reads.data_keys).issubset(available_keys):
        return False
    if contract.reads_one_of:
        return any(set(option.data_keys).issubset(available_keys) for option in contract.reads_one_of)
    return True


def _record_contract_writes(stage: AgentReady, available_keys: set[str]) -> None:
    contract = stage.describe()
    available_keys.update(contract.writes.data_keys)
    available_keys.update(contract.writes.segment_data_keys)


def _assert_no_duplicates(values: list[str], label: str) -> None:
    assert len(values) == len(set(values)), f"{label} contains duplicate values: {values!r}"


def test_agent_stage_registry_contracts_and_coverage_matrix(tmp_path: Path) -> None:
    cases = _coverage_cases(tmp_path)
    by_class = {case.cls: case for case in cases}
    discovered = _discover_agent_ready_classes()

    assert len(cases) == EXPECTED_AUDIO_AGENT_READY_STAGE_COUNT
    assert len(discovered) == EXPECTED_AUDIO_AGENT_READY_STAGE_COUNT
    assert set(by_class) == discovered
    assert len(by_class) == len(cases), "coverage matrix has duplicate stage classes"

    composite_names = {"AudioDataFilterStage", "ManifestReader", "SplitASRAlignJoinStage"}
    scenarios = {case.scenario for case in cases}
    assert VALID_AGENT_SCENARIOS.issubset(scenarios)

    for case in cases:
        stage = case.factory()
        contract = stage.describe()
        assert isinstance(stage, AgentReady)
        assert isinstance(contract, StageContract)
        assert contract.wrappable is (case.cls.__name__ not in composite_names)


def test_agent_all_44_stage_contracts_are_planner_safe(tmp_path: Path) -> None:
    cases = _coverage_cases(tmp_path)
    assert len(cases) == EXPECTED_AUDIO_AGENT_READY_STAGE_COUNT

    for case in cases:
        stage = case.factory()
        contract = stage.describe()
        name = case.cls.__name__

        assert case.scenario in VALID_AGENT_SCENARIOS, f"{name} has unknown scenario {case.scenario!r}"
        assert isinstance(contract, StageContract), f"{name}.describe() did not return StageContract"
        assert contract.cardinality in {"1:1", "1:1 nested-list", "1:N fan-out", "N:1", "filter"}
        if contract.iteration_key:
            assert contract.cardinality in {"1:1 nested-list", "1:N fan-out", "N:1"}

        _assert_no_duplicates(contract.reads.data_keys, f"{name}.reads.data_keys")
        _assert_no_duplicates(contract.reads.segment_data_keys, f"{name}.reads.segment_data_keys")
        _assert_no_duplicates(contract.writes.data_keys, f"{name}.writes.data_keys")
        _assert_no_duplicates(contract.writes.segment_data_keys, f"{name}.writes.segment_data_keys")
        _assert_no_duplicates(contract.metadata_reads, f"{name}.metadata_reads")
        _assert_no_duplicates(contract.metadata_writes, f"{name}.metadata_writes")
        _assert_no_duplicates(contract.gates.runtime_secrets, f"{name}.gates.runtime_secrets")

        assert set(contract.reads.accepts).issubset(VALID_ACCEPTS)
        assert set(contract.writes.produces).issubset(VALID_PRODUCES)
        for index, option in enumerate(contract.reads_one_of):
            _assert_no_duplicates(option.data_keys, f"{name}.reads_one_of[{index}].data_keys")
            _assert_no_duplicates(option.segment_data_keys, f"{name}.reads_one_of[{index}].segment_data_keys")
            assert set(option.accepts).issubset(VALID_ACCEPTS)
            assert set(option.produces).issubset(VALID_PRODUCES)

        if contract.cardinality == "filter" and contract.cardinality_options:
            assert "filter" in contract.cardinality_options, f"{name} filter contract lacks filter option"

        if case.scenario == "composite":
            assert contract.wrappable is False, f"{name} composite should not be directly wrappable"


def test_agent_ingress_transform_pipeline_custom_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.stages.audio.tagging import resample_audio as resample_module

    source = _write_wav(tmp_path / "source.wav", channels=2)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"agent_audio_path": str(source), "text": "hello world", "keep": True}) + "\n")

    reader = ManifestReaderStage()
    tasks = reader.process(FileGroupTask(dataset_name="agent", data=[str(manifest)]))
    assert len(tasks) == 1
    task = tasks[0]

    duration = GetAudioDurationStage(audio_filepath_key="agent_audio_path", duration_key="agent_duration")
    task = duration.process(task)
    assert task.data["agent_duration"] > 0

    mono = MonoConversionStage(
        audio_filepath_key="agent_audio_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        duration_key="agent_duration",
        output_audio_filepath_key="agent_mono_path",
        output_sample_rate=16000,
        input_residency="file",
        keep_waveform_in_task=True,
        write_to_disk=True,
        update_audio_filepath=True,
        output_dir=str(tmp_path / "mono"),
    )
    task = mono.process(task)
    _assert_data_writes_present(mono, task.data)
    assert task.data["agent_waveform"].shape[0] == 1

    monkeypatch.setattr(resample_module.subprocess, "run", _copy_file_for_fake_ffmpeg)
    resample = ResampleAudioStage(
        resampled_audio_dir=str(tmp_path / "resampled"),
        audio_filepath_key="agent_audio_path",
        resampled_audio_filepath_key="agent_resampled_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        duration_key="agent_duration",
        input_residency="waveform",
        keep_waveform_in_task=True,
        write_to_disk=True,
    )
    task = resample.process(task)
    _assert_data_writes_present(resample, task.data)
    assert os.path.exists(task.data["agent_resampled_path"])

    writer_task = AudioTask(
        dataset_name=task.dataset_name,
        data={key: value for key, value in task.data.items() if key != "agent_waveform"},
        _metadata=task._metadata,
        _stage_perf=list(task._stage_perf),
    )
    writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
    writer.setup()
    writer.process(writer_task)
    assert (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip()

    docs = AudioToDocumentStage(keep_keys=["text", "agent_duration", "agent_resampled_path"]).process_batch([task])
    assert docs[0].to_pandas().iloc[0]["agent_resampled_path"] == task.data["agent_resampled_path"]


def test_agent_segmentation_quality_pipeline_annotate_and_filter(tmp_path: Path) -> None:
    audio_path = _write_wav(tmp_path / "quality.wav", duration_sec=1.0)
    task = AudioTask(
        dataset_name="agent",
        data={
            "agent_audio_path": str(audio_path),
            "agent_waveform": _waveform(duration_sec=1.0),
            "agent_sr": 16000,
            "keep": True,
        },
    )

    vad = VADSegmentationStage(
        audio_filepath_key="agent_audio_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        start_ms_key="agent_start_ms",
        end_ms_key="agent_end_ms",
        duration_key="agent_duration",
        nested=True,
        input_residency="waveform",
    )
    vad._vad_model = object()
    vad._get_vad_segments = MethodType(lambda self, waveform, sample_rate: [{"start": 0.0, "end": 0.45}, {"start": 0.45, "end": 0.9}], vad)
    task = vad.process(task)
    assert len(task.data["agent_segments"]) == 2
    for segment in task.data["agent_segments"]:
        segment["start"] = segment["agent_start_ms"] / 1000.0
        segment["end"] = segment["agent_end_ms"] / 1000.0
        segment["speaker"] = "spk0"
        segment["text"] = "speech"

    utmos = UTMOSFilterStage(
        mos_threshold=4.0,
        action="annotate",
        mode="segments",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        score_key="agent_utmos",
    )
    utmos._model = _FakeUTMOSModel(2.0)
    task = utmos.process(task)
    assert len(task.data["agent_segments"]) == 2
    assert all(segment["agent_utmos"] == 2.0 for segment in task.data["agent_segments"])

    sigmos = SIGMOSFilterStage(
        noise_threshold=4.0,
        ovrl_threshold=4.0,
        action="annotate",
        mode="segments",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        noise_key="agent_sigmos_noise",
        ovrl_key="agent_sigmos_ovrl",
    )
    sigmos._model = _FakeSIGMOSModel(2.0)
    task = sigmos.process(task)
    assert len(task.data["agent_segments"]) == 2
    assert all(segment["agent_sigmos_noise"] == 2.0 for segment in task.data["agent_segments"])

    band = BandFilterStage(
        band_value="full_band",
        action="annotate",
        mode="segments",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        prediction_key="agent_band",
    )
    band._predictor = _FakeBandPredictor("narrow_band")
    task = band.process(task)
    assert all(segment["agent_band"] == "narrow_band" for segment in task.data["agent_segments"])

    squim = TorchSquimQualityMetricsStage(
        audio_filepath_key="agent_audio_path",
        segments_key="agent_segments",
        metrics_key="agent_metrics",
        resources=Resources(gpus=0.0),
    )
    squim._compute_metrics_batched = MethodType(lambda self, waveforms: [(3.1, 0.9, 12.0)] * len(waveforms), squim)
    task = squim.process_batch([task])[0]
    assert all("pesq_squim" in segment["agent_metrics"] for segment in task.data["agent_segments"])

    bandwidth = BandwidthEstimationStage(
        audio_filepath_key="agent_audio_path",
        segments_key="agent_segments",
        metrics_key="agent_metrics",
    )
    assert bandwidth.validate_input(task)
    task = bandwidth.process(task)
    assert all("bandwidth" in segment["agent_metrics"] for segment in task.data["agent_segments"])

    assert PreserveByValueStage("keep", True).process_batch([task]) == [task]

    filter_stage = UTMOSFilterStage(
        mos_threshold=4.0,
        action="filter",
        mode="segments",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        score_key="agent_utmos",
    )
    filter_stage._model = _FakeUTMOSModel(2.0)
    filtered = filter_stage.process(
        AudioTask(
            data={
                "agent_segments": [
                    {"agent_waveform": _waveform(duration_sec=0.2), "agent_sr": 16000},
                    {"agent_waveform": _waveform(duration_sec=0.2), "agent_sr": 16000},
                ]
            }
        )
    )
    assert filtered == []


def test_agent_yaml_custom_key_quality_pipeline_to_document(tmp_path: Path) -> None:
    """Agent-authored tutorial-style YAML can set custom keys and reach a clean document."""
    audio_path = _write_wav(tmp_path / "yaml_custom_quality.wav", duration_sec=1.0, channels=2)
    pipeline = _pipeline_from_agent_yaml(
        {
            "processors": [
                {
                    "_target_": "nemo_curator.stages.audio.preprocessing.mono_conversion.MonoConversionStage",
                    "audio_filepath_key": "agent_audio_path",
                    "waveform_key": "agent_waveform",
                    "sample_rate_key": "agent_sr",
                    "is_mono_key": "agent_is_mono",
                    "duration_key": "agent_duration",
                    "num_samples_key": "agent_num_samples",
                    "output_sample_rate": 16000,
                    "strict_sample_rate": False,
                    "input_residency": "file",
                    "keep_waveform_in_task": True,
                    "write_to_disk": False,
                },
                {
                    "_target_": "nemo_curator.stages.audio.segmentation.vad_segmentation.VADSegmentationStage",
                    "audio_filepath_key": "agent_audio_path",
                    "waveform_key": "agent_waveform",
                    "sample_rate_key": "agent_sr",
                    "segments_key": "agent_segments",
                    "segment_num_key": "agent_segment_num",
                    "duration_key": "agent_segment_duration",
                    "original_file_key": "agent_original_file",
                    "nested": True,
                    "input_residency": "waveform",
                    "keep_segment_waveform_in_task": True,
                    "min_duration_sec": 0.05,
                },
                {
                    "_target_": "nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage",
                    "action": "annotate",
                    "mode": "segments",
                    "input_residency": "waveform",
                    "waveform_key": "agent_waveform",
                    "sample_rate_key": "agent_sr",
                    "segments_key": "agent_segments",
                    "score_key": "agent_utmos",
                    "mos_threshold": 4.0,
                    "resources": {"gpus": 0.0},
                },
                {
                    "_target_": "nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage",
                    "action": "annotate",
                    "mode": "segments",
                    "input_residency": "waveform",
                    "waveform_key": "agent_waveform",
                    "sample_rate_key": "agent_sr",
                    "segments_key": "agent_segments",
                    "noise_key": "agent_sigmos_noise",
                    "ovrl_key": "agent_sigmos_ovrl",
                    "noise_threshold": 4.0,
                    "ovrl_threshold": 4.0,
                    "resources": {"gpus": 0.0},
                },
                {
                    "_target_": "nemo_curator.stages.audio.filtering.band.BandFilterStage",
                    "action": "filter",
                    "mode": "segments",
                    "input_residency": "waveform",
                    "waveform_key": "agent_waveform",
                    "sample_rate_key": "agent_sr",
                    "segments_key": "agent_segments",
                    "prediction_key": "agent_band",
                    "band_value": "full_band",
                    "resources": {"gpus": 0.0},
                },
                {
                    "_target_": "nemo_curator.stages.audio.io.convert.AudioToDocumentStage",
                    "keep_keys": ["agent_audio_path", "agent_duration", "agent_is_mono", "agent_segments"],
                    "segments_key": "agent_segments",
                    "serialize_segments": True,
                },
            ],
        }
    )
    pipeline.build()
    _patch_common_agent_pipeline_fakes(pipeline.stages)

    outputs = _run_inline_agent_pipeline(
        pipeline.stages,
        [
            AudioTask(
                dataset_name="agent_yaml",
                data={"agent_audio_path": str(audio_path), "text": "synthetic speech"},
                _metadata={"trace_id": "yaml-custom"},
                _stage_perf=["seed"],
            )
        ],
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], DocumentBatch)
    row = outputs[0].to_pandas().iloc[0].to_dict()
    assert row["agent_audio_path"] == str(audio_path)
    assert row["agent_is_mono"] is True
    assert len(row["agent_segments"]) == 2
    for segment in row["agent_segments"]:
        assert segment["agent_utmos"] == pytest.approx(4.6)
        assert segment["agent_sigmos_noise"] == 4.6
        assert segment["agent_band"] == "full_band"
        assert "agent_waveform" not in segment
        assert not any(type(value).__name__ == "Tensor" for value in segment.values())


def test_agent_yaml_timing_metrics_pipeline_uses_file_segments(tmp_path: Path) -> None:
    """A second YAML path covers file-backed VAD segments feeding SQUIM and bandwidth."""
    audio_path = _write_wav(tmp_path / "yaml_timing_metrics.wav", duration_sec=1.0)
    pipeline = _pipeline_from_agent_yaml(
        {
            "processors": [
                {
                    "_target_": "nemo_curator.stages.audio.inference.vad.whisperx_vad.WhisperXVADStage",
                    "audio_filepath_key": "agent_audio_path",
                    "segments_key": "agent_timing_segments",
                    "resources": {"gpus": 0.0},
                },
                {
                    "_target_": "nemo_curator.stages.audio.metrics.squim.TorchSquimQualityMetricsStage",
                    "audio_filepath_key": "agent_audio_path",
                    "segments_key": "agent_timing_segments",
                    "metrics_key": "agent_metrics",
                    "resources": {"gpus": 0.0},
                },
                {
                    "_target_": "nemo_curator.stages.audio.metrics.bandwidth.BandwidthEstimationStage",
                    "audio_filepath_key": "agent_audio_path",
                    "segments_key": "agent_timing_segments",
                    "metrics_key": "agent_metrics",
                },
                {
                    "_target_": "nemo_curator.stages.audio.io.convert.AudioToDocumentStage",
                    "keep_keys": ["agent_audio_path", "agent_timing_segments"],
                    "segments_key": "agent_timing_segments",
                    "serialize_segments": True,
                },
            ],
        }
    )
    pipeline.build()
    _patch_common_agent_pipeline_fakes(pipeline.stages)

    outputs = _run_inline_agent_pipeline(
        pipeline.stages,
        [AudioTask(dataset_name="agent_yaml", data={"agent_audio_path": str(audio_path), "text": "speech"})],
    )

    assert len(outputs) == 1
    row = outputs[0].to_pandas().iloc[0].to_dict()
    assert len(row["agent_timing_segments"]) == 2
    metrics = row["agent_timing_segments"][0]["agent_metrics"]
    assert metrics["pesq_squim"] == 3.2
    assert metrics["stoi_squim"] == 0.95
    assert "bandwidth" in metrics


def test_agent_yaml_audio_data_filter_topologies_match_contracts() -> None:
    """ReadSpeech-style AudioDataFilter YAML expands into compatible concrete modules."""

    def build_names(*, vad: bool, speaker: bool) -> tuple[list[str], list[Any]]:
        pipeline = _pipeline_from_agent_yaml(
            {
                "processors": [
                    {
                        "_target_": (
                            "nemo_curator.stages.audio.advanced_pipelines.audio_data_filter."
                            "audio_data_filter.AudioDataFilterStage"
                        ),
                        "config": {
                            "mono_conversion": {"output_sample_rate": 16000, "strict_sample_rate": False},
                            "vad": {"enable": vad, "gpus": 0.0, "min_duration_sec": 0.05},
                            "band_filter": {"enable": True, "gpus": 0.0},
                            "utmos": {"enable": True, "gpus": 0.0},
                            "sigmos": {"enable": True, "gpus": 0.0},
                            "speaker_separation": {"enable": speaker, "gpus": 0.0, "min_duration": 0.05},
                        },
                    }
                ],
            }
        )
        pipeline.build()
        return [stage.__class__.__name__ for stage in pipeline.stages], pipeline.stages

    expected = {
        (False, False): [
            "MonoConversionStage",
            "BandFilterStage",
            "UTMOSFilterStage",
            "SIGMOSFilterStage",
            "TimestampMapperStage",
        ],
        (True, False): [
            "MonoConversionStage",
            "VADSegmentationStage",
            "BandFilterStage",
            "UTMOSFilterStage",
            "SIGMOSFilterStage",
            "TimestampMapperStage",
        ],
        (False, True): [
            "MonoConversionStage",
            "BandFilterStage",
            "UTMOSFilterStage",
            "SIGMOSFilterStage",
            "SpeakerSeparationStage",
            "BandFilterStage",
            "UTMOSFilterStage",
            "SIGMOSFilterStage",
            "TimestampMapperStage",
        ],
        (True, True): [
            "MonoConversionStage",
            "VADSegmentationStage",
            "BandFilterStage",
            "UTMOSFilterStage",
            "SIGMOSFilterStage",
            "SegmentConcatenationStage",
            "SpeakerSeparationStage",
            "VADSegmentationStage",
            "BandFilterStage",
            "UTMOSFilterStage",
            "SIGMOSFilterStage",
            "TimestampMapperStage",
        ],
    }

    for (vad, speaker), expected_names in expected.items():
        names, stages = build_names(vad=vad, speaker=speaker)
        assert names == expected_names
        for stage in stages:
            assert isinstance(stage.describe(), StageContract)
        vad_stages = [stage for stage in stages if isinstance(stage, VADSegmentationStage)]
        if vad and speaker:
            assert [stage.nested for stage in vad_stages] == [True, False]
        elif vad:
            assert [stage.nested for stage in vad_stages] == [False]


def test_agent_yaml_audio_data_filter_full_pipeline_dataflow(tmp_path: Path) -> None:
    """Run the full composite topology inline with fake models and verify key handoff."""
    audio_path = _write_wav(tmp_path / "audio_data_filter_full.wav", duration_sec=1.0)
    pipeline = _pipeline_from_agent_yaml(
        {
            "processors": [
                {
                    "_target_": (
                        "nemo_curator.stages.audio.advanced_pipelines.audio_data_filter."
                        "audio_data_filter.AudioDataFilterStage"
                    ),
                    "config": {
                        "mono_conversion": {"output_sample_rate": 16000, "strict_sample_rate": False},
                        "vad": {"enable": True, "gpus": 0.0, "min_duration_sec": 0.05},
                        "band_filter": {"enable": True, "gpus": 0.0},
                        "utmos": {"enable": True, "gpus": 0.0},
                        "sigmos": {"enable": True, "gpus": 0.0},
                        "concatenation": {"silence_duration_sec": 0.0},
                        "speaker_separation": {"enable": True, "gpus": 0.0, "min_duration": 0.05},
                        "timestamp_mapper": {
                            "passthrough_keys": [
                                "speaker_id",
                                "num_speakers",
                                "sample_rate",
                                "band_prediction",
                                "utmos_mos",
                                "sigmos_noise",
                                "sigmos_ovrl",
                            ]
                        },
                    },
                }
            ],
        }
    )
    pipeline.build()
    _patch_common_agent_pipeline_fakes(pipeline.stages)

    outputs = _run_inline_agent_pipeline(
        pipeline.stages,
        [
            AudioTask(
                dataset_name="agent_filter",
                data={"audio_filepath": str(audio_path), "text": "speech", "audio_item_id": "filter_item"},
                _metadata={"trace_id": "audio-data-filter"},
                _stage_perf=["seed"],
            )
        ],
    )

    assert len(outputs) == 2
    # Multi-speaker timing: TimestampMapper maps each per-speaker VAD segment's
    # start_ms/end_ms through the concat->original mappings as CONCAT-TIME (the
    # separator emits full-length per-speaker stems, so VAD_Speaker's start/end are
    # concat-time, not clip-relative). The fake VAD_Speaker returns the same
    # [0.0, 0.1]s window for every stem, so both speakers map to original [0, 100]ms
    # here — a fixture limitation, not the mapper. The point verified is that the
    # mapped start/end is used: an earlier guard instead discarded start/end and
    # spanned each speaker's diar-segment UNION, which would give (0, 200)/(200, 400)
    # and duplicate whole-clip rows on real multi-VAD-segment audio. Per-speaker
    # distinctness is proven directly in
    # test_timestamp_mapper_multispeaker_maps_distinct_windows below.
    for task in outputs:
        assert isinstance(task, AudioTask)
        assert task.data["original_file"] == str(audio_path)
        assert (task.data["original_start_ms"], task.data["original_end_ms"]) == (0, 100)
        assert task.data["duration"] == pytest.approx(0.1)
        assert task.data["band_prediction"] == "full_band"
        assert task.data["utmos_mos"] == pytest.approx(4.6)
        assert task.data["sigmos_noise"] == 4.6
        assert task.data["speaker_id"] in {"speaker_0", "speaker_1"}
        assert task._metadata["trace_id"] == "audio-data-filter"
        assert "segment_mappings" in task._metadata
        assert "seed" in task._stage_perf
    assert {task.data["speaker_id"] for task in outputs} == {"speaker_0", "speaker_1"}


def test_timestamp_mapper_multispeaker_maps_distinct_windows() -> None:
    """Regression: SpeakerSep->VAD per-speaker segments map to DISTINCT original windows.

    After SpeakerSeparation the separator emits full-length stems (each speaker's
    audio overlaid on a silent track at its concat-time position), so VAD_Speaker's
    start_ms/end_ms are concat-time and must be mapped directly through the
    concat->original mappings. A removed guard used to discard start_ms/end_ms
    whenever diar_segments were also present and span the diar UNION instead.

    The fixture is DISCRIMINATING: each speaker's VAD start_ms/end_ms is a strict
    sub-interval of its diar segment, so the two branches produce different output.
    Direct mapping (the fix) yields the narrow refined window; the removed
    diar-union guard would yield the wider whole-segment window. Asserting the
    narrow windows therefore fails on the guarded code and pins the fix.
    """
    # Two concat segments; original coords differ from concat so translation is visible.
    mappings = [
        {"original_file": "/audio.wav", "original_start_ms": 0, "concat_start_ms": 0, "concat_end_ms": 400},
        {"original_file": "/audio.wav", "original_start_ms": 1000, "concat_start_ms": 400, "concat_end_ms": 800},
    ]
    mapper = TimestampMapperStage(passthrough_keys=["speaker_id"])
    spans = {}
    for speaker_id, (start_ms, end_ms), diar in [
        # VAD window (100,300) sits inside diar seg [0.0,0.4]=concat[0,400] -> fix maps to original (100,300)
        ("speaker_0", (100, 300), [[0.0, 0.4]]),
        # VAD window (500,700) sits inside diar seg [0.4,0.8]=concat[400,800] -> fix maps to original (1100,1300)
        ("speaker_1", (500, 700), [[0.4, 0.8]]),
    ]:
        task = AudioTask(
            dataset_name="multispeaker",
            data={
                "start_ms": start_ms,
                "end_ms": end_ms,
                "diar_segments": diar,  # present alongside start/end — must NOT trigger union collapse
                "speaker_id": speaker_id,
                "original_file": "/audio.wav",
            },
            _metadata={"segment_mappings": mappings},
        )
        out = mapper.process(task)
        assert isinstance(out, AudioTask)  # not dropped
        spans[speaker_id] = (out.data["original_start_ms"], out.data["original_end_ms"])
    # narrow refined windows; the removed guard would give (0,400) and (1000,1400)
    assert spans == {"speaker_0": (100, 300), "speaker_1": (1100, 1300)}


def test_agent_speech_tagging_pipeline_with_fake_inference(tmp_path: Path) -> None:
    audio_path = _write_wav(tmp_path / "speech.wav", duration_sec=1.0)
    task = _audio_task(audio_path)

    whisperx = WhisperXVADStage(segments_key="agent_vad_segments", resources=Resources(gpus=0.0))
    whisperx._vad_model = _FakeVADModel()
    task = whisperx.process(task)
    assert task.data["agent_vad_segments"] == [{"start": 0.0, "end": 0.4}]

    pyannote = PyAnnoteDiarizationStage(hf_token="fake", write_rttm=False, segments_key="agent_segments", overlap_segments_key="agent_overlaps")  # noqa: S106 - not a real credential
    pyannote.process = MethodType(
        lambda self, t: t.data.update(
            {
                self.segments_key: [{"speaker": "spk0", "start": 0.0, "end": 0.4}],
                self.overlap_segments_key: [],
            }
        )
        or t,
        pyannote,
    )
    task = pyannote.process(task)

    sortformer = InferenceSortformerStage(
        diar_model=object(),
        filepath_key="audio_filepath",
        diar_segments_key="agent_diar_segments",
        rttm_out_dir=None,
    )
    sortformer.diarize = MethodType(lambda self, paths: [[{"speaker": "spk0", "start": 0.0, "end": 0.4}]], sortformer)
    task = sortformer.process(task)
    assert task.data["agent_diar_segments"][0]["speaker"] == "spk0"

    asr = InferenceAsrNemoStage(asr_model=object(), filepath_key="audio_filepath", pred_text_key="agent_pred_text")
    asr.transcribe = MethodType(lambda self, files: ["hello world" for _ in files], asr)
    task = asr.process_batch([task])[0]
    assert task.data["agent_pred_text"] == "hello world"

    aligner = NeMoASRAlignerStage(text_key="agent_text", words_key="agent_words", alignment_key="agent_alignment", segments_key="agent_segments")

    def fake_align(self: NeMoASRAlignerStage, tasks: list[AudioTask]) -> list[AudioTask]:
        for item in tasks:
            item.data[self.alignment_key] = [{"word": "hello", "start": 0.0, "end": 0.4}]
            item.data[self.text_key] = "hello"
        return tasks

    aligner.process_batch = MethodType(fake_align, aligner)
    task = aligner.process_batch([task])[0]

    merge = MergeAlignmentDiarizationStage(
        text_key="agent_text",
        words_key="agent_words",
        alignment_key="agent_alignment",
        segments_key="agent_segments",
    )
    task = merge.process(task)
    assert task.data["agent_segments"][0]["agent_text"] == "hello"
    assert task.data["agent_segments"][0]["agent_words"][0]["word"] == "hello"

    itn = InverseTextNormalizationStage(text_key="agent_text", segments_key="agent_segments")
    itn._normalizer = _FakeNormalizer()
    task = itn.process(task)

    chinese = ChineseConversionStage(text_key="agent_text", segments_key="agent_segments")
    chinese._converter = _FakeConverter()
    task = chinese.process(task)
    assert "agent_text_ITN" in task.data["agent_segments"][0]
    assert "agent_text_simplified" in task.data["agent_segments"][0]

    wer = ComputeWERStage(hypothesis_text_key="agent_text", reference_text_key="text_ref", segments_key="agent_segments")
    wer._normalizer = _FakeNormalizer()
    task.data["agent_segments"][0]["text_ref"] = "hello"
    assert wer.validate_input(task)
    task = wer.process(task)
    assert "wer" in task.data["agent_segments"][0]["metrics"]

    pairwise = GetPairwiseWerStage(text_key="text", pred_text_key="agent_pred_text", wer_key="agent_wer_pct")
    task = pairwise.process(task)
    assert task.data["agent_wer_pct"] == 0.0

    prep = PrepareModuleSegmentsStage(segments_key="agent_segments", duration_key="duration")
    prepared = prep.process(task)
    assert "agent_segments" in prepared.data


def test_agent_optional_fanout_for_vad_and_diarizers_with_custom_keys(tmp_path: Path) -> None:  # noqa: PLR0915 (complexity accepted: end-to-end fan-out scenario across three diarizer stages)
    audio_path = _write_wav(tmp_path / "fanout.wav", duration_sec=1.0)

    default_whisperx = WhisperXVADStage(resources=Resources(gpus=0.0))
    assert default_whisperx.describe().cardinality == "1:1"
    assert default_whisperx.ray_stage_spec() == {}

    whisperx = WhisperXVADStage(
        resources=Resources(gpus=0.0),
        fanout=True,
        segments_key="agent_vad_segments",
        start_key="agent_start",
        end_key="agent_end",
        duration_key="agent_duration",
        segment_num_key="agent_segment_num",
        original_file_key="agent_original_file",
    )
    whisperx._vad_model = SimpleNamespace(
        get_vad_segments=lambda _audio, _max_length, sample_rate=16000: [
            {"start": 0.0, "end": 0.4},
            {"start": 0.4, "end": 0.9},
        ],
    )
    whisperx_children = whisperx.process(_audio_task(audio_path))

    assert isinstance(whisperx_children, list)
    assert len(whisperx_children) == 2
    assert whisperx.describe().cardinality == "1:N fan-out"
    assert whisperx.describe().iteration_key == "agent_vad_segments"
    assert whisperx.ray_stage_spec()[RayStageSpecKeys.IS_FANOUT_STAGE] is True
    assert "agent_vad_segments" not in whisperx_children[0].data
    assert whisperx_children[0].data["agent_start"] == 0.0
    assert whisperx_children[0].data["agent_end"] == 0.4
    assert whisperx_children[0].data["agent_duration"] == 0.4
    assert whisperx_children[0].data["agent_segment_num"] == 0
    assert whisperx_children[0].data["agent_original_file"] == str(audio_path)
    assert whisperx_children[0]._metadata == {"source": "agent"}
    assert whisperx_children[0]._stage_perf == ["upstream"]

    default_pyannote = PyAnnoteDiarizationStage(hf_token="fake", write_rttm=False)  # noqa: S106 - not a real credential
    assert default_pyannote.describe().cardinality == "1:1"
    assert default_pyannote.ray_stage_spec() == {}

    pyannote = PyAnnoteDiarizationStage(
        hf_token="fake",  # noqa: S106 - not a real credential
        write_rttm=False,
        min_length=0.1,
        fanout=True,
        segments_key="agent_segments",
        overlap_segments_key="agent_overlaps",
        start_key="agent_start",
        end_key="agent_end",
        speaker_key="agent_speaker",
        duration_key="agent_duration",
        segment_num_key="agent_segment_num",
        original_file_key="agent_original_file",
    )
    pyannote._pipeline = lambda _payload, hook=None: _FakePyAnnoteDiarization(
        [(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")]
    )
    pyannote_children = pyannote.process(_audio_task(audio_path))

    assert isinstance(pyannote_children, list)
    assert len(pyannote_children) == 2
    assert pyannote.describe().cardinality == "1:N fan-out"
    assert pyannote.describe().iteration_key == "agent_segments"
    assert pyannote.ray_stage_spec()[RayStageSpecKeys.IS_FANOUT_STAGE] is True
    assert "agent_segments" not in pyannote_children[0].data
    assert "agent_overlaps" not in pyannote_children[0].data
    assert pyannote_children[0].data["agent_speaker"] == "agent_item_SPEAKER_00"
    assert pyannote_children[1].data["agent_speaker"] == "agent_item_SPEAKER_01"
    assert pyannote_children[0]._metadata == {"source": "agent"}
    assert pyannote_children[0]._stage_perf == ["upstream"]

    default_sortformer = InferenceSortformerStage(diar_model=object())
    assert default_sortformer.describe().cardinality == "1:1"
    assert default_sortformer.ray_stage_spec() == {}

    sortformer = InferenceSortformerStage(
        diar_model=object(),
        fanout=True,
        diar_segments_key="agent_diar_segments",
        start_key="agent_start",
        end_key="agent_end",
        speaker_key="agent_speaker",
        duration_key="agent_duration",
        segment_num_key="agent_segment_num",
        original_file_key="agent_original_file",
    )
    def fake_sortformer_diarize(
        self: InferenceSortformerStage,  # noqa: ARG001 - bound via MethodType; receiver unused by the fake
        paths: list[str],  # noqa: ARG001
    ) -> list[list[dict[str, Any]]]:
        return [[{"speaker": "spk0", "start": 0.0, "end": 0.25}, {"speaker": "spk1", "start": 0.25, "end": 0.75}]]

    sortformer.diarize = MethodType(fake_sortformer_diarize, sortformer)
    sortformer_children = sortformer.process(_audio_task(audio_path))

    assert isinstance(sortformer_children, list)
    assert len(sortformer_children) == 2
    assert sortformer.describe().cardinality == "1:N fan-out"
    assert sortformer.describe().iteration_key == "agent_diar_segments"
    assert sortformer.ray_stage_spec()[RayStageSpecKeys.IS_FANOUT_STAGE] is True
    assert "agent_diar_segments" not in sortformer_children[0].data
    assert sortformer_children[0].data["agent_speaker"] == "spk0"
    assert sortformer_children[1].data["agent_speaker"] == "spk1"
    assert sortformer_children[0].data["agent_original_file"] == str(audio_path)
    assert sortformer_children[0]._metadata == {"source": "agent"}
    assert sortformer_children[0]._stage_perf == ["upstream"]


def test_agent_split_join_timestamp_extract_pipeline(tmp_path: Path) -> None:
    audio_path = _write_wav(tmp_path / "split.wav", duration_sec=1.0)
    task = AudioTask(
        dataset_name="agent",
        data={
            "resampled_audio_filepath": str(audio_path),
            "duration": 1.0,
            "audio_item_id": "split_item",
            "segments": [{"start": 0.0, "end": 0.4}],
            "original_file": str(audio_path),
            "start_ms": 0,
            "end_ms": 400,
        },
        _metadata={"segment_mappings": [{"concat_start_ms": 0, "concat_end_ms": 400, "original_file": str(audio_path), "original_start_ms": 0, "original_end_ms": 400}]},
        _stage_perf=["split-input"],
    )

    split = SplitLongAudioStage(suggested_max_len=10.0)
    task = split.process(task)
    assert task.data["split_filepaths"] == [str(audio_path)]
    task.data["split_metadata"][0]["text"] = "hello"
    task.data["split_metadata"][0]["alignment"] = [{"word": "hello", "start": 0.0, "end": 0.4}]

    join = JoinSplitAudioMetadataStage()
    task = join.process(task)
    assert task.data["text"] == "hello"
    assert task.data["alignment"][0]["word"] == "hello"

    timestamp = TimestampMapperStage(passthrough_keys=["text", "alignment"])
    task = timestamp.process(task)
    assert isinstance(task, AudioTask)
    assert task.data["original_file"] == str(audio_path)
    assert task.data["original_start_ms"] == 0
    assert task.data["original_end_ms"] == 400
    assert "resampled_audio_filepath" not in task.data

    extraction = SegmentExtractionStage(output_dir=str(tmp_path / "extract"), output_key="agent_extracted_path")
    extracted = extraction.process_batch([task])[0]
    # output_key holds the list of ALL written segment paths (a scalar was
    # last-write-wins for multi-interval entries)
    extracted_paths = extracted.data["agent_extracted_path"]
    assert isinstance(extracted_paths, list)
    assert extracted_paths
    assert all(os.path.exists(p) for p in extracted_paths)

    concat = SegmentConcatenationStage(silence_duration_sec=0.0)
    parent = AudioTask(
        dataset_name="agent",
        data={
            "segments": [
                {
                    "waveform": _waveform(duration_sec=0.2),
                    "sample_rate": 16000,
                    "start_ms": 0,
                    "end_ms": 200,
                    "segment_num": 0,
                    "original_file": str(audio_path),
                }
            ]
        },
        _metadata={"parent": True},
        _stage_perf=["concat-input"],
    )
    concatenated = concat.process(parent)
    assert concatenated._metadata["parent"] is True
    assert "segment_mappings" in concatenated._metadata
    assert concatenated._stage_perf == ["concat-input"]


def test_agent_alm_and_pretrain_pipeline(tmp_path: Path) -> None:
    audio_path = _write_wav(tmp_path / "alm.wav", duration_sec=1.0)
    alm_task = AudioTask(
        dataset_name="agent",
        data={
            "audio_filepath": str(audio_path),
            "audio_sample_rate": 16000,
            "swift_audio_filepath": "",
            "segments": _base_segments(),
        },
    )

    builder = ALMDataBuilderStage(target_window_duration=0.9, tolerance=0.2, min_speakers=1)
    alm_task = builder.process(alm_task)
    assert "windows" in alm_task.data

    overlap = ALMDataOverlapStage(windows_key="windows", filtered_windows_key="agent_filtered_windows")
    alm_task = overlap.process(alm_task)
    assert "agent_filtered_windows" in alm_task.data

    manifest = tmp_path / "long_form.jsonl"
    row = {"id": "row1", "audio_filepath": audio_path.name, "segments": _base_segments()}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    reader = ReadLongFormManifestStage(str(manifest), str(tmp_path))
    tasks = reader.process(EmptyTask())
    assert len(tasks) == 1
    pretrain_task = tasks[0]

    overlap_filter = OverlapFilterStage()
    pretrain_task = overlap_filter.process(pretrain_task)
    planner = SnippetCutPlannerStage(max_duration_sec=1.0, min_duration_sec=0.1)
    pretrain_task = planner.process(pretrain_task)
    assert pretrain_task.data["_snippet_plan"]

    rep_filter = SnippetRepetitionFilterStage(tokenizer_path=str(tmp_path))
    rep_filter._snippet_is_repetitive = MethodType(lambda self, text, snippet, task_id: False, rep_filter)
    pretrain_task = rep_filter.process(pretrain_task)

    extractor = SnippetExtractionStage(
        output_dir=str(tmp_path / "snippets"),
        output_audio_tar_path=str(tmp_path / "snippets.tar"),
        dry_run=True,
    )
    snippet_tasks = extractor.process(pretrain_task)
    assert snippet_tasks
    assert snippet_tasks[0].data["snippet_id"] is not None

    writer = SnippetManifestWriterStage(str(tmp_path / "snippets.jsonl"))
    writer.setup()
    writer.process(snippet_tasks[0])
    assert list(tmp_path.glob("snippets.jsonl.shard-*"))

    aggregator = PretrainMetricsAggregatorStage(str(tmp_path / "metrics.json"))
    aggregator.setup()
    aggregator.process(snippet_tasks[0])
    assert list(tmp_path.glob("metrics.json.shard-*"))


def test_agent_validate_input_uses_dict_keys_after_contract_fix(tmp_path: Path) -> None:
    audio_path = _write_wav(tmp_path / "validate.wav", duration_sec=1.0)
    task = AudioTask(
        data={
            "audio_filepath": str(audio_path),
            "duration": 1.0,
            "segments": [{"start": 0.0, "end": 0.5, "text": "hello", "text_ref": "hello"}],
        }
    )

    assert BandwidthEstimationStage().validate_input(task)
    # squim's default filepath key is "resampled_audio_filepath" (tagging
    # tutorial compat); point it at this task's plain audio_filepath
    assert TorchSquimQualityMetricsStage(audio_filepath_key="audio_filepath").validate_input(task)
    assert ComputeWERStage().validate_input(task)


def test_agent_contract_key_planner_chains_custom_quality_pipeline() -> None:
    stages: list[AgentReady] = [
        VADSegmentationStage(
            audio_filepath_key="agent_audio_path",
            waveform_key="agent_waveform",
            sample_rate_key="agent_sr",
            segments_key="agent_segments",
            nested=True,
            input_residency="waveform",
        ),
        UTMOSFilterStage(
            action="annotate",
            mode="segments",
            waveform_key="agent_waveform",
            sample_rate_key="agent_sr",
            segments_key="agent_segments",
            score_key="agent_utmos",
        ),
        SIGMOSFilterStage(
            action="annotate",
            mode="segments",
            waveform_key="agent_waveform",
            sample_rate_key="agent_sr",
            segments_key="agent_segments",
            noise_key="agent_noise",
            ovrl_key="agent_ovrl",
        ),
        BandFilterStage(
            action="annotate",
            mode="segments",
            waveform_key="agent_waveform",
            sample_rate_key="agent_sr",
            segments_key="agent_segments",
            prediction_key="agent_band",
        ),
        TorchSquimQualityMetricsStage(
            audio_filepath_key="agent_audio_path",
            segments_key="agent_segments",
            metrics_key="agent_metrics",
            resources=Resources(gpus=0.0),
        ),
        BandwidthEstimationStage(
            audio_filepath_key="agent_audio_path",
            segments_key="agent_segments",
            metrics_key="agent_metrics",
        ),
    ]

    available = {"agent_audio_path", "agent_waveform", "agent_sr"}
    for stage in stages:
        assert _contract_reads_satisfied(stage, available), f"agent cannot feed {type(stage).__name__}"
        _record_contract_writes(stage, available)

    assert {"agent_segments", "agent_utmos", "agent_noise", "agent_ovrl", "agent_band", "agent_metrics"}.issubset(
        available
    )


def test_agent_transform_residency_controls_and_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_curator.stages.audio.tagging import resample_audio as resample_module

    audio_path = _write_wav(tmp_path / "transform.wav", sample_rate=16000, duration_sec=0.4, channels=2)

    missing_waveform = MonoConversionStage(
        audio_filepath_key="agent_audio_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        input_residency="waveform",
    ).process(AudioTask(data={"agent_audio_path": str(audio_path)}))
    assert missing_waveform == []

    strict_mismatch = MonoConversionStage(
        audio_filepath_key="agent_audio_path",
        strict_sample_rate=True,
        output_sample_rate=48000,
    ).process(AudioTask(data={"agent_audio_path": str(audio_path)}))
    assert strict_mismatch == []

    with pytest.raises(ValueError, match="At least one"):
        ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "resample_bad"),
            keep_waveform_in_task=False,
            write_to_disk=False,
        )

    def write_valid_resample_output(cmd: list[str], **_: Any) -> SimpleNamespace:
        sf.write(cmd[-1], _waveform(duration_sec=0.4, channels=2).numpy().T, 16000, format="WAV", subtype="PCM_16")
        assert sf.info(cmd[-1]).frames > 0
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(resample_module.subprocess, "run", write_valid_resample_output)
    task = AudioTask(
        data={
            "agent_waveform": _waveform(duration_sec=0.4, channels=2),
            "agent_sr": 16000,
        }
    )
    resample = ResampleAudioStage(
        resampled_audio_dir=str(tmp_path / "resample"),
        audio_filepath_key="agent_audio_path",
        resampled_audio_filepath_key="agent_resampled_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        duration_key="agent_duration",
        audio_item_id_key="agent_audio_id",
        input_residency="waveform",
        keep_waveform_in_task=True,
        write_to_disk=False,
    )
    out = resample.process(task)

    assert "agent_resampled_path" not in out.data
    assert "agent_audio_path" not in out.data
    assert out.data["agent_waveform"].shape[0] == 2
    assert out.data["agent_sr"] == 16000
    assert out.data["agent_duration"] > 0
    assert out.data["agent_audio_id"]


def test_agent_quality_task_mode_annotate_filter_and_missing_input() -> None:
    task = AudioTask(
        data={
            "agent_waveform": _waveform(duration_sec=0.3, channels=2),
            "agent_sr": 16000,
        }
    )

    utmos = UTMOSFilterStage(
        mos_threshold=4.0,
        action="annotate",
        mode="task",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        score_key="agent_utmos",
    )
    utmos._model = _FakeUTMOSModel(2.0)
    task = utmos.process(task)
    assert isinstance(task, AudioTask)
    assert task.data["agent_utmos"] == 2.0

    sigmos = SIGMOSFilterStage(
        noise_threshold=4.0,
        ovrl_threshold=4.0,
        action="annotate",
        mode="task",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        noise_key="agent_noise",
        ovrl_key="agent_ovrl",
    )
    sigmos._model = _FakeSIGMOSModel(2.5)
    task = sigmos.process(task)
    assert isinstance(task, AudioTask)
    assert task.data["agent_noise"] == 2.5
    assert task.data["agent_ovrl"] == 2.5

    band = BandFilterStage(
        band_value="full_band",
        action="annotate",
        mode="task",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        prediction_key="agent_band",
    )
    band._predictor = _FakeBandPredictor("narrow_band")
    task = band.process(task)
    assert isinstance(task, AudioTask)
    assert task.data["agent_band"] == "narrow_band"

    pass_filter = UTMOSFilterStage(
        mos_threshold=4.0,
        action="filter",
        mode="task",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
    )
    pass_filter._model = _FakeUTMOSModel(4.5)
    assert isinstance(pass_filter.process(AudioTask(data={"agent_waveform": _waveform(), "agent_sr": 16000})), AudioTask)

    fail_filter = UTMOSFilterStage(
        mos_threshold=4.0,
        action="filter",
        mode="task",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
    )
    fail_filter._model = _FakeUTMOSModel(2.0)
    assert fail_filter.process(AudioTask(data={"agent_waveform": _waveform(), "agent_sr": 16000})) == []

    missing_sr = UTMOSFilterStage(
        action="annotate",
        mode="task",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        score_key="agent_utmos",
    )
    missing_sr._model = _FakeUTMOSModel(5.0)
    out = missing_sr.process(AudioTask(data={"agent_waveform": _waveform()}))
    assert isinstance(out, AudioTask)
    assert "agent_utmos" not in out.data


def test_agent_segment_filter_keeps_partial_survivors_with_segment_keys() -> None:
    task = AudioTask(
        data={
            "agent_segments": [
                {"id": 0, "agent_waveform": _waveform(duration_sec=0.2), "agent_sr": 16000},
                {"id": 1, "agent_waveform": _waveform(duration_sec=0.2), "agent_sr": 16000},
                {"id": 2, "agent_waveform": _waveform(duration_sec=0.2), "agent_sr": 16000},
            ]
        }
    )

    stage = UTMOSFilterStage(
        mos_threshold=4.0,
        action="filter",
        mode="segments",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        score_key="agent_utmos",
    )
    stage._model = _SequentialUTMOSModel([4.2, 2.0, 4.1])

    out = stage.process(task)
    assert isinstance(out, AudioTask)
    assert [segment["id"] for segment in out.data["agent_segments"]] == [0, 2]
    assert [segment["agent_utmos"] for segment in out.data["agent_segments"]] == pytest.approx([4.2, 4.1])


def test_agent_speaker_separation_fanout_feeds_downstream_filter() -> None:
    def fake_speaker_audio_data(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "spk0": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.25, diar_segments=[(0.0, 0.25)]),
            "spk1": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.30, diar_segments=[(0.25, 0.55)]),
        }

    stage = SpeakerSeparationStage(
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        speaker_id_key="agent_speaker",
        num_speakers_key="agent_num_speakers",
        duration_key="agent_duration",
        diar_segments_key="agent_diar_segments",
        min_duration=0.1,
        resources=Resources(gpus=0.0),
    )
    stage._separator = SimpleNamespace(get_speaker_audio_data=fake_speaker_audio_data)

    parent = AudioTask(
        dataset_name="agent",
        data={
            "agent_waveform": _waveform(duration_sec=0.6),
            "agent_sr": 16000,
            "agent_duration": 0.6,
            "text": "speaker parent",
        },
        _metadata={"trace": "kept"},
        _stage_perf=["fanout-input"],
    )
    speakers = stage.process(parent)

    assert len(speakers) == 2
    assert {speaker.data["agent_speaker"] for speaker in speakers} == {"spk0", "spk1"}
    for speaker in speakers:
        assert speaker.data["agent_num_speakers"] == 2
        assert speaker.data["agent_waveform"].shape[0] == 1
        assert speaker.data["agent_sr"] == 16000
        assert speaker.data["agent_duration"] >= 0.1
        assert speaker.data["agent_diar_segments"]
        assert speaker.data["text"] == "speaker parent"
        assert speaker._metadata == {"trace": "kept"}
        assert speaker._stage_perf == ["fanout-input"]

    band = BandFilterStage(
        band_value="full_band",
        action="annotate",
        mode="task",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        prediction_key="agent_band",
    )
    band._predictor = _FakeBandPredictor("full_band")

    annotated = [band.process(speaker) for speaker in speakers]
    assert all(isinstance(speaker, AudioTask) for speaker in annotated)
    assert all(speaker.data["agent_band"] == "full_band" for speaker in annotated if isinstance(speaker, AudioTask))


def test_agent_text_and_validation_noop_edges(tmp_path: Path) -> None:
    audio_path = _write_wav(tmp_path / "noop.wav")
    task = AudioTask(
        data={
            "agent_segments": [
                {"agent_text": "hello"},
                {"agent_text": ""},
                {"other_text": "ignored"},
            ]
        }
    )

    itn = InverseTextNormalizationStage(text_key="agent_text", segments_key="agent_segments")
    itn._normalizer = _FakeNormalizer()
    task = itn.process(task)
    assert task.data["agent_segments"][0]["agent_text_ITN"] == "hello"
    assert "agent_text_ITN" not in task.data["agent_segments"][1]
    assert "agent_text_ITN" not in task.data["agent_segments"][2]

    class _FailingConverter:
        def convert(self, _text: str) -> str:
            msg = "conversion failed"
            raise RuntimeError(msg)

    chinese = ChineseConversionStage(text_key="agent_text", segments_key="agent_segments")
    chinese._converter = _FailingConverter()
    task = chinese.process(task)
    assert task.data["agent_segments"][0]["agent_text_simplified"] == "hello"

    assert not BandwidthEstimationStage(audio_filepath_key="missing_path").validate_input(AudioTask(data={}))
    assert not TorchSquimQualityMetricsStage(audio_filepath_key="missing_path").validate_input(AudioTask(data={}))
    assert not ComputeWERStage(segments_key="missing_segments").validate_input(AudioTask(data={"agent_audio_path": str(audio_path)}))

    preserve = PreserveByValueStage("keep", True)
    kept = preserve.process_batch([AudioTask(data={"id": 1, "keep": True}), AudioTask(data={"id": 2, "keep": False})])
    assert [task.data["id"] for task in kept] == [1]
    with pytest.raises(ValueError, match="failed validation"):
        preserve.process_batch([AudioTask(data={"id": 3})])


def test_agent_planner_detects_missing_keys_and_collisions() -> None:
    """A simulated planner must refuse unsatisfiable chains and spot key collisions.

    The existing pipeline tests only prove that *valid* chains work. This guards the
    other half of the agent's job: rejecting incompatible wiring before runtime.
    """
    # 1. A consolidator that needs a nested segments list is unsatisfiable before a
    #    producer declares it.
    available: set[str] = {"audio_filepath", "waveform", "sample_rate"}
    concat = SegmentConcatenationStage(segments_key="segments")
    assert _contract_reads_satisfied(concat, available) is False

    nested_vad = VADSegmentationStage(nested=True, segments_key="segments")
    _record_contract_writes(nested_vad, available)
    assert "segments" in available
    assert _contract_reads_satisfied(concat, available) is True

    # 2. Cardinality intent: a fan-out producer and a fan-in consolidator declare
    #    distinct cardinalities, so a planner can tell that fan-out output is NOT a
    #    nested list it can consolidate.
    assert VADSegmentationStage(nested=False, segments_key="segments").describe().cardinality == "1:N fan-out"
    assert nested_vad.describe().cardinality == "1:1 nested-list"
    assert concat.describe().cardinality == "N:1"

    # 3. Key collision: two scorers writing the same output key collide; distinct
    #    keys do not. This is exactly the A/B scoring case full key authority enables.
    produced: set[str] = {"waveform", "sample_rate"}
    _record_contract_writes(UTMOSFilterStage(score_key="utmos_mos"), produced)
    same_key = UTMOSFilterStage(score_key="utmos_mos")
    assert set(same_key.describe().writes.data_keys) & produced, "duplicate score key should collide"
    distinct_key = UTMOSFilterStage(score_key="utmos_mos_model_b")
    assert not (set(distinct_key.describe().writes.data_keys) & produced), "renamed score key must not collide"


def test_agent_fanout_children_have_isolated_metadata() -> None:
    """Fan-out children must own independent _metadata / _stage_perf copies.

    Pins the de-aliasing fix behaviorally: mutating one child must not leak into a
    sibling or the parent (the shared-reference bug class).
    """

    def fake_speaker_audio_data(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "spk0": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.25, diar_segments=[(0.0, 0.25)]),
            "spk1": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.30, diar_segments=[(0.25, 0.55)]),
        }

    stage = SpeakerSeparationStage(
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        min_duration=0.1,
        resources=Resources(gpus=0.0),
    )
    stage._separator = SimpleNamespace(get_speaker_audio_data=fake_speaker_audio_data)

    parent = AudioTask(
        dataset_name="agent",
        data={"agent_waveform": _waveform(duration_sec=0.6), "agent_sr": 16000},
        _metadata={"trace": "kept"},
        _stage_perf=["fanout-input"],
    )
    children = stage.process(parent)
    assert len(children) == 2

    children[0]._metadata["mutated"] = True
    children[0]._stage_perf.append("child0-only")

    assert "mutated" not in children[1]._metadata
    assert "mutated" not in parent._metadata
    assert "child0-only" not in children[1]._stage_perf
    assert "child0-only" not in parent._stage_perf
    assert children[1]._metadata["trace"] == "kept"


def test_agent_mono_output_gating_and_auto_residency(tmp_path: Path) -> None:
    """Output-destination flags gate their keys, and ``auto`` residency prefers the tensor.

    Verifies the §3.4 invariant (a disabled destination's key is absent, not stale) and
    that ``auto`` residency does not touch disk when a waveform is already present.
    """
    # auto residency: waveform present, audio_filepath points at a non-existent file ->
    # the stage must use the in-memory waveform and never read disk.
    waveform_only = AudioTask(
        dataset_name="agent",
        data={
            "agent_audio_path": str(tmp_path / "does_not_exist.wav"),
            "agent_waveform": _waveform(duration_sec=0.5, channels=2),
            "agent_sr": 16000,
        },
    )
    mono_mem = MonoConversionStage(
        audio_filepath_key="agent_audio_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        output_audio_filepath_key="agent_mono_path",
        output_sample_rate=16000,
        input_residency="auto",
        keep_waveform_in_task=True,
        write_to_disk=False,
    )
    out = mono_mem.process(waveform_only)
    assert isinstance(out, AudioTask)
    assert "agent_waveform" in out.data, "tensor must be kept when keep_waveform_in_task=True"
    assert "agent_mono_path" not in out.data, "disk-path key must be absent when write_to_disk=False"

    # disk-only: keep_waveform_in_task=False -> the waveform key must be omitted.
    src = _write_wav(tmp_path / "mono_src.wav", channels=1)
    disk_task = AudioTask(dataset_name="agent", data={"agent_audio_path": str(src), "agent_sr": 16000})
    mono_disk = MonoConversionStage(
        audio_filepath_key="agent_audio_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        output_audio_filepath_key="agent_mono_path",
        output_sample_rate=16000,
        input_residency="file",
        keep_waveform_in_task=False,
        write_to_disk=True,
        output_dir=str(tmp_path / "mono_out"),
    )
    out_disk = mono_disk.process(disk_task)
    assert "agent_mono_path" in out_disk.data, "disk-path key must be present when write_to_disk=True"
    assert "agent_waveform" not in out_disk.data, "tensor must be omitted when keep_waveform_in_task=False"


def test_agent_metadata_survives_multi_stage_pipeline() -> None:
    """Seeded _metadata / _stage_perf must survive a full multi-stage pipeline.

    The per-stage SegmentConcat regression checks one stage; this checks the systemic
    invariant across VAD (nested) -> quality annotate.
    """
    task = AudioTask(
        dataset_name="agent",
        data={"agent_waveform": _waveform(duration_sec=1.0), "agent_sr": 16000},
        _metadata={"trace_id": "abc", "quality": {"keep": True}},
        _stage_perf=["ingress"],
    )

    vad = VADSegmentationStage(
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        nested=True,
        input_residency="waveform",
    )
    vad._vad_model = object()
    vad._get_vad_segments = MethodType(
        lambda self, waveform, sample_rate: [{"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0}],
        vad,
    )
    task = vad.process(task)
    for segment in task.data["agent_segments"]:
        segment["text"] = "speech"

    utmos = UTMOSFilterStage(
        mos_threshold=1.0,
        action="annotate",
        mode="segments",
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        segments_key="agent_segments",
        score_key="agent_utmos",
    )
    utmos._model = _FakeUTMOSModel(4.5)
    task = utmos.process(task)

    assert task._metadata["trace_id"] == "abc"
    assert task._metadata["quality"] == {"keep": True}
    assert "ingress" in task._stage_perf


def test_agent_squim_metrics_map_to_correct_segments() -> None:
    """SQUIM sorts waveforms by length for batching, then maps results back.

    A transposition in that map would pass every "key present" check but be silently
    wrong, so we encode each waveform's length in its fake score and assert each
    segment receives its OWN score after the sort-then-map round trip.
    """
    segments = [{"start": 0, "end": 1, "text": "a"}, {"start": 1, "end": 2, "text": "b"}, {"start": 2, "end": 3, "text": "c"}]
    task = AudioTask(dataset_name="agent", data={"agent_segments": segments, "agent_audio_path": "ignored"})

    squim = TorchSquimQualityMetricsStage(
        audio_filepath_key="agent_audio_path",
        segments_key="agent_segments",
        metrics_key="agent_metrics",
        resources=Resources(gpus=0.0),
    )
    # Control collection so the test exercises only the sort-then-map logic: three
    # waveforms of distinct lengths, mapped to segment indices 0, 1, 2.
    lengths = {0: 30, 1: 10, 2: 20}
    squim._collect_waveforms_for_entry = MethodType(
        lambda self, task_idx, data_entry: [(task_idx, idx, torch.ones(lengths[idx])) for idx in range(3)],
        squim,
    )
    # Encode each waveform's length in its score so a swap is detectable.
    squim._compute_metrics_batched = MethodType(
        lambda self, waveforms: [(float(w.shape[-1]), 0.0, 0.0) for w in waveforms],
        squim,
    )

    out = squim.process_batch([task])[0]
    result_segments = out.data["agent_segments"]
    assert result_segments[0]["agent_metrics"]["pesq_squim"] == 30
    assert result_segments[1]["agent_metrics"]["pesq_squim"] == 10
    assert result_segments[2]["agent_metrics"]["pesq_squim"] == 20


def test_agent_filepath_protocol_preserves_first_original_through_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A derivative chain keeps the FIRST original in original_audio_filepath.

    Mono(write+update) -> Resample(write+update): audio_filepath must point at the
    latest derivative while original_audio_filepath must still hold the manifest source
    (the 'first overwrite only' semantic), proving the canonical-key protocol composes.
    """
    from nemo_curator.stages.audio.tagging import resample_audio as resample_module

    source = _write_wav(tmp_path / "source.wav", sample_rate=16000, channels=1)
    task = AudioTask(dataset_name="agent", data={"audio_filepath": str(source), "audio_sample_rate": 16000})

    mono = MonoConversionStage(
        output_sample_rate=16000,
        input_residency="file",
        keep_waveform_in_task=True,
        write_to_disk=True,
        update_audio_filepath=True,
        output_dir=str(tmp_path / "mono"),
    )
    task = mono.process(task)
    assert task.data["original_audio_filepath"] == str(source)
    mono_path = task.data["audio_filepath"]
    assert mono_path != str(source)

    monkeypatch.setattr(resample_module.subprocess, "run", _copy_file_for_fake_ffmpeg)
    resample = ResampleAudioStage(
        resampled_audio_dir=str(tmp_path / "resampled"),
        input_residency="file",
        keep_waveform_in_task=False,
        write_to_disk=True,
        update_audio_filepath=True,
    )
    task = resample.process(task)

    # audio_filepath advanced to the resampled derivative...
    assert task.data["audio_filepath"] == task.data["resampled_audio_filepath"]
    assert task.data["audio_filepath"] != mono_path
    # ...but the very first source is still preserved (not overwritten by the 2nd update).
    assert task.data["original_audio_filepath"] == str(source)


def test_agent_vad_empty_and_degenerate_cardinality() -> None:
    """Zero-detection cases must not crash downstream wiring.

    Nested VAD yields an empty segment list (1:1); fan-out VAD yields no child tasks;
    a segment-mode filter that drops everything returns no task.
    """
    data = {"waveform": _waveform(duration_sec=0.5), "sample_rate": 16000}

    nested = VADSegmentationStage(nested=True, input_residency="waveform")
    nested._vad_model = object()
    nested._get_vad_segments = MethodType(lambda self, waveform, sample_rate: [], nested)
    nested_task = nested.process(AudioTask(dataset_name="agent", data=dict(data)))
    assert isinstance(nested_task, AudioTask)
    assert nested_task.data["segments"] == []

    fanout = VADSegmentationStage(nested=False, input_residency="waveform")
    fanout._vad_model = object()
    fanout._get_vad_segments = MethodType(lambda self, waveform, sample_rate: [], fanout)
    assert fanout.process(AudioTask(dataset_name="agent", data=dict(data))) == []

    # A segment-mode filter whose every segment fails drops the whole task.
    seg_task = AudioTask(
        dataset_name="agent",
        data={
            "waveform": _waveform(duration_sec=0.5),
            "sample_rate": 16000,
            "segments": [
                {"start": 0.0, "end": 0.25, "waveform": _waveform(duration_sec=0.25), "sample_rate": 16000},
                {"start": 0.25, "end": 0.5, "waveform": _waveform(duration_sec=0.25), "sample_rate": 16000},
            ],
        },
    )
    utmos = UTMOSFilterStage(mos_threshold=4.0, action="filter", mode="segments")
    utmos._model = _FakeUTMOSModel(1.0)  # below threshold -> every segment dropped
    assert utmos.process(seg_task) == []


def test_agent_vad_nested_and_fanout_produce_equivalent_segments() -> None:
    """Nested and fan-out VAD must agree on segment boundaries, only on packaging."""
    timestamps = [{"start": 0.0, "end": 0.4}, {"start": 0.4, "end": 0.9}]
    data = {"waveform": _waveform(duration_sec=1.0), "sample_rate": 16000, "audio_filepath": "src.wav"}

    nested = VADSegmentationStage(nested=True, input_residency="waveform")
    nested._vad_model = object()
    nested._get_vad_segments = MethodType(lambda self, waveform, sample_rate: list(timestamps), nested)
    nested_task = nested.process(AudioTask(dataset_name="agent", data=dict(data)))
    nested_segments = nested_task.data["segments"]

    fanout = VADSegmentationStage(nested=False, input_residency="waveform")
    fanout._vad_model = object()
    fanout._get_vad_segments = MethodType(lambda self, waveform, sample_rate: list(timestamps), fanout)
    children = fanout.process(AudioTask(dataset_name="agent", data=dict(data)))

    assert len(nested_segments) == len(children) == 2
    nested_bounds = [(seg["start_ms"], seg["end_ms"]) for seg in nested_segments]
    fanout_bounds = [(child.data["start_ms"], child.data["end_ms"]) for child in children]
    assert nested_bounds == fanout_bounds == [(0, 400), (400, 900)]


def test_agent_audio_to_document_never_leaks_non_serializable() -> None:
    """The JSONL/document boundary must never carry tensors or, by default, segments."""
    task = AudioTask(
        dataset_name="agent",
        data={
            "audio_filepath": "src.wav",
            "text": "hello world",
            "waveform": _waveform(duration_sec=0.5),  # tensor must be stripped
            "segments": [{"start": 0.0, "end": 0.5, "text": "hello"}],  # stripped by default
        },
    )

    default_docs = AudioToDocumentStage().process_batch([task])
    row = default_docs[0].to_pandas().iloc[0].to_dict()
    assert "waveform" not in row
    assert "segments" not in row
    assert row["text"] == "hello world"
    json.dumps(row)  # must be JSON-serializable (would raise on a tensor leak)

    keep_segments = AudioToDocumentStage(serialize_segments=True).process_batch([task])
    seg_row = keep_segments[0].to_pandas().iloc[0].to_dict()
    assert "waveform" not in seg_row  # tensor still stripped
    assert seg_row["segments"][0]["text"] == "hello"  # segments retained on request


def test_agent_resample_skips_existing_output_only_when_writing_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running resample must be idempotent on disk but never wrongly skip otherwise."""
    from nemo_curator.stages.audio.tagging import resample_audio as resample_module

    calls = {"n": 0}

    def counting_ffmpeg(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        return _copy_file_for_fake_ffmpeg(cmd, **kwargs)

    monkeypatch.setattr(resample_module.subprocess, "run", counting_ffmpeg)
    source = _write_wav(tmp_path / "src.wav", sample_rate=16000, channels=1)

    disk_stage = ResampleAudioStage(
        resampled_audio_dir=str(tmp_path / "out"),
        input_residency="file",
        write_to_disk=True,
        keep_waveform_in_task=False,
    )

    def disk_task() -> AudioTask:
        return AudioTask(dataset_name="agent", data={"audio_filepath": str(source), "audio_item_id": "fixed_id"})

    disk_stage.process(disk_task())
    assert calls["n"] == 1
    disk_stage.process(disk_task())  # same output path already exists -> skipped
    assert calls["n"] == 1

    # write_to_disk=False uses a fresh temp output each run, so it must always convert.
    calls["n"] = 0
    mem_stage = ResampleAudioStage(
        resampled_audio_dir=str(tmp_path / "unused"),
        input_residency="file",
        write_to_disk=False,
        keep_waveform_in_task=True,
    )
    mem_stage.process(AudioTask(dataset_name="agent", data={"audio_filepath": str(source), "audio_item_id": "m"}))
    mem_stage.process(AudioTask(dataset_name="agent", data={"audio_filepath": str(source), "audio_item_id": "m"}))
    assert calls["n"] == 2


def test_agent_manifest_writer_truncates_on_setup(tmp_path: Path) -> None:
    """A fresh run (setup) truncates the output so reruns do not accumulate duplicates."""
    out_path = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(output_path=str(out_path))
    task = AudioTask(dataset_name="agent", data={"audio_filepath": "src.wav", "text": "row"})

    writer.setup()
    writer.process(task)
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 2  # appends within a run

    writer.setup()  # new run truncates
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_agent_or_shaped_stages_accept_each_declared_input_shape(tmp_path: Path) -> None:
    """Each reads_one_of shape an OR-shaped stage declares must pass validate_input."""
    audio_path = _write_wav(tmp_path / "or.wav", sample_rate=16000)

    squim = TorchSquimQualityMetricsStage(audio_filepath_key="agent_audio_path", segments_key="agent_segments")
    # both declared read shapes include the filepath (SQUIM always loads the
    # audio file; segments alone are NOT a sufficient shape)
    assert not squim.validate_input(AudioTask(data={"agent_segments": [{"start": 0, "end": 1}]}))
    assert squim.validate_input(
        AudioTask(data={"agent_audio_path": str(audio_path), "agent_segments": [{"start": 0, "end": 1}]})
    )
    assert squim.validate_input(AudioTask(data={"agent_audio_path": str(audio_path)}))
    assert not squim.validate_input(AudioTask(data={}))

    bandwidth = BandwidthEstimationStage(audio_filepath_key="agent_audio_path")
    assert bandwidth.validate_input(AudioTask(data={"agent_audio_path": str(audio_path), "segments": [{"start": 0, "end": 1}]}))
    assert bandwidth.validate_input(AudioTask(data={"agent_audio_path": str(audio_path), "duration": 1.0}))
    assert not bandwidth.validate_input(AudioTask(data={"duration": 1.0}))


def test_agent_contract_self_consistency_meta_invariants(tmp_path: Path) -> None:
    """Cross-stage contract invariants every AgentReady stage must satisfy.

    A meta-test over all 44 contracts: declaring a disk output implies the disk gate,
    declaring a tensor output implies at least one declared write key, and every
    reads_one_of alternative is a non-empty key set.
    """
    cases = _coverage_cases(tmp_path)
    assert len(cases) == EXPECTED_AUDIO_AGENT_READY_STAGE_COUNT

    for case in cases:
        contract = case.factory().describe()
        name = case.cls.__name__

        if "disk" in contract.writes.produces:
            assert contract.gates.writes_to_disk, f"{name} produces disk but gates.writes_to_disk is False"
        if "tensor" in contract.writes.produces:
            assert contract.writes.data_keys or contract.writes.segment_data_keys, (
                f"{name} produces a tensor but declares no write keys"
            )
        for index, option in enumerate(contract.reads_one_of):
            assert option.data_keys or option.segment_data_keys, (
                f"{name}.reads_one_of[{index}] is an empty key set"
            )
