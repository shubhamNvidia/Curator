"""One-shot card backfill for the 28 existing audio stages.

Run once during Phase 1 implementation; the cards are then version-controlled
and edited by hand. Generated cards land under
``Curator/nemo_curator/stages/audio/_cards/<StageName>/stage_card.yaml``.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path


CARDS_ROOT = Path(__file__).resolve().parents[1] / "nemo_curator" / "stages" / "audio" / "_cards"


def W(name: str, body: str) -> None:
    d = CARDS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "stage_card.yaml").write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    print(f"wrote {d / 'stage_card.yaml'}")


# ---------------------------------------------------------------------------
# common.py: 5 cards
# ---------------------------------------------------------------------------

W("GetAudioDurationStage", """
schema_version: '1.0'
name: GetAudioDurationStage
target: nemo_curator.stages.audio.common.GetAudioDurationStage
version: 1.0.0
public_api: true
description: |
  Compute audio duration in seconds from the file at audio_filepath_key and
  store the result under duration_key. Uses soundfile.info for a metadata-only
  read (does not decode samples).
summary: Read audio duration via soundfile and stamp it into task.data.
category: source
capabilities: [get_duration]
also_handles: []
modality_in: audio
modality_out: audio
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [duration]
produces_cardinality: '1:1'
requires_on_disk_path: true
requires_file_extensions: [.wav, .flac, .ogg, .mp3]
produces_keys_after_run: [duration]
drops_keys_after_run: []
params:
  - {name: audio_filepath_key, type: str, default: audio_filepath}
  - {name: duration_key, type: str, default: duration}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_cheap
tags: [duration, metadata, cpu]
source_file: nemo_curator/stages/audio/common.py
smoke_test_status: unknown
""")


W("PreserveByValueStage", """
schema_version: '1.0'
name: PreserveByValueStage
target: nemo_curator.stages.audio.common.PreserveByValueStage
version: 1.0.0
public_api: true
description: |
  Generic key/value filter. Drops tasks whose task.data[input_value_key]
  fails the comparison `operator` against target_value. Operator is one of
  {lt, le, eq, ne, ge, gt}. Returns None from process(); all work happens in
  process_batch().
summary: Drop tasks that don't satisfy a simple key-vs-value comparison.
category: filter
capabilities: [value_filter]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: []
produces_cardinality: '1:0|1'
produces_keys_after_run: []
drops_keys_after_run: []
params:
  - {name: input_value_key, type: str, required: true, description: "task.data key to compare."}
  - {name: target_value, type: str, required: true, description: "Threshold value (int or str)."}
  - {name: operator, type: enum, default: eq, choices: [lt, le, eq, ne, ge, gt]}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: true
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_cheap
tags: [filter, cpu, generic]
source_file: nemo_curator/stages/audio/common.py
smoke_test_status: unknown
""")


W("ManifestReaderStage", """
schema_version: '1.0'
name: ManifestReaderStage
target: nemo_curator.stages.audio.common.ManifestReaderStage
version: 1.0.0
public_api: true
description: |
  Reads each JSONL line from a FileGroupTask and emits one AudioTask per line.
  Streams via fsspec; supports local + cloud (s3, gs) paths. Typically used as
  the second half of the ManifestReader composite.
summary: Stream JSONL manifests line-by-line into AudioTasks.
category: source
capabilities: [read_manifest]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: [audio_filepath]
produces_cardinality: '1:N'
params: []
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_cheap
tags: [reader, manifest, jsonl]
source_file: nemo_curator/stages/audio/common.py
smoke_test_status: unknown
""")


W("ManifestReader", """
schema_version: '1.0'
name: ManifestReader
target: nemo_curator.stages.audio.common.ManifestReader
version: 1.0.0
public_api: true
description: |
  Composite reader: FilePartitioningStage discovers JSONL manifest files
  and groups them, then ManifestReaderStage emits one AudioTask per line.
summary: High-level JSONL manifest source.
category: composite
capabilities: [read_manifest]
also_handles: []
inputs: {top_level: [], data: []}
outputs: {top_level: [], data: [audio_filepath]}
produces_cardinality: '1:N'
params:
  - {name: manifest_path, type: path, required: true}
  - {name: files_per_partition, type: int, default: 1, min: 1, max: 100000}
  - {name: blocksize, type: str, default: null}
  - {name: file_extensions, type: list, default: [.jsonl, .json]}
  - {name: storage_options, type: dict, default: null}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_cheap
tags: [reader, manifest, composite]
source_file: nemo_curator/stages/audio/common.py
smoke_test_status: unknown
""")


W("ManifestWriterStage", """
schema_version: '1.0'
name: ManifestWriterStage
target: nemo_curator.stages.audio.common.ManifestWriterStage
version: 1.0.0
public_api: true
description: |
  Append each AudioTask's task.data dict to an output JSONL manifest. Truncates
  on the driver in setup() so reruns produce a clean file; per-node setup_on_node
  only creates the directory.
summary: JSONL manifest sink.
category: sink
capabilities: [write_manifest]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: []
produces_cardinality: '1:1'
produces_on_disk_files: true
params:
  - {name: output_path, type: path, required: true}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_cheap
tags: [writer, manifest, jsonl, sink]
source_file: nemo_curator/stages/audio/common.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# io/
# ---------------------------------------------------------------------------

W("SegmentExtractionStage", """
schema_version: '1.0'
name: SegmentExtractionStage
target: nemo_curator.stages.audio.io.extract_segments.SegmentExtractionStage
version: 1.0.0
public_api: true
description: |
  Slice the original audio file based on offset/duration in each task's
  manifest entry and write the slice as a standalone segment file (wav/flac/ogg)
  to output_dir. Operates batched; one source file is opened once per batch.
summary: Write per-segment audio files from manifest entries.
category: io
capabilities: [segment_extract]
also_handles: []
inputs:
  top_level: []
  data: [original_file, original_start_ms, original_end_ms]
outputs:
  top_level: []
  data: [audio_filepath]
produces_cardinality: '1:1'
produces_on_disk_files: true
params:
  - {name: output_dir, type: path, required: true}
  - {name: output_format, type: enum, default: wav, choices: [wav, flac, ogg]}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 64
supports_process_batch: true
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [io, extract, segment, wav]
source_file: nemo_curator/stages/audio/io/extract_segments.py
smoke_test_status: unknown
""")


W("AudioToDocumentStage", """
schema_version: '1.0'
name: AudioToDocumentStage
target: nemo_curator.stages.audio.io.convert.AudioToDocumentStage
version: 1.0.0
public_api: true
description: |
  Aggregate a batch of AudioTask entries into a single DocumentBatch (Pandas-backed).
  Strips non-serializable keys (raw tensors, waveforms) defensively before serializing.
  The only existing many-to-one stage in the catalog.
summary: Collapse a batch of AudioTasks into one DocumentBatch.
category: io
capabilities: [audio_to_document]
also_handles: []
modality_in: audio
modality_out: text
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: []
produces_cardinality: 'N:1'
params: []
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 64
supports_process_batch: true
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [bridge, audio_to_text]
source_file: nemo_curator/stages/audio/io/convert.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# preprocessing/
# ---------------------------------------------------------------------------

W("MonoConversionStage", """
schema_version: '1.0'
name: MonoConversionStage
target: nemo_curator.stages.audio.preprocessing.mono_conversion.MonoConversionStage
version: 1.0.0
public_api: true
description: |
  Load the file referenced by audio_filepath_key, downmix to mono, validate
  sample rate against output_sample_rate. In strict mode (default) the stage
  drops files whose native SR differs from output_sample_rate (returns [] from
  process()). Carries the mono waveform forward in task.data['waveform'].
summary: Force mono and verify sample rate; load waveform into memory.
category: preprocess
capabilities: [normalize_mono, normalize_sample_rate]
also_handles: []
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [waveform, sample_rate]
produces_cardinality: '1:0|1'
requires_sample_rate: null
produces_in_memory_waveform: true
params:
  - {name: output_sample_rate, type: int, default: 48000, min: 8000, max: 96000}
  - {name: audio_filepath_key, type: str, default: audio_filepath}
  - {name: strict_sample_rate, type: bool, default: true}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [mono, sample_rate, in_memory]
source_file: nemo_curator/stages/audio/preprocessing/mono_conversion.py
smoke_test_status: unknown
""")


W("SegmentConcatenationStage", """
schema_version: '1.0'
name: SegmentConcatenationStage
target: nemo_curator.stages.audio.preprocessing.concatenation.SegmentConcatenationStage
version: 1.0.0
public_api: true
description: |
  Consume a single AudioTask whose task.data['segments'] holds a list of
  VAD-emitted nested segments. Concatenate them in order with silence_duration_sec
  of silence between each, and emit ONE AudioTask carrying the combined waveform
  plus segment_mappings in task._metadata.
summary: Concat nested VAD segments back into a single waveform with silence gaps.
category: preprocess
capabilities: [segment_concat]
also_handles: []
inputs:
  top_level: []
  data: [segments]
outputs:
  top_level: []
  data: [waveform, sample_rate, num_segments, total_duration_sec, original_file]
produces_cardinality: '1:1'
produces_in_memory_waveform: true
params:
  - {name: silence_duration_sec, type: float, default: 0.5, min: 0.0, max: 5.0}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [concat, vad, in_memory]
source_file: nemo_curator/stages/audio/preprocessing/concatenation.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# segmentation/
# ---------------------------------------------------------------------------

W("VADSegmentationStage", """
schema_version: '1.0'
name: VADSegmentationStage
target: nemo_curator.stages.audio.segmentation.vad_segmentation.VADSegmentationStage
version: 1.0.0
public_api: true
description: |
  Silero VAD-based segmentation. In default (non-nested) mode emits one AudioTask
  per detected speech segment (1:N fan-out). In nested mode emits one task whose
  task.data['segments'] holds the list. Honors min_duration_sec/max_duration_sec,
  so the planner can drop a downstream duration filter when this stage is in the IR.
summary: Voice-activity segmentation with built-in min/max duration gates.
category: segmentation
capabilities: [vad]
also_handles: [duration_filter, silence_removal]
inputs:
  top_level: []
  data: [waveform, sample_rate]
outputs:
  top_level: []
  data: [waveform, sample_rate, start_ms, end_ms, segment_num, duration]
produces_cardinality: '1:N'
nested_segment_key: segments
requires_in_memory_waveform: true
params:
  - {name: min_interval_ms, type: int, default: 500, min: 0, max: 60000}
  - {name: min_duration_sec, type: float, default: 2.0, min: 0.0, max: 3600.0}
  - {name: max_duration_sec, type: float, default: 60.0, min: 0.1, max: 3600.0}
  - {name: threshold, type: float, default: 0.5, min: 0.0, max: 1.0}
  - {name: speech_pad_ms, type: int, default: 300, min: 0, max: 5000}
  - {name: waveform_key, type: str, default: waveform}
  - {name: sample_rate_key, type: str, default: sample_rate}
  - {name: nested, type: bool, default: false, description: "If true, output one task with task.data['segments'] instead of fan-out."}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
models:
  - {name: silero-vad, provider: torch_hub, license: MIT}
license: MIT
commercial_safe: true
cost_hint: cheap
quality_hint: high
tags: [vad, silero, segmentation, fanout]
source_file: nemo_curator/stages/audio/segmentation/vad_segmentation.py
smoke_test_status: unknown
""")


W("SpeakerSeparationStage", """
schema_version: '1.0'
name: SpeakerSeparationStage
target: nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage
version: 1.0.0
public_api: true
description: |
  Use the NVIDIA Sortformer diarizer to identify per-speaker activity and
  fan out one AudioTask per speaker carrying only that speaker's segments.
  Optionally excludes overlapping regions and enforces min_duration / gap_threshold
  cleanup between adjacent same-speaker chunks.
summary: Split a multi-speaker recording into one task per speaker.
category: segmentation
capabilities: [speaker_separation, speaker_diarization]
also_handles: []
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [audio_filepath, speaker_id, start_ms, end_ms]
produces_cardinality: '1:N'
requires_in_memory_waveform: false
params:
  - {name: model_path, type: str, default: 'nvidia/diar_sortformer_4spk-v1'}
  - {name: exclude_overlaps, type: bool, default: true}
  - {name: min_duration, type: float, default: 0.8, min: 0.0, max: 60.0}
  - {name: gap_threshold, type: float, default: 0.1, min: 0.0, max: 5.0}
  - {name: buffer_time, type: float, default: 0.5, min: 0.0, max: 5.0}
resources: {cpus: 1.0, gpus: 1.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
incompatible_with_inference_server: false
models:
  - {name: diar_sortformer_4spk-v1, provider: huggingface, repo_id: 'nvidia/diar_sortformer_4spk-v1', license: CC-BY-4.0}
license: Apache-2.0
commercial_safe: true
cost_hint: expensive
quality_hint: high
tags: [speaker_separation, sortformer, gpu, fanout]
source_file: nemo_curator/stages/audio/segmentation/speaker_separation.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# filtering/
# ---------------------------------------------------------------------------

W("BandFilterStage", """
schema_version: '1.0'
name: BandFilterStage
target: nemo_curator.stages.audio.filtering.band.BandFilterStage
version: 1.0.0
public_api: true
description: |
  Classify each clip as 'full_band' or 'narrow_band' using the
  nvidia/nemocurator-speech-bandwidth-filter joblib model and drop clips that
  don't match band_value.
summary: Keep only clips matching the chosen bandwidth class.
category: filter
capabilities: [quality_filter_band]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: [bandwidth_class]
produces_cardinality: '1:0|1'
requires_in_memory_waveform: true
params:
  - {name: model_path, type: str, default: null, description: "Optional local .joblib path."}
  - {name: cache_dir, type: path, default: null}
  - {name: band_value, type: enum, default: full_band, choices: [full_band, narrow_band]}
resources: {cpus: 4.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
models:
  - {name: nemocurator-speech-bandwidth-filter, provider: huggingface, repo_id: 'nvidia/nemocurator-speech-bandwidth-filter', license: Apache-2.0}
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
quality_hint: medium
tags: [bandwidth, filter, classifier, cpu]
source_file: nemo_curator/stages/audio/filtering/band.py
smoke_test_status: unknown
""")


W("UTMOSFilterStage", """
schema_version: '1.0'
name: UTMOSFilterStage
target: nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage
version: 1.0.0
public_api: true
description: |
  Run the utmos22_strong MOS estimator (torch.hub tarepan/SpeechMOS) on each
  clip and drop clips whose predicted MOS is below mos_threshold. Audio is
  resampled to 16 kHz internally.
summary: Drop clips whose UTMOS score is below a floor.
category: filter
capabilities: [quality_filter_mos]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: [utmos_mos]
produces_cardinality: '1:0|1'
requires_sample_rate: null
requires_in_memory_waveform: true
params:
  - {name: mos_threshold, type: float, default: 3.5, min: 1.0, max: 5.0}
  - {name: sample_rate, type: int, default: 16000, min: 16000, max: 16000, description: "Internal resample target; not user-tunable in practice."}
resources: {cpus: 1.0, gpus: 0.5, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
models:
  - {name: utmos22_strong, provider: torch_hub, license: MIT}
license: Apache-2.0
commercial_safe: true
cost_hint: medium
quality_hint: high
tags: [mos, utmos, filter, gpu]
source_file: nemo_curator/stages/audio/filtering/utmos.py
smoke_test_status: unknown
""")


W("SIGMOSFilterStage", """
schema_version: '1.0'
name: SIGMOSFilterStage
target: nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage
version: 1.0.0
public_api: true
description: |
  Microsoft SIG-Challenge MOS estimator (ONNX). Outputs seven quality
  sub-scores (noise / ovrl / sig / col / disc / loud / reverb) and drops
  clips that fall below any of the *_threshold floors. Model auto-downloads
  on first use and caches under ~/.cache/nemo_curator/sigmos_model/.
summary: Multi-axis MOS filter (noise / overall / signal / loudness / reverb / ...).
category: filter
capabilities: [quality_filter_mos]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: [sigmos_noise, sigmos_ovrl, sigmos_sig, sigmos_col, sigmos_disc, sigmos_loud, sigmos_reverb]
produces_cardinality: '1:0|1'
requires_in_memory_waveform: true
params:
  - {name: model_dir, type: path, default: ~/.cache/nemo_curator/sigmos_model/}
  - {name: model_path, type: path, default: null}
  - {name: noise_threshold, type: float, default: 4.0, min: 1.0, max: 5.0}
  - {name: ovrl_threshold, type: float, default: 3.5, min: 1.0, max: 5.0}
  - {name: sig_threshold, type: float, default: null, min: 1.0, max: 5.0}
  - {name: col_threshold, type: float, default: null, min: 1.0, max: 5.0}
  - {name: disc_threshold, type: float, default: null, min: 1.0, max: 5.0}
  - {name: loud_threshold, type: float, default: null, min: 1.0, max: 5.0}
  - {name: reverb_threshold, type: float, default: null, min: 1.0, max: 5.0}
resources: {cpus: 1.0, gpus: 0.5, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
models:
  - {name: sig-mos-onnx, provider: github, license: Apache-2.0, description: "Microsoft SIG-Challenge ONNX"}
license: Apache-2.0
commercial_safe: true
cost_hint: medium
quality_hint: high
tags: [mos, sigmos, filter, onnx, gpu]
source_file: nemo_curator/stages/audio/filtering/sigmos.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# tagging/
# ---------------------------------------------------------------------------

W("ResampleAudioStage", """
schema_version: '1.0'
name: ResampleAudioStage
target: nemo_curator.stages.audio.tagging.resample_audio.ResampleAudioStage
version: 1.0.0
public_api: true
description: |
  Shell out to ffmpeg to resample the file at audio_filepath_key to
  target_sample_rate / target_format / target_nchannels, writing a NEW file
  to resampled_audio_filepath_key. Required by ASR / diarization / WhisperX
  paths that expect 16 kHz mono on disk.
summary: ffmpeg-based on-disk resample to 16 kHz mono (default).
category: tagging
capabilities: [normalize_sample_rate]
also_handles: [normalize_mono]
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [resampled_audio_filepath, duration, audio_item_id]
produces_cardinality: '1:1'
requires_on_disk_path: true
produces_on_disk_files: true
params:
  - {name: input_format, type: str, default: wav}
  - {name: target_sample_rate, type: int, default: 16000, min: 8000, max: 96000}
  - {name: target_format, type: enum, default: wav, choices: [wav, flac, ogg]}
  - {name: target_nchannels, type: int, default: 1, min: 1, max: 2}
  - {name: audio_filepath_key, type: str, default: audio_filepath}
  - {name: resampled_audio_filepath_key, type: str, default: resampled_audio_filepath}
  - {name: duration_key, type: str, default: duration}
  - {name: audio_item_id_key, type: str, default: audio_item_id}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [resample, ffmpeg, on_disk]
source_file: nemo_curator/stages/audio/tagging/resample_audio.py
smoke_test_status: unknown
""")


W("SplitLongAudioStage", """
schema_version: '1.0'
name: SplitLongAudioStage
target: nemo_curator.stages.audio.tagging.split.SplitLongAudioStage
version: 1.0.0
public_api: true
description: |
  Split a very long audio file into approximately suggested_max_len chunks
  (writes new WAV files alongside the source); keeps the task 1:1 but stamps
  child-file metadata into task.data for downstream rejoin.
summary: Cut long recordings into ~max_len chunks for downstream batched processing.
category: tagging
capabilities: [long_audio_split]
also_handles: []
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [audio_filepath, split_files]
produces_cardinality: '1:1'
requires_on_disk_path: true
produces_on_disk_files: true
params:
  - {name: suggested_max_len, type: float, default: 3600.0, min: 1.0, max: 36000.0}
  - {name: min_len, type: float, default: 1.0, min: 0.0, max: 3600.0}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [split, long_audio, on_disk]
source_file: nemo_curator/stages/audio/tagging/split.py
smoke_test_status: unknown
""")


W("JoinSplitAudioMetadataStage", """
schema_version: '1.0'
name: JoinSplitAudioMetadataStage
target: nemo_curator.stages.audio.tagging.split.JoinSplitAudioMetadataStage
version: 1.0.0
public_api: true
description: |
  Inverse of SplitLongAudioStage: take chunk-level results and stitch them
  back together onto the original file by remapping timestamps and merging
  per-chunk annotations.
summary: Re-attach chunk-level results onto the original long file.
category: tagging
capabilities: [long_audio_join]
also_handles: [timestamp_remap]
inputs:
  top_level: []
  data: [split_files]
outputs:
  top_level: []
  data: [audio_filepath, segments]
produces_cardinality: '1:1'
params: []
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [join, long_audio]
source_file: nemo_curator/stages/audio/tagging/split.py
smoke_test_status: unknown
""")


W("SplitASRAlignJoinStage", """
schema_version: '1.0'
name: SplitASRAlignJoinStage
target: nemo_curator.stages.audio.tagging.split.SplitASRAlignJoinStage
version: 1.0.0
public_api: true
description: |
  CompositeStage that wraps SplitLongAudio + NeMo ASR + JoinSplitAudioMetadata
  into a single named building block for long-audio transcription with
  alignment.
summary: One-call split-ASR-align-join for long files.
category: composite
capabilities: [asr_align, long_audio_split, long_audio_join]
also_handles: []
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [audio_filepath, text, words, segments]
produces_cardinality: '1:1'
params:
  - {name: suggested_max_len, type: float, default: 3600.0}
  - {name: min_len, type: float, default: 1.0}
  - {name: model_name, type: str, default: 'nvidia/parakeet-tdt_ctc-1.1b'}
  - {name: is_fastconformer, type: bool, default: true}
  - {name: decoder_type, type: enum, default: rnnt, choices: [rnnt, ctc]}
  - {name: max_len, type: float, default: 40.0}
  - {name: batch_size, type: int, default: 100, min: 1, max: 4096}
  - {name: transcribe_batch_size, type: int, default: 32}
  - {name: split_batch_size, type: int, default: 5000}
  - {name: num_workers, type: int, default: 10}
  - {name: infer_segment_only, type: bool, default: false}
  - {name: compute_timestamps, type: bool, default: true}
  - {name: timestamp_type, type: enum, default: word, choices: [word, char]}
  - {name: text_key, type: str, default: text}
  - {name: words_key, type: str, default: words}
  - {name: disable_word_confidence, type: bool, default: false}
  - {name: segments_key, type: str, default: segments}
resources: {cpus: 1.0, gpus: 1.0, gpu_memory_gb: 0.0}
batch_size: 100
supports_process_batch: true
preferred_executor: any
models:
  - {name: parakeet-tdt_ctc-1.1b, provider: huggingface, repo_id: 'nvidia/parakeet-tdt_ctc-1.1b', license: CC-BY-4.0}
license: Apache-2.0
commercial_safe: true
cost_hint: very_expensive
quality_hint: high
tags: [asr, alignment, long_audio, composite, gpu]
source_file: nemo_curator/stages/audio/tagging/split.py
smoke_test_status: unknown
""")


W("NeMoASRAlignerStage", """
schema_version: '1.0'
name: NeMoASRAlignerStage
target: nemo_curator.stages.audio.tagging.inference.nemo_asr_align.NeMoASRAlignerStage
version: 1.0.0
public_api: true
description: |
  NeMo ASR + word/segment alignment. Loads a Parakeet-class model and emits
  per-clip transcripts plus word-level timestamps when compute_timestamps is on.
summary: NeMo ASR transcription with word-level alignment.
category: tagging
capabilities: [asr, asr_align]
also_handles: []
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [text, words, segments]
produces_cardinality: '1:1'
requires_on_disk_path: true
params:
  - {name: model_name, type: str, default: 'nvidia/parakeet-tdt_ctc-1.1b'}
  - {name: min_len, type: float, default: 1.0}
  - {name: max_len, type: float, default: 40.0}
  - {name: is_fastconformer, type: bool, default: true}
  - {name: decoder_type, type: enum, default: rnnt, choices: [rnnt, ctc]}
  - {name: transcribe_batch_size, type: int, default: 32}
  - {name: num_workers, type: int, default: 10}
  - {name: batch_size, type: int, default: 100}
  - {name: split_batch_size, type: int, default: 5000}
  - {name: compute_timestamps, type: bool, default: true}
  - {name: timestamp_type, type: enum, default: word, choices: [word, char]}
  - {name: infer_segment_only, type: bool, default: false}
  - {name: segments_key, type: str, default: segments}
  - {name: text_key, type: str, default: text}
  - {name: words_key, type: str, default: words}
  - {name: disable_word_confidence, type: bool, default: false}
resources: {cpus: 1.0, gpus: 1.0, gpu_memory_gb: 0.0}
batch_size: 100
supports_process_batch: true
preferred_executor: any
models:
  - {name: parakeet-tdt_ctc-1.1b, provider: huggingface, repo_id: 'nvidia/parakeet-tdt_ctc-1.1b', license: CC-BY-4.0}
license: Apache-2.0
commercial_safe: true
cost_hint: expensive
quality_hint: high
tags: [asr, alignment, parakeet, gpu]
source_file: nemo_curator/stages/audio/tagging/inference/nemo_asr_align.py
smoke_test_status: unknown
""")


W("MergeAlignmentDiarizationStage", """
schema_version: '1.0'
name: MergeAlignmentDiarizationStage
target: nemo_curator.stages.audio.tagging.merge_alignment_diarization.MergeAlignmentDiarizationStage
version: 1.0.0
public_api: true
description: |
  Take word-level alignment (from NeMoASRAlignerStage) and diarization segments
  (from PyAnnoteDiarizationStage / InferenceSortformerStage) and merge them
  into a single rich-transcript representation where each word is attributed
  to a speaker.
summary: Join word alignment with diarization to produce per-speaker word streams.
category: tagging
capabilities: [diarization_align_merge]
also_handles: []
inputs:
  top_level: []
  data: [words, segments]
outputs:
  top_level: []
  data: [text, words, segments]
produces_cardinality: '1:1'
params:
  - {name: text_key, type: str, default: text}
  - {name: words_key, type: str, default: words}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [merge, diarization, alignment]
source_file: nemo_curator/stages/audio/tagging/merge_alignment_diarization.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# inference/
# ---------------------------------------------------------------------------

W("InferenceAsrNemoStage", """
schema_version: '1.0'
name: InferenceAsrNemoStage
target: nemo_curator.stages.audio.inference.asr.asr_nemo.InferenceAsrNemoStage
version: 1.0.0
public_api: true
description: |
  Light-weight NeMo ASR transcription wrapper. Loads a model_name checkpoint
  (no alignment) and writes the prediction string to pred_text_key. Cheaper
  than NeMoASRAlignerStage when timestamps aren't needed.
summary: NeMo ASR transcription only (no alignment).
category: inference
capabilities: [asr]
also_handles: []
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [pred_text]
produces_cardinality: '1:1'
requires_on_disk_path: true
params:
  - {name: model_name, type: str, required: true, description: "NeMo ASR checkpoint name (e.g. nvidia/parakeet-ctc-1.1b)."}
  - {name: filepath_key, type: str, default: audio_filepath}
  - {name: pred_text_key, type: str, default: pred_text}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 16
supports_process_batch: true
preferred_executor: any
models:
  - {name: parakeet-tdt_ctc-1.1b, provider: huggingface, repo_id: 'nvidia/parakeet-tdt_ctc-1.1b', license: CC-BY-4.0}
license: Apache-2.0
commercial_safe: true
cost_hint: expensive
quality_hint: high
tags: [asr, nemo, parakeet, gpu]
source_file: nemo_curator/stages/audio/inference/asr/asr_nemo.py
smoke_test_status: unknown
""")


W("InferenceSortformerStage", """
schema_version: '1.0'
name: InferenceSortformerStage
target: nemo_curator.stages.audio.inference.sortformer.InferenceSortformerStage
version: 1.0.0
public_api: true
description: |
  Streaming Sortformer diarizer. Default model nvidia/diar_streaming_sortformer_4spk-v2.1.
  Emits a list of (speaker_id, start, end) tuples under diar_segments_key. Up to 4
  speakers; chunked streaming inference with configurable left/right context.
summary: Streaming NVIDIA Sortformer diarizer (up to 4 speakers).
category: inference
capabilities: [speaker_diarization, speaker_count]
also_handles: []
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [diar_segments, num_speakers]
produces_cardinality: '1:1'
requires_on_disk_path: true
params:
  - {name: model_name, type: str, default: 'nvidia/diar_streaming_sortformer_4spk-v2.1'}
  - {name: filepath_key, type: str, default: audio_filepath}
  - {name: diar_segments_key, type: str, default: diar_segments}
  - {name: chunk_len, type: int, default: 340}
  - {name: chunk_left_context, type: int, default: 1}
  - {name: chunk_right_context, type: int, default: 40}
  - {name: fifo_len, type: int, default: 40}
  - {name: spkcache_update_period, type: int, default: 300}
  - {name: spkcache_len, type: int, default: 188}
  - {name: inference_batch_size, type: int, default: 1}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 8.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
models:
  - {name: diar_streaming_sortformer_4spk-v2.1, provider: huggingface, repo_id: 'nvidia/diar_streaming_sortformer_4spk-v2.1', license: CC-BY-4.0}
license: Apache-2.0
commercial_safe: true
cost_hint: expensive
quality_hint: high
tags: [diarization, sortformer, streaming, gpu]
source_file: nemo_curator/stages/audio/inference/sortformer.py
smoke_test_status: unknown
""")


W("WhisperXVADStage", """
schema_version: '1.0'
name: WhisperXVADStage
target: nemo_curator.stages.audio.inference.vad.whisperx_vad.WhisperXVADStage
version: 1.0.0
public_api: true
description: |
  WhisperX/pyannote VAD wrapper. Operates on resampled_audio_filepath (16 kHz mono)
  and writes a list of speech segments under segments_key.
summary: WhisperX VAD on 16 kHz resampled audio.
category: inference
capabilities: [vad]
also_handles: []
inputs:
  top_level: []
  data: [resampled_audio_filepath]
outputs:
  top_level: []
  data: [vad_segments]
produces_cardinality: '1:1'
requires_sample_rate: 16000
requires_on_disk_path: true
params:
  - {name: min_length, type: float, default: 0.5, min: 0.0, max: 3600.0}
  - {name: max_length, type: float, default: 40.0, min: 0.1, max: 3600.0}
  - {name: vad_onset, type: float, default: 0.5, min: 0.0, max: 1.0}
  - {name: vad_offset, type: float, default: 0.363, min: 0.0, max: 1.0}
  - {name: segments_key, type: str, default: vad_segments}
  - {name: audio_filepath_key, type: str, default: resampled_audio_filepath}
resources: {cpus: 1.0, gpus: 1.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
models:
  - {name: pyannote-vad, provider: huggingface, license: MIT}
license: Apache-2.0
commercial_safe: true
cost_hint: medium
quality_hint: high
tags: [vad, whisperx, pyannote, gpu]
source_file: nemo_curator/stages/audio/inference/vad/whisperx_vad.py
smoke_test_status: unknown
""")


W("PyAnnoteDiarizationStage", """
schema_version: '1.0'
name: PyAnnoteDiarizationStage
target: nemo_curator.stages.audio.inference.speaker_diarization.pyannote.PyAnnoteDiarizationStage
version: 1.0.0
public_api: true
description: |
  pyannote/speaker-diarization-3.1 wrapper. Requires an HF token (env
  HF_TOKEN) at run time. Emits a list of (speaker_id, start, end) under
  segments_key and an overlap_segments list under overlap_segments_key.
summary: pyannote speaker-diarization 3.1.
category: inference
capabilities: [speaker_diarization, speaker_count]
also_handles: []
inputs:
  top_level: []
  data: [resampled_audio_filepath]
outputs:
  top_level: []
  data: [segments, overlap_segments]
produces_cardinality: '1:1'
requires_sample_rate: 16000
requires_on_disk_path: true
requires_external_token_env: HF_TOKEN
params:
  - {name: model_name, type: str, default: 'pyannote/speaker-diarization-3.1'}
  - {name: segmentation_batch_size, type: int, default: 128}
  - {name: embedding_batch_size, type: int, default: 128}
  - {name: min_length, type: float, default: 0.5}
  - {name: max_length, type: float, default: 40.0}
  - {name: audio_filepath_key, type: str, default: resampled_audio_filepath}
  - {name: segments_key, type: str, default: segments}
  - {name: overlap_segments_key, type: str, default: overlap_segments}
resources: {cpus: 1.0, gpus: 1.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
models:
  - {name: pyannote-speaker-diarization-3.1, provider: huggingface, repo_id: 'pyannote/speaker-diarization-3.1', license: MIT}
license: Apache-2.0
commercial_safe: true
cost_hint: expensive
quality_hint: high
tags: [diarization, pyannote, gpu]
source_file: nemo_curator/stages/audio/inference/speaker_diarization/pyannote.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# alm/
# ---------------------------------------------------------------------------

W("ALMDataBuilderStage", """
schema_version: '1.0'
name: ALMDataBuilderStage
target: nemo_curator.stages.audio.alm.alm_data_builder.ALMDataBuilderStage
version: 1.0.0
public_api: true
description: |
  Pack diarized + aligned segments into Audio Language Model training windows
  of approximately target_window_duration seconds with the given tolerance.
  Filters windows that fall below min_bandwidth / min_sample_rate and that
  contain too few or too many speakers.
summary: Build ALM training windows from diarized/aligned segments.
category: alm
capabilities: [alm_package]
also_handles: []
inputs:
  top_level: []
  data: [segments, words]
outputs:
  top_level: []
  data: [alm_windows]
produces_cardinality: '1:1'
params:
  - {name: target_window_duration, type: float, default: 120.0, min: 1.0, max: 600.0}
  - {name: tolerance, type: float, default: 0.1, min: 0.0, max: 1.0}
  - {name: min_bandwidth, type: int, default: 8000, min: 0, max: 24000}
  - {name: min_sample_rate, type: int, default: 16000, min: 8000, max: 96000}
  - {name: min_speakers, type: int, default: 2, min: 1, max: 8}
  - {name: max_speakers, type: int, default: 5, min: 1, max: 8}
  - {name: truncation, type: bool, default: true}
  - {name: drop_fields, type: str, default: words}
  - {name: drop_fields_top_level, type: str, default: 'words,segments'}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [alm, packaging, cpu]
source_file: nemo_curator/stages/audio/alm/alm_data_builder.py
smoke_test_status: unknown
""")


W("ALMDataOverlapStage", """
schema_version: '1.0'
name: ALMDataOverlapStage
target: nemo_curator.stages.audio.alm.alm_data_overlap.ALMDataOverlapStage
version: 1.0.0
public_api: true
description: |
  Overlap-based dedup pass for ALM windows. Drops a window if it overlaps an
  already-kept window above overlap_percentage. Use after ALMDataBuilderStage
  to control redundancy in the training set.
summary: Dedup ALM windows by overlap percentage.
category: alm
capabilities: [alm_overlap]
also_handles: []
inputs:
  top_level: []
  data: [alm_windows]
outputs:
  top_level: []
  data: [alm_windows]
produces_cardinality: '1:1'
params:
  - {name: overlap_percentage, type: int, default: 0, min: 0, max: 100}
  - {name: target_duration, type: float, default: 120.0, min: 1.0, max: 600.0}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [alm, dedup, cpu]
source_file: nemo_curator/stages/audio/alm/alm_data_overlap.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# postprocessing/
# ---------------------------------------------------------------------------

W("TimestampMapperStage", """
schema_version: '1.0'
name: TimestampMapperStage
target: nemo_curator.stages.audio.postprocessing.timestamp_mapper.TimestampMapperStage
version: 1.0.0
public_api: true
description: |
  Normalize the output schema. Constructs the canonical (original_file,
  original_start_ms, original_end_ms, duration_ms, duration) and copies the
  whitelist of passthrough_keys from the input. When diarization is present,
  also emits diar_segments + speaking_duration.
summary: Pipeline-output normalization to original-file timeline.
category: postprocess
capabilities: [timestamp_remap]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: [original_file, original_start_ms, original_end_ms, duration_ms, duration]
produces_cardinality: '1:1'
params:
  - {name: passthrough_keys, type: list, default: null, description: "Whitelist of keys to copy through. None = default whitelist."}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_cheap
tags: [postprocess, normalize, cpu]
source_file: nemo_curator/stages/audio/postprocessing/timestamp_mapper.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# metrics/
# ---------------------------------------------------------------------------

W("GetPairwiseWerStage", """
schema_version: '1.0'
name: GetPairwiseWerStage
target: nemo_curator.stages.audio.metrics.get_wer.GetPairwiseWerStage
version: 1.0.0
public_api: true
description: |
  Compute editdistance-based WER between text_key (reference) and pred_text_key
  (ASR hypothesis), writing the result to wer_key (a percent, 0-100). Pairs
  naturally with InferenceAsrNemoStage upstream and a PreserveByValueStage
  downstream to enforce asr_wer_max.
summary: Compute WER between reference and ASR hypothesis.
category: metric
capabilities: [wer]
also_handles: []
inputs:
  top_level: []
  data: [text, pred_text]
outputs:
  top_level: []
  data: [wer]
produces_cardinality: '1:1'
params:
  - {name: text_key, type: str, default: text}
  - {name: pred_text_key, type: str, default: pred_text}
  - {name: wer_key, type: str, default: wer}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_cheap
tags: [wer, metric, cpu]
source_file: nemo_curator/stages/audio/metrics/get_wer.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# datasets/
# ---------------------------------------------------------------------------

W("CreateInitialManifestReadSpeechStage", """
schema_version: '1.0'
name: CreateInitialManifestReadSpeechStage
target: nemo_curator.stages.audio.datasets.readspeech.create_initial_manifest.CreateInitialManifestReadSpeechStage
version: 1.0.0
public_api: true
description: |
  Dataset reader for the DNS Challenge Read Speech corpus. Optionally downloads
  the corpus and emits one AudioTask per file (capped at max_samples). Use
  for quick experiments without a pre-existing manifest.
summary: Read Speech (DNS) dataset source — auto-download + tasks.
category: source
capabilities: [dataset_create]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: [audio_filepath, text]
produces_cardinality: '1:N'
params:
  - {name: max_samples, type: int, default: 5000, min: 1, max: 1000000}
  - {name: auto_download, type: bool, default: true}
  - {name: filepath_key, type: str, default: audio_filepath}
  - {name: text_key, type: str, default: text}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [dataset, reader, dns]
source_file: nemo_curator/stages/audio/datasets/readspeech/create_initial_manifest.py
smoke_test_status: unknown
""")


W("CreateInitialManifestFleursStage", """
schema_version: '1.0'
name: CreateInitialManifestFleursStage
target: nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.CreateInitialManifestFleursStage
version: 1.0.0
public_api: true
description: |
  Dataset reader for the google/fleurs multilingual corpus. Configurable by
  lang + split; emits one AudioTask per fleurs example.
summary: FLEURS dataset source — multilingual auto-download.
category: source
capabilities: [dataset_create]
also_handles: []
inputs:
  top_level: []
  data: []
outputs:
  top_level: []
  data: [audio_filepath, text]
produces_cardinality: '1:N'
params:
  - {name: lang, type: str, default: '', description: "FLEURS language code, e.g. en_us."}
  - {name: split, type: enum, default: '', choices: ['', train, validation, test]}
  - {name: raw_data_dir, type: path, default: ''}
  - {name: filepath_key, type: str, default: audio_filepath}
  - {name: text_key, type: str, default: text}
resources: {cpus: 1.0, gpus: 0.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: cheap
tags: [dataset, reader, fleurs, multilingual]
source_file: nemo_curator/stages/audio/datasets/fleurs/create_initial_manifest.py
smoke_test_status: unknown
""")


# ---------------------------------------------------------------------------
# advanced_pipelines/
# ---------------------------------------------------------------------------

W("AudioDataFilterStage", """
schema_version: '1.0'
name: AudioDataFilterStage
target: nemo_curator.stages.audio.advanced_pipelines.audio_data_filter.audio_data_filter.AudioDataFilterStage
version: 1.0.0
public_api: true
description: |
  Pre-blessed CompositeStage: MonoConversion -> [optional VAD fanout] ->
  Band/UTMOS/SIGMOS filters -> [optional SegmentConcat] -> [optional
  SpeakerSep + per-speaker filters] -> TimestampMapper. The most common
  "give me clean single-speaker clips" preset. Configurable via YAML
  config_path or config dict.
summary: Blessed clean-single-speaker preset composite.
category: composite
capabilities: [vad, quality_filter_mos, quality_filter_band, speaker_separation, speaker_diarization, segment_concat]
also_handles: [duration_filter, silence_removal, normalize_mono, normalize_sample_rate, timestamp_remap]
inputs:
  top_level: []
  data: [audio_filepath]
outputs:
  top_level: []
  data: [original_file, original_start_ms, original_end_ms, duration]
produces_cardinality: '1:N'
params:
  - {name: config_path, type: path, default: null}
  - {name: config, type: dict, default: null}
  - {name: name, type: str, default: AudioDataFilter}
resources: {cpus: 1.0, gpus: 1.0, gpu_memory_gb: 0.0}
batch_size: 1
supports_process_batch: false
preferred_executor: any
license: Apache-2.0
commercial_safe: true
cost_hint: very_expensive
quality_hint: high
tags: [preset, composite, clean_speech, gpu]
source_file: nemo_curator/stages/audio/advanced_pipelines/audio_data_filter/audio_data_filter.py
smoke_test_status: unknown
""")


if __name__ == "__main__":
    print(f"wrote {len(list(CARDS_ROOT.rglob('stage_card.yaml')))} cards to {CARDS_ROOT}", file=sys.stderr)
