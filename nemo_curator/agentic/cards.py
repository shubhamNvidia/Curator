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
"""Schemas for machine-readable cards consumed by the agent.

Four card kinds:

- :class:`StageCard` — one per :class:`ProcessingStage` class. Tells the agent
  what the stage consumes, produces, costs, and is allowed to do.
- :class:`DatasetCard` — one per processed input or output dataset. Captures
  the deterministic profile from Layer 1.
- :class:`ModelCard` — model metadata referenced from stage cards (license,
  provider, checksum). Cards are JSON / YAML on disk.
- :class:`RunCard` — emitted by every pipeline execution. The contract for
  ``curator-adv replay`` and the eval harness.

Design notes — see ``ADV/adv_phase2/AUDIO_INTERNALS.md`` section 14 for the
deltas these schemas honor. In summary:

- ``StageCard.name`` MUST equal the Python class name (i.e. the registry key
  in ``nemo_curator.stages.base._STAGE_REGISTRY``).
- ``StageCard.target`` is the dotted module path used by Hydra ``_target_``
  in canonical ``stages:`` YAML.
- ``StageCard.inputs`` / ``outputs`` are split into ``top_level`` (task
  attributes like ``filepath_key``) and ``data`` (keys of ``task.data``),
  mirroring the tuple returned by ``ProcessingStage.inputs()`` / ``outputs()``.
- ``ResourceSpec`` enforces the same XOR as ``stages.resources.Resources``:
  ``gpus > 0`` and ``gpu_memory_gb > 0`` are mutually exclusive.
- ``produces_cardinality`` is the *agentic* hint with no source of truth in
  the codebase today; it is authored per stage card by humans.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ----------------------------------------------------------------------------
# Enumerations
# ----------------------------------------------------------------------------


class Cardinality(str, Enum):
    """How a stage's ``process()`` maps inputs to outputs.

    Not declared anywhere in the existing code — inferred from each stage's
    ``process()`` return-type contract:

    - ``ONE_TO_ONE``: single :class:`AudioTask` returned. Most stages.
    - ``ONE_TO_MANY``: ``list[AudioTask]`` returned with new ``task_id``s.
      Examples: :class:`VADSegmentationStage` (``nested=False``),
      :class:`SpeakerSeparationStage`.
    - ``ONE_TO_ZERO_OR_ONE``: drop-style filters.
      Examples: :class:`UTMOSFilterStage`, :class:`PreserveByValueStage`.
    - ``ONE_TO_ONE_NESTED``: emits a single task with
      ``task.data["segments"] = [...]`` for downstream filters.
      Example: :class:`VADSegmentationStage` (``nested=True``).
    - ``MANY_TO_ONE``: collapses a batch into a single output task.
      Only :class:`AudioToDocumentStage` today.
    """

    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    ONE_TO_ZERO_OR_ONE = "1:0|1"
    ONE_TO_ONE_NESTED = "1:1+nested"
    MANY_TO_ONE = "N:1"


class StageCategory(str, Enum):
    """Coarse classification used by the planner for ranking and ordering.

    Only categories present in the current catalog are listed. New categories
    (e.g. annotation, augmentation) will be added when the modules that
    populate them are written.
    """

    SOURCE = "source"                # readers, dataset creators
    PREPROCESS = "preprocess"        # mono conversion
    SEGMENTATION = "segmentation"    # VAD, speaker separation, long-split
    INFERENCE = "inference"          # ASR, diarization
    FILTER = "filter"                # MOS / bandwidth / preserve-by-value
    TAGGING = "tagging"              # resample-on-disk, ASR-align, merge
    METRIC = "metric"                # WER
    POSTPROCESS = "postprocess"      # timestamp mapping
    ALM = "alm"                      # audio language model packaging
    IO = "io"                        # extract segments, audio→document
    SINK = "sink"                    # manifest writer
    COMPOSITE = "composite"          # CompositeStage subclasses


class Modality(str, Enum):
    """Modalities a stage consumes or produces."""

    AUDIO = "audio"
    TEXT = "text"


class Phase(str, Enum):
    """Coarse pipeline phase used as a hard ordering constraint.

    The validator's topo sort respects phase order *first*, then the I/O
    DAG. A stage in phase N must run BEFORE every stage in phase M > N.
    The enum's ``_order`` index is the canonical rank.

    Phases (low → high):

    1. ``source``       Readers and dataset creators (ManifestReader, ...).
    2. ``preprocess``   On-disk format normalization (ResampleAudioStage).
    3. ``load``         Bring audio into memory (MonoConversionStage).
    4. ``segment``      Segmentation / fan-out (VADSegmentationStage).
    5. ``analyze``      Filters / inference / scoring (MOS, Band, ASR,
                        diarization, separation).
    6. ``concat``       Re-stitch fanned segments (SegmentConcatenationStage).
    7. ``package``      Window-level packaging (ALM stages).
    8. ``polish``       Row-schema normalization; CLEARS task.data
                        (TimestampMapperStage).
    9. ``materialize``  Write per-clip audio to disk (SegmentExtractionStage).
    10. ``sink``        Final manifest / document write (ManifestWriterStage,
                        AudioToDocumentStage).
    """

    SOURCE = "source"
    PREPROCESS = "preprocess"
    LOAD = "load"
    SEGMENT = "segment"
    ANALYZE = "analyze"
    CONCAT = "concat"
    PACKAGE = "package"
    POLISH = "polish"
    MATERIALIZE = "materialize"
    SINK = "sink"


PHASE_ORDER: dict[Phase, int] = {p: i for i, p in enumerate(Phase)}


class InputShape(str, Enum):
    """What kind of input shape a stage NEEDS upstream of it.

    Used to detect when SegmentConcatenationStage must be auto-inserted:
    if an upstream stage emits ``fan_out_*`` shape and a downstream stage
    requires ``whole_file``, we insert concat between them.
    """

    ANY = "any"                          # tolerates anything
    WHOLE_FILE = "whole_file"            # ONE coherent task per source file
    NESTED_SEGMENTS = "nested_segments"  # needs task.data["segments"]
    FANNED_OUT = "fanned_out"            # needs the 1:N tasks from VAD


class OutputShape(str, Enum):
    """What kind of output shape this stage emits.

    Used to detect upstream/downstream mismatches against
    :class:`InputShape`.
    """

    PASSTHROUGH = "passthrough"          # 1:1, task.data preserved
    FAN_OUT_SEGMENTS = "fan_out_segments"  # 1:N by VAD segment
    FAN_OUT_SPEAKERS = "fan_out_speakers"  # 1:N by speaker
    FAN_IN = "fan_in"                    # N:1 collapse
    REBUILD_TASK = "rebuild_task"        # CLEARS task.data; writes row schema
    WRITE_FILES = "write_files"          # materializes per-clip files on disk
    NESTED_SEGMENTS = "nested_segments"  # 1:1 + task.data["segments"]


class CapabilityTag(str, Enum):
    """Predicates the planner uses to satisfy :class:`IntentCategories`.

    A stage may advertise zero or more of these. Multiple stages may share a
    tag (e.g. both UTMOS and SIGMOS satisfy ``quality_filter_mos``); the cost
    ladder + license gate resolve the tie.

    Only tags satisfied by the current 28-stage catalog are listed. New tags
    are added together with the stages that satisfy them.
    """

    READ_MANIFEST = "read_manifest"
    WRITE_MANIFEST = "write_manifest"
    GET_DURATION = "get_duration"
    NORMALIZE_MONO = "normalize_mono"
    NORMALIZE_SAMPLE_RATE = "normalize_sample_rate"
    VAD = "vad"
    DURATION_FILTER = "duration_filter"
    SILENCE_REMOVAL = "silence_removal"
    SPEAKER_DIARIZATION = "speaker_diarization"
    SPEAKER_SEPARATION = "speaker_separation"
    SPEAKER_COUNT = "speaker_count"
    ASR = "asr"
    ASR_ALIGN = "asr_align"
    QUALITY_FILTER_MOS = "quality_filter_mos"
    QUALITY_FILTER_BAND = "quality_filter_band"
    SEGMENT_EXTRACT = "segment_extract"
    SEGMENT_CONCAT = "segment_concat"
    WER = "wer"
    ALM_PACKAGE = "alm_package"
    ALM_OVERLAP = "alm_overlap"
    DATASET_CREATE = "dataset_create"
    AUDIO_TO_DOCUMENT = "audio_to_document"
    LONG_AUDIO_SPLIT = "long_audio_split"
    LONG_AUDIO_JOIN = "long_audio_join"
    TIMESTAMP_REMAP = "timestamp_remap"
    DIARIZATION_ALIGN_MERGE = "diarization_align_merge"
    VALUE_FILTER = "value_filter"


class LicenseKind(str, Enum):
    """SPDX-style license buckets the gate checks against ``IntentCategories.commercial_only``.

    Anything not in ``COMMERCIAL_OK`` is gated when commercial mode is on.
    """

    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    BSD_3_CLAUSE = "BSD-3-Clause"
    BSD_2_CLAUSE = "BSD-2-Clause"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    GPL_3_0 = "GPL-3.0"
    GPL_2_0 = "GPL-2.0"
    LGPL_3_0 = "LGPL-3.0"
    AGPL_3_0 = "AGPL-3.0"
    PROPRIETARY = "Proprietary"
    UNKNOWN = "Unknown"


COMMERCIAL_OK: frozenset[LicenseKind] = frozenset({
    LicenseKind.APACHE_2_0,
    LicenseKind.MIT,
    LicenseKind.BSD_3_CLAUSE,
    LicenseKind.BSD_2_CLAUSE,
    LicenseKind.CC_BY_4_0,
})


# ----------------------------------------------------------------------------
# Sub-schemas reused across cards
# ----------------------------------------------------------------------------


class ResourceSpec(BaseModel):
    """Mirror of :class:`nemo_curator.stages.resources.Resources`.

    Enforces the same XOR: ``gpus > 0`` and ``gpu_memory_gb > 0`` are mutually
    exclusive (Curator's ``Resources.__post_init__`` raises if both are set).
    """

    model_config = ConfigDict(extra="forbid")

    cpus: float = Field(default=1.0, ge=0.0)
    gpus: float = Field(default=0.0, ge=0.0)
    gpu_memory_gb: float = Field(default=0.0, ge=0.0)
    nvdecs: int = Field(default=0, ge=0)
    nvencs: int = Field(default=0, ge=0)
    entire_gpu: bool = False

    @model_validator(mode="after")
    def _xor_gpu_modes(self) -> ResourceSpec:
        if self.gpus > 0 and self.gpu_memory_gb > 0:
            msg = "gpus and gpu_memory_gb are mutually exclusive"
            raise ValueError(msg)
        return self

    def to_resources_kwargs(self) -> dict[str, Any]:
        """Return kwargs ready for ``Resources(**kwargs)`` in canonical YAML."""

        kwargs: dict[str, Any] = {"cpus": self.cpus}
        if self.gpus:
            kwargs["gpus"] = self.gpus
        if self.gpu_memory_gb:
            kwargs["gpu_memory_gb"] = self.gpu_memory_gb
        return kwargs


class IOSpec(BaseModel):
    """Mirrors the tuple returned by ``ProcessingStage.inputs()`` / ``outputs()``.

    The base class returns ``tuple[list[str], list[str]]`` = ``(top_level,
    data_keys)``.
    """

    model_config = ConfigDict(extra="forbid")

    top_level: list[str] = Field(default_factory=list)
    data: list[str] = Field(default_factory=list)

    def as_tuple(self) -> tuple[list[str], list[str]]:
        return list(self.top_level), list(self.data)


class ParamSpec(BaseModel):
    """One row of the stage's ``__init__`` keyword surface."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["int", "float", "str", "bool", "list", "dict", "path", "enum", "tuple"]
    default: Any | None = None
    required: bool = False
    min: float | None = None
    max: float | None = None
    choices: list[Any] | None = None
    description: str | None = None
    examples: list[Any] | None = None
    affects_behavior: bool = True


class ModelRef(BaseModel):
    """Pointer to a :class:`ModelCard` referenced from a stage card."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: Literal["huggingface", "torch_hub", "onnx", "github", "internal", "other"] = "huggingface"
    repo_id: str | None = None
    revision: str | None = None
    license: LicenseKind = LicenseKind.UNKNOWN
    requires_token_env: str | None = None
    download_size_mb: float | None = None
    description: str | None = None


class ThresholdBand(BaseModel):
    """One row in a stage's threshold guidance.

    Lets a stage card teach the planner what a numeric parameter value means
    in plain English. ``param`` names the parameter on the stage, ``value``
    is a concrete number on that parameter's scale, and ``label`` is a short
    intent-phrase that the user might use in a prompt ("clean", "broadcast",
    "studio"). The planner reads these via ``stage_inspect`` and picks a
    threshold from the bands without the system prompt having to encode a
    one-size-fits-all conversion table.
    """

    model_config = ConfigDict(extra="forbid")

    param: str
    value: float
    label: str


class ThresholdSetting(BaseModel):
    """One stage / param / value triple, used inside a :class:`ThresholdCombo`.

    Combos are cross-stage presets — a single user-facing label resolves to
    threshold values on potentially several stages (e.g. the project's
    "clean speech" default touches UTMOSFilterStage, SIGMOSFilterStage and
    BandFilterStage). Each setting in the combo says exactly which stage
    and which param to dial.

    ``value`` accepts a float for numeric thresholds (most cases) or a
    string for enum-valued params such as :class:`BandFilterStage`'s
    ``band_value`` (``full_band`` / ``narrow_band``).
    """

    model_config = ConfigDict(extra="forbid")

    stage: str
    param: str
    value: float | str


class ThresholdCombo(BaseModel):
    """Named cross-stage preset for a recognizable user intent.

    The planner consults these when the user mentions a multi-word quality
    phrase (e.g. "clean non-noisy", "TTS curation"). A combo is a labelled
    bundle of :class:`ThresholdSetting` entries that may span multiple
    stages — preferred over per-axis bands when the user's words match one
    of the bundle's labels/aliases.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    settings: list[ThresholdSetting] = Field(default_factory=list)


class OrderingHints(BaseModel):
    """Soft ordering preferences for a stage.

    Hard ordering — producer-before-consumer on data keys, source-first,
    sink-last, MANY_TO_ONE adjacent-to-sink — is enforced deterministically
    by the validator's topological sort using each card's ``inputs`` and
    ``outputs``. ``OrderingHints`` carries the *preferences* that aren't
    derivable from the I/O contract: "filter cheap before expensive",
    "score after speaker fan-out", and so on. The validator emits INFO
    findings when these are violated; it never rewrites the IR for soft
    preferences.

    Both ``prefer_after`` and ``prefer_before`` accept either stage class
    names (e.g. ``"VADSegmentationStage"``) or capability tag strings
    (e.g. ``"vad"`` — the value of :class:`CapabilityTag.VAD`). Capability
    references match any registered stage with that tag.
    """

    model_config = ConfigDict(extra="forbid")

    prefer_after: list[str] = Field(default_factory=list)
    prefer_before: list[str] = Field(default_factory=list)
    rationale: str = ""


class SelectionHints(BaseModel):
    """Optional planner-facing routing hints surfaced via ``stage_inspect``.

    Fields:

    - ``prefer_when`` / ``avoid_when`` / ``notes`` — free-form bullets the
      planner reads as routing intuition.
    - ``threshold_bands`` — per-axis (single param) numeric guidance.
    - ``combo_presets`` — named cross-stage bundles. If a user prompt's
      phrasing matches a combo's label or aliases, the planner SHOULD
      apply the whole bundle verbatim rather than re-deriving thresholds
      from individual bands. Combos are the project-blessed defaults.
    - ``ordering_hints`` — soft "prefer me after / before X" preferences
      that aren't expressible as I/O dependencies.
    """

    model_config = ConfigDict(extra="forbid")

    prefer_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    threshold_bands: list[ThresholdBand] = Field(default_factory=list)
    combo_presets: list[ThresholdCombo] = Field(default_factory=list)
    ordering_hints: OrderingHints = Field(default_factory=OrderingHints)


# ----------------------------------------------------------------------------
# Stage card
# ----------------------------------------------------------------------------


class StageCard(BaseModel):
    """The agent's view of a single :class:`ProcessingStage` class.

    Loaded from ``stage_card.yaml`` files co-located with each stage module.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity ----------------------------------------------------------------
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(..., description="Must equal the Python class name (registry key).")
    target: str = Field(..., description="Dotted module path for Hydra _target_.")
    version: str = "1.0.0"
    public_api: bool = False
    deprecated: bool = False
    deprecated_reason: str | None = None
    description: str = Field(..., description="One-paragraph human summary.")
    summary: str = Field(..., description="One-sentence summary for the planner.")

    # Classification ----------------------------------------------------------
    category: StageCategory
    phase: Phase = Field(
        default=Phase.ANALYZE,
        description=(
            "Coarse pipeline phase. Hard ordering constraint enforced by "
            "the validator: lower phase index runs before higher. See "
            "the :class:`Phase` enum for semantics."
        ),
    )
    input_shape: InputShape = Field(
        default=InputShape.ANY,
        description="What task shape this stage REQUIRES upstream of it.",
    )
    output_shape: OutputShape = Field(
        default=OutputShape.PASSTHROUGH,
        description="What task shape this stage EMITS to downstream stages.",
    )
    terminal: bool = Field(
        default=False,
        description=(
            "If true, this stage MUST be the last reference to in-memory "
            "audio (e.g. it clears task.data or writes a final manifest). "
            "Used by the validator to detect 'something runs after polish'."
        ),
    )
    capabilities: list[CapabilityTag] = Field(default_factory=list)
    also_handles: list[CapabilityTag] = Field(
        default_factory=list,
        description=(
            "Tags this stage transparently satisfies via internal params, "
            "e.g. VADSegmentationStage(min_duration_sec, max_duration_sec) "
            "also_handles=[duration_filter, silence_removal]."
        ),
    )
    modality_in: Modality = Modality.AUDIO
    modality_out: Modality = Modality.AUDIO

    # I/O & dataflow ----------------------------------------------------------
    inputs: IOSpec = Field(default_factory=IOSpec)
    outputs: IOSpec = Field(default_factory=IOSpec)
    produces_cardinality: Cardinality = Cardinality.ONE_TO_ONE
    nested_segment_key: str | None = Field(
        default=None,
        description="If cardinality=1:1+nested, key in task.data holding the list.",
    )

    # Execution preconditions / postconditions --------------------------------
    requires_sample_rate: int | None = Field(
        default=None,
        description="Downstream expects this SR; auto-insert picks Mono/Resample.",
    )
    requires_mono: bool = False
    requires_in_memory_waveform: bool = False
    requires_on_disk_path: bool = False
    requires_file_extensions: list[str] | None = None
    requires_external_token_env: str | None = None
    produces_on_disk_files: bool = False
    produces_in_memory_waveform: bool = False
    produces_keys_after_run: list[str] = Field(default_factory=list)
    drops_keys_after_run: list[str] = Field(default_factory=list)

    # Parameter surface -------------------------------------------------------
    params: list[ParamSpec] = Field(default_factory=list)

    # Resources & batching ----------------------------------------------------
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    batch_size: int = 1
    supports_process_batch: bool = False

    # Backend hints -----------------------------------------------------------
    preferred_executor: Literal["xenna", "ray_data", "ray_actor_pool", "any"] = "any"
    incompatible_with_inference_server: bool = False

    # Models & licensing ------------------------------------------------------
    models: list[ModelRef] = Field(default_factory=list)
    license: LicenseKind = LicenseKind.APACHE_2_0
    license_constraints: list[str] = Field(default_factory=list)
    commercial_safe: bool = True

    # Cost / quality hints (rough; cost ladder uses these) --------------------
    cost_hint: Literal["very_cheap", "cheap", "medium", "expensive", "very_expensive"] = "cheap"
    quality_hint: Literal["low", "medium", "high"] | None = None

    # Tagging for catalog filtering ------------------------------------------
    tags: list[str] = Field(default_factory=list)

    # Planner routing hints — distinguishes stages sharing a capability tag --
    selection_hints: SelectionHints = Field(default_factory=SelectionHints)

    # Provenance --------------------------------------------------------------
    source_file: str | None = Field(
        default=None,
        description="Path to the source module relative to repo root.",
    )
    last_smoke_test: datetime | None = None
    smoke_test_status: Literal["pass", "fail", "unknown", "skipped"] = "unknown"

    # ---- Validators --------------------------------------------------------

    @field_validator("name")
    @classmethod
    def _name_is_class_like(cls, v: str) -> str:
        if not v or not v[0].isalpha() or not v.replace("_", "").isalnum():
            msg = f"StageCard.name must be a Python class identifier; got {v!r}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _commercial_consistency(self) -> StageCard:
        if self.commercial_safe and self.license not in COMMERCIAL_OK:
            for m in self.models:
                if m.license not in COMMERCIAL_OK:
                    msg = (
                        f"StageCard {self.name}: commercial_safe=true but model "
                        f"{m.name} has non-commercial-safe license {m.license}"
                    )
                    raise ValueError(msg)
        if self.produces_cardinality == Cardinality.ONE_TO_ONE_NESTED and not self.nested_segment_key:
            msg = "produces_cardinality=1:1+nested requires nested_segment_key"
            raise ValueError(msg)
        return self


# ----------------------------------------------------------------------------
# Dataset card
# ----------------------------------------------------------------------------


class DatasetProfile(BaseModel):
    """Deterministic findings of Layer 1 profiler."""

    model_config = ConfigDict(extra="forbid")

    total_files: int = 0
    decodable_files: int = 0
    decode_failure_rate: float = 0.0
    formats: dict[str, int] = Field(default_factory=dict)
    sample_rates_hz: dict[str, int] = Field(default_factory=dict)
    channel_distribution: dict[str, int] = Field(default_factory=dict)
    duration_p05_sec: float | None = None
    duration_p50_sec: float | None = None
    duration_p95_sec: float | None = None
    total_duration_hours: float | None = None
    notes: list[str] = Field(default_factory=list)


class DatasetCard(BaseModel):
    """One per input or output dataset reasoned about by the agent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    name: str
    uri: str
    uri_scheme: Literal["file", "hf", "s3", "tar", "http", "manifest"] = "file"
    description: str = ""
    license: LicenseKind = LicenseKind.UNKNOWN
    commercial_safe: bool = True
    profile: DatasetProfile = Field(default_factory=DatasetProfile)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sample_count_for_profile: int = 0


# ----------------------------------------------------------------------------
# Model card
# ----------------------------------------------------------------------------


class ModelCard(BaseModel):
    """Per-model metadata referenced from stage cards."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    name: str
    provider: Literal["huggingface", "torch_hub", "onnx", "github", "internal", "other"]
    repo_id: str | None = None
    revision: str | None = None
    description: str = ""
    license: LicenseKind = LicenseKind.UNKNOWN
    commercial_safe: bool = True
    requires_token_env: str | None = None
    download_size_mb: float | None = None
    expected_input_sample_rate: int | None = None
    expected_input_channels: int | None = None
    expected_input_format: str | None = None
    output_schema: dict[str, str] = Field(default_factory=dict)
    benchmarks: dict[str, float] = Field(default_factory=dict)
    languages: list[str] | None = None
    tags: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Run card
# ----------------------------------------------------------------------------


class RunStageRecord(BaseModel):
    """Per-stage rolled-up stats from one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    stage_name: str
    target: str
    cardinality: Cardinality
    tasks_in: int = 0
    tasks_out: int = 0
    elapsed_sec: float = 0.0
    cache_hit: bool = False
    checkpoint_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CriticReport(BaseModel):
    """Layer 5 output critic report."""

    model_config = ConfigDict(extra="forbid")

    deterministic_pass: bool = True
    deterministic_findings: list[str] = Field(default_factory=list)
    drop_rates_by_stage: dict[str, float] = Field(default_factory=dict)
    output_profile: DatasetProfile = Field(default_factory=DatasetProfile)
    llm_used: bool = False
    llm_intent_alignment_score: float | None = None
    llm_findings: list[str] = Field(default_factory=list)


class RunCard(BaseModel):
    """Emitted by every pipeline run; the contract for replay and the eval gate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    user_prompt: str | None = None
    intent_categories: dict[str, Any] | None = None
    input_dataset_card: DatasetCard | None = None
    output_dataset_card: DatasetCard | None = None
    pipeline_ir_path: str | None = None
    compiled_yaml_path: str | None = None
    executor: Literal["xenna", "ray_data", "ray_actor_pool"] = "xenna"
    stage_records: list[RunStageRecord] = Field(default_factory=list)
    critic: CriticReport | None = None
    total_elapsed_sec: float = 0.0
    success: bool = False
    failure_reason: str | None = None
    cache_dir: str | None = None
    target_dir: str
    env: dict[str, str] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Convenience IO helpers
# ----------------------------------------------------------------------------


def load_stage_card(path: str | Path) -> StageCard:
    """Load a stage card from a YAML or JSON file."""

    import yaml

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return StageCard.model_validate(data)


def dump_stage_card(card: StageCard, path: str | Path) -> None:
    """Write a stage card to YAML, preserving the canonical ordering."""

    import yaml

    Path(path).write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )


__all__ = [
    "COMMERCIAL_OK",
    "Cardinality",
    "CapabilityTag",
    "CriticReport",
    "DatasetCard",
    "DatasetProfile",
    "IOSpec",
    "InputShape",
    "LicenseKind",
    "Modality",
    "ModelCard",
    "ModelRef",
    "OutputShape",
    "PHASE_ORDER",
    "ParamSpec",
    "Phase",
    "ResourceSpec",
    "RunCard",
    "RunStageRecord",
    "StageCard",
    "StageCategory",
    "dump_stage_card",
    "load_stage_card",
]
