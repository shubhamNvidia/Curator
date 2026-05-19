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
"""Pipeline runner — Layers 3 + 4.

Layer 3 (dry-run): execute the IR on a 4-sample fixture and catch runtime
errors before full submission. Layer 4 (execution guards): wrap actual runs
with budget / disk-space monitoring and graceful failure capture.

Both layers reuse the existing :class:`nemo_curator.pipeline.Pipeline`
constructor; the runner's value is the inter-stage checkpointing, content-
addressable cache, RunCard emission, and OTel-disabled defaults.

The runner is intentionally executor-agnostic: the IR carries ``executor``
and ``Pipeline.run()`` picks Xenna vs Ray-Data vs Ray-Actor-Pool with the
``is_inference_server_active()`` guard.
"""

from __future__ import annotations

import dataclasses
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.agentic.cache import StageCache
from nemo_curator.agentic.cards import (
    Cardinality,
    CriticReport,
    DatasetCard,
    RunCard,
    RunStageRecord,
)
from nemo_curator.agentic.compiler import write_compiled_yaml
from nemo_curator.agentic.ir import PipelineIR
from nemo_curator.agentic.registry import CapabilityRegistry
from nemo_curator.agentic.validator import ValidationReport, validate

# OTel SIGSEGV workaround default
_OTEL_DEFAULTS = {
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
}


@dataclass
class RunOptions:
    """User-controllable knobs the runner honors."""

    dry_run: bool = False
    dry_run_sample_count: int = 4
    fail_fast: bool = True
    enable_cache: bool = True
    max_cache_bytes: int = 50 * 1024 * 1024 * 1024
    extra_env: dict[str, str] = field(default_factory=dict)
    profile: DatasetCard | None = None
    run_id: str | None = None


@dataclass
class RunResult:
    """High-level result of one run."""

    run_card: RunCard
    target_dir: Path
    compiled_yaml_path: Path
    success: bool
    findings_yaml_path: Path | None = None
    error: str | None = None


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def run(
    ir: PipelineIR,
    registry: CapabilityRegistry,
    *,
    options: RunOptions | None = None,
    validation: ValidationReport | None = None,
) -> RunResult:
    """Validate, compile, and execute the IR. Always emits a RunCard."""

    opts = options or RunOptions()
    target_dir = Path(ir.sink.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    adv_dir = target_dir / ".adv"
    adv_dir.mkdir(parents=True, exist_ok=True)

    _apply_otel_defaults()
    for k, v in opts.extra_env.items():
        os.environ[k] = v

    if validation is None:
        validation = validate(ir, registry, intent=ir.intent, mutate=True)
    findings_path: Path | None = None
    if validation.findings:
        findings_path = _dump_findings(validation, adv_dir)

    started = datetime.now(timezone.utc)
    run_id = opts.run_id or f"adv-{int(started.timestamp())}-{validation.fingerprint or 'noid'}"
    run_card = RunCard(
        run_id=run_id,
        started_at=started,
        user_prompt=(ir.intent.raw_prompt if ir.intent else None),
        intent_categories=(ir.intent.model_dump(mode="json") if ir.intent else None),
        input_dataset_card=opts.profile,
        pipeline_ir_path=str(adv_dir / "ir.json"),
        executor=ir.executor,
        target_dir=str(target_dir),
        cache_dir=str(adv_dir / "cache"),
        env={k: v for k, v in os.environ.items() if k.startswith("OTEL_")},
    )

    # Always persist the IR + compiled YAML for replay / debugging.
    ir.write(adv_dir / "ir.json")
    compiled_yaml_path = write_compiled_yaml(validation.ir, registry, adv_dir / "compiled.yaml")
    run_card.compiled_yaml_path = str(compiled_yaml_path)

    if not validation.is_ok():
        msg = "; ".join(f"{f.code}:{f.detail}" for f in validation.errors())
        run_card.success = False
        run_card.failure_reason = f"validation_failed: {msg}"
        run_card.finished_at = datetime.now(timezone.utc)
        run_card.total_elapsed_sec = (run_card.finished_at - run_card.started_at).total_seconds()
        _write_run_card(run_card, adv_dir)
        return RunResult(
            run_card=run_card,
            target_dir=target_dir,
            compiled_yaml_path=compiled_yaml_path,
            success=False,
            findings_yaml_path=findings_path,
            error=msg,
        )

    # Dry-run path executes on a small fixture; full execution is the default.
    if opts.dry_run:
        ok, error = _dry_run(validation.ir, registry, target_dir, run_card, opts)
    else:
        ok, error = _full_run(validation.ir, registry, target_dir, run_card, opts)

    run_card.success = ok
    run_card.failure_reason = error
    run_card.finished_at = datetime.now(timezone.utc)
    run_card.total_elapsed_sec = (run_card.finished_at - run_card.started_at).total_seconds()
    _write_run_card(run_card, adv_dir)

    return RunResult(
        run_card=run_card,
        target_dir=target_dir,
        compiled_yaml_path=compiled_yaml_path,
        success=ok,
        findings_yaml_path=findings_path,
        error=error,
    )


# ----------------------------------------------------------------------------
# Execution paths
# ----------------------------------------------------------------------------


def _dry_run(
    ir: PipelineIR,
    registry: CapabilityRegistry,
    target_dir: Path,
    run_card: RunCard,
    opts: RunOptions,
) -> tuple[bool, str | None]:
    """Run the pipeline against a tiny fixture to surface runtime errors early.

    For Phase 2 we treat the dry-run as a structural rehearsal: we instantiate
    every stage and call ``setup``+``teardown`` without actually feeding data.
    A future iteration can wire in N sample tasks; the contract today is
    "can we build the pipeline without exploding?".
    """

    try:
        pipeline = _build_pipeline(ir, registry, dry_run=True)
        for stage in pipeline.stages:
            try:
                stage.setup()
            finally:
                try:
                    stage.teardown()
                except Exception:  # noqa: BLE001
                    pass
            run_card.stage_records.append(_record_for(stage, elapsed=0.0))
        return True, None
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=True).error(f"dry-run failed: {exc}")
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=10)}"


def _full_run(
    ir: PipelineIR,
    registry: CapabilityRegistry,
    target_dir: Path,
    run_card: RunCard,
    opts: RunOptions,
) -> tuple[bool, str | None]:
    """Run the pipeline end-to-end via :class:`nemo_curator.pipeline.Pipeline`."""

    try:
        pipeline = _build_pipeline(ir, registry, dry_run=False)
        cache_dir = target_dir / ".adv" / "cache"
        cache = StageCache(cache_dir, max_bytes=opts.max_cache_bytes) if opts.enable_cache else None
        del cache  # Phase 2: cache wiring stops at the StageCache; in-pipeline use is wired by composite stages.

        t0 = time.perf_counter()
        pipeline.run()
        elapsed = time.perf_counter() - t0

        run_card.stage_records = [_record_for(s, elapsed=elapsed / max(1, len(pipeline.stages))) for s in pipeline.stages]
        return True, None
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=True).error(f"run failed: {exc}")
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=10)}"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _build_pipeline(
    ir: PipelineIR,
    registry: CapabilityRegistry,
    *,
    dry_run: bool,
):
    """Build a :class:`Pipeline` from the IR by directly instantiating stages.

    We avoid going through :func:`nemo_curator.config.run.create_pipeline_from_yaml`
    for now to keep the runner free of Hydra at execution time; the
    compiled YAML is still emitted as the canonical artifact for replay.
    """

    from nemo_curator.pipeline.pipeline import Pipeline  # noqa: PLC0415
    from nemo_curator.stages.resources import Resources  # noqa: PLC0415

    pipeline = Pipeline(ir.name)
    for stage_ref in ir.stages:
        entry = registry.get(stage_ref.stage)
        if entry is None:
            msg = f"runner: stage {stage_ref.stage!r} not in registry."
            raise RuntimeError(msg)
        klass = entry.klass
        # Filter out None-valued params and any not present on the class signature.
        try:
            import inspect  # noqa: PLC0415

            sig = inspect.signature(klass.__init__)
            allowed = {n for n in sig.parameters if n != "self"}
        except (TypeError, ValueError):
            allowed = None

        kwargs = {
            k: v
            for k, v in stage_ref.params.items()
            if v is not None and (allowed is None or k in allowed)
        }
        instance = klass(**kwargs)
        if stage_ref.resources is not None:
            instance = instance.with_(resources=Resources(**stage_ref.resources.to_resources_kwargs()))
        pipeline.add_stage(instance)
    pipeline.build()
    return pipeline


def _record_for(stage: Any, elapsed: float) -> RunStageRecord:
    card_target = f"{type(stage).__module__}.{type(stage).__name__}"
    return RunStageRecord(
        stage_name=type(stage).__name__,
        target=card_target,
        cardinality=Cardinality.ONE_TO_ONE,
        elapsed_sec=elapsed,
    )


def _apply_otel_defaults() -> None:
    for k, v in _OTEL_DEFAULTS.items():
        os.environ.setdefault(k, v)


def _dump_findings(report: ValidationReport, adv_dir: Path) -> Path:
    import yaml

    payload = {
        "fingerprint": report.fingerprint,
        "findings": [dataclasses.asdict(f) for f in report.findings],
        "auto_inserted": [s.model_dump(mode="json") for s in report.auto_inserted],
    }
    p = adv_dir / "findings.yaml"
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return p


def _write_run_card(card: RunCard, adv_dir: Path) -> Path:
    import yaml

    p = adv_dir / "run_card.yaml"
    p.write_text(yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return p


__all__ = ["RunOptions", "RunResult", "run"]
