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

"""Configuration-strategy resolver (1A.2): outcome -> concrete parameter.

Path A (knowledge-driven, the default and only path implemented here): resolve a
user-facing *outcome* to a concrete stage parameter using the card, so the agent
never exposes an internal threshold and never invents one. Three inputs, in
priority order:

* ``explicit`` — the user gave a value (``{param: value}``): used as-is.
* ``use_case`` — a named card ``preset`` (e.g. ``tts_reference``): the bundle applied.
* ``label`` — an *outcome label* (e.g. ``studio``): mapped via the card's
  ``metrics`` ``anchors`` to a concrete value — a threshold on the stage's own
  ``threshold_param`` (self-filtering stage), or, for an annotator, a
  ``PreserveByValueStage`` filter whose operator comes from the metric's
  ``direction`` (``higher_better`` -> ``ge``, ``lower_better`` -> ``le``).

Everything is generic: metric/param/label/direction are read from the card; there
are no metric names in the logic. **Path B (data-informed relative selection) is
deferred** behind the ``data_driven`` flag — a *relative* request with the flag
off returns an ``ask`` (get an explicit/absolute bar) rather than guessing.
"""

from __future__ import annotations

import re
from typing import Any

from nemo_curator.audio_agent.contracts import ConfigStrategyEntry

_OP_FOR_DIRECTION = {"higher_better": "ge", "lower_better": "le"}
_RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def resolve(
    stage_id: str,
    *,
    label: str | None = None,
    use_case: str | None = None,
    explicit: dict[str, Any] | None = None,
    data_driven: bool = False,
) -> dict[str, Any]:
    """Resolve an outcome to concrete config for ``stage_id`` (Path A).

    Returns ``{stage, params, filter_stage, strategy, asks}``:
    ``params`` set on the stage; ``filter_stage`` a ``{ref, params}`` to insert
    (annotator case); ``strategy`` the auditable :class:`ConfigStrategyEntry`
    records; ``asks`` any user-facing questions the host must resolve.
    """
    from nemo_curator.audio_agent.index import get_index

    card = get_index().card(stage_id) or {}
    metrics = card.get("metrics") or {}
    presets = card.get("presets") or {}
    out: dict[str, Any] = {"stage": stage_id, "params": {}, "filter_stage": None, "strategy": [], "asks": []}

    if explicit:
        for param, value in explicit.items():
            out["params"][param] = value
            out["strategy"].append(
                ConfigStrategyEntry(
                    param=param, value=value, kind="absolute", mode="knowledge_driven",
                    source={"from": "user_explicit"}, recompute_on="none",
                    rationale="explicit user value",
                ).to_dict()
            )
        return out

    if use_case:
        bundle = presets.get(use_case)
        if not isinstance(bundle, dict):
            out["asks"].append(f"unknown use_case {use_case!r}; known presets: {sorted(presets)}")
            return out
        for param, value in bundle.items():
            out["params"][param] = value
            out["strategy"].append(
                ConfigStrategyEntry(
                    param=param, value=value, kind="absolute", mode="knowledge_driven",
                    source={"from": "card_preset", "ref": use_case}, recompute_on="none",
                    rationale=f"card preset {use_case!r}",
                ).to_dict()
            )
        return out

    if label:
        _resolve_label(stage_id, label, metrics, out)
        return out

    out["asks"].append("provide one of: label (outcome), use_case (preset), or explicit ({param: value})")
    return out


def _resolve_label(stage_id: str, label: str, metrics: dict[str, Any], out: dict[str, Any]) -> None:
    """Map an outcome label to a value via a metric's anchors (numeric or categorical)."""
    for metric_key, mblock in metrics.items():
        if not isinstance(mblock, dict):
            continue
        for anchor_key, anchor_label in (mblock.get("anchors") or {}).items():
            if str(anchor_label).lower() != label.lower():
                continue
            scale = mblock.get("scale") or {}
            direction = scale.get("direction")
            numeric = ("min" in scale) or ("max" in scale) or bool(_RANGE_RE.match(str(anchor_key)))
            value: Any = _value_from_range(str(anchor_key), direction) if numeric else anchor_key

            tparam = mblock.get("threshold_param")
            if tparam:  # self-filtering stage: set its own threshold
                out["params"][tparam] = value
                out["strategy"].append(
                    ConfigStrategyEntry(
                        param=tparam, value=value, metric=metric_key, kind="absolute",
                        mode="knowledge_driven", source={"from": "card_anchor", "ref": label},
                        recompute_on="none", rationale=f"label {label!r} -> {metric_key}={value}",
                    ).to_dict()
                )
            else:  # annotator: emit a PreserveByValueStage filter, operator from direction
                op = _OP_FOR_DIRECTION.get(direction, "ge")
                out["filter_stage"] = {
                    "ref": "PreserveByValueStage",
                    "params": {"input_value_key": metric_key, "operator": op, "target_value": value},
                }
                out["strategy"].append(
                    ConfigStrategyEntry(
                        param="target_value", value=value, metric=metric_key, kind="absolute",
                        mode="knowledge_driven", source={"from": "card_anchor", "ref": label},
                        recompute_on="none",
                        rationale=f"label {label!r} -> filter {metric_key} {op} {value} (PreserveByValueStage)",
                    ).to_dict()
                )
            return

    known = sorted({str(lbl) for m in metrics.values() if isinstance(m, dict) for lbl in (m.get("anchors") or {}).values()})
    out["asks"].append(
        f"no anchor for label {label!r} on {stage_id}; known outcome labels: {known or '(none — this stage has no metric anchors)'}"
    )


def _value_from_range(range_key: str, direction: str | None) -> Any:  # noqa: ANN401
    """The concrete cutoff for a ``"lo-hi"`` anchor range given the metric direction.

    ``higher_better`` keeps values at/above the good range -> use its low bound;
    ``lower_better`` keeps values at/below -> use its high bound.
    """
    m = _RANGE_RE.match(range_key)
    if not m:
        return range_key
    lo, hi = float(m.group(1)), float(m.group(2))
    return hi if direction == "lower_better" else lo
