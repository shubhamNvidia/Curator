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

"""Auto-derivation of agent-facing stage metadata.

Turns a stage's constructor surface (dataclass fields or ``__init__`` signature)
into a list of :class:`~nemo_curator.stages.audio._agent_ready.ParamSpec`, and
assembles the full discovery contract by merging those params and semantic key
roles into the stage's hand-written ``describe()``. Stages therefore never list
params by hand:

    contract = build_contract(stage)         # dynamic, resolved key values
    contract = SomeStage.describe_static()   # static, instance-free (planning)
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import re
import sys
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

from nemo_curator.stages.audio._agent_ready import (
    ParamSpec,
    StageContract,
    StaticHints,
)
from nemo_curator.stages.audio._roles import (
    LITERAL_KEY_ROLES,
    role_for_field,
    role_overrides_for,
)

# Framework-level constructor fields that are not stage-semantic knobs.
EXCLUDED_PARAM_NAMES = frozenset({"name", "resources", "batch_size", "runtime_env", "num_workers"})

_MISSING = dataclasses.MISSING


# --------------------------------------------------------------------------- #
# Type rendering / Literal handling
# --------------------------------------------------------------------------- #
def _module_globals(obj: Any) -> dict[str, Any]:  # noqa: ANN401
    mod = sys.modules.get(getattr(obj, "__module__", "") or "")
    return getattr(mod, "__dict__", {})


def _resolve_hint(raw: Any, globalns: dict[str, Any]) -> Any:  # noqa: ANN401
    """Resolve a possibly-stringized annotation to a type object, best-effort.

    ``from __future__ import annotations`` makes every annotation a string; we
    eval it in the owning module's namespace. Heavy/forward refs that fail to
    resolve are kept as their raw string (rendered verbatim, no Literal/choices).
    """
    if raw is None or raw is inspect.Parameter.empty:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return eval(raw, dict(globalns))  # noqa: S307 - annotations come from our own source
    except Exception:  # noqa: BLE001
        return raw


def _scalar_name(tp: type) -> str:
    return {int: "int", float: "float", str: "str", bool: "bool"}.get(tp, getattr(tp, "__name__", str(tp)))


def _render_type(hint: Any) -> str:  # noqa: ANN401
    if hint is None or hint is inspect.Parameter.empty:
        return "Any"
    if isinstance(hint, str):
        return hint
    origin = get_origin(hint)
    if origin is Literal:
        elem = {type(a).__name__ for a in get_args(hint)}
        if elem == {"str"}:
            return "str"
        if elem == {"int"}:
            return "int"
        if elem <= {"int", "float"}:
            return "float"
        return "str"
    if origin is Union or origin is getattr(types, "UnionType", ()):
        args = get_args(hint)
        nullable = any(a is type(None) for a in args)
        inner = [_render_type(a) for a in args if a is not type(None)]
        rendered = " | ".join(inner) if inner else "Any"
        return f"{rendered} | None" if nullable else rendered
    if origin in (list, typing.List):  # noqa: UP006
        sub = get_args(hint)
        return f"list[{_render_type(sub[0])}]" if sub else "list"
    if origin in (dict, typing.Dict):  # noqa: UP006
        return "dict"
    if origin in (tuple, typing.Tuple):  # noqa: UP006
        return "tuple"
    if isinstance(hint, type):
        return _scalar_name(hint)
    return str(hint)


def _literal_choices(hint: Any) -> list[Any] | None:  # noqa: ANN401
    if isinstance(hint, str) or hint is None:
        return None
    if get_origin(hint) is Literal:
        return list(get_args(hint))
    if get_origin(hint) is Union or get_origin(hint) is getattr(types, "UnionType", ()):
        for a in get_args(hint):
            if get_origin(a) is Literal:
                return list(get_args(a))
    return None


# --------------------------------------------------------------------------- #
# Docstring Args parsing (single maintained source for param descriptions)
# --------------------------------------------------------------------------- #
_ARG_HDR = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$")
_SECTION_HDR = re.compile(
    r"^\s*(Returns?|Raises?|Yields?|Notes?|Examples?|Attributes?|See Also|Warning|Todo)\s*:\s*$"
)
_ARG_LINE = re.compile(r"^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*)\s*(\([^)]*\))?\s*:\s*(?P<desc>.*)$")


def _parse_args_section(doc: str) -> dict[str, str]:
    lines = doc.splitlines()
    out: dict[str, str] = {}
    in_args = False
    current: str | None = None
    arg_indent = 0
    for line in lines:
        if _ARG_HDR.match(line):
            in_args = True
            current = None
            continue
        if not in_args:
            continue
        if _SECTION_HDR.match(line):
            break
        if not line.strip():
            continue
        m = _ARG_LINE.match(line)
        if m and (current is None or len(m.group("indent")) <= arg_indent + 1 or len(m.group("indent")) <= 8):  # noqa: PLR2004
            current = m.group("name")
            arg_indent = len(m.group("indent"))
            out[current] = m.group("desc").strip()
        elif current is not None:
            out[current] = (out[current] + " " + line.strip()).strip()
    return out


def _docstring_arg_descriptions(cls: type) -> dict[str, str]:
    out: dict[str, str] = {}
    for klass in reversed(cls.__mro__):  # base first, so subclass docstrings win
        doc = klass.__dict__.get("__doc__")
        if doc:
            out.update(_parse_args_section(doc))
    return out


# --------------------------------------------------------------------------- #
# Param derivation
# --------------------------------------------------------------------------- #
def _as_class(stage_or_cls: Any) -> type:  # noqa: ANN401
    return stage_or_cls if isinstance(stage_or_cls, type) else type(stage_or_cls)


def _call_factory(factory: Any) -> tuple[Any, bool]:  # noqa: ANN401
    try:
        return factory(), False
    except Exception:  # noqa: BLE001
        return "<unresolved default>", False


def _dataclass_params(cls: type, descriptions: dict[str, str]) -> list[ParamSpec]:
    globalns = _module_globals(cls)
    params: list[ParamSpec] = []
    for f in dataclasses.fields(cls):
        if not f.init or f.name in EXCLUDED_PARAM_NAMES or f.name.startswith("_"):
            continue
        if f.default is not _MISSING:
            default, required = f.default, False
        elif f.default_factory is not _MISSING:
            default, required = _call_factory(f.default_factory)
        else:
            default, required = None, True
        hint = _resolve_hint(f.type, globalns)
        params.append(
            ParamSpec(
                name=f.name,
                type=_render_type(hint),
                default=default,
                required=required,
                choices=_literal_choices(hint),
                description=descriptions.get(f.name),
                role=role_for_field(f.name) if f.name.endswith("_key") else None,
            )
        )
    return params


def _init_params(cls: type, descriptions: dict[str, str]) -> list[ParamSpec]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    globalns = _module_globals(cls)
    params: list[ParamSpec] = []
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in EXCLUDED_PARAM_NAMES or name.startswith("_"):
            continue
        required = p.default is inspect.Parameter.empty
        default = None if required else p.default
        hint = _resolve_hint(p.annotation, globalns)
        params.append(
            ParamSpec(
                name=name,
                type=_render_type(hint),
                default=default,
                required=required,
                choices=_literal_choices(hint),
                description=descriptions.get(name),
                role=role_for_field(name) if name.endswith("_key") else None,
            )
        )
    return params


def stage_params(stage_or_cls: Any) -> list[ParamSpec]:  # noqa: ANN401
    """Auto-derive the configurable parameters of a stage.

    Dataclass stages use :func:`dataclasses.fields`; plain ``__init__`` stages
    use the constructor signature. ``Literal[...]`` annotations become
    ``choices``; per-param descriptions come from the class docstring ``Args:``
    section; ``*_key`` params get a semantic ``role``.
    """
    cls = _as_class(stage_or_cls)
    descriptions = _docstring_arg_descriptions(cls)
    if dataclasses.is_dataclass(cls):
        return _dataclass_params(cls, descriptions)
    return _init_params(cls, descriptions)


# --------------------------------------------------------------------------- #
# Contract assembly
# --------------------------------------------------------------------------- #
def _derived_dispatch(cls: type, declared: str) -> str:
    """Drive ``dispatch`` from the framework's batch-support truth source."""
    from nemo_curator.stages.base import ProcessingStage

    if declared in ("process", "process_batch"):
        return declared
    overrides_batch = getattr(cls, "process_batch", None) is not ProcessingStage.process_batch
    return "process_batch" if overrides_batch else "process"


def _task_type_name(t: Any) -> str | None:  # noqa: ANN401
    """The class name of a generic arg, or None for a TypeVar/non-type."""
    return t.__name__ if isinstance(t, type) else None


def _task_types(cls: type) -> tuple[str | None, str | None]:
    """``(accepts, produces)`` task-type names from the ``ProcessingStage[X, Y]`` generic.

    Walks the MRO's ``__orig_bases__`` for the parametrized ProcessingStage base
    (e.g. ``ProcessingStage[AudioTask, DocumentBatch]``). Returns ``(None, None)``
    when unparametrized (bare TypeVars) or not found — those become ``uncertain``
    at the task-type check rather than a false mismatch.
    """
    from nemo_curator.stages.base import ProcessingStage

    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()) or ():
            origin = get_origin(base)
            if origin is None:
                continue
            try:
                is_ps = origin is ProcessingStage or (isinstance(origin, type) and issubclass(origin, ProcessingStage))
            except TypeError:
                is_ps = False
            if not is_ps:
                continue
            args = get_args(base)
            if len(args) == 2:  # noqa: PLR2004 - ProcessingStage[X, Y] has exactly two type args
                return _task_type_name(args[0]), _task_type_name(args[1])
    return None, None


def _first_doc_line(cls: type) -> str | None:
    doc = inspect.getdoc(cls)
    if not doc:
        return None
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return None


def _contract_referenced_keys(contract: StageContract) -> set[str]:
    keys: set[str] = set()
    for spec in [contract.reads, contract.writes, *contract.reads_one_of]:
        keys.update(spec.data_keys)
        keys.update(spec.segment_data_keys)
    keys.update(contract.metadata_reads)
    keys.update(contract.metadata_writes)
    for conditional in contract.conditional_writes:
        keys.update(conditional.writes.data_keys)
        keys.update(conditional.writes.segment_data_keys)
        keys.update(conditional.metadata_writes)
    return keys


def _key_attr_names(stage: Any) -> list[str]:  # noqa: ANN401
    return [a for a in dir(stage) if a.endswith("_key") and not a.startswith("__")]


def _resolve_key_roles(stage: Any, contract: StageContract) -> dict[str, str]:  # noqa: ANN401
    """Map resolved key *values* referenced by the contract to semantic roles."""
    cls = _as_class(stage)
    overrides = role_overrides_for(cls)
    roles: dict[str, str] = {}
    # 1. From *_key attributes on the instance (their resolved values).
    if not isinstance(stage, type):
        for attr in _key_attr_names(stage):
            with contextlib.suppress(Exception):
                val = getattr(stage, attr)
                if isinstance(val, str) and val:
                    role = overrides.get(attr) or role_for_field(attr)
                    if role != "unknown":
                        roles[val] = role
    # 2. Literal-default fallback for producer keys with no *_key field.
    for key in _contract_referenced_keys(contract):
        if key not in roles:
            literal = LITERAL_KEY_ROLES.get(key)
            if literal is not None:
                roles[key] = literal
    return roles


def build_contract(stage: Any) -> StageContract:  # noqa: ANN401
    """Return ``stage.describe()`` enriched with auto-derived params + key roles.

    Hand-written ``params`` in ``describe()`` override auto-derived ones.
    ``dispatch`` is derived from batch support; ``batch_only``/``stage_id`` are
    filled when unset. The single entry point used by catalog + serialization.
    """
    base = stage.describe()
    derived = stage_params(stage)
    if base.params:
        by_name = {p.name: p for p in derived}
        by_name.update({p.name: p for p in base.params})
        params = list(by_name.values())
    else:
        params = derived
    key_roles = _resolve_key_roles(stage, base) or dict(base.key_roles)
    cls = _as_class(stage)
    accepts_tt, produces_tt = _task_types(cls)
    return dataclasses.replace(
        base,
        params=params,
        key_roles=key_roles,
        dispatch=_derived_dispatch(cls, base.dispatch),
        batch_only=base.batch_only or bool(getattr(cls, "BATCH_ONLY", False)),
        stage_id=base.stage_id or cls.__name__,
        description=base.description or _first_doc_line(cls),
        accepts_task_type=base.accepts_task_type or accepts_tt,
        produces_task_type=base.produces_task_type or produces_tt,
    )


def _static_key_roles(cls: type) -> dict[str, str]:
    """Key roles from ``*_key`` field DEFAULT values (no instantiation)."""
    overrides = role_overrides_for(cls)
    roles: dict[str, str] = {}
    for p in stage_params(cls):
        if not p.name.endswith("_key") or not isinstance(p.default, str) or not p.default:
            continue
        role = overrides.get(p.name) or role_for_field(p.name)
        if role != "unknown":
            roles[p.default] = role
    return roles


def static_contract(cls: type) -> StageContract:
    """Instance-free discovery contract (params, roles, gates, dispatch, hints).

    Does not run ``__init__``; safe for stages with required constructor args.
    Reads/writes/cardinality are NOT resolved here — use :func:`build_contract`
    on an instance for those.
    """
    hints: StaticHints = getattr(cls, "AGENT_STATIC", None) or StaticHints()
    accepts_tt, produces_tt = _task_types(cls)
    return StageContract(
        contract_resolution="static_params_and_hints",
        cardinality_options=list(hints.cardinality_options),
        gates=hints.gates,
        dispatch=_derived_dispatch(cls, hints.dispatch),
        error_policy=hints.error_policy,
        description=hints.description or _first_doc_line(cls),
        stage_id=hints.stage_id or cls.__name__,
        params=stage_params(cls),
        key_roles=_static_key_roles(cls),
        batch_only=bool(getattr(cls, "BATCH_ONLY", False)),
        accepts_task_type=accepts_tt,
        produces_task_type=produces_tt,
    )
