"""Dump signatures of every audio ProcessingStage. Helper for stage card backfill."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import importlib
    import pkgutil

    import nemo_curator.stages.audio as audio_pkg

    for _, name, _ in pkgutil.walk_packages(audio_pkg.__path__, prefix=audio_pkg.__name__ + "."):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            print(f"# skip {name}: {exc}", file=sys.stderr)

    from nemo_curator.stages.base import _STAGE_REGISTRY, ProcessingStage  # noqa: PLC0415

    audio_classes = [
        c for n, c in sorted(_STAGE_REGISTRY.items())
        if "nemo_curator.stages.audio" in (c.__module__ or "")
    ]

    for klass in audio_classes:
        is_composite = any(b.__name__ == "CompositeStage" for b in klass.__mro__)
        kind = "Composite" if is_composite else "Stage"
        print(f"## {klass.__name__}  ({kind})")
        print(f"target: {klass.__module__}.{klass.__name__}")
        try:
            sig = inspect.signature(klass.__init__)
        except (TypeError, ValueError):
            sig = None
        if sig:
            print(f"signature: {sig}")
        attrs = {a: getattr(klass, a, None) for a in ("name", "resources", "batch_size", "runtime_env")}
        print(f"class-attrs: {attrs}")
        for m in ("inputs", "outputs"):
            if m in klass.__dict__:
                src = inspect.getsource(klass.__dict__[m])
                print(f"{m}:\n{src}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
