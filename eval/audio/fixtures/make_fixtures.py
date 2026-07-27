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

"""Generate tiny self-contained audio + manifests for the opt-in E2E smoke.

Writes short mono sine WAVs (stdlib `wave`, no numpy/soundfile needed) and
rewrites the tiny manifests with ABSOLUTE paths so `AUDIO_AGENT_EVAL_EXECUTE=1`
can run a real (model-free) duration smoke without any external dataset.

    python eval/audio/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
import math
import os
import struct
import wave

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUDIO = os.path.join(_HERE, "audio")
_MANIFESTS = os.path.join(_HERE, "manifests")


def _write_sine(path: str, *, seconds: float = 1.0, sr: int = 16000, freq: float = 220.0) -> None:
    n = int(seconds * sr)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * (i / sr)))) for i in range(n)
        )
        w.writeframes(frames)


def main() -> int:
    os.makedirs(_AUDIO, exist_ok=True)
    os.makedirs(_MANIFESTS, exist_ok=True)

    clips = []
    for i, freq in enumerate((220.0, 330.0, 440.0)):
        p = os.path.join(_AUDIO, f"clip{i}.wav")
        _write_sine(p, seconds=1.0 + 0.5 * i, sr=16000, freq=freq)
        clips.append(p)
    longform = os.path.join(_AUDIO, "longform0.wav")
    _write_sine(longform, seconds=8.0, sr=16000, freq=180.0)  # "long-ish" for the smoke

    texts = ["the quick brown fox", "jumps over the lazy dog", "hello world this is a test"]
    with open(os.path.join(_MANIFESTS, "tiny_with_transcripts.jsonl"), "w", encoding="utf-8") as f:
        for p, t in zip(clips, texts):
            f.write(json.dumps({"audio_filepath": p, "text": t}) + "\n")
    with open(os.path.join(_MANIFESTS, "tiny_no_transcripts.jsonl"), "w", encoding="utf-8") as f:
        for p in clips:
            f.write(json.dumps({"audio_filepath": p}) + "\n")

    print(f"[make_fixtures] wrote {len(clips)} clips + longform to {_AUDIO}")
    print("[make_fixtures] rewrote tiny_with_transcripts.jsonl / tiny_no_transcripts.jsonl with absolute paths")
    print("Now run: AUDIO_AGENT_EVAL_EXECUTE=1 python -m eval.audio.run_eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
