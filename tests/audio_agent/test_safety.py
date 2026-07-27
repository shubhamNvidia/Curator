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

"""Unit tests for the deterministic safety guardrails (nemo_curator.audio_agent._safety)."""

import hashlib

from nemo_curator.audio_agent import _safety


class TestRedact:
    def test_strips_secrets_and_transcripts(self) -> None:
        obj = {
            "api_key": "sk-123",
            "hf_token": "hf_x",
            "text": "hello world",
            "score": 3.5,
            "nested": {"password": "p", "keep": 1},
            "list": [{"secret": "s"}],
        }
        out = _safety.redact(obj)
        assert out["api_key"] == "<redacted-secret>"
        assert out["hf_token"] == "<redacted-secret>"
        assert out["nested"]["password"] == "<redacted-secret>"
        assert out["nested"]["keep"] == 1
        assert out["list"][0]["secret"] == "<redacted-secret>"
        assert out["text"].startswith("<redacted-transcript:")
        assert out["score"] == 3.5

    def test_can_keep_transcripts(self) -> None:
        assert _safety.redact({"text": "hi"}, redact_transcripts=False)["text"] == "hi"


class TestWorkspaceLock:
    def test_off_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)
        assert _safety.workspace_root() is None
        assert _safety.path_violations(["/etc/passwd", "/anywhere/x.wav"]) == []

    def test_blocks_outside_allows_inside(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        inside = str(tmp_path / "data" / "a.wav")
        violations = _safety.path_violations([inside, "/etc/passwd"])
        assert any("passwd" in v for v in violations)
        assert all("a.wav" not in v for v in violations)

    def test_allows_remote_uris(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        assert _safety.path_violations(["s3://bucket/key", "http://h/x", None]) == []


class TestSmokeToken:
    def test_not_derivable_from_public_config_hash(self) -> None:
        """H2: the token must not be a plain hash of the (public) config_hash."""
        ch = "deadbeefcafe1234"
        plain = hashlib.sha256(f"audio_agent_smoke|{ch}".encode()).hexdigest()[:24]
        assert _safety.smoke_token(ch) != plain

    def test_roundtrip_verifies_and_rejects(self) -> None:
        ch = "abc123"
        tok = _safety.smoke_token(ch)
        assert _safety.verify_smoke_token(tok, ch) is True
        assert _safety.verify_smoke_token("wrong-token", ch) is False
        assert _safety.verify_smoke_token(tok, "other-hash") is False
        assert _safety.verify_smoke_token(None, ch) is False
        assert _safety.verify_smoke_token(tok, None) is False

    def test_secret_env_changes_token(self, monkeypatch) -> None:
        monkeypatch.setenv("AUDIO_AGENT_SMOKE_SECRET", "secret-a")
        _safety._smoke_secret.cache_clear()
        tok_a = _safety.smoke_token("h")
        monkeypatch.setenv("AUDIO_AGENT_SMOKE_SECRET", "secret-b")
        _safety._smoke_secret.cache_clear()
        tok_b = _safety.smoke_token("h")
        _safety._smoke_secret.cache_clear()
        assert tok_a != tok_b


class TestRequireSmoke:
    def test_env_toggle(self, monkeypatch) -> None:
        monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
        assert _safety.require_smoke() is False
        for truthy in ("1", "true", "YES", "on"):
            monkeypatch.setenv("AUDIO_AGENT_REQUIRE_SMOKE", truthy)
            assert _safety.require_smoke() is True
        monkeypatch.setenv("AUDIO_AGENT_REQUIRE_SMOKE", "0")
        assert _safety.require_smoke() is False
