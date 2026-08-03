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

    def test_strips_secret_values_embedded_in_error_strings(self) -> None:
        out = _safety.redact(
            {
                "reason": (
                    "credential=plain-value "
                    "HF_TOKEN=plain-token "
                    "AWS_ACCESS_KEY_ID=plain-access "
                    "Authorization: Bearer bearer-value"
                )
            }
        )
        reason = out["reason"]
        assert "plain-value" not in reason
        assert "plain-token" not in reason
        assert "plain-access" not in reason
        assert "bearer-value" not in reason
        assert reason.count("<redacted-secret>") == 4

    def test_strips_quoted_json_secret_assignments_and_multiword_values(self) -> None:
        redacted = _safety.redact_secret_text(
            '{"api_key": "sk-demo-secret", "password": "two words secret"}'
        )
        assert "sk-demo-secret" not in redacted
        assert "two words secret" not in redacted
        assert redacted.count("<redacted-secret>") == 2

    def test_strips_basic_auth_url_userinfo_and_jwt(self) -> None:
        basic = "dXNlcjpwYXNzd29yZA=="
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        redacted = _safety.redact_secret_text(
            f"Authorization: Basic {basic}; "
            "registry=https://alice:correct-horse@example.test/v2; "
            f"assertion {jwt}"
        )
        assert basic not in redacted
        assert "alice:correct-horse" not in redacted
        assert jwt not in redacted
        assert "https://<redacted-secret>@example.test/v2" in redacted
        assert redacted.count("<redacted-secret>") == 3

    def test_strips_conservative_standalone_token_prefixes(self) -> None:
        # Assembled from split literals so these FAKE fixtures never appear as a
        # contiguous token in source (GitHub secret-scanning push protection matches the
        # xoxb-/ghp_/AKIA/sk-proj- prefixes). The runtime strings are unchanged, so the
        # redaction coverage is identical.
        secrets = (
            "sk-" + "proj-abcdefghijklmnopqrstuv",
            "ghp" + "_abcdefghijklmnopqrstuvwxyz123456",
            "AKIA" + "ABCDEFGHIJKLMNOP",
            "xoxb" + "-123456789012-abcdefghijklmnop",
        )
        redacted = _safety.redact_secret_text(" ".join(secrets))
        assert all(secret not in redacted for secret in secrets)
        assert redacted.count("<redacted-secret>") == len(secrets)

    def test_keeps_noncredential_basic_text_and_short_prefixed_terms(self) -> None:
        text = "basic authentication failed while importing sk-learn and checking ghp_status"
        assert _safety.redact_secret_text(text) == text


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

    def test_file_uri_is_still_a_local_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        inside = (tmp_path / "data.jsonl").as_uri()
        assert _safety.path_violations([inside]) == []
        assert any("passwd" in v for v in _safety.path_violations(["file:///etc/passwd"]))

    def test_local_uri_alias_is_still_a_local_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        inside = f"local://{tmp_path}/data.jsonl"
        assert _safety.path_violations([inside]) == []
        assert any("passwd" in v for v in _safety.path_violations(["local:///etc/passwd"]))

    def test_recipe_path_params_flattens_list_valued_sources(self) -> None:
        stage = type("Stage", (), {"params": {"manifest_path": ["a.jsonl", "b.jsonl"]}})()
        recipe = type("Recipe", (), {"stages": [stage]})()
        assert _safety.recipe_path_params(recipe) == ["a.jsonl", "b.jsonl"]

    def test_recipe_path_params_ignores_semantic_path_key_fields(self) -> None:
        stage = type(
            "Stage",
            (),
            {
                "params": {
                    "audio_filepath_key": "audio_filepath",
                    "audio_path_resolution": "relative",
                    "split_filepaths_key": "split_filepaths",
                    "model_path": "nvidia/model-name",
                    "manifest_path": "inside.jsonl",
                }
            },
        )()
        recipe = type("Recipe", (), {"stages": [stage]})()
        assert _safety.recipe_path_params(recipe) == ["inside.jsonl"]


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
