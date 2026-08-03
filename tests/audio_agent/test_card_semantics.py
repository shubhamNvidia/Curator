# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape checks for advisory card semantics.

These tests intentionally do not assert a vocabulary for scope or meaning.  The
core makes the prose retrievable; the host LLM remains responsible for applying
it to user intent.
"""

from nemo_curator.audio_agent.card_conformance import (
    _semantic_fact_violations,
    check_card,
)


def test_semantic_facts_accept_compact_and_rich_prose() -> None:
    assert not _semantic_fact_violations(
        "ExampleStage",
        {
            "score": "A compact factual statement.",
            "duration": {
                "meaning": "Duration of the original recording.",
                "unit": "seconds",
                "provenance": "Computed from the decoded source.",
                "scope": "original recording, copied to emitted children",
                "propagation": "Copied unchanged across fan-out.",
                "counterexamples": ["It is not recomputed for each child."],
            },
        },
    )


def test_semantic_facts_reject_unreadable_shapes_without_interpreting_meaning() -> None:
    violations = _semantic_fact_violations(
        "ExampleStage",
        {
            "": "missing anchor",
            "score": {"meaning": "", "counterexamples": "not-a-list"},
            "duration": 3,
        },
    )
    assert any("keys must be non-empty" in item for item in violations)
    assert any(".meaning must be a non-empty string" in item for item in violations)
    assert any(".counterexamples must be a non-empty list" in item for item in violations)
    assert any("semantic_facts['duration'] must be prose or a mapping" in item for item in violations)


def test_semantic_facts_require_an_honest_evidence_tier() -> None:
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": {"params": "mechanical"},
            "semantic_facts": {"duration": "Duration of the selected audio."},
        },
    )

    assert any("verified.semantic_facts" in item for item in violations)


def test_verified_trust_metadata_must_be_a_mapping() -> None:
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": "mechanical",
            "semantic_facts": {"duration": "Duration of the selected audio."},
        },
    )

    assert any("verified must be a mapping" in item for item in violations)
