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


def test_a_card_key_nobody_reads_is_a_violation_not_a_shrug() -> None:
    """An unknown top-level key fails silently in the worst way: no error anywhere, the card
    still passes conformance, and its content simply never reaches the host critic. Two shipped
    cards wrote ``gotchas`` and ``relationships`` for what the readers call ``counterexamples``
    and ``comparison``, so the disambiguation prose written to stop a stage being confused with
    its neighbour was read by nobody at all.
    """
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": {"params": "mechanical"},
            "gotchas": ["what the readers call counterexamples"],
            "relationships": {"OtherStage": "what the readers call comparison"},
        },
    )

    assert any("unknown top-level field 'gotchas'" in item for item in violations)
    assert any("unknown top-level field 'relationships'" in item for item in violations)


def test_a_row_dropping_stage_cannot_ship_a_card_that_omits_the_filter_tag() -> None:
    """The contract is the stricter statement and the one an author writes alone, having just
    made the stage drop rows. Without the tag nothing assembling a recipe knows it can, so a
    stage silently discarding most of a corpus reads as a pass-through exactly where the
    decision to include it is made.
    """
    violations = check_card(
        "SampleRateFilterStage",
        {
            "category": "preprocess",
            "summary": "Selects rows by sample rate.",
            "verified": {"params": "mechanical"},
            "tags": [],
        },
    )

    assert any("cardinality='filter'" in item and "is_filter" in item for item in violations)


def test_filtering_within_a_row_is_not_required_to_claim_a_row_cardinality() -> None:
    """The tag is the broader planner-facing notion. ``OverlapFilterStage`` shrinks a segment
    list while every row survives, so tag-without-cardinality is a correct pairing; demanding
    the converse would make it declare a row cardinality it does not have.
    """
    violations = check_card(
        "OverlapFilterStage",
        {
            "category": "preprocess",
            "summary": "Drops overlapping segments within a row.",
            "verified": {"params": "mechanical"},
            "tags": ["is_filter"],
        },
    )

    assert not any("is_filter" in item for item in violations)


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


def test_a_stage_advertising_a_score_must_say_what_the_score_means() -> None:
    """The metrics checks validate a block only IF one is present, so a scorer with no block at
    all passed silently. ``BandwidthEstimationStage`` shipped ``produces_score`` with no scale,
    range or direction; nothing downstream could then derive a comparison operator, and the
    generic filter the resolver emits for an annotate-only scorer is where an inverted filter
    comes from.
    """
    violations = check_card(
        "BandwidthEstimationStage",
        {
            "category": "quality",
            "summary": "Estimates effective bandwidth.",
            "verified": {"params": "mechanical"},
            "provenance": {"card_version": 1},
            "tags": ["produces_score"],
        },
    )

    assert any("produces_score" in item and "no metrics block" in item for item in violations)


def test_a_categorical_score_is_not_required_to_invent_a_numeric_direction() -> None:
    """The guard requires a metrics BLOCK, not a direction. ``BandFilterStage`` predicts
    full_band/narrow_band -- there is no 'higher is better' to declare -- so demanding a scale
    would force a card to state something untrue to pass a gate.
    """
    violations = check_card(
        "BandFilterStage",
        {
            "category": "quality",
            "summary": "Classifies bandwidth and filters.",
            "verified": {"params": "mechanical"},
            "provenance": {"card_version": 1},
            "tags": ["produces_score", "is_filter"],
            "metrics": {"band_prediction": {"threshold_param": "band_value"}},
        },
    )

    assert not any("produces_score" in item for item in violations)


def test_a_card_must_date_itself_so_a_stale_guess_is_distinguishable_from_a_fresh_one() -> None:
    """A card is read as current. Without ``provenance`` there is nothing to say when its
    ``best_guess`` facts were last checked against the code. 47 of 49 cards carried it by
    convention and the two that did not were simply the newest -- how an unenforced convention
    always fails.
    """
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": {"params": "mechanical"},
        },
    )

    assert any("missing required field 'provenance'" in item for item in violations)


def test_a_stage_shipping_without_a_card_fails_the_gate_unless_waived(monkeypatch) -> None:  # noqa: ANN001
    """An uncarded stage was reported and never failed, so a stage could ship plannable with no
    card semantics behind it: ``discover`` lists it, the planner may pick it, and the host critic
    gets no meaning, scope or counterexample to reason from. ``--allow-uncarded`` stays as the
    deliberate waiver for a work-in-progress stage.
    """
    from nemo_curator.audio_agent import card_conformance as cc

    monkeypatch.setattr(
        cc,
        "audit",
        lambda: {
            "violations": {},
            "orphan_cards": [],
            "uncarded_stages": ["AStageNobodyCarded"],
            "gate_unverified": [],
            "carded_count": 1,
            "stage_count": 2,
        },
    )
    monkeypatch.setattr(cc, "audit_blueprints", lambda: {"violations": {}, "blueprint_count": 0})

    assert cc.main([]) == 1
    assert cc.main(["--allow-uncarded"]) == 0
