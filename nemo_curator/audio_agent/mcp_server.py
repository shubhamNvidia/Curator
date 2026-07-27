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

"""MCP adapter — exposes the audio-agent verbs as typed tools for MCP hosts.

Thin wrapper over the exact same deterministic core as the CLI/`verbs`, so
Claude/Cursor (or any MCP client) get native, typed tools. ``mcp`` is an optional
dependency; the module imports without it and ``main`` prints an install hint.

    pip install "mcp[cli]"
    python -m nemo_curator.audio_agent.mcp_server        # stdio server
"""

from __future__ import annotations

from typing import Any


def build_server() -> Any:  # noqa: ANN401 - returns a FastMCP instance
    """Construct and return the FastMCP server (imports ``mcp`` lazily)."""
    from mcp.server.fastmcp import FastMCP

    from nemo_curator import audio_agent as aa

    server = FastMCP("nemo-curator-audio-agent")

    @server.tool()
    def discover() -> dict[str, Any]:
        """List agent-ready audio stages with category and one-liner."""
        return aa.discover()

    @server.tool()
    def describe(name: str) -> dict[str, Any]:
        """Return the static contract (and card) for one stage."""
        return aa.describe(name)

    @server.tool()
    def catalog_tree() -> dict[str, Any]:
        """Return the L0 category tree for coarse-to-fine routing."""
        return aa.catalog_tree()

    @server.tool()
    def cards(category: str | None = None, names: list[str] | None = None) -> dict[str, Any]:
        """L1 one-liners for a category, or L2 full cards for named finalists."""
        return aa.cards(category=category, names=names)

    @server.tool()
    def context(
        goal: dict[str, Any] | None = None,
        data: str | None = None,
        stages: list[str] | None = None,
        roles: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble a PlanningContext (category tree + profile + env + blueprints)."""
        return aa.context(goal, data=data, stages=stages, roles=roles)

    @server.tool()
    def validate(
        recipe: dict[str, Any],
        data: str | None = None,
        expected_outputs: list[str] | None = None,
        acceptance_criteria: list[dict[str, Any]] | None = None,
        request_type: str | None = None,
    ) -> dict[str, Any]:
        """Validate a recipe (roles/keys/cards/gates/output-completeness) -> Verdict.

        ``acceptance_criteria`` + ``request_type`` add the 1A.1 acceptance checks
        (criterion fields must be producible; request-type sanity)."""
        return aa.validate(
            recipe, data=data, expected_outputs=expected_outputs,
            acceptance_criteria=acceptance_criteria, request_type=request_type,
        )

    @server.tool()
    def smoke(
        recipe: dict[str, Any],
        sample: int = 10,
        data: str | None = None,
        output_dir: str | None = None,
        bootstrap_ray: bool = False,
        calibration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a recipe on a bounded sample and return evidence (incl. a ``smoke_token``).

        ``bootstrap_ray`` auto-starts a local Ray head when none is reachable;
        ``output_dir`` sets where sampled outputs go; ``calibration`` seeds mode
        selection from a prior smoke. Parity with the ``smoke`` verb/CLI."""
        return aa.smoke(
            recipe, sample=sample, data=data, output_dir=output_dir,
            bootstrap_ray=bootstrap_ray, calibration=calibration,
        )

    @server.tool()
    def run(
        recipe: dict[str, Any],
        confirm: bool | str = False,
        data: str | None = None,
        output_dir: str | None = None,
        checkpoint_path: str | None = None,
        bootstrap_ray: bool = False,
        smoke_token: str | None = None,
        calibration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Confirm-gated full run. Refuses without explicit confirmation.

        ``smoke_token`` satisfies ``AUDIO_AGENT_REQUIRE_SMOKE`` (pass the token from a
        prior ``smoke``); ``bootstrap_ray`` auto-starts Ray; ``checkpoint_path`` enables
        partial-run resume; ``output_dir``/``calibration`` mirror the verb. Full parity
        with the ``run`` verb/CLI."""
        return aa.run(
            recipe, confirm=confirm, data=data, output_dir=output_dir,
            checkpoint_path=checkpoint_path, bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token, calibration=calibration,
        )

    @server.tool()
    def report(output: str, data: str | None = None) -> dict[str, Any]:
        """Post-hoc evidence report from an output manifest/dir."""
        return aa.report(output, data=data)

    @server.tool()
    def verify(
        acceptance_criteria: list[dict[str, Any]],
        evidence: dict[str, Any] | None = None,
        frozen_criteria: list[dict[str, Any]] | None = None,
        recipe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify acceptance criteria against evidence -> AcceptanceReport (1A.1/1A.3).

        ``overall`` is ``met`` iff every ``must`` criterion is met; states are
        met / not_met / unverifiable / unachievable (never silently relaxed). Pass
        ``frozen_criteria`` or a ``recipe`` to run the honesty guard (flags a
        weaker-than-confirmed contract and forces overall=not_met)."""
        return aa.verify(acceptance_criteria, evidence=evidence, frozen_criteria=frozen_criteria, recipe=recipe)

    @server.tool()
    def resolve(
        stage: str,
        label: str | None = None,
        use_case: str | None = None,
        explicit: dict[str, Any] | None = None,
        data_driven: bool = False,
    ) -> dict[str, Any]:
        """Resolve an outcome (label/use_case/explicit) to concrete stage config (1A.2).

        Maps a user-facing outcome to params (or a PreserveByValueStage filter) via
        the card's metrics anchors/presets, with an auditable strategy trail. Never
        exposes or invents internal thresholds."""
        return aa.resolve(stage, label=label, use_case=use_case, explicit=explicit, data_driven=data_driven)

    @server.tool()
    def runs(run_id: str | None = None) -> dict[str, Any]:
        """List local run records (provenance), or load one by run_id. Local history,
        not shared memory/learning."""
        return aa.runs(run_id=run_id)

    @server.tool()
    def plan_continuation(recipe: dict[str, Any], parent_run_id: str, data: str | None = None) -> dict[str, Any]:
        """Plan a follow-up run incrementally against a prior run: reuse the parent's
        output where the new recipe safely extends it (else full_rerun with the
        divergence point). Reuse requires the same source data."""
        return aa.plan_continuation(recipe, parent_run_id, data=data)

    @server.tool()
    def calibrate(smoke_report: dict[str, Any]) -> dict[str, Any]:
        """Extract measured per-stage resources from a smoke report (1C.2), to pass to
        run(calibration=...) so the planner uses measured over card best-guess numbers."""
        return aa.calibrate(smoke_report)

    return server


def main() -> int:
    try:
        server = build_server()
    except ModuleNotFoundError:
        print("MCP is not installed. Install it with:  pip install 'mcp[cli]'")
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
