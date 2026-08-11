# Smoke, the confirm gate, the run, and acceptance verification

Loaded on demand from `SKILL.md` steps 5 to 7. Read this once the semantic critique passes.

## 5. Smoke (empirical loop, <= 2 iterations)

```bash
python -m nemo_curator.audio_agent smoke --recipe recipe.yaml --sample 10 --data /path/to/data --bootstrap-ray
```

`--bootstrap-ray` lets the agent start a correctly-configured local Ray head
itself (free port, plasma on /tmp, API limit) so no manual Ray setup is needed.
If a cluster already exists, set `RAY_ADDRESS` and omit the flag.

Show the user `retained` / `rejected` + examples. If `goals_met` is false (0
retained, errors), read the structured `diagnosis` (when present). Adjust a
threshold only for a grounded data/filter failure; environment/action-required
failures go through the decision policy in `SKILL.md` step 2.

On a GPU smoke the result also carries a `calibration` block (measured per-stage
VRAM/throughput). Pass it to the full run so the resource planner can raise a
card/default estimate when the smoke observed a larger peak; because a bounded
smoke cannot prove the full-run maximum, calibration never lowers that baseline:
`run ... --calibration calib.json` (extract it with `calibrate --smoke
smoke.json`). On a CPU smoke there's no VRAM to measure, so the planner keeps
using the card facts.

## 6. Confirm gate -> run

Present the plan, the semantic critique (`intent_status: pass`), the smoke
evidence, the scale/time estimate, **and the acceptance-criteria contract** —
stating for each criterion **what its metric captures and what it does NOT**
(e.g. "UTMOS measures naturalness/overall quality, not background-noise level —
add a noise/SIGMOS criterion?"). Never silently decide which metric stands for a
fuzzy word ("clean", "good"); surface it here. Then ask the user to confirm. Only
then:

```bash
python -m nemo_curator.audio_agent run --recipe recipe.yaml --confirm <config_hash> --data /path/to/data --bootstrap-ray
```

Passing the `config_hash` (from the refusal output) enforces plan-execution
integrity: what was approved is exactly what runs. `--bootstrap-ray` starts the
Ray head if needed (same as smoke).

Guardrails enforced in the tool: paths are restricted to `AUDIO_AGENT_WORKSPACE`
(when set); secrets/transcripts are stripped from tool output; and if
`AUDIO_AGENT_REQUIRE_SMOKE` is set, also pass `--smoke-token <token>` (from the
`smoke` output) or `run` refuses. The resource planner auto-picks streaming/batch
and refuses if the recipe can't fit the machine.

## 7. Report + verify acceptance

Summarize the returned `report` (retained/rejected, per-filter counts, failure
reasons, output paths) in plain language. `run` also returns `acceptance`, verified
against the recipe's embedded contract and terminal-output evidence; treat that as
the primary post-run verdict. Use standalone `verify` only for an explicitly
post-hoc evidence set or a newly proposed contract:

```bash
python -m nemo_curator.audio_agent verify --criteria criteria.yaml --evidence evidence.json \
  --recipe recipe.yaml   # frozen contract -> runs the honesty guard
```

Report the `AcceptanceReport`: `overall` (`met` iff every `must` criterion is met)
plus each criterion's state — `met` / `not_met` / `unverifiable` (no evidence, e.g.
WER with no references) / `unachievable` (the data cannot reach an absolute target).
Only declare success when `overall` is `met`. `unachievable`/`not_met` are honest
outcomes: offer options (adjust thresholds, provide references, relabel an absolute
bar with the user's consent) — never silently relax a `must`.

**Reviewer charter (you, the host, are the reviewer).** After the deterministic
verify:
1. Resolve any `semantic_fit` criteria (they come back `unverifiable` — that's your
   job): judge, grounded in the evidence/examples, whether the result coheres with
   intent.
2. Read the `honesty` section — the guard flags goalpost-moving (a confirmed `must`
   dropped/downgraded/relaxed vs the frozen contract). If non-empty, `overall` is
   forced `not_met`: do **not** present success. The contract is frozen into the
   recipe and covered by `config_hash`, so relaxing a bar means re-confirming a new
   contract with the user, never editing it silently.
3. You may surface semantic concerns but **may not override the deterministic
   verdict** — if unresolved, escalate to the user.
