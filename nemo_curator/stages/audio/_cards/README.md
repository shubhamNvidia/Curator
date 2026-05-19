# Audio Stage Cards

One card per stage class. Each card lives in its own subdirectory
(`_cards/<StageName>/stage_card.yaml`) so the registry's `rglob("stage_card.yaml")`
discovery picks it up without collisions.

These are machine-readable summaries of the stage's contract that the agentic
planner reasons over. They do **not** replace the Python class — the
`drift lint` step in CI catches mismatches between a card's `params` list and
the class's `__init__` signature.

See `nemo_curator/agentic/cards.py` for the full `StageCard` schema.
