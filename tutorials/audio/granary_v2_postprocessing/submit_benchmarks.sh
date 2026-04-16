#!/bin/bash
# Submit postprocessing jobs per benchmark with configurable per-benchmark chunk sizes.
#
# Scans top-level subdirectories of <input_dir> and submits each benchmark
# independently via submit.sh. Per-benchmark MANIFESTS_PER_JOB overrides are
# defined in BENCHMARK_CHUNKS below — all others get the DEFAULT (128).
#
# Usage:
#   bash submit_benchmarks.sh <output_dir> <input_dir>
#
# Override defaults via env:
#   DEFAULT_MANIFESTS_PER_JOB=64 CPUS_PER_JOB=32 bash submit_benchmarks.sh <output_dir> <input_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit.sh"

OUTPUT_DIR="${1:?Usage: bash submit_benchmarks.sh <output_dir> <input_dir>}"
INPUT_DIR="${2:?}"

# --------------------------------------------------------------------------
# Per-benchmark chunk size overrides.
# Add/edit entries here: ["benchmark_name"]=N
# --------------------------------------------------------------------------
declare -A BENCHMARK_CHUNKS=(
    ["ytc"]=8
)

DEFAULT_MANIFESTS_PER_JOB="${DEFAULT_MANIFESTS_PER_JOB:-8}"
export CPUS_PER_JOB="${CPUS_PER_JOB:-16}"

# --------------------------------------------------------------------------

mapfile -t BENCH_DIRS < <(find "${INPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#BENCH_DIRS[@]} -eq 0 ]]; then
    echo "Error: no subdirectories found under ${INPUT_DIR}" >&2
    exit 1
fi

echo "Output dir : ${OUTPUT_DIR}"
echo "Input dir  : ${INPUT_DIR}"
echo "Benchmarks : ${#BENCH_DIRS[@]}"
echo "CPUs/job   : ${CPUS_PER_JOB}"
echo ""

for BENCH_DIR in "${BENCH_DIRS[@]}"; do
    BENCH_NAME=$(basename "${BENCH_DIR}")

    if [[ -v BENCHMARK_CHUNKS["${BENCH_NAME}"] ]]; then
        CHUNKS="${BENCHMARK_CHUNKS[${BENCH_NAME}]}"
    else
        CHUNKS="${DEFAULT_MANIFESTS_PER_JOB}"
    fi

    # Check if every manifest in this benchmark already has a non-empty output.
    # If so, skip the benchmark entirely — no jobs submitted.
    bench_done=$(python3 - "${INPUT_DIR}" "${OUTPUT_DIR}" "${BENCH_DIR}" <<'PYEOF'
import sys, os
from pathlib import Path
input_dir, output_dir, bench_dir = sys.argv[1], sys.argv[2], sys.argv[3]
manifests = list(Path(bench_dir).rglob("*.jsonl"))
if not manifests:
    print("no"); sys.exit(0)
for m in manifests:
    out = os.path.join(output_dir, os.path.relpath(str(m), input_dir))
    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        print("no"); sys.exit(0)
print("yes")
PYEOF
)

    if [[ "${bench_done}" == "yes" ]]; then
        echo ">>> ${BENCH_NAME}  — already done, skipping"
        continue
    fi

    echo ">>> ${BENCH_NAME}  (MANIFESTS_PER_JOB=${CHUNKS})"
    # INPUT_ROOT tells submit.sh (and the Slurm job) to use the original root
    # dir as the path anchor, so output mirrors full hierarchy: ytc/en9/manifest.jsonl
    MANIFESTS_PER_JOB="${CHUNKS}" INPUT_ROOT="${INPUT_DIR}" bash "${SUBMIT_SCRIPT}" "${OUTPUT_DIR}" "${BENCH_DIR}"
done
