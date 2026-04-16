#!/bin/bash
# Submit Slurm jobs for the postprocessing pipeline.
# Manifests are grouped into chunks of MANIFESTS_PER_JOB so the number of
# submitted jobs stays manageable even for large datasets.
#
# Chunks where every output already exists are skipped — no job submitted.
# This makes resuming efficient: only incomplete work is requeued.
#
# Usage:
#   bash submit.sh <output_dir> <input_dir_1> [input_dir_2 ...]
#
# Tune chunk size and CPUs via environment variables:
#   MANIFESTS_PER_JOB=32 CPUS_PER_JOB=64 bash submit.sh <output_dir> <input_dir>
#
# Multiple input dirs are submitted as sequential waves:
# wave N+1 starts only after every job in wave N finishes (afterany).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"

MANIFESTS_PER_JOB="${MANIFESTS_PER_JOB:-8}"
CPUS_PER_JOB="${CPUS_PER_JOB:-8}"

# INPUT_ROOT overrides the path-anchor used for output mirroring.
# Set by submit_benchmarks.sh so that benchmark subdirs (e.g. .../ytc) still
# produce correctly nested output paths (e.g. output/ytc/en9/manifest.jsonl).
# When calling submit.sh directly, leave unset — INPUT_DIR is used as the anchor.
INPUT_ROOT="${INPUT_ROOT:-}"

OUTPUT_DIR="${1:?Usage: bash submit.sh <output_dir> <input_dir_1> [input_dir_2 ...]}"
shift
INPUT_DIRS=("$@")

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
    echo "Error: at least one input_dir is required." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PREV_WAVE_IDS=()

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    mapfile -t MANIFESTS < <(find "${INPUT_DIR}" -name "*.jsonl" | sort)

    if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
        echo "Warning: no *.jsonl found under ${INPUT_DIR}, skipping." >&2
        continue
    fi

    DEPEND_FLAG=""
    if [[ ${#PREV_WAVE_IDS[@]} -gt 0 ]]; then
        DEP_LIST=$(IFS=:; echo "${PREV_WAVE_IDS[*]}")
        DEPEND_FLAG="--dependency=afterany:${DEP_LIST}"
    fi

    N_JOBS=$(( (${#MANIFESTS[@]} + MANIFESTS_PER_JOB - 1) / MANIFESTS_PER_JOB ))
    echo "Wave: ${INPUT_DIR}"
    echo "  Manifests : ${#MANIFESTS[@]}  |  per job : ${MANIFESTS_PER_JOB}  |  max jobs : ${N_JOBS}"
    [[ -n "${DEPEND_FLAG}" ]] && echo "  Depends on: ${PREV_WAVE_IDS[*]}"

    CURRENT_WAVE_IDS=()
    n_submitted=0
    n_skipped=0
    chunk=()
    # Use INPUT_ROOT as the path anchor if set, otherwise fall back to INPUT_DIR
    ROOT_DIR="${INPUT_ROOT:-${INPUT_DIR}}"

    for MANIFEST in "${MANIFESTS[@]}"; do
        chunk+=("${MANIFEST}")

        if [[ ${#chunk[@]} -eq ${MANIFESTS_PER_JOB} ]]; then
            # Check if every output in this chunk already exists and is non-empty
            all_done=$(python3 - "${ROOT_DIR}" "${OUTPUT_DIR}" "${chunk[@]}" <<'PYEOF'
import sys, os
input_dir, output_dir = sys.argv[1], sys.argv[2]
for m in sys.argv[3:]:
    out = os.path.join(output_dir, os.path.relpath(m, input_dir))
    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        print("no"); sys.exit(0)
print("yes")
PYEOF
)
            if [[ "${all_done}" == "yes" ]]; then
                (( n_skipped++ )) || true
            else
                JOB_ID=$(sbatch \
                    ${DEPEND_FLAG} \
                    --cpus-per-task="${CPUS_PER_JOB}" \
                    --parsable \
                    "${RUN_SCRIPT}" "${INPUT_DIR}" "${OUTPUT_DIR}" --manifests "${chunk[@]}")
                CURRENT_WAVE_IDS+=("${JOB_ID}")
                echo "  ${JOB_ID}  ←  ${#chunk[@]} manifests"
                (( n_submitted++ )) || true
            fi
            chunk=()
        fi
    done

    # Handle remaining manifests
    if [[ ${#chunk[@]} -gt 0 ]]; then
        all_done=$(python3 - "${ROOT_DIR}" "${OUTPUT_DIR}" "${chunk[@]}" <<'PYEOF'
import sys, os
input_dir, output_dir = sys.argv[1], sys.argv[2]
for m in sys.argv[3:]:
    out = os.path.join(output_dir, os.path.relpath(m, input_dir))
    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        print("no"); sys.exit(0)
print("yes")
PYEOF
)
        if [[ "${all_done}" == "yes" ]]; then
            (( n_skipped++ )) || true
        else
            JOB_ID=$(sbatch \
                ${DEPEND_FLAG} \
                --cpus-per-task="${CPUS_PER_JOB}" \
                --parsable \
                "${RUN_SCRIPT}" "${INPUT_DIR}" "${OUTPUT_DIR}" --manifests "${chunk[@]}")
            CURRENT_WAVE_IDS+=("${JOB_ID}")
            echo "  ${JOB_ID}  ←  ${#chunk[@]} manifests (last chunk)"
            (( n_submitted++ )) || true
        fi
    fi

    echo "  Submitted : ${n_submitted}  |  Already done (skipped) : ${n_skipped}"
    PREV_WAVE_IDS=("${CURRENT_WAVE_IDS[@]}")
    echo ""
done

if [[ ${#PREV_WAVE_IDS[@]} -gt 0 ]]; then
    echo "All waves submitted."
    echo "Monitor : squeue -u ${USER}"
    echo "Job IDs : ${PREV_WAVE_IDS[*]}"
else
    echo "Nothing to submit — all manifests already done."
fi
