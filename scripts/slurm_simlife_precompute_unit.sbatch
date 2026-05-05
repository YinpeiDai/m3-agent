#!/bin/bash
#SBATCH --job-name=sl-pre
#SBATCH --array=0-319%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=m3_agent_outputs/simlife_logs/precompute_%A_%a.out
#SBATCH --error=m3_agent_outputs/simlife_logs/precompute_%A_%a.err

# Stage A: per-video-unit precomputation (clips + faces + voices + memories).
# Set --partition / --account / --qos for your cluster before submitting.
# Array length must match `wc -l data/simlife/units.txt` (default 0-319).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

UNITS_FILE=data/simlife/units.txt
PY=/data/daiyp/micromamba/envs/m3a/bin/python

# 1-indexed sed line number from 0-indexed SLURM_ARRAY_TASK_ID
LINE=$((SLURM_ARRAY_TASK_ID + 1))
UNIT=$(sed -n "${LINE}p" "$UNITS_FILE")
if [[ -z "$UNIT" ]]; then
    echo "No unit at line $LINE of $UNITS_FILE"
    exit 1
fi

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID unit=$UNIT host=$(hostname)"
"$PY" m3_agent/simlife_precompute_unit.py --unit "$UNIT"
echo "[$(date)] done $UNIT"
