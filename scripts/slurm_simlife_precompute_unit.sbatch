#!/bin/bash
#SBATCH --job-name=sl-pre
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1%10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=m3_agent_outputs/simlife_logs/precompute_%A_%a.log


# Stage A: per-video-unit precomputation (clips + faces + voices + memories).
# Array length must match `wc -l data/simlife/units.txt` (default 0-319).

source /home/daiyp/.bashrc
cd /home/daiyp/m3-agent

UNITS_FILE=data/simlife/units.txt

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

# 1-indexed sed line number from 0-indexed SLURM_ARRAY_TASK_ID
LINE=$((SLURM_ARRAY_TASK_ID + 1))
UNIT=$(sed -n "${LINE}p" "$UNITS_FILE")

if [[ -z "$UNIT" ]]; then
    echo "No unit at line $LINE of $UNITS_FILE"
    exit 1
fi

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID unit=$UNIT host=$(hostname)"
/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/m3a/bin/python m3_agent/simlife_precompute_unit.py --unit "$UNIT"
echo "[$(date)] done $UNIT"
