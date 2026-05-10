#!/bin/bash
#SBATCH --job-name=sl-ovr
#SBATCH --array=0-82%12
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=m3_agent_outputs/simlife_logs/apply_overrides_%A_%a.log

# Stage A.5: apply per-chain dialogue overrides on top of the default per-unit
# intermediate JSONs. Walks SimLife-Data-HF/video_chains/<vc>/overlay/ for
# chains that ship a per-chain dialogue override and writes per-(chain, unit)
# voice JSONs to data/intermediate/per_chain/<chain>/<unit>/. With
# REGENERATE_MEMORIES=1 it also re-runs Qwen2.5-Omni for clips whose voices
# changed (required for the override to actually take effect at Stage B —
# see the atomic pairing rule in m3_agent/simlife_apply_dialogue_overrides.py).
#
# Run AFTER Stage A precompute for the affected units (the script reads the
# precomputed face JSONs to determine n_clips per unit).
#
# Tune --array to match the count of overlay chains:
#   ls SimLife-Data-HF/video_chains | while read d; do
#     [[ -d SimLife-Data-HF/video_chains/$d/overlay ]] && echo "$d"
#   done | wc -l
# (currently 83 — array 0-82).

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

source /home/daiyp/.bashrc
cd /home/daiyp/m3-agent
micromamba activate m3a 

REGENERATE_MEMORIES="${REGENERATE_MEMORIES:-1}"

# Build a sorted list of override-chain ids on the fly (chains with an
# overlay/ subdir under video_chains/) so we don't need a separate manifest.
CHAIN=$(for d in SimLife-Data-HF/video_chains/vc_*; do
            [[ -d "$d/overlay" ]] && basename "$d"
        done | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [[ -z "$CHAIN" ]]; then
    echo "No chain at array index $SLURM_ARRAY_TASK_ID"
    exit 0
fi

EXTRA=()
if [[ "$REGENERATE_MEMORIES" == "1" ]]; then
    EXTRA+=(--regenerate_memories)
fi

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID chain=$CHAIN regen_memories=$REGENERATE_MEMORIES host=$(hostname)"
python m3_agent/simlife_apply_dialogue_overrides.py --chain "$CHAIN" "${EXTRA[@]}"
echo "[$(date)] done $CHAIN"
