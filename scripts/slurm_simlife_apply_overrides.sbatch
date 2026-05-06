#!/bin/bash
#SBATCH --job-name=sl-ovr
#SBATCH --array=0-69%16
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=m3_agent_outputs/simlife_logs/apply_overrides_%A_%a.log

# Stage A.5: apply per-chain dialogue overrides on top of the default per-unit
# intermediate JSONs. Walks SimLife-Data-HF/task_dialogue_audio/ for chains with
# replacement WAVs + asr.jsonl, and writes per-(chain, unit) voice JSONs to
# data/intermediate/per_chain/<chain>/<unit>/. With REGENERATE_MEMORIES=1 it
# also re-runs Qwen2.5-Omni for clips whose voices changed (this is what makes
# the override actually take effect at Stage B — see the atomic pairing rule
# in m3_agent/simlife_apply_dialogue_overrides.py).
#
# Run AFTER Stage A precompute for the affected units (the script reads the
# precomputed face JSONs to determine n_clips per unit).
#
# Tune --array to ``ls SimLife-Data-HF/task_dialogue_audio | wc -l`` (default
# 0-69 for the current 70 chains).


source /home/daiyp/.bashrc
cd /home/daiyp/m3-agent

REGENERATE_MEMORIES="${REGENERATE_MEMORIES:-1}"
PY=/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/m3a/bin/python

# Build a sorted list of override-chain ids on the fly so we don't need a
# separate manifest file.
CHAIN=$(ls SimLife-Data-HF/task_dialogue_audio 2>/dev/null \
        | grep '^vc_' \
        | sort \
        | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [[ -z "$CHAIN" ]]; then
    echo "No chain at array index $SLURM_ARRAY_TASK_ID"
    exit 0
fi

EXTRA=()
if [[ "$REGENERATE_MEMORIES" == "1" ]]; then
    EXTRA+=(--regenerate_memories)
fi

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID chain=$CHAIN regen_memories=$REGENERATE_MEMORIES host=$(hostname)"
"$PY" m3_agent/simlife_apply_dialogue_overrides.py --chain "$CHAIN" "${EXTRA[@]}"
echo "[$(date)] done $CHAIN"
