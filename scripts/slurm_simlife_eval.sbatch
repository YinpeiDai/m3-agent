#!/bin/bash
#SBATCH --job-name=sl-eval
#SBATCH --array=0-7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=m3_agent_outputs/simlife_logs/eval_%A_%a.out
#SBATCH --error=m3_agent_outputs/simlife_logs/eval_%A_%a.err

# Stage C: sharded evaluation. Each job loads the M3-Agent-Control vLLM
# model on 2 GPUs and processes a deterministic 1/N slice of the chain
# pickles. Set --partition / --account / --qos for your cluster before
# submitting. After all shards finish, run scripts/simlife_combine_eval.sh
# (or the combiner directly) to merge per-shard outputs.
#
# Tune NUM_SHARDS to balance "throughput" vs "GPU footprint":
#   - 50 chains, ~536 questions: NUM_SHARDS=8 puts ~6-7 chains (~67 questions)
#     per shard. Sequential rounds ≈ 5; with batch_size=4, that's ~17 batches
#     × 5 rounds × ~30s per round ≈ 40 minutes per shard, comfortably under
#     the 12 h limit.
#   - For a quicker run, raise NUM_SHARDS to 16 or 25 (one chain per shard).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

NUM_SHARDS="${NUM_SHARDS:-8}"
DATA_FILE="${DATA_FILE:-data/annotations/simlife.json}"
PY=/data/daiyp/micromamba/envs/m3a/bin/python

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SHARDS" ]]; then
    echo "shard $SLURM_ARRAY_TASK_ID >= NUM_SHARDS=$NUM_SHARDS, nothing to do"
    exit 0
fi

echo "[$(date)] shard=$SLURM_ARRAY_TASK_ID/$NUM_SHARDS data=$DATA_FILE host=$(hostname)"
"$PY" m3_agent/control.py \
    --data_file "$DATA_FILE" \
    --num_shards "$NUM_SHARDS" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --tensor_parallel_size 2
echo "[$(date)] done shard=$SLURM_ARRAY_TASK_ID"
