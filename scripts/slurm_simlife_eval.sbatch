#!/bin/bash
#SBATCH --job-name=sl-eval
#SBATCH --array=0-7
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=46G
#SBATCH --time=5-00:00:00
#SBATCH --output=m3_agent_outputs/simlife_logs/eval_%A_%a.log


source /home/daiyp/.bashrc
cd /home/daiyp/m3-agent

micromamba activate m3a 

NUM_SHARDS="${NUM_SHARDS:-8}"
DATA_FILE="${DATA_FILE:-data/annotations/simlife.json}"

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SHARDS" ]]; then
    echo "shard $SLURM_ARRAY_TASK_ID >= NUM_SHARDS=$NUM_SHARDS, nothing to do"
    exit 0
fi

echo "[$(date)] shard=$SLURM_ARRAY_TASK_ID/$NUM_SHARDS data=$DATA_FILE host=$(hostname)"
PYTHONPATH=. python m3_agent/control.py \
    --data_file "$DATA_FILE" \
    --num_shards "$NUM_SHARDS" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --tensor_parallel_size 2
echo "[$(date)] done shard=$SLURM_ARRAY_TASK_ID"
