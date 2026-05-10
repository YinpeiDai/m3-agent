#!/bin/bash
#SBATCH --job-name=sl-asm
#SBATCH --array=0-119%32
#SBATCH --account=chaijy0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=m3_agent_outputs/simlife_logs/assemble_%A_%a.log

# Stage B: per-video-chain graph assembly. CPU-only; uses text-embedding-3-large API
# (re-embeds calendar-prefixed memory text per chain).
# One job per unique video_chain_id. Array length must match
# `wc -l data/simlife/data_chains.jsonl` (currently 120 chains, so 0-119).

source /home/daiyp/.bashrc
cd /home/daiyp/m3-agent

micromamba activate m3a 

CHAINS_FILE=data/simlife/data_chains.jsonl

LINE=$((SLURM_ARRAY_TASK_ID + 1))
CHAIN=$(sed -n "${LINE}p" "$CHAINS_FILE" | python -c 'import sys, json; print(json.loads(sys.stdin.read())["chain_id"])')
if [[ -z "$CHAIN" ]]; then
    echo "No chain at line $LINE of $CHAINS_FILE"
    exit 1
fi

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID chain=$CHAIN host=$(hostname)"
# Build both memory-graph variants by default (audio + noaudio). Override with
# VARIANT=audio or VARIANT=noaudio before sbatch to build only one.
VARIANT="${VARIANT:-both}"
python m3_agent/simlife_assemble_chain.py --chain "$CHAIN" --variant "$VARIANT"
echo "[$(date)] done $CHAIN ($VARIANT)"
