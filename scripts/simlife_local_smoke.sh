#!/bin/bash
# Local smoke test for the SimLife adapter — run this BEFORE submitting to SLURM.
#
# What it does:
#   1. Build the SimLife manifests (idempotent; cheap).
#   2. Pick one video chain and the first 1–2 video units it references.
#   3. Run Stage A (precompute clips/faces/voices/memories) on those units.
#   4. Run Stage B (assemble the per-chain VideoGraph pickle).
#   5. Inspect the resulting pickle.
#
# Knobs — override via env vars, e.g.:
#   N_UNITS=3 SKIP_MEMORY=1 ./scripts/simlife_local_smoke.sh
#   CHAIN=vc_000001 ./scripts/simlife_local_smoke.sh
set -euo pipefail

PY="${PY:-/data/daiyp/micromamba/envs/m3a/bin/python}"
CHAIN="${CHAIN:-}"          # empty -> first chain in data_chains.jsonl
N_UNITS="${N_UNITS:-1}"     # how many units (from the start of the chain) to precompute
SKIP_MEMORY="${SKIP_MEMORY:-0}"   # 1 to skip Qwen captioning (use synthetic memories for stage B)
QWEN_GPU="${QWEN_GPU:-1}"   # which GPU index to bind Qwen to via CUDA_VISIBLE_DEVICES

cd "$(dirname "$0")/.."

echo "==> [1/5] Build manifests"
"$PY" m3_agent/simlife_data_prep.py

# Pick a chain (first line of data_chains.jsonl unless CHAIN is set).
if [[ -z "$CHAIN" ]]; then
    CHAIN=$(head -1 data/simlife/data_chains.jsonl | "$PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["chain_id"])')
fi
echo "    using chain: $CHAIN"

# Read its first N_UNITS video_ids.
mapfile -t UNITS < <("$PY" - "$CHAIN" "$N_UNITS" <<'PYEOF'
import json, sys
target, n = sys.argv[1], int(sys.argv[2])
with open("data/simlife/data_chains.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row["chain_id"] == target:
            for v in row["video_ids"][:n]:
                print(v)
            break
PYEOF
)
echo "    units to precompute (${#UNITS[@]}): ${UNITS[*]}"

echo
echo "==> [2/5] Stage A — precompute per unit"
for unit in "${UNITS[@]}"; do
    echo "    ----- $unit -----"
    if [[ "$SKIP_MEMORY" == "1" ]]; then
        "$PY" m3_agent/simlife_precompute_unit.py --unit "$unit" --skip_memory --log_level INFO
    else
        # Qwen2.5-Omni is ~15 GB; pin to a less-busy GPU.
        CUDA_VISIBLE_DEVICES="$QWEN_GPU" \
            "$PY" m3_agent/simlife_precompute_unit.py --unit "$unit" --log_level INFO
    fi
done

echo
echo "==> [3/5] Sanity-check intermediate JSONs"
for unit in "${UNITS[@]}"; do
    d="data/intermediate/$unit"
    nf=$(ls "$d"/clip_*_faces.json 2>/dev/null | wc -l)
    nv=$(ls "$d"/clip_*_voices.json 2>/dev/null | wc -l)
    nm=$(ls "$d"/clip_*_memory.json 2>/dev/null | wc -l)
    echo "    $unit: faces=$nf voices=$nv memory=$nm"
done

# If memory generation was skipped, synthesize a minimal stub so stage B has
# something to ingest. Real runs will skip this branch entirely.
if [[ "$SKIP_MEMORY" == "1" ]]; then
    echo "    (memory skipped — writing synthetic stub for stage B)"
    "$PY" - <<PYEOF
import json, glob, os
for unit in ${UNITS[*]@Q}.split():
    d = f"data/intermediate/{unit}"
    for fp in glob.glob(f"{d}/clip_*_voices.json"):
        k = int(os.path.basename(fp).split("_")[1])
        mp = f"{d}/clip_{k}_memory.json"
        if os.path.exists(mp):
            continue
        voices = json.load(open(fp))
        speakers = []
        for v in voices:
            s = v.get("speaker") or "unknown"
            if s not in speakers:
                speakers.append(s)
        # One trivial episodic line so process_memories has something to insert.
        epi = ["<face_0> appears."] if json.load(open(f"{d}/clip_{k}_faces.json")) else []
        json.dump({"episodic": epi, "semantic": [], "voice_speaker_order": speakers}, open(mp, "w"))
PYEOF
fi

echo
echo "==> [4/5] Stage B — assemble chain graph"
"$PY" m3_agent/simlife_assemble_chain.py --chain "$CHAIN" --overwrite --log_level INFO

PKL="data/memory_graphs/${CHAIN}.pkl"
echo
echo "==> [5/5] Inspect $PKL"
"$PY" - "$PKL" <<'PYEOF'
import sys, pickle
sys.path.insert(0, ".")
import mmagent.videograph
sys.modules["videograph"] = mmagent.videograph
g = pickle.load(open(sys.argv[1], "rb"))
types = {}
for n in g.nodes.values():
    types[n.type] = types.get(n.type, 0) + 1
print(f"  nodes={len(g.nodes)} edges={len(g.edges)} types={types}")
print(f"  characters={len(getattr(g, 'character_mappings', {}))}")
print(f"  text_nodes_by_clip={ {k: len(v) for k, v in g.text_nodes_by_clip.items()} }")
sample_text = next(
    (n.metadata['contents'][0] for n in g.nodes.values()
     if n.type in ('episodic', 'semantic') and n.metadata.get('contents')),
    None,
)
if sample_text:
    s = sample_text if len(sample_text) <= 160 else sample_text[:160] + "..."
    print(f"  sample text node: {s!r}")
PYEOF

echo
echo "OK — local smoke test passed for $CHAIN."
echo "Next: submit the full SLURM arrays once this looks right:"
echo "  sbatch scripts/slurm_simlife_precompute_unit.sbatch"
echo "  sbatch scripts/slurm_simlife_assemble_chain.sbatch"
