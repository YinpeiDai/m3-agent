"""Build SimLife manifests for the M3-Agent adapter.

Outputs (under data/simlife/ and data/annotations/):
- units.txt              one "video_XXXXXX" per line, union of all task video_ids
- data_units.jsonl       per-unit precompute manifest (Stage A)
- data_chains.jsonl      per-video-chain assembly manifest (Stage B). Multiple
                         task JSONs that share a video_chain_id collapse into
                         one entry — they have identical video_ids, only their
                         stop_day_position / questions differ.
- ../annotations/simlife.json
                         control.py-compatible QA bundle, keyed by chain id.
                         Each entry's qa_list merges questions from every task
                         that targets that chain. We use the *_vision question
                         fields (question_vision / options_vision /
                         answer_vision), since M3-Agent is a VLM.

This script is fast and side-effect-free outside its output paths; rerunning is safe.
"""
import argparse
import collections
import glob
import json
import math
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIMLIFE_ROOT = os.path.join(REPO_ROOT, "SimLife-Data-HF")
TASKS_DIR = os.path.join(SIMLIFE_ROOT, "tasks")
UNITS_DIR = os.path.join(SIMLIFE_ROOT, "video_units")

OUT_SIMLIFE = os.path.join(REPO_ROOT, "data", "simlife")
OUT_ANNOTATIONS = os.path.join(REPO_ROOT, "data", "annotations")

CLIP_INTERVAL_SEC = 30
CLIPS_PER_UNIT_OFFSET = 100  # global_clip_id = unit_idx * 100 + k


def unit_dir_name(video_id):
    return f"video_{int(video_id):06d}"


def chain_basename(video_chain_id):
    return f"vc_{int(video_chain_id):06d}"


def load_unit_durations(units):
    durations = {}
    for u in units:
        meta_path = os.path.join(UNITS_DIR, u, "metadata.json")
        with open(meta_path) as f:
            durations[u] = float(json.load(f)["duration_sec"])
    return durations


def question_to_qa(question, task_id, before_clip):
    """Build a control.py-compatible QA dict from a SimLife question, using the
    *_vision variants (the model is a VLM and the vision-targeted phrasing
    avoids leaking the canonical character names).
    """
    qid = f"task_{int(task_id):06d}_Q{int(question['task_question_id']):06d}"
    options = question["options_vision"]
    options_str = " | ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(options))
    prompt = f"{question['question_vision']}\nOptions: {options_str}"
    return {
        "question_id": qid,
        "question": prompt,
        "answer": question["answer_vision"],
        "options": options,
        "question_type": question["question_type"],
        "format": question.get("format"),
        "task_id": int(task_id),
        "before_clip": before_clip,
    }


def compute_before_clip(task, unit_durations):
    """Translate (stop_day_position, stop_video_time_local) into the integer
    clip-id used as ``metadata['timestamp']`` on each memory text node.

    Convention (matches M3-Bench / robot.json): ``before_clip`` is the LAST
    clip whose memories the agent is allowed to see. The clip CONTAINING the
    stop time is dropped, since it may include content from after the stop.

    Layout:
        global_clip_id = unit_idx * 100 + k_within_unit
    """
    stop_pos = int(task["stop_day_position"])
    video_ids = task["video_ids"]

    # Sum durations of every fully-shown unit (those strictly before stop_pos).
    cum = 0.0
    for v in video_ids[:stop_pos]:
        cum += unit_durations[unit_dir_name(v)]
    stop_local = float(task["stop_video_time_global_sec"]) - cum
    stop_local = max(0.0, stop_local)
    k_at_stop = int(stop_local // CLIP_INTERVAL_SEC)

    if k_at_stop > 0:
        return stop_pos * CLIPS_PER_UNIT_OFFSET + (k_at_stop - 1)

    # k_at_stop == 0: the stop time is in the first 30s of unit stop_pos.
    # Drop unit stop_pos entirely and fall back to the last clip of the
    # previous unit.
    if stop_pos == 0:
        return -1  # nothing visible
    prev_unit = unit_dir_name(video_ids[stop_pos - 1])
    prev_n_clips = max(1, math.ceil(unit_durations[prev_unit] / CLIP_INTERVAL_SEC))
    return (stop_pos - 1) * CLIPS_PER_UNIT_OFFSET + (prev_n_clips - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simlife_root", default=SIMLIFE_ROOT)
    parser.add_argument("--out_simlife", default=OUT_SIMLIFE)
    parser.add_argument("--out_annotations", default=OUT_ANNOTATIONS)
    args = parser.parse_args()

    os.makedirs(args.out_simlife, exist_ok=True)
    os.makedirs(args.out_annotations, exist_ok=True)

    task_paths = sorted(glob.glob(os.path.join(args.simlife_root, "tasks", "*.json")))
    if not task_paths:
        raise SystemExit(f"No task JSONs found under {args.simlife_root}/tasks")

    tasks = [json.load(open(p)) for p in task_paths]

    # Group tasks by video_chain_id. All tasks sharing a chain have identical
    # video_ids; we verify that explicitly to catch corrupt task files early.
    by_chain = collections.defaultdict(list)
    for t in tasks:
        by_chain[int(t["video_chain_id"])].append(t)
    for vc, group in by_chain.items():
        base = tuple(group[0]["video_ids"])
        for t in group[1:]:
            if tuple(t["video_ids"]) != base:
                raise SystemExit(f"Tasks in chain {vc} have inconsistent video_ids")

    # Flatten unit set for Stage A.
    unit_set = set()
    for t in tasks:
        unit_set.update(unit_dir_name(v) for v in t["video_ids"])
    units_sorted = sorted(unit_set)
    unit_durations = load_unit_durations(units_sorted)

    # units.txt + data_units.jsonl
    units_txt = os.path.join(args.out_simlife, "units.txt")
    with open(units_txt, "w") as f:
        for u in units_sorted:
            f.write(u + "\n")

    units_jsonl = os.path.join(args.out_simlife, "data_units.jsonl")
    with open(units_jsonl, "w") as f:
        for u in units_sorted:
            row = {
                "id": u,
                "src_dir": os.path.join("SimLife-Data-HF", "video_units", u),
                "clip_path": os.path.join("data", "clips", u),
                "intermediate_outputs": os.path.join("data", "intermediate", u),
            }
            f.write(json.dumps(row) + "\n")

    # data_chains.jsonl + annotations (one entry per vc)
    chains_jsonl = os.path.join(args.out_simlife, "data_chains.jsonl")
    annotations = {}
    chains_sorted = sorted(by_chain.keys())
    with open(chains_jsonl, "w") as f_chains:
        for vc in chains_sorted:
            group = by_chain[vc]
            base = group[0]
            chain_id = chain_basename(vc)
            video_ids = [unit_dir_name(v) for v in base["video_ids"]]
            base_mem_path = os.path.join("data", "memory_graphs", f"{chain_id}.pkl")
            # Stage B writes one pickle per memory variant — audio (Qwen
            # with the audio modality on) and noaudio (vision-only).
            # ``mem_path`` (without suffix) is kept for backward
            # compatibility with any legacy single-variant pipeline.
            mem_path_audio = os.path.join("data", "memory_graphs", f"{chain_id}_audio.pkl")
            mem_path_noaudio = os.path.join("data", "memory_graphs", f"{chain_id}_noaudio.pkl")

            row = {
                "chain_id": chain_id,
                "video_chain_id": vc,
                "video_ids": video_ids,
                "intermediate_dirs": [
                    os.path.join("data", "intermediate", u) for u in video_ids
                ],
                "mem_path": base_mem_path,                # legacy, unsuffixed
                "mem_path_audio": mem_path_audio,
                "mem_path_noaudio": mem_path_noaudio,
                "task_ids": [int(t["task_id"]) for t in group],
            }
            f_chains.write(json.dumps(row) + "\n")

            # Merge questions across all tasks targeting this chain. Each
            # question carries its own before_clip so control.py can truncate
            # the memory graph per-question.
            qa_list = []
            for task in group:
                before_clip = compute_before_clip(task, unit_durations)
                for q in task["questions"]:
                    qa_list.append(question_to_qa(q, task["task_id"], before_clip))
            annotations[chain_id] = {
                "video_path": "",
                # Default ``mem_path`` points at the audio-variant pickle so
                # control.py works without changes; the noaudio pickle
                # path is exposed alongside for evaluators that route
                # per-question.
                "mem_path": mem_path_audio,
                "mem_path_audio": mem_path_audio,
                "mem_path_noaudio": mem_path_noaudio,
                "video_chain_id": vc,
                "task_ids": [int(t["task_id"]) for t in group],
                "qa_list": qa_list,
            }

    ann_path = os.path.join(args.out_annotations, "simlife.json")
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(
        f"tasks={len(tasks)} chains={len(chains_sorted)} units={len(units_sorted)} "
        f"questions={sum(len(v['qa_list']) for v in annotations.values())}"
    )
    print(f"  -> {units_txt}")
    print(f"  -> {units_jsonl}")
    print(f"  -> {chains_jsonl}")
    print(f"  -> {ann_path}")


if __name__ == "__main__":
    main()
