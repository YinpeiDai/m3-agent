"""Build SimLife manifests for the M3-Agent adapter.

Outputs (under data/simlife/ and data/annotations/):
- units.txt              one "video_XXXXXX" per line, union of all task video_ids
- data_units.jsonl       per-unit precompute manifest (Stage A)
- data_chains.jsonl      per-video-chain assembly manifest (Stage B). Multiple
                         task JSONs that share a video_chain_id collapse into
                         one entry — they have identical video_ids, only their
                         stop_day_position / questions differ. Includes the
                         per-unit ``day_calendar`` (date + day_of_week) read
                         from ``video_chains/<vc>/chain.json`` so Stage B can
                         inject calendar info into memory text without
                         re-parsing chain JSON.
- ../annotations/simlife.json
                         control.py-compatible QA bundle, keyed by chain id.
                         Each entry's qa_list merges questions from every task
                         that targets that chain. Each source question expands
                         into 3 entries (one per hint variant: no_hint,
                         partial_hint, full_hint), each tagged with
                         ``variant='audio'|'noaudio'`` from the task's
                         top-level ``is_omni`` flag.

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
CHAINS_DIR = os.path.join(SIMLIFE_ROOT, "video_chains")

OUT_SIMLIFE = os.path.join(REPO_ROOT, "data", "simlife")
OUT_ANNOTATIONS = os.path.join(REPO_ROOT, "data", "annotations")

CLIP_INTERVAL_SEC = 30
CLIPS_PER_UNIT_OFFSET = 100  # global_clip_id = unit_idx * 100 + k

HINT_LEVELS = ("no_hint", "partial_hint", "full_hint")


def unit_dir_name(video_id):
    return f"video_{int(video_id):06d}"


def chain_basename(video_chain_id):
    return f"vc_{int(video_chain_id):06d}"


def load_chain_json(vc):
    """Read ``video_chains/<vc>/chain.json`` and return the raw dict.

    chain.json is the source of truth for video order, per-unit duration,
    and the day_calendar (date + day_of_week per video_id).
    """
    p = os.path.join(CHAINS_DIR, chain_basename(vc), "chain.json")
    with open(p) as f:
        return json.load(f)


def _answer_letter(answer, options):
    """Return the option letter ('A'..'D') matching ``answer`` in ``options``.

    Falls back to the raw answer string if no exact match is found, so
    pathological data surfaces visibly rather than silently coercing to
    'A'. SimLife's per-question options are short and unique, so an
    exact string match is always sufficient — no normalization needed.
    """
    for i, opt in enumerate(options):
        if opt == answer:
            return chr(ord('A') + i)
    return answer


def question_to_qa_rows(question, task_id, before_clip, variant):
    """Expand one source question into 3 rows (one per hint level).

    Each output row carries:
      - ``question_id``    suffixed with the hint level for uniqueness
      - ``question``       the hint-specific text + options block
      - ``answer``         the option *letter* ('A', 'B', ...). The raw
                           answer string is kept under ``answer_text`` for
                           debugging / human inspection.
      - ``hint``           ``no_hint`` | ``partial_hint`` | ``full_hint``
      - ``variant``        ``audio`` if the source task is_omni else ``noaudio``
      - ``before_clip``    per-task stop point (last visible clip id)

    The new SimLife schema has no separate _vision fields — names like
    "Father Sim" appear directly in question text. control.py's
    ``SIMLIFE_CHARACTER_HINT`` already pre-binds the four mains to fixed
    character ids, so no name-leakage handling is needed here.
    """
    rows = []
    qid_int = int(question["task_question_id"])
    options = question["options"]
    options_str = " | ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(options))
    raw_answer = question["answer"]
    answer_letter = _answer_letter(raw_answer, options)
    qtype = question.get("question_type")
    qfmt = question.get("format")

    for hint in HINT_LEVELS:
        text_key = f"question_{hint}"
        if text_key not in question:
            # Defensive — earlier dataset revisions had different fields. We
            # skip this hint variant rather than silently falling back to
            # another, so missing fields surface as a manifest gap.
            continue
        prompt = f"{question[text_key]}\nOptions: {options_str}"
        rows.append({
            "question_id": f"task_{int(task_id):06d}_Q{qid_int:06d}_{hint}",
            "question": prompt,
            "answer": answer_letter,
            "answer_text": raw_answer,
            "options": options,
            "question_type": qtype,
            "format": qfmt,
            "task_id": int(task_id),
            "task_question_id": qid_int,
            "hint": hint,
            "variant": variant,
            "before_clip": before_clip,
        })
    return rows


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

    # The new task schema doesn't carry ``video_ids`` directly — chain.json
    # is the single source of truth. Group tasks by vc and load each
    # chain's chain.json once. Then attach the chain's video_ids back onto
    # every task in that group so the rest of the pipeline (compute
    # before_clip, etc.) doesn't need to know about the schema split.
    by_chain = collections.defaultdict(list)
    for t in tasks:
        by_chain[int(t["video_chain_id"])].append(t)

    chain_jsons = {vc: load_chain_json(vc) for vc in by_chain.keys()}
    for vc, group in by_chain.items():
        chain_video_ids = chain_jsons[vc]["video_ids"]
        for t in group:
            t["video_ids"] = chain_video_ids

    # Flatten unit set for Stage A.
    unit_set = set()
    for vc, cj in chain_jsons.items():
        unit_set.update(unit_dir_name(v) for v in cj["video_ids"])
    units_sorted = sorted(unit_set)

    # Per-unit duration: prefer chain.json's video_durations_sec (since it's
    # already loaded for the calendar), fall back to per-unit metadata.json
    # for any unit that doesn't appear in any chain we processed.
    unit_durations = {}
    for vc, cj in chain_jsons.items():
        durs = cj.get("video_durations_sec") or []
        for vid, dur in zip(cj["video_ids"], durs):
            unit_durations.setdefault(unit_dir_name(vid), float(dur))
    for u in units_sorted:
        if u in unit_durations:
            continue
        meta_path = os.path.join(args.simlife_root, "video_units", u, "metadata.json")
        with open(meta_path) as f:
            unit_durations[u] = float(json.load(f)["duration_sec"])

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
    n_audio_q = n_noaudio_q = 0
    n_omni_tasks = sum(1 for t in tasks if t.get("is_omni"))
    n_silent_tasks = len(tasks) - n_omni_tasks
    with open(chains_jsonl, "w") as f_chains:
        for vc in chains_sorted:
            group = by_chain[vc]
            chain_id = chain_basename(vc)
            cj = chain_jsons[vc]
            video_ids = [unit_dir_name(v) for v in cj["video_ids"]]
            base_mem_path = os.path.join("data", "memory_graphs", f"{chain_id}.pkl")
            # Stage B writes one pickle per memory variant — audio (Qwen
            # with the audio modality on) and noaudio (vision-only). Both
            # are needed because some tasks targeting this chain are
            # is_omni=True and others may be False.
            mem_path_audio = os.path.join("data", "memory_graphs", f"{chain_id}_audio.pkl")
            mem_path_noaudio = os.path.join("data", "memory_graphs", f"{chain_id}_noaudio.pkl")

            # Stage B reads day_calendar to inject [YYYY-MM-DD, DayOfWeek]
            # into each clip's memory text. We embed it directly in the
            # chain manifest so Stage B doesn't have to re-open chain.json.
            day_calendar = []
            for entry in cj.get("day_calendar", []):
                day_calendar.append({
                    "video_id": unit_dir_name(entry["video_id"]),
                    "day_position": int(entry.get("day_position", 0)),
                    "date": entry.get("date"),
                    "day_of_week": entry.get("day_of_week"),
                })

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
                "day_calendar": day_calendar,
                "video_durations_sec": list(cj.get("video_durations_sec") or []),
            }
            f_chains.write(json.dumps(row) + "\n")

            # Merge questions across all tasks targeting this chain. Each
            # question expands into 3 hint variants; the variant
            # (audio/noaudio) is fixed per task by ``is_omni``.
            qa_list = []
            for task in group:
                before_clip = compute_before_clip(task, unit_durations)
                variant = "audio" if task.get("is_omni") else "noaudio"
                for q in task["questions"]:
                    rows = question_to_qa_rows(q, task["task_id"], before_clip, variant)
                    qa_list.extend(rows)
                    if variant == "audio":
                        n_audio_q += len(rows)
                    else:
                        n_noaudio_q += len(rows)
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
        f"tasks={len(tasks)} (is_omni: audio={n_omni_tasks}, noaudio={n_silent_tasks}) "
        f"chains={len(chains_sorted)} units={len(units_sorted)} "
        f"qa_rows={sum(len(v['qa_list']) for v in annotations.values())} "
        f"(audio={n_audio_q}, noaudio={n_noaudio_q}; 3 hint levels per source question)"
    )
    print(f"  -> {units_txt}")
    print(f"  -> {units_jsonl}")
    print(f"  -> {chains_jsonl}")
    print(f"  -> {ann_path}")


if __name__ == "__main__":
    main()
