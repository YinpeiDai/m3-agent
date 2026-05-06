# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""M3-Agent-Control eval driver.

Reads an annotation JSON (``data/annotations/<dataset>.json``) keyed by
chain id, runs the M3-Agent-Control vLLM model on every question, with
each round routing search queries against the matching chain pickle on
disk, and writes the results as JSONL.

Per-question variant routing: each question's ``variant`` field
(``"audio"`` or ``"noaudio"``) decides which graph pickle to load —
``mem_path_audio`` vs ``mem_path_noaudio`` from the annotation. Missing
labels default to ``audio``.

Sharded execution: ``--num_shards N --shard S`` partitions the work by
chain so that all questions touching the same pickle land in the same
shard (preserves OS-cache + per-worker graph-cache locality).
"""
import argparse
import copy
import json
import multiprocessing
import os
import re
import sys
import time

import openai
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import mmagent.videograph
from mmagent.prompts import prompt_agent_verify_answer_referencing
from mmagent.retrieve import search
from mmagent.utils.chat_api import generate_messages
from mmagent.utils.general import load_video_graph

# load_video_graph unpickles a VideoGraph that was originally pickled
# from the bare ``videograph`` module name (M3-Bench history); alias so
# pickle.load resolves the class.
sys.modules["videograph"] = mmagent.videograph

# ---------------------------------------------------------------------------
# Module config + globals
# ---------------------------------------------------------------------------
processing_config = json.load(open("configs/processing_config.json"))
api_config = json.load(open("configs/api_config.json"))

MODEL_NAME = "models/M3-Agent-Control"
GPT_MODEL = "gpt-4-0125-preview"

_gpt_client = openai.AzureOpenAI(
    azure_endpoint=api_config[GPT_MODEL]["azure_endpoint"],
    api_version=api_config[GPT_MODEL]["api_version"],
    api_key=api_config[GPT_MODEL]["api_key"],
)

SYSTEM_PROMPT = (
    "You are given a question and some relevant knowledge. Your task is to reason "
    "about whether the provided knowledge is sufficient to answer the question. If "
    "it is sufficient, output [Answer] followed by the answer. If it is not "
    "sufficient, output [Search] and generate a query that will be encoded into "
    "embeddings for a vector similarity search. The query will help retrieve "
    "additional information from a memory bank.\n\nQuestion: {question}"
)
INSTRUCTION = (
    "\n\nOutput the answer in the format:\n"
    "Action: [Answer] or [Search]\n"
    "Content: {content}\n\n"
    "If the answer cannot be derived yet, the {content} should be a single search "
    "query that would help retrieve the missing information. The search {content} "
    "needs to be different from the previous.\n"
    "You can get the mapping relationship between character ID and name by using "
    "search query such as: \"What is the name of <character_{i}>\" or \"What is "
    "the character id of {name}\".\n"
    "After obtaining the mapping, it is best to use character ID instead of name "
    "for searching.\n"
    "If the answer can be derived from the provided knowledge, the {content} is "
    "the specific answer to the question. Only name can appear in the answer, not "
    "character ID like <character_{i}>."
)
ACTION_PATTERN = re.compile(r"Action: \[(.*)\].*Content: (.*)", re.DOTALL)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# CPU-affinity / GPT-4 verifier helpers
# ---------------------------------------------------------------------------
def _detect_usable_cpus():
    """Return the number of CPUs this process is *actually allowed* to use.

    ``multiprocessing.cpu_count()`` returns the node's physical CPU total,
    which under SLURM ignores ``--cpus-per-task`` — spawning that many
    workers on a 96-core node with a 1-core allocation would have all
    workers fighting for the single allocated core via cgroups.

    Order of preference:
      1. ``SLURM_CPUS_PER_TASK`` env var (definitive when set by srun).
      2. ``os.sched_getaffinity(0)`` (respects cgroups + taskset on Linux).
      3. ``os.cpu_count()`` as a last resort.
    """
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n and n.isdigit() and int(n) > 0:
        return int(n)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _gpt_call(messages, timeout=30):
    response = _gpt_client.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0,
        timeout=timeout, max_tokens=2048,
    )
    return response.choices[0].message.content, response.usage.total_tokens


def _gpt_call_with_retry(messages, timeout=30, retries=20):
    for i in range(retries):
        try:
            return _gpt_call(messages, timeout)
        except Exception as e:
            time.sleep(20)
            print(f"Retry {i} times, exception: {e} from message {messages}")
    raise RuntimeError(f"Failed to get GPT-4 response after {retries} retries")


def eval_answer(question, predict, ground_truth):
    """Use GPT-4 as the verifier — returns True iff it judges ``predict`` correct."""
    if predict == "":
        return False
    try:
        messages = generate_messages([{
            "type": "text",
            "content": prompt_agent_verify_answer_referencing.format(
                question=question,
                ground_truth_answer=ground_truth,
                agent_answer=predict,
            ),
        }])
        response = _gpt_call_with_retry(messages)
        result = response[0].lower()
    except Exception as e:
        print(f"Error verifying qa: {question} | {e}")
        return False
    return "yes" in result


# ---------------------------------------------------------------------------
# Per-worker graph cache + retrieval consumer
# ---------------------------------------------------------------------------
# Each multiprocessing worker keeps the *raw* loaded VideoGraph (no
# truncation) for every chain pickle it has touched, then deepcopies +
# truncates per call so different before_clip values stay independent.
# With chain-locality sharding all questions in a shard touch only a few
# mem_paths, so the cache stays small but the repeated disk + unpickle
# cost goes to ~0 from the second call onward.
_graph_cache = {}


def _get_truncated_graph(mem_path, before_clip):
    if mem_path not in _graph_cache:
        _graph_cache[mem_path] = load_video_graph(mem_path)
    g = copy.deepcopy(_graph_cache[mem_path])
    if before_clip is not None:
        g.truncate_memory_by_clip(before_clip, False)
    g.refresh_equivalences()
    return g


def consumer(data):
    """One per-question retrieval step. Runs inside the worker pool.

    Parses the latest assistant message; if it's [Answer], marks the
    question finished. Otherwise treats it as a [Search] query, loads
    (cached) the chain pickle, runs ``search``, and appends the result
    text as the next user message so the agent can continue.
    """
    if data["finish"]:
        return data

    before_clip = data.get("before_clip", None)
    response = data["conversations"][-1]["content"]
    match = ACTION_PATTERN.search(response.split("</think>")[-1])
    if match:
        action, content = match.group(1), match.group(2)
    else:
        action, content = "Search", None

    if action == "Answer":
        data["response"] = content
        data["finish"] = True
        return data

    new_memories = {}
    if content:
        mem_node = _get_truncated_graph(data["mem_path"], before_clip)
        if "character id" in content:
            memories, _, _ = search(
                mem_node, content, [], mem_wise=True, topk=20, before_clip=before_clip,
            )
        else:
            memories, current_clips, _ = search(
                mem_node, content, data["currenr_clips"],
                threshold=0.5, topk=processing_config["topk"], before_clip=before_clip,
            )
            data["currenr_clips"] = current_clips
        new_memories.update(memories)

    search_result = (
        "Searched knowledge: "
        + json.dumps(new_memories, ensure_ascii=False)
            .encode("utf-8", "ignore").decode("utf-8")
    )
    if not new_memories:
        search_result += "\n(The search result is empty. Please try searching from another perspective.)"
    data["conversations"].append({"role": "user", "content": search_result})
    return data


# ---------------------------------------------------------------------------
# Annotation loading + sharding
# ---------------------------------------------------------------------------
def _chain_key(v):
    """Canonical chain identifier used for sharding (audio path is stable)."""
    return v.get("mem_path_audio") or v.get("mem_path")


def _build_batches(datas, my_chain_keys, batch_size):
    """Flatten the chain-keyed annotation dict into per-batch lists of
    per-question entries, honouring each question's ``variant`` label.
    """
    batched, batch = [], []
    use_counts = {"audio": 0, "noaudio": 0}
    skipped_no_path = 0
    for v in datas.values():
        if _chain_key(v) not in my_chain_keys:
            continue
        chain_paths = {
            "audio": v.get("mem_path_audio") or v.get("mem_path"),
            "noaudio": v.get("mem_path_noaudio"),
        }
        for qa in v["qa_list"]:
            label = qa.get("variant") or qa.get("mode") or "audio"
            if label not in ("audio", "noaudio"):
                label = "audio"
            mem_path = chain_paths.get(label)
            if mem_path is None:
                skipped_no_path += 1
                continue
            use_counts[label] += 1
            entry = {
                "id": qa["question_id"],
                "variant": label,
                "mem_path": mem_path,
                "question": qa["question"],
                "answer": qa["answer"],
            }
            if "before_clip" in qa:
                entry["before_clip"] = qa["before_clip"]
            batch.append(entry)
            if len(batch) == batch_size:
                batched.append(batch)
                batch = []
    if batch:
        batched.append(batch)
    return batched, use_counts, skipped_no_path


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/annotations/robot.json")

    # Sharding: split unique chain pickles across N shards by deterministic
    # index-modulo, so all questions touching the same pickle stay together
    # (preserves disk/OS-cache locality and lets each shard use a smaller
    # per-process graph cache).
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel eval shards.")
    parser.add_argument("--shard", type=int, default=0,
                        help="Which shard this process owns, in [0, num_shards).")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Override results path. Defaults to "
                             "data/results/<dataset>.jsonl, or "
                             "data/results/<dataset>.shard<S>of<N>.jsonl when sharded.")

    # GPU-utilisation knobs.
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="vLLM tensor-parallel size. Set to 1 if your job has only 1 GPU.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="Fraction of GPU memory vLLM grabs for weights + KV cache.")
    parser.add_argument("--max_num_seqs", type=int, default=512,
                        help="Max concurrent sequences vLLM keeps in flight. "
                             "Should be >= processing_config['batch_size'].")
    parser.add_argument("--no_prefix_caching", action="store_true",
                        help="Disable vLLM prefix caching. Keep ON (default) for SimLife "
                             "— the system_prompt + instruction are identical across "
                             "every question, so KV-cache reuse cuts prefill cost.")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Cap vLLM's context length. Lower this if KV cache "
                             "is pushing the OOM line.")
    return parser.parse_args()


def _resolve_output_path(args):
    dataset_name = os.path.basename(args.data_file).rsplit(".", 1)[0]
    if args.output_path:
        out = args.output_path
    elif args.num_shards > 1:
        out = os.path.join(
            "data/results",
            f"{dataset_name}.shard{args.shard:03d}of{args.num_shards:03d}.jsonl",
        )
    else:
        out = os.path.join("data/results", f"{dataset_name}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


def _build_llm(args):
    llm_kwargs = dict(
        model=MODEL_NAME,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=not args.no_prefix_caching,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    model = LLM(**llm_kwargs)
    print(f"vLLM config: TP={args.tensor_parallel_size} "
          f"max_num_seqs={args.max_num_seqs} "
          f"gpu_mem_util={args.gpu_memory_utilization} "
          f"prefix_caching={not args.no_prefix_caching} "
          f"batch_size(config)={processing_config['batch_size']}")
    return model


def _print_summary(args, output_path, results, batches, timing, total_elapsed,
                   max_num_seqs):
    """Final per-shard summary: per-variant accuracy + per-stage timing."""
    by_variant = {}
    for r in results:
        v = r.get("variant", "?")
        bucket = by_variant.setdefault(v, [0, 0])
        bucket[0] += 1
        if r.get("gpt_eval"):
            bucket[1] += 1
    n_total = len(results)
    n_correct = sum(b[1] for b in by_variant.values())
    avg_active = (timing["active_seqs_sum"] / timing["rounds"]) if timing["rounds"] else 0.0
    pct = lambda part: (part / total_elapsed * 100) if total_elapsed else 0.0

    print()
    print("=" * 60)
    print(f"shard {args.shard}/{args.num_shards}  done in {total_elapsed:.1f}s "
          f"({total_elapsed / 60:.1f} min) — wrote {output_path}")
    print(f"  total entries: {n_total}    correct: {n_correct} "
          f"({(n_correct / n_total * 100) if n_total else 0:.1f}%)")
    for v in sorted(by_variant):
        t, c = by_variant[v]
        pct_v = (c / t * 100) if t else 0
        print(f"    variant={v:8s}  {t} entries  correct={c}  ({pct_v:.1f}%)")
    print(f"  batches:      {len(batches)}    rounds across batches: {timing['rounds']}")
    print(f"  vLLM gen:     {timing['gen']:.1f}s ({pct(timing['gen']):.1f}% wall)")
    print(f"  consumer:     {timing['consumer']:.1f}s ({pct(timing['consumer']):.1f}% wall)")
    print(f"  eval (gpt-4): {timing['eval']:.1f}s ({pct(timing['eval']):.1f}% wall)")
    print(f"  avg active seqs/round: {avg_active:.1f} "
          f"(vs max_num_seqs={max_num_seqs}, "
          f"batch_size={processing_config['batch_size']})")
    print(f"  per-entry avg: {(total_elapsed / n_total) if n_total else 0:.2f}s")
    print("=" * 60)


def _vllm_generate_round(model, batched_data, total_round, round_idx):
    """Build prompts for the still-active entries and run vLLM once."""
    vllm_inputs = []
    for entry in batched_data:
        if entry["finish"]:
            continue
        entry["conversations"][-1]["content"] += INSTRUCTION
        if round_idx == total_round - 1:
            entry["conversations"][-1]["content"] += (
                "\n(The Action of this round must be [Answer]. "
                "If there is insufficient information, you can make reasonable guesses.)"
            )
        text = tokenizer.apply_chat_template(
            entry["conversations"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        vllm_inputs.append({"prompt_token_ids": text})
    outputs = model.generate(
        prompts=vllm_inputs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    # Splice generated text back into the still-active entries.
    out_idx = 0
    for entry in batched_data:
        if entry["finish"]:
            continue
        entry["conversations"].append(
            {"role": "assistant", "content": outputs[out_idx].outputs[0].text}
        )
        out_idx += 1
    assert out_idx == len(vllm_inputs)
    return len(vllm_inputs)


def main():
    args = _parse_args()
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"Bad shard config: shard={args.shard} num_shards={args.num_shards}")

    output_path = _resolve_output_path(args)
    model = _build_llm(args)

    # Annotation loading + sharding.
    datas = json.load(open(args.data_file))
    all_chain_keys = sorted({_chain_key(v) for v in datas.values() if _chain_key(v)})
    my_chain_keys = {p for i, p in enumerate(all_chain_keys)
                     if i % args.num_shards == args.shard}
    print(f"shard {args.shard}/{args.num_shards}: "
          f"{len(my_chain_keys)}/{len(all_chain_keys)} chains -> {output_path}")

    batches, use_counts, skipped = _build_batches(
        datas, my_chain_keys, processing_config["batch_size"],
    )
    n_total_entries = sum(len(b) for b in batches)
    print(f"shard {args.shard}: {n_total_entries} entries in {len(batches)} batches "
          f"(audio={use_counts['audio']}, noaudio={use_counts['noaudio']}"
          f"{f', skipped_no_path={skipped}' if skipped else ''})")

    # Persistent worker pool: SLURM-aware sizing avoids spawning a worker
    # per physical core on a node where the job only has --cpus-per-task=K.
    usable_cpus = _detect_usable_cpus()
    pool_size = max(1, min(usable_cpus, processing_config["batch_size"]))
    pool = multiprocessing.Pool(processes=pool_size)
    print(f"consumer pool: {pool_size} workers "
          f"(usable_cpus={usable_cpus}, "
          f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')}, "
          f"batch_size={processing_config['batch_size']}; "
          f"graph cache shared per-worker across rounds)")

    # Per-stage timers — accumulated across every batch so the final
    # summary breaks down where wall-clock went.
    total_t0 = time.time()
    timing = {
        "gen": 0.0,             # vLLM model.generate()
        "consumer": 0.0,        # parse + retrieve + truncate + refresh
        "eval": 0.0,            # GPT-4 verifier round-trip
        "rounds": 0,            # round invocations across all batches
        "active_seqs_sum": 0,   # cumulative active sequences across rounds
        "questions_done": 0,
    }

    results = []
    n_correct_so_far = 0

    pbar = tqdm(batches, desc="batches", unit="batch", dynamic_ncols=True)
    total_round = processing_config["total_round"]
    for batch_idx, batch in enumerate(pbar):
        batch_t0 = time.time()
        batch_time = {"gen": 0.0, "consumer": 0.0, "eval": 0.0}

        for entry in batch:
            entry["conversations"] = [
                {"role": "system",
                 "content": SYSTEM_PROMPT.format(question=entry["question"])},
                {"role": "user", "content": "Searched knowledge: {}"},
            ]
            entry["finish"] = False
            entry["currenr_clips"] = []

        for round_idx in range(total_round):
            gen_t0 = time.time()
            n_active = _vllm_generate_round(model, batch, total_round, round_idx)
            gen_dt = time.time() - gen_t0
            timing["gen"] += gen_dt
            batch_time["gen"] += gen_dt
            timing["active_seqs_sum"] += n_active
            timing["rounds"] += 1

            cons_t0 = time.time()
            batch = pool.map(consumer, batch)
            cons_dt = time.time() - cons_t0
            timing["consumer"] += cons_dt
            batch_time["consumer"] += cons_dt

            pbar.set_postfix(
                round=f"{round_idx + 1}/{total_round}",
                live=f"{n_active} active",
                gen=f"{gen_dt:.1f}s",
                cons=f"{cons_dt:.1f}s",
                refresh=False,
            )

        # GPT-4 eval per question (rate-limited via the inner sleep).
        eval_t0 = time.time()
        for entry in batch:
            if "response" in entry:
                entry["gpt_eval"] = eval_answer(
                    entry["question"], entry["response"], entry["answer"],
                )
                time.sleep(0.5)
            else:
                entry["gpt_eval"] = False
            results.append(entry)
            if entry["gpt_eval"]:
                n_correct_so_far += 1
        eval_dt = time.time() - eval_t0
        timing["eval"] += eval_dt
        batch_time["eval"] += eval_dt
        timing["questions_done"] += len(batch)

        batch_dt = time.time() - batch_t0
        avg_per_batch = (time.time() - total_t0) / (batch_idx + 1)
        pbar.set_postfix(
            batch=f"{batch_dt:.1f}s",
            gen=f"{batch_time['gen']:.1f}s",
            cons=f"{batch_time['consumer']:.1f}s",
            eval=f"{batch_time['eval']:.1f}s",
            avg_batch=f"{avg_per_batch:.1f}s",
            acc=f"{n_correct_so_far}/{timing['questions_done']}",
        )

    pool.close()
    pool.join()

    total_elapsed = time.time() - total_t0

    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _print_summary(args, output_path, results, batches, timing, total_elapsed,
                   args.max_num_seqs)


if __name__ == "__main__":
    main()
