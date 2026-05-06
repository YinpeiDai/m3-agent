"""Combine per-shard eval results from sharded control.py runs.

Reads ``data/results/<dataset>.shard*.jsonl`` and writes the concatenated
results to ``data/results/<dataset>.jsonl`` plus a one-line summary with
overall accuracy. Safe to re-run; existing combined file is overwritten.

Each shard line is a single JSON object emitted by ``control.py``; we merge
without deduplication beyond ``question_id``. If the same ``question_id``
appears in multiple shards (shouldn't happen given mem_path-based sharding),
the last one wins.
"""
import argparse
import glob
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="simlife",
                        help="Basename matching control.py's <dataset>.shard*.jsonl files")
    parser.add_argument("--results_dir", default="data/results")
    parser.add_argument("--out_path", default=None,
                        help="Override output path; defaults to <results_dir>/<dataset>.jsonl")
    parser.add_argument("--require_complete", action="store_true",
                        help="Fail if any shard file is missing.")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, f"{args.dataset}.shard*.jsonl")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise SystemExit(f"No shard files matching {pattern!r}")

    # Detect (S, N) from filenames so we can flag gaps.
    shard_re = re.compile(rf"{re.escape(args.dataset)}\.shard(\d+)of(\d+)\.jsonl$")
    seen = []
    expected_total = None
    for p in shard_paths:
        m = shard_re.search(os.path.basename(p))
        if not m:
            continue
        s, n = int(m.group(1)), int(m.group(2))
        seen.append(s)
        if expected_total is None:
            expected_total = n
        elif n != expected_total:
            raise SystemExit(f"Inconsistent num_shards in filenames: {p}")
    missing = []
    if expected_total is not None:
        missing = sorted(set(range(expected_total)) - set(seen))
        if missing and args.require_complete:
            raise SystemExit(f"Missing shards: {missing}")

    out_path = args.out_path or os.path.join(args.results_dir, f"{args.dataset}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Key by (id, variant) so the same question evaluated under both
    # audio + noaudio keeps both rows; legacy single-variant entries
    # without a 'variant' field land under variant=''.
    by_key = {}
    for p in shard_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row.get("id") or row.get("question_id")
                variant = row.get("variant", "")
                by_key[(qid, variant)] = row

    by_variant = {}  # variant -> [total, correct]
    for (qid, variant), r in by_key.items():
        bucket = by_variant.setdefault(variant or "?", [0, 0])
        bucket[0] += 1
        if r.get("gpt_eval"):
            bucket[1] += 1
    n_total = sum(b[0] for b in by_variant.values())
    n_correct = sum(b[1] for b in by_variant.values())

    with open(out_path, "w") as f:
        for key in sorted(by_key.keys()):
            f.write(json.dumps(by_key[key], ensure_ascii=False) + "\n")

    overall_acc = (n_correct / n_total) if n_total else 0.0
    print(f"shards={len(shard_paths)} entries={n_total} correct={n_correct} "
          f"accuracy={overall_acc:.4f}")
    for variant in sorted(by_variant):
        t, c = by_variant[variant]
        acc = (c / t) if t else 0.0
        label = variant or "(no variant tag)"
        print(f"  variant={label:18s} entries={t:6d} correct={c:6d} accuracy={acc:.4f}")
    print(f"  wrote {out_path}")
    if missing:
        print(f"  WARNING: missing shards {missing} (re-run those then re-combine)")


if __name__ == "__main__":
    main()
