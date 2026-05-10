"""Combine per-shard eval results from sharded control.py runs.

Reads ``data/results/<dataset>.shard*.jsonl`` and writes the concatenated
results to ``data/results/<dataset>.jsonl`` plus a one-line summary with
per-variant/per-hint counts. No correctness scoring here — control.py
collects raw answers; grading happens in a downstream evaluator.

Each shard line is a single JSON object emitted by ``control.py``. Rows
are deduped by ``(question_id, variant, hint)`` so that running the same
question under different hint levels keeps all rows. If the same key
appears in multiple shards (shouldn't happen given chain-based
sharding), the last one wins.
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

    # Key by (id, variant, hint) so the same question evaluated under
    # different hint levels keeps all rows. Legacy entries without
    # variant/hint land under sentinel keys.
    by_key = {}
    for p in shard_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row.get("id") or row.get("question_id")
                key = (qid, row.get("variant", ""), row.get("hint", ""))
                by_key[key] = row

    by_variant = {}
    by_hint = {}
    for (_qid, variant, hint), _r in by_key.items():
        by_variant[variant or "?"] = by_variant.get(variant or "?", 0) + 1
        by_hint[hint or "?"] = by_hint.get(hint or "?", 0) + 1
    n_total = len(by_key)

    with open(out_path, "w") as f:
        for key in sorted(by_key.keys()):
            f.write(json.dumps(by_key[key], ensure_ascii=False) + "\n")

    print(f"shards={len(shard_paths)} entries={n_total}")
    for variant in sorted(by_variant):
        label = variant or "(no variant tag)"
        print(f"  variant={label:18s} entries={by_variant[variant]:6d}")
    for hint in ("no_hint", "partial_hint", "full_hint"):
        if hint in by_hint:
            print(f"  hint={hint:14s} entries={by_hint[hint]:6d}")
    for hint in sorted(set(by_hint) - {"no_hint", "partial_hint", "full_hint"}):
        print(f"  hint={hint:14s} entries={by_hint[hint]:6d}")
    print(f"  wrote {out_path}")
    if missing:
        print(f"  WARNING: missing shards {missing} (re-run those then re-combine)")


if __name__ == "__main__":
    main()
