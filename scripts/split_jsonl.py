from __future__ import annotations
import argparse, json, os, random, sys, hashlib
from typing import Any, Dict, List, Tuple, Iterable, Optional
from collections import defaultdict, Counter
from pathlib import Path

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                bad += 1
    if bad:
        print(f"[WARN] Skipped {bad} malformed line(s).", file=sys.stderr)
    return rows

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_val_size(val_size: float|int, n: int) -> int:
    if isinstance(val_size, float):
        if not (0 < val_size <= 1):
            raise ValueError("--val-size float must be in (0,1]")
        k = int(round(n * val_size))
    else:
        k = int(val_size)
    k = max(1, min(k, n-1)) if n >= 2 else 0
    return k

def dedupe(rows: List[Dict[str, Any]], key: Optional[str]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        if key and key in r:
            k = ("key", str(r[key]))
        else:
            # hash stable text of the object
            k = ("hash", hashlib.md5(json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest())
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def stratified_split(rows: List[Dict[str, Any]], key: str, k_val: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g = r.get(key, "_MISSING_")
        buckets[str(g)].append(r)

    val, train = [], []
    # target global val size
    n = len(rows)
    frac = k_val / max(1, n)
    for g, group in buckets.items():
        grp = list(group)
        rng.shuffle(grp)
        k_g = max(1, int(round(len(grp) * frac))) if len(grp) > 1 else min(1, k_val) if len(grp)==1 and k_val>0 else 0
        val.extend(grp[:k_g])
        train.extend(grp[k_g:])

    # If rounding drifted, fix counts
    # Too many in val -> move extras to train
    if len(val) > k_val:
        rng.shuffle(val)
        move = len(val) - k_val
        train.extend(val[-move:])
        val = val[:-move]
    # Too few in val -> move from train
    elif len(val) < k_val:
        rng.shuffle(train)
        move = min(k_val - len(val), len(train))
        val.extend(train[:move])
        train = train[move:]
    return train, val

def random_split(rows: List[Dict[str, Any]], k_val: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    data = list(rows)
    rng.shuffle(data)
    val = data[:k_val]
    train = data[k_val:]
    return train, val

def compute_stats(rows: List[Dict[str, Any]], stratify_key: Optional[str]) -> Dict[str, Any]:
    stats = {"count": len(rows)}
    if stratify_key:
        c = Counter(str(r.get(stratify_key, "_MISSING_")) for r in rows)
        stats["by_group"] = dict(c.most_common())
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL (e.g., train_synth_clean.jsonl)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--val-size", required=True, type=float, help="Validation size: fraction (0,1] or integer (>=1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify-key", default=None, help="JSON key to stratify by (optional)")
    ap.add_argument("--dedupe-key", default=None, help="JSON key to dedupe by (optional)")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        print("[ERR] No rows found.", file=sys.stderr)
        sys.exit(2)

    if args.dedupe_key is not None:
        before = len(rows)
        rows = dedupe(rows, args.dedupe_key)
        after = len(rows)
        if after < before:
            print(f"[INFO] Deduped: {before} -> {after}")

    k_val = normalize_val_size(args.val_size, len(rows))

    if args.stratify_key:
        train, val = stratified_split(rows, args.stratify_key, k_val, seed=args.seed)
    else:
        train, val = random_split(rows, k_val, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_path = outdir / "train.jsonl"
    val_path = outdir / "val.jsonl"
    write_jsonl(train_path.as_posix(), train)
    write_jsonl(val_path.as_posix(), val)

    stats = {
        "input": args.input,
        "outdir": args.outdir,
        "seed": args.seed,
        "val_size": args.val_size,
        "actual_counts": {"train": len(train), "val": len(val)},
        "train_stats": compute_stats(train, args.stratify_key),
        "val_stats": compute_stats(val, args.stratify_key),
        "stratify_key": args.stratify_key,
    }
    with open(outdir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {train_path} ({len(train)}) and {val_path} ({len(val)})")
    print(f"[OK] Stats -> {outdir/'split_stats.json'}")

if __name__ == "__main__":
    main()