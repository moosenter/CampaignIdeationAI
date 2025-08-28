import json, pathlib, math, difflib
from tqdm import tqdm

IN_PATH = "data/train_synth.jsonl"
OUT_PATH = "data/train_synth_clean.jsonl"

# very simple near-dup key + fuzzy similarity on (title + big_idea)
def pack_text(rec):
    o = rec["output"]; 
    title = o.get("concept_title",""); idea = o.get("big_idea",""); km = o.get("key_message","")
    return f"{title} || {idea} || {km}"

def main():
    seen_keys = set()
    kept = []
    for line in tqdm(open(IN_PATH,"r",encoding="utf-8")):
        rec = json.loads(line)
        o = rec["output"]; i = rec["input"]
        # basic gating
        if len(o.get("channels",[])) == 0 or not o.get("concept_title"):
            continue
        if not (1 <= o.get("timeline_weeks", 0) <= 24):
            continue
        # coarse dedup key
        key = (i["industry"], i["objective"], tuple(sorted([c["name"] for c in o.get("channels",[]) if "name" in c]))[:3])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(rec)

    # fuzzy prune near-duplicates (O(n^2) on small batches; shard if large)
    kept2 = []
    texts = []
    for rec in kept:
        t = pack_text(rec)
        is_dup = False
        for prev in texts[-500:]:  # local window
            if difflib.SequenceMatcher(None, t, prev).ratio() > 0.92:
                is_dup = True; break
        if not is_dup:
            kept2.append(rec); texts.append(t)

    pathlib.Path(OUT_PATH).parent.mkdir(exist_ok=True, parents=True)
    with open(OUT_PATH,"w",encoding="utf-8") as f:
        for rec in kept2:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    print(f"Kept {len(kept2)} / {len(kept)} after fuzzy dedupe; wrote -> {OUT_PATH}")

if __name__ == "__main__":
    main()
