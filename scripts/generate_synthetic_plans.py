import os, json, math, time, random, pathlib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, jsonschema
from typing import Optional, Any, Dict, Tuple

BASE_MODEL = os.getenv("BASE_MODEL","meta-llama/Meta-Llama-3.1-8B-Instruct")
BRIEFS_PATH = os.getenv("BRIEFS_PATH","data/briefs_train.jsonl")
OUT_PATH = os.getenv("OUT_PATH","data/train_synth.jsonl")
HF_TOKEN = os.getenv("HF_TOKEN")
SCHEMA_PATH = "schema/campaign.schema.json"

SYS = ("You are a senior marketing strategist for Thailand. "
       "Return ONLY a single JSON object that strictly follows the provided schema. "
       "No prose, no markdown—JSON only.")

def load_briefs(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            yield j["input"]

def build_user(brief):
    return (f"Brief:\n- Industry: {brief['industry']}\n"
            f"- Audience: {json.dumps(brief['audience'],ensure_ascii=False)}\n"
            f"- Budget (THB): {brief['budget_thb']}\n"
            f"- Objective: {brief['objective']}\n"
            f"- Constraints: {json.dumps(brief.get('constraints',{}),ensure_ascii=False)}\n\n"
            f"JSON fields to produce: concept_title, big_idea, key_message, channels[], assets[], "
            f"timeline_weeks, budget_split[], kpis{{}}")

def build_chat(user):
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYS}"
            f"\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user}"
            f"\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")

def try_parse_json(txt: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse a JSON object from a mixed transcript.
    Strategy:
      1) Try parsing the whole string.
      2) Scan for balanced {...} blocks (quote-aware).
      3) Prefer blocks that appear after the last 'assistant' marker.
      4) Among candidates, pick the largest valid JSON.
      5) Fallback: attempt json-repair on the longest candidate.
    """
    s = txt.strip()

    # 1) Quick path: the whole string is JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Gather balanced {...} candidates (quote/escape aware)
    candidates: list[Tuple[int, int, str]] = []
    in_str = False
    esc = False
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidates.append((start, i + 1, s[start:i + 1]))
                        start = -1

    if not candidates:
        # 5) Last-ditch: try to repair the naive slice (if any)
        try:
            from json_repair import repair_json
            i1, i2 = s.find('{'), s.rfind('}')
            if i1 != -1 and i2 != -1 and i2 > i1:
                return json.loads(repair_json(s[i1:i2+1]))
        except Exception:
            return None
        return None

    # 3) Prefer candidates after the last 'assistant' marker
    anchor = s.lower().rfind("assistant")
    scoped = [c for c in candidates if c[0] >= (anchor if anchor != -1 else 0)]
    if scoped:
        candidates = scoped  # restrict to assistant region if present

    # 4) Try to parse candidates; choose the largest valid JSON
    best: Tuple[int, Dict[str, Any]] | None = None
    for _, _, chunk in candidates:
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if best is None or len(chunk) > best[0]:
            best = (len(chunk), obj)
    if best:
        return best[1]

    # 5) Fallback: json-repair on the longest candidate
    try:
        from json_repair import repair_json
        longest = max(candidates, key=lambda t: len(t[2]))[2]
        fixed = repair_json(longest)
        return json.loads(fixed)
    except Exception:
        return None

def normalize_plan(p):
    # keep plan tidy; ensure budget split ~ 1.0
    if "budget_split" in p and isinstance(p["budget_split"], list):
        s = sum(x[1] for x in p["budget_split"] if isinstance(x,list) and len(x)==2 and isinstance(x[1], (int,float)))
        if s > 0:
            p["budget_split"] = [[x[0], round(x[1]/s, 2)] for x in p["budget_split"] if len(x)==2]
    return p

def _default_activation_for(name: str) -> str:
    name_l = name.lower()
    if "facebook" in name_l:
        return "Paid + organic posts with short-form video"
    if "instagram" in name_l:
        return "Reels + Stories with branded content"
    if "tiktok" in name_l:
        return "Short-form UGC challenge + creator collab"
    if name_l in ("line", "line oa", "line official account"):
        return "OA broadcast + coupon/stamp card"
    if "influencer" in name_l:
        return "Creators produce seeded content"
    if "outdoor" in name_l or "ooh" in name_l:
        return "Billboards / OOH placements"
    return "Channel activation plan"

def normalize_to_schema(plan: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(plan)

    # assets -> list[str]
    if isinstance(out.get("assets"), list):
        new_assets: List[str] = []
        for a in out["assets"]:
            if isinstance(a, str):
                new_assets.append(a)
            elif isinstance(a, dict):
                t = a.get("type") or a.get("name") or "Asset"
                f = a.get("format") or a.get("description") or ""
                s = f"{t}: {f}".strip(": ").strip()
                new_assets.append(s if s else t)
        out["assets"] = new_assets

    # budget_split -> list[[label, fraction]]
    new_bs: List[List[Any]] = []
    bs = out.get("budget_split", [])
    if isinstance(bs, list) and bs:
        tmp_amounts: List[Tuple[str, float]] = []
        perc_pairs: List[Tuple[str, float]] = []

        for item in bs:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], (int, float)):
                tmp_amounts.append((str(item[0]), float(item[1])))
            elif isinstance(item, dict):
                label = item.get("channel") or item.get("name") or item.get("label") or "Other"
                if "percentage" in item and isinstance(item["percentage"], (int, float)):
                    perc_pairs.append((label, float(item["percentage"]) / 100.0))
                elif "allocation" in item and isinstance(item["allocation"], (int, float)):
                    tmp_amounts.append((label, float(item["allocation"])))

        if perc_pairs:
            s = sum(v for _, v in perc_pairs) or 1.0
            new_bs = [[k, round(v / s, 2)] for k, v in perc_pairs]
        elif tmp_amounts:
            s = sum(v for _, v in tmp_amounts) or 1.0
            new_bs = [[k, round(v / s, 2)] for k, v in tmp_amounts]

    out["budget_split"] = new_bs

    # channels -> list of {"name","activation","kpis"}
    fixed_channels: List[Dict[str, Any]] = []
    for ch in out.get("channels", []):
        if isinstance(ch, dict):
            name = ch.get("name") or ch.get("channel") or "Channel"
            activation = ch.get("activation") or _default_activation_for(name)
            kpis = ch.get("kpis")
            if not kpis:
                kpis = {}
                # migrate basic metrics to kpis
                for k in ("reach", "ctr", "cpl", "redemption", "membership_signup"):
                    if k in ch:
                        kpis[k] = ch[k]
            fixed_channels.append({"name": name, "activation": activation, "kpis": kpis})
        elif isinstance(ch, str):
            fixed_channels.append({"name": ch, "activation": _default_activation_for(ch), "kpis": {}})
    out["channels"] = fixed_channels

    # required keys
    out.setdefault("concept_title", "")
    out.setdefault("big_idea", "")
    out.setdefault("key_message", "")
    out.setdefault("timeline_weeks", 4)
    out.setdefault("kpis", {})

    return out

def main():
    schema = json.load(open(SCHEMA_PATH))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
    model.eval()

    pathlib.Path(os.path.dirname(OUT_PATH)).mkdir(exist_ok=True, parents=True)
    out = open(OUT_PATH,"w",encoding="utf-8")

    briefs = list(load_briefs(BRIEFS_PATH))
    rng = random.Random(123)

    for brief in tqdm(briefs, total=len(briefs)):
        # small sampling jitter to diversify outputs
        temp = rng.uniform(0.6, 0.9)
        top_p = rng.uniform(0.85, 0.95)
        mx=1024
        user = build_user(brief)
        prompt = build_chat(user)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=mx, do_sample=True, temperature=temp, top_p=top_p)
        txt = tok.decode(out_ids[0], skip_special_tokens=True)
        js = try_parse_json(txt)
        if not js:
            continue
        js = normalize_plan(js)
        
        js = normalize_to_schema(js)

        # validate; skip if invalid
        try:
            jsonschema.validate(js, schema)
        except Exception:
            continue

        out.write(json.dumps({"input": brief, "output": js}, ensure_ascii=False)+"\n")

    out.close()
    print("Done ->", OUT_PATH)

if __name__ == "__main__":
    main()
