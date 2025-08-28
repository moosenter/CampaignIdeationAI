import json, random, pathlib, itertools, math
pathlib.Path("data").mkdir(exist_ok=True)
random.seed(42)

INDUSTRIES = [
  "FMCG snacks","Retail fashion","Banking","Telco","Automotive",
  "QSR","Beauty","Travel","Education","Healthcare"
]
AGE_BANDS = ["18-24","25-34","35-44","45-60"]
GEO = ["TH"]  # keep Thailand focus; add "SEA" if needed
OBJECTIVES = ["awareness","acquisition","retention","loyalty","upsell"]
BUDGETS_THB = [300_000, 600_000, 1_000_000, 1_500_000, 3_000_000]
TONES = ["playful","premium","trustworthy","innovative","minimal"]
MANDATORY = [[],["LINE OA"],["TikTok"],["Facebook","Instagram"],["Email","LINE OA"]]
BANNED = [[],["Twitter/X"],["Out of Home"],["YouTube Shorts"]]

def sample_brief():
    industry = random.choice(INDUSTRIES)
    age = random.choice(AGE_BANDS)
    obj = random.choice(OBJECTIVES)
    bud = random.choice(BUDGETS_THB)
    tone = random.choice(TONES)
    mandatory = random.choice(MANDATORY)
    banned = random.choice(BANNED) if not mandatory else []
    return {
      "industry": industry,
      "audience": {"geo": random.choice(GEO), "age": age},
      "budget_thb": bud,
      "objective": obj,
      "constraints": {"brand_tone": tone, "mandatory_channels": mandatory, "banned_channels": banned}
    }

def make(n=5000, train_ratio=0.9):
    seen = set()
    briefs = []
    while len(briefs) < n:
        b = sample_brief()
        key = (b["industry"], b["audience"]["age"], b["budget_thb"], b["objective"],
               tuple(b["constraints"]["mandatory_channels"]), tuple(b["constraints"]["banned_channels"]))
        if key in seen: 
            continue
        seen.add(key)
        briefs.append({"input": b})  # output to be filled later
    random.shuffle(briefs)
    split = int(train_ratio*len(briefs))
    with open("data/briefs_train.jsonl","w",encoding="utf-8") as f:
        for x in briefs[:split]: f.write(json.dumps(x, ensure_ascii=False)+"\n")
    with open("data/briefs_val.jsonl","w",encoding="utf-8") as f:
        for x in briefs[split:]: f.write(json.dumps(x, ensure_ascii=False)+"\n")
    print(f"Wrote {split} train and {len(briefs)-split} val briefs.")

if __name__ == "__main__":
    make(n=8000)  # change size here
