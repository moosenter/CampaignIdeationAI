from typing import Optional, Any, Dict, Tuple, List
import re
import json


def extract_first_json_block(text: str) -> str | None:
    """Return substring of the first top-level JSON object if found."""
    s = text.find("{"); e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s:e+1]

def normalize_budget_split(plan: Dict[str, Any]) -> None:
    """Normalize budget_split weights to sum ~1.0 (in-place)."""
    items = plan.get("budget_split")
    if not isinstance(items, list):
        return
    total = 0.0
    acc = []
    for it in items:
        if isinstance(it, list) and len(it) == 2 and isinstance(it[1], (int,float)):
            acc.append(it); total += float(it[1])
    if total > 0:
        plan["budget_split"] = [[k, round(v/total, 2)] for k, v in acc]

def safe_load_json(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        return None
    
# ---------- helpers

def _default_activation_for(name: str) -> str:
    n = (name or "").lower()
    if "line" in n: return "OA broadcast + coupon/stamp card"
    if "facebook" in n: return "Paid + organic posts with short-form video"
    if "instagram" in n: return "Reels + Stories with branded content"
    if "tiktok" in n: return "Short-form UGC challenge + creator collab"
    if "twitter" in n or "x" == n: return "Community engagement threads"
    if "email" in n: return "Newsletter & lifecycle automations"
    return "Channel activation plan"

_pct = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$")

def _to_fraction(x: Any) -> float:
    """
    Accepts 60, 0.6, '60', '60%', '0.6' -> returns 0..1
    """
    if isinstance(x, (int, float)):
        return float(x) if 0 <= x <= 1 else float(x)/100.0
    if isinstance(x, str):
        m = _pct.match(x)
        if m:
            v = float(m.group(1))
            return v if 0 <= v <= 1 else v/100.0
    return 0.0

def _renorm_pairs(pairs: List[Tuple[str, float]], decimals: int = 2) -> List[List[Any]]:
    """
    Normalize to sum ~1.0, round, then fix rounding drift on the largest item.
    """
    s = sum(v for _, v in pairs)
    if s <= 0:
        return []
    fracs = [(k, v/s) for k, v in pairs]
    rounded = [(k, round(v, decimals)) for k, v in fracs]
    # fix drift
    diff = round(1.0 - sum(v for _, v in rounded), decimals)
    if abs(diff) > 0 and rounded:
        # adjust the largest entry
        i = max(range(len(rounded)), key=lambda j: rounded[j][1])
        k, v = rounded[i]
        rounded[i] = (k, round(max(0.0, v + diff), decimals))
    return [[k, v] for k, v in rounded]

# ---------- main transformer

def align_plan_to_schema(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a raw assistant plan into your schema:
      channels[] -> {name, activation, kpis}
      assets -> [string]
      budget_split -> [[label, fraction]]
      timeline_weeks -> int >= 1
    """
    aligned: Dict[str, Any] = dict(plan)

    # 1) assets -> list[str] (e.g., {"name": "...", "description": "..."} -> "name: description")
    assets = aligned.get("assets", [])
    new_assets: List[str] = []
    if isinstance(assets, list):
        for a in assets:
            if isinstance(a, str):
                new_assets.append(a)
            elif isinstance(a, dict):
                nm = (a.get("name") or a.get("type") or "Asset").strip()
                desc = (a.get("description") or a.get("format") or "").strip()
                s = f"{nm}: {desc}".strip(": ").strip()
                new_assets.append(s if s else nm)
    aligned["assets"] = new_assets

    # 2) channels -> each must have name, activation, kpis
    fixed_channels: List[Dict[str, Any]] = []
    for ch in aligned.get("channels", []):
        if isinstance(ch, str):
            fixed_channels.append({"name": ch, "activation": _default_activation_for(ch), "kpis": {}})
        elif isinstance(ch, dict):
            name = ch.get("name") or ch.get("channel") or "Channel"
            activation = ch.get("activation") or ch.get("description") or _default_activation_for(name)
            kpis = ch.get("kpis") or {}
            # If simple metrics live on the channel dict, migrate them into kpis
            for k in ("reach", "ctr", "cpl", "redemption", "membership_signup", "engagement", "retention"):
                if k in ch and k not in kpis:
                    kpis[k] = ch[k]
            fixed_channels.append({"name": name, "activation": activation, "kpis": kpis})
    aligned["channels"] = fixed_channels

    # 3) budget_split -> [[label, fraction]]
    pairs: List[Tuple[str, float]] = []
    bs = aligned.get("budget_split", [])
    if isinstance(bs, list) and bs:
        for item in bs:
            if isinstance(item, list) and len(item) == 2:
                pairs.append((str(item[0]), _to_fraction(item[1])))
            elif isinstance(item, dict):
                label = item.get("channel") or item.get("name") or item.get("label") or "Other"
                if "percentage" in item:
                    pairs.append((label, _to_fraction(item["percentage"])))
                elif "weight" in item:
                    pairs.append((label, _to_fraction(item["weight"])))
                elif "allocation" in item:
                    # treat as absolute; normalize later
                    pairs.append((label, float(item["allocation"])))
    # If still empty, derive from channel weights if present
    if not pairs:
        for ch in fixed_channels:
            orig = next((c for c in plan.get("channels", []) if isinstance(c, dict) and (c.get("name") or c.get("channel")) == ch["name"]), None)
            if isinstance(orig, dict) and "weight" in orig:
                pairs.append((ch["name"], _to_fraction(orig["weight"])))
    aligned["budget_split"] = _renorm_pairs(pairs)

    # 4) timeline_weeks
    tw = aligned.get("timeline_weeks", 4)
    try:
        tw = int(tw)
    except Exception:
        tw = 4
    aligned["timeline_weeks"] = max(1, tw)

    # 5) required text fields
    aligned["concept_title"] = str(aligned.get("concept_title", "") or "")
    aligned["big_idea"] = str(aligned.get("big_idea", "") or "")
    aligned["key_message"] = str(aligned.get("key_message", "") or "")

    # 6) kpis ensure object
    if not isinstance(aligned.get("kpis"), dict):
        aligned["kpis"] = {}

    return aligned

def text_after_assistant(txt: str) -> str:
    """
    Return everything after the LAST line that equals 'assistant' (case-insensitive).
    Works on chat transcripts where roles appear on their own line.
    """
    last = None
    for m in re.finditer(r'(?mi)^\s*json\s*$', txt):
        last = m
    if last:
        return txt[last.end():].lstrip()
    # fallback: last occurrence anywhere
    i = txt.lower().rfind("assistant")
    return txt[i + len("assistant"):].lstrip() if i != -1 else txt

def extract_balanced_json(text: str) -> Optional[str]:
    """
    Return the first balanced {...} JSON object in `text`, respecting quotes/escapes.
    """
    start = -1
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
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
                        return text[start:i+1]
    return None

def json_after_assistant(txt: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the JSON block that follows the last 'assistant' marker.
    Tries normal json; if that fails, pulls a balanced {...} block and parses it.
    """
    tail = text_after_assistant(txt)
    # quick path: whole tail is JSON
    try:
        return json.loads(tail)
    except Exception:
        pass
    # robust path: find the balanced object
    block = extract_balanced_json(tail)
    if block:
        try:
            return json.loads(block)
        except Exception:
            # optional: try json-repair if you have it installed
            try:
                from json_repair import repair_json
                return json.loads(repair_json(block))
            except Exception:
                return None
    return None