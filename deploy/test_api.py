import os, sys, time, json, argparse
from typing import Dict, Any, List
import requests

try:
    from jsonschema import validate
    HAVE_JSONSCHEMA = True
except Exception:
    HAVE_JSONSCHEMA = False

DEFAULT_BASE = os.getenv("API_BASE", "http://localhost:8000")

CASES: List[Dict[str, Any]] = [
    {
        "name": "awareness_basic",
        "payload": {
            "industry":"FMCG snacks",
            "audience":{"geo":"TH","age":"18-24"},
            "budget_thb": 1000000,
            "objective":"awareness",
            "constraints":{"brand_tone":"playful","mandatory_channels":["LINE OA"],"banned_channels":[]},
            "language":"EN"
        }
    },
    {
        "name": "acquisition_line_first",
        "payload": {
            "industry":"Retail fashion",
            "audience":{"geo":"TH","age":"25-34"},
            "budget_thb": 600000,
            "objective":"acquisition",
            "constraints":{"brand_tone":"premium","mandatory_channels":["LINE OA","Instagram"],"banned_channels":["Twitter/X"]},
            "language":"EN"
        }
    },
    {
        "name": "loyalty_crm",
        "payload": {
            "industry":"Banking",
            "audience":{"geo":"TH","age":"25-45"},
            "budget_thb": 2000000,
            "objective":"loyalty",
            "constraints":{"brand_tone":"trustworthy","mandatory_channels":["Email","LINE OA"],"banned_channels":[]},
            "language":"EN"
        }
    }
]


def http_get(url: str, timeout: float = 120.0):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r

def http_post(url: str, payload: Dict[str, Any], timeout: float = 300.0):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r

def fetch_schema(base: str) -> Dict[str, Any]:
    if not HAVE_JSONSCHEMA:
        print("(!) jsonschema not installed; skipping strict validation", file=sys.stderr)
    r = http_get(f"{base}/schema")
    return r.json()

def health_check(base: str) -> Dict[str, Any]:
    r = http_get(f"{base}/health")
    return r.json()

def run_case(base: str, schema: Dict[str, Any], case: Dict[str, Any]) -> bool:
    name = case["name"]
    payload = case["payload"]
    print(f"\n=== Case: {name} ===")
    t0 = time.time()
    r = http_post(f"{base}/campaign/generate", payload)
    rt_ms = int((time.time()-t0)*1000)
    try:
        data = r.json()
    except Exception as e:
        print(f"[FAIL] Response not JSON: {e}")
        print(r.text[:1000])
        return False

    status = data.get("status")
    elapsed_ms = data.get("elapsed_ms")
    plan = data.get("plan")
    warnings = data.get("warnings") or []

    if status != "ok":
        print(f"[FAIL] status != ok -> {status}")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:1200])
        return False

    if "plan_raw" in plan:
        print("[WARN] Server could not produce valid JSON; plan_raw returned.")
        print(plan["plan_raw"])
        return False

    # validate against server schema if available
    if HAVE_JSONSCHEMA and schema:
        try:
            validate(plan, schema)
        except Exception as e:
            print(f"[WARN] JSON does not conform to schema: {e}")
            print(json.dumps(plan, indent=2, ensure_ascii=False)[:1200])

    # Summary
    title = plan.get("concept_title", "(no title)")
    timeline = plan.get("timeline_weeks", "?")
    budget_items = plan.get("budget_split", [])
    budget_pct = sum([x[1] for x in budget_items if isinstance(x, list) and len(x)==2 and isinstance(x[1], (int,float))])
    print(f"[OK] title={title!r}  server_elapsed={elapsed_ms}ms  rtt={rt_ms}ms  budget_sum≈{round(budget_pct,2)}  timeline={timeline}w")
    if warnings:
        print("warnings:", warnings)
    return True

def main():
    ap = argparse.ArgumentParser(description="Test Campaign Ideation API")
    ap.add_argument("--base", default=DEFAULT_BASE, help="API base URL (default: %(default)s)")
    ap.add_argument("--case", default="all", help="Case name to run (or 'all')")
    args = ap.parse_args()
    
    base = args.base.rstrip("/")
    print(f"Target API: {base}")
    
    try:
        schema = fetch_schema(base)
    except Exception as e:
        print("[WARN] /schema not available:", e)
        schema = {}

    ok_all = True
    selected = CASES if args.case == "all" else [c for c in CASES if c["name"] == args.case]
    if not selected:
        print(f"No matching case for: {args.case}")
        sys.exit(3)

    for c in selected:
        ok = run_case(base, schema, c)
        ok_all = ok_all and ok

    if not ok_all:
        sys.exit(1)
    print("\nAll selected cases passed ✔")

if __name__ == "__main__":
    main()