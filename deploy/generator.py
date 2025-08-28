from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM
import time, json
from typing import Dict, Any, Tuple, List
import torch

from config import SYSTEM_PROMPT, DEFAULT_SCHEMA, GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P
from prompts import build_user_prompt, as_chat_messages
from utils import extract_first_json_block, normalize_budget_split
from validators import validate_plan
from model_loader import load_llama

def generate_json_plan(tokenizer: AutoTokenizer,
                       model: AutoModelForCausalLM,
                       system_prompt: str,
                       user_prompt: str,
                       max_new_tokens: int = 1024,
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> str:
    """Return raw model output text."""
    messages = as_chat_messages(system_prompt, user_prompt)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_campaign_plan(brief: Dict[str, Any],
                           schema: Dict[str, Any] = DEFAULT_SCHEMA,
                           max_new_tokens: int = GEN_MAX_NEW_TOKENS,
                           temperature: float = GEN_TEMPERATURE,
                           top_p: float = GEN_TOP_P) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (plan_dict, meta)
      meta includes: elapsed_ms, attempts, warnings[]
    """
    tok, mdl = load_llama()
    messages = as_chat_messages(SYSTEM_PROMPT, build_user_prompt(brief))
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)

    t0 = time.time()
    warnings: List[str] = []

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    raw = tok.decode(out[0], skip_special_tokens=True)
    cand = extract_first_json_block(raw) or raw

    try:
        plan = json.loads(cand)
    except Exception as e:
        warnings.append("JSON parse failed; returning raw text in 'plan_raw'.")
        return {"plan_raw": raw}, {"elapsed_ms": int((time.time()-t0)*1000), "attempts": 1, "warnings": warnings}

    # normalize + validate
    normalize_budget_split(plan)
    ok, err = validate_plan(plan, schema)
    if not ok:
        warnings.append(f"Schema validation failed: {err}")

    meta = {
        "elapsed_ms": int((time.time()-t0)*1000),
        "attempts": 1,
        "warnings": warnings
    }
    return plan, meta