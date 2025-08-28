import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from jsonschema import validate, ValidationError
from tqdm import tqdm
from typing import Optional, Any, Dict, Tuple, List
import re
from utils import json_after_assistant, align_plan_to_schema

BASE_MODEL = os.getenv("BASE_MODEL","meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR","outputs/lora-llama31-8b")
VAL_PATH = "data/val.jsonl"
SCHEMA_PATH = "schema/campaign.schema.json"

SYS_PROMPT = ("You are a senior marketing strategist for Thailand. "
              "Return ONLY a single JSON object that strictly follows the provided schema. "
              "No prose, no markdown—JSON only.")

def build_user(inp):
    return (f"Brief:\n- Industry: {inp['industry']}\n"
            f"- Audience: {json.dumps(inp['audience'],ensure_ascii=False)}\n"
            f"- Budget (THB): {inp['budget_thb']}\n"
            f"- Objective: {inp['objective']}\n"
            f"- Constraints: {json.dumps(inp.get('constraints',{}),ensure_ascii=False)}\n\n"
            f"JSON fields to produce: concept_title, big_idea, key_message, channels[], assets[], timeline_weeks, budget_split[], kpis{{}}")

def build_chat(user):
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYS_PROMPT}"
            f"\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")

def load_jsonl(p): 
    return [json.loads(l) for l in open(p,"r",encoding="utf-8")]

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    schema = json.load(open(SCHEMA_PATH))
    val = load_jsonl(VAL_PATH)

    ok = 0
    for ex in tqdm(val):
        user = build_user(ex["input"])
        prompt = build_chat(user)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.7)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # take the last JSON-ish block
        start = text.find("{")
        end = text.rfind("}")
        try:
            js = align_plan_to_schema(json_after_assistant(text[start:end+1]))
            print(js)
            # js = json.loads(js)
            validate(js, schema)
            ok += 1
        except Exception:
            pass
        
        break

    print(f"Schema pass rate: {ok}/{len(val)} = {ok/len(val):.2%}")

if __name__ == "__main__":
    main()
