import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, jsonschema, tqdm

BASE_MODEL=os.getenv("BASE_MODEL","meta-llama/Meta-Llama-3.1-8B-Instruct")
IN_PATH="data/train_synth_clean.jsonl"
OUT_PATH="data/train_synth_bilingual.jsonl"
SCHEMA_PATH="schema/campaign.schema.json"

SYS=("You are a bilingual Thai/English marketing editor. "
     "Given a JSON plan, output the SAME JSON with fields rewritten in the TARGET language. "
     "Do not change numbers/structure. Only translate text content.")

PROMPT=("TARGET={target}\n\nJSON:\n{js}\n\n"
        "Return JSON only, same keys & schema.")

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    schema = json.load(open(SCHEMA_PATH))
    out = open(OUT_PATH,"w",encoding="utf-8")

    def trans(target, rec):
        j = json.dumps(rec["output"], ensure_ascii=False)
        msg = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYS}\n<|eot_id|>"
               f"<|start_header_id|>user<|end_header_id|>\n{PROMPT.format(target=target, js=j)}\n"
               f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
        ids = tok(msg, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=1024, do_sample=True, temperature=0.4, top_p=0.9)
        txt = tok.decode(gen[0], skip_special_tokens=True)
        s,e = txt.find("{"), txt.rfind("}")
        if s==-1 or e==-1: return None
        js2 = json.loads(txt[s:e+1])
        jsonschema.validate(js2, schema)
        return {"input": rec["input"], "output": js2}

    for line in tqdm.tqdm(open(IN_PATH,"r",encoding="utf-8")):
        rec = json.loads(line)
        out.write(json.dumps(rec, ensure_ascii=False)+"\n")          # keep original
        tr = trans("TH", rec)                                        # Thai version
        if tr: out.write(json.dumps(tr, ensure_ascii=False)+"\n")
    out.close(); print("Done ->", OUT_PATH)

if __name__ == "__main__":
    main()
