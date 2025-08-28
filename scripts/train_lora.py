# scripts/train_lora.py
import os, json, torch
from datasets import load_dataset, Dataset, Features, Value
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from dataclasses import dataclass
from typing import Dict
from peft import TaskType

BASE_MODEL = os.getenv("BASE_MODEL","meta-llama/Meta-Llama-3.1-8B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR","outputs/lora-llama31-8b")
TRAIN_PATH = "data/train.jsonl"
VAL_PATH   = "data/val.jsonl"

SYS_PROMPT = ("You are a senior marketing strategist for Thailand. "
              "Return ONLY a single JSON object that strictly follows the provided schema. "
              "No prose, no markdown—JSON only.")

def build_prompt(ex: Dict) -> str:
    inp = ex["input"]
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYS_PROMPT}"
        f"\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Brief:\n- Industry: {inp['industry']}\n"
        f"- Audience: {json.dumps(inp['audience'],ensure_ascii=False)}\n"
        f"- Budget (THB): {inp['budget_thb']}\n"
        f"- Objective: {inp['objective']}\n"
        f"- Constraints: {json.dumps(inp.get('constraints',{}),ensure_ascii=False)}\n\n"
        f"JSON fields to produce: concept_title, big_idea, key_message, channels[], assets[], timeline_weeks, budget_split[], kpis{{}}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        f"{json.dumps(ex['output'],ensure_ascii=False)}"
    )

def load_jsonl(path):
    return [{"text": build_prompt(json.loads(l))} for l in open(path, "r", encoding="utf-8")]

def gen_text(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                try:
                    from json_repair import repair_json
                    obj = json.loads(repair_json(line))
                except Exception:
                    continue
            # Build your train string here; adjust to your file structure
            inp  = obj.get("input", {})
            outp = obj.get("output", obj)
            text = (
                "<|system|>You are a marketing strategist. JSON only.\n"
                f"<|user|>{json.dumps(inp, ensure_ascii=False)}\n"
                f"<|assistant|>{json.dumps(outp, ensure_ascii=False)}"
            )
            yield {"text": text}

def main():
    # 4-bit quant for QLoRA
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
    )
    # QLoRA training prerequisites
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # Important for 4-bit training: make inputs require grads
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ---- Build datasets (must be 🤗 Dataset, not list) ----
    # If you already have train/val JSONL with your own gen_text():
    train_ds = Dataset.from_generator(lambda: gen_text("data/train.jsonl"),
                                      features=Features({"text": Value("string")}))
    eval_ds  = Dataset.from_generator(lambda: gen_text("data/val.jsonl"),
                                      features=Features({"text": Value("string")}))

    # ---- TRL config ----
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  # set True only if supported
        max_length=4096,
        packing=False,
        dataset_text_field="text",
    )

    # ---- Let TRL attach LoRA (do NOT call get_peft_model yourself) ----
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,   # TRL v0.21 uses processing_class (not tokenizer=)
        peft_config=peft_cfg,         # << TRL creates trainable LoRA adapters
    )

    # Sanity check: ensure LoRA params are trainable
    trainable, total = 0, 0
    for _, p in trainer.model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")
    assert trainable > 0, "No trainable parameters! LoRA not attached?"

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved LoRA adapter + tokenizer to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
