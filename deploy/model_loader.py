# Strictly load Meta-Llama-3.1-8B-Instruct only (gated on Hugging Face).
import os
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from config import MODEL_ID

ADAPTER_DIR = os.getenv("ADAPTER_DIR","outputs/lora-llama31-8b")

def _resolve_model_source(model_dir: str | None) -> str:
    """
    Decide whether to load from a local directory or from the HF repo id.
    Only Meta-Llama-3.1-8B-Instruct is allowed.
    """
    if model_dir:
        # Accept a local path as long as it exists; caller is responsible for placing the correct model there.
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir
    # default to HF repo id
    return MODEL_ID

def load_llama(model_dir: str | None = None,
               local_files_only: bool = False,
               hf_token: str | None = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load tokenizer and model for Meta-Llama-3.1-8B-Instruct.
    If using the HF repo (not local), you MUST have accepted the license and provide a token with gated access.
    """
    src = _resolve_model_source(model_dir)

    # Enforce model id if src is a repo id
    if src == MODEL_ID or os.path.isdir(src):
        pass
    else:
        raise ValueError("This app only supports Meta-Llama-3.1-8B-Instruct.")
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. For CPU, use the CPU option below.")

    kwargs = dict(local_files_only=local_files_only)
    if hf_token:
        kwargs["token"] = hf_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    use_mps = torch.backends.mps.is_available()
    dtype = torch.float16 if use_mps else torch.float32

    tok = AutoTokenizer.from_pretrained(src, use_fast=True, **kwargs)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype,quantization_config=bnb,device_map="auto", low_cpu_mem_usage=True,**kwargs)
    mdl = PeftModel.from_pretrained(mdl, ADAPTER_DIR)
    mdl.eval()
    return tok, mdl