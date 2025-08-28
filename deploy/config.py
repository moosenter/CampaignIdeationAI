# Project-wide constants and defaults.
import os
# Hard lock to Meta-Llama-3.1-8B-Instruct (HF gated repo)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Optional local directory to load the model from (offline). If set, this must contain the exact model.
MODEL_DIR = os.getenv("MODEL_DIR", "").strip() or None

# Hugging Face token for gated access. Must include "read" and "public gated repositories".
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

# Force offline mode (no HF calls). Set to "1" to require local files only.
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "0") in ("1","true","True")

# Generation defaults (tune as desired)
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "1024"))
GEN_TEMPERATURE    = float(os.getenv("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P          = float(os.getenv("GEN_TOP_P", "0.9"))

# Default JSON schema for a campaign plan
DEFAULT_SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["concept_title","big_idea","key_message","channels","assets","timeline_weeks","budget_split","kpis"],
  "properties": {
    "concept_title": {"type": "string","minLength": 3},
    "big_idea": {"type": "string"},
    "key_message": {"type": "string"},
    "channels": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name","activation","kpis"],
        "properties": {
          "name": {"type": "string"},
          "activation": {"type": "string"},
          "kpis": {"type": "object"}
        },
        "additionalProperties": True
      }
    },
    "assets": {"type": "array","items": {"type": "string"}},
    "timeline_weeks": {"type": "integer","minimum": 1},
    "budget_split": {
      "type": "array",
      "items": {
        "type": "array",
        "prefixItems": [
          {"type": "string"},
          {"type": "number","minimum": 0,"maximum": 1}
        ],
        "minItems": 2,
        "maxItems": 2
      }
    },
    "kpis": {"type": "object"}
  },
  "additionalProperties": True
}

# Channels commonly used in Thailand
CHANNEL_CATALOG = ["LINE OA","TikTok","Facebook","Instagram","YouTube","Email","Retail POS","Twitter/X"]

SYSTEM_PROMPT = (
    "You are a senior marketing strategist for Thailand.\n"
    "Return ONLY a single JSON object that strictly follows the provided schema.\n"
    "No prose, no markdown — JSON only.\n"
)