# prompts.py
import json
from typing import Dict, Any, List

def build_user_prompt(inp: Dict[str, Any]) -> str:
    return (
        "Brief:\n"
        f"- Industry: {inp['industry']}\n"
        f"- Audience: {json.dumps(inp['audience'], ensure_ascii=False)}\n"
        f"- Budget (THB): {inp['budget_thb']}\n"
        f"- Objective: {inp['objective']}\n"
        f"- Constraints: {json.dumps(inp.get('constraints', {}), ensure_ascii=False)}\n\n"
        "JSON fields to produce: concept_title, big_idea, key_message, channels[], assets[], "
        "timeline_weeks, budget_split[], kpis{}"
    )

def as_chat_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]