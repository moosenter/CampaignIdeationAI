from typing import Dict, Any, Tuple
from jsonschema import validate

def validate_plan(plan: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        validate(plan, schema)
        return True, ""
    except Exception as e:
        return False, str(e)