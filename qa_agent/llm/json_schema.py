"""Helpers for JSON Schemas sent to strict OpenAI-compatible APIs."""
from copy import deepcopy
from typing import Any

from ..models import QATestStep


def normalize_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a schema and close every object schema for strict structured output."""
    normalized = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    return normalized


def qa_test_plan_schema() -> dict[str, Any]:
    """Return a strict-compatible plan schema with explicit action parameters."""
    step_variants = []
    common_fields = ("url", "expected", "selector", "expected_text", "value")
    for action, _action_fields in QATestStep.ACTION_PARAMETER_FIELDS.items():
        parameter_fields = common_fields + (("option_label",) if action == "select_option" else ())
        parameters_schema = {
            "type": "object",
            "properties": {
                field: (
                    {"type": "string"}
                    if field == "option_label"
                    else {"type": ["string", "null"]}
                )
                for field in parameter_fields
            },
            "required": list(parameter_fields),
            "additionalProperties": False,
        }
        step_variants.append({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": [action]},
                "parameters": parameters_schema,
            },
            "required": ["action", "parameters"],
            "additionalProperties": False,
        })
    return {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "steps": {"type": "array", "items": {"anyOf": step_variants}},
        },
        "required": ["url", "steps"],
        "additionalProperties": False,
    }
