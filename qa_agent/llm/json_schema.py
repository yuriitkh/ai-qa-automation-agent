"""Helpers for JSON Schemas sent to strict OpenAI-compatible APIs."""
from copy import deepcopy
from typing import Any

from ..models import QATestStep


def normalize_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the strict-schema form expected by OpenAI-compatible endpoints.

    Strict JSON Schema mode requires every declared object property to appear
    in ``required`` and every object to reject additional properties. Optional
    application fields remain nullable where their source schema permits it;
    local Pydantic validation still defines the authoritative output contract.
    Defaults are annotations and are not part of the request contract.
    """
    normalized = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                value["additionalProperties"] = False
                properties = value.get("properties")
                if isinstance(properties, dict):
                    existing = value.get("required", [])
                    required = list(existing) if isinstance(existing, list) else []
                    value["required"] = list(dict.fromkeys(
                        [*required, *properties.keys()]
                    ))
            value.pop("default", None)
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
    for action, action_fields in QATestStep.ACTION_PARAMETER_FIELDS.items():
        parameter_fields = common_fields + (("option_label",) if action == "select_option" else ())
        parameters_schema = {
            "type": "object",
            "properties": {
                field: _parameter_schema(field, action_fields)
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
            "url": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "steps": {"type": "array", "minItems": 1, "items": {"anyOf": step_variants}},
        },
        "required": ["url", "steps"],
        "additionalProperties": False,
    }


def _parameter_schema(field: str, required_fields: tuple[str, ...]) -> dict[str, Any]:
    if field == "option_label" or field in required_fields:
        schema: dict[str, Any] = {"type": "string"}
        if field in {"url", "selector"}:
            schema.update({"minLength": 1, "pattern": r".*\S.*"})
        return schema
    return {"type": ["string", "null"]}
