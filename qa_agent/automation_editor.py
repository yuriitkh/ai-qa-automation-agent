"""Safe form parsing for the structured, local TestPlan editor."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from pydantic import ValidationError

from qa_agent.models import QATestPlan, QATestStep
from qa_agent.test_plan_validation import PlanValidationError, validate_executable_plan
from qa_agent.testplan_export import _is_unsafe_export_value


# Fields are derived from the browser runner contract. These optional fields are
# the additional parameters the runner explicitly reads with ``dict.get``.
ACTION_EDITOR_FIELDS: dict[str, tuple[str, ...]] = {
    "navigate": ("url",),
    "assert_page_loaded": (),
    "assert_title": ("expected",),
    "assert_visible": ("selector", "expected_text"),
    "click": ("selector",),
    "fill": ("selector", "value"),
    "assert_hidden": ("selector",),
    "assert_url": ("expected",),
    "select_option": ("selector", "option_label"),
    "assert_text_contains": ("expected_text", "selector"),
    "assert_checked": ("selector",),
    "assert_selected": ("selector", "expected"),
    "assert_enabled": ("selector",),
    "assert_disabled": ("selector",),
}

ACTION_LABELS = {
    "navigate": "Navigate to URL",
    "assert_page_loaded": "Assert page loaded",
    "assert_title": "Assert page title",
    "assert_visible": "Assert visible",
    "click": "Click",
    "fill": "Fill field",
    "assert_hidden": "Assert hidden",
    "assert_url": "Assert URL",
    "select_option": "Select option",
    "assert_text_contains": "Assert text contains",
    "assert_checked": "Assert checked",
    "assert_selected": "Assert selected",
    "assert_enabled": "Assert enabled",
    "assert_disabled": "Assert disabled",
}

FIELD_LABELS = {
    "url": "URL",
    "selector": "Selector",
    "value": "Value",
    "expected": "Expected value",
    "expected_text": "Expected text",
    "option_label": "Option label",
}

_ACTION_FIELD = re.compile(r"^action\.(\d+)\.param\.([a-z_]+)$")
_ACTION_TYPE = re.compile(r"^action\.(\d+)\.type$")
_REQUIRED_FIELDS = QATestStep.ACTION_PARAMETER_FIELDS


class AutomationEditorError(ValueError):
    def __init__(self, field_errors: dict[str, str]) -> None:
        self.field_errors = field_errors
        super().__init__(next(iter(field_errors.values()), "The automation form is invalid."))


def plan_from_form(values: dict[str, str]) -> QATestPlan:
    """Parse and validate one step's structured form without accepting extra fields."""
    errors: dict[str, str] = {}
    allowed_names = {"plan_url", "expected_version"}
    indices: set[int] = set()
    action_types: dict[int, str] = {}
    action_parameters: dict[int, dict[str, str]] = {}

    for name, value in values.items():
        if name in allowed_names:
            continue
        type_match = _ACTION_TYPE.fullmatch(name)
        if type_match:
            index = int(type_match.group(1))
            indices.add(index)
            action_types[index] = value
            continue
        field_match = _ACTION_FIELD.fullmatch(name)
        if field_match:
            index = int(field_match.group(1))
            field = field_match.group(2)
            indices.add(index)
            action_parameters.setdefault(index, {})[field] = value
            continue
        errors[name] = "This automation field is not supported."

    if not values.get("plan_url", "").strip():
        errors["plan_url"] = "Enter the URL for this automation plan."
    else:
        try:
            parsed = urlsplit(values["plan_url"].strip())
            _ = parsed.port
            if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
                errors["plan_url"] = "Enter a valid HTTP or HTTPS URL."
            elif parsed.username or parsed.password:
                errors["plan_url"] = "URLs containing credentials are not allowed."
        except ValueError:
            errors["plan_url"] = "Enter a valid HTTP or HTTPS URL."

    ordered_indices = sorted(indices)
    if ordered_indices and ordered_indices != list(range(len(ordered_indices))):
        errors["actions"] = "Action order is invalid. Reload the editor and try again."
    if not ordered_indices:
        errors["actions"] = "Add at least one action to this plan."

    plan_steps: list[QATestStep] = []
    for index in ordered_indices:
        action = action_types.get(index, "")
        type_key = f"action.{index}.type"
        if action not in ACTION_EDITOR_FIELDS:
            errors[type_key] = "Choose a supported action type."
            continue
        parameters = action_parameters.get(index, {})
        supported_fields = set(ACTION_EDITOR_FIELDS[action])
        for field in parameters:
            if field not in supported_fields:
                errors[f"action.{index}.param.{field}"] = (
                    "This parameter is not supported for the selected action."
                )
        for field in _REQUIRED_FIELDS[action]:
            if not parameters.get(field, "").strip():
                errors[f"action.{index}.param.{field}"] = (
                    f"{FIELD_LABELS.get(field, field)} is required for this action."
                )
        cleaned_parameters = {
            field: value
            for field, value in parameters.items()
            if field in supported_fields and value != ""
        }
        try:
            plan_steps.append(QATestStep(action=action, parameters=cleaned_parameters))
        except ValidationError:
            errors[type_key] = "The selected action parameters are invalid."

    if errors:
        raise AutomationEditorError(errors)

    try:
        plan = validate_executable_plan(QATestPlan(
            url=values["plan_url"].strip(),
            steps=plan_steps,
        ))
    except PlanValidationError as error:
        for issue in error.issues:
            match = re.fullmatch(r"steps\[(\d+)\](?:\.parameters\.([a-z_]+))?", issue.path)
            if match:
                index, field = match.groups()
                key = f"action.{index}.param.{field}" if field else f"action.{index}.type"
            elif issue.path == "url":
                key = "plan_url"
            else:
                key = "actions"
            errors[key] = issue.message
        raise AutomationEditorError(errors) from error

    if _is_unsafe_export_value(plan.url):
        raise AutomationEditorError({
            "plan_url": "Local file paths and URLs containing credentials are not allowed.",
        })
    for index, action in enumerate(plan.steps):
        for field, value in action.parameters.items():
            if not isinstance(value, str):
                continue
            if field == "url" and not _valid_http_url(value):
                raise AutomationEditorError({
                    f"action.{index}.param.url": "Enter a valid HTTP or HTTPS URL.",
                })
            if _is_unsafe_export_value(value):
                raise AutomationEditorError({
                    f"action.{index}.param.{field}": "Local file paths and URLs containing credentials are not allowed.",
                })
    return plan


def _valid_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        # Accessing port also detects malformed numeric ports.
        _ = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme.casefold() in {"http", "https"}
        and parsed.hostname
        and not parsed.username
        and not parsed.password
    )
