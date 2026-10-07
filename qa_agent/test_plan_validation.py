"""Structural checks for plans that the browser runner can execute."""

from typing import Any

from qa_agent.models import QATestPlan


_REQUIRED_PARAMETERS = {
    "navigate": ("url",),
    "assert_page_loaded": (),
    "assert_title": ("expected",),
    "assert_visible": ("selector",),
    "click": ("selector",),
    "fill": ("selector", "value"),
    "assert_hidden": ("selector",),
    "assert_url": ("expected",),
    "select_option": ("selector", "option_label"),
    "assert_text_contains": ("expected_text",),
    "assert_checked": ("selector",),
    "assert_selected": ("selector",),
    "assert_enabled": ("selector",),
    "assert_disabled": ("selector",),
}


def validate_executable_plan(value: Any) -> QATestPlan:
    """Revalidate a plan and ensure required action inputs are present.

    Pydantic does not revalidate an existing model instance by default, so
    model instances are converted to plain data before being validated again.
    This also protects the execution boundary from corrupted or legacy rows.
    """
    data = value.model_dump(mode="python") if isinstance(value, QATestPlan) else value
    plan = QATestPlan.model_validate(data)
    if not plan.url.strip():
        raise ValueError("QATestPlan requires a non-empty URL.")
    for step in plan.steps:
        for field in _REQUIRED_PARAMETERS[step.action]:
            parameter = step.parameters.get(field)
            if not isinstance(parameter, str):
                raise ValueError(
                    f"{step.action} requires a string {field} parameter."
                )
            if field in {"selector", "url"} and not parameter.strip():
                raise ValueError(f"{step.action} requires non-empty {field} text.")
        for field in ("expected_text", "expected"):
            parameter = step.parameters.get(field)
            if parameter is not None and not isinstance(parameter, str):
                raise ValueError(f"{step.action} {field} must be a string.")
        if step.action == "assert_text_contains":
            selector = step.parameters.get("selector")
            if selector is not None and (
                not isinstance(selector, str) or not selector.strip()
            ):
                raise ValueError("assert_text_contains selector must be non-empty text.")
    if isinstance(value, QATestPlan) and plan == value:
        return value
    return plan
