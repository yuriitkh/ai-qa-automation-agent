"""Structural checks for plans that the browser runner can execute."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from qa_agent.models import QATestPlan, QATestStep


class PlanValidationIssue(BaseModel):
    """Allowlisted, value-free detail about an invalid executable plan."""

    model_config = ConfigDict(frozen=True)

    code: Literal[
        "INVALID_PLAN_STRUCTURE",
        "MISSING_TARGET_URL",
        "EMPTY_ACTION_SEQUENCE",
        "INVALID_ACTION_TYPE",
        "UNSUPPORTED_ACTION",
        "INVALID_PARAMETERS",
        "MISSING_LOCATOR",
        "MISSING_INPUT_VALUE",
        "MISSING_NAVIGATION_URL",
        "MISSING_EXPECTED_VALUE",
        "MISSING_OPTION_LABEL",
        "INVALID_PARAMETER_TYPE",
        "INVALID_PARAMETER",
        "DISCOVERY_SELECTOR_MISMATCH",
        "ACTION_TARGET_MISMATCH",
        "UNGROUNDED_ASSERTION",
        "EXPECTED_RESULT_NOT_COVERED",
    ]
    path: str
    message: str


class PlanValidationError(ValueError):
    """Semantic plan error containing only safe, structured diagnostics."""

    def __init__(self, issues: tuple[PlanValidationIssue, ...] | list[PlanValidationIssue]):
        self.issues = tuple(issues)
        super().__init__(self.issues[0].message if self.issues else "Plan validation failed.")


_REQUIRED_PARAMETERS = QATestStep.ACTION_PARAMETER_FIELDS
_UNSUPPORTED_ACTION_MESSAGE = "The action is not supported by the browser runner."


def validate_executable_plan(value: Any) -> QATestPlan:
    """Revalidate a plan and ensure required action inputs are present.

    Pydantic does not revalidate an existing model instance by default, so
    model instances are converted to plain data before validation. Diagnostics
    use stable paths and messages and never include generated parameter values.
    """
    data = value.model_dump(mode="python") if isinstance(value, QATestPlan) else value
    _validate_action_contract(data)
    try:
        plan = QATestPlan.model_validate(data)
    except ValidationError as error:
        # The action contract above handles executable fields. This catch-all
        # covers malformed outer/model structure without exposing Pydantic's
        # input-bearing error details to progress or repair prompts.
        raise PlanValidationError([
            PlanValidationIssue(
                code="INVALID_PLAN_STRUCTURE",
                path="plan",
                message="The generated plan structure is invalid.",
            )
        ]) from error
    if isinstance(value, QATestPlan) and plan == value:
        return value
    return plan


def _validate_action_contract(data: Any) -> None:
    if not isinstance(data, dict):
        _fail("INVALID_PLAN_STRUCTURE", "plan", "The generated plan must be an object.")

    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        _fail("MISSING_TARGET_URL", "url", "The plan requires a non-empty target URL.")

    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("EMPTY_ACTION_SEQUENCE", "steps", "The plan requires at least one executable action.")

    for index, step in enumerate(steps):
        step_path = f"steps[{index}]"
        if not isinstance(step, dict):
            _fail("INVALID_PLAN_STRUCTURE", step_path, "The action entry must be an object.")
        action = step.get("action")
        action_path = f"{step_path}.action"
        if not isinstance(action, str):
            _fail("INVALID_ACTION_TYPE", action_path, "The action name must be text.")
        if action not in _REQUIRED_PARAMETERS:
            _fail("UNSUPPORTED_ACTION", action_path, _UNSUPPORTED_ACTION_MESSAGE)
        parameters = step.get("parameters", {})
        if not isinstance(parameters, dict):
            _fail("INVALID_PARAMETERS", f"{step_path}.parameters", "Action parameters must be an object.")

        for field in _REQUIRED_PARAMETERS[action]:
            path = f"{step_path}.parameters.{field}"
            parameter = parameters.get(field)
            if not isinstance(parameter, str):
                code, message = _missing_parameter_issue(action, field)
                _fail(code, path, message)
            if field in {"selector", "url"} and not parameter.strip():
                code, message = _missing_parameter_issue(action, field)
                _fail(code, path, message)

        for field in ("expected_text", "expected"):
            parameter = parameters.get(field)
            if parameter is not None and not isinstance(parameter, str):
                _fail(
                    "INVALID_PARAMETER_TYPE",
                    f"{step_path}.parameters.{field}",
                    "Expected values must be text when provided.",
                )
        if action == "assert_text_contains":
            selector = parameters.get("selector")
            if selector is not None and (
                not isinstance(selector, str) or not selector.strip()
            ):
                _fail(
                    "MISSING_LOCATOR",
                    f"{step_path}.parameters.selector",
                    "ASSERT_TEXT_CONTAINS requires a non-empty locator when one is provided.",
                )
        if action != "select_option" and "option_label" in parameters:
            _fail(
                "INVALID_PARAMETER",
                f"{step_path}.parameters.option_label",
                "Option labels are only supported by SELECT_OPTION actions.",
            )


def _missing_parameter_issue(action: str, field: str) -> tuple[str, str]:
    if field == "selector":
        return "MISSING_LOCATOR", f"{action.upper()} action requires a locator."
    if field == "value":
        return "MISSING_INPUT_VALUE", "FILL action requires both a locator and a value."
    if field == "url":
        return "MISSING_NAVIGATION_URL", "NAVIGATE action requires a URL."
    if field in {"expected", "expected_text"}:
        return "MISSING_EXPECTED_VALUE", f"{action.upper()} action requires an expected value."
    if field == "option_label":
        return "MISSING_OPTION_LABEL", "SELECT_OPTION action requires an option label."
    return "INVALID_PARAMETER_TYPE", f"{action.upper()} action has an invalid required parameter."


def _fail(code: str, path: str, message: str) -> None:
    raise PlanValidationError([PlanValidationIssue(code=code, path=path, message=message)])
