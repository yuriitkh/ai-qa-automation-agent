"""Value-free browser action diagnostics; no exception or page text is retained."""
import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from qa_agent.redaction import redact_secrets


def safe_selector_identity(selector, sensitive_values=()) -> str | None:
    if not isinstance(selector, str) or not selector:
        return None
    cleaned = redact_secrets(selector)
    if cleaned == selector and not any(value and value in selector for value in sensitive_values):
        if re.fullmatch(r"(?:#[A-Za-z_][\w-]{0,79}|[a-z][a-z0-9-]{0,39})", selector):
            return selector
    return "selector-sha256:" + hashlib.sha256(selector.encode()).hexdigest()[:16]


class ActionFailureDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    action_index: int = Field(ge=0)
    action: Literal["navigate", "assert_page_loaded", "assert_title", "assert_visible", "click", "check", "uncheck", "fill", "assert_hidden", "assert_url", "select_option", "assert_text_contains", "assert_checked", "assert_unchecked", "assert_selected", "assert_enabled", "assert_disabled", "assert_value"]
    selector_identity: str | None = None
    exception_category: Literal["TIMEOUT", "ASSERTION_FAILURE", "BROWSER_ERROR", "ACTION_ERROR"]
    exception_type: Literal["TimeoutError", "AssertionError", "Error", "Other"]
    timed_out: bool = False
    page_open: bool | None = None
    at_plan_url: bool | None = None
    document_state: Literal["loading", "interactive", "complete"] | None = None
    target_count: int | None = Field(default=None, ge=0)
    target_visible: bool | None = None
    target_enabled: bool | None = None
    evidence_indexes: tuple[int, ...] = ()
    classification: Literal["PRODUCT_FAILURE", "AUTOMATION_DRIFT", "AUTOMATION_EXECUTION_ERROR", "INFRASTRUCTURE_ERROR"] | None = None

    @field_validator("selector_identity")
    @classmethod
    def safe_identity(cls, value):
        if isinstance(value, str) and re.fullmatch(r"selector-sha256:[a-f0-9]{16}", value):
            return value
        return safe_selector_identity(value)

    @computed_field
    @property
    def recommended_action(self) -> str:
        if self.classification == "PRODUCT_FAILURE":
            return "Review the requirement and screenshot; investigate the product behavior."
        if self.exception_category == "BROWSER_ERROR":
            return "Check browser and target availability before retrying."
        if self.target_count == 0 or self.target_count is not None and self.target_count > 1:
            return "Review Discovery and the exact control identity before creating a new plan version."
        return "Inspect the screenshot and control state; review this action before retrying."

    def summary(self) -> str:
        target = f" Target: {self.selector_identity}." if self.selector_identity else ""
        return f"Action {self.action_index + 1} ({self.action}) failed: {self.exception_category}.{target} {self.recommended_action}"


def action_failure_diagnostic(runner_result) -> ActionFailureDiagnostic | None:
    for item in (runner_result or {}).get("steps", []):
        if isinstance(item, dict) and item.get("status") == "failed" and isinstance(item.get("diagnostic"), dict):
            try:
                return ActionFailureDiagnostic.model_validate(item["diagnostic"])
            except ValueError:
                return None
    return None


def redact_action_failure(diagnostic: ActionFailureDiagnostic | None, redact) -> ActionFailureDiagnostic | None:
    """Honor RunContext sensitivity without losing the selector's stable identity."""
    if diagnostic is not None and diagnostic.selector_identity:
        selector = diagnostic.selector_identity
        if redact(selector) != selector:
            return diagnostic.model_copy(update={
                "selector_identity": "selector-sha256:" + hashlib.sha256(selector.encode()).hexdigest()[:16]
            })
    return diagnostic


def capture_action_failure(page, action, index, plan, error) -> ActionFailureDiagnostic:
    causes, current = [], error
    while current is not None and all(current is not item for item in causes):
        causes.append(current)
        current = current.__cause__
    text = " ".join(str(item).casefold() for item in causes)
    timeout = any(type(item).__name__ == "TimeoutError" for item in causes) or "timeout" in text
    infrastructure = any(marker in text for marker in (
        "browser has been closed", "browser disconnected", "target page, context or browser has been closed",
        "page has been closed", "page crashed", "net::err_", "connection refused", "connection reset", "protocol error",
    ))
    category = "BROWSER_ERROR" if infrastructure else "TIMEOUT" if any(type(item).__name__ == "TimeoutError" for item in causes) else "ASSERTION_FAILURE" if action.action.startswith("assert_") and isinstance(error, AssertionError) else "ACTION_ERROR"
    data = dict(action_index=index, action=action.action,
        selector_identity=safe_selector_identity(action.parameters.get("selector"),
            [item.parameters.get("value") if item.action == "fill" else item.parameters.get("expected")
             for item in plan.steps if item.action in {"fill", "assert_value"}]),
        exception_category=category, timed_out=timeout,
        exception_type="TimeoutError" if category == "TIMEOUT" else "AssertionError" if isinstance(error, AssertionError) else "Error" if type(error).__name__ == "Error" else "Other")
    # Best-effort boolean/state probes cannot mask the original action failure.
    try:
        closed = page.is_closed()
        if isinstance(closed, bool):
            data["page_open"] = not closed
        if closed is False:
            data["at_plan_url"] = page.url == plan.url
            state = page.evaluate("() => document.readyState")
            if state in {"loading", "interactive", "complete"}:
                data["document_state"] = state
            selector = action.parameters.get("selector")
            if isinstance(selector, str):
                target = page.locator(selector)
                count = target.count()
                if isinstance(count, int) and not isinstance(count, bool):
                    data["target_count"] = count
                    if count == 1:
                        visible, enabled = target.is_visible(), target.is_enabled()
                        if isinstance(visible, bool):
                            data["target_visible"] = visible
                        if isinstance(enabled, bool):
                            data["target_enabled"] = enabled
    except Exception:
        pass
    return ActionFailureDiagnostic.model_validate(data)
