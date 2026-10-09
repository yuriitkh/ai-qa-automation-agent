"""Segment-local generation context and narrow, deterministic boundary checks."""
import re
from dataclasses import dataclass

from qa_agent.expected_result_coverage import assertion_subject_matches
from qa_agent.models import DiscoveryResult, DiscoveryStatus, InteractiveElement, QATestPlan, TestStep
from qa_agent.test_plan_validation import PlanValidationError, PlanValidationIssue


@dataclass(frozen=True)
class StepGenerationContext:
    segment_order: int
    previous_steps: tuple[TestStep, ...] = ()
    remaining_steps: tuple[TestStep, ...] = ()
    # Successful prior actions only; never retain entered values here.
    completed_actions: tuple[tuple[str, str | None], ...] = ()
    state_preserved: bool = False


_SUBMIT = re.compile(r"\b(?:submit|send|register|create\s+(?:an?\s+)?account|sign\s*up)\b", re.I)
_FILL = re.compile(r"\b(?:fill|enter|type|replace|change|update|correct|edit)\b", re.I)
_NAVIGATE = re.compile(r"\b(?:navigate|open|visit|go\s+to|reload|refresh|reset|return)\b", re.I)


def observed_controls(discovery: DiscoveryResult) -> dict[str, InteractiveElement]:
    """Use the deterministic snapshot when typed collections may contain AI suggestions."""
    controls = {item.selector: item for item in discovery.interactive_elements} if discovery.status == DiscoveryStatus.SUCCESS else {}
    for record in discovery.snapshot.get("interactive_elements", []):
        if not isinstance(record, dict) or not isinstance(record.get("selector"), str) or not record.get("kind"):
            continue
        try:
            item = InteractiveElement.model_validate(record)
        except ValueError:
            continue
        controls[item.selector] = item
    return controls


def validate_step_boundaries(plan: QATestPlan, step: TestStep, discovery: DiscoveryResult,
                             context: StepGenerationContext | None) -> None:
    if context is None:
        return
    intent = f"{step.name} {step.description}"
    future_submission = [item for item in context.remaining_steps if _SUBMIT.search(f"{item.name} {item.description}")]
    current_matches = assertion_subject_matches(step, plan, discovery=discovery)
    future_assertions = {
        index for item in context.remaining_steps
        for index, matches in assertion_subject_matches(item, plan, discovery=discovery).items() if matches
    }
    controls = observed_controls(discovery)
    filled = {selector for action, selector in context.completed_actions if action == "fill"}
    for index, action in enumerate(plan.steps):
        selector = action.parameters.get("selector")
        control = controls.get(selector)
        reason = None
        if action.action == "click" and future_submission and not _SUBMIT.search(intent) and control is not None:
            label = control.accessible_name or control.text
            named_later = bool(label and any(label.casefold() in f"{item.name} {item.description}".casefold() for item in future_submission))
            if control.button_type == "submit" or named_later:
                reason = "Submission belongs to a later TestStep; this step must preserve the prepared form."
        if action.action.startswith("assert_") and index in future_assertions and not current_matches.get(index):
            # A load assertion may legitimately check navigation performed here.
            if not (action.action == "assert_page_loaded" and _NAVIGATE.search(intent)):
                reason = "This assertion verifies a later TestStep rather than the current expected result."
        if context.state_preserved:
            if action.action == "fill" and selector in filled and not _FILL.search(intent):
                if control is None or control.has_value is not False:
                    reason = "This control was filled by a completed step; the current step does not request changing it."
            if action.action == "navigate" and context.completed_actions and not _NAVIGATE.search(intent):
                if "current_page" in discovery.strategies_used and action.parameters.get("url") == discovery.url:
                    reason = "The active segment already has this page; repeated navigation would reset its state."
        if reason:
            raise PlanValidationError([PlanValidationIssue(code="TESTSTEP_BOUNDARY_VIOLATION",
                path=f"steps[{index}]", message=reason)])
