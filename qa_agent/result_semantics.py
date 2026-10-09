"""Conservative public outcomes over compatible internal execution statuses."""

from __future__ import annotations

from typing import Any


def enum_value(value: Any) -> str | None:
    return value.value if hasattr(value, "value") else value


def result_outcome(status: Any, outcome: Any = None, *, complete: bool | None = None) -> str:
    """FAILED alone is never evidence of a product defect."""
    status, outcome = enum_value(status), enum_value(outcome)
    aliases = {
        "SETUP_FAILURE": "BLOCKED",
        "MISSING_AUTOMATION": "BLOCKED",
        "INVALID_TESTCASE": "BLOCKED",
        "PRECONDITION_NOT_ESTABLISHED": "BLOCKED",
        "CLEANUP_FAILURE": "INFRASTRUCTURE_ERROR",
        "SETUP_INFRASTRUCTURE_ERROR": "INFRASTRUCTURE_ERROR",
        "EXECUTION_ERROR": "AUTOMATION_EXECUTION_ERROR",
        "AI_GENERATION_ERROR": "AUTOMATION_GENERATION_ERROR",
    }
    outcome = aliases.get(outcome, outcome)
    if outcome in {
        "PRODUCT_FAILURE", "AUTOMATION_EXECUTION_ERROR", "AUTOMATION_DRIFT",
        "AUTOMATION_GENERATION_ERROR", "INFRASTRUCTURE_ERROR", "BLOCKED", "INCONCLUSIVE",
    }:
        return outcome
    if outcome == "PASSED":
        return "PASSED" if status not in {"FAILED", "BLOCKED", "RUNNING", "PENDING"} and complete is not False else "INCONCLUSIVE"
    if status == "BLOCKED":
        return "BLOCKED"
    if status in {"PENDING", "QUEUED", "RUNNING", "RETRYING", "NOT_RUN", "NOT_ATTEMPTED"}:
        return status
    return "INCONCLUSIVE"


def result_label(outcome: str) -> str:
    return {
        "PASSED": "Passed", "PRODUCT_FAILURE": "Failed",
        "AUTOMATION_EXECUTION_ERROR": "Automation Error", "AUTOMATION_DRIFT": "Automation Error",
        "AUTOMATION_GENERATION_ERROR": "Generation Error", "INFRASTRUCTURE_ERROR": "Infrastructure Error",
        "BLOCKED": "Blocked", "INCONCLUSIVE": "Inconclusive",
        "NOT_ATTEMPTED": "Not attempted", "NOT_RUN": "Not run",
    }.get(outcome, outcome.replace("_", " ").title())


def result_tone(outcome: str) -> str:
    return {
        "PASSED": "success", "PRODUCT_FAILURE": "product-failure",
        "AUTOMATION_DRIFT": "drift", "AUTOMATION_EXECUTION_ERROR": "infrastructure",
        "AUTOMATION_GENERATION_ERROR": "warning", "INFRASTRUCTURE_ERROR": "infrastructure",
        "BLOCKED": "warning",
    }.get(outcome, "neutral")


def result_explanation(outcome: str) -> str:
    return {
        "PASSED": "All test steps completed and the verified behavior passed.",
        "PRODUCT_FAILURE": "A trustworthy assertion detected unexpected product behavior.",
        "AUTOMATION_DRIFT": "The saved automation no longer matches the current UI.",
        "AUTOMATION_EXECUTION_ERROR": "The automation could not reliably complete or verify the expected behavior.",
        "AUTOMATION_GENERATION_ERROR": "Automation generation or semantic validation stopped before all steps could execute.",
        "INFRASTRUCTURE_ERROR": "A browser, storage, environment or runtime problem prevented reliable execution.",
        "BLOCKED": "An unmet prerequisite or dependency prevented execution.",
        "INCONCLUSIVE": "The available evidence does not establish whether the product behavior passed or failed.",
    }.get(outcome, "Execution has not reached a confirmed result.")


def recommended_action(outcome: str) -> str:
    return {
        "PASSED": "Review the recorded steps and evidence.",
        "PRODUCT_FAILURE": "View evidence and review defect details against the saved requirement.",
        "AUTOMATION_DRIFT": "Review the assertion and the exact saved TestPlan version against the current UI.",
        "AUTOMATION_EXECUTION_ERROR": "Review the assertion and the exact saved TestPlan version; inspect evidence before editing automation.",
        "AUTOMATION_GENERATION_ERROR": "Review the TestCase and validation diagnostics, then retry Automation.",
        "INFRASTRUCTURE_ERROR": "Open System Health and review configuration before retrying.",
        "BLOCKED": "Review prerequisites, approval and saved automation availability.",
    }.get(outcome, "Review the available evidence and developer details before drawing a product conclusion.")


def terminal_phase(outcome: str) -> str:
    if outcome == "PASSED":
        return "Completed — Passed"
    if outcome == "PRODUCT_FAILURE":
        return "Completed — Product Failure"
    if outcome == "BLOCKED":
        return "Blocked — Prerequisite Not Met"
    return f"Stopped — {result_label(outcome)}"


def execution_classification(execution: Any) -> str | None:
    """Use only the classification recorded with this exact execution attempt."""
    metadata = execution.runner_result or {}
    value = metadata.get("qa_classification")
    if value is not None:
        return result_outcome(execution.status, value)
    if enum_value(execution.status) == "PASSED":
        return "PASSED"
    return None


def execution_observation(execution: Any) -> Any:
    metadata = execution.runner_result or {}
    for step in metadata.get("steps", []):
        if isinstance(step, dict) and step.get("status") == "failed":
            for key in ("observed", "actual_result", "actual"):
                if key in step:
                    return step[key]
    return execution.actual_result


def history_outcome(record: Any) -> str:
    final_by_step = {item.test_step_id: item for item in record.executions}
    complete = record.finished_at is not None and bool(record.steps) and all(
        enum_value(step.status) == "PASSED"
        and step.id in final_by_step
        and enum_value(final_by_step[step.id].status) == "PASSED"
        and final_by_step[step.id].finished_at is not None
        for step in record.steps
    )
    return result_outcome(record.status, record.outcome, complete=complete)


def result_summary(outcome: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Count verified steps and identify an issue without claiming missing work passed."""
    completed = sum(step["outcome"] in {"PASSED", "PRODUCT_FAILURE"} for step in steps)
    issues = [step for step in steps if step["outcome"] not in {"PASSED", "READY", "PENDING"}]
    issue = next((step for step in issues if step["outcome"] == outcome), issues[0] if issues else None)
    position = steps.index(issue) + 1 if issue is not None else None
    stopping = (
        f"{'Failure' if outcome == 'PRODUCT_FAILURE' else 'Stopped'} at Step {position} of {len(steps)}: {issue['name']}."
        if issue is not None and outcome != "PASSED" else None
    )
    return {
        "outcome": outcome,
        "status": result_label(outcome),
        "completed_steps": completed,
        "total_steps": len(steps),
        "stopping_step": position if outcome != "PASSED" else None,
        "stopping_detail": stopping,
        "explanation": result_explanation(outcome),
        "recommended_action": recommended_action(outcome),
    }
