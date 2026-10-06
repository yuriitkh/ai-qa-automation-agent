"""Execute one exact plan version and classify its execution outcome."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from qa_agent.execution_repository import ExecutionRepository
from qa_agent.execution_progress import (
    ExecutionEventType,
    emit_progress_event,
)
from qa_agent.models import (
    Evidence,
    EvidenceType,
    Execution,
    ExecutionStatus,
    QATestPlan,
    TestPlanVersion,
    TestStep,
)
from qa_agent.presentation import failure_message


class PlanExecutionClassification(str, Enum):
    PASSED = "PASSED"
    PRODUCT_FAILURE = "PRODUCT_FAILURE"
    AUTOMATION_DRIFT = "AUTOMATION_DRIFT"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


@dataclass(frozen=True)
class PlanExecutionOutcome:
    execution: Execution
    classification: PlanExecutionClassification
    error: Exception | None = None


class PlanExecutionPersistenceError(RuntimeError):
    """An execution attempt could not be persisted."""


class PlanExecutionService:
    """Run and persist one TestStep against the exact supplied plan version.

    Plan selection, discovery, generation, recovery, and orchestration are
    deliberately outside this service.
    """

    def __init__(
        self,
        runner: Callable[[QATestPlan], dict[str, Any]],
        execution_repository: ExecutionRepository,
    ) -> None:
        self._runner = runner
        self._execution_repository = execution_repository

    def execute(
        self,
        test_step: TestStep,
        plan_version: TestPlanVersion,
    ) -> PlanExecutionOutcome:
        started_at = datetime.now(timezone.utc)
        emit_progress_event(
            ExecutionEventType.STEP_STARTED,
            step=test_step,
            status=ExecutionStatus.RUNNING.value,
            message="Running step.",
        )
        runner_result: dict[str, Any] | None = None
        execution_error: Exception | None = None
        try:
            runner_output = self._runner(plan_version.qa_test_plan)
            if not isinstance(runner_output, dict):
                raise TypeError("Browser runner must return a result dictionary.")
            runner_result = runner_output
            runner_status = runner_result.get("status")
            if runner_status not in {"passed", "failed"}:
                raise ValueError(
                    f"Browser runner returned unsupported status {runner_status!r}."
                )
        except Exception as error:
            execution_error = error
            runner_status = "failed"

        status = (
            ExecutionStatus.PASSED
            if runner_status == "passed"
            else ExecutionStatus.FAILED
        )
        execution_id = uuid4()
        execution = Execution(
            id=execution_id,
            test_step_id=test_step.id,
            test_plan_version_id=plan_version.id,
            planned_step_index=(
                self._failed_plan_step_index(plan_version, runner_result)
                if status == ExecutionStatus.FAILED
                else None
            ),
            status=status,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            actual_result="failed" if execution_error is not None else runner_status,
            error=(
                str(execution_error)
                if execution_error is not None
                else self._runner_error(runner_result)
                if status == ExecutionStatus.FAILED
                else None
            ),
            runner_result=runner_result,
            evidence=self._runner_evidence(runner_result, status, execution_id),
        )
        try:
            self._execution_repository.save(execution)
        except Exception as error:
            emit_progress_event(
                ExecutionEventType.STEP_FAILED,
                step=test_step,
                status=ExecutionStatus.FAILED.value,
                classification=PlanExecutionClassification.INFRASTRUCTURE_ERROR.value,
                message="The execution result could not be saved.",
            )
            raise PlanExecutionPersistenceError(str(error)) from error

        classification = self._classify(execution, execution_error)
        for evidence_index, _item in enumerate(execution.evidence):
            emit_progress_event(
                ExecutionEventType.EVIDENCE_CAPTURED,
                step=test_step,
                evidence_execution_id=execution.id,
                evidence_index=evidence_index,
                message="Screenshot evidence captured.",
            )
        if status == ExecutionStatus.PASSED:
            emit_progress_event(
                ExecutionEventType.STEP_PASSED,
                step=test_step,
                status=status.value,
                classification=classification.value,
                message="Step passed.",
            )
        else:
            emit_progress_event(
                ExecutionEventType.STEP_FAILED,
                step=test_step,
                status=status.value,
                classification=classification.value,
                message=failure_message(classification.value),
            )
        return PlanExecutionOutcome(
            execution=execution,
            classification=classification,
            error=execution_error,
        )

    @staticmethod
    def _classify(
        execution: Execution,
        execution_error: Exception | None,
    ) -> PlanExecutionClassification:
        if execution_error is not None:
            return PlanExecutionClassification.INFRASTRUCTURE_ERROR
        if execution.status == ExecutionStatus.PASSED:
            return PlanExecutionClassification.PASSED
        if _is_stale_ui_failure(execution.runner_result):
            return PlanExecutionClassification.AUTOMATION_DRIFT

        failed_action = _failed_action(execution.runner_result)
        if failed_action is not None and failed_action.startswith("assert_"):
            return PlanExecutionClassification.PRODUCT_FAILURE
        # A returned failure without explicit assertion evidence is not
        # attributed to the product; its cause is not sufficiently known.
        return PlanExecutionClassification.INFRASTRUCTURE_ERROR

    @staticmethod
    def _runner_error(runner_result: dict[str, Any]) -> str:
        for step_result in runner_result.get("steps", []):
            if isinstance(step_result, dict) and step_result.get("status") == "failed":
                return str(step_result.get("error") or "Browser plan failed.")
        return "Browser plan failed."

    @staticmethod
    def _failed_plan_step_index(
        plan_version: TestPlanVersion,
        runner_result: dict[str, Any] | None,
    ) -> int | None:
        if not isinstance(runner_result, dict):
            return None
        for index, step_result in enumerate(runner_result.get("steps", [])):
            if isinstance(step_result, dict) and step_result.get("status") == "failed":
                return index if index < len(plan_version.qa_test_plan.steps) else None
        return None

    @staticmethod
    def _runner_evidence(
        runner_result: dict[str, Any] | None,
        status: ExecutionStatus,
        execution_id,
    ) -> tuple[Evidence, ...]:
        if status != ExecutionStatus.FAILED or not isinstance(runner_result, dict):
            return ()
        items = runner_result.get("evidence", [])
        if not isinstance(items, list):
            return ()
        evidence = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != EvidenceType.SCREENSHOT.value:
                continue
            path = item.get("path")
            if not isinstance(path, str) or not path:
                continue
            evidence_id = uuid4()
            item["evidence_id"] = str(evidence_id)
            evidence.append(Evidence(
                id=evidence_id,
                execution_id=execution_id,
                type=EvidenceType.SCREENSHOT,
                path=path,
                description=item.get("description"),
            ))
        return tuple(evidence)


def _failed_action(runner_result: dict[str, Any] | None) -> str | None:
    if not isinstance(runner_result, dict):
        return None
    for step_result in runner_result.get("steps", []):
        if isinstance(step_result, dict) and step_result.get("status") == "failed":
            action = step_result.get("action")
            return action if isinstance(action, str) else None
    return None


def _is_stale_ui_failure(runner_result: dict[str, Any] | None) -> bool:
    """Retain the pipeline's existing narrow locator-drift predicate."""
    if not isinstance(runner_result, dict):
        return False

    for step_result in runner_result.get("steps", []):
        if not isinstance(step_result, dict) or step_result.get("status") != "failed":
            continue
        if step_result.get("action") not in {"click", "fill"}:
            continue

        error = str(step_result.get("error") or "").casefold()
        selector_missing = "selector" in error and (
            "not found" in error or "was not found" in error
        )
        element_unavailable = any(
            marker in error
            for marker in (
                "resolved to hidden",
                "resolved to 0 elements",
                "element is not visible",
                "waiting for element to be visible",
                "not attached to the dom",
                "detached from the dom",
            )
        )
        locator_actionability_timeout = (
            "timeout" in error
            and "waiting for locator" in error
            and (
                "could not click element matching selector" in error
                or "could not fill element matching selector" in error
            )
        )
        if selector_missing or element_unavailable or locator_actionability_timeout:
            return True
    return False
