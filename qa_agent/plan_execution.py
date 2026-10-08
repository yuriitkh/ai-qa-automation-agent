"""Execute one exact plan version and classify its execution outcome."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from inspect import getattr_static
from typing import Any, Callable
from uuid import uuid4

from qa_agent.execution_repository import ExecutionRepository
from qa_agent.evidence_policy import EvidenceExecutionIdentity, EvidenceScope, evidence_execution_scope
from qa_agent.execution_progress import (
    ExecutionEventType,
    emit_progress_event,
)
from qa_agent.models import (
    AssertionGrounding,
    Evidence,
    EvidenceType,
    Execution,
    ExecutionStatus,
    QATestPlan,
    TestPlanVersion,
    TestStep,
    TestCase,
)
from qa_agent.presentation import failure_message


class PlanExecutionClassification(str, Enum):
    PASSED = "PASSED"
    PRODUCT_FAILURE = "PRODUCT_FAILURE"
    AUTOMATION_DRIFT = "AUTOMATION_DRIFT"
    AUTOMATION_EXECUTION_ERROR = "AUTOMATION_EXECUTION_ERROR"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


@dataclass(frozen=True)
class PlanExecutionOutcome:
    execution: Execution
    classification: PlanExecutionClassification
    error: Exception | None = None


class PlanExecutionPersistenceError(RuntimeError):
    """An execution attempt could not be persisted."""


class TestCasePlanRunner:
    """Bind a runner to one TestCase and route steps by segment order."""

    def __init__(self, runner: Callable[[QATestPlan], dict[str, Any]]) -> None:
        self._runner = runner
        self._session_context: Any | None = None
        self._session: Any | None = None
        self._segment_by_step: dict[Any, int] = {}
        self._closed = False

    @property
    def browser_session_started(self) -> bool:
        return bool(getattr(self._session, "is_started", False))

    def bind(self, test_case: TestCase) -> None:
        if self._session_context is not None or self._segment_by_step:
            raise RuntimeError("TestCase runner is already bound.")
        self._segment_by_step = {
            step.id: segment.order
            for segment in test_case.segments
            for step in segment.steps
        }
        try:
            getattr_static(self._runner, "open_test_case_session")
        except AttributeError:
            opener = None
        else:
            opener = getattr(self._runner, "open_test_case_session")
        if callable(opener):
            context = opener(test_case)
            session = context.__enter__()
            self._session_context = context
            self._session = session

    def for_step(self, test_step: TestStep) -> Callable[[QATestPlan], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("TestCase runner is closed.")
        segment_order = self._segment_by_step.get(test_step.id)
        if segment_order is None:
            raise ValueError("TestStep does not belong to the bound TestCase.")
        if self._session is None:
            return self._runner
        return lambda plan: self._session.run_plan(segment_order, plan)

    def close(self, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        context = self._session_context
        self._session_context = None
        if context is not None:
            if primary_error is None:
                context.__exit__(None, None, None)
            else:
                context.__exit__(
                    type(primary_error),
                    primary_error,
                    primary_error.__traceback__,
                )


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

    def new_test_case_runner(self) -> TestCasePlanRunner:
        return TestCasePlanRunner(self._runner)

    def execute(
        self,
        test_step: TestStep,
        plan_version: TestPlanVersion,
        *,
        runner: Callable[[QATestPlan], dict[str, Any]] | None = None,
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
        execution_id = uuid4()
        try:
            active_runner = runner if runner is not None else self._runner
            with evidence_execution_scope(EvidenceExecutionIdentity(execution_id, test_step.id)):
                runner_output = active_runner(plan_version.qa_test_plan)
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
            evidence=self._runner_evidence(runner_result, execution_id),
        )
        classification = self._classify(execution, execution_error, plan_version)
        grounding_reason = self._grounding_failure_reason(
            execution, plan_version, classification
        )
        if grounding_reason is not None:
            # This static explanation intentionally replaces runner text that
            # can contain sensitive expected/actual page values.
            execution = execution.model_copy(update={"error": grounding_reason})
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
        plan_version: TestPlanVersion | None = None,
    ) -> PlanExecutionClassification:
        if execution_error is not None:
            return PlanExecutionClassification.INFRASTRUCTURE_ERROR
        if execution.status == ExecutionStatus.PASSED:
            return PlanExecutionClassification.PASSED
        if _is_infrastructure_failure(execution.runner_result):
            return PlanExecutionClassification.INFRASTRUCTURE_ERROR
        if _is_stale_ui_failure(execution.runner_result):
            return PlanExecutionClassification.AUTOMATION_DRIFT
        if _is_assertion_target_unavailable(execution.runner_result):
            return PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR

        failed_action = _failed_action(execution.runner_result)
        if failed_action is not None and failed_action.startswith("assert_"):
            if plan_version is not None:
                grounding = _assertion_grounding(execution, plan_version)
                if plan_version.origin is not None and plan_version.origin.value == "HUMAN_EDITED":
                    pass
                elif grounding == AssertionGrounding.REQUIREMENT_GROUNDED:
                    pass
                elif grounding == AssertionGrounding.OBSERVATION_GROUNDED:
                    return PlanExecutionClassification.AUTOMATION_DRIFT
                else:
                    return PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR
            return PlanExecutionClassification.PRODUCT_FAILURE
        if failed_action is not None:
            return PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR
        # A returned failure without explicit assertion evidence is not
        # attributed to the product; its cause is not sufficiently known.
        return PlanExecutionClassification.INFRASTRUCTURE_ERROR

    @staticmethod
    def _grounding_failure_reason(
        execution: Execution,
        plan_version: TestPlanVersion,
        classification: PlanExecutionClassification,
    ) -> str | None:
        failed_action = _failed_action(execution.runner_result)
        if failed_action is None or not failed_action.startswith("assert_"):
            return None
        interaction = execution.planned_interaction(plan_version)
        if interaction is None or interaction.action != failed_action:
            return None
        if plan_version.origin is not None and plan_version.origin.value == "HUMAN_EDITED":
            return None
        grounding = _assertion_grounding(execution, plan_version)
        if grounding == AssertionGrounding.OBSERVATION_GROUNDED:
            return (
                "This assertion matched deterministic page evidence when the plan was "
                "generated. The current failure is treated as possible automation drift."
            )
        if (
            classification == PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR
            and grounding in {
                None,
                AssertionGrounding.INFERRED,
                AssertionGrounding.UNKNOWN,
            }
        ):
            if grounding is None:
                return (
                    "This saved assertion has no grounding metadata, so its failure "
                    "cannot be attributed to product behavior."
                )
            return (
                "This assertion was not grounded in the requirement or deterministic "
                "page evidence, so its failure cannot be attributed to product behavior."
            )
        return None

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
        execution_id,
    ) -> tuple[Evidence, ...]:
        if not isinstance(runner_result, dict):
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
            try:
                scope = EvidenceScope(item["scope"]) if isinstance(item.get("scope"), str) else None
            except ValueError:
                scope = None
            evidence.append(Evidence(
                id=evidence_id,
                execution_id=execution_id,
                type=EvidenceType.SCREENSHOT,
                path=path,
                description=item.get("description"),
                scope=scope,
                event=item.get("event") if isinstance(item.get("event"), str) else None,
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


def _assertion_grounding(
    execution: Execution,
    plan_version: TestPlanVersion,
) -> AssertionGrounding | None:
    """Read metadata only for the exact failed action in this saved version."""
    index = execution.planned_step_index
    if index is None or plan_version.id != execution.test_plan_version_id:
        return None
    failed_action = _failed_action(execution.runner_result)
    interaction = execution.planned_interaction(plan_version)
    if (
        interaction is None
        or not interaction.action.startswith("assert_")
        or interaction.action != failed_action
    ):
        return None
    for entry in plan_version.assertion_grounding or ():
        if entry.step_index == index:
            return entry.category
    return None


def _is_stale_ui_failure(runner_result: dict[str, Any] | None) -> bool:
    """Retain the pipeline's existing narrow locator-drift predicate."""
    if not isinstance(runner_result, dict):
        return False

    for step_result in runner_result.get("steps", []):
        if not isinstance(step_result, dict) or step_result.get("status") != "failed":
            continue
        action = step_result.get("action")
        error = str(step_result.get("error") or "").casefold()
        selector_missing = (
            "selector" in error
            and any(marker in error for marker in ("not found", "was not found"))
        ) or any(marker in error for marker in (
            "resolved to 0 elements",
            "resolved to 2 elements",
            "matched multiple elements",
            "strict mode violation",
            "not attached to the dom",
            "detached from the dom",
        ))
        element_unavailable = any(
            marker in error
            for marker in (
                "resolved to hidden",
                "resolved to 0 elements",
                "element is not visible",
                "waiting for element to be visible",
            )
        ) and action in {"click", "fill", "select_option"}
        locator_actionability_timeout = (
            "timeout" in error
            and "waiting for locator" in error
            and (
                "could not click element matching selector" in error
                or "could not fill element matching selector" in error
                or action == "select_option"
            )
        )
        option_missing = action == "select_option" and any(marker in error for marker in (
            "did not match any options",
            "no options matched",
            "option label was not found",
        ))
        if (
            (selector_missing and action in {"click", "fill", "select_option"})
            or element_unavailable
            or locator_actionability_timeout
            or option_missing
        ):
            return True
    return False


def _is_assertion_target_unavailable(runner_result: dict[str, Any] | None) -> bool:
    """Separate a missing generated assertion target from product mismatches."""
    if not isinstance(runner_result, dict):
        return False
    unavailable_markers = (
        "not found",
        "was not found",
        "resolved to 0 elements",
        "resolved to hidden",
        "element is not visible",
        "not attached to the dom",
        "detached from the dom",
        "strict mode violation",
        "matched multiple elements",
    )
    for step_result in runner_result.get("steps", []):
        if not isinstance(step_result, dict) or step_result.get("status") != "failed":
            continue
        action = step_result.get("action")
        error = str(step_result.get("error") or "").casefold()
        if (
            isinstance(action, str)
            and action in {
                "assert_visible",
                "assert_checked",
                "assert_selected",
                "assert_enabled",
                "assert_disabled",
            }
            and (
                ("selector" in error and any(marker in error for marker in unavailable_markers))
                or ("timeout" in error and "waiting for locator" in error)
            )
        ):
            return True
    return False


def _is_infrastructure_failure(runner_result: dict[str, Any] | None) -> bool:
    if not isinstance(runner_result, dict):
        return False
    markers = (
        "browser has been closed",
        "browser disconnected",
        "browser launch failed",
        "target page, context or browser has been closed",
        "page has been closed",
        "page crashed",
        "net::err_",
        "connection refused",
        "connection reset",
        "name_not_resolved",
        "protocol error",
    )
    return any(
        isinstance(step_result, dict)
        and step_result.get("status") == "failed"
        and any(marker in str(step_result.get("error") or "").casefold() for marker in markers)
        for step_result in runner_result.get("steps", [])
    )
