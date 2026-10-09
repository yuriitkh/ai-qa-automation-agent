"""Exact plan-version selection and execution for non-learning workflows."""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable
from uuid import UUID

from qa_agent.execution_control import (
    current_cancellation,
)
from qa_agent.models import (
    Execution,
    ExecutionStatus,
    FailurePolicy,
    TestCase,
    TestPlanVersion,
    TestRun,
    TestStep,
)
from qa_agent.execution_progress import (
    ExecutionEventType,
    emit_progress_event,
)
from qa_agent.plan_execution import (
    PlanExecutionClassification,
    PlanExecutionPersistenceError,
    PlanExecutionService,
)
from qa_agent.plan_store import PlanStore
from qa_agent.run_context import RunContext
from qa_agent.test_plan_validation import validate_executable_plan


@dataclass(frozen=True)
class StepPlanSelection:
    """Pin one TestStep to one immutable TestPlanVersion identity."""

    test_step_id: UUID
    test_plan_version_id: UUID


@dataclass(frozen=True)
class PlanVersionSet:
    """In-memory exact selections; tuple form preserves duplicate detection."""

    selections: tuple[StepPlanSelection, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selections", tuple(self.selections))

    @classmethod
    def from_mapping(cls, selections: dict[UUID, UUID]) -> "PlanVersionSet":
        return cls(tuple(
            StepPlanSelection(step_id, version_id)
            for step_id, version_id in selections.items()
        ))


class PlanSelectionError(ValueError):
    """A pinned set is incomplete or does not match the TestCase and store."""


@dataclass(frozen=True)
class ResolvedStepPlan:
    test_step: TestStep
    plan_version: TestPlanVersion


@dataclass(frozen=True)
class ResolvedPlanVersionSet:
    """Selections resolved and ordered by their TestCase TestStep order."""

    steps: tuple[ResolvedStepPlan, ...]


class WorkflowOutcome(str, Enum):
    CANCELLED = "CANCELLED"
    PASSED = "PASSED"
    PRODUCT_FAILURE = "PRODUCT_FAILURE"
    AUTOMATION_DRIFT = "AUTOMATION_DRIFT"
    AUTOMATION_EXECUTION_ERROR = "AUTOMATION_EXECUTION_ERROR"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    SETUP_FAILURE = "SETUP_FAILURE"


@dataclass(frozen=True)
class PinnedStepExecution:
    test_step_id: UUID
    test_plan_version_id: UUID
    classification: PlanExecutionClassification
    execution: Execution


@dataclass(frozen=True)
class PinnedExecutionResult:
    outcome: WorkflowOutcome
    test_run: TestRun
    step_executions: tuple[PinnedStepExecution, ...]
    error: Exception | None = field(default=None, repr=False, compare=False)
    cleanup_error: Exception | None = field(default=None, repr=False, compare=False)

    @property
    def executions(self) -> list[Execution]:
        return self.test_run.executions


class PinnedExecutionService:
    """Execute a TestCase against only the exact selected plan versions.

    This service has no discovery, generation, repair, regeneration, or
    current-version lookup dependency.
    """

    def __init__(
        self,
        plan_store: PlanStore,
        plan_execution: PlanExecutionService,
    ) -> None:
        self._plan_store = plan_store
        self._plan_execution = plan_execution

    def resolve(
        self,
        test_case: TestCase,
        selection: PlanVersionSet,
    ) -> ResolvedPlanVersionSet:
        ordered_steps = sorted(test_case.steps, key=lambda step: step.order)
        required = {step.id: step for step in ordered_steps}
        selected: dict[UUID, UUID] = {}

        for item in selection.selections:
            if item.test_step_id not in required:
                raise PlanSelectionError(
                    f"Plan selection names unknown TestStep {item.test_step_id}."
                )
            if item.test_step_id in selected:
                raise PlanSelectionError(
                    f"TestStep {item.test_step_id} has multiple selected versions."
                )
            selected[item.test_step_id] = item.test_plan_version_id

        missing = [step.id for step in ordered_steps if step.id not in selected]
        if missing:
            raise PlanSelectionError(
                "Plan selection is missing TestSteps: "
                + ", ".join(str(step_id) for step_id in missing)
            )

        resolved: list[ResolvedStepPlan] = []
        for step in ordered_steps:
            version_id = selected[step.id]
            version = self._plan_store.get_version(version_id)
            if version is None:
                raise PlanSelectionError(
                    f"Unknown TestPlanVersion {version_id} for TestStep {step.id}."
                )
            test_plan = self._plan_store.find_test_plan(step.id)
            if test_plan is None or test_plan.test_step_id != step.id:
                raise PlanSelectionError(
                    f"No stored TestPlan belongs to TestStep {step.id}."
                )
            if version.test_plan_id != test_plan.id:
                raise PlanSelectionError(
                    f"TestPlanVersion {version_id} does not belong to TestStep {step.id}."
                )
            try:
                validate_executable_plan(version.qa_test_plan)
            except (TypeError, ValueError) as error:
                raise PlanSelectionError(
                    f"Selected TestPlanVersion for TestStep {step.id} is not executable."
                ) from error
            resolved.append(ResolvedStepPlan(step, version))

        return ResolvedPlanVersionSet(tuple(resolved))

    def execute(
        self,
        test_case: TestCase,
        resolved: ResolvedPlanVersionSet,
        run_context: RunContext,
    ) -> PinnedExecutionResult:
        expected_step_ids = [
            step.id for step in sorted(test_case.steps, key=lambda step: step.order)
        ]
        resolved_step_ids = [item.test_step.id for item in resolved.steps]
        if resolved_step_ids != expected_step_ids:
            raise PlanSelectionError(
                "Resolved plan versions must cover this TestCase exactly in step order."
            )
        case_runner = self._plan_execution.new_test_case_runner()
        try:
            case_runner.bind(test_case)
            result = self._execute_steps(
                test_case, resolved, run_context, case_runner
            )
        except BaseException as error:
            case_runner.close(primary_error=error)
            raise
        else:
            try:
                case_runner.close()
            except Exception as error:
                # Cleanup must finish on the owning thread. Even a cleanup
                # error must not lose already persisted execution references.
                if current_cancellation() is None or not current_cancellation().requested:
                    raise
                result = replace(result, cleanup_error=error)
            return result

    def _execute_steps(
        self,
        test_case: TestCase,
        resolved: ResolvedPlanVersionSet,
        run_context: RunContext,
        case_runner,
    ) -> PinnedExecutionResult:
        executions: list[Execution] = []
        step_executions: list[PinnedStepExecution] = []
        blocked_step_ids: list[UUID] = []
        error: Exception | None = None

        for index, selected in enumerate(resolved.steps):
            if current_cancellation() is not None and current_cancellation().requested:
                break
            emit_progress_event(
                ExecutionEventType.PLAN_REUSED,
                step=selected.test_step,
                plan_origin=(selected.plan_version.origin.value
                             if selected.plan_version.origin is not None
                             else "PINNED"),
                plan_version=selected.plan_version.version,
                plan_version_id=selected.plan_version.id,
                message="Pinned automation loaded.",
            )
            try:
                result = self._plan_execution.execute(
                    selected.test_step,
                    selected.plan_version,
                    runner=case_runner.for_step(selected.test_step),
                )
            except PlanExecutionPersistenceError as caught:
                error = caught
                break

            execution = result.execution
            executions.append(execution)
            step_executions.append(PinnedStepExecution(
                test_step_id=selected.test_step.id,
                test_plan_version_id=selected.plan_version.id,
                classification=result.classification,
                execution=execution,
            ))
            if execution.status == ExecutionStatus.CANCELLED or (current_cancellation() is not None and current_cancellation().requested):
                break
            if result.error is not None:
                error = result.error
                break

            if (
                isinstance(execution.runner_result, dict)
                and execution.runner_result.get("cookie_consent_requires_attention") is True
            ):
                blocked_step_ids = [
                    following.test_step.id
                    for following in resolved.steps[index + 1:]
                ]
                for following in resolved.steps[index + 1:]:
                    emit_progress_event(
                        ExecutionEventType.STEP_BLOCKED,
                        step=following.test_step,
                        status=ExecutionStatus.BLOCKED.value,
                        message="Blocked because cookie consent requires attention.",
                    )
                break

            if (
                execution.status == ExecutionStatus.FAILED
                and selected.test_step.failure_policy == FailurePolicy.BLOCK_REST
            ):
                blocked_step_ids = [
                    following.test_step.id
                    for following in resolved.steps[index + 1:]
                ]
                for following in resolved.steps[index + 1:]:
                    emit_progress_event(
                        ExecutionEventType.STEP_BLOCKED,
                        step=following.test_step,
                        status=ExecutionStatus.BLOCKED.value,
                        message="Blocked by the preceding step's failure policy.",
                    )
                break

        test_run = TestRun.from_test_case(
            test_case,
            executions,
            blocked_step_ids=blocked_step_ids,
            run_context=run_context,
            cancelled=current_cancellation() is not None and current_cancellation().requested,
        )
        outcome = WorkflowOutcome.CANCELLED if test_run.cancelled else _workflow_outcome(step_executions, error)
        return PinnedExecutionResult(
            outcome=outcome,
            test_run=test_run,
            step_executions=tuple(step_executions),
            error=error,
        )


def _workflow_outcome(
    step_executions: Iterable[PinnedStepExecution],
    error: Exception | None,
) -> WorkflowOutcome:
    if error is not None:
        return WorkflowOutcome.INFRASTRUCTURE_ERROR
    classifications = {item.classification for item in step_executions}
    if PlanExecutionClassification.INFRASTRUCTURE_ERROR in classifications:
        return WorkflowOutcome.INFRASTRUCTURE_ERROR
    if PlanExecutionClassification.AUTOMATION_DRIFT in classifications:
        return WorkflowOutcome.AUTOMATION_DRIFT
    if PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR in classifications:
        return WorkflowOutcome.AUTOMATION_EXECUTION_ERROR
    if PlanExecutionClassification.PRODUCT_FAILURE in classifications:
        return WorkflowOutcome.PRODUCT_FAILURE
    return WorkflowOutcome.PASSED
