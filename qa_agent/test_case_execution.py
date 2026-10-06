"""Application service for running a persisted TestCase with saved plans."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from qa_agent.browser_runner import BrowserRunner
from qa_agent.execution_repository import ExecutionRepository
from qa_agent.models import QATestPlan
from qa_agent.pipeline import PipelineResult
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.pinned_execution import (
    PinnedExecutionService,
    PlanSelectionError,
    PlanVersionSet,
    StepPlanSelection,
)
from qa_agent.plan_store import PlanStore
from qa_agent.run_context import RunContext
from qa_agent.run_history import RunHistoryService, WorkflowType
from qa_agent.setup_orchestration import SetupCleanupCoordinator
from qa_agent.test_case_repository import TestCaseRepository
from qa_agent.workflows import (
    AutomationWorkflow,
    PinnedWorkflowResult,
    RegressionWorkflow,
    ValidationWorkflow,
)


class RunUnavailableError(ValueError):
    """A safe, user-facing reason a persisted TestCase cannot be run."""


@dataclass(frozen=True)
class WorkflowAvailability:
    automation_available: bool
    validation_available: bool
    regression_available: bool
    usable_plan_count: int
    total_step_count: int
    plan_versions: tuple[tuple[UUID, int, UUID], ...]
    reason: str | None = None


RunnerFactory = Callable[[Path | None], Callable[[QATestPlan], dict[str, Any]]]


class TestCaseExecutionService:
    """Resolve saved plan versions, then delegate execution to a real workflow."""

    __test__ = False

    def __init__(
        self,
        test_cases: TestCaseRepository,
        plan_store: PlanStore,
        execution_repository: ExecutionRepository,
        run_history: RunHistoryService,
        *,
        evidence_directory: str | Path | None = None,
        runner_factory: RunnerFactory | None = None,
        setup_cleanup_factory: Callable[[], SetupCleanupCoordinator] | None = None,
        automation_workflow: AutomationWorkflow | None = None,
    ) -> None:
        self._test_cases = test_cases
        self._plan_store = plan_store
        self._execution_repository = execution_repository
        self._run_history = run_history
        self._evidence_directory = (
            Path(evidence_directory).expanduser().resolve()
            if evidence_directory is not None else None
        )
        self._runner_factory = runner_factory or (
            lambda directory: BrowserRunner(directory, headless=True)
        )
        self._setup_cleanup_factory = setup_cleanup_factory or (
            lambda: SetupCleanupCoordinator({})
        )
        self._automation_workflow = automation_workflow

    def workflow_availability(self, test_case_id: UUID) -> WorkflowAvailability:
        test_case = self._test_cases.get(test_case_id)
        if test_case is None:
            raise RunUnavailableError("TestCase not found.")
        versions: list[tuple[UUID, int, UUID]] = []
        steps = sorted(test_case.steps, key=lambda item: item.order)
        for step in steps:
            version = self._plan_store.find(step.id)
            plan = self._plan_store.find_test_plan(step.id)
            saved_version = self._plan_store.get_version(version.id) if version else None
            if (
                version is not None
                and plan is not None
                and saved_version == version
                and plan.test_step_id == step.id
                and version.test_plan_id == plan.id
                and bool(version.qa_test_plan.steps)
            ):
                versions.append((step.id, version.version, version.id))
        complete = bool(steps) and len(versions) == len(steps)
        ready_reason = None if complete else "No complete automation version has been generated yet."
        return WorkflowAvailability(
            automation_available=self._automation_workflow is not None,
            validation_available=complete,
            regression_available=complete,
            usable_plan_count=len(versions),
            total_step_count=len(steps),
            plan_versions=tuple(versions),
            reason=ready_reason,
        )

    def run(
        self,
        test_case_id: UUID,
        workflow_type: WorkflowType,
    ) -> PinnedWorkflowResult | PipelineResult:
        if workflow_type == WorkflowType.AUTOMATION:
            if self._automation_workflow is None:
                raise RunUnavailableError(
                    "Automation is not available for persisted TestCases yet."
                )
            test_case = self._test_cases.get(test_case_id)
            if test_case is None:
                raise RunUnavailableError("TestCase not found.")
            return self._automation_workflow.run_test_case(test_case, RunContext())
        if workflow_type not in {WorkflowType.VALIDATION, WorkflowType.REGRESSION}:
            raise RunUnavailableError(
                "Choose Automation, Validation, or Regression before starting a run."
            )

        test_case = self._test_cases.get(test_case_id)
        if test_case is None:
            raise RunUnavailableError("TestCase not found.")

        selections: list[StepPlanSelection] = []
        missing_steps = []
        for step in sorted(test_case.steps, key=lambda item: item.order):
            version = self._plan_store.find(step.id)
            if version is None:
                missing_steps.append(step.name)
            else:
                selections.append(StepPlanSelection(step.id, version.id))
        if missing_steps:
            raise RunUnavailableError(
                "This TestCase is not ready to run because one or more steps have no saved plan."
            )
        availability = self.workflow_availability(test_case_id)
        if not availability.validation_available:
            raise RunUnavailableError(
                "This TestCase is not ready to run because one or more saved plans are not usable."
            )

        runner = self._runner_factory(self._evidence_directory)
        plan_execution = PlanExecutionService(runner, self._execution_repository)
        executor = PinnedExecutionService(self._plan_store, plan_execution)
        workflow_class = (
            ValidationWorkflow
            if workflow_type == WorkflowType.VALIDATION
            else RegressionWorkflow
        )
        workflow = workflow_class(
            executor,
            self._setup_cleanup_factory(),
            self._run_history,
        )
        try:
            return workflow.run(
                test_case,
                PlanVersionSet(tuple(selections)),
                RunContext(),
            )
        except PlanSelectionError as error:
            raise RunUnavailableError(
                "Saved plans changed before this run started. Reload the TestCase and try again."
            ) from error
