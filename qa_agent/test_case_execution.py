"""Application service for running a persisted TestCase with saved plans."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from qa_agent.browser_runner import BrowserRunner
from qa_agent.execution_repository import ExecutionRepository
from qa_agent.models import QATestPlan
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
from qa_agent.workflows import PinnedWorkflowResult, RegressionWorkflow, ValidationWorkflow


class RunUnavailableError(ValueError):
    """A safe, user-facing reason a persisted TestCase cannot be run."""


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

    def run(
        self,
        test_case_id: UUID,
        workflow_type: WorkflowType,
    ) -> PinnedWorkflowResult:
        if workflow_type not in {WorkflowType.VALIDATION, WorkflowType.REGRESSION}:
            raise RunUnavailableError(
                "Automation is not available for persisted TestCases yet."
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
