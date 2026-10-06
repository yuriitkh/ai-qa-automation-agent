"""Thin automation, validation, and regression workflow entry points."""

from dataclasses import dataclass, field

from qa_agent.models import TestCase, TestRun
from qa_agent.pipeline import PipelineResult, QATestPipeline
from qa_agent.pinned_execution import (
    PinnedExecutionResult,
    PinnedExecutionService,
    PlanVersionSet,
    ResolvedPlanVersionSet,
    WorkflowOutcome,
)
from qa_agent.run_context import RunContext
from qa_agent.setup_orchestration import (
    CleanupOutcome,
    SetupCleanupCoordinator,
    SetupRunOutcome,
    TestCaseRunOutcome,
)


@dataclass(frozen=True)
class PinnedWorkflowResult:
    """Primary workflow outcome with setup and cleanup reported separately."""

    outcome: WorkflowOutcome
    lifecycle: TestCaseRunOutcome = field(repr=False)
    execution: PinnedExecutionResult | None = field(default=None, repr=False)

    @property
    def run_context(self) -> RunContext:
        return self.lifecycle.run_context

    @property
    def setup(self) -> SetupRunOutcome:
        return self.lifecycle.setup

    @property
    def cleanup(self) -> CleanupOutcome:
        return self.lifecycle.cleanup

    @property
    def test_run(self) -> TestRun | None:
        return self.execution.test_run if self.execution is not None else None

    @property
    def product_started(self) -> bool:
        return self.lifecycle.product_started

    @property
    def product_error_type(self) -> str | None:
        return self.lifecycle.product_error_type

    @property
    def product_error(self) -> str | None:
        return self.lifecycle.product_error


class _PinnedTestCaseWorkflow:
    def __init__(
        self,
        executor: PinnedExecutionService,
        setup_cleanup: SetupCleanupCoordinator | None = None,
    ) -> None:
        self._executor = executor
        self._setup_cleanup = setup_cleanup or SetupCleanupCoordinator({})

    def run(
        self,
        test_case: TestCase,
        selected_versions: PlanVersionSet,
        run_context: RunContext | None = None,
    ) -> PinnedWorkflowResult:
        context = run_context if run_context is not None else RunContext()
        # Reject bad pins before provisioning resources. Resolution is exact and
        # never consults the store's current/latest version for execution.
        resolved = self._executor.resolve(test_case, selected_versions)
        execution_result: PinnedExecutionResult | None = None

        def execute_pinned(received_context: RunContext) -> PinnedExecutionResult:
            nonlocal execution_result
            execution_result = self._executor.execute(
                test_case, resolved, received_context
            )
            return execution_result

        lifecycle = self._setup_cleanup.run(test_case, context, execute_pinned)
        if not lifecycle.setup.succeeded:
            outcome = WorkflowOutcome.SETUP_FAILURE
        elif lifecycle.product_error_type is not None:
            outcome = WorkflowOutcome.INFRASTRUCTURE_ERROR
        elif execution_result is None:
            outcome = WorkflowOutcome.INFRASTRUCTURE_ERROR
        else:
            outcome = execution_result.outcome

        return PinnedWorkflowResult(
            outcome=outcome,
            lifecycle=lifecycle,
            execution=execution_result,
        )


class ValidationWorkflow(_PinnedTestCaseWorkflow):
    """Verify one exact plan-version set from a configured initial state."""


class RegressionWorkflow(_PinnedTestCaseWorkflow):
    """Repeat execution of selected versions without any learning behavior."""


class AutomationWorkflow:
    """Expose the existing self-healing pipeline as the learning workflow."""

    def __init__(self, pipeline: QATestPipeline) -> None:
        self._pipeline = pipeline

    def run(
        self,
        task: str,
        base_url: str | None = None,
        run_context: RunContext | None = None,
    ) -> PipelineResult:
        return self._pipeline.run(task, base_url, run_context)
