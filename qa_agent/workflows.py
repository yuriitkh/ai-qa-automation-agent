"""Thin automation, validation, and regression workflow entry points."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from qa_agent.execution_control import (
    check_cancelled,
    completion_boundary,
)
from qa_agent.models import ExecutionStatus, TestCase, TestRun
from qa_agent.expected_result_coverage import (
    ExpectedResultCoverageError,
    expected_result_coverage,
)
from qa_agent.plan_execution import PlanExecutionClassification
from qa_agent.pipeline import PipelineResult, QATestPipeline
from qa_agent.pinned_execution import (
    PinnedExecutionResult,
    PinnedExecutionService,
    PlanVersionSet,
    ResolvedPlanVersionSet,
    WorkflowOutcome,
)
from qa_agent.run_context import RunContext
from qa_agent.run_history import RunHistoryService, WorkflowType
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
    test_run: TestRun
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
    def product_started(self) -> bool:
        return self.lifecycle.product_started

    @property
    def product_error_type(self) -> str | None:
        return self.lifecycle.product_error_type

    @property
    def product_error(self) -> str | None:
        return self.lifecycle.product_error


class _PinnedTestCaseWorkflow:
    workflow_type: WorkflowType

    def __init__(
        self,
        executor: PinnedExecutionService,
        setup_cleanup: SetupCleanupCoordinator | None = None,
        run_history: RunHistoryService | None = None,
    ) -> None:
        self._executor = executor
        self._setup_cleanup = setup_cleanup or SetupCleanupCoordinator({})
        self._run_history = run_history

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
        uncovered_step_ids = tuple(
            item.test_step.id
            for item in resolved.steps
            if not expected_result_coverage(
                item.test_step,
                item.plan_version.qa_test_plan,
            ).is_sufficient
        )
        coverage_sufficient = not uncovered_step_ids
        if self.workflow_type == WorkflowType.VALIDATION and not coverage_sufficient:
            raise ExpectedResultCoverageError()
        started_at = datetime.now(timezone.utc)
        execution_result: PinnedExecutionResult | None = None

        def execute_pinned(received_context: RunContext) -> PinnedExecutionResult:
            nonlocal execution_result
            check_cancelled()
            execution_result = self._executor.execute(
                test_case, resolved, received_context
            )
            return execution_result

        lifecycle = self._setup_cleanup.run(test_case, context, execute_pinned)
        finished_at = datetime.now(timezone.utc)
        if execution_result is not None and execution_result.cleanup_error is not None:
            from qa_agent.setup_orchestration import CleanupOutcome, CleanupFailure
            lifecycle = replace(lifecycle, cleanup=CleanupOutcome((*lifecycle.cleanup.failures, CleanupFailure("Browser cleanup", type(execution_result.cleanup_error).__name__, "Browser cleanup did not complete normally."))))
        if (
            self.workflow_type == WorkflowType.REGRESSION
            and uncovered_step_ids
            and execution_result is not None
            and execution_result.outcome == WorkflowOutcome.PASSED
        ):
            coverage_error = ExpectedResultCoverageError()
            uncovered_step_id = uncovered_step_ids[0]
            failed_execution = next(
                (
                    execution
                    for execution in execution_result.test_run.executions
                    if execution.test_step_id == uncovered_step_id
                    and execution.status == ExecutionStatus.PASSED
                ),
                None,
            )
            if failed_execution is not None:
                failed_execution = failed_execution.model_copy(update={
                    "status": ExecutionStatus.FAILED,
                    "error": str(coverage_error),
                    "runner_result": {
                        **(failed_execution.runner_result or {}),
                        "qa_classification": "AUTOMATION_EXECUTION_ERROR",
                    },
                })
                test_run = execution_result.test_run.model_copy(update={
                    "executions": [
                        failed_execution if execution.id == failed_execution.id else execution
                        for execution in execution_result.test_run.executions
                    ],
                })
                step_executions = tuple(
                    replace(
                        item,
                        classification=PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR,
                        execution=failed_execution,
                    )
                    if item.execution.id == failed_execution.id
                    else item
                    for item in execution_result.step_executions
                )
            else:
                test_run = execution_result.test_run
                step_executions = execution_result.step_executions
            execution_result = replace(
                execution_result,
                outcome=WorkflowOutcome.AUTOMATION_EXECUTION_ERROR,
                error=coverage_error,
                test_run=test_run,
                step_executions=step_executions,
            )
        if not lifecycle.setup.succeeded:
            outcome = WorkflowOutcome.SETUP_FAILURE
        elif lifecycle.product_error_type is not None:
            outcome = WorkflowOutcome.INFRASTRUCTURE_ERROR
        elif execution_result is None:
            outcome = WorkflowOutcome.INFRASTRUCTURE_ERROR
        else:
            outcome = execution_result.outcome

        test_run = (
            execution_result.test_run
            if execution_result is not None
            else TestRun.from_test_case(test_case, [], run_context=context)
        )
        with completion_boundary() as token:
            if token is not None and token.requested:
                outcome = WorkflowOutcome.CANCELLED
                finished_at = datetime.now(timezone.utc)
                test_run = TestRun.from_test_case(test_case, test_run.executions, blocked_step_ids=test_run.blocked_step_ids, run_context=context, cancelled=True).model_copy(update={"id": test_run.id, "started_at": started_at, "finished_at": finished_at})
                if execution_result is not None:
                    execution_result = replace(execution_result, outcome=outcome, test_run=test_run)
            result = PinnedWorkflowResult(
                outcome=outcome,
                test_run=test_run,
                lifecycle=lifecycle,
                execution=execution_result,
            )
            if self._run_history is not None:
                self._run_history.record_completed_run(
                    test_case,
                    result.test_run,
                    workflow_type=self.workflow_type,
                    outcome=result.outcome,
                    setup=result.setup,
                    cleanup=result.cleanup,
                    started_at=started_at,
                    finished_at=finished_at,
                )
            if token is not None:
                token.sealed = True
        return result


class ValidationWorkflow(_PinnedTestCaseWorkflow):
    """Verify one exact plan-version set from a configured initial state."""

    workflow_type = WorkflowType.VALIDATION


class RegressionWorkflow(_PinnedTestCaseWorkflow):
    """Repeat execution of selected versions without any learning behavior."""

    workflow_type = WorkflowType.REGRESSION


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

    def run_test_case(
        self,
        test_case: TestCase,
        run_context: RunContext | None = None,
    ) -> PipelineResult:
        return self._pipeline.run_test_case(test_case, run_context)
