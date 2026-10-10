"""Application service for running a persisted TestCase with saved plans."""

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from qa_agent.execution_control import is_cancelled
from qa_agent.browser_runner import BrowserRunner
from qa_agent.automation_lifecycle import (
    AutomationLifecycleService,
    AutomationStatus,
    plan_fingerprint_for_versions,
)
from qa_agent.expected_result_coverage import (
    has_sufficient_test_case_coverage,
    ExpectedResultCoverageError,
)
from qa_agent.execution_repository import ExecutionRepository
from qa_agent.execution_progress import get_active_execution_progress
from qa_agent.models import QATestPlan, TestCase
from qa_agent.pipeline import PipelineResult
from qa_agent.reliability import AutomationReviewRequired
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
from qa_agent.test_case_review import TestCaseReviewService, TestCaseReviewStatus
from qa_agent.test_plan_validation import validate_executable_plan
from qa_agent.workflows import (
    AutomationWorkflow,
    PinnedWorkflowResult,
    RegressionWorkflow,
    ValidationWorkflow,
)


logger = logging.getLogger(__name__)


class RunUnavailableError(ValueError):
    """A safe, user-facing reason a persisted TestCase cannot be run."""

    def __init__(self, message: str, *, category: str = "MISSING_AUTOMATION") -> None:
        self.category = category
        super().__init__(message)


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
        automation_lifecycle: AutomationLifecycleService | None = None,
        test_case_review: TestCaseReviewService | None = None,
        candidate_review_required: Callable[[UUID], bool] | None = None,
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
        self._automation_lifecycle = automation_lifecycle
        self._test_case_review = test_case_review
        self._candidate_review_required = candidate_review_required or (lambda _: False)

    def workflow_availability(self, test_case_id: UUID) -> WorkflowAvailability:
        test_case = self._test_cases.get(test_case_id)
        if test_case is None:
            raise RunUnavailableError("TestCase not found.")
        return self.workflow_availability_for_test_case(test_case)

    def workflow_availability_for_test_case(self, test_case: TestCase) -> WorkflowAvailability:
        """Assess a supplied case snapshot without reloading its definition."""
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
                try:
                    validate_executable_plan(version.qa_test_plan)
                except (TypeError, ValueError):
                    continue
                versions.append((step.id, version.version, version.id))
        complete = bool(steps) and len(versions) == len(steps)
        coverage_sufficient = (
            complete and has_sufficient_test_case_coverage(test_case, self._plan_store)
        )
        if self._automation_lifecycle is not None and self._automation_lifecycle.status(test_case) in {AutomationStatus.NEEDS_UPDATE, AutomationStatus.AUTOMATION_FAILED}:
            return WorkflowAvailability(automation_available=self._automation_workflow is not None,
                validation_available=False, regression_available=False,
                usable_plan_count=0, total_step_count=len(steps), plan_versions=tuple(versions),
                reason="The TestCase changed or generation needs attention. Generate updated automation before approving Validation.")
        has_saved_plan = any(
            self._plan_store.find(step.id) is not None for step in steps
        )
        ready_reason = (
            None
            if coverage_sufficient
            else (
                "Automation does not verify the TestStep expected result."
                if complete
                else (
                    "One or more saved plans are not usable."
                    if has_saved_plan
                    else "No complete automation version has been generated yet."
                )
            )
        )
        needs_candidate_review = any(self._requires_candidate_review(version_id) for _, _, version_id in versions)
        if needs_candidate_review and (
            self._test_case_review is None
            or self._test_case_review.record(test_case.id) is None
            or not self._test_case_review.validation_approved_for(test_case)
        ):
            return WorkflowAvailability(
                automation_available=self._automation_workflow is not None,
                validation_available=False, regression_available=False,
                usable_plan_count=len(versions), total_step_count=len(steps), plan_versions=tuple(versions),
                reason="Review repaired automation and explicitly Approve for Validation before execution.",
            )
        return WorkflowAvailability(
            automation_available=self._automation_workflow is not None,
            validation_available=complete and coverage_sufficient,
            regression_available=complete,
            usable_plan_count=len(versions),
            total_step_count=len(steps),
            plan_versions=tuple(versions),
            reason=ready_reason,
        )

    def run_pinned_regression(
        self,
        test_case: TestCase,
        selected_versions: PlanVersionSet,
    ) -> PinnedWorkflowResult:
        """Execute a supplied TestCase snapshot against only its exact saved pins."""
        approval_error = self.pinned_regression_approval_error(test_case, selected_versions)
        if approval_error:
            raise RunUnavailableError(approval_error, category="MISSING_AUTOMATION")
        runner = self._runner_factory(self._evidence_directory)
        plan_execution = PlanExecutionService(runner, self._execution_repository)
        executor = PinnedExecutionService(self._plan_store, plan_execution)
        workflow = RegressionWorkflow(
            executor,
            self._setup_cleanup_factory(),
            self._run_history,
        )
        try:
            return workflow.run(test_case, selected_versions, RunContext())
        except ExpectedResultCoverageError as error:
            raise RunUnavailableError(str(error), category="MISSING_AUTOMATION") from error
        except PlanSelectionError as error:
            raise RunUnavailableError(
                "A pinned saved plan is no longer available for this Suite Run.",
                category="MISSING_AUTOMATION",
            ) from error

    def pinned_regression_approval_error(
        self,
        test_case: TestCase,
        selected_versions: PlanVersionSet,
        *,
        check_current_automation: bool = False,
    ) -> str | None:
        """Check TestCase and plan approval against the exact versions a suite pins."""
        if self._test_case_review is not None and self._test_case_review.status(test_case.id) != TestCaseReviewStatus.APPROVED:
            return "Approve the TestCase before adding it to a Suite Run."
        # Eligibility uses today's lifecycle; queued runs and retries retain
        # their approved immutable pins even when the latest plans change.
        if check_current_automation and self._automation_lifecycle is not None and self._automation_lifecycle.status(test_case) in {
            AutomationStatus.NOT_AUTOMATED,
            AutomationStatus.NEEDS_UPDATE,
            AutomationStatus.AUTOMATION_FAILED,
        }:
            return "Generate or repair automation for the current TestCase before adding it to a Suite Run."
        needs_candidate_review = any(self._requires_candidate_review(item.test_plan_version_id) for item in selected_versions.selections)
        if needs_candidate_review and (
            self._test_case_review is None or self._test_case_review.record(test_case.id) is None
        ):
            return "Review repaired automation and explicitly Approve for Validation before running the suite."
        if self._test_case_review is None:
            return None
        record = self._test_case_review.record(test_case.id)
        if record is None:
            return None  # Legacy TestCases remain approved by default.
        selected = {
            item.test_step_id: item.test_plan_version_id
            for item in selected_versions.selections
        }
        fingerprint = plan_fingerprint_for_versions(test_case, selected)
        if record.approved_plan_fingerprint != fingerprint:
            return "Approve the current saved automation for Validation before running the suite."
        return None

    def _requires_candidate_review(self, version_id):
        version = self._plan_store.get_version(version_id)
        return bool(version and version.manual_recovery) or self._candidate_review_required(version_id)

    def run(
        self,
        test_case_id: UUID,
        workflow_type: WorkflowType,
    ) -> PinnedWorkflowResult | PipelineResult:
        progress = get_active_execution_progress()
        if workflow_type == WorkflowType.AUTOMATION:
            if self._automation_workflow is None:
                raise RunUnavailableError(
                    "Automation is not available for persisted TestCases yet.",
                    category="MISSING_AUTOMATION",
                )
            test_case = self._test_cases.get(test_case_id)
            if test_case is None:
                raise RunUnavailableError(
                    "TestCase not found.", category="INVALID_TESTCASE"
                )
            if (
                self._test_case_review is not None
                and self._test_case_review.status(test_case.id) != TestCaseReviewStatus.APPROVED
            ):
                raise RunUnavailableError(
                    "Approve the TestCase before generating automation.",
                    category="INVALID_TESTCASE",
                )
            run_context = progress.run_context if progress is not None else RunContext()
            if progress is not None:
                progress.bind_run_context(run_context)
                progress.test_case_loaded(test_case)
            try:
                regenerate = self._automation_lifecycle is not None and self._automation_lifecycle.status(test_case) in {AutomationStatus.NEEDS_UPDATE, AutomationStatus.AUTOMATION_FAILED}
                result = self._automation_workflow.run_test_case(test_case, run_context, **({"regenerate": True} if regenerate else {}))
            except Exception as error:
                if self._automation_lifecycle is not None and not is_cancelled(error):
                    self._lifecycle_update(
                        self._automation_lifecycle.mark_automation_completed if isinstance(error.__cause__, AutomationReviewRequired) else self._automation_lifecycle.mark_automation_failed, test_case
                    )
                raise
            if self._automation_lifecycle is not None and not result.test_run.cancelled:
                self._lifecycle_update(
                    self._automation_lifecycle.mark_automation_completed, test_case
                )
            return result
        if workflow_type not in {WorkflowType.VALIDATION, WorkflowType.REGRESSION}:
            raise RunUnavailableError(
                "Choose Automation, Validation, or Regression before starting a run.",
                category="EXECUTION_ERROR",
            )

        test_case = self._test_cases.get(test_case_id)
        if test_case is None:
            raise RunUnavailableError("TestCase not found.", category="INVALID_TESTCASE")
        if self._automation_lifecycle is not None and self._automation_lifecycle.status(test_case) in {AutomationStatus.NEEDS_UPDATE, AutomationStatus.AUTOMATION_FAILED}:
            raise RunUnavailableError("Generate updated automation before running the current TestCase.")
        review_record = (
            self._test_case_review.record(test_case.id)
            if self._test_case_review is not None else None
        )
        if (
            self._test_case_review is not None
            and self._test_case_review.status(test_case.id) != TestCaseReviewStatus.APPROVED
        ):
            raise RunUnavailableError(
                "Approve the TestCase before starting a run.", category="INVALID_TESTCASE"
            )
        if (
            review_record is not None
            and workflow_type in {WorkflowType.VALIDATION, WorkflowType.REGRESSION}
            and not self._test_case_review.validation_approved_for(test_case)
        ):
            raise RunUnavailableError(
                "Approve the current saved automation before starting this workflow.",
                category="MISSING_AUTOMATION",
            )
        run_context = progress.run_context if progress is not None else RunContext()
        if progress is not None:
            progress.bind_run_context(run_context)
            progress.test_case_loaded(test_case)

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
                "This TestCase is not ready to run because one or more steps have no saved plan.",
                category="MISSING_AUTOMATION",
            )
        availability = self.workflow_availability(test_case_id)
        workflow_available = (
            availability.validation_available
            if workflow_type == WorkflowType.VALIDATION
            else availability.regression_available
        )
        if not workflow_available:
            raise RunUnavailableError(
                availability.reason
                or "This TestCase is not ready to run because one or more saved plans are not usable.",
                category="MISSING_AUTOMATION",
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
        selected_plan_fingerprint = plan_fingerprint_for_versions(
            test_case,
            {selection.test_step_id: selection.test_plan_version_id for selection in selections},
        )
        if (
            review_record is not None
            and workflow_type in {WorkflowType.VALIDATION, WorkflowType.REGRESSION}
            and selected_plan_fingerprint != review_record.approved_plan_fingerprint
        ):
            raise RunUnavailableError(
                "Saved plans changed after approval. Review and approve the current automation again.",
                category="MISSING_AUTOMATION",
            )
        try:
            result = workflow.run(
                test_case,
                PlanVersionSet(tuple(selections)),
                run_context,
            )
            if workflow_type == WorkflowType.VALIDATION and self._automation_lifecycle is not None and not result.test_run.cancelled:
                self._lifecycle_update(
                    self._automation_lifecycle.mark_validation_completed,
                    test_case,
                    passed=result.outcome.value == "PASSED",
                    validated_plan_fingerprint=selected_plan_fingerprint,
                )
            return result
        except ExpectedResultCoverageError as error:
            raise RunUnavailableError(
                str(error), category="MISSING_AUTOMATION"
            ) from error
        except PlanSelectionError as error:
            raise RunUnavailableError(
                "Saved plans changed before this run started. Reload the TestCase and try again.",
                category="MISSING_AUTOMATION",
            ) from error

    @staticmethod
    def _lifecycle_update(method, *args, **kwargs) -> None:
        try:
            method(*args, **kwargs)
        except Exception as error:
            logger.error("Could not persist automation lifecycle status (%s)", type(error).__name__)
