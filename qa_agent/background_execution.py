"""Background orchestration for Web-triggered TestCase workflow runs."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from uuid import UUID

from qa_agent.execution_progress import (
    ExecutionEventType,
    ExecutionProgressReporter,
    ExecutionProgressStore,
    active_execution_progress,
)
from qa_agent.pipeline import PipelineStageError
from qa_agent.run_history import RunHistoryService, WorkflowType
from qa_agent.run_context import RunContext
from qa_agent.test_case_execution import RunUnavailableError, TestCaseExecutionService


logger = logging.getLogger(__name__)


class BackgroundRunService:
    """Submit one local workflow run and retain its live progress snapshot."""

    def __init__(
        self,
        run_service: TestCaseExecutionService,
        run_history: RunHistoryService,
        progress_store: ExecutionProgressStore | None = None,
        *,
        max_workers: int = 4,
        max_pending: int = 16,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if max_pending < 0:
            raise ValueError("max_pending cannot be negative.")
        self._run_service = run_service
        self._run_history = run_history
        self.progress_store = progress_store or ExecutionProgressStore()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="qa-agent-run",
        )
        self._capacity = BoundedSemaphore(max_workers + max_pending)
        self._lock = Lock()
        self._closed = False

    def start(self, test_case_id: UUID, workflow_type: WorkflowType) -> str:
        progress_id = self.progress_store.create(test_case_id, workflow_type)
        reporter = ExecutionProgressReporter(self.progress_store, progress_id)
        reporter.emit(
            ExecutionEventType.RUN_REQUESTED,
            message="Run requested.",
        )
        if not self._capacity.acquire(blocking=False):
            reporter.finish(
                outcome="EXECUTION_ERROR",
                error_category="EXECUTION_ERROR",
                message="The local run queue is busy. Try again shortly.",
            )
            return progress_id
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Background run service is closed.")
                self._executor.submit(
                    self._execute_with_release,
                    progress_id,
                    test_case_id,
                    workflow_type,
                )
        except Exception:
            self._capacity.release()
            logger.exception("Could not submit TestCase execution job")
            reporter.finish(
                outcome="EXECUTION_ERROR",
                error_category="EXECUTION_ERROR",
                message="An unexpected execution error occurred.",
            )
        return progress_id

    def _execute_with_release(
        self,
        progress_id: str,
        test_case_id: UUID,
        workflow_type: WorkflowType,
    ) -> None:
        try:
            self._execute(progress_id, test_case_id, workflow_type)
        finally:
            self._capacity.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _execute(
        self,
        progress_id: str,
        test_case_id: UUID,
        workflow_type: WorkflowType,
    ) -> None:
        reporter = ExecutionProgressReporter(
            self.progress_store,
            progress_id,
            RunContext(),
        )
        with active_execution_progress(reporter):
            reporter.emit(
                ExecutionEventType.RUN_STARTED,
                message="Run started.",
            )
            try:
                result = self._run_service.run(test_case_id, workflow_type)
            except RunUnavailableError as error:
                category = getattr(error, "category", "MISSING_AUTOMATION")
                logger.info("TestCase run was not available (%s)", category)
                reporter.finish(
                    outcome=category,
                    error_category=category,
                    message=None,
                )
                return
            except PipelineStageError as error:
                category = _pipeline_error_category(error)
                logger.exception("TestCase pipeline stopped during %s", error.stage)
                reporter.finish(
                    outcome=category,
                    error_category=category,
                    message=None,
                )
                return
            except Exception:
                logger.exception("Unexpected TestCase workflow failure")
                reporter.finish(
                    outcome="EXECUTION_ERROR",
                    error_category="EXECUTION_ERROR",
                    message="An unexpected execution error occurred.",
                )
                return

            try:
                test_run = getattr(result, "test_run", None)
                run_id = getattr(test_run, "id", None)
                record = self._run_history.get(run_id) if isinstance(run_id, UUID) else None
                run_status = _enum_value(
                    record.status if record is not None else getattr(test_run, "status", None)
                )
                outcome = _enum_value(
                    record.outcome
                    if record is not None
                    else getattr(result, "outcome", None)
                )
                if outcome is None:
                    outcome = "PASSED" if run_status == "PASSED" else "FAILED"
                reporter.finish(
                    run_id=run_id if isinstance(run_id, UUID) else None,
                    run_status=run_status,
                    outcome=outcome,
                    duration_ms=record.duration_ms if record is not None else None,
                    error_category=(
                        outcome
                        if outcome in {
                            "PRODUCT_FAILURE",
                            "AUTOMATION_DRIFT",
                            "INFRASTRUCTURE_ERROR",
                            "SETUP_FAILURE",
                        }
                        else None
                    ),
                )
            except Exception:
                logger.exception("Could not summarize completed TestCase workflow")
                reporter.finish(
                    outcome="EXECUTION_ERROR",
                    error_category="EXECUTION_ERROR",
                    message="An unexpected execution error occurred.",
                )


def _pipeline_error_category(error: PipelineStageError) -> str:
    stage = error.stage.casefold()
    if "plan generation" in stage or "regeneration" in stage:
        return "AUTOMATION_GENERATION_ERROR"
    if "plan lookup" in stage:
        return "MISSING_AUTOMATION"
    if "decomposition" in stage or "target resolution" in stage:
        return "INVALID_TESTCASE"
    return "INFRASTRUCTURE_ERROR"


def _enum_value(value) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)
