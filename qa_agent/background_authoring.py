"""Bounded background jobs for asynchronous AI TestCase authoring."""

from __future__ import annotations

import logging
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from uuid import UUID

from qa_agent.execution_progress import (
    AuthoringEventType,
    AuthoringProgressReporter,
    ExecutionProgressStore,
)
from qa_agent.test_case_authoring import (
    TestCaseAuthoringError,
    TestCaseAuthoringService,
    TestCaseDraftStore,
)


logger = logging.getLogger(__name__)

_SAFE_FAILURES = {
    "AI_PROVIDER_ERROR": (
        "AI providers could not complete the request. "
        "Review Developer details or use manual authoring."
    ),
    "AI_RATE_LIMIT": (
        "All configured AI providers are rate limited. Try again later or use manual authoring."
    ),
    "AI_TIMEOUT": (
        "AI providers timed out. Try again or use manual authoring."
    ),
    "AI_GENERATION_ERROR": "The AI could not generate a complete TestCase structure.",
    "AI_OUTPUT_VALIDATION_ERROR": (
        "The AI response could not be converted into a valid TestCase."
    ),
    "AUTHORING_EXECUTION_ERROR": "An unexpected authoring error occurred.",
}


class BackgroundAuthoringService:
    """Run bounded authoring jobs without sharing execution worker capacity."""

    def __init__(
        self,
        authoring_service: TestCaseAuthoringService,
        draft_store: TestCaseDraftStore,
        progress_store: ExecutionProgressStore,
        *,
        max_workers: int = 2,
        max_pending: int = 8,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if max_pending < 0:
            raise ValueError("max_pending cannot be negative.")
        self._authoring_service = authoring_service
        self._draft_store = draft_store
        self._progress_store = progress_store
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="qa-agent-authoring",
        )
        self._capacity = BoundedSemaphore(max_workers + max_pending)
        self._lock = Lock()
        self._closed = False

    def start(
        self,
        *,
        name: str,
        base_url: str,
        scenario: str,
        source_draft_token: str | None = None,
        source_draft_id: UUID | None = None,
    ) -> str:
        progress_id = self._progress_store.create_authoring(
            name=name,
            base_url=base_url,
            scenario=scenario,
            source_draft_token=source_draft_token,
            source_draft_id=source_draft_id,
        )
        reporter = AuthoringProgressReporter(self._progress_store, progress_id)
        reporter.emit(AuthoringEventType.AUTHORING_REQUESTED)
        reporter.emit(AuthoringEventType.INPUT_VALIDATED)
        if not self._capacity.acquire(blocking=False):
            reporter.finish_failure(
                "AUTHORING_EXECUTION_ERROR",
                "The authoring queue is busy. Try again shortly.",
            )
            return progress_id
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Background authoring service is closed.")
                self._executor.submit(
                    self._execute_and_release,
                    progress_id,
                    name,
                    base_url,
                    scenario,
                    source_draft_token,
                    source_draft_id,
                )
        except Exception as error:
            self._capacity.release()
            logger.error("Could not submit authoring job (%s)", type(error).__name__)
            reporter.finish_failure(
                "AUTHORING_EXECUTION_ERROR",
                _SAFE_FAILURES["AUTHORING_EXECUTION_ERROR"],
            )
        return progress_id

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _execute_and_release(
        self,
        progress_id: str,
        name: str,
        base_url: str,
        scenario: str,
        source_draft_token: str | None,
        source_draft_id: UUID | None,
    ) -> None:
        try:
            self._execute(
                progress_id,
                name,
                base_url,
                scenario,
                source_draft_token,
                source_draft_id,
            )
        finally:
            self._capacity.release()

    def _execute(
        self,
        progress_id: str,
        name: str,
        base_url: str,
        scenario: str,
        source_draft_token: str | None,
        source_draft_id: UUID | None,
    ) -> None:
        reporter = AuthoringProgressReporter(self._progress_store, progress_id)
        try:
            reporter.emit(AuthoringEventType.AUTHORING_STARTED)
            validated = self._authoring_service.validate_input(name, scenario, base_url)
            source_draft = (
                self._draft_store.get(source_draft_token)
                if source_draft_token is not None else None
            )
            draft = self._authoring_service.generate(
                validated.name,
                validated.scenario,
                validated.base_url,
                progress_callback=reporter.emit,
                provider_progress_callback=reporter.provider_progress,
                usage_workflow_id=progress_id,
            )
            prior_workflows = (
                source_draft.usage_workflow_ids
                if source_draft is not None else ()
            )
            draft = replace(
                draft,
                source_draft_id=source_draft_id,
                usage_workflow_ids=tuple(dict.fromkeys(
                    (*prior_workflows, *draft.usage_workflow_ids)
                )),
            )
            token = self._draft_store.put(draft)
            try:
                reporter.emit(AuthoringEventType.DRAFT_CREATED)
                reporter.finish_success(f"/test-cases/review/{token}")
            except Exception:
                self._draft_store.take(token)
                raise
            if source_draft_token is not None:
                self._draft_store.take(source_draft_token)
        except TestCaseAuthoringError as error:
            category = error.category
            if category == "INVALID_AUTHORING_INPUT":
                message = str(error)
            elif category in _SAFE_FAILURES:
                message = _SAFE_FAILURES[category]
            else:
                category = "AUTHORING_EXECUTION_ERROR"
                message = _SAFE_FAILURES[category]
            logger.warning("TestCase authoring ended (%s)", category)
            reporter.finish_failure(
                category,
                message,
                provider_failures=error.provider_failures,
            )
        except Exception as error:
            logger.error("Unexpected authoring failure (%s)", type(error).__name__)
            reporter.finish_failure(
                "AUTHORING_EXECUTION_ERROR",
                _SAFE_FAILURES["AUTHORING_EXECUTION_ERROR"],
            )
