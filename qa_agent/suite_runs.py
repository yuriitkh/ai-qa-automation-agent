"""Durable, sequential execution of ordered Test Suite members."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from threading import BoundedSemaphore, Lock
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.execution_control import (
    CancellationToken,
    current_cancellation,
    cancellation_scope,
    check_cancelled,
    is_cancelled,
    OperationCancelled,
)
from qa_agent.cookie_consent import (
    DEFAULT_COOKIE_CONSENT_POLICY,
    CookieConsentPolicy,
    cookie_consent_scope,
)
from qa_agent.evidence_policy import DEFAULT_EVIDENCE_POLICY, EvidencePolicy, evidence_policy_scope
from qa_agent.models import TestCase
from qa_agent.pinned_execution import PlanVersionSet, StepPlanSelection
from qa_agent.run_history import RunHistoryService, WorkflowType
from qa_agent.result_semantics import result_outcome, result_label
from qa_agent.test_case_execution import TestCaseExecutionService
from qa_agent.test_case_repository import TestCaseRepository
from qa_agent.test_suites import TestSuite, TestSuiteService


logger = logging.getLogger(__name__)


class AIPolicy(str, Enum):
    DISABLED = "DISABLED"
    ALLOWED = "ALLOWED"


class SuiteRunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_FAILURES = "COMPLETED_WITH_FAILURES"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"


class SuiteRunItemStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    PASSED = "PASSED"
    PASSED_AFTER_RETRY = "PASSED_AFTER_RETRY"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    NOT_RUN = "NOT_RUN"
    INTERRUPTED = "INTERRUPTED"


class SuiteRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    workflow_type: WorkflowType = WorkflowType.REGRESSION
    execution_type: Literal["SEQUENTIAL"] = "SEQUENTIAL"
    ai_policy: AIPolicy = AIPolicy.DISABLED
    retry_count: int = Field(default=0, ge=0, le=2)
    evidence_policy: EvidencePolicy = DEFAULT_EVIDENCE_POLICY
    cookie_policy: CookieConsentPolicy = DEFAULT_COOKIE_CONSENT_POLICY


class PinnedSuitePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_step_id: UUID
    test_plan_version_id: UUID
    version_number: int


class SuiteRunAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_number: int
    run_id: UUID | None = None
    run_public_id: str | None = None
    run_status: str
    outcome: str | None = None
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    failure_classifications: list[str] = Field(default_factory=list)
    error_category: str | None = None


class SuiteRunItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_index: int
    test_case_id: UUID
    test_case_public_id: str | None = None
    test_case_name: str
    status: SuiteRunItemStatus = SuiteRunItemStatus.QUEUED
    classification: str | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    pinned_plans: list[PinnedSuitePlan] = Field(default_factory=list)
    test_case_snapshot: dict[str, Any] = Field(default_factory=dict, repr=False)
    attempts: list[SuiteRunAttempt] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error_category: str | None = None


class SuiteRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    public_id: str | None = None
    suite_id: UUID
    suite_name: str
    config: SuiteRunConfig
    status: SuiteRunStatus = SuiteRunStatus.QUEUED
    items: list[SuiteRunItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_category: str | None = None

    @property
    def flaky_count(self) -> int:
        return sum(item.status == SuiteRunItemStatus.PASSED_AFTER_RETRY and suite_item_outcome(item) == "PASSED" for item in self.items)

    @property
    def passed_count(self) -> int:
        return self.outcome_counts["passed"]

    @property
    def failed_count(self) -> int:
        return self.outcome_counts["product_failures"]

    @property
    def outcome_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(("passed", "product_failures", "automation_errors", "generation_errors", "infrastructure_errors", "blocked", "inconclusive", "pending", "cancelled", "not_attempted"), 0)
        categories = {
            "PASSED": "passed", "PRODUCT_FAILURE": "product_failures",
            "AUTOMATION_EXECUTION_ERROR": "automation_errors", "AUTOMATION_DRIFT": "automation_errors",
            "AUTOMATION_GENERATION_ERROR": "generation_errors", "INFRASTRUCTURE_ERROR": "infrastructure_errors",
            "BLOCKED": "blocked", "INCONCLUSIVE": "inconclusive", "CANCELLED": "cancelled", "NOT_ATTEMPTED": "not_attempted",
        }
        for item in self.items:
            counts[categories.get(suite_item_outcome(item), "pending")] += 1
        return counts


class SuiteRunRepository:
    """Repository contract implemented by the in-memory and SQLite stores."""

    def save(self, run: SuiteRun) -> SuiteRun:
        raise NotImplementedError

    def get(self, run_id: UUID) -> SuiteRun | None:
        raise NotImplementedError

    def get_by_public_id(self, public_id: str) -> SuiteRun | None:
        raise NotImplementedError

    def list_recent(self, limit: int = 50) -> list[SuiteRun]:
        raise NotImplementedError

    def interrupt_incomplete(self) -> int:
        raise NotImplementedError


class InMemorySuiteRunRepository(SuiteRunRepository):
    def __init__(self) -> None:
        self._runs: dict[UUID, SuiteRun] = {}
        self._next_id = 1
        self._lock = Lock()

    def save(self, run: SuiteRun) -> SuiteRun:
        with self._lock:
            existing = self._runs.get(run.id)
            stored = run
            if existing is None and run.public_id is None:
                stored = run.model_copy(update={"public_id": f"SUITE-RUN-{self._next_id:06d}"})
                self._next_id += 1
            elif run.public_id is None:
                stored = run.model_copy(update={"public_id": existing.public_id})
            self._runs[run.id] = stored
            return stored

    def get(self, run_id: UUID) -> SuiteRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def get_by_public_id(self, public_id: str) -> SuiteRun | None:
        with self._lock:
            return next((run for run in self._runs.values() if run.public_id == public_id), None)

    def list_recent(self, limit: int = 50) -> list[SuiteRun]:
        with self._lock:
            return list(reversed(list(self._runs.values())[-max(0, limit):]))

    def interrupt_incomplete(self) -> int:
        with self._lock:
            return self._interrupt(lambda run: self._runs.__setitem__(run.id, run))

    def _interrupt(self, save) -> int:
        changed = 0
        now = datetime.now(timezone.utc)
        for run in list(self._runs.values()):
            if run.status == SuiteRunStatus.CANCELLATION_REQUESTED:
                save(recover_cancelled_suite(run))
                changed += 1
                continue
            if run.status not in {SuiteRunStatus.QUEUED, SuiteRunStatus.RUNNING}:
                continue
            updated_items = []
            for item in run.items:
                if item.status == SuiteRunItemStatus.RUNNING:
                    updated_items.append(item.model_copy(update={"status": SuiteRunItemStatus.INTERRUPTED, "finished_at": now}))
                elif item.status in {SuiteRunItemStatus.QUEUED, SuiteRunItemStatus.RETRYING}:
                    updated_items.append(item.model_copy(update={"status": SuiteRunItemStatus.NOT_RUN}))
                else:
                    updated_items.append(item)
            save(run.model_copy(update={"status": SuiteRunStatus.INTERRUPTED, "finished_at": now, "items": updated_items}))
            changed += 1
        return changed


class SuiteEligibility(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_index: int
    test_case_id: UUID
    test_case_public_id: str | None = None
    test_case_name: str
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    pinned_plans: list[PinnedSuitePlan] = Field(default_factory=list)
    test_case_snapshot: dict[str, Any] = Field(default_factory=dict, repr=False)


class SuiteStartResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    suite: TestSuite | None
    run: SuiteRun | None = None
    eligibility: list[SuiteEligibility] = Field(default_factory=list)
    suite_error: str | None = None

    @property
    def can_start(self) -> bool:
        return self.suite is not None and bool(self.eligibility) and all(item.eligible for item in self.eligibility)


class SuiteRunService:
    """Preflight and durably execute a saved-plan Regression Suite Run."""

    def __init__(
        self,
        suites: TestSuiteService,
        test_cases: TestCaseRepository,
        execution: TestCaseExecutionService,
        history: RunHistoryService,
        repository: SuiteRunRepository,
        *,
        max_workers: int = 2,
        max_pending: int = 4,
    ) -> None:
        if max_workers < 1 or max_pending < 0:
            raise ValueError("Suite run worker limits are invalid.")
        self._suites = suites
        self._test_cases = test_cases
        self._execution = execution
        self._history = history
        self._repository = repository
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qa-suite-run")
        self._capacity = BoundedSemaphore(max_workers + max_pending)
        self._max_capacity = max_workers + max_pending
        self._active_jobs = 0
        self._lock = Lock()
        self._closed = False
        self._controls: dict[UUID, CancellationToken] = {}
        self._repository.interrupt_incomplete()

    def readiness_state(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "closed": self._closed,
                "active": self._active_jobs,
                "capacity": self._max_capacity,
                "available": not self._closed and self._active_jobs < self._max_capacity,
            }

    @property
    def repository(self) -> SuiteRunRepository:
        return self._repository

    def preview(self, suite_id: UUID) -> SuiteStartResult:
        suite = self._suites.get(suite_id)
        if suite is None:
            return SuiteStartResult(suite=None, suite_error="Test Suite not found.")
        member_ids = self._suites.all_member_ids(suite_id)
        if not member_ids:
            return SuiteStartResult(suite=suite, suite_error="Add at least one TestCase before starting a Suite Run.")
        entries: list[SuiteEligibility] = []
        for index, case_id in enumerate(member_ids):
            case = self._test_cases.get(case_id)
            if case is None:
                entries.append(SuiteEligibility(
                    order_index=index,
                    test_case_id=case_id,
                    test_case_name="Missing TestCase",
                    eligible=False,
                    reasons=["This suite member no longer exists. Remove it from the suite before running."],
                ))
                continue
            availability = self._execution.workflow_availability_for_test_case(case)
            pins = [
                PinnedSuitePlan(test_step_id=step_id, test_plan_version_id=version_id, version_number=version_number)
                for step_id, version_number, version_id in availability.plan_versions
            ]
            ready = availability.regression_available and len(pins) == len(case.steps)
            reasons = [] if ready else [availability.reason or "A complete saved Regression plan is not available."]
            if ready:
                selected = PlanVersionSet(tuple(
                    StepPlanSelection(pin.test_step_id, pin.test_plan_version_id)
                    for pin in pins
                ))
                approval_error = self._execution.pinned_regression_approval_error(
                    case, selected, check_current_automation=True,
                )
                if approval_error:
                    ready = False
                    reasons.append(approval_error)
            entries.append(SuiteEligibility(
                order_index=index,
                test_case_id=case.id,
                test_case_public_id=case.public_id,
                test_case_name=case.name,
                eligible=ready,
                reasons=reasons,
                pinned_plans=pins,
                test_case_snapshot=case.model_dump(mode="json"),
            ))
        return SuiteStartResult(suite=suite, eligibility=entries)

    def start(
        self,
        suite_id: UUID,
        *,
        ai_policy: AIPolicy = AIPolicy.DISABLED,
        retry_count: int = 0,
        evidence_policy: EvidencePolicy = DEFAULT_EVIDENCE_POLICY,
        cookie_policy: CookieConsentPolicy = DEFAULT_COOKIE_CONSENT_POLICY,
    ) -> SuiteStartResult:
        preview = self.preview(suite_id)
        if preview.suite is None or preview.suite_error or not preview.can_start:
            return preview
        config = SuiteRunConfig(
            ai_policy=ai_policy,
            retry_count=retry_count,
            evidence_policy=evidence_policy,
            cookie_policy=cookie_policy,
        )
        items = [
            SuiteRunItem(
                order_index=item.order_index,
                test_case_id=item.test_case_id,
                test_case_public_id=item.test_case_public_id,
                test_case_name=item.test_case_name,
                pinned_plans=item.pinned_plans,
                test_case_snapshot=item.test_case_snapshot,
            )
            for item in preview.eligibility
        ]
        run = self._repository.save(SuiteRun(
            suite_id=suite_id,
            suite_name=preview.suite.name,
            config=config,
            items=items,
        ))
        if not self._capacity.acquire(blocking=False):
            failed = run.model_copy(update={
                "status": SuiteRunStatus.FAILED,
                "finished_at": datetime.now(timezone.utc),
                "error_category": "QUEUE_FULL",
            })
            self._repository.save(failed)
            return preview.model_copy(update={"run": failed})
        reserved = False
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Suite run service is closed.")
                self._active_jobs += 1
                reserved = True
                self._controls[run.id] = CancellationToken()
                self._executor.submit(self._execute_with_release, run.id)
        except Exception:
            if reserved:
                with self._lock:
                    self._active_jobs -= 1
                    self._controls.pop(run.id, None)
            self._capacity.release()
            logger.exception("Could not submit local Suite Run job")
            failed = run.model_copy(update={
                "status": SuiteRunStatus.FAILED,
                "finished_at": datetime.now(timezone.utc),
                "error_category": "QUEUE_ERROR",
            })
            self._repository.save(failed)
            return preview.model_copy(update={"run": failed})
        return preview.model_copy(update={"run": run})

    def get(self, public_id: str) -> SuiteRun | None:
        return self._repository.get_by_public_id(public_id)

    def cancel(self, public_id: str) -> bool:
        run = self.get(public_id)
        if run is None:
            return False
        with self._lock:
            token = self._controls.get(run.id)
        if token is None:
            return False
        with token.lock:
            current = self._repository.get(run.id)
            if current is None or current.status not in {
                SuiteRunStatus.QUEUED, SuiteRunStatus.RUNNING,
                SuiteRunStatus.CANCELLATION_REQUESTED,
            }:
                return False
            return token.request(lambda: self._repository.save(
                current.model_copy(update={"status": SuiteRunStatus.CANCELLATION_REQUESTED})
            ))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            controls = list(self._controls.values())
        for token in controls:
            token.request()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _execute_with_release(self, run_id: UUID) -> None:
        token = self._controls[run_id]
        try:
            with cancellation_scope(token):
                self._execute(run_id)
        finally:
            with token.lock:
                token.sealed = True
            with self._lock:
                self._active_jobs -= 1
                self._controls.pop(run_id, None)
            self._capacity.release()

    def _execute(self, run_id: UUID) -> None:
        run = self._repository.get(run_id)
        if run is None:
            return
        now = datetime.now(timezone.utc)
        run = self._save(run.model_copy(update={"status": SuiteRunStatus.RUNNING, "started_at": now}))
        try:
            for index, item in enumerate(run.items):
                check_cancelled()
                run = self._run_item(run, index, item)
            final_status = (
                SuiteRunStatus.COMPLETED_WITH_FAILURES
                if any(item.status in {SuiteRunItemStatus.FAILED, SuiteRunItemStatus.INTERRUPTED} for item in run.items)
                else SuiteRunStatus.COMPLETED
            )
            token = current_cancellation()
            with token.lock:
                if token.requested:
                    raise OperationCancelled()
                self._save(run.model_copy(update={"status": final_status, "finished_at": datetime.now(timezone.utc)}))
                token.sealed = True
        except Exception as error:
            logger.error("Suite Run stopped safely (%s)", type(error).__name__)
            token = current_cancellation()
            with token.lock:
                current = self._repository.get(run_id) or run
                now = datetime.now(timezone.utc)
                cancelled = is_cancelled(error) or current_cancellation().requested
                items = [
                    item.model_copy(update={"status": SuiteRunItemStatus.NOT_ATTEMPTED if cancelled else SuiteRunItemStatus.NOT_RUN})
                    if item.status in {SuiteRunItemStatus.QUEUED, SuiteRunItemStatus.RETRYING}
                    else item.model_copy(update={"status": SuiteRunItemStatus.CANCELLED}) if cancelled and item.status == SuiteRunItemStatus.RUNNING
                    else item
                    for item in current.items
                ]
                self._save(current.model_copy(update={
                    "items": items,
                    "status": SuiteRunStatus.CANCELLED if is_cancelled(error) or current_cancellation().requested else SuiteRunStatus.FAILED,
                    "finished_at": now,
                    "error_category": "CANCELLED" if is_cancelled(error) or current_cancellation().requested else type(error).__name__,
                }))
                token.sealed = True

    def _run_item(self, run: SuiteRun, index: int, item: SuiteRunItem) -> SuiteRun:
        started_at = datetime.now(timezone.utc)
        item = item.model_copy(update={"status": SuiteRunItemStatus.RUNNING, "started_at": started_at})
        run = self._replace_item(run, index, item)
        attempts = list(item.attempts)
        max_attempts = run.config.retry_count + 1
        succeeded = False
        for attempt_number in range(1, max_attempts + 1):
            check_cancelled()
            attempt_started = datetime.now(timezone.utc)
            run_id: UUID | None = None
            public_id: str | None = None
            status = "FAILED"
            outcome = "EXECUTION_ERROR"
            classifications: list[str] = []
            error_category: str | None = None
            try:
                test_case = TestCase.model_validate(item.test_case_snapshot)
                selected = PlanVersionSet(tuple(
                    StepPlanSelection(plan.test_step_id, plan.test_plan_version_id)
                    for plan in item.pinned_plans
                ))
                with cookie_consent_scope(run.config.cookie_policy):
                    with evidence_policy_scope(run.config.evidence_policy):
                        with cancellation_scope(CancellationToken(parent=current_cancellation())):
                            check_cancelled()
                            result = self._execution.run_pinned_regression(test_case, selected)
                run_id = result.test_run.id
                record = self._history.get(run_id)
                if record is None:
                    raise RuntimeError("RUN_HISTORY_MISSING")
                public_id = record.public_id
                status = record.status.value
                outcome = record.outcome
                classifications = sorted({
                    step.classification.value
                    for step in (result.execution.step_executions if result.execution else ())
                })
                succeeded = status == "PASSED" and outcome == "PASSED"
            except Exception as error:
                error_category = "CANCELLED" if is_cancelled(error) else type(error).__name__
                if is_cancelled(error):
                    status = outcome = "CANCELLED"
                logger.warning("Suite item attempt failed safely (%s)", error_category)
                succeeded = False
            finished = datetime.now(timezone.utc)
            attempts.append(SuiteRunAttempt(
                attempt_number=attempt_number,
                run_id=run_id,
                run_public_id=public_id,
                run_status=status,
                outcome=outcome,
                started_at=attempt_started,
                finished_at=finished,
                duration_ms=max(0, int((finished - attempt_started).total_seconds() * 1000)),
                failure_classifications=classifications,
                error_category=error_category,
            ))
            if succeeded or current_cancellation().requested:
                break
            if attempt_number < max_attempts:
                item = item.model_copy(update={"status": SuiteRunItemStatus.RETRYING, "attempts": attempts})
                run = self._replace_item(run, index, item)
        item_finished_at = datetime.now(timezone.utc)
        last_attempt = attempts[-1]
        final = item.model_copy(update={
            "status": (
                SuiteRunItemStatus.PASSED_AFTER_RETRY
                if succeeded and len(attempts) > 1
                else SuiteRunItemStatus.PASSED
                if succeeded
                else SuiteRunItemStatus.CANCELLED if last_attempt.outcome == "CANCELLED"
                else SuiteRunItemStatus.FAILED
            ),
            "attempts": attempts,
            "classification": last_attempt.outcome,
            "finished_at": item_finished_at,
            "duration_ms": max(0, int((item_finished_at - started_at).total_seconds() * 1000)),
            "error_category": None if succeeded else (attempts[-1].error_category or attempts[-1].outcome),
        })
        return self._replace_item(run, index, final)

    def _replace_item(self, run: SuiteRun, index: int, item: SuiteRunItem) -> SuiteRun:
        items = list(run.items)
        items[index] = item
        return self._save(run.model_copy(update={"items": items}))

    def _save(self, run: SuiteRun) -> SuiteRun:
        token = current_cancellation()
        if token is None:
            return self._repository.save(run)
        with token.lock:
            if token.requested and run.status in {SuiteRunStatus.QUEUED, SuiteRunStatus.RUNNING}:
                run = run.model_copy(update={"status": SuiteRunStatus.CANCELLATION_REQUESTED})
            return self._repository.save(run)


def suite_run_report_json(run: SuiteRun) -> str:
    """Return a stable JSON representation of the persisted run snapshot."""
    payload = json.loads(run.model_dump_json())
    payload["outcome_counts"] = run.outcome_counts
    payload["display_status"] = suite_status_label(run)
    for item, source in zip(payload.get("items", []), run.items):
        item.pop("test_case_snapshot", None)
        item["display_outcome"] = suite_item_outcome(source)
        item["display_status"] = suite_item_label(source)
        for attempt, recorded_attempt in zip(item["attempts"], source.attempts):
            attempt["display_status"] = result_label(suite_attempt_outcome(recorded_attempt))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def suite_attempt_outcome(attempt: SuiteRunAttempt) -> str:
    return result_outcome(
        attempt.run_status, attempt.outcome,
        complete=attempt.run_id is not None and attempt.error_category is None,
    )


def suite_item_outcome(item: SuiteRunItem) -> str:
    if item.status in {SuiteRunItemStatus.QUEUED, SuiteRunItemStatus.RUNNING, SuiteRunItemStatus.RETRYING}:
        return item.status.value
    if item.status == SuiteRunItemStatus.NOT_ATTEMPTED:
        return "NOT_ATTEMPTED"
    if item.status == SuiteRunItemStatus.BLOCKED:
        return "BLOCKED"
    if item.status in {SuiteRunItemStatus.NOT_RUN, SuiteRunItemStatus.INTERRUPTED}:
        return "INCONCLUSIVE"
    attempt = item.attempts[-1] if item.attempts else None
    if attempt is None:
        return result_outcome(item.status, item.classification, complete=False)
    return result_outcome(
        attempt.run_status, attempt.outcome,
        complete=(
            item.status in {SuiteRunItemStatus.PASSED, SuiteRunItemStatus.PASSED_AFTER_RETRY}
            and attempt.run_id is not None and attempt.error_category is None
        ),
    )


def suite_status_label(run: SuiteRun) -> str:
    if run.status == SuiteRunStatus.CANCELLATION_REQUESTED:
        return "Stopping…"
    if run.status == SuiteRunStatus.CANCELLED:
        return "Stopped by user"
    if run.status in {SuiteRunStatus.QUEUED, SuiteRunStatus.RUNNING}:
        return run.status.value.title()
    if run.status in {SuiteRunStatus.INTERRUPTED, SuiteRunStatus.FAILED}:
        return "Stopped — Inconclusive" if run.status == SuiteRunStatus.INTERRUPTED else "Stopped — Infrastructure Error"
    return "Completed — Passed" if run.items and run.passed_count == len(run.items) else "Completed with issues"


def suite_item_label(item: SuiteRunItem) -> str:
    outcome = suite_item_outcome(item)
    return "Passed after retry" if outcome == "PASSED" and item.status == SuiteRunItemStatus.PASSED_AFTER_RETRY else result_label(outcome)


def recover_cancelled_suite(run: SuiteRun) -> SuiteRun:
    """No completion timestamp is invented for work lost at process exit."""
    items = [item.model_copy(update={"status": SuiteRunItemStatus.CANCELLED if item.status == SuiteRunItemStatus.RUNNING else SuiteRunItemStatus.NOT_ATTEMPTED}) if item.status in {SuiteRunItemStatus.RUNNING, SuiteRunItemStatus.QUEUED, SuiteRunItemStatus.RETRYING} else item for item in run.items]
    return run.model_copy(update={"status": SuiteRunStatus.CANCELLED, "items": items, "error_category": "CANCELLED"})
