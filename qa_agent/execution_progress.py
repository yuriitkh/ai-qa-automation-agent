"""Thread-safe in-memory execution lifecycle events for local Web runs."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from secrets import token_urlsafe
from threading import RLock
from typing import Any, Iterator
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.models import TestCase, TestStep
from qa_agent.presentation import failure_message
from qa_agent.redaction import redact_secrets
from qa_agent.run_context import RunContext


class ExecutionEventType(str, Enum):
    RUN_REQUESTED = "RUN_REQUESTED"
    RUN_STARTED = "RUN_STARTED"
    TESTCASE_LOADED = "TESTCASE_LOADED"
    SETUP_STARTED = "SETUP_STARTED"
    SETUP_SUCCEEDED = "SETUP_SUCCEEDED"
    SETUP_FAILED = "SETUP_FAILED"
    AUTOMATION_PREPARATION_STARTED = "AUTOMATION_PREPARATION_STARTED"
    PLAN_REUSED = "PLAN_REUSED"
    PLAN_GENERATION_STARTED = "PLAN_GENERATION_STARTED"
    PLAN_GENERATED = "PLAN_GENERATED"
    PLAN_GENERATION_FAILED = "PLAN_GENERATION_FAILED"
    STEP_STARTED = "STEP_STARTED"
    STEP_PASSED = "STEP_PASSED"
    STEP_FAILED = "STEP_FAILED"
    STEP_BLOCKED = "STEP_BLOCKED"
    EVIDENCE_CAPTURED = "EVIDENCE_CAPTURED"
    CLEANUP_STARTED = "CLEANUP_STARTED"
    CLEANUP_SUCCEEDED = "CLEANUP_SUCCEEDED"
    CLEANUP_FAILED = "CLEANUP_FAILED"
    RUN_FINISHED = "RUN_FINISHED"


class ProgressState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class ProgressStepState(str, Enum):
    PENDING = "PENDING"
    PREPARING_AUTOMATION = "PREPARING_AUTOMATION"
    READY = "READY"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ProgressStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    order: int
    name: str
    state: ProgressStepState = ProgressStepState.PENDING
    automation_state: str | None = None


class ExecutionProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    event_type: ExecutionEventType
    timestamp: datetime
    test_case_id: UUID
    workflow_type: str
    step_id: UUID | None = None
    step_order: int | None = None
    step_name: str | None = None
    status: str | None = None
    classification: str | None = None
    message: str | None = None
    run_id: UUID | None = None
    run_status: str | None = None
    outcome: str | None = None
    error_category: str | None = None
    duration_ms: int | None = None
    evidence_execution_id: UUID | None = None
    evidence_index: int | None = None
    evidence_name: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize only presentation-safe event fields."""
        return {
            "id": str(self.id),
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "test_case_id": str(self.test_case_id),
            "workflow": self.workflow_type,
            "step_id": str(self.step_id) if self.step_id else None,
            "step_order": self.step_order,
            "step_name": self.step_name,
            "status": self.status,
            "classification": self.classification,
            "message": self.message,
            "run_id": str(self.run_id) if self.run_id else None,
            "run_status": self.run_status,
            "outcome": self.outcome,
            "error_category": self.error_category,
            "duration_ms": self.duration_ms,
            "evidence": ({
                "execution_id": str(self.evidence_execution_id),
                "index": self.evidence_index,
                "name": self.evidence_name,
            } if self.evidence_execution_id is not None else None),
        }


class ExecutionProgressSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    progress_id: str
    test_case_id: UUID
    test_case_name: str | None = None
    workflow_type: str
    state: ProgressState
    phase: str
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    steps: tuple[ProgressStep, ...] = ()
    events: tuple[ExecutionProgressEvent, ...] = ()
    final_run_id: UUID | None = None
    run_status: str | None = None
    outcome: str | None = None
    error_category: str | None = None
    error_message: str | None = None
    elapsed_ms: int = 0

    @property
    def final_run_url(self) -> str | None:
        return f"/runs/{self.final_run_id}" if self.final_run_id else None

    def to_public_dict(self) -> dict[str, Any]:
        """Return an allowlisted JSON view; internal state and paths stay private."""
        return {
            "progress_id": self.progress_id,
            "test_case_id": str(self.test_case_id),
            "test_case_name": self.test_case_name,
            "workflow": self.workflow_type,
            "state": self.state.value,
            "phase": self.phase,
            "requested_at": self.requested_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_ms": self.elapsed_ms,
            "steps": [{
                "id": str(step.id),
                "order": step.order,
                "name": step.name,
                "state": step.state.value,
                "automation_state": step.automation_state,
            } for step in self.steps],
            "events": [event.to_public_dict() for event in self.events],
            "final_run_id": str(self.final_run_id) if self.final_run_id else None,
            "final_run_url": self.final_run_url,
            "run_status": self.run_status,
            "outcome": self.outcome,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "finished": self.state == ProgressState.FINISHED,
        }


@dataclass
class _ProgressRecord:
    progress_id: str
    test_case_id: UUID
    workflow_type: str
    requested_at: datetime
    test_case_name: str | None = None
    state: ProgressState = ProgressState.QUEUED
    phase: str = "Queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    steps: dict[UUID, ProgressStep] = field(default_factory=dict)
    events: list[ExecutionProgressEvent] = field(default_factory=list)
    final_run_id: UUID | None = None
    run_status: str | None = None
    outcome: str | None = None
    error_category: str | None = None
    duration_ms: int | None = None
    error_message: str | None = None


class ExecutionProgressStore:
    """Small locked progress store; running records are never pruned."""

    def __init__(
        self,
        *,
        finished_ttl: timedelta = timedelta(hours=24),
        max_finished_records: int = 500,
        clock=None,
    ) -> None:
        if finished_ttl.total_seconds() <= 0:
            raise ValueError("finished_ttl must be positive.")
        if max_finished_records < 1:
            raise ValueError("max_finished_records must be positive.")
        self._finished_ttl = finished_ttl
        self._max_finished_records = max_finished_records
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._records: dict[str, _ProgressRecord] = {}
        self._lock = RLock()

    def create(self, test_case_id: UUID, workflow_type: str | Enum) -> str:
        now = self._now()
        progress_id = token_urlsafe(24)
        with self._lock:
            self._prune(now)
            self._records[progress_id] = _ProgressRecord(
                progress_id=progress_id,
                test_case_id=test_case_id,
                workflow_type=(
                    workflow_type.value
                    if isinstance(workflow_type, Enum)
                    else str(workflow_type)
                ),
                requested_at=now,
            )
        return progress_id

    def get(self, progress_id: str) -> ExecutionProgressSnapshot | None:
        now = self._now()
        with self._lock:
            self._prune(now)
            record = self._records.get(progress_id)
            return self._snapshot(record, now) if record is not None else None

    def register_test_case(
        self,
        progress_id: str,
        test_case_id: UUID,
        name: str,
        steps: list[ProgressStep],
    ) -> None:
        with self._lock:
            record = self._require_record(progress_id)
            if record.test_case_id != test_case_id:
                raise ValueError("Progress record belongs to a different TestCase.")
            record.test_case_name = name
            record.steps = {step.id: step for step in steps}

    def append_for(
        self,
        progress_id: str,
        event_type: ExecutionEventType,
        *,
        step_id: UUID | None = None,
        step_order: int | None = None,
        step_name: str | None = None,
        status: str | None = None,
        classification: str | None = None,
        message: str | None = None,
        run_id: UUID | None = None,
        run_status: str | None = None,
        outcome: str | None = None,
        error_category: str | None = None,
        duration_ms: int | None = None,
        evidence_execution_id: UUID | None = None,
        evidence_index: int | None = None,
    ) -> ExecutionProgressEvent:
        now = self._now()
        with self._lock:
            record = self._require_record(progress_id)
            timestamp = max(now, record.events[-1].timestamp) if record.events else now
            event = ExecutionProgressEvent(
                event_type=event_type,
                timestamp=timestamp,
                test_case_id=record.test_case_id,
                workflow_type=record.workflow_type,
                step_id=step_id,
                step_order=step_order,
                step_name=step_name,
                status=status,
                classification=classification,
                message=message,
                run_id=run_id,
                run_status=run_status,
                outcome=outcome,
                error_category=error_category,
                duration_ms=duration_ms,
                evidence_execution_id=evidence_execution_id,
                evidence_index=evidence_index,
                evidence_name="Screenshot" if evidence_execution_id is not None else None,
            )
            self._append_to_record(record, event)
            return event

    def _append_to_record(
        self, record: _ProgressRecord, event: ExecutionProgressEvent
    ) -> None:
        if record.state == ProgressState.FINISHED:
            raise RuntimeError("Cannot append events to finished progress.")
        record.events.append(event)
        if event.event_type == ExecutionEventType.RUN_STARTED:
            record.state = ProgressState.RUNNING
            record.started_at = event.timestamp
            record.phase = "Loading TestCase"
        elif event.event_type == ExecutionEventType.TESTCASE_LOADED:
            record.phase = "TestCase loaded"
        elif event.event_type == ExecutionEventType.SETUP_STARTED:
            record.phase = "Establishing preconditions"
        elif event.event_type in {
            ExecutionEventType.SETUP_SUCCEEDED,
            ExecutionEventType.SETUP_FAILED,
        }:
            record.phase = "Preparing workflow"
        elif event.event_type == ExecutionEventType.AUTOMATION_PREPARATION_STARTED:
            record.phase = "Preparing automation"
        elif event.event_type == ExecutionEventType.PLAN_GENERATION_STARTED:
            record.phase = "Generating automation"
            self._update_step(record, event.step_id, state=ProgressStepState.PREPARING_AUTOMATION)
        elif event.event_type == ExecutionEventType.PLAN_REUSED:
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.READY,
                automation_state="Reused",
            )
            record.phase = "Preparing automation"
        elif event.event_type == ExecutionEventType.PLAN_GENERATED:
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.READY,
                automation_state="Generated",
            )
            record.phase = "Preparing automation"
        elif event.event_type == ExecutionEventType.PLAN_GENERATION_FAILED:
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.FAILED,
                automation_state="Failed",
            )
            record.phase = "Preparing automation"
        elif event.event_type == ExecutionEventType.STEP_STARTED:
            self._update_step(record, event.step_id, state=ProgressStepState.RUNNING)
            record.phase = "Running test"
        elif event.event_type == ExecutionEventType.STEP_PASSED:
            self._update_step(record, event.step_id, state=ProgressStepState.PASSED)
        elif event.event_type == ExecutionEventType.STEP_FAILED:
            self._update_step(record, event.step_id, state=ProgressStepState.FAILED)
        elif event.event_type == ExecutionEventType.STEP_BLOCKED:
            self._update_step(record, event.step_id, state=ProgressStepState.BLOCKED)
        elif event.event_type == ExecutionEventType.CLEANUP_STARTED:
            record.phase = "Cleaning up"
        elif event.event_type in {
            ExecutionEventType.CLEANUP_SUCCEEDED,
            ExecutionEventType.CLEANUP_FAILED,
        }:
            record.phase = "Finishing run"
        elif event.event_type == ExecutionEventType.RUN_FINISHED:
            record.state = ProgressState.FINISHED
            record.finished_at = event.timestamp
            record.phase = "Finished"
            record.final_run_id = event.run_id
            record.run_status = event.run_status
            record.outcome = event.outcome
            record.error_category = event.error_category
            record.error_message = event.message
            record.duration_ms = event.duration_ms

    @staticmethod
    def _update_step(
        record: _ProgressRecord,
        step_id: UUID | None,
        *,
        state: ProgressStepState,
        automation_state: str | None = None,
    ) -> None:
        if step_id is None or step_id not in record.steps:
            return
        current = record.steps[step_id]
        record.steps[step_id] = current.model_copy(update={
            "state": state,
            "automation_state": (
                automation_state
                if automation_state is not None
                else current.automation_state
            ),
        })

    def _require_record(self, progress_id: str) -> _ProgressRecord:
        record = self._records.get(progress_id)
        if record is None:
            raise KeyError("Execution progress is no longer available.")
        return record

    def _snapshot(
        self, record: _ProgressRecord, now: datetime
    ) -> ExecutionProgressSnapshot:
        start = record.started_at or record.requested_at
        end = record.finished_at or now
        elapsed_ms = max(0, int((end - start).total_seconds() * 1000))
        return ExecutionProgressSnapshot(
            progress_id=record.progress_id,
            test_case_id=record.test_case_id,
            test_case_name=record.test_case_name,
            workflow_type=record.workflow_type,
            state=record.state,
            phase=record.phase,
            requested_at=record.requested_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            steps=tuple(sorted(record.steps.values(), key=lambda item: item.order)),
            events=tuple(record.events),
            final_run_id=record.final_run_id,
            run_status=record.run_status,
            outcome=record.outcome,
            error_category=record.error_category,
            error_message=record.error_message,
            elapsed_ms=(
                record.duration_ms
                if record.state == ProgressState.FINISHED and record.duration_ms is not None
                else elapsed_ms
            ),
        )

    def _prune(self, now: datetime) -> None:
        expired = [
            key for key, item in self._records.items()
            if item.state == ProgressState.FINISHED
            and item.finished_at is not None
            and now - item.finished_at >= self._finished_ttl
        ]
        for key in expired:
            self._records.pop(key, None)
        finished = sorted(
            (
                item for item in self._records.values()
                if item.state == ProgressState.FINISHED
            ),
            key=lambda item: item.finished_at or item.requested_at,
        )
        for item in finished[:-self._max_finished_records]:
            self._records.pop(item.progress_id, None)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("Progress clock must return timezone-aware datetimes.")
        return value.astimezone(timezone.utc)


class ExecutionProgressReporter:
    """Safe event-writing facade bound to one progress ID and RunContext."""

    def __init__(
        self,
        store: ExecutionProgressStore,
        progress_id: str,
        run_context: RunContext | None = None,
    ) -> None:
        self._store = store
        self.progress_id = progress_id
        self.run_context = run_context or RunContext()

    def bind_run_context(self, run_context: RunContext) -> None:
        self.run_context = run_context

    def test_case_loaded(self, test_case: TestCase) -> None:
        name = self.safe_text(test_case.name)
        steps = [
            ProgressStep(
                id=step.id,
                order=step.order,
                name=self.safe_text(step.name),
            )
            for step in sorted(test_case.steps, key=lambda item: item.order)
        ]
        self._store.register_test_case(
            self.progress_id, test_case.id, name, steps
        )
        self.emit(
            ExecutionEventType.TESTCASE_LOADED,
            message="TestCase loaded.",
        )

    def emit(
        self,
        event_type: ExecutionEventType,
        *,
        step: TestStep | None = None,
        status: str | None = None,
        classification: str | None = None,
        message: str | None = None,
        run_id: UUID | None = None,
        run_status: str | None = None,
        outcome: str | None = None,
        error_category: str | None = None,
        duration_ms: int | None = None,
        evidence_execution_id: UUID | None = None,
        evidence_index: int | None = None,
    ) -> None:
        self._store.append_for(
            self.progress_id,
            event_type,
            step_id=step.id if step else None,
            step_order=step.order if step else None,
            step_name=self.safe_text(step.name) if step else None,
            status=status,
            classification=classification,
            message=self.safe_text(message) if message else None,
            run_id=run_id,
            run_status=run_status,
            outcome=outcome,
            error_category=error_category,
            duration_ms=duration_ms,
            evidence_execution_id=evidence_execution_id,
            evidence_index=evidence_index,
        )

    def finish(
        self,
        *,
        run_id: UUID | None = None,
        run_status: str | None = None,
        outcome: str,
        error_category: str | None = None,
        duration_ms: int | None = None,
        message: str | None = None,
    ) -> None:
        self.emit(
            ExecutionEventType.RUN_FINISHED,
            run_id=run_id,
            run_status=run_status,
            outcome=outcome,
            error_category=error_category,
            duration_ms=duration_ms,
            classification=outcome,
            status=run_status,
            message=self.safe_text(message or failure_message(outcome)),
        )

    def safe_text(self, value: Any) -> str:
        text = str(value)
        for secret in sorted(_sensitive_values(self.run_context), key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        text = redact_secrets(text)
        text = re.sub(r'(?i)\b[A-Z]:[\\/][^\r\n<>"\']*', "[PATH]", text)
        text = re.sub(r'(?<![:/\w])/(?!/)[^\s<>"\']+', "[PATH]", text)
        return text[:500]


_ACTIVE_PROGRESS: ContextVar[ExecutionProgressReporter | None] = ContextVar(
    "qa_agent_active_execution_progress", default=None
)


@contextmanager
def active_execution_progress(
    reporter: ExecutionProgressReporter,
) -> Iterator[None]:
    token: Token[ExecutionProgressReporter | None] = _ACTIVE_PROGRESS.set(reporter)
    try:
        yield
    finally:
        _ACTIVE_PROGRESS.reset(token)


def get_active_execution_progress() -> ExecutionProgressReporter | None:
    return _ACTIVE_PROGRESS.get()


def emit_progress_event(
    event_type: ExecutionEventType,
    *,
    step: TestStep | None = None,
    status: str | None = None,
    classification: str | None = None,
    message: str | None = None,
    evidence_execution_id: UUID | None = None,
    evidence_index: int | None = None,
) -> None:
    reporter = get_active_execution_progress()
    if reporter is not None:
        reporter.emit(
            event_type,
            step=step,
            status=status,
            classification=classification,
            message=message,
            evidence_execution_id=evidence_execution_id,
            evidence_index=evidence_index,
        )


def _sensitive_values(run_context: RunContext) -> set[str]:
    values: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value:
                values.add(value)
        elif isinstance(value, bytes):
            if value:
                values.add(value.decode("utf-8", errors="replace"))
        elif isinstance(value, dict):
            for key, child in value.items():
                collect(key)
                collect(child)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                collect(child)
        elif value is not None:
            values.add(str(value))

    for item in run_context.values.values():
        if item.sensitive:
            collect(item.value)
    return values
