"""Thread-safe in-memory execution lifecycle events for local Web runs."""

from __future__ import annotations

import logging
import re
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from secrets import token_urlsafe
from threading import RLock
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qa_agent.execution_control import completion_boundary
from qa_agent.execution_diagnostics import ActionFailureDiagnostic, redact_action_failure
from qa_agent.models import TestCase, TestStep
from qa_agent.presentation import failure_message
from qa_agent.result_semantics import result_outcome, result_label, terminal_phase, result_summary
from qa_agent.redaction import redact_secrets, redact_diagnostic
from qa_agent.llm.errors import ProviderFailureDetail, SAFE_FAILURE_DETAILS
from qa_agent.run_context import RunContext
from qa_agent.test_plan_validation import PlanValidationIssue

logger = logging.getLogger(__name__)


class ExecutionEventType(str, Enum):
    RELIABILITY_GENERATING = "RELIABILITY_GENERATING"
    RELIABILITY_CHECKING = "RELIABILITY_CHECKING"
    RELIABILITY_RETRY = "RELIABILITY_RETRY"
    RELIABILITY_FALLBACK = "RELIABILITY_FALLBACK"
    RELIABILITY_REPAIR = "RELIABILITY_REPAIR"
    RELIABILITY_READY_FOR_REVIEW = "RELIABILITY_READY_FOR_REVIEW"
    RELIABILITY_NEEDS_ATTENTION = "RELIABILITY_NEEDS_ATTENTION"
    RUN_REQUESTED = "RUN_REQUESTED"
    RUN_STARTED = "RUN_STARTED"
    TESTCASE_LOADED = "TESTCASE_LOADED"
    SETUP_STARTED = "SETUP_STARTED"
    SETUP_SUCCEEDED = "SETUP_SUCCEEDED"
    SETUP_FAILED = "SETUP_FAILED"
    AUTOMATION_PREPARATION_STARTED = "AUTOMATION_PREPARATION_STARTED"
    PLAN_REUSED = "PLAN_REUSED"
    PLAN_GENERATION_STARTED = "PLAN_GENERATION_STARTED"
    PLAN_REPAIR_STARTED = "PLAN_REPAIR_STARTED"
    PLAN_REPAIR_SUCCEEDED = "PLAN_REPAIR_SUCCEEDED"
    PLAN_REPAIR_FAILED = "PLAN_REPAIR_FAILED"
    PLAN_GENERATED = "PLAN_GENERATED"
    PLAN_GENERATION_FAILED = "PLAN_GENERATION_FAILED"
    STEP_STARTED = "STEP_STARTED"
    STEP_PASSED = "STEP_PASSED"
    STEP_CANCELLED = "STEP_CANCELLED"
    STEP_FAILED = "STEP_FAILED"
    STEP_BLOCKED = "STEP_BLOCKED"
    EVIDENCE_CAPTURED = "EVIDENCE_CAPTURED"
    CLEANUP_STARTED = "CLEANUP_STARTED"
    CLEANUP_SUCCEEDED = "CLEANUP_SUCCEEDED"
    CLEANUP_FAILED = "CLEANUP_FAILED"
    RUN_FINISHED = "RUN_FINISHED"


class ProgressState(str, Enum):
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    CANCELLED = "CANCELLED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class AuthoringEventType(str, Enum):
    AUTHORING_REQUESTED = "AUTHORING_REQUESTED"
    INPUT_VALIDATED = "INPUT_VALIDATED"
    AUTHORING_STARTED = "AUTHORING_STARTED"
    LLM_REQUEST_STARTED = "LLM_REQUEST_STARTED"
    LLM_PROVIDER_SELECTED = "LLM_PROVIDER_SELECTED"
    LLM_PROVIDER_FAILED = "LLM_PROVIDER_FAILED"
    LLM_PROVIDER_FALLBACK = "LLM_PROVIDER_FALLBACK"
    LLM_PROVIDER_COMPLETED = "LLM_PROVIDER_COMPLETED"
    LLM_RESPONSE_RECEIVED = "LLM_RESPONSE_RECEIVED"
    TESTCASE_VALIDATION_STARTED = "TESTCASE_VALIDATION_STARTED"
    TESTCASE_VALIDATED = "TESTCASE_VALIDATED"
    DRAFT_CREATED = "DRAFT_CREATED"
    AUTHORING_FINISHED = "AUTHORING_FINISHED"
    AUTHORING_FAILED = "AUTHORING_FAILED"
    AUTHORING_CANCELLED = "AUTHORING_CANCELLED"


class ProgressStepState(str, Enum):
    PENDING = "PENDING"
    PREPARING_AUTOMATION = "PREPARING_AUTOMATION"
    READY = "READY"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class ProgressStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    order: int
    name: str
    state: ProgressStepState = ProgressStepState.PENDING
    execution_state: ProgressStepState = ProgressStepState.PENDING
    automation_state: str | None = None
    plan_origin: str | None = None
    plan_version: int | None = None
    plan_version_id: UUID | None = None
    failure_classification: str | None = None
    message: str | None = None
    evidence_count: int = 0
    action_failure: ActionFailureDiagnostic | None = None


class AutomationGenerationFailure(BaseModel):
    """Safe, step-specific details for a failed Automation plan attempt."""

    model_config = ConfigDict(frozen=True)

    test_case_id: UUID
    workflow_type: str
    step_id: UUID
    step_order: int
    step_name: str
    prior_plan_exists: bool
    new_plan_saved: bool
    failure_category: str
    safe_reason: str
    technical_classification: str
    validation_issues: tuple[PlanValidationIssue, ...] = ()


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
    plan_origin: str | None = None
    plan_version: int | None = None
    plan_version_id: UUID | None = None
    reliability_operation_id: UUID | None = None
    failure_code: str | None = None
    prior_plan_exists: bool | None = None
    new_plan_saved: bool | None = None
    message: str | None = None
    validation_issues: tuple[PlanValidationIssue, ...] = ()
    run_id: UUID | None = None
    run_status: str | None = None
    outcome: str | None = None
    error_category: str | None = None
    duration_ms: int | None = None
    evidence_execution_id: UUID | None = None
    evidence_index: int | None = None
    evidence_name: str | None = None
    action_failure: ActionFailureDiagnostic | None = None

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
            "plan_origin": self.plan_origin,
            "plan_version": self.plan_version,
            "plan_version_id": str(self.plan_version_id) if self.plan_version_id else None,
            "reliability_operation_id": str(self.reliability_operation_id) if self.reliability_operation_id else None,
            "failure_code": self.failure_code,
            "prior_plan_exists": self.prior_plan_exists,
            "new_plan_saved": self.new_plan_saved,
            "message": self.message,
            "validation_issues": [issue.model_dump(mode="json") for issue in self.validation_issues],
            "run_id": str(self.run_id) if self.run_id else None,
            "run_status": self.run_status,
            "outcome": self.outcome,
            "error_category": self.error_category,
            "duration_ms": self.duration_ms,
            "action_failure": self.action_failure.model_dump(mode="json") if self.action_failure else None,
            "evidence": ({
                "execution_id": str(self.evidence_execution_id),
                "index": self.evidence_index,
                "name": self.evidence_name,
            } if self.evidence_execution_id is not None else None),
        }


class ExecutionProgressSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    progress_id: str
    kind: str = "EXECUTION"
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
    provider_diagnostics: tuple[dict[str, Any], ...] = ()
    final_run_id: UUID | None = None
    run_status: str | None = None
    outcome: str | None = None
    error_category: str | None = None
    error_message: str | None = None
    automation_generation_failure: AutomationGenerationFailure | None = None
    elapsed_ms: int = 0

    @model_validator(mode="after")
    def terminal_snapshot_is_coherent(self) -> "ExecutionProgressSnapshot":
        if self.state == ProgressState.FINISHED:
            if self.finished_at is None:
                raise ValueError("Finished progress must have a terminal phase and timestamp.")
            if self.run_status == "RUNNING":
                raise ValueError("Finished progress cannot contain a running TestRun.")
            if any(
                step.state == ProgressStepState.RUNNING
                or step.execution_state == ProgressStepState.RUNNING
                for step in self.steps
            ):
                raise ValueError("Finished progress cannot contain a running step.")
        return self

    @property
    def final_run_url(self) -> str | None:
        return f"/runs/{self.final_run_id}" if self.final_run_id else None

    def to_public_dict(self) -> dict[str, Any]:
        """Return an allowlisted JSON view; internal state and paths stay private."""
        complete = bool(self.steps) and all(step.execution_state == ProgressStepState.PASSED for step in self.steps)
        public_outcome = result_outcome(
            self.run_status if self.state == ProgressState.FINISHED else self.state,
            self.outcome or self.error_category, complete=complete,
        )
        summary = result_summary(public_outcome, [
            {
                "name": step.name,
                "outcome": result_outcome(step.state, step.failure_classification or (
                    "PASSED" if step.execution_state == ProgressStepState.PASSED else None
                )) if step.state in {ProgressStepState.PASSED, ProgressStepState.FAILED, ProgressStepState.BLOCKED}
                else step.state.value,
            } for step in self.steps
        ])
        if self.automation_generation_failure:
            summary["explanation"] = self.automation_generation_failure.safe_reason
        if self.outcome == "AUTOMATION_REVIEW_REQUIRED":
            summary["explanation"] = "A repaired automation candidate is saved and requires human review before execution."
            summary["recommended_action"] = "Review the saved automation, then explicitly approve it for Validation."
        if self.outcome == "CANCELLED":
            summary["explanation"] = "The operation was stopped by user. Completed results and evidence remain available."
            summary["recommended_action"] = "Review available results before starting another operation."
        summary["history_note"] = (
            "No persisted Run was created. Partial progress diagnostics are available for this request."
            if self.state == ProgressState.FINISHED and self.final_run_id is None else None
        )
        return {
            "progress_id": self.progress_id,
            "kind": self.kind,
            "test_case_id": str(self.test_case_id),
            "test_case_name": self.test_case_name,
            "workflow": self.workflow_type,
            "state": self.state.value,
            "phase": self.phase,
            "display_status": result_label(public_outcome),
            "summary": summary,
            "diagnostic_stages": _diagnostic_stages(self.events),
            "provider_diagnostics": list(self.provider_diagnostics),
            "requested_at": self.requested_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_ms": self.elapsed_ms,
            "steps": [{
                "id": str(step.id),
                "order": step.order,
                "name": step.name,
                "state": step.state.value,
                "execution_state": step.execution_state.value,
                "automation_state": step.automation_state,
                "plan_origin": step.plan_origin,
                "plan_version": step.plan_version,
                "plan_version_id": str(step.plan_version_id) if step.plan_version_id else None,
                "display_status": result_label(result_outcome(
                    step.state, step.failure_classification or ("PASSED" if step.state == ProgressStepState.PASSED else None),
                )) if step.state in {ProgressStepState.PASSED, ProgressStepState.FAILED, ProgressStepState.BLOCKED} else step.state.value.replace("_", " ").title(),
                "failure_classification": step.failure_classification,
                "message": step.message,
                "evidence_count": step.evidence_count,
                "action_failure": step.action_failure.model_dump(mode="json") if step.action_failure else None,
            } for step in self.steps],
            "events": [event.to_public_dict() for event in self.events],
            "final_run_id": str(self.final_run_id) if self.final_run_id else None,
            "final_run_url": self.final_run_url,
            "run_status": self.run_status,
            "outcome": self.outcome,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "automation_generation_failure": (
                self.automation_generation_failure.model_dump(mode="json")
                if self.automation_generation_failure is not None else None
            ),
            "finished": self.state == ProgressState.FINISHED,
        }


def _diagnostic_stages(events) -> list[dict[str, Any]]:
    """Group consecutive stages while preserving recorded chronology and attempts."""
    groups: list[dict[str, Any]] = []
    attempts: dict[tuple[str, UUID | None], int] = {}
    starts: dict[tuple[str, UUID | None], datetime] = {}
    previous = None
    for event in events:
        name = event.event_type.value
        stage = (
            "Plan validation" if name in {"RELIABILITY_CHECKING", "RELIABILITY_READY_FOR_REVIEW"}
            else "Repair attempt" if name == "RELIABILITY_REPAIR"
            else "Automation generation" if name.startswith("RELIABILITY_")
            else "TestCase loading" if name == "TESTCASE_LOADED"
            else "Repair attempt" if name.startswith("PLAN_REPAIR")
            else "Plan validation" if name == "PLAN_GENERATED" or event.validation_issues
            else "Automation generation" if name.startswith("PLAN_GENERATION")
            else "Browser execution" if name.startswith(("STEP_", "EVIDENCE_"))
            else "Reporting and completion" if name.startswith("CLEANUP_") or name == "RUN_FINISHED"
            else "Preparation"
        )
        signature = (stage, name, event.step_id, event.status, event.classification, event.message, event.evidence_execution_id, event.evidence_index)
        if signature == previous:
            continue
        previous = signature
        attempt_stage = "Automation generation" if name in {"PLAN_GENERATED", "PLAN_GENERATION_FAILED"} else stage
        key = (attempt_stage, event.step_id)
        if name in {"STEP_STARTED", "PLAN_GENERATION_STARTED", "PLAN_REPAIR_STARTED"}:
            attempts[key] = attempts.get(key, 0) + 1
            starts[key] = event.timestamp
        row = event.to_public_dict()
        row["reason"] = redact_diagnostic(event.message or name.replace("_", " ").title())
        row["attempt_number"] = attempts.get(key)
        row["display_status"] = (
            result_label(result_outcome(event.status or event.run_status, event.classification or event.outcome))
            if event.classification or event.outcome else
            "Error" if name.endswith("FAILED") else "Started" if name.endswith("STARTED") else "Recorded"
        )
        if event.duration_ms is None and key in starts and name in {"STEP_PASSED", "STEP_FAILED", "PLAN_GENERATED", "PLAN_GENERATION_FAILED", "PLAN_REPAIR_SUCCEEDED", "PLAN_REPAIR_FAILED"}:
            row["duration_ms"] = max(0, int((event.timestamp - starts.pop(key)).total_seconds() * 1000))
        if not groups or groups[-1]["stage"] != stage:
            groups.append({"stage": stage, "events": []})
        groups[-1]["events"].append(row)
    return groups


class AuthoringProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    event_type: AuthoringEventType
    timestamp: datetime
    message: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }


class AuthoringProgressSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    progress_id: str
    kind: str = "AUTHORING"
    test_case_name: str
    base_url: str
    state: ProgressState
    phase: str
    provider_name: str | None = None
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_ms: int = 0
    events: tuple[AuthoringProgressEvent, ...] = ()
    finished: bool = False
    success: bool | None = None
    error_category: str | None = None
    error_message: str | None = None
    provider_failures: tuple[ProviderFailureDetail, ...] = ()
    review_url: str | None = None

    @model_validator(mode="after")
    def terminal_state_is_coherent(self) -> "AuthoringProgressSnapshot":
        terminal = self.state in {ProgressState.FINISHED, ProgressState.FAILED, ProgressState.CANCELLED}
        if self.finished != terminal:
            raise ValueError("Authoring finished flag must match its lifecycle state.")
        if self.finished != (self.finished_at is not None):
            raise ValueError("Finished authoring progress requires a completion timestamp.")
        if self.success is True and (
            self.state != ProgressState.FINISHED or not self.review_url
        ):
            raise ValueError("Successful authoring requires a review URL.")
        if self.success is False and (
            self.state != ProgressState.FAILED
            or not self.error_category
            or not self.error_message
        ):
            raise ValueError("Failed authoring requires a safe category and message.")
        if self.success is not True and self.review_url is not None:
            raise ValueError("Only successful authoring may expose a review URL.")
        return self

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "progress_id": self.progress_id,
            "kind": self.kind,
            "test_case_name": self.test_case_name,
            "base_url": self.base_url,
            "workflow": "AUTHORING",
            "state": self.state.value,
            "phase": self.phase,
            "provider_name": self.provider_name,
            "requested_at": self.requested_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_ms": self.elapsed_ms,
            "events": [event.to_public_dict() for event in self.events],
            "finished": self.finished,
            "success": self.success,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "provider_failures": [
                failure.to_public_dict() for failure in self.provider_failures
            ],
            "review_url": self.review_url,
        }


@dataclass
class _AuthoringProgressRecord:
    progress_id: str
    test_case_name: str
    base_url: str
    original_base_url: str
    scenario: str
    source_draft_token: str | None
    source_draft_id: UUID | None
    requested_at: datetime
    state: ProgressState = ProgressState.QUEUED
    phase: str = "Preparing"
    provider_name: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[AuthoringProgressEvent] = field(default_factory=list)
    success: bool | None = None
    error_category: str | None = None
    error_message: str | None = None
    provider_failures: tuple[ProviderFailureDetail, ...] = ()
    review_url: str | None = None


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
    provider_diagnostics: tuple[dict[str, Any], ...] = ()
    final_run_id: UUID | None = None
    run_status: str | None = None
    outcome: str | None = None
    error_category: str | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    automation_generation_failure: AutomationGenerationFailure | None = None
    running_step_ids: set[UUID] = field(default_factory=set)


class ExecutionProgressStore:
    """Shared in-memory store for execution and authoring progress records."""

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
        self._authoring_records: dict[str, _AuthoringProgressRecord] = {}
        self._lock = RLock()

    def create(self, test_case_id: UUID, workflow_type: str | Enum) -> str:
        now = self._now()
        with self._lock:
            self._prune(now)
            progress_id = token_urlsafe(24)
            while progress_id in self._records or progress_id in self._authoring_records:
                progress_id = token_urlsafe(24)
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

    def create_authoring(
        self,
        *,
        name: str,
        base_url: str,
        scenario: str,
        source_draft_token: str | None = None,
        source_draft_id: UUID | None = None,
    ) -> str:
        now = self._now()
        safe_url = redact_secrets(_safe_authoring_base_url(base_url))
        with self._lock:
            self._prune(now)
            progress_id = token_urlsafe(24)
            while progress_id in self._records or progress_id in self._authoring_records:
                progress_id = token_urlsafe(24)
            self._authoring_records[progress_id] = _AuthoringProgressRecord(
                progress_id=progress_id,
                test_case_name=name,
                base_url=safe_url,
                original_base_url=base_url,
                scenario=scenario,
                source_draft_token=source_draft_token,
                source_draft_id=source_draft_id,
                requested_at=now,
            )
        return progress_id

    def get_authoring_source_draft_id(self, progress_id: str) -> UUID | None:
        """Return the server-held persistent Draft source for a job."""
        now = self._now()
        with self._lock:
            self._prune(now)
            record = self._authoring_records.get(progress_id)
            return record.source_draft_id if record is not None else None

    def get_authoring(self, progress_id: str) -> AuthoringProgressSnapshot | None:
        now = self._now()
        with self._lock:
            self._prune(now)
            record = self._authoring_records.get(progress_id)
            return self._authoring_snapshot(record, now) if record is not None else None

    def get_authoring_retry_data(
        self, progress_id: str
    ) -> tuple[str, str, str, str | None] | None:
        """Return original input and source draft for a valid failed request retry."""
        now = self._now()
        with self._lock:
            self._prune(now)
            record = self._authoring_records.get(progress_id)
            if record is None or record.state not in {ProgressState.FAILED, ProgressState.CANCELLED} or record.success is True:
                return None
            return (
                record.test_case_name,
                record.original_base_url,
                record.scenario,
                record.source_draft_token,
            )

    def append_authoring(
        self,
        progress_id: str,
        event_type: AuthoringEventType,
        *,
        message: str | None = None,
        provider_name: str | None = None,
    ) -> AuthoringProgressEvent:
        now = self._now()
        with self._lock:
            record = self._require_authoring_record(progress_id)
            if record.state in {ProgressState.FINISHED, ProgressState.FAILED, ProgressState.CANCELLED}:
                raise RuntimeError("Cannot append events to finished authoring progress.")
            _validate_authoring_event_order(record, event_type)
            timestamp = max(now, record.events[-1].timestamp) if record.events else now
            event = AuthoringProgressEvent(
                event_type=event_type,
                timestamp=timestamp,
                message=redact_secrets(message or _authoring_event_message(event_type)),
            )
            record.events.append(event)
            if event_type == AuthoringEventType.AUTHORING_STARTED:
                record.state = ProgressState.RUNNING
                record.started_at = event.timestamp
                record.phase = "Understanding scenario"
            elif event_type == AuthoringEventType.LLM_REQUEST_STARTED:
                record.phase = "Generating TestCase"
            elif event_type in {
                AuthoringEventType.LLM_PROVIDER_SELECTED,
                AuthoringEventType.LLM_PROVIDER_FALLBACK,
                AuthoringEventType.LLM_PROVIDER_COMPLETED,
            }:
                record.provider_name = (
                    _safe_provider_display_name(provider_name)
                    if provider_name else "Configured provider"
                )
            elif event_type in {
                AuthoringEventType.LLM_RESPONSE_RECEIVED,
                AuthoringEventType.TESTCASE_VALIDATION_STARTED,
            }:
                record.phase = "Validating result"
            elif event_type in {
                AuthoringEventType.TESTCASE_VALIDATED,
                AuthoringEventType.DRAFT_CREATED,
            }:
                record.phase = "Preparing review"
            return event

    def finish_authoring(
        self,
        progress_id: str,
        *,
        review_url: str | None = None,
        error_category: str | None = None,
        error_message: str | None = None,
        provider_failures: tuple[ProviderFailureDetail, ...] = (),
    ) -> None:
        with self._lock:
            record = self._require_authoring_record(progress_id)
            if record.state in {ProgressState.FINISHED, ProgressState.FAILED, ProgressState.CANCELLED}:
                return
            success = review_url is not None
            if success:
                if error_category is not None or error_message is not None:
                    raise ValueError("Successful authoring cannot contain failure details.")
                event_type = AuthoringEventType.AUTHORING_FINISHED
                safe_message = _authoring_event_message(event_type)
            else:
                if not error_category or not error_message:
                    raise ValueError("Failed authoring requires a safe category and message.")
                event_type = AuthoringEventType.AUTHORING_CANCELLED if error_category == "CANCELLED" else AuthoringEventType.AUTHORING_FAILED
                safe_message = error_message
            if error_category != "CANCELLED":
                _validate_authoring_event_order(record, event_type)
            now = self._now()
            timestamp = max(now, record.events[-1].timestamp) if record.events else now
            record.events.append(AuthoringProgressEvent(
                event_type=event_type,
                timestamp=timestamp,
                message=redact_secrets(safe_message),
            ))
            record.state = ProgressState.CANCELLED if error_category == "CANCELLED" else ProgressState.FINISHED if success else ProgressState.FAILED
            record.finished_at = timestamp
            record.phase = "Stopped by user" if error_category == "CANCELLED" else "Ready for review" if success else "Finished"
            record.success = None if error_category == "CANCELLED" else success
            record.error_category = error_category
            record.error_message = redact_secrets(error_message) if error_message else None
            record.provider_failures = tuple(provider_failures[:8])
            record.review_url = review_url

    def request_cancellation(self, progress_id: str, *, authoring: bool = False) -> None:
        with self._lock:
            record = self._require_authoring_record(progress_id) if authoring else self._require_record(progress_id)
            if record.state not in {ProgressState.FINISHED, ProgressState.FAILED, ProgressState.CANCELLED}:
                record.state = ProgressState.CANCELLATION_REQUESTED
                record.phase = "Stopping… Cancellation requested; waiting for safe cleanup."

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
        plan_origin: str | None = None,
        plan_version: int | None = None,
        plan_version_id: UUID | None = None,
        reliability_operation_id: UUID | None = None,
        failure_code: str | None = None,
        prior_plan_exists: bool | None = None,
        new_plan_saved: bool | None = None,
        message: str | None = None,
        validation_issues: tuple[PlanValidationIssue, ...] = (),
        run_id: UUID | None = None,
        run_status: str | None = None,
        outcome: str | None = None,
        error_category: str | None = None,
        duration_ms: int | None = None,
        evidence_execution_id: UUID | None = None,
        evidence_index: int | None = None,
        action_failure: ActionFailureDiagnostic | None = None,
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
                plan_origin=plan_origin,
                plan_version=plan_version,
                plan_version_id=plan_version_id,
                reliability_operation_id=reliability_operation_id,
                failure_code=failure_code,
                prior_plan_exists=prior_plan_exists,
                new_plan_saved=new_plan_saved,
                message=message,
                validation_issues=validation_issues,
                run_id=run_id,
                run_status=run_status,
                outcome=outcome,
                error_category=error_category,
                duration_ms=duration_ms,
                evidence_execution_id=evidence_execution_id,
                evidence_index=evidence_index,
                evidence_name="Screenshot" if evidence_execution_id is not None else None,
                action_failure=action_failure,
            )
            self._append_to_record(record, event)
            return event

    def set_provider_diagnostics(self, progress_id: str, diagnostics: tuple[dict[str, Any], ...]) -> None:
        with self._lock:
            record = self._require_record(progress_id)
            if record.state != ProgressState.FINISHED:
                record.provider_diagnostics = diagnostics

    def finish_for(
        self,
        progress_id: str,
        *,
        run_id: UUID | None = None,
        run_status: str | None = None,
        outcome: str,
        error_category: str | None = None,
        duration_ms: int | None = None,
        status: str | None = None,
        classification: str | None = None,
        message: str | None = None,
    ) -> None:
        """Record the terminal event once; repeated finalization is harmless."""
        with self._lock:
            record = self._require_record(progress_id)
            if record.state == ProgressState.FINISHED:
                return
            if run_status == "RUNNING":
                run_status = None
                outcome = "EXECUTION_ERROR"
                error_category = "EXECUTION_ERROR"
                duration_ms = None
                message = "The workflow returned before execution reached a terminal state."
            self.append_for(
                progress_id,
                ExecutionEventType.RUN_FINISHED,
                run_id=run_id,
                run_status=run_status,
                outcome=outcome,
                error_category=error_category,
                duration_ms=duration_ms,
                status=status,
                classification=classification or outcome,
                message=message,
            )

    def _append_to_record(
        self, record: _ProgressRecord, event: ExecutionProgressEvent
    ) -> None:
        if record.state == ProgressState.FINISHED:
            raise RuntimeError("Cannot append events to finished progress.")
        if event.event_type == ExecutionEventType.RUN_FINISHED and record.state == ProgressState.FINISHED:
            raise RuntimeError("Execution progress is already finished.")
        stopping = record.state == ProgressState.CANCELLATION_REQUESTED
        if event.event_type in {ExecutionEventType.STEP_PASSED, ExecutionEventType.STEP_FAILED, ExecutionEventType.STEP_CANCELLED}:
            if event.step_id not in record.running_step_ids:
                raise ValueError("A step cannot finish before it has started.")
        record.events.append(event)
        if event.event_type.value.startswith("RELIABILITY_"):
            record.phase = {
                "RELIABILITY_GENERATING": "Generating automation", "RELIABILITY_CHECKING": "Checking automation quality",
                "RELIABILITY_RETRY": "Retrying provider request", "RELIABILITY_FALLBACK": "Switching provider",
                "RELIABILITY_REPAIR": "Repairing unapproved candidate", "RELIABILITY_READY_FOR_REVIEW": "Ready for review",
                "RELIABILITY_NEEDS_ATTENTION": "Needs attention",
            }[event.event_type.value]
        elif event.event_type == ExecutionEventType.RUN_STARTED:
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
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.PREPARING_AUTOMATION,
                automation_state="Preparing",
                message=event.message,
            )
        elif event.event_type == ExecutionEventType.PLAN_REPAIR_STARTED:
            record.phase = "Repairing automation"
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.PREPARING_AUTOMATION,
                automation_state="Repairing",
                message=event.message,
            )
        elif event.event_type == ExecutionEventType.PLAN_REPAIR_SUCCEEDED:
            record.phase = "Validating automation"
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.READY,
                automation_state="Repaired",
                plan_origin=event.plan_origin or "REPAIRED",
                plan_version=event.plan_version,
                plan_version_id=event.plan_version_id,
                message=event.message,
            )
        elif event.event_type == ExecutionEventType.PLAN_REPAIR_FAILED:
            record.phase = "Preparing automation"
            self._update_step(
                record, event.step_id,
                automation_state="Needs attention",
                message=event.message,
            )
        elif event.event_type == ExecutionEventType.PLAN_REUSED:
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.READY,
                automation_state="Reused",
                plan_origin=event.plan_origin,
                plan_version=event.plan_version,
                plan_version_id=event.plan_version_id,
                message=event.message,
            )
            record.phase = "Preparing automation"
        elif event.event_type == ExecutionEventType.PLAN_GENERATED:
            automation_state = {
                "REPAIRED": "Repaired",
                "REGENERATED": "Regenerated",
            }.get(event.plan_origin, "Generated")
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.READY,
                automation_state=automation_state,
                plan_origin=event.plan_origin,
                plan_version=event.plan_version,
                plan_version_id=event.plan_version_id,
                message=event.message,
            )
            record.phase = "Preparing automation"
        elif event.event_type == ExecutionEventType.PLAN_GENERATION_FAILED:
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.FAILED,
                execution_state=ProgressStepState.NOT_ATTEMPTED,
                automation_state="Failed",
                failure_classification=event.classification,
                message=event.message,
            )
            record.phase = "Preparing automation"
            if event.classification == "AUTOMATION_GENERATION_ERROR" and event.step_id is not None:
                record.automation_generation_failure = AutomationGenerationFailure(
                    test_case_id=record.test_case_id,
                    workflow_type=record.workflow_type,
                    step_id=event.step_id,
                    step_order=event.step_order or 0,
                    step_name=event.step_name or "Test step",
                    prior_plan_exists=bool(event.prior_plan_exists),
                    new_plan_saved=bool(event.new_plan_saved),
                    failure_category=event.classification,
                    safe_reason=event.message or "No reliable executable action could be produced for this step.",
                    technical_classification=event.failure_code or "PLAN_GENERATION_FAILED",
                    validation_issues=event.validation_issues,
                )
        elif event.event_type == ExecutionEventType.STEP_STARTED:
            if event.step_id is not None:
                step = record.steps.get(event.step_id)
                if step is not None and step.state not in {
                    ProgressStepState.PENDING,
                    ProgressStepState.READY,
                    ProgressStepState.FAILED,
                }:
                    raise ValueError("A step cannot start from its current progress state.")
                record.running_step_ids.add(event.step_id)
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.RUNNING,
                execution_state=ProgressStepState.RUNNING,
                clear_failure=True,
                plan_version=event.plan_version,
                plan_version_id=event.plan_version_id,
                plan_origin=event.plan_origin,
                message=event.message,
            )
            record.phase = "Running test"
        elif event.event_type == ExecutionEventType.STEP_PASSED:
            if event.step_id is not None:
                record.running_step_ids.discard(event.step_id)
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.PASSED,
                execution_state=ProgressStepState.PASSED,
                clear_failure=True,
                message=event.message,
            )
        elif event.event_type == ExecutionEventType.STEP_CANCELLED:
            if event.step_id is not None:
                record.running_step_ids.discard(event.step_id)
            self._update_step(record, event.step_id, state=ProgressStepState.CANCELLED, execution_state=ProgressStepState.CANCELLED, failure_classification="CANCELLED", message="Stopped by user.")
        elif event.event_type == ExecutionEventType.STEP_FAILED:
            if event.step_id is not None:
                record.running_step_ids.discard(event.step_id)
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.FAILED,
                execution_state=ProgressStepState.FAILED,
                failure_classification=event.classification,
                message=event.message,
                action_failure=event.action_failure,
            )
        elif event.event_type == ExecutionEventType.STEP_BLOCKED:
            self._update_step(
                record, event.step_id,
                state=ProgressStepState.BLOCKED,
                execution_state=ProgressStepState.BLOCKED,
                message=event.message,
            )
        elif event.event_type == ExecutionEventType.EVIDENCE_CAPTURED:
            self._update_step(
                record, event.step_id,
                evidence_increment=1,
            )
        elif event.event_type == ExecutionEventType.CLEANUP_STARTED:
            record.phase = "Cleaning up"
        elif event.event_type in {
            ExecutionEventType.CLEANUP_SUCCEEDED,
            ExecutionEventType.CLEANUP_FAILED,
        }:
            record.phase = "Finishing run"
        elif event.event_type == ExecutionEventType.RUN_FINISHED:
            for step_id, step in tuple(record.steps.items()):
                if step.state in {
                    ProgressStepState.RUNNING,
                }:
                    record.steps[step_id] = step.model_copy(update={
                        "state": ProgressStepState.CANCELLED if event.outcome == "CANCELLED" else ProgressStepState.FAILED,
                        "execution_state": ProgressStepState.CANCELLED if event.outcome == "CANCELLED" else ProgressStepState.FAILED,
                        "failure_classification": event.outcome or "INCONCLUSIVE",
                    })
                elif step.state == ProgressStepState.PREPARING_AUTOMATION:
                    record.steps[step_id] = step.model_copy(update={
                        "state": ProgressStepState.NOT_ATTEMPTED if event.outcome == "CANCELLED" else ProgressStepState.FAILED,
                        "automation_state": "Cancelled" if event.outcome == "CANCELLED" else "Failed",
                        "execution_state": ProgressStepState.NOT_ATTEMPTED,
                        "failure_classification": event.outcome or "INCONCLUSIVE",
                    })
                elif step.state in {ProgressStepState.PENDING, ProgressStepState.READY}:
                    record.steps[step_id] = step.model_copy(update={
                        "state": ProgressStepState.NOT_ATTEMPTED,
                        "execution_state": ProgressStepState.NOT_ATTEMPTED,
                    })
            record.running_step_ids.clear()
            failure = record.automation_generation_failure
            if failure is not None:
                for step_id, step in tuple(record.steps.items()):
                    if step.order > failure.step_order and step.state in {
                        ProgressStepState.PENDING,
                        ProgressStepState.READY,
                    }:
                        record.steps[step_id] = step.model_copy(update={
                            "state": ProgressStepState.NOT_ATTEMPTED,
                            "execution_state": ProgressStepState.NOT_ATTEMPTED,
                        })
            record.state = ProgressState.FINISHED
            record.finished_at = event.timestamp
            record.phase = terminal_phase(result_outcome(
                event.run_status, event.outcome,
                complete=bool(record.steps) and all(step.execution_state == ProgressStepState.PASSED for step in record.steps.values()),
            ))
            if event.outcome == "AUTOMATION_REVIEW_REQUIRED":
                record.phase = "Ready for review — Human approval required"
            record.final_run_id = event.run_id
            record.run_status = event.run_status
            record.outcome = event.outcome
            record.error_category = event.error_category
            record.error_message = event.message
            record.duration_ms = event.duration_ms

        if stopping and event.event_type != ExecutionEventType.RUN_FINISHED:
            record.state = ProgressState.CANCELLATION_REQUESTED
            record.phase = "Stopping… Cancellation requested; waiting for safe cleanup."

    @staticmethod
    def _update_step(
        record: _ProgressRecord,
        step_id: UUID | None,
        *,
        state: ProgressStepState | None = None,
        automation_state: str | None = None,
        execution_state: ProgressStepState | None = None,
        plan_origin: str | None = None,
        plan_version: int | None = None,
        plan_version_id: UUID | None = None,
        failure_classification: str | None = None,
        clear_failure: bool = False,
        message: str | None = None,
        evidence_increment: int = 0,
        action_failure: ActionFailureDiagnostic | None = None,
    ) -> None:
        if step_id is None or step_id not in record.steps:
            return
        current = record.steps[step_id]
        record.steps[step_id] = current.model_copy(update={
            "state": state or current.state,
            "automation_state": (
                automation_state
                if automation_state is not None
                else current.automation_state
            ),
            "execution_state": execution_state or current.execution_state,
            "plan_origin": plan_origin if plan_origin is not None else current.plan_origin,
            "plan_version": plan_version if plan_version is not None else current.plan_version,
            "plan_version_id": plan_version_id if plan_version_id is not None else current.plan_version_id,
            "failure_classification": None if clear_failure else (
                failure_classification
                if failure_classification is not None
                else current.failure_classification
            ),
            "message": message if message is not None else current.message,
            "evidence_count": current.evidence_count + evidence_increment,
            "action_failure": None if clear_failure else (action_failure or current.action_failure),
        })

    def _require_record(self, progress_id: str) -> _ProgressRecord:
        record = self._records.get(progress_id)
        if record is None:
            raise KeyError("Execution progress is no longer available.")
        return record

    def _require_authoring_record(self, progress_id: str) -> _AuthoringProgressRecord:
        record = self._authoring_records.get(progress_id)
        if record is None:
            raise KeyError("Authoring progress is no longer available.")
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
            provider_diagnostics=record.provider_diagnostics,
            final_run_id=record.final_run_id,
            run_status=record.run_status,
            outcome=record.outcome,
            error_category=record.error_category,
            error_message=record.error_message,
            automation_generation_failure=record.automation_generation_failure,
            elapsed_ms=(
                record.duration_ms
                if record.state == ProgressState.FINISHED and record.duration_ms is not None
                else elapsed_ms
            ),
        )

    def _authoring_snapshot(
        self, record: _AuthoringProgressRecord, now: datetime
    ) -> AuthoringProgressSnapshot:
        start = record.started_at or record.requested_at
        end = record.finished_at or now
        return AuthoringProgressSnapshot(
            progress_id=record.progress_id,
            test_case_name=redact_secrets(record.test_case_name),
            base_url=record.base_url,
            state=record.state,
            phase=record.phase,
            provider_name=record.provider_name,
            requested_at=record.requested_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            elapsed_ms=max(0, int((end - start).total_seconds() * 1000)),
            events=tuple(record.events),
            finished=record.state in {ProgressState.FINISHED, ProgressState.FAILED, ProgressState.CANCELLED},
            success=record.success,
            error_category=record.error_category,
            error_message=record.error_message,
            provider_failures=record.provider_failures,
            review_url=record.review_url,
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
        authoring_expired = [
            key for key, item in self._authoring_records.items()
            if item.state in {ProgressState.FINISHED, ProgressState.FAILED, ProgressState.CANCELLED}
            and item.finished_at is not None
            and now - item.finished_at >= self._finished_ttl
        ]
        for key in authoring_expired:
            self._authoring_records.pop(key, None)
        finished_authoring = sorted(
            (
                item for item in self._authoring_records.values()
                if item.state in {ProgressState.FINISHED, ProgressState.FAILED, ProgressState.CANCELLED}
            ),
            key=lambda item: item.finished_at or item.requested_at,
        )
        for item in finished_authoring[:-self._max_finished_records]:
            self._authoring_records.pop(item.progress_id, None)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("Progress clock must return timezone-aware datetimes.")
        return value.astimezone(timezone.utc)


class AuthoringProgressReporter:
    """Safe event writer for an authoring request in the shared progress store."""

    def __init__(self, store: ExecutionProgressStore, progress_id: str) -> None:
        self._store = store
        self.progress_id = progress_id

    def emit(
        self,
        event_type: AuthoringEventType | str,
        message: str | None = None,
    ) -> None:
        self._store.append_authoring(
            self.progress_id,
            AuthoringEventType(event_type),
            message=message,
        )

    def provider_progress(
        self,
        event: str,
        provider: str,
        *,
        next_provider: str | None = None,
        category: str | None = None,
        http_status: int | None = None,
        safe_detail: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        provider = _safe_provider_display_name(provider)
        next_provider = (
            _safe_provider_display_name(next_provider) if next_provider else None
        )
        category = category if category in SAFE_FAILURE_DETAILS else "OTHER_PROVIDER_ERROR"
        safe_detail = (
            safe_detail if safe_detail in SAFE_FAILURE_DETAILS.values()
            or safe_detail in {
                "Response did not match the TestCase schema",
                "Response did not contain a complete TestCase structure",
                "Structured output is not supported",
            }
            else SAFE_FAILURE_DETAILS[category]
        )
        status = f" HTTP {http_status}" if isinstance(http_status, int) else ""
        retry = (
            f" Try again in {retry_after_seconds} seconds."
            if category == "RATE_LIMIT" and isinstance(retry_after_seconds, int)
            and retry_after_seconds > 0 else " Try again later."
            if category == "RATE_LIMIT" else ""
        )
        if event == "selected":
            event_type = AuthoringEventType.LLM_PROVIDER_SELECTED
            message = f"Provider: {provider}"
        elif event == "failed":
            event_type = AuthoringEventType.LLM_PROVIDER_FAILED
            message = f"{provider} failed: {safe_detail} [{category}{status}].{retry}"
        elif event == "fallback" and next_provider:
            event_type = AuthoringEventType.LLM_PROVIDER_FALLBACK
            message = f"{provider} failed [{category}{status}]. Trying {next_provider}..."
        elif event == "completed":
            event_type = AuthoringEventType.LLM_PROVIDER_COMPLETED
            message = f"Completed with {provider}."
        else:
            return
        self._store.append_authoring(
            self.progress_id,
            event_type,
            message=message,
            provider_name=(next_provider if event == "fallback" else provider),
        )

    def finish_success(self, review_url: str) -> None:
        self._store.finish_authoring(self.progress_id, review_url=review_url)

    def finish_failure(
        self,
        category: str,
        message: str,
        *,
        provider_failures: tuple[ProviderFailureDetail, ...] = (),
    ) -> None:
        with completion_boundary() as token:
            if token is not None and token.requested:
                category, message = "CANCELLED", "Stopped by user."
            self._store.finish_authoring(
                self.progress_id,
                error_category=category,
                error_message=redact_secrets(message),
                provider_failures=provider_failures,
            )
            if token is not None:
                token.sealed = True


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

    def request_cancellation(self) -> None:
        self._store.request_cancellation(self.progress_id)

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
        plan_origin: str | None = None,
        plan_version: int | None = None,
        plan_version_id: UUID | None = None,
        reliability_operation_id: UUID | None = None,
        failure_code: str | None = None,
        prior_plan_exists: bool | None = None,
        new_plan_saved: bool | None = None,
        message: str | None = None,
        validation_issues: tuple[PlanValidationIssue, ...] = (),
        run_id: UUID | None = None,
        run_status: str | None = None,
        outcome: str | None = None,
        error_category: str | None = None,
        duration_ms: int | None = None,
        evidence_execution_id: UUID | None = None,
        evidence_index: int | None = None,
        action_failure: ActionFailureDiagnostic | None = None,
    ) -> None:
        self._store.append_for(
            self.progress_id,
            event_type,
            step_id=step.id if step else None,
            step_order=step.order if step else None,
            step_name=self.safe_text(step.name) if step else None,
            status=status,
            classification=classification,
            plan_origin=plan_origin,
            plan_version=plan_version,
            plan_version_id=plan_version_id,
            reliability_operation_id=reliability_operation_id,
            failure_code=failure_code,
            prior_plan_exists=prior_plan_exists,
            new_plan_saved=new_plan_saved,
            message=self.safe_text(message) if message else None,
            validation_issues=tuple(
                PlanValidationIssue(
                    code=issue.code,
                    path=self.safe_text(issue.path),
                    message=self.safe_text(issue.message),
                )
                for issue in validation_issues
            ),
            run_id=run_id,
            run_status=run_status,
            outcome=outcome,
            error_category=error_category,
            duration_ms=duration_ms,
            evidence_execution_id=evidence_execution_id,
            evidence_index=evidence_index,
            action_failure=redact_action_failure(action_failure, self.safe_text),
        )

    def capture_provider_diagnostics(self, trace) -> None:
        """Reuse recorded provider attempts without invoking or changing routing."""
        try:
            diagnostics = []
            for step in getattr(trace, "steps", ()):
                number = 0
                for attempt in step.provider_attempts:
                    request_sent = attempt.outcome.value != 'UNAVAILABLE'
                    if request_sent:
                        number += 1
                    row = {
                        "step_id": str(step.test_step_id),
                        "step_number": step.order + 1,
                        "attempt_number": number if request_sent else None,
                        "request_sent": request_sent,
                        "provider": self.safe_text(attempt.provider_name),
                        "request_kind": attempt.request_kind.value,
                        "status": attempt.outcome.value,
                    }
                    if attempt.model:
                        row["model"] = self.safe_text(attempt.model)
                    if attempt.duration_ms is not None:
                        row["duration_ms"] = attempt.duration_ms
                    if attempt.error_class:
                        row["reason"] = self.safe_text(attempt.error_class)
                    diagnostics.append(row)
            self._store.set_provider_diagnostics(self.progress_id, tuple(diagnostics))
        except Exception:
            logger.warning("Recorded provider diagnostics could not be displayed.")

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
        snapshot = self._store.get(self.progress_id)
        if (
            message is None
            and outcome == "AUTOMATION_GENERATION_ERROR"
            and snapshot is not None
            and snapshot.automation_generation_failure is not None
        ):
            failure = snapshot.automation_generation_failure
            message = (
                f"Automation stopped at Step {failure.step_order + 1}: "
                f"{failure.step_name}. {failure.safe_reason}"
            )
        with completion_boundary() as token:
            if token is not None and token.requested:
                outcome = error_category = "CANCELLED"
                run_status = "CANCELLED"
                message = "Stopped by user."
            self._store.finish_for(
                self.progress_id,
                run_id=run_id,
                run_status=run_status,
                outcome=outcome,
                error_category=error_category,
                duration_ms=duration_ms,
                status=run_status,
                classification=outcome,
                message=self.safe_text(message or failure_message(outcome)),
            )
            if token is not None:
                token.sealed = True

    def safe_text(self, value: Any) -> str:
        text = str(value)
        for secret in sorted(_sensitive_values(self.run_context), key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        return redact_diagnostic(text)[:500]


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
    plan_origin: str | None = None,
    plan_version: int | None = None,
    plan_version_id: UUID | None = None,
    reliability_operation_id: UUID | None = None,
    failure_code: str | None = None,
    prior_plan_exists: bool | None = None,
    new_plan_saved: bool | None = None,
    message: str | None = None,
    validation_issues: tuple[PlanValidationIssue, ...] = (),
    evidence_execution_id: UUID | None = None,
    evidence_index: int | None = None,
    action_failure: ActionFailureDiagnostic | None = None,
) -> None:
    reporter = get_active_execution_progress()
    if reporter is not None:
        reporter.emit(
            event_type,
            step=step,
            status=status,
            classification=classification,
            plan_origin=plan_origin,
            plan_version=plan_version,
            plan_version_id=plan_version_id,
            reliability_operation_id=reliability_operation_id,
            failure_code=failure_code,
            prior_plan_exists=prior_plan_exists,
            new_plan_saved=new_plan_saved,
            message=message,
            validation_issues=validation_issues,
            evidence_execution_id=evidence_execution_id,
            evidence_index=evidence_index,
            action_failure=action_failure,
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


def _safe_authoring_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return ""


def _authoring_event_message(event_type: AuthoringEventType) -> str:
    return {
        AuthoringEventType.AUTHORING_REQUESTED: "Scenario received.",
        AuthoringEventType.INPUT_VALIDATED: "Input validated.",
        AuthoringEventType.AUTHORING_STARTED: "Understanding scenario.",
        AuthoringEventType.LLM_REQUEST_STARTED: "Generating TestCase.",
        AuthoringEventType.LLM_PROVIDER_SELECTED: "Provider selected.",
        AuthoringEventType.LLM_PROVIDER_FAILED: "Provider unavailable.",
        AuthoringEventType.LLM_PROVIDER_FALLBACK: "Trying the next provider.",
        AuthoringEventType.LLM_PROVIDER_COMPLETED: "Provider completed.",
        AuthoringEventType.LLM_RESPONSE_RECEIVED: "AI response received.",
        AuthoringEventType.TESTCASE_VALIDATION_STARTED: "Validating TestCase structure.",
        AuthoringEventType.TESTCASE_VALIDATED: "TestCase structure validated.",
        AuthoringEventType.DRAFT_CREATED: "Preparing review.",
        AuthoringEventType.AUTHORING_FINISHED: "TestCase draft ready.",
        AuthoringEventType.AUTHORING_FAILED: "TestCase authoring failed.",
    }[event_type]


def _validate_authoring_event_order(
    record: _AuthoringProgressRecord,
    event_type: AuthoringEventType,
) -> None:
    if event_type == AuthoringEventType.AUTHORING_FAILED:
        if not record.events or record.events[0].event_type != AuthoringEventType.AUTHORING_REQUESTED:
            raise ValueError("Authoring failure must follow an authoring request.")
        return
    provider_events = {
        AuthoringEventType.LLM_PROVIDER_SELECTED,
        AuthoringEventType.LLM_PROVIDER_FAILED,
        AuthoringEventType.LLM_PROVIDER_FALLBACK,
        AuthoringEventType.LLM_PROVIDER_COMPLETED,
    }
    if event_type in provider_events:
        lifecycle_events = [event.event_type for event in record.events if event.event_type not in provider_events]
        if not lifecycle_events or lifecycle_events[-1] != AuthoringEventType.LLM_REQUEST_STARTED:
            raise ValueError("Provider progress is only valid while generating a TestCase.")
        return
    predecessors = {
        AuthoringEventType.AUTHORING_REQUESTED: None,
        AuthoringEventType.INPUT_VALIDATED: AuthoringEventType.AUTHORING_REQUESTED,
        AuthoringEventType.AUTHORING_STARTED: AuthoringEventType.INPUT_VALIDATED,
        AuthoringEventType.LLM_REQUEST_STARTED: AuthoringEventType.AUTHORING_STARTED,
        AuthoringEventType.LLM_RESPONSE_RECEIVED: AuthoringEventType.LLM_REQUEST_STARTED,
        AuthoringEventType.TESTCASE_VALIDATION_STARTED: AuthoringEventType.LLM_RESPONSE_RECEIVED,
        AuthoringEventType.TESTCASE_VALIDATED: AuthoringEventType.TESTCASE_VALIDATION_STARTED,
        AuthoringEventType.DRAFT_CREATED: AuthoringEventType.TESTCASE_VALIDATED,
        AuthoringEventType.AUTHORING_FINISHED: AuthoringEventType.DRAFT_CREATED,
    }
    expected_previous = predecessors[event_type]
    if expected_previous is None:
        if record.events:
            raise ValueError("Authoring request can only be the first event.")
        return
    if not record.events or record.events[-1].event_type != expected_previous:
        if event_type == AuthoringEventType.LLM_RESPONSE_RECEIVED:
            latest = next(
                (event.event_type for event in reversed(record.events)
                 if event.event_type not in provider_events),
                None,
            )
            if latest == AuthoringEventType.LLM_REQUEST_STARTED:
                return
        raise ValueError(
            f"Authoring event {event_type.value} is out of order."
        )


def _safe_provider_display_name(value: str) -> str:
    known = {
        "groq": "Groq",
        "gemini": "Gemini",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
    }
    normalized = " ".join(redact_secrets(str(value)).split()).strip()
    if normalized.casefold() in known:
        return known[normalized.casefold()]
    if normalized and len(normalized) <= 80 and not any(
        unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in normalized
    ):
        return normalized
    return "Configured provider"
