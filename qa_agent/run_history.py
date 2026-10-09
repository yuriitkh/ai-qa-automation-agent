"""Safe, durable history snapshots for completed QA workflow runs."""

from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.cookie_consent import CookieConsentRecord, current_cookie_consent_record
from qa_agent.evidence_policy import EvidencePolicy, EvidenceScope, current_evidence_policy
from qa_agent.execution_repository import ExecutionRepository
from qa_agent.models import (
    Execution,
    ExecutionStatus,
    PlanVersionOrigin,
    TestCase,
    TestRun,
)
from qa_agent.plan_store import PlanStore
from qa_agent.public_ids import format_run_public_id, parse_run_public_id
from qa_agent.redaction import redact_secrets, redact_diagnostic
from qa_agent.result_semantics import execution_classification, execution_observation
from qa_agent.run_context import RunContext
from qa_agent.setup_orchestration import CleanupOutcome, SetupRunOutcome


class WorkflowType(str, Enum):
    AUTOMATION = "AUTOMATION"
    VALIDATION = "VALIDATION"
    REGRESSION = "REGRESSION"


class HistoryStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    order: int
    name: str
    description: str
    expected: str
    status: ExecutionStatus | None = None


class HistorySegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    order: int
    base_url: str | None = None
    test_step_ids: list[UUID]


class HistoryPrecondition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    order: int
    description: str
    provided_data_keys: list[str] = Field(default_factory=list)
    status: str | None = None
    error: str | None = None
    error_type: str | None = None


class HistoryEvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    execution_id: UUID
    type: str
    name: str
    description: str | None = None
    scope: EvidenceScope | None = None
    event: str | None = None


class HistoryExecutionReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: UUID
    test_step_id: UUID
    test_plan_version_id: UUID
    plan_version_number: int | None = None
    plan_version_origin: PlanVersionOrigin | None = None
    status: ExecutionStatus
    classification: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    safe_error: str | None = None
    safe_diagnostics: str | None = None
    safe_actual_result: Any = None
    evidence: list[HistoryEvidenceReference] = Field(default_factory=list)


class HistoryCleanupFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    error_type: str
    message: str


class RunHistoryRecord(BaseModel):
    """Safe browse/report snapshot; full Executions remain in their repository."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    public_id: str | None = Field(default=None, pattern=r"^RUN-\d{6,}$")
    test_case_id: UUID
    test_case_public_id: str | None = Field(default=None, pattern=r"^TC-\d{4,}$")
    test_case_name: str
    test_case_description: str
    base_url: str | None = None
    workflow_type: WorkflowType
    # None marks a historical record created before evidence policies existed.
    evidence_policy: EvidencePolicy | None = None
    cookie_consent: CookieConsentRecord | None = None
    outcome: str | None = None
    status: ExecutionStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    steps: list[HistoryStep] = Field(default_factory=list)
    segments: list[HistorySegment] = Field(default_factory=list)
    preconditions: list[HistoryPrecondition] = Field(default_factory=list)
    setup_status: str | None = None
    cleanup_succeeded: bool | None = None
    cleanup_failures: list[HistoryCleanupFailure] = Field(default_factory=list)
    failed_step_ids: list[UUID] = Field(default_factory=list)
    blocked_step_ids: list[UUID] = Field(default_factory=list)
    executions: list[HistoryExecutionReference] = Field(default_factory=list)
    run_context_safe: dict[str, dict[str, Any]] = Field(default_factory=dict)
    trace_id: UUID | None = None

    @classmethod
    def from_completed_run(
        cls,
        test_case: TestCase,
        test_run: TestRun,
        *,
        workflow_type: WorkflowType,
        outcome: str | Enum | None = None,
        setup: SetupRunOutcome | None = None,
        cleanup: CleanupOutcome | None = None,
        trace_id: UUID | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> "RunHistoryRecord":
        if test_run.test_case_id != test_case.id:
            raise ValueError("TestRun does not belong to the supplied TestCase.")
        run_context = test_run.run_context
        setup_by_id = {
            item.precondition_id: item for item in setup.preconditions
        } if setup is not None else {}
        final_execution_by_step = {
            step_id: test_run.final_execution_for_step(step_id)
            for step_id in test_run.test_step_ids
        }
        ordered_steps = sorted(test_case.steps, key=lambda step: step.order)
        history_steps = [
            HistoryStep(
                id=step.id,
                order=step.order,
                name=_safe_text(step.name, run_context),
                description=_safe_text(step.description, run_context),
                expected=_safe_text(step.expected, run_context),
                status=(
                    ExecutionStatus.BLOCKED
                    if step.id in test_run.blocked_step_ids
                    else final_execution_by_step[step.id].status
                    if final_execution_by_step[step.id] is not None
                    else ExecutionStatus.NOT_ATTEMPTED if step.id in test_run.not_attempted_step_ids
                    else None
                ),
            )
            for step in ordered_steps
        ]
        preconditions = [
            HistoryPrecondition(
                id=item.id,
                order=item.order,
                description=_safe_text(item.description, run_context),
                provided_data_keys=list(item.provided_data_keys),
                status=(setup_by_id[item.id].status.value if item.id in setup_by_id else None),
                error=(
                    _safe_text(setup_by_id[item.id].error, run_context)
                    if item.id in setup_by_id and setup_by_id[item.id].error
                    else None
                ),
                error_type=(setup_by_id[item.id].error_type if item.id in setup_by_id else None),
            )
            for item in sorted(test_case.preconditions, key=lambda item: item.order)
        ]
        all_executions = list(test_run.executions)
        actual_start = started_at or min(
            (execution.started_at for execution in all_executions),
            default=test_run.started_at,
        )
        actual_finish = finished_at or test_run.finished_at
        duration_ms = (
            max(0, int((actual_finish - actual_start).total_seconds() * 1000))
            if actual_finish is not None
            else None
        )
        raw_outcome = outcome.value if isinstance(outcome, Enum) else outcome
        if raw_outcome is None:
            raw_outcome = (
                "PASSED"
                if test_run.status == ExecutionStatus.PASSED
                else "FAILED"
            )
        safe_context = _safe_context(run_context)
        return cls(
            run_id=test_run.id,
            test_case_public_id=test_case.public_id,
            test_case_id=test_case.id,
            test_case_name=_safe_text(test_case.name, run_context),
            test_case_description=_safe_text(test_case.description, run_context),
            base_url=_safe_text(test_case.base_url, run_context) if test_case.base_url else None,
            workflow_type=workflow_type,
            evidence_policy=current_evidence_policy(),
            cookie_consent=current_cookie_consent_record(),
            outcome=str(raw_outcome),
            status=test_run.status,
            started_at=actual_start,
            finished_at=actual_finish,
            duration_ms=duration_ms,
            steps=history_steps,
            segments=[
                HistorySegment(
                    id=segment.id,
                    order=segment.order,
                    base_url=(
                        _safe_text(segment.base_url, run_context)
                        if segment.base_url else None
                    ),
                    test_step_ids=[step.id for step in segment.steps],
                )
                for segment in sorted(test_case.segments, key=lambda item: item.order)
            ],
            preconditions=preconditions,
            setup_status=setup.status.value if setup is not None else None,
            cleanup_succeeded=cleanup.succeeded if cleanup is not None else None,
            cleanup_failures=[
                HistoryCleanupFailure(
                    label=_safe_text(failure.label, run_context),
                    error_type=failure.error_type,
                    message=_safe_text(failure.message, run_context),
                )
                for failure in cleanup.failures
            ] if cleanup is not None else [],
            failed_step_ids=[
                step.id for step in ordered_steps
                if (
                    final_execution_by_step[step.id] is not None
                    and final_execution_by_step[step.id].status == ExecutionStatus.FAILED
                )
            ],
            blocked_step_ids=list(test_run.blocked_step_ids),
            executions=[
                HistoryExecutionReference(
                    execution_id=execution.id,
                    test_step_id=execution.test_step_id,
                    test_plan_version_id=execution.test_plan_version_id,
                    status=execution.status,
                    classification=execution_classification(execution),
                    started_at=execution.started_at,
                    finished_at=execution.finished_at,
                    safe_error=_safe_text(execution.error, run_context),
                    safe_diagnostics=redact_diagnostic(
                        _safe_text(_runner_diagnostic(execution), run_context) or ""
                    ) or None,
                    safe_actual_result=_safe_value(execution_observation(execution), run_context),
                    evidence=[
                        HistoryEvidenceReference(
                            id=item.id,
                            execution_id=item.execution_id,
                            type=item.type.value,
                            name=_safe_text(item.path.replace("\\", "/").rsplit("/", 1)[-1], run_context) or "evidence",
                            description=_safe_text(item.description, run_context),
                            scope=item.scope,
                            event=item.event,
                        )
                        for item in execution.evidence
                    ],
                )
                for execution in all_executions
            ],
            run_context_safe=safe_context,
            trace_id=trace_id,
        )


class RunHistoryRepository(Protocol):
    def save(self, record: RunHistoryRecord) -> RunHistoryRecord: ...

    def get(self, run_id: UUID) -> RunHistoryRecord | None: ...

    def list_recent(self, limit: int = 50) -> list[RunHistoryRecord]: ...

    def list_for_test_case(
        self, test_case_id: UUID, limit: int = 50
    ) -> list[RunHistoryRecord]: ...


class InMemoryRunHistoryRepository:
    """Insertion-ordered run history for tests and short-lived applications."""

    def __init__(self) -> None:
        self._records: dict[UUID, RunHistoryRecord] = {}
        self._next_public_id = 1

    def save(self, record: RunHistoryRecord) -> RunHistoryRecord:
        if record.run_id in self._records:
            raise ValueError(f"Run {record.run_id} already exists in history.")
        public_id = record.public_id
        sequence = parse_run_public_id(public_id)
        if public_id is None:
            public_id = format_run_public_id(self._next_public_id)
            sequence = self._next_public_id
        elif sequence is None:
            raise ValueError("Run public ID must use the RUN-000001 format.")
        if any(item.public_id == public_id for item in self._records.values()):
            raise ValueError(f"Run public ID {public_id} already exists in history.")
        self._next_public_id = max(self._next_public_id, (sequence or 0) + 1)
        saved = record.model_copy(update={"public_id": public_id}, deep=True)
        self._records[record.run_id] = saved
        return saved.model_copy(deep=True)

    def get(self, run_id: UUID) -> RunHistoryRecord | None:
        record = self._records.get(run_id)
        return record.model_copy(deep=True) if record is not None else None

    def get_by_public_id(self, public_id: str) -> RunHistoryRecord | None:
        record = next(
            (item for item in self._records.values() if item.public_id == public_id),
            None,
        )
        return record.model_copy(deep=True) if record is not None else None

    def list_recent(self, limit: int = 50) -> list[RunHistoryRecord]:
        _validate_limit(limit)
        return [
            record.model_copy(deep=True)
            for record in list(self._records.values())[::-1][:limit]
        ]

    def list_for_test_case(
        self, test_case_id: UUID, limit: int = 50
    ) -> list[RunHistoryRecord]:
        _validate_limit(limit)
        return [
            record.model_copy(deep=True)
            for record in reversed(tuple(self._records.values()))
            if record.test_case_id == test_case_id
        ][:limit]


class RunHistoryDetail(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    record: RunHistoryRecord
    executions: dict[UUID, Execution] = Field(default_factory=dict)


class RunHistoryService:
    """Build safe records, persist them, and load execution-backed details."""

    def __init__(
        self,
        repository: RunHistoryRepository,
        execution_repository: ExecutionRepository | None = None,
        plan_store: PlanStore | None = None,
    ) -> None:
        self._repository = repository
        self._execution_repository = execution_repository
        self._plan_store = plan_store

    def record_completed_run(
        self,
        test_case: TestCase,
        test_run: TestRun,
        *,
        workflow_type: WorkflowType,
        outcome: str | Enum | None = None,
        setup: SetupRunOutcome | None = None,
        cleanup: CleanupOutcome | None = None,
        trace_id: UUID | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> RunHistoryRecord:
        record = RunHistoryRecord.from_completed_run(
            test_case,
            test_run,
            workflow_type=workflow_type,
            outcome=outcome,
            setup=setup,
            cleanup=cleanup,
            trace_id=trace_id,
            started_at=started_at,
            finished_at=finished_at,
        )
        if self._plan_store is not None:
            references = []
            for reference in record.executions:
                try:
                    version = self._plan_store.get_version(reference.test_plan_version_id)
                except Exception:
                    version = None
                if version is None:
                    references.append(reference)
                    continue
                references.append(reference.model_copy(update={
                    "plan_version_number": version.version,
                    "plan_version_origin": version.origin,
                }))
            record = record.model_copy(update={"executions": references})
        return self._repository.save(record)

    def get(self, run_id: UUID) -> RunHistoryRecord | None:
        return self._repository.get(run_id)

    def list_recent(self, limit: int = 50) -> list[RunHistoryRecord]:
        return self._repository.list_recent(limit)

    def list_for_test_case(
        self, test_case_id: UUID, limit: int = 50
    ) -> list[RunHistoryRecord]:
        return self._repository.list_for_test_case(test_case_id, limit)

    def get_detail(self, run_id: UUID) -> RunHistoryDetail | None:
        record = self._repository.get(run_id)
        if record is None:
            return None
        executions: dict[UUID, Execution] = {}
        if self._execution_repository is not None:
            for reference in record.executions:
                execution = self._execution_repository.get(reference.execution_id)
                if (
                    execution is not None
                    and execution.test_step_id == reference.test_step_id
                    and execution.test_plan_version_id == reference.test_plan_version_id
                ):
                    executions[reference.execution_id] = execution
        return RunHistoryDetail(record=record, executions=executions)


def _safe_context(run_context: RunContext) -> dict[str, dict[str, Any]]:
    try:
        raw = run_context.safe_dump()
    except (TypeError, ValueError):
        raw = {}
        for key, item in run_context.values.items():
            value: Any = "[REDACTED]" if item.sensitive else _safe_value(item.value, run_context)
            raw[key] = {"value": value, "sensitive": item.sensitive, "source": item.source}
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(raw):
        item = raw[key]
        safe_key = _safe_text(key, run_context) or "[REDACTED]"
        result[safe_key] = {
            "value": _safe_value(item.get("value"), run_context),
            "sensitive": bool(item.get("sensitive", False)),
            "source": _safe_text(item.get("source"), run_context),
        }
    return result


def _runner_diagnostic(execution: Execution) -> str | None:
    for step in (execution.runner_result or {}).get("steps", []):
        if isinstance(step, dict) and step.get("status") == "failed" and isinstance(step.get("error"), str):
            return step["error"]
    return execution.error


def _safe_text(value: str | None, run_context: RunContext) -> str | None:
    if value is None:
        return None
    result = value
    secrets = sorted(_sensitive_strings(run_context), key=len, reverse=True)
    for secret in secrets:
        result = result.replace(secret, "[REDACTED]")
    return redact_secrets(result)


def _safe_value(value: Any, run_context: RunContext) -> Any:
    if isinstance(value, str):
        return _safe_text(value, run_context)
    if isinstance(value, dict):
        return {
            str(_safe_text(str(key), run_context)): _safe_value(value[key], run_context)
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(child, run_context) for child in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    try:
        return _safe_text(str(value), run_context)
    except Exception:
        return "[UNSERIALIZABLE]"


def _sensitive_strings(run_context: RunContext) -> set[str]:
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


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Run history limit must be a non-negative integer.")
