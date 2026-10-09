"""Bounded generation recovery and safe, persistent reliability decisions."""

from __future__ import annotations

from contextvars import ContextVar, copy_context
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Event, RLock, Thread
from typing import Any, Callable, Literal
from uuid import UUID, uuid4
import sqlite3
import time

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from qa_agent.llm.errors import NonRetryableLLMError, RetryableLLMError, category_for_error
from qa_agent.llm_usage import current_llm_usage_context, OP_GENERATE_AUTOMATION_PLAN
from qa_agent.redaction import redact_diagnostic
from qa_agent.test_plan_validation import PlanValidationError

OVERALL_TIMEOUT_SECONDS = 60.0
PROVIDER_TIMEOUT_SECONDS = 30.0
MAX_BACKOFF_SECONDS = 5.0
QUALITY_GATES = ("schema_and_actions", "locator_identity", "assertion_grounding", "expected_result_coverage")


class ReliabilitySettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    additional_retries: StrictBool = False
    provider_fallback: StrictBool = True
    automatic_plan_repair: StrictBool = False
    max_total_attempts: int = Field(default=2, strict=True, ge=1, le=3)


class ReliabilityAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    action: Literal["INITIAL_GENERATION", "PROVIDER_RETRY", "PROVIDER_FALLBACK", "TARGETED_REPAIR"]
    reason: str
    provider: str | None = None
    model: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    status: Literal["RUNNING", "QUALITY_PENDING", "ACCEPTED", "REJECTED", "CANCELLED"] = "RUNNING"
    error_category: str | None = None
    quality_gates: dict[str, str] = Field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None


class ReliabilityDecision(BaseModel):
    timestamp: datetime
    attempt_index: int | None = None
    action: str
    category: str | None = None
    reason: str


class ReliabilityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID = Field(default_factory=uuid4)
    test_case_id: UUID | None = None
    test_step_id: UUID
    settings: ReliabilitySettings
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_ms: int | None = None
    outcome: Literal["RUNNING", "READY_FOR_REVIEW", "NEEDS_ATTENTION", "CANCELLED", "INTERRUPTED"] = "RUNNING"
    attempts: list[ReliabilityAttempt] = Field(default_factory=list)
    decisions: list[ReliabilityDecision] = Field(default_factory=list)
    candidate_version_id: UUID | None = None
    repaired: bool = False
    final_category: str | None = None


class InMemoryReliabilityRepository:
    def __init__(self):
        self._settings = ReliabilitySettings()
        self._records: dict[UUID, ReliabilityRecord] = {}
        self._lock = RLock()

    def settings(self) -> ReliabilitySettings:
        with self._lock:
            return self._settings

    def save_settings(self, settings: ReliabilitySettings) -> None:
        with self._lock:
            self._settings = ReliabilitySettings.model_validate(settings.model_dump())

    def save(self, record: ReliabilityRecord) -> None:
        with self._lock:
            self._records[record.id] = record.model_copy(deep=True)

    def get(self, operation_id: UUID) -> ReliabilityRecord | None:
        with self._lock:
            record = self._records.get(operation_id)
            return record.model_copy(deep=True) if record else None

    def list_records(self) -> list[ReliabilityRecord]:
        with self._lock:
            return sorted((item.model_copy(deep=True) for item in self._records.values()), key=lambda item: item.started_at, reverse=True)


class SQLiteReliabilityRepository:
    """Additive tables; existing provider, plan, run and usage rows are untouched."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS automation_reliability_settings (id INTEGER PRIMARY KEY CHECK(id = 1), settings_json TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS automation_reliability_operations (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, record_json TEXT NOT NULL)")
            # Preserve room for an established three-provider chain on first migration.
            providers = set()
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            for table in ("provider_settings", "custom_provider_settings"):
                if table in tables:
                    providers.update(row[0] for row in connection.execute(f"SELECT provider_id FROM {table} WHERE enabled = 1"))
            defaults = ReliabilitySettings(max_total_attempts=3 if len(providers) >= 3 else 2)
            connection.execute("INSERT OR IGNORE INTO automation_reliability_settings VALUES (1, ?)", (defaults.model_dump_json(),))

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=1)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def settings(self) -> ReliabilitySettings:
        with self._connect() as connection:
            row = connection.execute("SELECT settings_json FROM automation_reliability_settings WHERE id = 1").fetchone()
        return ReliabilitySettings.model_validate_json(row[0])

    def save_settings(self, settings: ReliabilitySettings) -> None:
        validated = ReliabilitySettings.model_validate(settings.model_dump())
        with self._connect() as connection:
            connection.execute("UPDATE automation_reliability_settings SET settings_json = ? WHERE id = 1", (validated.model_dump_json(),))

    def save(self, record: ReliabilityRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO automation_reliability_operations VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET record_json = excluded.record_json",
                (str(record.id), record.started_at.isoformat(), record.model_dump_json()),
            )

    def get(self, operation_id: UUID) -> ReliabilityRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT record_json FROM automation_reliability_operations WHERE id = ?", (str(operation_id),)).fetchone()
        return ReliabilityRecord.model_validate_json(row[0]) if row else None

    def list_records(self) -> list[ReliabilityRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT record_json FROM automation_reliability_operations ORDER BY started_at DESC, id").fetchall()
        return [ReliabilityRecord.model_validate_json(row[0]) for row in rows]


class ReliabilityStopped(RuntimeError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(safe_reason(category))


class AutomationReviewRequired(RuntimeError):
    """A saved repaired candidate must pass explicit human approval before execution."""

    def __init__(self, operation_id: UUID | None = None):
        self.operation_id = operation_id
        super().__init__("Repaired automation is saved for review. Review it and approve the saved automation for Validation before execution.")


def classify_failure(error: BaseException) -> str:
    if isinstance(error, ReliabilityStopped):
        return error.category
    if isinstance(error, PlanValidationError):
        codes = {issue.code for issue in error.issues}
        if "UNGROUNDED_ASSERTION" in codes:
            return "ASSERTION_NOT_GROUNDED"
        if "REPAIR_CHANGED_SEMANTICS" in codes:
            return "UNSAFE_REPAIR"
        if codes & {"DISCOVERY_SELECTOR_MISMATCH", "ACTION_TARGET_MISMATCH", "MISSING_LOCATOR"}:
            return "LOCATOR_IDENTITY_UNCERTAIN"
        if "EXPECTED_RESULT_NOT_COVERED" in codes:
            return "EXPECTED_RESULT_NOT_COVERED"
        if "UNSUPPORTED_ACTION" in codes:
            return "UNSUPPORTED_ACTION"
        return "PLAN_SCHEMA_INVALID"
    if isinstance(error, (RetryableLLMError, NonRetryableLLMError)):
        return category_for_error(error)
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "TIMEOUT" if isinstance(error, TimeoutError) else "PROVIDER_UNAVAILABLE"
    if isinstance(error, (OSError, sqlite3.Error)):
        return "INFRASTRUCTURE_ERROR"
    return "UNKNOWN_ERROR"


def safe_reason(category: str) -> str:
    return {
        "TIMEOUT": "The provider request timed out.",
        "RATE_LIMIT": "The provider rate limit prevented this request.",
        "PROVIDER_UNAVAILABLE": "The configured provider is temporarily unavailable.",
        "AUTH_ERROR": "The provider credentials need attention; this provider will not be retried.",
        "MODEL_NOT_FOUND": "The configured model is unavailable.",
        "INVALID_REQUEST": "The provider rejected the request configuration.",
        "SCHEMA_ERROR": "The provider rejected the structured-output contract.",
        "INVALID_RESPONSE": "The provider did not return a usable structured response.",
        "PLAN_SCHEMA_INVALID": "The candidate does not meet the executable plan contract.",
        "UNSUPPORTED_ACTION": "The candidate contains an unsupported action.",
        "EXPECTED_RESULT_NOT_COVERED": "The candidate does not verify the original expected result.",
        "ASSERTION_NOT_GROUNDED": "The assertion is not grounded in requirements or deterministic evidence. Human review is required.",
        "LOCATOR_IDENTITY_UNCERTAIN": "The control identity is uncertain. Human review is required.",
        "INSUFFICIENT_TESTCASE_REQUIREMENTS": "Clarify the TestCase requirements before generating automation.",
        "ATTEMPT_LIMIT": "The total generation attempt limit was reached.",
        "TIME_LIMIT": "The generation time limit was reached; no further recovery will start.",
        "CANCELLED": "Generation was cancelled; no further recovery will start.",
        "NO_PROVIDER": "No eligible configured provider is available.",
        "NO_FALLBACK": "No eligible configured fallback provider remains.",
        "INFRASTRUCTURE_ERROR": "Local infrastructure could not safely complete generation.",
        "PRODUCT_FAILURE": "Confirmed product failures cannot be repaired by generation recovery.",
        "UNSAFE_REPAIR": "Repair changed an existing action or assertion. Human review is required.",
    }.get(category, "Generation stopped because this failure does not permit safe automatic recovery.")


_ACTIVE_OPERATION: ContextVar[Any] = ContextVar("automation_reliability_operation", default=None)


def current_reliability_operation():
    return _ACTIVE_OPERATION.get()


def provider_timeout(default: float) -> float:
    operation = current_reliability_operation()
    return max(0.001, min(default, operation.remaining_seconds, PROVIDER_TIMEOUT_SECONDS)) if operation else default


def _safe_metadata(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    from qa_agent.execution_progress import get_active_execution_progress
    progress = get_active_execution_progress()
    value = progress.safe_text(value) if progress else redact_diagnostic(value)
    return " ".join(value.split())[:180]


class ReliabilityOperation:
    def __init__(self, supervisor, record: ReliabilityRecord, cancellation: Event):
        self.supervisor, self.record, self.cancellation = supervisor, record, cancellation
        self.started = time.monotonic()
        self.deadline = self.started + supervisor.timeout_seconds
        self.request_action = "INITIAL_GENERATION"
        self.request_reason = "Initial generation from the unchanged TestStep requirement."
        self.last_provider = None
        self.usage = None
        self.cost = None

    @property
    def remaining_seconds(self):
        return max(0.0, self.deadline - time.monotonic())

    def check(self):
        self.ensure_active()
        if len(self.record.attempts) >= self.record.settings.max_total_attempts:
            raise ReliabilityStopped("ATTEMPT_LIMIT")

    def ensure_active(self):
        if self.cancellation.is_set():
            raise ReliabilityStopped("CANCELLED")
        if self.remaining_seconds <= 0:
            raise ReliabilityStopped("TIME_LIMIT")

    def decision(self, action: str, category: str | None, reason: str):
        self.record.decisions.append(ReliabilityDecision(
            timestamp=datetime.now(timezone.utc), attempt_index=len(self.record.attempts) or None,
            action=action, category=category, reason=reason,
        ))
        self.supervisor.repository.save(self.record)

    def emit(self, kind: str, message: str):
        from qa_agent.execution_progress import ExecutionEventType, emit_progress_event
        emit_progress_event(getattr(ExecutionEventType, kind), step=self.supervisor.active_step(self.record.id), message=message, reliability_operation_id=self.record.id)

    def invoke(self, provider, call: Callable, *, action: str, reason: str):
        self.check()
        attempt = ReliabilityAttempt(
            index=len(self.record.attempts) + 1, action=action, reason=reason,
            provider=_safe_metadata(getattr(provider, "display_name", None) or getattr(provider, "name", None) or type(provider).__name__) if provider else None,
            model=_safe_metadata(getattr(provider, "model", None) or getattr(provider, "_model", None)) if provider else None,
            started_at=datetime.now(timezone.utc), quality_gates=dict.fromkeys(QUALITY_GATES, "NOT_RUN"),
        )
        if not self.supervisor._capacity.acquire(blocking=False):
            raise ReliabilityStopped("INFRASTRUCTURE_ERROR")
        self.record.attempts.append(attempt)
        try:
            self.supervisor.repository.save(self.record)
        except Exception:
            self.supervisor._capacity.release()
            raise
        self.last_provider = provider
        self.usage = self.cost = None
        done = Event()
        returned = {}
        context = copy_context()
        started = time.monotonic()

        def active_call():
            self.ensure_active()
            return call()

        def worker():
            try:
                returned["value"] = context.run(active_call)
            except BaseException as error:
                returned["error"] = error
            finally:
                self.supervisor._capacity.release()
                done.set()

        try:
            try:
                Thread(target=worker, name="automation-provider-request", daemon=True).start()
            except Exception:
                self.supervisor._capacity.release()
                raise
            request_deadline = min(self.deadline, started + PROVIDER_TIMEOUT_SECONDS)
            while not done.wait(min(0.05, max(0.001, request_deadline - time.monotonic()))):
                if self.cancellation.is_set():
                    raise ReliabilityStopped("CANCELLED")
                if time.monotonic() >= request_deadline:
                    raise ReliabilityStopped("TIME_LIMIT")
            if self.cancellation.is_set():
                raise ReliabilityStopped("CANCELLED")
            if self.remaining_seconds <= 0:
                raise ReliabilityStopped("TIME_LIMIT")
            if "error" in returned:
                raise returned["error"]
        except BaseException as error:
            attempt.status = "CANCELLED" if classify_failure(error) == "CANCELLED" else "REJECTED"
            attempt.error_category = classify_failure(error)
            raise
        else:
            attempt.status = "QUALITY_PENDING"
            return returned["value"]
        finally:
            attempt.finished_at = datetime.now(timezone.utc)
            attempt.duration_ms = max(0, int((time.monotonic() - started) * 1000))
            if done.is_set() and self.usage is not None:
                attempt.input_tokens, attempt.output_tokens = self.usage.input_tokens, self.usage.output_tokens
                attempt.estimated_cost_usd = self.cost
            self.supervisor.repository.save(self.record)

    def capture_usage(self, usage, cost):
        self.usage, self.cost = usage, cost

    def provider_recovery(self, error: BaseException, *, has_fallback: bool) -> str:
        category = classify_failure(error)
        if isinstance(error, NonRetryableLLMError) or category in {"CANCELLED", "TIME_LIMIT", "INFRASTRUCTURE_ERROR", "UNKNOWN_ERROR", "PRODUCT_FAILURE"}:
            return "STOP"
        self.check()
        settings = self.record.settings
        if settings.additional_retries and category in {"TIMEOUT", "RATE_LIMIT", "PROVIDER_UNAVAILABLE"}:
            delay = min(MAX_BACKOFF_SECONDS, 0.5 * 2 ** (len(self.record.attempts) - 1))
            retry_after = getattr(error, "retry_after_seconds", None)
            if isinstance(retry_after, int) and retry_after > 0:
                delay = max(delay, retry_after)
            if delay <= MAX_BACKOFF_SECONDS and delay < self.remaining_seconds:
                self.decision("PROVIDER_RETRY", category, safe_reason(category))
                self.emit("RELIABILITY_RETRY", f"{safe_reason(category)} Retrying provider request for attempt {len(self.record.attempts) + 1} after a bounded delay.")
                if self.cancellation.wait(delay):
                    raise ReliabilityStopped("CANCELLED")
                self.check()
                return "PROVIDER_RETRY"
        if settings.provider_fallback and has_fallback and isinstance(error, RetryableLLMError):
            self.decision("PROVIDER_FALLBACK", category, safe_reason(category))
            self.emit("RELIABILITY_FALLBACK", f"{safe_reason(category)} Switching to the next eligible configured provider for attempt {len(self.record.attempts) + 1}.")
            return "PROVIDER_FALLBACK"
        reason = "No eligible configured fallback provider remains. " if settings.provider_fallback and not has_fallback else "No permitted provider recovery remains. "
        self.decision("STOP", category, reason + safe_reason(category))
        return "STOP"


class AutomationReliabilitySupervisor:
    def __init__(self, repository=None, *, timeout_seconds: float = OVERALL_TIMEOUT_SECONDS, max_inflight: int = 4):
        if timeout_seconds <= 0 or timeout_seconds > OVERALL_TIMEOUT_SECONDS:
            raise ValueError("Generation time limit must be positive and no more than 60 seconds.")
        self.repository = repository or InMemoryReliabilityRepository()
        self.timeout_seconds = timeout_seconds
        self._capacity = BoundedSemaphore(max_inflight)
        self._active: dict[UUID, tuple[Event, Any]] = {}
        self._lock = RLock()

    def active_step(self, operation_id):
        with self._lock:
            return self._active.get(operation_id, (None, None))[1]

    def cancel(self, operation_id: UUID) -> bool:
        with self._lock:
            active = self._active.get(operation_id)
            if active:
                active[0].set()
            return active is not None

    def cancel_all(self) -> None:
        with self._lock:
            for cancellation, _ in self._active.values():
                cancellation.set()

    def requires_review(self, version_id: UUID) -> ReliabilityRecord | None:
        return next((record for record in self.repository.list_records() if record.candidate_version_id == version_id and record.repaired), None)

    def reconcile_interrupted(self):
        for record in self.repository.list_records():
            if record.outcome == "RUNNING":
                record.outcome, record.final_category = "INTERRUPTED", "UNKNOWN_ERROR"
                record.decisions.append(ReliabilityDecision(timestamp=datetime.now(timezone.utc), action="STOP", reason="An application restart interrupted this operation; completion time and outcome are unknown."))
                self.repository.save(record)

    def generate(self, step, request: Callable[[str | None, Exception | None], Any], validate: Callable, *, cancellation: Event | None = None):
        usage_context = current_llm_usage_context(OP_GENERATE_AUTOMATION_PLAN)
        record = ReliabilityRecord(
            test_step_id=step.id, test_case_id=UUID(usage_context.related_test_case_id) if usage_context.related_test_case_id else None,
            settings=self.repository.settings(), started_at=datetime.now(timezone.utc),
        )
        self.repository.save(record)
        operation = ReliabilityOperation(self, record, cancellation or Event())
        with self._lock:
            self._active[record.id] = (operation.cancellation, step)
        token = _ACTIVE_OPERATION.set(operation)
        repair_error = None
        try:
            if not step.description.strip() or not step.expected.strip():
                raise ReliabilityStopped("INSUFFICIENT_TESTCASE_REQUIREMENTS")
            while True:
                operation.check()
                operation.emit("RELIABILITY_GENERATING" if repair_error is None else "RELIABILITY_REPAIR", "Generating automation." if repair_error is None else "Repairing an unapproved candidate while preserving the original requirement.")
                try:
                    previous_count = len(record.attempts)
                    value = request("TARGETED_REPAIR" if repair_error else None, repair_error)
                    if len(record.attempts) == previous_count:
                        raise ReliabilityStopped("UNKNOWN_ERROR")
                    operation.emit("RELIABILITY_CHECKING", "Checking automation quality using the mandatory gates.")
                    validated = validate(value)
                    operation.ensure_active()
                except Exception as error:
                    category = classify_failure(error)
                    if record.attempts and record.attempts[-1].status == "QUALITY_PENDING":
                        attempt = record.attempts[-1]
                        attempt.status, attempt.error_category = ("CANCELLED" if category == "CANCELLED" else "REJECTED"), category
                        failed_gate = gate_for_category(category)
                        if category == "UNSAFE_REPAIR":
                            attempt.quality_gates = dict.fromkeys(QUALITY_GATES, "PASSED") | {"repair_integrity": "FAILED"}
                        elif failed_gate:
                            for name in QUALITY_GATES:
                                attempt.quality_gates[name] = "FAILED" if name == failed_gate else "PASSED" if QUALITY_GATES.index(name) < QUALITY_GATES.index(failed_gate) else "NOT_RUN"
                    self.repository.save(record)
                    if self._repair_allowed(operation, error, step):
                        operation.check()
                        operation.decision("TARGETED_REPAIR", category, safe_reason(category))
                        record.repaired = True
                        operation.request_action, operation.request_reason = "TARGETED_REPAIR", safe_reason(category)
                        repair_error = error
                        continue
                    operation.decision("HUMAN_REVIEW_REQUIRED" if category in {"ASSERTION_NOT_GROUNDED", "LOCATOR_IDENTITY_UNCERTAIN", "INSUFFICIENT_TESTCASE_REQUIREMENTS", "UNSAFE_REPAIR"} else "STOP", category, safe_reason(category))
                    raise
                record.attempts[-1].status = "ACCEPTED"
                record.attempts[-1].quality_gates = dict.fromkeys(QUALITY_GATES, "PASSED")
                record.outcome = "READY_FOR_REVIEW"
                operation.decision("HUMAN_REVIEW_REQUIRED", None, "Quality gates accepted the candidate; generation is not Browser Validation or a product PASS.")
                operation.emit("RELIABILITY_READY_FOR_REVIEW", "The automation candidate passed quality checks and is ready for review.")
                return validated, record.id, record.repaired
        except BaseException as error:
            category = "CANCELLED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else classify_failure(error)
            record.final_category = category
            record.outcome = "CANCELLED" if category == "CANCELLED" else "NEEDS_ATTENTION"
            if not record.decisions or record.decisions[-1].category != category:
                operation.decision("STOP", category, safe_reason(category))
            operation.emit("RELIABILITY_NEEDS_ATTENTION", safe_reason(category))
            raise
        finally:
            record.finished_at = datetime.now(timezone.utc)
            record.elapsed_ms = max(0, int((time.monotonic() - operation.started) * 1000))
            try:
                self.repository.save(record)
            finally:
                _ACTIVE_OPERATION.reset(token)
                with self._lock:
                    self._active.pop(record.id, None)

    @staticmethod
    def _repair_allowed(operation, error, step):
        if not operation.record.settings.automatic_plan_repair or operation.record.repaired:
            return False
        category = classify_failure(error)
        if category in {"PLAN_SCHEMA_INVALID", "INVALID_RESPONSE"}:
            if isinstance(error, PlanValidationError) and any(issue.code in {"MISSING_INPUT_VALUE", "MISSING_EXPECTED_VALUE", "MISSING_OPTION_LABEL"} for issue in error.issues):
                return False
            return True
        if category == "EXPECTED_RESULT_NOT_COVERED":
            from qa_agent.expected_result_coverage import expected_result_coverage, ExpectedResultCoverageStatus
            return expected_result_coverage(step, None).status != ExpectedResultCoverageStatus.UNKNOWN
        return False

    def attach_candidate(self, operation_id: UUID, version_id: UUID):
        record = self.repository.get(operation_id)
        if record:
            record.candidate_version_id = version_id
            self.repository.save(record)

    def statistics(self) -> dict[str, Any]:
        records = self.repository.list_records()
        completed = [record for record in records if record.finished_at is not None]
        recovered = [record for record in completed if len(record.attempts) > 1]
        attempts = [attempt for record in records for attempt in record.attempts]
        rejections, providers = {}, {}
        for attempt in attempts:
            if "FAILED" in attempt.quality_gates.values():
                category = attempt.error_category or "UNKNOWN_ERROR"
                rejections[category] = rejections.get(category, 0) + 1
            key = (attempt.provider, attempt.model)
            providers[key] = providers.get(key, 0) + 1
        return {
            "generation_operations": len(records), "completed_operations": len(completed),
            "first_attempt_success_rate": sum(record.outcome == "READY_FOR_REVIEW" and len(record.attempts) == 1 for record in completed) / len(completed) if completed else None,
            "recovery_attempts": sum(max(0, len(record.attempts) - 1) for record in records),
            "fallback_count": sum(attempt.action == "PROVIDER_FALLBACK" for attempt in attempts),
            "recovery_success_rate": sum(record.outcome == "READY_FOR_REVIEW" for record in recovered) / len(recovered) if recovered else None,
            "quality_gate_rejections": rejections,
            "average_attempts": sum(len(record.attempts) for record in completed) / len(completed) if completed else None,
            "average_generation_duration_ms": sum(record.elapsed_ms for record in completed) / len(completed) if completed else None,
            "provider_model_attempts": [{"provider": key[0], "model": key[1], "count": count} for key, count in providers.items()],
            "input_tokens": sum(attempt.input_tokens for attempt in attempts if attempt.input_tokens is not None) if any(attempt.input_tokens is not None for attempt in attempts) else None,
            "output_tokens": sum(attempt.output_tokens for attempt in attempts if attempt.output_tokens is not None) if any(attempt.output_tokens is not None for attempt in attempts) else None,
            "usage_known_attempts": sum(attempt.input_tokens is not None and attempt.output_tokens is not None for attempt in attempts),
            "estimated_cost_usd": sum(attempt.estimated_cost_usd for attempt in attempts if attempt.estimated_cost_usd is not None) if any(attempt.estimated_cost_usd is not None for attempt in attempts) else None,
            "cost_known_attempts": sum(attempt.estimated_cost_usd is not None for attempt in attempts),
            "total_attempts": len(attempts),
        }


def gate_for_category(category):
    return {
        "PLAN_SCHEMA_INVALID": "schema_and_actions", "UNSUPPORTED_ACTION": "schema_and_actions",
        "LOCATOR_IDENTITY_UNCERTAIN": "locator_identity", "ASSERTION_NOT_GROUNDED": "assertion_grounding",
        "EXPECTED_RESULT_NOT_COVERED": "expected_result_coverage",
    }.get(category)
