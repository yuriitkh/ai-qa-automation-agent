"""Structured Execution Trace: provider-independent observability for one run.

The trace is a pure observability layer. It depends only on domain models and
the shared redaction helper -- never on Playwright, LLM SDKs, CLI, Web UI,
HTML renderers, or databases. Recording is best-effort: a trace failure must
never break the QA pipeline or change its behavior.

The active recorder is propagated through a ContextVar so the LLM router can
record provider attempts without any signature change on the generation,
discovery, or provider interfaces.
"""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.locator_recovery import LocatorRecoveryResult, RecoveryStatus
from qa_agent.models import (
    AIDiscoveryResult,
    DiscoveryResult,
    DiscoveryStatus,
    Evidence,
    Execution,
    ExecutionStatus,
    InteractiveElement,
    TestCase,
    TestPlanVersion,
    TestStep,
)
from qa_agent.redaction import redact_secrets, safe_failure_reason

logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = "1"

_current_recorder: ContextVar["ExecutionTraceRecorder | None"] = ContextVar(
    "execution_trace_recorder", default=None
)


class TraceStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"


class RequestKind(str, Enum):
    TEST_PLAN = "TEST_PLAN"
    DISCOVERY = "DISCOVERY"


class ProviderAttemptOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    NON_RETRYABLE_ERROR = "NON_RETRYABLE_ERROR"
    UNAVAILABLE = "UNAVAILABLE"


class TraceTotals(BaseModel):
    steps: int = 0
    provider_attempts: int = 0
    execution_attempts: int = 0
    locator_recoveries: int = 0
    regenerations: int = 0


class DecompositionStepTrace(BaseModel):
    id: UUID
    order: int
    name: str
    description: str
    expected: str


class DecompositionTrace(BaseModel):
    test_case_id: UUID
    name: str
    description: str
    base_url: str | None = None
    steps: list[DecompositionStepTrace] = Field(default_factory=list)
    duration_ms: int | None = None


class DiscoveryTrace(BaseModel):
    status: DiscoveryStatus
    url: str
    title: str = ""
    strategies_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    interactive_elements: list[InteractiveElement] = Field(default_factory=list)
    navigation_path_count: int = 0
    direct_navigation_path_count: int = 0
    navigation_sequence_count: int = 0
    duration_ms: int | None = None


class DiscoveryFallbackTrace(BaseModel):
    invoked: bool
    navigation_paths_added: int = 0
    direct_navigation_paths_added: int = 0
    interactive_elements_added: int = 0
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int | None = None


class ProviderAttemptTrace(BaseModel):
    provider_name: str
    request_kind: RequestKind
    outcome: ProviderAttemptOutcome
    model: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    duration_ms: int | None = None
    is_selected: bool = False
    # Future: populated once providers expose token usage.
    token_usage: dict[str, int] | None = None


class PlanCacheTrace(BaseModel):
    hit: bool
    version_id: UUID | None = None
    version_number: int | None = None


class PlanGenerationTrace(BaseModel):
    test_plan_id: UUID | None = None
    test_plan_version_id: UUID | None = None
    version_number: int | None = None
    steps_count: int = 0
    duration_ms: int | None = None


class RunnerStepTrace(BaseModel):
    action: str
    status: str
    error: str | None = None


class ExecutionAttemptTrace(BaseModel):
    execution_id: UUID
    plan_version_id: UUID
    plan_version_number: int | None = None
    status: ExecutionStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    planned_step_index: int | None = None
    actual_result: Any = None
    error: str | None = None
    runner_steps: list[RunnerStepTrace] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class LocatorRecoveryTrace(BaseModel):
    status: RecoveryStatus
    original_selector: str
    candidate_selector: str | None = None
    signals: list[str] = Field(default_factory=list)
    reason: str = ""


class RegenerationTrace(BaseModel):
    from_version: int
    to_version: int
    reason: str
    trigger_error: str | None = None


class StepTrace(BaseModel):
    test_step_id: UUID
    order: int
    name: str
    description: str
    expected: str
    plan_cache: PlanCacheTrace | None = None
    discoveries: list[DiscoveryTrace] = Field(default_factory=list)
    discovery_fallback: DiscoveryFallbackTrace | None = None
    provider_attempts: list[ProviderAttemptTrace] = Field(default_factory=list)
    plan_generation: PlanGenerationTrace | None = None
    execution_attempts: list[ExecutionAttemptTrace] = Field(default_factory=list)
    locator_recovery: LocatorRecoveryTrace | None = None
    regeneration: RegenerationTrace | None = None


class ExecutionTrace(BaseModel):
    """Immutable snapshot of one pipeline run, safe for any presenter."""

    model_config = ConfigDict(frozen=True)

    trace_id: UUID = Field(default_factory=uuid4)
    schema_version: str = Field(default=TRACE_SCHEMA_VERSION)
    task: str
    target_url: str | None = None
    status: TraceStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error: str | None = None
    error_stage: str | None = None
    decomposition: DecompositionTrace | None = None
    steps: list[StepTrace] = Field(default_factory=list)
    totals: TraceTotals = Field(default_factory=TraceTotals)


def elapsed_ms(started: float) -> int:
    """Monotonic millisecond duration since ``started``."""
    return max(0, int((time.perf_counter() - started) * 1000))


def record_safely(recorder: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a recorder method as best-effort; never propagate failures."""
    if recorder is None:
        return None
    try:
        return getattr(recorder, method)(*args, **kwargs)
    except Exception:
        logger.debug("Execution trace recording failed", exc_info=True)
        return None


def get_active_trace_recorder() -> "ExecutionTraceRecorder | None":
    """Return the recorder active for the current run, if any."""
    return _current_recorder.get()


@contextmanager
def active_trace_recorder(
    recorder: "ExecutionTraceRecorder | None",
) -> Iterator["ExecutionTraceRecorder | None"]:
    """Make ``recorder`` visible to the LLM router for the duration of a run."""
    token = _current_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _current_recorder.reset(token)


class ExecutionTraceRecorder:
    """Mutable collector that builds one :class:`ExecutionTrace` per run."""

    def __init__(self, task: str) -> None:
        self._task = redact_secrets(task)
        self._target_url: str | None = None
        self._started_at = datetime.now(timezone.utc)
        self._error: str | None = None
        self._error_stage: str | None = None
        self._decomposition: DecompositionTrace | None = None
        self._steps: list[StepTrace] = []
        self._current_step: StepTrace | None = None

    def record_target_url(self, target_url: str) -> None:
        self._target_url = redact_secrets(target_url)

    def record_decomposition(
        self, test_case: TestCase, duration_ms: int | None
    ) -> None:
        self._decomposition = DecompositionTrace(
            test_case_id=test_case.id,
            name=redact_secrets(test_case.name),
            description=redact_secrets(test_case.description),
            base_url=test_case.base_url,
            steps=[
                DecompositionStepTrace(
                    id=step.id,
                    order=step.order,
                    name=redact_secrets(step.name),
                    description=redact_secrets(step.description),
                    expected=redact_secrets(step.expected),
                )
                for step in sorted(test_case.steps, key=lambda item: item.order)
            ],
            duration_ms=duration_ms,
        )

    def begin_step(self, test_step: TestStep) -> None:
        self._current_step = StepTrace(
            test_step_id=test_step.id,
            order=test_step.order,
            name=redact_secrets(test_step.name),
            description=redact_secrets(test_step.description),
            expected=redact_secrets(test_step.expected),
        )
        self._steps.append(self._current_step)

    def record_cache(self, hit: bool, version: TestPlanVersion | None) -> None:
        step = self._current_step
        if step is None:
            return
        step.plan_cache = PlanCacheTrace(
            hit=hit,
            version_id=version.id if version is not None else None,
            version_number=version.version if version is not None else None,
        )

    def record_discovery(
        self, result: DiscoveryResult, duration_ms: int | None
    ) -> None:
        step = self._current_step
        if step is None:
            return
        step.discoveries.append(DiscoveryTrace(
            status=result.status,
            url=result.url,
            title=redact_secrets(result.title),
            strategies_used=list(result.strategies_used),
            warnings=[redact_secrets(warning) for warning in result.warnings],
            interactive_elements=list(result.interactive_elements),
            navigation_path_count=len(result.navigation_paths),
            direct_navigation_path_count=len(result.direct_navigation_paths),
            navigation_sequence_count=len(result.navigation),
            duration_ms=duration_ms,
        ))

    def record_discovery_fallback(
        self, suggestions: AIDiscoveryResult, duration_ms: int | None
    ) -> None:
        step = self._current_step
        if step is None:
            return
        step.discovery_fallback = DiscoveryFallbackTrace(
            invoked=True,
            navigation_paths_added=len(suggestions.navigation_paths),
            direct_navigation_paths_added=len(suggestions.direct_navigation_paths),
            interactive_elements_added=len(suggestions.interactive_elements),
            warnings=[redact_secrets(warning) for warning in suggestions.warnings],
            duration_ms=duration_ms,
        )

    def record_provider_attempt(
        self,
        provider_name: str,
        request_kind: RequestKind,
        outcome: ProviderAttemptOutcome,
        *,
        model: str | None = None,
        error: Exception | None = None,
        http_status: int | None = None,
        duration_ms: int | None = None,
        is_selected: bool = False,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        step = self._current_step
        if step is None:
            return
        step.provider_attempts.append(ProviderAttemptTrace(
            provider_name=provider_name,
            request_kind=request_kind,
            outcome=outcome,
            model=model,
            error_class=type(error).__name__ if error is not None else None,
            error_message=safe_failure_reason(error) if error is not None else None,
            http_status=http_status,
            duration_ms=duration_ms,
            is_selected=is_selected,
            token_usage=token_usage,
        ))

    def record_plan_generation(
        self,
        test_plan_id: UUID,
        test_plan_version_id: UUID,
        version_number: int,
        steps_count: int,
        duration_ms: int | None,
    ) -> None:
        step = self._current_step
        if step is None:
            return
        step.plan_generation = PlanGenerationTrace(
            test_plan_id=test_plan_id,
            test_plan_version_id=test_plan_version_id,
            version_number=version_number,
            steps_count=steps_count,
            duration_ms=duration_ms,
        )

    def record_execution_attempt(
        self, execution: Execution, plan_version_number: int | None = None
    ) -> None:
        step = self._current_step
        if step is None:
            return
        duration_ms = None
        if execution.finished_at is not None:
            duration_ms = max(
                0,
                int(
                    (execution.finished_at - execution.started_at).total_seconds()
                    * 1000
                ),
            )
        runner_result = (
            execution.runner_result
            if isinstance(execution.runner_result, dict)
            else {}
        )
        step.execution_attempts.append(ExecutionAttemptTrace(
            execution_id=execution.id,
            plan_version_id=execution.test_plan_version_id,
            plan_version_number=plan_version_number,
            status=execution.status,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            duration_ms=duration_ms,
            planned_step_index=execution.planned_step_index,
            actual_result=execution.actual_result,
            error=redact_secrets(execution.error) if execution.error else None,
            runner_steps=[
                RunnerStepTrace(
                    action=str(item.get("action", "")),
                    status=str(item.get("status", "")),
                    error=(
                        redact_secrets(str(item.get("error") or "")) or None
                    ),
                )
                for item in runner_result.get("steps", [])
                if isinstance(item, dict)
            ],
            evidence=list(execution.evidence),
        ))

    def record_locator_recovery(
        self, recovery: LocatorRecoveryResult | None
    ) -> None:
        step = self._current_step
        if step is None or recovery is None:
            return
        step.locator_recovery = LocatorRecoveryTrace(
            status=recovery.status,
            original_selector=recovery.original_selector,
            candidate_selector=(
                recovery.candidate.selector
                if recovery.candidate is not None
                else None
            ),
            signals=list(recovery.signals),
            reason=redact_secrets(recovery.reason),
        )

    def record_regeneration(
        self,
        from_version: int,
        to_version: int,
        reason: str,
        trigger_error: str | None = None,
    ) -> None:
        step = self._current_step
        if step is None:
            return
        step.regeneration = RegenerationTrace(
            from_version=from_version,
            to_version=to_version,
            reason=redact_secrets(reason),
            trigger_error=redact_secrets(trigger_error) if trigger_error else None,
        )

    def finalize(
        self,
        status: TraceStatus,
        error: Exception | None = None,
        error_stage: str | None = None,
    ) -> ExecutionTrace:
        finished_at = datetime.now(timezone.utc)
        if error is not None:
            self._error = safe_failure_reason(error)
            self._error_stage = error_stage
        return ExecutionTrace(
            task=self._task,
            target_url=self._target_url,
            status=status,
            started_at=self._started_at,
            finished_at=finished_at,
            duration_ms=max(
                0,
                int((finished_at - self._started_at).total_seconds() * 1000),
            ),
            error=self._error,
            error_stage=self._error_stage,
            decomposition=self._decomposition,
            steps=self._steps,
            totals=TraceTotals(
                steps=len(self._steps),
                provider_attempts=sum(
                    len(step.provider_attempts) for step in self._steps
                ),
                execution_attempts=sum(
                    len(step.execution_attempts) for step in self._steps
                ),
                locator_recoveries=sum(
                    1 for step in self._steps if step.locator_recovery is not None
                ),
                regenerations=sum(
                    1 for step in self._steps if step.regeneration is not None
                ),
            ),
        ).model_copy(deep=True)
