import time
import inspect
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from qa_agent.execution_control import (
    check_cancelled,
    current_cancellation,
    is_cancelled,
    completion_boundary,
)
from qa_agent.browser_discovery import capture_discovery_result, extract_target_url
from qa_agent.assertion_grounding import validate_assertion_grounding
from qa_agent.expected_result_coverage import validate_expected_result_coverage
from qa_agent.generation_context import StepGenerationContext, validate_step_boundaries
from qa_agent.browser_runner import BrowserRunner
from qa_agent.execution_repository import ExecutionRepository, InMemoryExecutionRepository
from qa_agent.execution_trace import (
    ExecutionTrace,
    ExecutionTraceRecorder,
    TraceStatus,
    active_trace_recorder,
    elapsed_ms,
    record_safely,
)
from qa_agent.llm_usage import OP_DISCOVERY, llm_usage_scope
from qa_agent.diagnostic_mode import current_diagnostic_level, diagnostic_scope
from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    Execution,
    ExecutionStatus,
    FailurePolicy,
    PlanVersionOrigin,
    QATestPlan,
    LocatorIdentityEntry,
    TestCase,
    TestPlanVersion,
    TestRun,
    TestStep,
)
from qa_agent.plan_store import InMemoryPlanStore, PlanStore
from qa_agent.locator_recovery import (
    RecoveryStatus,
    identity_for_element,
    recover_locator,
)
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import GeneratedTestPlan, TestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError, validate_executable_plan
from qa_agent.reliability import AutomationReviewRequired, ReliabilityStopped
from qa_agent.discovery_fallback import DiscoveryFallback
from qa_agent.plan_execution import (
    PlanExecutionClassification,
    PlanExecutionOutcome,
    PlanExecutionPersistenceError,
    PlanExecutionService,
)
from qa_agent.run_context import RunContext
from qa_agent.run_history import RunHistoryService, WorkflowType
from qa_agent.execution_progress import ExecutionEventType, emit_progress_event
from qa_agent.llm.errors import NonRetryableLLMError, RetryableLLMError


class PipelineStageError(RuntimeError):
    """Add pipeline-stage context while preserving the original exception."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.trace: ExecutionTrace | None = None
        super().__init__(f"Pipeline stage '{stage}' failed: {message}")


@dataclass(frozen=True)
class PipelineResult:
    """Application result retaining each plan/version pair and its executions."""

    test_case: TestCase
    test_plans: list[GeneratedTestPlan]
    executions: list[Execution]
    trace: ExecutionTrace | None = field(default=None, compare=False)
    # Step ids a BLOCK_REST failure policy prevented from executing.
    blocked_step_ids: list[UUID] = field(default_factory=list)
    run_context: RunContext = field(default_factory=RunContext, compare=False)
    cancelled: bool = False
    test_run: TestRun = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "test_run",
            TestRun.from_test_case(
                self.test_case,
                self.executions,
                blocked_step_ids=self.blocked_step_ids,
                run_context=self.run_context,
                cancelled=self.cancelled,
            ),
        )

    def __iter__(self):
        """Allow existing callers to keep iterating over run results."""
        return iter(self.executions)

    def __len__(self) -> int:
        return len(self.executions)

    def __getitem__(self, index: int | slice) -> Execution | list[Execution]:
        return self.executions[index]


class QATestPipeline:
    """Coordinate decomposition, discovery, plan generation, and execution."""

    def __init__(
        self,
        decomposer: TestCaseDecomposer,
        plan_generator: TestPlanGenerator,
        discovery: Callable[[str], DiscoveryResult] = capture_discovery_result,
        runner: Callable[[QATestPlan], dict[str, Any]] | None = None,
        evidence_directory: str | None = None,
        plan_store: PlanStore | None = None,
        execution_repository: ExecutionRepository | None = None,
        discovery_fallback: DiscoveryFallback | None = None,
        run_history: RunHistoryService | None = None,
        candidate_review_approved: Callable[[TestCase], bool] | None = None,
    ) -> None:
        self._decomposer = decomposer
        self._plan_generator = plan_generator
        self._discovery = discovery
        self._discovery_fallback = discovery_fallback
        self._run_history = run_history
        self._candidate_review_approved = candidate_review_approved or (lambda _: False)
        self._runner = runner if runner is not None else BrowserRunner(evidence_directory)
        self._plan_store = plan_store if plan_store is not None else InMemoryPlanStore()
        self._execution_repository = (
            execution_repository
            if execution_repository is not None
            else InMemoryExecutionRepository()
        )
        self._plan_execution = PlanExecutionService(
            self._runner, self._execution_repository
        )

    def run(
        self,
        task: str,
        base_url: str | None = None,
        run_context: RunContext | None = None,
    ) -> PipelineResult:
        return self._run_entry(task, base_url, run_context, supplied_test_case=None)

    def run_test_case(
        self,
        test_case: TestCase,
        run_context: RunContext | None = None,
        *, regenerate: bool = False,
    ) -> PipelineResult:
        """Run a persisted canonical TestCase without decomposing its text again."""
        return self._run_entry(
            test_case.description,
            test_case.base_url,
            run_context,
            supplied_test_case=test_case,
            regenerate=regenerate,
        )

    def _run_entry(
        self,
        task: str,
        base_url: str | None,
        run_context: RunContext | None,
        *,
        supplied_test_case: TestCase | None,
        regenerate: bool = False,
    ) -> PipelineResult:
        """Run each decomposed step and return plans and executions together.

        The execution trace is an observability layer only: recording is
        best-effort and never changes pipeline behavior, fallback semantics,
        or recovery/regeneration logic.
        """
        active_run_context = run_context if run_context is not None else RunContext()
        trace = self._create_trace_recorder(task)
        case_runner = self._plan_execution.new_test_case_runner()
        cleanup_outcome = None
        level = current_diagnostic_level()
        try:
            if level is None:
                level = self._plan_generator.supervisor.effective_settings()[0].diagnostic_level
        except Exception:
            pass
        with diagnostic_scope(level), active_trace_recorder(trace):
            try:
                try:
                    result = self._run(
                        task,
                        base_url,
                        trace,
                        active_run_context,
                        supplied_test_case=supplied_test_case,
                        case_runner=case_runner,
                        regenerate=regenerate,
                    )
                except BaseException as error:
                    case_runner.close(primary_error=error)
                    raise
                else:
                    try:
                        case_runner.close()
                    except Exception as error:
                        if current_cancellation() is None or not current_cancellation().requested:
                            raise
                        from qa_agent.setup_orchestration import CleanupOutcome, CleanupFailure
                        cleanup_outcome = CleanupOutcome((CleanupFailure("Browser cleanup", type(error).__name__, "Browser cleanup did not complete normally."),))
            except PipelineStageError as error:
                error.trace = record_safely(
                    trace,
                    "finalize",
                    TraceStatus.ERROR,
                    error=error,
                    error_stage=error.stage,
                )
                raise
            except Exception as error:
                record_safely(trace, "finalize", TraceStatus.ERROR, error=error)
                raise
        with completion_boundary() as token:
            if token is not None and token.requested and not result.cancelled:
                result = replace(result, cancelled=True)
            if result.cancelled:
                finished_at = datetime.now(timezone.utc)
                # The step snapshot precedes owner-thread cleanup. Cancellation
                # can still win here; retain its identity and original findings.
                if result.trace is not None:
                    object.__setattr__(result, "trace", result.trace.model_copy(update={
                        "status": TraceStatus.ERROR,
                        "finished_at": finished_at,
                        "duration_ms": max(0, int((finished_at - result.trace.started_at).total_seconds() * 1000)),
                    }))
                object.__setattr__(result, "test_run", result.test_run.model_copy(update={"finished_at": finished_at}))
            if self._run_history is not None:
                self._run_history.record_completed_run(
                    result.test_case,
                    result.test_run,
                    workflow_type=WorkflowType.AUTOMATION,
                    outcome=_automation_run_outcome(result.test_run),
                    cleanup=cleanup_outcome,
                    trace_id=result.trace.trace_id if result.trace is not None else None,
                    started_at=result.trace.started_at if result.trace is not None else None,
                    finished_at=result.test_run.finished_at if result.cancelled else result.trace.finished_at if result.trace is not None else None,
                )
            if token is not None:
                token.sealed = True
        return result

    def _create_trace_recorder(self, task: str) -> ExecutionTraceRecorder:
        return ExecutionTraceRecorder(task=task)

    def _discover(self, target_url: str, case_runner=None, test_step=None) -> DiscoveryResult:
        """Observe the active page; isolate fresh probes for non-session adapters."""
        check_cancelled()
        if case_runner is not None and test_step is not None and case_runner.browser_session_started:
            observed = case_runner.capture_discovery(test_step)
            if observed is not None:
                return observed
        if case_runner is not None and case_runner.browser_session_started:
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="qa-browser-discovery",
            ) as worker:
                context = copy_context()
                return worker.submit(context.run, self._discovery, target_url).result()
        return self._discovery(target_url)

    def _run(
        self,
        task: str,
        base_url: str | None,
        trace: ExecutionTraceRecorder,
        run_context: RunContext,
        *,
        supplied_test_case: TestCase | None = None,
        case_runner=None,
        regenerate: bool = False,
    ) -> PipelineResult:
        decomposition_started = time.perf_counter()
        try:
            test_case = (
                supplied_test_case
                if supplied_test_case is not None
                else self._decomposer.decompose(task, base_url)
            )
        except Exception as error:
            raise PipelineStageError("decomposition", str(error)) from error
        if case_runner is not None:
            case_runner.bind(test_case)
        record_safely(
            trace,
            "record_decomposition",
            test_case,
            elapsed_ms(decomposition_started),
        )
        emit_progress_event(
            ExecutionEventType.AUTOMATION_PREPARATION_STARTED,
            message="Preparing automation for the TestCase.",
        )

        segment_by_step = {
            step.id: segment
            for segment in test_case.segments
            for step in segment.steps
        }
        try:
            target_url = (
                test_case.base_url
                or next((segment.base_url for segment in test_case.segments if segment.base_url), None)
                or extract_target_url(task)
            )
        except Exception as error:
            raise PipelineStageError("target resolution", str(error)) from error
        record_safely(trace, "record_target_url", target_url)

        executions: list[Execution] = []
        generated_plans: list[GeneratedTestPlan] = []
        blocked_step_ids: list[UUID] = []
        ordered_steps = sorted(test_case.steps, key=lambda step: step.order)
        cancelled = False
        for step_index, test_step in enumerate(ordered_steps):
            try:
                check_cancelled()
                step_target_url = segment_by_step[test_step.id].base_url or target_url
                record_safely(trace, "begin_step", test_step)
                try:
                    cached_version = self._plan_store.find(test_step.id)
                except Exception as error:
                    if is_cancelled(error):
                        raise
                    raise PipelineStageError(
                        f"plan lookup (step {test_step.order}: {test_step.name})",
                        str(error),
                    ) from error

                record_safely(
                    trace,
                    "record_cache",
                    cached_version is not None,
                    cached_version,
                )

                if cached_version is not None and not regenerate:
                    try:
                        cached_plan = self._plan_store.find_test_plan(test_step.id)
                        if cached_plan is None:
                            raise ValueError(
                                "Cached TestPlan is unavailable for the stored TestPlanVersion."
                            )
                        generated_plan = GeneratedTestPlan(
                            test_plan=cached_plan,
                            test_plan_version=cached_version,
                        )
                        if cached_plan.test_step_id != test_step.id:
                            raise ValueError("Cached TestPlan belongs to a different TestStep.")
                        if cached_version.test_plan_id != cached_plan.id:
                            raise ValueError("Cached TestPlanVersion belongs to a different TestPlan.")
                        cached_executable_plan = validate_executable_plan(
                            cached_version.qa_test_plan
                        )
                        generated_plan = GeneratedTestPlan(
                            test_plan=cached_plan,
                            test_plan_version=cached_version.model_copy(update={
                                "qa_test_plan": cached_executable_plan,
                            }),
                        )
                    except Exception as error:
                        if is_cancelled(error):
                            raise
                        emit_progress_event(
                            ExecutionEventType.PLAN_GENERATION_FAILED,
                            step=test_step,
                            classification="MISSING_AUTOMATION",
                            failure_code="SAVED_PLAN_INVALID",
                            prior_plan_exists=True,
                            new_plan_saved=False,
                            message="Saved automation could not be loaded.",
                        )
                        raise PipelineStageError(
                            f"plan lookup (step {test_step.order}: {test_step.name})",
                            str(error),
                        ) from error
                    emit_progress_event(
                        ExecutionEventType.PLAN_REUSED,
                        step=test_step,
                        plan_origin=cached_version.origin.value if cached_version.origin is not None else "SAVED",
                        plan_version=cached_version.version,
                        plan_version_id=cached_version.id,
                        message="Saved automation loaded.",
                    )
                else:
                    emit_progress_event(
                        ExecutionEventType.PLAN_GENERATION_STARTED,
                        step=test_step,
                        message="Preparing automation for this step.",
                    )
                    discovery_started = time.perf_counter()
                    try:
                        with llm_usage_scope(
                            operation_type=OP_DISCOVERY,
                            related_test_case_id=test_case.id,
                            related_test_case_public_id=test_case.public_id,
                        ):
                            discovery_result = self._discover(step_target_url, case_runner, test_step)
                        if (discovery_result.status != DiscoveryStatus.SUCCESS
                                and self._discovery_fallback is not None):
                            fallback_started = time.perf_counter()
                            try:
                                with llm_usage_scope(
                                    operation_type=OP_DISCOVERY,
                                    related_test_case_id=test_case.id,
                                    related_test_case_public_id=test_case.public_id,
                                ):
                                    check_cancelled()
                                    suggestions = self._discovery_fallback.discover(
                                        task, step_target_url, test_step, discovery_result
                                    )
                            except Exception as fallback_error:
                                # Evidence-first ordering (same principle as the
                                # P0-2 fix): the deterministic discovery already
                                # returned a result and the fallback attempt
                                # happened, so record both before the failure
                                # propagates. record_safely is best-effort and
                                # cannot mask the original error.
                                record_safely(
                                    trace,
                                    "record_discovery",
                                    discovery_result,
                                    elapsed_ms(discovery_started),
                                )
                                record_safely(
                                    trace,
                                    "record_discovery_fallback_failure",
                                    fallback_error,
                                    elapsed_ms(fallback_started),
                                )
                                raise
                            record_safely(
                                trace,
                                "record_discovery_fallback",
                                suggestions,
                                elapsed_ms(fallback_started),
                            )
                            discovery_result = discovery_result.model_copy(update={
                                "status": (DiscoveryStatus.PARTIAL if discovery_result.status == DiscoveryStatus.FAILED
                                           and (suggestions.navigation_paths or suggestions.direct_navigation_paths
                                                or suggestions.interactive_elements)
                                           else discovery_result.status),
                                "navigation_paths": discovery_result.navigation_paths + suggestions.navigation_paths,
                                "direct_navigation_paths": discovery_result.direct_navigation_paths + suggestions.direct_navigation_paths,
                                "interactive_elements": discovery_result.interactive_elements + suggestions.interactive_elements,
                                "warnings": discovery_result.warnings + suggestions.warnings,
                            })
                        record_safely(
                            trace,
                            "record_discovery",
                            discovery_result,
                            elapsed_ms(discovery_started),
                        )
                        if discovery_result.status == DiscoveryStatus.FAILED:
                            details = "; ".join(discovery_result.warnings) or "No details provided."
                            raise RuntimeError(f"Browser discovery returned FAILED: {details}")
                    except Exception as error:
                        if is_cancelled(error):
                            raise
                        emit_progress_event(
                            ExecutionEventType.PLAN_GENERATION_FAILED,
                            step=test_step,
                            classification="INFRASTRUCTURE_ERROR",
                            failure_code="DISCOVERY_FAILED",
                            prior_plan_exists=False,
                            new_plan_saved=False,
                            message="The browser could not inspect the target page.",
                        )
                        raise PipelineStageError(
                            f"discovery (step {test_step.order}: {test_step.name})",
                            str(error),
                        ) from error

                    generation_started = time.perf_counter()
                    segment = segment_by_step[test_step.id]
                    completed_ids = {item.test_step_id for item in executions if item.status == ExecutionStatus.PASSED}
                    segment_ids = {item.id for item in segment.steps}
                    step_context = StepGenerationContext(
                        segment_order=segment.order,
                        previous_steps=tuple(item for item in ordered_steps if item.id in segment_ids and item.order < test_step.order),
                        remaining_steps=tuple(item for item in ordered_steps if item.id in segment_ids and item.order > test_step.order),
                        completed_actions=tuple((action.action, action.parameters.get("selector"))
                            for prior in generated_plans if prior.test_plan.test_step_id in completed_ids & segment_ids
                            for action in prior.test_plan_version.qa_test_plan.steps),
                        state_preserved=case_runner is not None and case_runner.browser_session_started,
                    )
                    try:
                        with llm_usage_scope(
                            related_test_case_id=test_case.id,
                            related_test_case_public_id=test_case.public_id,
                        ):
                            check_cancelled()
                            generated_plan = self._plan_generator.generate_with_plan(
                                test_step,
                                discovery_result,
                                **({"existing_test_plan": self._plan_store.find_test_plan(test_step.id), "version_number": cached_version.version + 1} if cached_version is not None else {}),
                                **_requirement_context_kwargs(
                                    self._plan_generator, test_case.description
                                ),
                                **_optional_generation_kwargs(self._plan_generator, step_context=step_context),
                            )
                        generated_plan = _validate_generated_plan(
                            generated_plan,
                            test_step,
                            discovery_result=discovery_result,
                            requirement_context=test_case.description,
                            step_context=step_context,
                            expected_version=cached_version.version + 1 if cached_version is not None else 1,
                        )
                        generated_plan = _with_discovered_locator_identity(
                            generated_plan, discovery_result
                        )
                        generated_plan = _with_plan_origin(
                            generated_plan, PlanVersionOrigin.REGENERATED if cached_version is not None else PlanVersionOrigin.AI_GENERATED
                        )
                    except Exception as error:
                        if is_cancelled(error):
                            raise
                        failure_code, safe_reason = _generation_failure_details(error)
                        gate = getattr(error, "rejected_gate", None)
                        if getattr(error, "provider_response_succeeded", False) and gate:
                            safe_reason = (f"Provider responded successfully, but {gate.replace('_', ' ').title()} validation rejected the candidate. {safe_reason} "
                                           "Review the requirement and observed target; never substitute an unobserved selector. ")
                            if "cookie" in (test_step.description + test_step.expected).casefold():
                                safe_reason += "Use Leave unchanged and explicit visitor setup for cookie checks. "
                            safe_reason += "Review this TestStep and the value-free generation diagnostics."
                        emit_progress_event(
                            ExecutionEventType.PLAN_GENERATION_FAILED,
                            step=test_step,
                            classification="AUTOMATION_GENERATION_ERROR",
                            failure_code=failure_code,
                            prior_plan_exists=False,
                            new_plan_saved=False,
                            message=safe_reason,
                            validation_issues=(error.issues if isinstance(error, PlanValidationError) else ()),
                            reliability_operation_id=getattr(error, "reliability_operation_id", None),
                        )
                        raise PipelineStageError(
                            f"plan generation (step {test_step.order}: {test_step.name})",
                            str(error),
                        ) from error
                    record_safely(
                        trace,
                        "record_plan_generation",
                        generated_plan.test_plan.id,
                        generated_plan.test_plan_version.id,
                        generated_plan.test_plan_version.version,
                        len(generated_plan.test_plan_version.qa_test_plan.steps),
                        elapsed_ms(generation_started),
                    )

                    try:
                        with completion_boundary():
                            check_cancelled()
                            self._plan_store.save(
                                test_step.id,
                                generated_plan.test_plan_version,
                                test_plan=generated_plan.test_plan,
                            )
                    except Exception as error:
                        if is_cancelled(error):
                            raise
                        emit_progress_event(
                            ExecutionEventType.PLAN_GENERATION_FAILED,
                            step=test_step,
                            classification="INFRASTRUCTURE_ERROR",
                            failure_code="PLAN_PERSISTENCE_FAILED",
                            prior_plan_exists=False,
                            new_plan_saved=False,
                            message="Generated automation could not be saved.",
                        )
                        raise PipelineStageError(
                            f"plan save (step {test_step.order}: {test_step.name})",
                            str(error),
                        ) from error
                    emit_progress_event(
                        ExecutionEventType.PLAN_GENERATED,
                        step=test_step,
                        plan_origin=(generated_plan.test_plan_version.origin.value
                                     if generated_plan.test_plan_version.origin is not None
                                     else "AI_GENERATED"),
                        plan_version=generated_plan.test_plan_version.version,
                        plan_version_id=generated_plan.test_plan_version.id,
                        message="Automation generated for this step.",
                    )

                plan_version = generated_plan.test_plan_version
                generated_plans.append(generated_plan)
                supervisor = getattr(self._plan_generator, "supervisor", None)
                candidate = supervisor.requires_review(plan_version.id) if supervisor else None
                if (candidate is not None or plan_version.manual_recovery is not None) and not self._candidate_review_approved(test_case):
                    operation_id = candidate.id if candidate else plan_version.manual_recovery.operation_id
                    review = AutomationReviewRequired(operation_id)
                    emit_progress_event(ExecutionEventType.RELIABILITY_READY_FOR_REVIEW, step=test_step, message=str(review), reliability_operation_id=operation_id)
                    raise PipelineStageError("automation review", str(review)) from review
                execution_outcome = self._execute_plan(
                    test_step, plan_version, trace,
                    runner=case_runner.for_step(test_step) if case_runner is not None else None,
                )
                execution = execution_outcome.execution
                executions.append(execution)
                if current_cancellation() is not None and current_cancellation().requested:
                    cancelled = True
                    break

                if (
                    isinstance(execution.runner_result, dict)
                    and execution.runner_result.get("cookie_consent_requires_attention") is True
                ):
                    blocked_step_ids = [
                        step.id for step in ordered_steps[step_index + 1:]
                    ]
                    _emit_blocked_steps(ordered_steps[step_index + 1:])
                    break

                if execution_outcome.classification != PlanExecutionClassification.AUTOMATION_DRIFT:
                    # Explicit failure policy: a step whose final outcome is
                    # FAILED may still continue the TestCase (CONTINUE) or stop
                    # it entirely (BLOCK_REST); the decision is never implicit.
                    if _should_block_rest(test_step, execution):
                        blocked_step_ids = [
                            step.id for step in ordered_steps[step_index + 1:]
                        ]
                        _emit_blocked_steps(ordered_steps[step_index + 1:])
                        break
                    continue

                if supervisor is not None:
                    # Execution drift is outside generation recovery; saved versions stay pinned.
                    emit_progress_event(ExecutionEventType.PLAN_REPAIR_FAILED, step=test_step, message="Automation drift requires human review; no automatic locator repair was attempted.")
                    if _should_block_rest(test_step, execution):
                        blocked_step_ids = [step.id for step in ordered_steps[step_index + 1:]]
                        _emit_blocked_steps(ordered_steps[step_index + 1:])
                        break
                    continue

                rediscovery_started = time.perf_counter()
                try:
                    with llm_usage_scope(
                        operation_type=OP_DISCOVERY,
                        related_test_case_id=test_case.id,
                        related_test_case_public_id=test_case.public_id,
                    ):
                        rediscovery_result = self._discover(step_target_url, case_runner, test_step)
                except Exception as error:
                    if is_cancelled(error):
                        raise
                    # The rediscovery callable itself failed, so no result object
                    # exists; record a FAILED attempt anyway so an attempted
                    # rediscovery is never invisible in the trace. The pipeline
                    # still aborts before recovery/regeneration, and the AI
                    # fallback remains unused on this path.
                    record_safely(
                        trace,
                        "record_discovery",
                        DiscoveryResult(
                            status=DiscoveryStatus.FAILED,
                            url=step_target_url,
                            warnings=[str(error)],
                        ),
                        elapsed_ms(rediscovery_started),
                    )
                    raise PipelineStageError(
                        f"rediscovery (step {test_step.order}: {test_step.name})",
                        str(error),
                    ) from error

                # Record the attempt BEFORE the FAILED check so a rediscovery
                # that halts the run is still represented in the trace; a
                # successful rediscovery is recorded exactly once here, as before.
                record_safely(
                    trace,
                    "record_discovery",
                    rediscovery_result,
                    elapsed_ms(rediscovery_started),
                )
                if rediscovery_result.status == DiscoveryStatus.FAILED:
                    details = "; ".join(rediscovery_result.warnings) or "No details provided."
                    reason = f"Browser rediscovery returned FAILED: {details}"
                    raise PipelineStageError(
                        f"rediscovery (step {test_step.order}: {test_step.name})",
                        reason,
                    ) from RuntimeError(reason)

                failed_interaction = execution.planned_interaction(plan_version)
                recovery = None
                if failed_interaction is not None and failed_interaction.action in {"click", "check", "uncheck", "fill"}:
                    original_identity = next((
                        item for item in (plan_version.locator_identity or ())
                        if item.step_index == execution.planned_step_index
                    ), None)
                    recovery = recover_locator(
                        failed_interaction, rediscovery_result, original_identity
                    )
                record_safely(trace, "record_locator_recovery", recovery)

                if recovery is not None and recovery.status in {
                    RecoveryStatus.MATCHED_HIGH_CONFIDENCE,
                    RecoveryStatus.MATCHED_ACCEPTABLE,
                }:
                    candidate = recovery.candidate
                    if candidate is None:
                        raise PipelineStageError(
                            f"locator recovery (step {test_step.order}: {test_step.name})",
                            "MATCHED result did not contain a candidate.",
                        )
                    if execution.planned_step_index is None:
                        raise PipelineStageError(
                            f"locator recovery (step {test_step.order}: {test_step.name})",
                            "Recovery did not identify the planned interaction.",
                        )
                    try:
                        repaired_plan = _replace_interaction_selector(
                            plan_version.qa_test_plan,
                            execution.planned_step_index,
                            candidate.selector,
                        )
                        repaired_version = TestPlanVersion(
                            test_plan_id=generated_plan.test_plan.id,
                            version=plan_version.version + 1,
                            origin=PlanVersionOrigin.REPAIRED,
                            qa_test_plan=repaired_plan,
                            assertion_grounding=plan_version.assertion_grounding,
                            locator_identity=_replace_locator_identity(
                                plan_version.locator_identity,
                                execution.planned_step_index,
                                identity_for_element(
                                    execution.planned_step_index,
                                    candidate,
                                    failed_interaction.action,
                                ),
                            ),
                        )
                        repaired = GeneratedTestPlan(
                            test_plan=generated_plan.test_plan,
                            test_plan_version=repaired_version,
                        )
                        with completion_boundary():
                            check_cancelled()
                            self._plan_store.save(
                                test_step.id, repaired_version, test_plan=repaired.test_plan
                            )
                    except Exception as error:
                        if is_cancelled(error):
                            raise
                        raise PipelineStageError(
                            f"deterministic repair (step {test_step.order}: {test_step.name})",
                            str(error),
                        ) from error
                    generated_plans.append(repaired)
                    emit_progress_event(
                        ExecutionEventType.PLAN_REPAIR_SUCCEEDED,
                        step=test_step,
                        plan_origin=PlanVersionOrigin.REPAIRED.value,
                        plan_version=repaired_version.version,
                        plan_version_id=repaired_version.id,
                        message="Saved automation repaired using current page evidence.",
                    )
                    repaired_outcome = self._execute_plan(
                        test_step, repaired_version, trace,
                        runner=case_runner.for_step(test_step) if case_runner is not None else None,
                    )
                    repaired_execution = repaired_outcome.execution
                    executions.append(repaired_execution)
                    if _should_block_rest(test_step, repaired_execution):
                        blocked_step_ids = [
                            step.id for step in ordered_steps[step_index + 1:]
                        ]
                        _emit_blocked_steps(ordered_steps[step_index + 1:])
                        break
                    continue

                emit_progress_event(
                    ExecutionEventType.PLAN_REPAIR_FAILED,
                    step=test_step,
                    message=_safe_locator_recovery_message(recovery),
                )
                if _should_block_rest(test_step, execution):
                    blocked_step_ids = [
                        step.id for step in ordered_steps[step_index + 1:]
                    ]
                    _emit_blocked_steps(ordered_steps[step_index + 1:])
                    break
                # Ambiguous, conflicting, or missing evidence is automation drift.
                # Keep the original failed execution and leave the saved version
                # untouched for review; do not ask an LLM to choose a replacement.
                continue

            except Exception as error:
                if not is_cancelled(error):
                    raise
                cancelled = True
                break

        record_safely(trace, "record_blocked_steps", blocked_step_ids)
        run = TestRun.from_test_case(
            test_case,
            executions,
            blocked_step_ids=blocked_step_ids,
            run_context=run_context,
            cancelled=cancelled or (current_cancellation() is not None and current_cancellation().requested),
        )
        status = (
            TraceStatus.ERROR if run.cancelled else
            TraceStatus.FAILED
            if run.status == ExecutionStatus.FAILED
            else TraceStatus.PASSED
        )
        final_trace = record_safely(trace, "finalize", status)
        return PipelineResult(
            test_case=test_case,
            test_plans=generated_plans,
            executions=executions,
            blocked_step_ids=blocked_step_ids,
            trace=final_trace,
            run_context=run_context,
            cancelled=cancelled or (current_cancellation() is not None and current_cancellation().requested),
        )


    def _execute_plan(
        self,
        test_step: TestStep,
        plan_version: TestPlanVersion,
        trace: ExecutionTraceRecorder,
        *,
        runner=None,
    ) -> PlanExecutionOutcome:
        try:
            outcome = self._plan_execution.execute(
                test_step, plan_version, runner=runner
            )
        except PlanExecutionPersistenceError as error:
            raise PipelineStageError(
                f"execution repository (step {test_step.order}: {test_step.name}, "
                f"plan version {plan_version.version})",
                str(error),
            ) from error.__cause__

        # Trace remains a pipeline concern. Record immediately after the
        # service persists the execution, including before re-raising runner
        # errors, so every persisted attempt has exactly one trace record.
        record_safely(
            trace,
            "record_execution_attempt",
            outcome.execution,
            plan_version.version,
        )
        if outcome.error is not None:
            raise PipelineStageError(
                f"execution (step {test_step.order}: {test_step.name}, "
                f"plan version {plan_version.version})",
                str(outcome.error),
            ) from outcome.error
        return outcome


def _with_discovered_locator_identity(
    generated_plan: GeneratedTestPlan,
    discovery_result: DiscoveryResult,
) -> GeneratedTestPlan:
    """Attach identity evidence from the exact discovered interaction target."""
    version = generated_plan.test_plan_version
    candidates_by_selector: dict[str, list[Any]] = {}
    for element in discovery_result.interactive_elements:
        candidates_by_selector.setdefault(element.selector, []).append(element)

    entries = {item.step_index: item for item in (version.locator_identity or ())}
    for index, interaction in enumerate(version.qa_test_plan.steps):
        if interaction.action not in {"click", "check", "uncheck", "fill"}:
            continue
        selector = interaction.parameters.get("selector")
        matches = candidates_by_selector.get(selector, []) if isinstance(selector, str) else []
        if len(matches) == 1:
            entries[index] = identity_for_element(index, matches[0], interaction.action)

    identity = tuple(entries[index] for index in sorted(entries)) or None
    return GeneratedTestPlan(
        test_plan=generated_plan.test_plan,
        test_plan_version=version.model_copy(update={"locator_identity": identity}),
    )


def _replace_locator_identity(
    current: tuple[LocatorIdentityEntry, ...] | None,
    step_index: int,
    replacement: LocatorIdentityEntry,
) -> tuple[LocatorIdentityEntry, ...]:
    entries = {item.step_index: item for item in (current or ())}
    entries[step_index] = replacement
    return tuple(entries[index] for index in sorted(entries))


def _safe_locator_recovery_message(recovery) -> str:
    if recovery is not None and recovery.status == RecoveryStatus.AMBIGUOUS:
        return "Multiple controls match the saved identity. Manual attention is required."
    if recovery is not None and recovery.status == RecoveryStatus.REJECTED_CONFLICT:
        return "Similar controls were found, but their identity conflicts with the saved target. Manual attention is required."
    return "No safe locator replacement was found. Manual attention is required."


def _with_plan_origin(
    generated_plan: GeneratedTestPlan,
    origin: PlanVersionOrigin,
) -> GeneratedTestPlan:
    version = generated_plan.test_plan_version
    if version.origin in {origin, PlanVersionOrigin.REPAIRED}:
        return generated_plan
    return GeneratedTestPlan(
        test_plan=generated_plan.test_plan,
        test_plan_version=version.model_copy(update={"origin": origin}),
    )


def _validate_generated_plan(
    generated_plan: GeneratedTestPlan,
    test_step: TestStep,
    *,
    discovery_result: DiscoveryResult,
    requirement_context: str | None = None,
    step_context: StepGenerationContext | None = None,
    expected_version: int,
) -> GeneratedTestPlan:
    """Revalidate plan structure and ownership before any version is saved."""
    if not isinstance(generated_plan, GeneratedTestPlan):
        raise TypeError("Plan generator must return a GeneratedTestPlan.")
    if generated_plan.test_plan.test_step_id != test_step.id:
        raise ValueError("Generated TestPlan belongs to a different TestStep.")
    version = generated_plan.test_plan_version
    if version.test_plan_id != generated_plan.test_plan.id:
        raise ValueError("Generated TestPlanVersion belongs to a different TestPlan.")
    if version.version != expected_version:
        raise ValueError("Plan generator returned an unexpected TestPlanVersion number.")
    executable_plan = validate_executable_plan(version.qa_test_plan)
    grounding = validate_assertion_grounding(
        executable_plan,
        test_step,
        discovery_result,
        requirement_context=requirement_context,
    )
    validate_expected_result_coverage(test_step, executable_plan, discovery=discovery_result)
    validate_step_boundaries(executable_plan, test_step, discovery_result, step_context)
    return GeneratedTestPlan(
        test_plan=generated_plan.test_plan,
        test_plan_version=version.model_copy(update={
            "qa_test_plan": executable_plan,
            "assertion_grounding": grounding,
        }),
    )


def _requirement_context_kwargs(generator: TestPlanGenerator, context: str) -> dict[str, str]:
    """Pass the original requirement to capable generators without breaking older extensions."""
    return _optional_generation_kwargs(generator, requirement_context=context)


def _optional_generation_kwargs(generator, **context):
    try:
        parameters = inspect.signature(generator.generate_with_plan).parameters.values()
    except (TypeError, ValueError):
        return {}
    names = {parameter.name for parameter in parameters}
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    return {name: value for name, value in context.items() if accepts_kwargs or name in names}


def _generation_failure_details(error: Exception) -> tuple[str, str]:
    """Map internal generation exceptions to stable, safe progress diagnostics."""
    if isinstance(error, ReliabilityStopped):
        return error.category, str(error)
    if isinstance(error, PlanValidationError):
        reason = error.issues[0].message if error.issues else "Generated automation failed validation."
        return "PLAN_VALIDATION_FAILED", reason
    if isinstance(error, ValidationError):
        return "PLAN_VALIDATION_FAILED", "Generated automation failed validation."
    if isinstance(error, (RetryableLLMError, NonRetryableLLMError)):
        return "LLM_PROVIDER_FAILURE", "The automation provider could not return a usable plan."
    if isinstance(error, (TypeError, ValueError)):
        return "PLAN_VALIDATION_FAILED", "Generated automation failed validation."
    return "PLAN_GENERATION_FAILED", "No reliable executable action could be produced for this step."


def _automation_run_outcome(test_run: TestRun) -> str:
    """Retain the classification made against each attempt's exact plan."""
    from qa_agent.result_semantics import execution_classification
    if test_run.status == ExecutionStatus.PASSED:
        return "PASSED"

    if test_run.cancelled:
        return "CANCELLED"
    failed_executions = [
        execution
        for execution in test_run.final_executions
        if execution.status == ExecutionStatus.FAILED
    ]
    classifications = {
        execution_classification(execution)
        for execution in failed_executions
    }
    if PlanExecutionClassification.INFRASTRUCTURE_ERROR in classifications:
        return PlanExecutionClassification.INFRASTRUCTURE_ERROR.value
    if PlanExecutionClassification.AUTOMATION_DRIFT in classifications:
        return PlanExecutionClassification.AUTOMATION_DRIFT.value
    if PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR in classifications:
        return PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR.value
    if PlanExecutionClassification.PRODUCT_FAILURE in classifications:
        return PlanExecutionClassification.PRODUCT_FAILURE.value
    return "INCONCLUSIVE"


def _emit_blocked_steps(steps: list[TestStep]) -> None:
    for step in steps:
        emit_progress_event(
            ExecutionEventType.STEP_BLOCKED,
            step=step,
            status=ExecutionStatus.BLOCKED.value,
            message="Blocked by the preceding step's failure policy.",
        )


def _should_block_rest(test_step: TestStep, execution: Execution) -> bool:
    """Honor an explicit BLOCK_REST failure policy after a step ends FAILED."""
    return (
        test_step.failure_policy == FailurePolicy.BLOCK_REST
        and execution.status == ExecutionStatus.FAILED
    )


def _replace_interaction_selector(
    plan: QATestPlan, step_index: int | None, selector: str
) -> QATestPlan:
    """Copy a plan version with only the failed interaction's selector changed."""
    if step_index is None or step_index >= len(plan.steps):
        raise ValueError("Failed execution does not identify a planned interaction.")
    steps = list(plan.steps)
    interaction = steps[step_index]
    if interaction.action not in {"click", "check", "uncheck", "fill"}:
        raise ValueError("Failed planned interaction is not locator-recoverable.")
    parameters = dict(interaction.parameters)
    parameters["selector"] = selector
    steps[step_index] = interaction.model_copy(update={"parameters": parameters})
    return plan.model_copy(update={"steps": steps})
