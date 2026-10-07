import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from qa_agent.browser_discovery import capture_discovery_result, extract_target_url
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
from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    Execution,
    ExecutionStatus,
    FailurePolicy,
    PlanVersionOrigin,
    QATestPlan,
    TestCase,
    TestPlanVersion,
    TestRun,
    TestStep,
)
from qa_agent.plan_store import InMemoryPlanStore, PlanStore
from qa_agent.locator_recovery import RecoveryStatus, recover_locator
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import GeneratedTestPlan, TestPlanGenerator
from qa_agent.test_plan_validation import validate_executable_plan
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
    ) -> None:
        self._decomposer = decomposer
        self._plan_generator = plan_generator
        self._discovery = discovery
        self._discovery_fallback = discovery_fallback
        self._run_history = run_history
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
    ) -> PipelineResult:
        """Run a persisted canonical TestCase without decomposing its text again."""
        return self._run_entry(
            test_case.description,
            test_case.base_url,
            run_context,
            supplied_test_case=test_case,
        )

    def _run_entry(
        self,
        task: str,
        base_url: str | None,
        run_context: RunContext | None,
        *,
        supplied_test_case: TestCase | None,
    ) -> PipelineResult:
        """Run each decomposed step and return plans and executions together.

        The execution trace is an observability layer only: recording is
        best-effort and never changes pipeline behavior, fallback semantics,
        or recovery/regeneration logic.
        """
        active_run_context = run_context if run_context is not None else RunContext()
        trace = self._create_trace_recorder(task)
        with active_trace_recorder(trace):
            try:
                result = self._run(
                    task,
                    base_url,
                    trace,
                    active_run_context,
                    supplied_test_case=supplied_test_case,
                )
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
        if self._run_history is not None:
            self._run_history.record_completed_run(
                result.test_case,
                result.test_run,
                workflow_type=WorkflowType.AUTOMATION,
                outcome=_automation_run_outcome(result.test_run),
                trace_id=result.trace.trace_id if result.trace is not None else None,
                started_at=result.trace.started_at if result.trace is not None else None,
                finished_at=result.trace.finished_at if result.trace is not None else None,
            )
        return result

    def _create_trace_recorder(self, task: str) -> ExecutionTraceRecorder:
        return ExecutionTraceRecorder(task=task)

    def _run(
        self,
        task: str,
        base_url: str | None,
        trace: ExecutionTraceRecorder,
        run_context: RunContext,
        *,
        supplied_test_case: TestCase | None = None,
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

        try:
            target_url = test_case.base_url or extract_target_url(task)
        except Exception as error:
            raise PipelineStageError("target resolution", str(error)) from error
        record_safely(trace, "record_target_url", target_url)

        executions: list[Execution] = []
        generated_plans: list[GeneratedTestPlan] = []
        blocked_step_ids: list[UUID] = []
        ordered_steps = sorted(test_case.steps, key=lambda step: step.order)
        for step_index, test_step in enumerate(ordered_steps):
            record_safely(trace, "begin_step", test_step)
            try:
                cached_version = self._plan_store.find(test_step.id)
            except Exception as error:
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

            if cached_version is not None:
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
                    discovery_result = self._discovery(target_url)
                    if (discovery_result.status != DiscoveryStatus.SUCCESS
                            and self._discovery_fallback is not None):
                        fallback_started = time.perf_counter()
                        try:
                            suggestions = self._discovery_fallback.discover(
                                task, target_url, test_step, discovery_result
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
                try:
                    generated_plan = self._plan_generator.generate_with_plan(
                        test_step, discovery_result
                    )
                    generated_plan = _validate_generated_plan(
                        generated_plan,
                        test_step,
                        expected_version=1,
                    )
                    generated_plan = _with_plan_origin(
                        generated_plan, PlanVersionOrigin.AI_GENERATED
                    )
                except Exception as error:
                    failure_code, safe_reason = _generation_failure_details(error)
                    emit_progress_event(
                        ExecutionEventType.PLAN_GENERATION_FAILED,
                        step=test_step,
                        classification="AUTOMATION_GENERATION_ERROR",
                        failure_code=failure_code,
                        prior_plan_exists=False,
                        new_plan_saved=False,
                        message=safe_reason,
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
                    self._plan_store.save(
                        test_step.id,
                        generated_plan.test_plan_version,
                        test_plan=generated_plan.test_plan,
                    )
                except Exception as error:
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
                    message="Automation generated for this step.",
                )

            plan_version = generated_plan.test_plan_version
            generated_plans.append(generated_plan)
            execution_outcome = self._execute_plan(test_step, plan_version, trace)
            execution = execution_outcome.execution
            executions.append(execution)

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

            rediscovery_started = time.perf_counter()
            try:
                rediscovery_result = self._discovery(target_url)
            except Exception as error:
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
                        url=target_url,
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
            if failed_interaction is not None and failed_interaction.action in {"click", "fill"}:
                recovery = recover_locator(failed_interaction, rediscovery_result)
            record_safely(trace, "record_locator_recovery", recovery)

            if recovery is not None and recovery.status == RecoveryStatus.MATCHED:
                candidate = recovery.candidate
                if candidate is None:
                    raise PipelineStageError(
                        f"locator recovery (step {test_step.order}: {test_step.name})",
                        "MATCHED result did not contain a candidate.",
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
                    )
                    repaired = GeneratedTestPlan(
                        test_plan=generated_plan.test_plan,
                        test_plan_version=repaired_version,
                    )
                    self._plan_store.save(
                        test_step.id, repaired_version, test_plan=repaired.test_plan
                    )
                except Exception as error:
                    raise PipelineStageError(
                        f"deterministic repair (step {test_step.order}: {test_step.name})",
                        str(error),
                    ) from error
                generated_plans.append(repaired)
                repaired_outcome = self._execute_plan(
                    test_step, repaired_version, trace
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

            regeneration_started = time.perf_counter()
            emit_progress_event(
                ExecutionEventType.PLAN_GENERATION_STARTED,
                step=test_step,
                message="Generating updated automation for this step.",
            )
            try:
                regenerated_plan = self._plan_generator.generate_with_plan(
                    test_step,
                    rediscovery_result,
                    existing_test_plan=generated_plan.test_plan,
                    version_number=plan_version.version + 1,
                )
                regenerated_plan = _validate_generated_plan(
                    regenerated_plan,
                    test_step,
                    expected_version=plan_version.version + 1,
                )
                if regenerated_plan.test_plan.id != generated_plan.test_plan.id:
                    raise ValueError("Regeneration must reuse the existing TestPlan.")
                regenerated_plan = _with_plan_origin(
                    regenerated_plan, PlanVersionOrigin.REGENERATED
                )
            except Exception as error:
                failure_code, safe_reason = _generation_failure_details(error)
                emit_progress_event(
                    ExecutionEventType.PLAN_GENERATION_FAILED,
                    step=test_step,
                    classification="AUTOMATION_GENERATION_ERROR",
                    failure_code=failure_code,
                    prior_plan_exists=True,
                    new_plan_saved=False,
                    message=safe_reason,
                )
                raise PipelineStageError(
                    f"regeneration (step {test_step.order}: {test_step.name})",
                    str(error),
                ) from error
            record_safely(
                trace,
                "record_plan_generation",
                regenerated_plan.test_plan.id,
                regenerated_plan.test_plan_version.id,
                regenerated_plan.test_plan_version.version,
                len(regenerated_plan.test_plan_version.qa_test_plan.steps),
                elapsed_ms(regeneration_started),
            )
            record_safely(
                trace,
                "record_regeneration",
                plan_version.version,
                regenerated_plan.test_plan_version.version,
                "stale_ui_failure",
                execution.error,
            )

            generated_plans.append(regenerated_plan)
            try:
                self._plan_store.save(
                    test_step.id,
                    regenerated_plan.test_plan_version,
                    test_plan=regenerated_plan.test_plan,
                )
            except Exception as error:
                emit_progress_event(
                    ExecutionEventType.PLAN_GENERATION_FAILED,
                    step=test_step,
                    classification="INFRASTRUCTURE_ERROR",
                    failure_code="PLAN_PERSISTENCE_FAILED",
                    prior_plan_exists=True,
                    new_plan_saved=False,
                    message="Updated automation could not be saved.",
                )
                raise PipelineStageError(
                    f"plan save (step {test_step.order}: {test_step.name})",
                    str(error),
                ) from error
            emit_progress_event(
                ExecutionEventType.PLAN_GENERATED,
                step=test_step,
                message="Updated automation generated for this step.",
            )
            regenerated_outcome = self._execute_plan(
                test_step, regenerated_plan.test_plan_version, trace
            )
            regenerated_execution = regenerated_outcome.execution
            executions.append(regenerated_execution)
            if _should_block_rest(test_step, regenerated_execution):
                blocked_step_ids = [
                    step.id for step in ordered_steps[step_index + 1:]
                ]
                _emit_blocked_steps(ordered_steps[step_index + 1:])
                break

        record_safely(trace, "record_blocked_steps", blocked_step_ids)
        run = TestRun.from_test_case(
            test_case,
            executions,
            blocked_step_ids=blocked_step_ids,
            run_context=run_context,
        )
        status = (
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
        )


    def _execute_plan(
        self,
        test_step: TestStep,
        plan_version: TestPlanVersion,
        trace: ExecutionTraceRecorder,
    ) -> PlanExecutionOutcome:
        try:
            outcome = self._plan_execution.execute(test_step, plan_version)
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


def _with_plan_origin(
    generated_plan: GeneratedTestPlan,
    origin: PlanVersionOrigin,
) -> GeneratedTestPlan:
    version = generated_plan.test_plan_version
    if version.origin == origin:
        return generated_plan
    return GeneratedTestPlan(
        test_plan=generated_plan.test_plan,
        test_plan_version=version.model_copy(update={"origin": origin}),
    )


def _validate_generated_plan(
    generated_plan: GeneratedTestPlan,
    test_step: TestStep,
    *,
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
    return GeneratedTestPlan(
        test_plan=generated_plan.test_plan,
        test_plan_version=version.model_copy(update={"qa_test_plan": executable_plan}),
    )


def _generation_failure_details(error: Exception) -> tuple[str, str]:
    """Map internal generation exceptions to stable, safe progress diagnostics."""
    if isinstance(error, ValidationError):
        return "PLAN_VALIDATION_FAILED", "Generated automation failed validation."
    if isinstance(error, (RetryableLLMError, NonRetryableLLMError)):
        return "LLM_PROVIDER_FAILURE", "The automation provider could not return a usable plan."
    if isinstance(error, (TypeError, ValueError)):
        return "PLAN_VALIDATION_FAILED", "Generated automation failed validation."
    return "PLAN_GENERATION_FAILED", "No reliable executable action could be produced for this step."


def _automation_run_outcome(test_run: TestRun) -> str:
    """Preserve product-failure classification in completed Automation runs."""
    if test_run.status == ExecutionStatus.PASSED:
        return "PASSED"

    failed_executions = [
        execution
        for execution in test_run.final_executions
        if execution.status == ExecutionStatus.FAILED
    ]
    classifications = {
        PlanExecutionService._classify(execution, None)
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
    return "FAILED"


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
    if interaction.action not in {"click", "fill"}:
        raise ValueError("Failed planned interaction is not locator-recoverable.")
    parameters = dict(interaction.parameters)
    parameters["selector"] = selector
    steps[step_index] = interaction.model_copy(update={"parameters": parameters})
    return plan.model_copy(update={"steps": steps})
