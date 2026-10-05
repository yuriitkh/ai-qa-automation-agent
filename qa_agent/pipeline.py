import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

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
    Evidence,
    EvidenceType,
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
from qa_agent.discovery_fallback import DiscoveryFallback


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
    test_run: TestRun = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "test_run",
            TestRun.from_test_case(self.test_case, self.executions),
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
    ) -> None:
        self._decomposer = decomposer
        self._plan_generator = plan_generator
        self._discovery = discovery
        self._discovery_fallback = discovery_fallback
        self._runner = runner if runner is not None else BrowserRunner(evidence_directory)
        self._plan_store = plan_store if plan_store is not None else InMemoryPlanStore()
        self._execution_repository = (
            execution_repository
            if execution_repository is not None
            else InMemoryExecutionRepository()
        )

    def run(self, task: str, base_url: str | None = None) -> PipelineResult:
        """Run each decomposed step and return plans and executions together.

        The execution trace is an observability layer only: recording is
        best-effort and never changes pipeline behavior, fallback semantics,
        or recovery/regeneration logic.
        """
        trace = self._create_trace_recorder(task)
        with active_trace_recorder(trace):
            try:
                return self._run(task, base_url, trace)
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

    def _create_trace_recorder(self, task: str) -> ExecutionTraceRecorder:
        return ExecutionTraceRecorder(task=task)

    def _run(
        self,
        task: str,
        base_url: str | None,
        trace: ExecutionTraceRecorder,
    ) -> PipelineResult:
        decomposition_started = time.perf_counter()
        try:
            test_case = self._decomposer.decompose(task, base_url)
        except Exception as error:
            raise PipelineStageError("decomposition", str(error)) from error
        record_safely(
            trace,
            "record_decomposition",
            test_case,
            elapsed_ms(decomposition_started),
        )

        try:
            target_url = test_case.base_url or extract_target_url(task)
        except Exception as error:
            raise PipelineStageError("target resolution", str(error)) from error
        record_safely(trace, "record_target_url", target_url)

        executions: list[Execution] = []
        generated_plans: list[GeneratedTestPlan] = []
        ordered_steps = sorted(test_case.steps, key=lambda step: step.order)
        for test_step in ordered_steps:
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
                except Exception as error:
                    raise PipelineStageError(
                        f"plan lookup (step {test_step.order}: {test_step.name})",
                        str(error),
                    ) from error
            else:
                discovery_started = time.perf_counter()
                try:
                    discovery_result = self._discovery(target_url)
                    if (discovery_result.status != DiscoveryStatus.SUCCESS
                            and self._discovery_fallback is not None):
                        fallback_started = time.perf_counter()
                        suggestions = self._discovery_fallback.discover(
                            task, target_url, test_step, discovery_result
                        )
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
                    raise PipelineStageError(
                        f"discovery (step {test_step.order}: {test_step.name})",
                        str(error),
                    ) from error

                generation_started = time.perf_counter()
                try:
                    generated_plan = self._plan_generator.generate_with_plan(
                        test_step, discovery_result
                    )
                    if generated_plan.test_plan.test_step_id != test_step.id:
                        raise ValueError("Generated TestPlan belongs to a different TestStep.")
                except Exception as error:
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
                    raise PipelineStageError(
                        f"plan save (step {test_step.order}: {test_step.name})",
                        str(error),
                    ) from error

            plan_version = generated_plan.test_plan_version
            generated_plans.append(generated_plan)
            execution = self._execute_plan(test_step, plan_version)
            executions.append(execution)
            record_safely(
                trace,
                "record_execution_attempt",
                execution,
                plan_version.version,
            )

            if execution.status != ExecutionStatus.FAILED or not _is_stale_ui_failure(
                execution.runner_result
            ):
                continue

            rediscovery_started = time.perf_counter()
            try:
                rediscovery_result = self._discovery(target_url)
                if rediscovery_result.status == DiscoveryStatus.FAILED:
                    details = "; ".join(rediscovery_result.warnings) or "No details provided."
                    raise RuntimeError(f"Browser rediscovery returned FAILED: {details}")
            except Exception as error:
                raise PipelineStageError(
                    f"rediscovery (step {test_step.order}: {test_step.name})",
                    str(error),
                ) from error
            record_safely(
                trace,
                "record_discovery",
                rediscovery_result,
                elapsed_ms(rediscovery_started),
            )

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
                repaired_execution = self._execute_plan(test_step, repaired_version)
                executions.append(repaired_execution)
                record_safely(
                    trace,
                    "record_execution_attempt",
                    repaired_execution,
                    repaired_version.version,
                )
                continue

            regeneration_started = time.perf_counter()
            try:
                regenerated_plan = self._plan_generator.generate_with_plan(
                    test_step,
                    rediscovery_result,
                    existing_test_plan=generated_plan.test_plan,
                    version_number=plan_version.version + 1,
                )
                if regenerated_plan.test_plan.id != generated_plan.test_plan.id:
                    raise ValueError("Regeneration must reuse the existing TestPlan.")
                if regenerated_plan.test_plan.test_step_id != test_step.id:
                    raise ValueError("Regenerated TestPlan belongs to a different TestStep.")
                if regenerated_plan.test_plan_version.version != plan_version.version + 1:
                    raise ValueError("Regeneration returned an unexpected TestPlanVersion number.")
            except Exception as error:
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
                raise PipelineStageError(
                    f"plan save (step {test_step.order}: {test_step.name})",
                    str(error),
                ) from error
            regenerated_execution = self._execute_plan(
                test_step, regenerated_plan.test_plan_version
            )
            executions.append(regenerated_execution)
            record_safely(
                trace,
                "record_execution_attempt",
                regenerated_execution,
                regenerated_plan.test_plan_version.version,
            )

        run = TestRun.from_test_case(test_case, executions)
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
            trace=final_trace,
        )

    def _execute_plan(self, test_step: TestStep, plan_version: TestPlanVersion) -> Execution:
        started_at = datetime.now(timezone.utc)
        runner_result: dict[str, Any] | None = None
        try:
            runner_output = self._runner(plan_version.qa_test_plan)
            if not isinstance(runner_output, dict):
                raise TypeError("Browser runner must return a result dictionary.")
            runner_result = runner_output
            runner_status = runner_result.get("status")
            if runner_status not in {"passed", "failed"}:
                raise ValueError(
                    f"Browser runner returned unsupported status {runner_status!r}."
                )
        except Exception as error:
            failed_execution = Execution(
                test_step_id=test_step.id,
                test_plan_version_id=plan_version.id,
                planned_step_index=self._failed_plan_step_index(
                    plan_version, runner_result
                ),
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                actual_result="failed",
                error=str(error),
                runner_result=runner_result,
            )
            self._save_execution(failed_execution, test_step, plan_version)
            raise PipelineStageError(
                f"execution (step {test_step.order}: {test_step.name}, "
                f"plan version {plan_version.version})",
                str(error),
            ) from error

        status = (
            ExecutionStatus.PASSED
            if runner_status == "passed"
            else ExecutionStatus.FAILED
        )
        execution_id = uuid4()
        execution = Execution(
            id=execution_id,
            test_step_id=test_step.id,
            test_plan_version_id=plan_version.id,
            planned_step_index=(
                self._failed_plan_step_index(plan_version, runner_result)
                if status == ExecutionStatus.FAILED
                else None
            ),
            status=status,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            actual_result=runner_status,
            error=self._runner_error(runner_result) if status == ExecutionStatus.FAILED else None,
            runner_result=runner_result,
            evidence=self._runner_evidence(runner_result, status, execution_id),
        )
        self._save_execution(execution, test_step, plan_version)
        return execution

    def _save_execution(
        self,
        execution: Execution,
        test_step: TestStep,
        plan_version: TestPlanVersion,
    ) -> None:
        try:
            self._execution_repository.save(execution)
        except Exception as error:
            raise PipelineStageError(
                f"execution repository (step {test_step.order}: {test_step.name}, "
                f"plan version {plan_version.version})",
                str(error),
            ) from error

    @staticmethod
    def _runner_error(runner_result: dict[str, Any]) -> str:
        for step_result in runner_result.get("steps", []):
            if isinstance(step_result, dict) and step_result.get("status") == "failed":
                return str(step_result.get("error") or "Browser plan failed.")
        return "Browser plan failed."

    @staticmethod
    def _failed_plan_step_index(
        plan_version: TestPlanVersion,
        runner_result: dict[str, Any] | None,
    ) -> int | None:
        if not isinstance(runner_result, dict):
            return None
        for index, step_result in enumerate(runner_result.get("steps", [])):
            if isinstance(step_result, dict) and step_result.get("status") == "failed":
                return index if index < len(plan_version.qa_test_plan.steps) else None
        return None

    @staticmethod
    def _runner_evidence(
        runner_result: dict[str, Any], status: ExecutionStatus, execution_id
    ) -> tuple[Evidence, ...]:
        if status != ExecutionStatus.FAILED:
            return ()
        items = runner_result.get("evidence", [])
        if not isinstance(items, list):
            return ()
        evidence = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != EvidenceType.SCREENSHOT.value:
                continue
            path = item.get("path")
            if not isinstance(path, str) or not path:
                continue
            evidence_id = uuid4()
            item["evidence_id"] = str(evidence_id)
            evidence.append(Evidence(
                id=evidence_id,
                execution_id=execution_id,
                type=EvidenceType.SCREENSHOT,
                path=path,
                description=item.get("description"),
            ))
        return tuple(evidence)


def _is_stale_ui_failure(runner_result: dict[str, Any] | None) -> bool:
    """Recognize locator/UI mismatches only; assertions and infra do not qualify."""
    if not isinstance(runner_result, dict):
        return False

    for step_result in runner_result.get("steps", []):
        if not isinstance(step_result, dict) or step_result.get("status") != "failed":
            continue
        if step_result.get("action") not in {"click", "fill"}:
            continue

        error = str(step_result.get("error") or "").casefold()
        selector_missing = "selector" in error and (
            "not found" in error or "was not found" in error
        )
        element_unavailable = any(
            marker in error
            for marker in (
                "resolved to hidden",
                "resolved to 0 elements",
                "element is not visible",
                "waiting for element to be visible",
                "not attached to the dom",
                "detached from the dom",
            )
        )
        locator_actionability_timeout = (
            "timeout" in error
            and "waiting for locator" in error
            and (
                "could not click element matching selector" in error
                or "could not fill element matching selector" in error
            )
        )
        if selector_missing or element_unavailable or locator_actionability_timeout:
            return True
    return False


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
