import io
import os
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock, patch

from qa_agent.execution_trace import (
    ExecutionTrace,
    ExecutionTraceRecorder,
    ProviderAttemptOutcome,
    RequestKind,
    RecoveryStatus,
    SegmentTrace,
    TraceStatus,
    active_trace_recorder,
)
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import NonRetryableLLMError, RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.models import (
    AIDiscoveryResult,
    DiscoveryResult,
    DiscoveryStatus,
    ExecutionSegment,
    ExecutionStatus,
    FailurePolicy,
    InteractiveElement,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.pipeline import PipelineResult, PipelineStageError, QATestPipeline
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.test_plan_generator import GeneratedTestPlan, LLMTestPlanGenerator


def _make_step(
    order: int,
    failure_policy: FailurePolicy = FailurePolicy.CONTINUE,
) -> DomainTestStep:
    return DomainTestStep(
        name=f"Step {order}",
        description=f"Perform action {order}",
        expected=f"Action {order} is completed.",
        order=order,
        failure_policy=failure_policy,
    )


def _test_step() -> DomainTestStep:
    return DomainTestStep(
        name="Open the example homepage",
        description="Open the example homepage.",
        expected="The example homepage is opened.",
        order=0,
    )


def _plan() -> QATestPlan:
    return QATestPlan(
        url="https://example.com/",
        steps=[QATestStep(action="assert_page_loaded")],
    )


class _FakeDecomposer:
    def __init__(
        self,
        result: DomainTestCase | None = None,
        events: list[tuple[Any, ...]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.error = error

    def decompose(self, task: str, base_url: str | None = None) -> DomainTestCase:
        self.events.append(("decomposer", task, base_url))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class _FakeGenerator:
    def __init__(self, events: list[tuple[Any, ...]] | None = None) -> None:
        self.events = events if events is not None else []
        self.calls: list[tuple[DomainTestStep, DiscoveryResult, int]] = []
        self.generated: list[GeneratedTestPlan] = []
        self.versions: list[DomainTestPlanVersion] = []

    def generate_with_plan(
        self,
        test_step: DomainTestStep,
        discovery_result: DiscoveryResult,
        *,
        existing_test_plan: DomainTestPlan | None = None,
        version_number: int = 1,
    ) -> GeneratedTestPlan:
        self.events.append(("generator", test_step, discovery_result))
        self.calls.append((test_step, discovery_result, version_number))
        plan = existing_test_plan or DomainTestPlan(
            test_step_id=test_step.id, name=test_step.name
        )
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=version_number,
            qa_test_plan=QATestPlan(
                url=discovery_result.url,
                steps=[QATestStep(action="assert_page_loaded")],
            ),
        )
        self.versions.append(version)
        generated = GeneratedTestPlan(test_plan=plan, test_plan_version=version)
        self.generated.append(generated)
        return generated


class _StubProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        *,
        plan: QATestPlan | None = None,
        error: Exception | None = None,
        available: bool = True,
        model: str | None = None,
    ) -> None:
        self.name = name
        self.plan = plan if plan is not None else _plan()
        self.error = error
        self.available = available
        self.model = model if model is not None else f"{name}-model"

    @property
    def is_available(self) -> bool:
        return self.available

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        if self.error is not None:
            raise self.error
        return self.plan

    def create_discovery(
        self, task: str, target_url: str, page_snapshot: str
    ) -> AIDiscoveryResult:
        if self.error is not None:
            raise self.error
        return AIDiscoveryResult()


class ExecutionTracePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.steps = [_make_step(1), _make_step(0)]
        self.test_case = DomainTestCase(
            name="Example flow",
            description="Open the example homepage.",
            base_url="https://example.com/",
            steps=self.steps,
        )
        self.discovery_result = DiscoveryResult(
            status=DiscoveryStatus.PARTIAL,
            url="https://example.com/",
            title="Example",
        )
        self.decomposer = _FakeDecomposer(self.test_case, self.events)
        self.generator = _FakeGenerator(self.events)
        self.runner_calls: list[QATestPlan] = []

        def discover(url: str) -> DiscoveryResult:
            self.events.append(("discovery", url))
            return self.discovery_result

        def run(plan: QATestPlan) -> dict[str, Any]:
            self.events.append(("runner", plan))
            self.runner_calls.append(plan)
            return {
                "status": "passed",
                "url": plan.url,
                "steps": [
                    {"action": "assert_page_loaded", "status": "passed", "error": ""}
                ],
            }

        self.discover = discover
        self.run_plan = run

    def _pipeline(self, **overrides: Any) -> QATestPipeline:
        params: dict[str, Any] = dict(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=self.run_plan,
        )
        params.update(overrides)
        return QATestPipeline(**params)

    def test_trace_is_created_for_successful_run(self) -> None:
        result = self._pipeline().run("Open the example homepage.")

        trace = result.trace
        self.assertIsInstance(trace, ExecutionTrace)
        self.assertEqual(trace.task, "Open the example homepage.")
        self.assertEqual(trace.target_url, "https://example.com/")
        self.assertEqual(trace.status, TraceStatus.PASSED)
        self.assertIsNotNone(trace.started_at)
        self.assertIsNotNone(trace.finished_at)
        self.assertIsNotNone(trace.duration_ms)
        self.assertIsNone(trace.error)
        self.assertIsNone(trace.error_stage)
        self.assertEqual(trace.schema_version, "2")

        decomposition = trace.decomposition
        assert decomposition is not None
        self.assertEqual(decomposition.test_case_id, self.test_case.id)
        self.assertEqual(decomposition.name, "Example flow")
        self.assertEqual(decomposition.base_url, "https://example.com/")
        self.assertEqual(
            [step.order for step in decomposition.steps], [0, 1]
        )
        self.assertIsNotNone(decomposition.duration_ms)

        self.assertEqual(len(trace.steps), 2)
        self.assertEqual(len(trace.segments), 1)
        segment_trace = trace.segments[0]
        self.assertEqual(segment_trace.segment_id, self.test_case.segments[0].id)
        self.assertEqual(segment_trace.order, 0)
        self.assertEqual(segment_trace.base_url, self.test_case.base_url)
        self.assertEqual(
            segment_trace.test_step_ids,
            [step.test_step_id for step in trace.steps],
        )
        self.assertEqual(segment_trace.blocked_test_step_ids, [])
        self.assertEqual(len(self.runner_calls), 2)
        self.assertEqual(sum(event[0] == "discovery" for event in self.events), 2)
        for step_trace in trace.steps:
            self.assertEqual(len(step_trace.discoveries), 1)
            discovery = step_trace.discoveries[0]
            self.assertEqual(discovery.status, DiscoveryStatus.PARTIAL)
            self.assertEqual(discovery.url, "https://example.com/")
            self.assertEqual(discovery.title, "Example")
            self.assertIsNotNone(discovery.duration_ms)
            self.assertEqual(step_trace.plan_cache.hit, False)
            self.assertIsNone(step_trace.plan_cache.version_id)
            self.assertIsNone(step_trace.discovery_fallback)
            self.assertEqual(step_trace.provider_attempts, [])
            self.assertIsNotNone(step_trace.plan_generation)
            self.assertEqual(step_trace.plan_generation.version_number, 1)
            self.assertEqual(step_trace.plan_generation.steps_count, 1)
            self.assertIsNotNone(step_trace.plan_generation.duration_ms)
            self.assertEqual(len(step_trace.execution_attempts), 1)
            attempt = step_trace.execution_attempts[0]
            self.assertEqual(attempt.status, ExecutionStatus.PASSED)
            self.assertEqual(attempt.plan_version_number, 1)
            self.assertIsNone(attempt.error)
            self.assertEqual(attempt.runner_steps[0].action, "assert_page_loaded")
            self.assertEqual(attempt.runner_steps[0].status, "passed")
            self.assertIsNone(step_trace.locator_recovery)
            self.assertIsNone(step_trace.regeneration)

        self.assertEqual(trace.totals.steps, 2)
        self.assertEqual(trace.totals.execution_attempts, 2)
        self.assertEqual(trace.totals.provider_attempts, 0)
        self.assertEqual(trace.totals.locator_recoveries, 0)
        self.assertEqual(trace.totals.regenerations, 0)

        serialized = trace.model_dump_json()
        serialized_data = ExecutionTrace.model_validate_json(serialized)
        self.assertEqual(
            [segment.segment_id for segment in serialized_data.segments],
            [segment.segment_id for segment in trace.segments],
        )
        self.assertEqual(
            [step.test_step_id for step in serialized_data.steps],
            [step.test_step_id for step in trace.steps],
        )

    def test_trace_is_created_for_failed_run(self) -> None:
        def failing_runner(plan: QATestPlan) -> dict[str, Any]:
            return {
                "status": "failed",
                "url": plan.url,
                "steps": [
                    {
                        "action": "assert_visible",
                        "status": "failed",
                        "error": "wrong title",
                    }
                ],
            }

        result = self._pipeline(runner=failing_runner).run(
            "Open the example homepage."
        )

        trace = result.trace
        assert trace is not None
        self.assertEqual(trace.status, TraceStatus.FAILED)
        self.assertIsNone(trace.error)
        for step_trace in trace.steps:
            self.assertEqual(len(step_trace.execution_attempts), 1)
            attempt = step_trace.execution_attempts[0]
            self.assertEqual(attempt.status, ExecutionStatus.FAILED)
            self.assertEqual(attempt.error, "wrong title")
            self.assertEqual(attempt.runner_steps[0].action, "assert_visible")
            self.assertEqual(attempt.runner_steps[0].status, "failed")
            self.assertEqual(attempt.runner_steps[0].error, "wrong title")
            # Assertion failures must not trigger recovery or regeneration.
            self.assertIsNone(step_trace.locator_recovery)
            self.assertIsNone(step_trace.regeneration)
        self.assertEqual(trace.totals.execution_attempts, 2)
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)

    def test_explicit_segments_reference_the_canonical_step_traces(self) -> None:
        steps = [_make_step(index) for index in range(4)]
        case = DomainTestCase(
            name="Segmented flow",
            description="Two segments with ordered steps.",
            base_url="https://example.com/",
            segments=[
                ExecutionSegment(order=0, base_url="https://example.com/", steps=steps[:2]),
                ExecutionSegment(order=1, base_url="https://example.org/", steps=steps[2:]),
            ],
        )
        result = self._pipeline(
            decomposer=_FakeDecomposer(case, self.events)
        ).run("Run segmented flow")

        trace = result.trace
        assert trace is not None
        self.assertEqual(len(trace.segments), 2)
        self.assertEqual([segment.order for segment in trace.segments], [0, 1])
        self.assertEqual([segment.segment_id for segment in trace.segments], [segment.id for segment in case.segments])
        self.assertEqual(trace.segments[0].base_url, case.base_url)
        self.assertEqual(trace.segments[1].base_url, "https://example.org/")
        self.assertEqual(
            [segment.test_step_ids for segment in trace.segments],
            [[steps[0].id, steps[1].id], [steps[2].id, steps[3].id]],
        )
        self.assertEqual(
            [step.test_step_id for step in trace.steps],
            [step.id for step in steps],
        )
        self.assertNotIn("steps", SegmentTrace.model_fields)
        self.assertEqual(len(self.runner_calls), 4)
        self.assertEqual(len(self.generator.calls), 4)
        self.assertEqual(sum(event[0] == "discovery" for event in self.events), 4)

    def test_continue_failure_remains_traceable_across_segment_boundary(self) -> None:
        steps = [_make_step(index) for index in range(3)]
        case = DomainTestCase(
            name="Continuing flow",
            description="Continue into the next segment after failure.",
            base_url="https://example.com/",
            segments=[
                ExecutionSegment(order=0, steps=[steps[0]]),
                ExecutionSegment(order=1, steps=steps[1:]),
            ],
        )
        calls = 0

        def fail_then_pass(plan: QATestPlan) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"status": "failed", "steps": [
                    {"action": "assert_visible", "status": "failed", "error": "mismatch"}
                ]}
            return {"status": "passed", "steps": []}

        result = self._pipeline(
            decomposer=_FakeDecomposer(case, self.events),
            runner=fail_then_pass,
        ).run("Continue flow")

        trace = result.trace
        assert trace is not None
        self.assertEqual(calls, 3)
        self.assertEqual(
            [step.test_step_id for step in trace.steps], [step.id for step in steps]
        )
        self.assertEqual(trace.steps[0].execution_attempts[0].status, ExecutionStatus.FAILED)
        self.assertEqual(trace.steps[1].execution_attempts[0].status, ExecutionStatus.PASSED)
        self.assertEqual(trace.segments[1].test_step_ids, [steps[1].id, steps[2].id])

    def test_block_rest_records_blocked_membership_without_fake_step_traces(self) -> None:
        steps = [
            _make_step(0, FailurePolicy.BLOCK_REST),
            _make_step(1),
            _make_step(2),
        ]
        case = DomainTestCase(
            name="Blocked flow",
            description="Block the later segment.",
            base_url="https://example.com/",
            segments=[
                ExecutionSegment(order=0, steps=[steps[0]]),
                ExecutionSegment(order=1, steps=steps[1:]),
            ],
        )

        calls = 0

        def fail(plan: QATestPlan) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {"status": "failed", "steps": [
                {"action": "assert_visible", "status": "failed", "error": "mismatch"}
            ]}

        result = self._pipeline(
            decomposer=_FakeDecomposer(case, self.events), runner=fail
        ).run("Block flow")

        trace = result.trace
        assert trace is not None
        self.assertEqual([step.test_step_id for step in trace.steps], [steps[0].id])
        self.assertEqual(trace.segments[1].test_step_ids, [steps[1].id, steps[2].id])
        self.assertEqual(trace.segments[1].blocked_test_step_ids, [steps[1].id, steps[2].id])
        self.assertEqual(trace.steps[0].execution_attempts[0].status, ExecutionStatus.FAILED)
        self.assertEqual(calls, 1)
        self.assertEqual(result.blocked_step_ids, [steps[1].id, steps[2].id])

    def test_trace_is_attached_to_pipeline_stage_error(self) -> None:
        def failed_discovery(url: str) -> DiscoveryResult:
            return DiscoveryResult(
                status=DiscoveryStatus.FAILED,
                url=url,
                warnings=["browser unavailable"],
            )

        pipeline = self._pipeline(discovery=failed_discovery)

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Open the example homepage.")

        error = raised.exception
        self.assertEqual(error.stage, "discovery (step 0: Step 0)")
        self.assertIsInstance(error.trace, ExecutionTrace)
        trace = error.trace
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertEqual(trace.error_stage, "discovery (step 0: Step 0)")
        self.assertIn("browser unavailable", trace.error or "")
        self.assertEqual(len(trace.segments), 1)
        self.assertEqual(
            trace.segments[0].test_step_ids,
            [step.id for step in sorted(self.steps, key=lambda item: item.order)],
        )
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0].discoveries[0].status, DiscoveryStatus.FAILED)

    def test_trace_is_attached_to_decomposition_error(self) -> None:
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(error=ValueError("invalid task"), events=self.events),
            discovery=self.discover,
            plan_generator=self.generator,
            runner=self.run_plan,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("bad task")

        error = raised.exception
        self.assertEqual(error.stage, "decomposition")
        self.assertIsInstance(error.trace, ExecutionTrace)
        self.assertEqual(error.trace.status, TraceStatus.ERROR)
        self.assertEqual(error.trace.error_stage, "decomposition")
        self.assertIn("invalid task", error.trace.error or "")
        self.assertEqual(error.trace.steps, [])
        self.assertEqual(error.trace.decomposition, None)

    def test_locator_recovery_is_recorded(self) -> None:
        step = _make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        store = InMemoryPlanStore()
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url="https://example.com/",
                steps=[
                    QATestStep(
                        action="click",
                        parameters={"selector": "#old", "text": "Continue"},
                    )
                ],
            ),
        )
        store.save(step.id, version_one, test_plan=plan)
        discovery_result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.com/",
            interactive_elements=[
                InteractiveElement(
                    kind="button",
                    tag="button",
                    role="button",
                    text="Continue",
                    selector="#new",
                )
            ],
        )
        calls = 0

        def runner(executable: QATestPlan) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "status": "failed",
                    "steps": [
                        {
                            "action": "click",
                            "status": "failed",
                            "error": "Selector '#old' was not found on the page.",
                        }
                    ],
                }
            self.assertEqual(executable.steps[0].parameters["selector"], "#new")
            return {"status": "passed", "steps": []}

        result = QATestPipeline(
            decomposer=_FakeDecomposer(case),
            discovery=lambda _: discovery_result,
            plan_generator=_FakeGenerator(),
            runner=runner,
            plan_store=store,
        ).run("Check")

        trace = result.trace
        assert trace is not None
        step_trace = trace.steps[0]
        self.assertEqual(step_trace.plan_cache.hit, True)
        self.assertEqual(step_trace.plan_cache.version_id, version_one.id)
        self.assertEqual(step_trace.plan_cache.version_number, 1)
        # A cache hit skips the initial discovery, so only the rediscovery
        # after the stale-UI failure is recorded for this step.
        self.assertEqual(len(step_trace.discoveries), 1)
        self.assertEqual(step_trace.discoveries[0].status, DiscoveryStatus.SUCCESS)
        recovery = step_trace.locator_recovery
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.status, RecoveryStatus.MATCHED)
        self.assertEqual(recovery.original_selector, "#old")
        self.assertEqual(recovery.candidate_selector, "#new")
        self.assertIn("exact_text", recovery.signals)
        self.assertIsNone(step_trace.regeneration)
        self.assertEqual(
            [item.status for item in step_trace.execution_attempts],
            [ExecutionStatus.FAILED, ExecutionStatus.PASSED],
        )
        self.assertEqual(step_trace.execution_attempts[0].planned_step_index, 0)
        self.assertEqual(step_trace.execution_attempts[1].plan_version_number, 2)
        self.assertEqual(trace.totals.locator_recoveries, 1)
        self.assertEqual(trace.totals.regenerations, 0)
        self.assertEqual(trace.totals.execution_attempts, 2)
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)

    def test_rejected_locator_conflict_is_recorded_without_regeneration(self) -> None:
        step = _make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        store = InMemoryPlanStore()
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url="https://example.com/",
                steps=[
                    QATestStep(
                        action="click",
                        parameters={"selector": "#old", "text": "Continue"},
                    )
                ],
            ),
        )
        store.save(step.id, version_one, test_plan=plan)
        discovery_result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.com/",
            interactive_elements=[
                InteractiveElement(
                    kind="button",
                    tag="button",
                    role="button",
                    text="Other",
                    selector="#other",
                )
            ],
        )
        calls = 0

        def runner(executable: QATestPlan) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "status": "failed",
                    "steps": [
                        {
                            "action": "click",
                            "status": "failed",
                            "error": "Selector '#old' was not found on the page.",
                        }
                    ],
                }
            return {"status": "passed", "steps": []}

        generator = _FakeGenerator()
        result = QATestPipeline(
            decomposer=_FakeDecomposer(case),
            discovery=lambda _: discovery_result,
            plan_generator=generator,
            runner=runner,
            plan_store=store,
        ).run("Check")

        trace = result.trace
        assert trace is not None
        step_trace = trace.steps[0]
        recovery = step_trace.locator_recovery
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.status, RecoveryStatus.REJECTED_CONFLICT)
        self.assertEqual(recovery.original_selector, "#old")
        self.assertIsNone(recovery.candidate_selector)
        self.assertIn("conflict", recovery.reason)
        self.assertIsNone(step_trace.regeneration)
        self.assertEqual(
            [item.status for item in step_trace.execution_attempts],
            [ExecutionStatus.FAILED],
        )
        self.assertEqual(generator.versions, [])
        self.assertEqual([pair.test_plan_version.version for pair in result.test_plans], [1])
        self.assertEqual(trace.totals.locator_recoveries, 1)
        self.assertEqual(trace.totals.regenerations, 0)
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)

    def test_discovery_fallback_is_recorded(self) -> None:
        fallback_suggestions = AIDiscoveryResult(
            navigation_paths=[],
            interactive_elements=[
                InteractiveElement(
                    kind="link", selector="#ai-link", text="AI found"
                )
            ],
            warnings=["ai fallback hint"],
        )
        fallback = Mock()
        fallback.discover.return_value = fallback_suggestions
        partial = DiscoveryResult(
            status=DiscoveryStatus.FAILED,
            url="https://example.com/",
            warnings=["deterministic discovery failed"],
        )

        result = QATestPipeline(
            decomposer=_FakeDecomposer(self.test_case, self.events),
            discovery=lambda _: partial,
            discovery_fallback=fallback,
            plan_generator=_FakeGenerator(self.events),
            runner=self.run_plan,
        ).run("Open the example homepage.")

        trace = result.trace
        assert trace is not None
        for step_trace in trace.steps:
            fallback_trace = step_trace.discovery_fallback
            self.assertIsNotNone(fallback_trace)
            self.assertTrue(fallback_trace.invoked)
            # Successful fallback: never marked as failed.
            self.assertIsNone(fallback_trace.error)
            self.assertEqual(fallback_trace.interactive_elements_added, 1)
            self.assertEqual(fallback_trace.warnings, ["ai fallback hint"])
            self.assertIsNotNone(fallback_trace.duration_ms)
            # The merged discovery result is the one recorded for the step,
            # exactly once — no duplicate discovery entries.
            self.assertEqual(len(step_trace.discoveries), 1)
            merged = step_trace.discoveries[0]
            self.assertEqual(merged.status, DiscoveryStatus.PARTIAL)
            self.assertIn("deterministic discovery failed", merged.warnings)
            self.assertIn("ai fallback hint", merged.warnings)
            self.assertEqual(len(merged.interactive_elements), 1)

    def test_trace_does_not_change_pipeline_result(self) -> None:
        result = self._pipeline().run("Open the example homepage.")

        self.assertEqual(
            [event[0] for event in self.events],
            [
                "decomposer",
                "discovery",
                "generator",
                "runner",
                "discovery",
                "generator",
                "runner",
            ],
        )
        self.assertEqual(len(result.test_plans), 2)
        self.assertEqual(len(result.executions), 2)
        self.assertEqual(self.runner_calls, [plan for plan in self.runner_calls])
        self.assertEqual(list(result), result.executions)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], result.executions[0])
        self.assertEqual(result[:], result.executions)
        self.assertEqual(result.test_run.test_case_id, self.test_case.id)
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)

    def test_trace_failure_does_not_break_execution(self) -> None:
        class _BrokenRecorder(ExecutionTraceRecorder):
            def record_discovery(
                self, result: DiscoveryResult, duration_ms: int | None
            ) -> None:
                raise RuntimeError("trace recorder is broken")

            def record_execution_attempt(
                self,
                execution: Any,
                plan_version_number: int | None = None,
            ) -> None:
                raise RuntimeError("trace execution recording is broken")

            def finalize(
                self,
                status: TraceStatus,
                error: Exception | None = None,
                error_stage: str | None = None,
            ) -> ExecutionTrace:
                raise RuntimeError("trace finalize is broken")

        class _BrokenTracePipeline(QATestPipeline):
            def _create_trace_recorder(self, task: str) -> ExecutionTraceRecorder:
                return _BrokenRecorder(task=task)

        pipeline = _BrokenTracePipeline(
            decomposer=_FakeDecomposer(self.test_case, self.events),
            discovery=self.discover,
            plan_generator=_FakeGenerator(self.events),
            runner=self.run_plan,
        )

        result = pipeline.run("Open the example homepage.")

        self.assertEqual(len(result.executions), 2)
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)
        self.assertEqual(
            [event[0] for event in self.events],
            [
                "decomposer",
                "discovery",
                "generator",
                "runner",
                "discovery",
                "generator",
                "runner",
            ],
        )
        self.assertIsNone(result.trace)

    def test_pipeline_result_backward_compatibility(self) -> None:
        positional = PipelineResult(self.test_case, [], [])
        self.assertIsNone(positional.trace)
        self.assertEqual(list(positional), [])
        self.assertEqual(len(positional), 0)

        trace = ExecutionTrace(
            task="Open the example homepage.",
            status=TraceStatus.PASSED,
            started_at=datetime.now(timezone.utc),
        )
        keyword = PipelineResult(
            test_case=self.test_case,
            test_plans=[],
            executions=[],
            trace=trace,
        )
        self.assertIs(keyword.trace, trace)
        self.assertIsNone(keyword.trace.error)
        self.assertEqual(keyword.test_run.test_case_id, self.test_case.id)

    def test_sensitive_values_are_redacted(self) -> None:
        secret = "super-secret-trace-token-123"
        with patch.dict(os.environ, {"QA_TRACE_TEST_API_KEY": secret}):
            def failing_runner(plan: QATestPlan) -> dict[str, Any]:
                return {
                    "status": "failed",
                    "url": plan.url,
                    "steps": [
                        {
                            "action": "assert_visible",
                            "status": "failed",
                            "error": f"Selector '{secret}' was not found.",
                        }
                    ],
                }

            result = self._pipeline(runner=failing_runner).run(
                "Open the example homepage."
            )

        trace = result.trace
        assert trace is not None
        for step_trace in trace.steps:
            attempt = step_trace.execution_attempts[0]
            self.assertIn("[REDACTED]", attempt.error or "")
            self.assertNotIn(secret, attempt.error or "")
            self.assertIn("[REDACTED]", attempt.runner_steps[0].error or "")
            self.assertNotIn(secret, attempt.runner_steps[0].error or "")
        # The domain Execution keeps the raw error: existing behavior is unchanged.
        self.assertIn(secret, result.executions[0].error)

    def test_task_text_is_redacted(self) -> None:
        secret = "super-secret-task-token-456"
        with patch.dict(os.environ, {"QA_TRACE_TEST_API_KEY": secret}):
            result = self._pipeline().run(f"Open the {secret} homepage.")

        trace = result.trace
        assert trace is not None
        self.assertNotIn(secret, trace.task)
        self.assertIn("[REDACTED]", trace.task)


class ExecutionTraceRouterTests(unittest.TestCase):
    def _recorder_with_step(self) -> ExecutionTraceRecorder:
        recorder = ExecutionTraceRecorder(task="Check")
        recorder.begin_step(_test_step())
        return recorder

    def test_provider_fallback_is_recorded_in_trace(self) -> None:
        plan = _plan()
        failing = _StubProvider(
            "failing",
            plan=plan,
            error=RetryableLLMError("request failed with HTTP 429"),
        )
        succeeding = _StubProvider("succeeding", plan=plan)
        router = LLMRouter([failing, succeeding])
        recorder = self._recorder_with_step()

        with (
            active_trace_recorder(recorder),
            redirect_stdout(io.StringIO()),
        ):
            result = router.create_test_plan(
                "Check", "https://example.com/", "{}"
            )

        self.assertEqual(result, plan)
        trace = recorder.finalize(TraceStatus.PASSED)
        attempts = trace.steps[0].provider_attempts
        self.assertEqual(len(attempts), 2)
        first, second = attempts
        self.assertEqual(first.provider_name, "failing")
        self.assertEqual(first.request_kind, RequestKind.TEST_PLAN)
        self.assertEqual(first.outcome, ProviderAttemptOutcome.RETRYABLE_ERROR)
        self.assertEqual(first.model, "failing-model")
        self.assertEqual(first.error_class, "RetryableLLMError")
        self.assertIn("HTTP 429", first.error_message or "")
        self.assertIsNone(first.http_status)
        self.assertIsNotNone(first.duration_ms)
        self.assertFalse(first.is_selected)
        self.assertEqual(second.provider_name, "succeeding")
        self.assertEqual(second.outcome, ProviderAttemptOutcome.SUCCESS)
        self.assertEqual(second.model, "succeeding-model")
        self.assertIsNone(second.error_class)
        self.assertIsNone(second.error_message)
        self.assertTrue(second.is_selected)
        self.assertEqual(trace.totals.provider_attempts, 2)

    def test_unavailable_providers_are_recorded_in_trace(self) -> None:
        plan = _plan()
        skipped = _StubProvider("skipped", plan=plan, available=False)
        succeeding = _StubProvider("succeeding", plan=plan)
        router = LLMRouter([skipped, succeeding])
        recorder = self._recorder_with_step()

        with (
            active_trace_recorder(recorder),
            redirect_stdout(io.StringIO()),
        ):
            router.create_test_plan("Check", "https://example.com/", "{}")

        trace = recorder.finalize(TraceStatus.PASSED)
        attempts = trace.steps[0].provider_attempts
        self.assertEqual(
            [item.outcome for item in attempts],
            [ProviderAttemptOutcome.UNAVAILABLE, ProviderAttemptOutcome.SUCCESS],
        )
        self.assertEqual(
            [item.provider_name for item in attempts], ["skipped", "succeeding"]
        )
        self.assertFalse(attempts[0].is_selected)
        self.assertTrue(attempts[1].is_selected)
        self.assertIsNone(attempts[0].duration_ms)

    def test_non_retryable_error_is_recorded_and_halts_fallback(self) -> None:
        plan = _plan()
        failing = _StubProvider(
            "failing", plan=plan, error=NonRetryableLLMError("invalid prompt")
        )
        succeeding = _StubProvider("succeeding", plan=plan)
        router = LLMRouter([failing, succeeding])
        recorder = self._recorder_with_step()

        with (
            active_trace_recorder(recorder),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(NonRetryableLLMError):
                router.create_test_plan("Check", "https://example.com/", "{}")

        trace = recorder.finalize(TraceStatus.ERROR)
        attempts = trace.steps[0].provider_attempts
        self.assertEqual(len(attempts), 1)
        self.assertEqual(
            attempts[0].outcome, ProviderAttemptOutcome.NON_RETRYABLE_ERROR
        )
        self.assertEqual(attempts[0].error_class, "NonRetryableLLMError")
        self.assertFalse(attempts[0].is_selected)

    def test_unclassified_error_is_recorded_and_halts_fallback(self) -> None:
        # P2-2 regression: a raw exception at the provider boundary stays
        # unclassified — no fallback, original identity preserved — but the
        # attempted provider is recorded exactly once in the trace.
        boom = RuntimeError("boom")
        failing_calls: list[str] = []
        succeeding_calls: list[str] = []

        class _Failing(LLMProvider):
            name = "failing_raw"
            model = "failing-raw-model"

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                failing_calls.append(task)
                raise boom

        class _Succeeding(LLMProvider):
            name = "succeeding_raw"
            model = "succeeding-raw-model"

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                succeeding_calls.append(task)
                return _plan()

        router = LLMRouter([_Failing(), _Succeeding()])
        recorder = self._recorder_with_step()

        with (
            active_trace_recorder(recorder),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError) as context:
                router.create_test_plan("Check", "https://example.com/", "{}")

        # The original exception is re-raised unchanged — not wrapped.
        self.assertIs(context.exception, boom)

        # No fallback: the second provider was never invoked.
        self.assertEqual(len(failing_calls), 1)
        self.assertEqual(succeeding_calls, [])

        # Exactly one attempt row, deliberately unclassified.
        trace = recorder.finalize(TraceStatus.ERROR)
        attempts = trace.steps[0].provider_attempts
        self.assertEqual(len(attempts), 1)
        attempt = attempts[0]
        self.assertEqual(
            attempt.outcome, ProviderAttemptOutcome.UNCLASSIFIED_ERROR
        )
        self.assertEqual(attempt.request_kind, RequestKind.TEST_PLAN)
        self.assertEqual(attempt.provider_name, "failing_raw")
        self.assertEqual(attempt.model, "failing-raw-model")
        self.assertEqual(attempt.error_class, "RuntimeError")
        self.assertEqual(attempt.error_message, "boom")
        self.assertIsNotNone(attempt.duration_ms)
        self.assertFalse(attempt.is_selected)
        self.assertEqual(trace.totals.provider_attempts, 1)

    def test_discovery_attempts_are_recorded_in_trace(self) -> None:
        failing = _StubProvider(
            "failing", error=RetryableLLMError("temporary failure")
        )
        succeeding = _StubProvider("succeeding")
        router = LLMRouter([failing, succeeding])
        recorder = self._recorder_with_step()

        with (
            active_trace_recorder(recorder),
            redirect_stdout(io.StringIO()),
        ):
            result = router.create_discovery(
                "Discover", "https://example.com/", "{}"
            )

        self.assertIsInstance(result, AIDiscoveryResult)
        trace = recorder.finalize(TraceStatus.PASSED)
        attempts = trace.steps[0].provider_attempts
        self.assertEqual(
            [item.request_kind for item in attempts],
            [RequestKind.DISCOVERY, RequestKind.DISCOVERY],
        )
        self.assertEqual(
            [item.outcome for item in attempts],
            [ProviderAttemptOutcome.RETRYABLE_ERROR, ProviderAttemptOutcome.SUCCESS],
        )

    def test_unclassified_discovery_error_is_recorded_and_halts_fallback(
        self,
    ) -> None:
        # P2-2 regression: same contract on the discovery route — the raw
        # exception propagates unchanged, no fallback, one trace row.
        boom = RuntimeError("discovery boom")
        failing_calls: list[str] = []
        succeeding_calls: list[str] = []

        class _Failing(LLMProvider):
            name = "failing_raw"
            model = "failing-raw-model"

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                raise AssertionError("create_test_plan must not run here")

            def create_discovery(
                self, task: str, target_url: str, page_snapshot: str
            ) -> AIDiscoveryResult:
                failing_calls.append(task)
                raise boom

        class _Succeeding(LLMProvider):
            name = "succeeding_raw"
            model = "succeeding-raw-model"

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                raise AssertionError("create_test_plan must not run here")

            def create_discovery(
                self, task: str, target_url: str, page_snapshot: str
            ) -> AIDiscoveryResult:
                succeeding_calls.append(task)
                return AIDiscoveryResult()

        router = LLMRouter([_Failing(), _Succeeding()])
        recorder = self._recorder_with_step()

        with (
            active_trace_recorder(recorder),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError) as context:
                router.create_discovery("Discover", "https://example.com/", "{}")

        # The original exception is re-raised unchanged — not wrapped.
        self.assertIs(context.exception, boom)

        # No fallback: the second provider was never invoked.
        self.assertEqual(len(failing_calls), 1)
        self.assertEqual(succeeding_calls, [])

        # Exactly one attempt row on the DISCOVERY route.
        trace = recorder.finalize(TraceStatus.ERROR)
        attempts = trace.steps[0].provider_attempts
        self.assertEqual(len(attempts), 1)
        attempt = attempts[0]
        self.assertEqual(
            attempt.outcome, ProviderAttemptOutcome.UNCLASSIFIED_ERROR
        )
        self.assertEqual(attempt.request_kind, RequestKind.DISCOVERY)
        self.assertEqual(attempt.provider_name, "failing_raw")
        self.assertEqual(attempt.error_class, "RuntimeError")
        self.assertEqual(attempt.error_message, "discovery boom")
        self.assertIsNotNone(attempt.duration_ms)
        self.assertFalse(attempt.is_selected)
        self.assertEqual(trace.totals.provider_attempts, 1)

    def test_router_runs_without_active_recorder(self) -> None:
        plan = _plan()
        router = LLMRouter([_StubProvider("succeeding", plan=plan)])

        with redirect_stdout(io.StringIO()):
            result = router.create_test_plan(
                "Check", "https://example.com/", "{}"
            )

        self.assertEqual(result, plan)

    def test_sensitive_values_are_redacted_in_provider_attempts(self) -> None:
        secret = "super-secret-provider-token-789"
        plan = _plan()
        with patch.dict(os.environ, {"QA_TRACE_TEST_API_KEY": secret}):
            failing = _StubProvider(
                "failing",
                plan=plan,
                error=RetryableLLMError(f"auth failed for {secret}"),
            )
            succeeding = _StubProvider("succeeding", plan=plan)
            router = LLMRouter([failing, succeeding])
            recorder = self._recorder_with_step()

            with (
                active_trace_recorder(recorder),
                redirect_stdout(io.StringIO()),
            ):
                router.create_test_plan("Check", "https://example.com/", "{}")

        trace = recorder.finalize(TraceStatus.PASSED)
        message = trace.steps[0].provider_attempts[0].error_message or ""
        self.assertIn("[REDACTED]", message)
        self.assertNotIn(secret, message)

    def test_provider_attempts_flow_through_pipeline(self) -> None:
        plan = _plan()
        failing = _StubProvider(
            "failing", plan=plan, error=RetryableLLMError("temporary failure")
        )
        succeeding = _StubProvider("succeeding", plan=plan)
        router = LLMRouter([failing, succeeding])
        generator = LLMTestPlanGenerator(router)
        test_case = DomainTestCase(
            name="Example flow",
            description="Open the example homepage.",
            base_url="https://example.com/",
            steps=[_make_step(0), _make_step(1)],
        )
        discovery_result = DiscoveryResult(
            status=DiscoveryStatus.PARTIAL, url="https://example.com/"
        )
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(test_case),
            discovery=lambda _: discovery_result,
            plan_generator=generator,
            runner=lambda _: {
                "status": "passed",
                "steps": [],
            },
        )

        with redirect_stdout(io.StringIO()):
            result = pipeline.run("Open the example homepage.")

        trace = result.trace
        assert trace is not None
        self.assertEqual(trace.status, TraceStatus.PASSED)
        for step_trace in trace.steps:
            attempts = step_trace.provider_attempts
            self.assertEqual(len(attempts), 2)
            self.assertEqual(
                [item.outcome for item in attempts],
                [
                    ProviderAttemptOutcome.RETRYABLE_ERROR,
                    ProviderAttemptOutcome.SUCCESS,
                ],
            )
            self.assertEqual(
                [item.provider_name for item in attempts], ["failing", "succeeding"]
            )
            self.assertEqual(
                [item.model for item in attempts], ["failing-model", "succeeding-model"]
            )
            self.assertFalse(attempts[0].is_selected)
            self.assertTrue(attempts[1].is_selected)
        self.assertEqual(trace.totals.provider_attempts, 4)
        self.assertEqual(trace.totals.steps, 2)


if __name__ == "__main__":
    unittest.main()
