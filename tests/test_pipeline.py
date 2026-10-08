import io
import unittest
# unittest does not import its mock submodule itself; without this import
# unittest.mock.Mock() below only works when another test module happened
# to import unittest.mock first (full-suite runs), breaking standalone runs.
import unittest.mock
from contextlib import redirect_stdout
from typing import Any

from qa_agent.automation_lifecycle import (
    AutomationLifecycleService,
    AutomationStatus,
    InMemoryAutomationLifecycleRepository,
)
from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    ExecutionStatus,
    FailurePolicy,
    InteractiveElement,
    LocatorIdentityEntry,
    PlanVersionOrigin,
    QATestPlan,
    QATestStep,
    RunContext,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.execution_trace import ProviderAttemptOutcome, TraceStatus
from qa_agent.locator_recovery import RecoveryStatus
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.pipeline import (
    PipelineResult,
    PipelineStageError,
    QATestPipeline,
    _with_discovered_locator_identity,
    _automation_run_outcome,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService, WorkflowType
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import GeneratedTestPlan, LLMTestPlanGenerator


class QATestPipelineTests(unittest.TestCase):
    def test_checkbox_actions_attach_only_discovered_value_free_identity(self) -> None:
        step = self.make_step(0)
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        generated = GeneratedTestPlan(
            test_plan=plan,
            test_plan_version=DomainTestPlanVersion(
                test_plan_id=plan.id,
                version=1,
                qa_test_plan=QATestPlan(url="http://127.0.0.1/", steps=[
                    QATestStep(action="check", parameters={"selector": "#marketing"}),
                    QATestStep(action="click", parameters={"selector": "#confirm-details"}),
                ]),
            ),
        )
        discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="http://127.0.0.1/",
            interactive_elements=[InteractiveElement(
                kind="checkbox", selector="#marketing", tag="input", role="checkbox",
                label="Marketing emails", accessible_name="Marketing emails",
            ), InteractiveElement(
                kind="button", selector="#confirm-details", tag="button", role="button",
                label="Confirm details", accessible_name="Confirm details",
                dialog_identity="id:details-dialog",
            )],
        )

        attached = _with_discovered_locator_identity(generated, discovery)

        identity = attached.test_plan_version.locator_identity[0]
        self.assertEqual(identity.step_index, 0)
        self.assertEqual(identity.label, "Marketing emails")
        self.assertEqual(identity.accessible_name, "Marketing emails")
        self.assertIsNone(identity.visible_text)
        self.assertEqual(
            attached.test_plan_version.locator_identity[1].dialog_identity,
            "id:details-dialog",
        )

    def test_custom_generator_missing_expected_coverage_is_rejected_before_save(self) -> None:
        step = self.make_step(0).model_copy(update={
            "name": "Verify the error message is displayed",
            "description": "Submit invalid data and inspect the response.",
            "expected": "An error message is displayed.",
        })
        case = DomainTestCase(
            name="Coverage boundary",
            description="Submit invalid data and verify its error message.",
            base_url="https://example.test/",
            steps=[step],
        )
        store = InMemoryPlanStore()
        runner_calls = []
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case),
            plan_generator=_FakeGenerator([]),
            discovery=lambda url: DiscoveryResult(
                status=DiscoveryStatus.SUCCESS,
                url=url,
            ),
            runner=lambda plan: runner_calls.append(plan) or {"status": "passed", "steps": []},
            plan_store=store,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run_test_case(case)

        self.assertIn("plan generation", raised.exception.stage)
        self.assertIsNone(store.find(step.id))
        self.assertEqual(runner_calls, [])

    def test_plan_missing_required_browser_parameter_is_not_persisted_or_executed(self) -> None:
        step = self.make_step(0)
        case = DomainTestCase(
            name="Invalid generated plan",
            description="Reject unsupported generated actions before persistence.",
            base_url="https://example.test/",
            steps=[step],
        )

        class InvalidGenerator(_FakeGenerator):
            def generate_with_plan(
                inner_self,
                test_step,
                discovery_result,
                *,
                existing_test_plan=None,
                version_number=1,
            ):
                generated = super().generate_with_plan(
                    test_step,
                    discovery_result,
                    existing_test_plan=existing_test_plan,
                    version_number=version_number,
                )
                invalid_plan = QATestPlan(
                    url=generated.test_plan_version.qa_test_plan.url,
                    steps=[QATestStep(action="click", parameters={})],
                )
                invalid_version = generated.test_plan_version.model_copy(
                    update={"qa_test_plan": invalid_plan}
                )
                return GeneratedTestPlan(
                    test_plan=generated.test_plan,
                    test_plan_version=invalid_version,
                )

        store = InMemoryPlanStore()
        runner_calls = []
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case),
            plan_generator=InvalidGenerator([]),
            discovery=lambda url: DiscoveryResult(
                status=DiscoveryStatus.SUCCESS,
                url=url,
            ),
            runner=lambda plan: runner_calls.append(plan) or {"status": "passed", "steps": []},
            plan_store=store,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run_test_case(case)

        self.assertIn("plan generation", raised.exception.stage)
        self.assertIsNone(store.find(step.id))
        self.assertEqual(runner_calls, [])

    def test_ai_discovery_runs_only_for_partial_or_failed_deterministic_results(self) -> None:
        from qa_agent.models import AIDiscoveryResult, NavigationPath

        for status in (DiscoveryStatus.SUCCESS, DiscoveryStatus.PARTIAL, DiscoveryStatus.FAILED):
            with self.subTest(status=status):
                case = DomainTestCase(name="Flow", description="Check", base_url="https://example.com/",
                                      steps=[self.make_step(0)])
                base = DiscoveryResult(status=status, url="https://example.com/",
                                       warnings=["missing path"] if status != DiscoveryStatus.SUCCESS else [])
                fallback = unittest.mock.Mock()
                fallback.discover.return_value = AIDiscoveryResult(
                    navigation_paths=[NavigationPath(menu_item_text="Account")])
                generator = _FakeGenerator([])
                result = QATestPipeline(
                    decomposer=_FakeDecomposer(case), discovery=lambda _: base,
                    discovery_fallback=fallback, plan_generator=generator,
                    runner=lambda plan: {"status": "passed", "steps": []},
                ).run("Check account")
                self.assertEqual(
                    result.test_plans[0].test_plan_version.origin,
                    PlanVersionOrigin.AI_GENERATED,
                )
                self.assertEqual(fallback.discover.call_count,
                                 0 if status == DiscoveryStatus.SUCCESS else 1)
                if status != DiscoveryStatus.SUCCESS:
                    self.assertEqual(generator.calls[0][1].navigation_paths[0].menu_item_text, "Account")

    def test_matched_stale_locator_is_repaired_deterministically_and_preserves_history(self) -> None:
        step = DomainTestStep(
            name="Click Continue",
            description="Click Continue to proceed.",
            expected="Continue is clicked.",
            order=0,
        )
        test_case = DomainTestCase(name="Flow", description="Check", base_url="https://example.com/", steps=[step])
        plan_store = InMemoryPlanStore()
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=plan.id, version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})
            ]),
            locator_identity=(LocatorIdentityEntry(
                step_index=0,
                accessible_name="Continue",
                visible_text="Continue",
                tag="button",
                role="button",
            ),),
        )
        plan_store.save(step.id, version_one, test_plan=plan)
        lifecycle = AutomationLifecycleService(
            InMemoryAutomationLifecycleRepository(), plan_store
        )
        self.assertEqual(
            lifecycle.mark_validation_completed(test_case, passed=True),
            AutomationStatus.AUTOMATION_READY,
        )
        discovery_result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS, url="https://example.com/",
            interactive_elements=[InteractiveElement(
                kind="button", tag="button", role="button", text="Continue", selector="#new"
            )],
        )
        calls = 0
        def runner(executable):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"status": "failed", "steps": [{"action": "click", "status": "failed",
                    "error": "Selector '#old' was not found on the page."}]}
            self.assertEqual(executable.steps[0].parameters["selector"], "#new")
            return {"status": "passed", "steps": [{"action": "click", "status": "passed"}]}
        generator = _FakeGenerator([])
        repository = InMemoryExecutionRepository()
        result = QATestPipeline(
            decomposer=_FakeDecomposer(test_case), discovery=lambda _: discovery_result,
            plan_generator=generator, runner=runner, plan_store=plan_store,
            execution_repository=repository,
        ).run("Check")

        self.assertEqual([p.test_plan_version.version for p in result.test_plans], [1, 2])
        self.assertEqual(
            [pair.test_plan_version.origin for pair in result.test_plans],
            [None, PlanVersionOrigin.REPAIRED],
        )
        self.assertEqual(result.test_plans[0].test_plan_version.qa_test_plan.steps[0].parameters["selector"], "#old")
        self.assertEqual(result.test_plans[1].test_plan_version.qa_test_plan.steps[0].parameters["selector"], "#new")
        self.assertEqual([e.status for e in result.executions], [ExecutionStatus.FAILED, ExecutionStatus.PASSED])
        self.assertEqual([e.test_plan_version_id for e in repository.list_for_test_step(step.id)],
                         [version_one.id, result.test_plans[-1].test_plan_version.id])
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)
        repaired_version = result.test_plans[-1].test_plan_version
        self.assertEqual(repaired_version.version, 2)
        self.assertEqual(repaired_version.origin, PlanVersionOrigin.REPAIRED)
        self.assertEqual(repaired_version.locator_identity[0].visible_text, "Continue")
        self.assertEqual(plan_store.get_version(version_one.id), version_one)
        self.assertEqual(plan_store.list_versions(step.id), (repaired_version, version_one))
        self.assertEqual(lifecycle.status(test_case), AutomationStatus.NEEDS_VALIDATION)
        self.assertEqual(generator.calls, [])

    def test_ambiguous_and_not_found_recovery_stop_without_llm_or_plan_change(self) -> None:
        for elements in ([], [
            InteractiveElement(kind="button", tag="button", role="button", text="Continue", selector="#a"),
            InteractiveElement(kind="button", tag="button", role="button", text="Continue", selector="#b"),
        ]):
            with self.subTest(elements=len(elements)):
                step = self.make_step(0)
                case = DomainTestCase(name="Flow", description="Check", base_url="https://example.com/", steps=[step])
                generator = _FakeGenerator([])
                run_count = 0
                def runner(plan):
                    nonlocal run_count
                    run_count += 1
                    return ({"status": "failed", "steps": [{"action": "click", "status": "failed",
                        "error": "Selector '#old' was not found on the page."}]} if run_count == 1
                        else {"status": "passed", "steps": []})
                cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
                v1 = DomainTestPlanVersion(test_plan_id=cached_plan.id, version=1,
                    qa_test_plan=QATestPlan(url="https://example.com/", steps=[QATestStep(
                        action="click", parameters={"selector": "#old", "text": "Continue"})]))
                store = InMemoryPlanStore(); store.save(step.id, v1, test_plan=cached_plan)
                discovery_result = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url="https://example.com/", interactive_elements=elements)
                result = QATestPipeline(decomposer=_FakeDecomposer(case), discovery=lambda _: discovery_result,
                    plan_generator=generator, runner=runner, plan_store=store).run("Check")
                self.assertEqual(generator.calls, [])
                self.assertEqual(run_count, 1)
                self.assertEqual([p.test_plan_version.version for p in result.test_plans], [1])
                self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)
                self.assertEqual(
                    result.trace.steps[0].locator_recovery.status,
                    RecoveryStatus.AMBIGUOUS if elements else RecoveryStatus.NO_MATCH,
                )
                self.assertEqual(store.find(step.id).id, v1.id)

    def test_unsafe_recovery_refusal_keeps_version_one_without_llm_call(
        self,
    ) -> None:
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=cached_plan.id,
            version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})]),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version_one, test_plan=cached_plan)

        # Deterministic rediscovery finds no candidate. The failure remains
        # Automation Drift and the pipeline does not ask an LLM to choose.
        discovery_result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.com/",
            interactive_elements=[],
        )
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)

        def stale_runner(plan: QATestPlan) -> dict[str, Any]:
            return {"status": "failed", "steps": [{
                "action": "click", "status": "failed",
                "error": "Selector '#old' was not found on the page."}]}

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=lambda _: discovery_result,
            plan_generator=generator,
            runner=stale_runner,
            plan_store=store,
            execution_repository=repository,
        )

        result = pipeline.run("Check")

        # The run is a completed failed execution, not a generation error.
        trace = result.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.FAILED)
        self.assertIsNone(trace.error_stage)
        self.assertIsNone(trace.error)

        # Recovery ran, but no successful regeneration was recorded.
        step_trace = trace.steps[0]
        self.assertEqual(
            step_trace.locator_recovery.status, RecoveryStatus.NO_MATCH
        )
        self.assertIsNone(step_trace.regeneration)
        self.assertIsNone(step_trace.plan_generation)
        self.assertEqual(trace.totals.regenerations, 0)

        # No regeneration attempt is made.
        self.assertEqual(generator.calls, [])

        # No v2 was saved; the plan store still points at v1.
        current = store.find(step.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.id, version_one.id)
        self.assertEqual(current.version, 1)
        self.assertIsNotNone(store.get_version(version_one.id))

        # The original v1 failed Execution remains persisted.
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].status, ExecutionStatus.FAILED)
        self.assertEqual(persisted[0].test_plan_version_id, version_one.id)
        self.assertEqual(
            persisted[0].error, "Selector '#old' was not found on the page."
        )

    def test_infrastructure_click_failure_does_not_trigger_locator_recovery(self) -> None:
        one_step_case = DomainTestCase(name="Flow", description="Check", base_url="https://example.com/", steps=[self.make_step(0)])
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(one_step_case), discovery=self.discover,
            plan_generator=self.generator, runner=lambda plan: {"status": "failed", "steps": [{
                "action": "click", "status": "failed", "error": "Browser has been closed unexpectedly."}]},
        )
        result = pipeline.run("Check")
        self.assertEqual(len(result.executions), 1)
        self.assertEqual(len(self.generator.calls), 1)
        self.assertEqual(sum(event[0] == "discovery" for event in self.events), 1)

    def setUp(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        # Deliberately reversed to check that orchestration honors TestStep.order.
        self.steps = [self.make_step(1), self.make_step(0)]
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
                "steps": [{"action": "assert_page_loaded", "status": "passed", "error": ""}],
            }

        self.discover = discover
        self.run_plan = run
        self.pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=self.run_plan,
        )

    @staticmethod
    def make_step(order: int) -> DomainTestStep:
        return DomainTestStep(
            name=f"Step {order}",
            description=f"Perform action {order}",
            expected=f"Action {order} is completed",
            order=order,
        )

    def test_full_pipeline_runs_steps_in_order_and_returns_executions(self) -> None:
        result = self.pipeline.run("Open the example homepage.")
        executions = result.executions

        self.assertIsInstance(result, PipelineResult)
        self.assertIs(result.test_case, self.test_case)
        self.assertEqual(len(result.test_plans), 2)
        self.assertEqual(len(executions), 2)
        self.assertEqual(
            [event[0] for event in self.events],
            ["decomposer", "discovery", "generator", "runner", "discovery", "generator", "runner"],
        )
        self.assertEqual([event[1].order for event in self.events if event[0] == "generator"], [0, 1])
        self.assertTrue(all(execution.status == ExecutionStatus.PASSED for execution in executions))
        self.assertEqual(list(result), executions)
        self.assertEqual(len(result), len(executions))
        self.assertIs(result[0], executions[0])
        self.assertEqual(result[:], executions)
        self.assertEqual(result.test_run.test_case_id, self.test_case.id)
        self.assertEqual(result.test_run.executions, executions)
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)

    def test_automation_stops_after_cookie_consent_attention(self) -> None:
        runner_calls = []

        def runner(plan):
            runner_calls.append(plan)
            return {
                "status": "failed",
                "cookie_consent_requires_attention": True,
                "steps": [{
                    "action": "cookie_consent_precondition",
                    "status": "failed",
                    "error": "Cookie consent requires attention.",
                }],
            }

        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=runner,
        )
        result = pipeline.run("Open the example homepage.")

        self.assertEqual(len(runner_calls), 1)
        self.assertEqual(len(result.executions), 1)
        self.assertEqual(result.blocked_step_ids, [self.steps[0].id])
        self.assertEqual(_automation_run_outcome(result.test_run), "AUTOMATION_EXECUTION_ERROR")

    def test_pipeline_preserves_supplied_run_context(self) -> None:
        context = RunContext()
        secret = "FAKE_PIPELINE_CONTEXT_SECRET_731"
        context.set_value("session_token", secret, sensitive=True)

        result = self.pipeline.run("Open the example homepage.", run_context=context)

        self.assertIs(result.run_context, context)
        self.assertIs(result.test_run.run_context, context)
        self.assertEqual(result.run_context.get_value("session_token"), secret)
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)
        self.assertEqual(result.test_run.failed_steps, [])
        self.assertEqual(result.test_run.blocked_steps, [])

    def test_pipeline_creates_a_distinct_run_context_for_each_run(self) -> None:
        first = self.pipeline.run("Open the example homepage.")
        second = self.pipeline.run("Open the example homepage.")

        self.assertIs(first.run_context, first.test_run.run_context)
        self.assertIs(second.run_context, second.test_run.run_context)
        self.assertIsNot(first.run_context, second.run_context)

    def test_pipeline_persists_automation_history_when_configured(self) -> None:
        repository = InMemoryRunHistoryRepository()
        history = RunHistoryService(repository)
        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=self.run_plan,
            run_history=history,
        )

        result = pipeline.run("Open the example homepage.")
        saved = history.get(result.test_run.id)

        self.assertIsNotNone(saved)
        self.assertEqual(saved.workflow_type, WorkflowType.AUTOMATION)
        self.assertEqual(saved.trace_id, result.trace.trace_id)
        self.assertEqual(saved.test_case_id, self.test_case.id)
        self.assertEqual(saved.executions[0].execution_id, result.executions[0].id)

    def test_run_test_case_executes_supplied_canonical_case_without_decomposing(self) -> None:
        result = self.pipeline.run_test_case(self.test_case)

        self.assertIs(result.test_case, self.test_case)
        self.assertEqual(
            [event[0] for event in self.events],
            ["discovery", "generator", "runner", "discovery", "generator", "runner"],
        )
        self.assertEqual([item.test_step_id for item in result.executions], [
            self.test_case.steps[1].id,
            self.test_case.steps[0].id,
        ])

    def test_failed_runner_evidence_is_attached_to_execution(self) -> None:
        def runner(plan: QATestPlan) -> dict[str, Any]:
            return {
                "status": "failed",
                "url": plan.url,
                "steps": [{"action": "assert_title", "status": "failed", "error": "wrong title"}],
                "evidence": [{
                    "type": "SCREENSHOT",
                    "path": "artifacts/failure.png",
                    "description": "Browser state after failed assertion.",
                }],
            }

        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=runner,
        )
        result = pipeline.run("Open the example homepage.")
        for execution in result.executions:
            self.assertEqual(execution.status, ExecutionStatus.FAILED)
            self.assertEqual(execution.error, "wrong title")
            self.assertEqual(execution.evidence[0].execution_id, execution.id)
            self.assertEqual(execution.evidence[0].type.value, "SCREENSHOT")
            self.assertEqual(execution.evidence[0].path, "artifacts/failure.png")

    def test_three_step_passes_are_returned_in_execution_order(self) -> None:
        ordered_steps = [self.make_step(order) for order in range(3)]
        test_case = DomainTestCase(
            name="Three checks",
            description="Run three independent checks.",
            base_url="https://example.com/",
            steps=[ordered_steps[2], ordered_steps[0], ordered_steps[1]],
        )
        generator = _FakeGenerator(self.events)
        runner_plans: list[QATestPlan] = []

        def runner(plan: QATestPlan) -> dict[str, Any]:
            runner_plans.append(plan)
            return {"status": "passed", "url": plan.url, "steps": []}

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(test_case, self.events),
            discovery=self.discover,
            plan_generator=generator,
            runner=runner,
        )

        result = pipeline.run("Run three checks.")

        expected_step_ids = [step.id for step in ordered_steps]
        self.assertEqual(len(result.executions), 3)
        self.assertEqual([pair.test_plan.test_step_id for pair in result.test_plans], expected_step_ids)
        self.assertEqual([run.test_step_id for run in result.executions], expected_step_ids)
        self.assertEqual(
            [run.test_plan_version_id for run in result.executions],
            [pair.test_plan_version.id for pair in result.test_plans],
        )
        self.assertEqual(len(runner_plans), 3)
        self.assertTrue(all(run.status == ExecutionStatus.PASSED for run in result.executions))

    def test_test_step_is_passed_to_discovery_and_generator(self) -> None:
        self.pipeline.run("Open the example homepage.")

        generated_steps = [event[1] for event in self.events if event[0] == "generator"]
        self.assertEqual([step.id for step in generated_steps], [self.steps[1].id, self.steps[0].id])
        self.assertEqual(
            [event[1] for event in self.events if event[0] == "discovery"],
            ["https://example.com/", "https://example.com/"],
        )

    def test_discovery_result_is_passed_to_plan_generator(self) -> None:
        self.pipeline.run("Open the example homepage.")

        results = [event[2] for event in self.events if event[0] == "generator"]
        self.assertEqual(results, [self.discovery_result, self.discovery_result])

    def test_generated_version_plan_is_passed_to_runner_and_mapped_to_execution(self) -> None:
        result = self.pipeline.run("Open the example homepage.")
        executions = result.executions

        generated_versions = self.generator.versions
        generated_plans = self.generator.plans
        self.assertEqual([pair.test_plan for pair in result.test_plans], generated_plans)
        self.assertEqual(
            [pair.test_plan_version.id for pair in result.test_plans],
            [pair.test_plan_version.id for pair in self.generator.generated],
        )
        self.assertTrue(all(
            pair.test_plan_version.origin == PlanVersionOrigin.AI_GENERATED
            for pair in result.test_plans
        ))
        self.assertEqual(self.runner_calls, [version.qa_test_plan for version in generated_versions])
        self.assertEqual(
            [run.test_plan_version_id for run in executions],
            [version.id for version in generated_versions],
        )
        ordered_steps = sorted(self.steps, key=lambda step: step.order)
        for step, plan, version, execution in zip(
            ordered_steps, generated_plans, generated_versions, executions
        ):
            self.assertEqual(plan.test_step_id, step.id)
            self.assertEqual(version.test_plan_id, plan.id)
            self.assertEqual(execution.test_step_id, step.id)
            self.assertEqual(execution.test_plan_version_id, version.id)
        self.assertEqual([run.actual_result for run in executions], ["passed", "passed"])
        self.assertEqual([run.runner_result["status"] for run in executions], ["passed", "passed"])
        self.assertTrue(all(run.finished_at is not None for run in executions))

    def test_decomposer_error_stops_pipeline_before_next_stages(self) -> None:
        decomposer_error = ValueError("invalid task")
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(error=decomposer_error, events=self.events),
            discovery=self.discover,
            plan_generator=self.generator,
            runner=self.run_plan,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("bad task")

        self.assertEqual(raised.exception.stage, "decomposition")
        self.assertIs(raised.exception.__cause__, decomposer_error)
        self.assertEqual([event[0] for event in self.events], ["decomposer"])

    def test_discovery_failure_does_not_call_generator(self) -> None:
        def failed_discovery(url: str) -> DiscoveryResult:
            return DiscoveryResult(
                status=DiscoveryStatus.FAILED,
                url=url,
                warnings=["browser unavailable"],
            )

        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=failed_discovery,
            plan_generator=self.generator,
            runner=self.run_plan,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Open the example homepage.")

        self.assertIn("discovery", raised.exception.stage)
        self.assertIn("browser unavailable", str(raised.exception))
        self.assertEqual(self.generator.calls, [])
        self.assertEqual(self.runner_calls, [])

    def test_discovery_exception_does_not_call_generator(self) -> None:
        discovery_error = RuntimeError("snapshot failed")

        def failed_discovery(url: str) -> DiscoveryResult:
            raise discovery_error

        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=failed_discovery,
            plan_generator=self.generator,
            runner=self.run_plan,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Open the example homepage.")

        self.assertIs(raised.exception.__cause__, discovery_error)
        self.assertEqual(self.generator.calls, [])

    def test_ai_discovery_fallback_failure_fails_at_discovery_stage(self) -> None:
        # P1-1: deterministic discovery is PARTIAL, the AI fallback is
        # invoked, and the fallback itself raises -> PipelineStageError.
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        store = InMemoryPlanStore()
        repository = InMemoryExecutionRepository()
        runner_calls: list[QATestPlan] = []

        def tracking_runner(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {"status": "passed", "steps": []}

        class _ExplodingFallback:
            def __init__(self) -> None:
                self.calls = 0

            def discover(
                self,
                task: str,
                target_url: str,
                test_step: DomainTestStep,
                deterministic_result: DiscoveryResult,
            ) -> Any:
                self.calls += 1
                raise RuntimeError("AI discovery provider failed")

        fallback = _ExplodingFallback()

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=lambda _: DiscoveryResult(
                status=DiscoveryStatus.PARTIAL,
                url="https://example.com/",
                warnings=["No navigation paths were discovered."],
            ),
            discovery_fallback=fallback,
            plan_generator=generator,
            runner=tracking_runner,
            plan_store=store,
            execution_repository=repository,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Check")

        error = raised.exception
        self.assertIn("discovery", error.stage)

        trace = error.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("discovery", trace.error_stage or "")

        # The AI fallback was genuinely invoked exactly once...
        self.assertEqual(fallback.calls, 1)

        # ...and the trace now proves it: the failed fallback attempt is
        # recorded (invoked=True + error) and never reported as a success,
        # and the deterministic discovery that triggered it is visible too.
        step_trace = trace.steps[0]
        fallback_trace = step_trace.discovery_fallback
        self.assertIsNotNone(fallback_trace)
        self.assertTrue(fallback_trace.invoked)
        self.assertEqual(fallback_trace.error, "AI discovery provider failed")
        self.assertEqual(fallback_trace.navigation_paths_added, 0)
        self.assertEqual(fallback_trace.direct_navigation_paths_added, 0)
        self.assertEqual(fallback_trace.interactive_elements_added, 0)
        self.assertIsNotNone(fallback_trace.duration_ms)

        # The deterministic discovery (PARTIAL) is recorded exactly once.
        self.assertEqual(len(step_trace.discoveries), 1)
        discovery_trace = step_trace.discoveries[0]
        self.assertEqual(discovery_trace.status, DiscoveryStatus.PARTIAL)
        self.assertEqual(discovery_trace.url, "https://example.com/")
        self.assertIn(
            "No navigation paths were discovered.", discovery_trace.warnings
        )

        # Nothing downstream happened.
        self.assertEqual(generator.calls, [])
        self.assertEqual(generator.generated, [])
        self.assertIsNone(store.find(step.id))
        self.assertEqual(runner_calls, [])
        self.assertEqual(repository.list_for_test_step(step.id), [])
        self.assertEqual(trace.totals.execution_attempts, 0)
        self.assertEqual(trace.totals.provider_attempts, 0)

    def test_stale_path_rediscovery_failed_halts_without_ai_fallback(self) -> None:
        # P1-2: cached v1 fails with a stale UI error, the stale-path
        # rediscovery returns FAILED, and the run dies at "rediscovery".
        # Characterizes that rediscovery never consults the AI fallback.
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=cached_plan.id,
            version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})]),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version_one, test_plan=cached_plan)
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        runner_calls: list[QATestPlan] = []

        def stale_then_stop(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {"status": "failed", "steps": [{
                "action": "click", "status": "failed",
                "error": "Selector '#old' was not found on the page."}]}

        class _RecordingFallback:
            def __init__(self) -> None:
                self.calls = 0

            def discover(self, *args: Any, **kwargs: Any) -> Any:
                self.calls += 1
                raise AssertionError("AI fallback must not run here")

        fallback = _RecordingFallback()

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=lambda _: DiscoveryResult(
                status=DiscoveryStatus.FAILED,
                url="https://example.com/",
                warnings=["Page never loaded."],
            ),
            discovery_fallback=fallback,
            plan_generator=generator,
            runner=stale_then_stop,
            plan_store=store,
            execution_repository=repository,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Check")

        error = raised.exception
        self.assertIn("rediscovery", error.stage)

        trace = error.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("rediscovery", trace.error_stage or "")

        # Matrix behavior: stale-path rediscovery does NOT use AI fallback.
        self.assertEqual(fallback.calls, 0)

        # The original v1 failed Execution remains persisted.
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].status, ExecutionStatus.FAILED)
        self.assertEqual(persisted[0].test_plan_version_id, version_one.id)

        # No v2, no regeneration, runner not called again.
        current = store.find(step.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.version, 1)
        self.assertEqual(current.id, version_one.id)
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(runner_calls), 1)

        # Trace kept the cache HIT and the single v1 attempt. The FAILED
        # rediscovery attempt is now visible in the trace (recorded before
        # the pipeline raises), while recovery/regeneration never ran.
        step_trace = trace.steps[0]
        self.assertTrue(step_trace.plan_cache.hit)
        self.assertEqual(len(step_trace.execution_attempts), 1)
        self.assertIsNone(step_trace.locator_recovery)
        self.assertIsNone(step_trace.regeneration)

        # The FAILED rediscovery attempt is present in the trace exactly once.
        self.assertEqual(len(step_trace.discoveries), 1)
        rediscovery_trace = step_trace.discoveries[0]
        self.assertEqual(rediscovery_trace.status, DiscoveryStatus.FAILED)
        self.assertEqual(rediscovery_trace.url, "https://example.com/")
        self.assertIn("Page never loaded.", rediscovery_trace.warnings)

        self.assertEqual(trace.totals.regenerations, 0)
        self.assertEqual(trace.totals.locator_recoveries, 0)

    def test_stale_path_rediscovery_exception_is_recorded_and_halts(self) -> None:
        # P1-2 regression (exception branch): the stale-path rediscovery
        # callable itself raises, so no DiscoveryResult object ever exists.
        # The pipeline must synthesize and record exactly one FAILED
        # rediscovery attempt in the trace before aborting at "rediscovery",
        # with the original exception preserved as the cause and no
        # recovery, regeneration, or AI fallback afterwards.
        boom = RuntimeError("rediscovery exploded")
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=cached_plan.id,
            version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})]),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version_one, test_plan=cached_plan)
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        discovery_calls: list[str] = []
        runner_calls: list[QATestPlan] = []

        def exploding_rediscovery(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            raise boom

        def stale_then_stop(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {"status": "failed", "steps": [{
                "action": "click", "status": "failed",
                "error": "Selector '#old' was not found on the page."}]}

        class _RecordingFallback:
            def __init__(self) -> None:
                self.calls = 0

            def discover(self, *args: Any, **kwargs: Any) -> Any:
                self.calls += 1
                raise AssertionError("AI fallback must not run here")

        fallback = _RecordingFallback()

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=exploding_rediscovery,
            discovery_fallback=fallback,
            plan_generator=generator,
            runner=stale_then_stop,
            plan_store=store,
            execution_repository=repository,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Check")

        # 1) Rediscovery was attempted exactly once, against the target URL
        # (the cache HIT skips initial discovery, so this is the only call).
        self.assertEqual(len(discovery_calls), 1)
        self.assertEqual(discovery_calls[0], "https://example.com/")

        error = raised.exception
        # 2) The run fails at the rediscovery stage...
        self.assertIn("rediscovery", error.stage)
        # 6) ...with the original exception preserved as the cause.
        self.assertIs(error.__cause__, boom)

        trace = error.trace
        self.assertIsNotNone(trace)
        # 3) Trace status is ERROR.
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("rediscovery", trace.error_stage or "")

        # 4)/5) The synthesized FAILED rediscovery attempt is visible in the
        # trace exactly once, carrying the target URL and the failure reason.
        step_trace = trace.steps[0]
        self.assertEqual(len(step_trace.discoveries), 1)
        rediscovery_trace = step_trace.discoveries[0]
        self.assertEqual(rediscovery_trace.status, DiscoveryStatus.FAILED)
        self.assertEqual(rediscovery_trace.url, "https://example.com/")
        self.assertTrue(
            any(
                "rediscovery exploded" in warning
                for warning in rediscovery_trace.warnings
            )
        )

        # 7) Locator recovery never ran.
        self.assertIsNone(step_trace.locator_recovery)
        self.assertEqual(trace.totals.locator_recoveries, 0)

        # 8) The AI discovery fallback was never consulted.
        self.assertEqual(fallback.calls, 0)

        # 9) No regeneration happened.
        self.assertIsNone(step_trace.regeneration)
        self.assertEqual(trace.totals.regenerations, 0)
        self.assertEqual(generator.calls, [])

        # 10) Nothing new was created: only the original v1 execution and
        # version exist, and the runner was invoked exactly once (the stale
        # v1 attempt that triggered the rediscovery).
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].status, ExecutionStatus.FAILED)
        self.assertEqual(persisted[0].test_plan_version_id, version_one.id)
        current = store.find(step.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.version, 1)
        self.assertEqual(current.id, version_one.id)
        self.assertEqual(len(runner_calls), 1)
        self.assertEqual(len(step_trace.execution_attempts), 1)
        self.assertTrue(step_trace.plan_cache.hit)

    def test_repaired_execution_failing_again_does_not_cascade(self) -> None:
        # P1-3: v1 stale failure -> MATCHED deterministic repair creates v2
        # -> v2 fails AGAIN with a stale UI error. The pipeline must NOT
        # enter a second recovery/regeneration cascade (no infinite loop).
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=cached_plan.id,
            version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})]),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version_one, test_plan=cached_plan)
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        discovery_calls: list[str] = []
        runner_calls: list[QATestPlan] = []

        def rediscover(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            return DiscoveryResult(
                status=DiscoveryStatus.SUCCESS,
                url=url,
                interactive_elements=[InteractiveElement(
                    kind="button", tag="button", role="button",
                    text="Continue", selector="#new",
                )],
            )

        def stale_twice(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            selector = plan.steps[0].parameters["selector"]
            return {"status": "failed", "steps": [{
                "action": "click", "status": "failed",
                "error": f"Selector '{selector}' was not found on the page."}]}

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=rediscover,
            plan_generator=generator,
            runner=stale_twice,
            plan_store=store,
            execution_repository=repository,
        )

        result = pipeline.run("Check")

        # Both attempts failed; final status is FAILED, not a hang or crash.
        self.assertEqual(
            [execution.status for execution in result.executions],
            [ExecutionStatus.FAILED, ExecutionStatus.FAILED],
        )
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)

        # v1 used version 1, repaired v2 used version 2; no v3 anywhere.
        self.assertEqual(
            [plan.test_plan_version.version for plan in result.test_plans],
            [1, 2],
        )
        self.assertEqual(
            [
                execution.test_plan_version_id
                for execution in result.executions
            ],
            [
                result.test_plans[0].test_plan_version.id,
                result.test_plans[1].test_plan_version.id,
            ],
        )
        current = store.find(step.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.version, 2)

        # Both executions are persisted.
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 2)
        self.assertEqual(
            [execution.status for execution in persisted],
            [ExecutionStatus.FAILED, ExecutionStatus.FAILED],
        )

        # No cascade: one rediscovery, one repaired run, no regeneration.
        self.assertEqual(len(discovery_calls), 1)
        self.assertEqual(len(runner_calls), 2)
        self.assertEqual(generator.calls, [])

        # Trace: two attempts, exactly one (MATCHED) recovery, zero
        # regenerations; repaired v2 selector was "#new".
        trace = result.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.FAILED)
        step_trace = trace.steps[0]
        self.assertEqual(len(step_trace.execution_attempts), 2)
        self.assertIsNotNone(step_trace.locator_recovery)
        self.assertEqual(
            step_trace.locator_recovery.status, RecoveryStatus.MATCHED
        )
        self.assertEqual(step_trace.locator_recovery.candidate_selector, "#new")
        self.assertEqual(trace.totals.locator_recoveries, 1)
        self.assertIsNone(step_trace.regeneration)
        self.assertEqual(trace.totals.regenerations, 0)

    def test_assertion_failure_after_regeneration_does_not_recover_again(
        self,
    ) -> None:
        # P1-4: v1 stale failure -> NOT_FOUND recovery -> LLM regeneration
        # creates v2 -> v2 fails with an ordinary assertion failure, which
        # must NOT trigger another recovery/regeneration.
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=cached_plan.id,
            version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})]),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version_one, test_plan=cached_plan)
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        discovery_calls: list[str] = []
        runner_calls: list[QATestPlan] = []

        def rediscover(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            # No matching element -> deterministic recovery is NOT_FOUND.
            return DiscoveryResult(
                status=DiscoveryStatus.SUCCESS,
                url=url,
                interactive_elements=[],
            )

        def stale_then_assertion(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            if len(runner_calls) == 1:
                return {"status": "failed", "steps": [{
                    "action": "click", "status": "failed",
                    "error": "Selector '#old' was not found on the page."}]}
            # Regenerated v2 fails with a plain assertion failure.
            return {"status": "failed", "steps": [{
                "action": "assert_visible", "status": "failed",
                "error": "Expected heading 'Welcome' was not visible."}]}

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=rediscover,
            plan_generator=generator,
            runner=stale_then_assertion,
            plan_store=store,
            execution_repository=repository,
        )

        result = pipeline.run("Check")

        # Unsafe rediscovery leaves v1 failed and does not create or run v2.
        self.assertEqual(
            [execution.status for execution in result.executions],
            [ExecutionStatus.FAILED],
        )
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(
            [
                execution.test_plan_version_id
                for execution in result.executions
            ],
            [version_one.id],
        )
        self.assertEqual([plan.test_plan_version.version for plan in result.test_plans], [1])
        current = store.find(step.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.id, version_one.id)
        self.assertEqual(current.version, 1)

        # Only rediscovery ran. No LLM or retry followed the stale failure.
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(discovery_calls), 1)
        self.assertEqual(len(runner_calls), 1)

        trace = result.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.FAILED)
        step_trace = trace.steps[0]
        self.assertEqual(len(step_trace.execution_attempts), 1)
        self.assertIsNone(step_trace.regeneration)
        self.assertEqual(trace.totals.regenerations, 0)

        self.assertIsNotNone(step_trace.locator_recovery)
        self.assertEqual(
            step_trace.locator_recovery.status, RecoveryStatus.NO_MATCH
        )
        self.assertEqual(trace.totals.locator_recoveries, 1)

        self.assertIsNone(step_trace.plan_generation)

    def test_generator_error_does_not_call_runner(self) -> None:
        generator_error = RuntimeError("LLM unavailable")
        generator = _FakeGenerator(self.events, error=generator_error)
        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=generator,
            runner=self.run_plan,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Open the example homepage.")

        self.assertIn("plan generation", raised.exception.stage)
        self.assertIs(raised.exception.__cause__, generator_error)
        self.assertEqual(self.runner_calls, [])

    def test_provider_exhaustion_fails_at_plan_generation_and_traces_attempts(
        self,
    ) -> None:
        # P0-3 characterization: every provider fails retryably, so the real
        # LLMRouter exhausts its chain and the run dies at plan generation
        # with full provider-attempt evidence in the trace.
        class _RetryableFailProvider(LLMProvider):
            def __init__(self, name: str) -> None:
                self.name = name
                self.model = f"{name}-model"

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                raise RetryableLLMError(f"{self.name} is down (HTTP 503)")

        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        events: list[tuple[Any, ...]] = []
        store = InMemoryPlanStore()
        repository = InMemoryExecutionRepository()
        runner_calls: list[QATestPlan] = []

        def tracking_runner(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {"status": "passed", "steps": []}

        router = LLMRouter([
            _RetryableFailProvider("stub_one"),
            _RetryableFailProvider("stub_two"),
        ])
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=lambda _: DiscoveryResult(
                status=DiscoveryStatus.SUCCESS, url="https://example.com/"
            ),
            plan_generator=LLMTestPlanGenerator(router),
            runner=tracking_runner,
            plan_store=store,
            execution_repository=repository,
        )

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(PipelineStageError) as raised:
                pipeline.run("Check")

        error = raised.exception
        self.assertIn("plan generation", error.stage)

        trace = error.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("plan generation", trace.error_stage or "")
        self.assertIn("All LLM providers failed", trace.error or "")

        # Exactly two retryable attempts; neither provider was selected.
        step_trace = trace.steps[0]
        attempts = step_trace.provider_attempts
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            [attempt.outcome for attempt in attempts],
            [
                ProviderAttemptOutcome.RETRYABLE_ERROR,
                ProviderAttemptOutcome.RETRYABLE_ERROR,
            ],
        )
        self.assertEqual(
            [attempt.provider_name for attempt in attempts],
            ["stub_one", "stub_two"],
        )
        self.assertFalse(any(attempt.is_selected for attempt in attempts))
        self.assertEqual(trace.totals.provider_attempts, 2)

        # No plan version, no execution, and the runner never ran.
        self.assertIsNone(step_trace.plan_generation)
        self.assertIsNone(store.find(step.id))
        self.assertEqual(runner_calls, [])
        self.assertEqual(repository.list_for_test_step(step.id), [])
        self.assertEqual(len(step_trace.execution_attempts), 0)
        self.assertEqual(trace.totals.execution_attempts, 0)
        self.assertIsNone(step_trace.locator_recovery)
        self.assertIsNone(step_trace.regeneration)

    def test_generic_provider_exception_is_unclassified_and_skips_fallback(
        self,
    ) -> None:
        # P2-2 regression: the router only classifies typed
        # RetryableLLMError/NonRetryableLLMError. A generic RuntimeError from
        # a provider stays UNCLASSIFIED — no fallback, identity preserved —
        # but is now recorded as exactly one UNCLASSIFIED_ERROR provider
        # attempt; the raw exception propagates and the run dies at plan
        # generation.
        from qa_agent.llm.errors import NonRetryableLLMError

        prebuilt_error = RuntimeError("unexpected provider failure")

        class _GenericFailProvider(LLMProvider):
            def __init__(self) -> None:
                self.name = "stub_generic"
                self.model = "stub-generic-model"
                self.calls = 0

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                self.calls += 1
                raise prebuilt_error

        class _SuccessProvider(LLMProvider):
            def __init__(self) -> None:
                self.name = "stub_second"
                self.model = "stub-second-model"
                self.calls = 0

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                self.calls += 1
                return QATestPlan(url=target_url, steps=[
                    QATestStep(action="assert_page_loaded", parameters={}),
                ])

        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        events: list[tuple[Any, ...]] = []
        store = InMemoryPlanStore()
        repository = InMemoryExecutionRepository()
        runner_calls: list[QATestPlan] = []

        def tracking_runner(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {"status": "passed", "steps": []}

        first_provider = _GenericFailProvider()
        second_provider = _SuccessProvider()

        router = LLMRouter([first_provider, second_provider])
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=lambda _: DiscoveryResult(
                status=DiscoveryStatus.SUCCESS, url="https://example.com/"
            ),
            plan_generator=LLMTestPlanGenerator(router),
            runner=tracking_runner,
            plan_store=store,
            execution_repository=repository,
        )

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(PipelineStageError) as raised:
                pipeline.run("Check")

        error = raised.exception
        self.assertIn("plan generation", error.stage)

        # The raw generic exception reaches the pipeline unclassified.
        self.assertIsNotNone(error.__cause__)
        self.assertIs(error.__cause__, prebuilt_error)
        self.assertEqual(str(error.__cause__), "unexpected provider failure")
        self.assertNotIsInstance(error.__cause__, (RetryableLLMError, NonRetryableLLMError))

        # No provider fallback: the second provider was never invoked.
        self.assertEqual(first_provider.calls, 1)
        self.assertEqual(second_provider.calls, 0)

        # Trace evidence: the attempted provider is recorded exactly once as
        # an UNCLASSIFIED failure — observable without being reclassified.
        trace = error.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("plan generation", trace.error_stage or "")
        self.assertIn("unexpected provider failure", trace.error or "")
        step_trace = trace.steps[0]
        attempts = step_trace.provider_attempts
        self.assertEqual(len(attempts), 1)
        self.assertEqual(
            attempts[0].outcome, ProviderAttemptOutcome.UNCLASSIFIED_ERROR
        )
        self.assertEqual(attempts[0].error_class, "RuntimeError")
        self.assertEqual(
            attempts[0].error_message, "unexpected provider failure"
        )
        self.assertFalse(attempts[0].is_selected)
        self.assertEqual(trace.totals.provider_attempts, 1)

        # No plan version, no execution, runner never ran.
        self.assertIsNone(step_trace.plan_generation)
        self.assertIsNone(store.find(step.id))
        self.assertEqual(runner_calls, [])
        self.assertEqual(repository.list_for_test_step(step.id), [])
        self.assertEqual(len(step_trace.execution_attempts), 0)

    def test_incompatible_generated_plan_fails_after_one_repair_without_version(
        self,
    ) -> None:
        # P2-4 characterization: an LLM plan that is schema-valid but
        # incompatible with discovered capabilities (assert_selected on a
        # non-radio/non-select input) is rejected inside the real
        # LLMTestPlanGenerator after the router already recorded a
        # SUCCESS+selected provider attempt. No TestPlanVersion, no
        # Execution, no plan_generation trace row.
        class _IncompatiblePlanProvider(LLMProvider):
            def __init__(self) -> None:
                self.name = "stub_incompatible"
                self.model = "stub-incompatible-model"
                self.calls = 0

            @property
            def is_available(self) -> bool:
                return True

            def create_test_plan(
                self, task: str, target_url: str, page_snapshot: str
            ) -> QATestPlan:
                self.calls += 1
                return QATestPlan(url=target_url, steps=[
                    QATestStep(
                        action="assert_selected",
                        parameters={"selector": "#email"},
                    ),
                ])

        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        events: list[tuple[Any, ...]] = []
        store = InMemoryPlanStore()
        repository = InMemoryExecutionRepository()
        runner_calls: list[QATestPlan] = []
        discovery_calls: list[str] = []

        def tracking_runner(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {"status": "passed", "steps": []}

        def tracking_discovery(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            return DiscoveryResult(
                status=DiscoveryStatus.SUCCESS,
                url="https://example.com/",
                interactive_elements=[InteractiveElement(
                    kind="input",
                    selector="#email",
                    tag="input",
                    role="textbox",
                    accessible_name="Email address",
                )],
            )

        provider = _IncompatiblePlanProvider()
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=tracking_discovery,
            plan_generator=LLMTestPlanGenerator(LLMRouter([provider])),
            runner=tracking_runner,
            plan_store=store,
            execution_repository=repository,
        )

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(PipelineStageError) as raised:
                pipeline.run("Check")

        error = raised.exception
        self.assertIn("plan generation", error.stage)
        # The capability-validation ValueError is the root cause.
        self.assertIsInstance(error.__cause__, ValueError)
        self.assertIn("does not target a radio or select", str(error.__cause__))

        trace = error.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("plan generation", trace.error_stage or "")

        # Both generation and the single bounded repair went through the
        # configured router and recorded the provider response before the
        # capability validator rejected it.
        step_trace = trace.steps[0]
        attempts = step_trace.provider_attempts
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(attempt.outcome == ProviderAttemptOutcome.SUCCESS for attempt in attempts))
        self.assertTrue(all(attempt.is_selected for attempt in attempts))

        # Rejected before any version or execution exists.
        self.assertIsNone(step_trace.plan_generation)
        self.assertIsNone(store.find(step.id))
        self.assertEqual(repository.list_for_test_step(step.id), [])
        self.assertEqual(runner_calls, [])
        self.assertEqual(len(step_trace.execution_attempts), 0)

        # Discovery ran once; semantic repair is bounded to one extra router call.
        self.assertEqual(len(discovery_calls), 1)
        self.assertEqual(provider.calls, 2)

    def test_runner_error_is_propagated_with_execution_stage_context(self) -> None:
        runner_error = RuntimeError("browser launch failed")
        repository = InMemoryExecutionRepository()

        def failed_runner(plan: QATestPlan) -> dict[str, Any]:
            raise runner_error

        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=failed_runner,
            execution_repository=repository,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Open the example homepage.")

        self.assertIn("execution", raised.exception.stage)
        self.assertIs(raised.exception.__cause__, runner_error)
        failed_attempts = repository.list_for_test_step(self.steps[1].id)
        self.assertEqual(len(failed_attempts), 1)
        self.assertEqual(failed_attempts[0].status, ExecutionStatus.FAILED)
        self.assertEqual(failed_attempts[0].error, "browser launch failed")
        self.assertEqual(
            failed_attempts[0].test_plan_version_id,
            self.generator.versions[0].id,
        )

    def test_runner_exception_persists_execution_and_records_trace_attempt(
        self,
    ) -> None:
        # P0-2 regression: when the runner raises, the FAILED Execution is
        # persisted AND mirrored by exactly one trace execution attempt
        # (recorded inside _execute_plan after persistence, before raising).
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        repository = InMemoryExecutionRepository()
        store = InMemoryPlanStore()

        def exploding_runner(plan: QATestPlan) -> dict[str, Any]:
            raise RuntimeError("browser launch failed")

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=lambda _: DiscoveryResult(
                status=DiscoveryStatus.SUCCESS, url="https://example.com/"
            ),
            plan_generator=generator,
            runner=exploding_runner,
            plan_store=store,
            execution_repository=repository,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Check")

        error = raised.exception
        self.assertIn("execution", error.stage)

        trace = error.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("execution", trace.error_stage or "")

        # Generation succeeded first and was recorded before the runner raised.
        step_trace = trace.steps[0]
        self.assertIsNotNone(step_trace.plan_generation)
        self.assertEqual(step_trace.plan_generation.version_number, 1)
        self.assertIsNotNone(step_trace.plan_cache)
        self.assertFalse(step_trace.plan_cache.hit)

        # INVARIANT: repository and trace are synchronized — every persisted
        # Execution has exactly one corresponding trace execution attempt.
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].status, ExecutionStatus.FAILED)
        self.assertEqual(persisted[0].error, "browser launch failed")
        self.assertEqual(
            persisted[0].test_plan_version_id, generator.versions[0].id
        )

        attempts = step_trace.execution_attempts
        self.assertEqual(len(attempts), 1)
        self.assertEqual(trace.totals.execution_attempts, 1)
        self.assertEqual(attempts[0].execution_id, persisted[0].id)
        self.assertEqual(
            attempts[0].plan_version_id, persisted[0].test_plan_version_id
        )
        self.assertEqual(attempts[0].plan_version_number, 1)
        self.assertEqual(attempts[0].status, ExecutionStatus.FAILED)
        self.assertEqual(attempts[0].error, "browser launch failed")
        self.assertEqual(len(persisted), len(attempts))

    def test_runner_exception_on_repaired_execution_records_both_trace_attempts(
        self,
    ) -> None:
        # P0-2 regression 1: stale v1 -> deterministic locator recovery
        # MATCHED -> repaired v2 runner raises. Both the v1 and the
        # repaired v2 executions must each be mirrored by exactly one
        # trace execution attempt.
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=cached_plan.id,
            version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})]),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version_one, test_plan=cached_plan)
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        runner_calls: list[QATestPlan] = []

        def rediscover(url: str) -> DiscoveryResult:
            return DiscoveryResult(
                status=DiscoveryStatus.SUCCESS,
                url=url,
                interactive_elements=[InteractiveElement(
                    kind="button", tag="button", role="button",
                    text="Continue", selector="#new",
                )],
            )

        def stale_then_raise(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            if len(runner_calls) == 1:
                selector = plan.steps[0].parameters["selector"]
                return {"status": "failed", "steps": [{
                    "action": "click", "status": "failed",
                    "error": f"Selector '{selector}' was not found on the page."}]}
            raise RuntimeError("browser launch failed")

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=rediscover,
            plan_generator=generator,
            runner=stale_then_raise,
            plan_store=store,
            execution_repository=repository,
        )

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Check")

        # The pipeline ends with an execution-stage error for the REPAIRED
        # v2 execution, not for v1.
        error = raised.exception
        self.assertIn("execution", error.stage)
        self.assertIn("plan version 2", error.stage)

        trace = error.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.ERROR)
        self.assertIn("execution", trace.error_stage or "")

        # Repository holds both FAILED executions: v1 stale failure and
        # the persisted v2 runner exception.
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 2)
        self.assertEqual(
            [execution.status for execution in persisted],
            [ExecutionStatus.FAILED, ExecutionStatus.FAILED],
        )

        # Deterministic recovery was MATCHED; no regeneration happened.
        step_trace = trace.steps[0]
        self.assertIsNotNone(step_trace.locator_recovery)
        self.assertEqual(
            step_trace.locator_recovery.status, RecoveryStatus.MATCHED
        )
        self.assertEqual(trace.totals.locator_recoveries, 1)
        self.assertIsNone(step_trace.regeneration)
        self.assertEqual(trace.totals.regenerations, 0)
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(runner_calls), 2)

        # Trace mirrors the repository exactly: v1 attempt, then v2 attempt.
        attempts = step_trace.execution_attempts
        self.assertEqual(len(attempts), 2)
        self.assertEqual(trace.totals.execution_attempts, 2)
        self.assertEqual(attempts[0].plan_version_number, 1)
        self.assertEqual(attempts[0].execution_id, persisted[0].id)
        self.assertEqual(attempts[0].status, ExecutionStatus.FAILED)
        self.assertEqual(attempts[1].plan_version_number, 2)
        self.assertEqual(attempts[1].execution_id, persisted[1].id)
        self.assertEqual(
            attempts[1].plan_version_id, persisted[1].test_plan_version_id
        )
        self.assertEqual(attempts[1].status, ExecutionStatus.FAILED)
        self.assertEqual(attempts[1].error, "browser launch failed")
        self.assertEqual(len(persisted), len(attempts))

        # The repaired v2 is the version bound to the persisted execution.
        current = store.find(step.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.version, 2)
        self.assertEqual(persisted[1].test_plan_version_id, current.id)

    def test_unsafe_recovery_does_not_attempt_a_second_execution(
        self,
    ) -> None:
        # Stale v1 with no saved identity must stop as drift without LLM
        # regeneration or a second browser execution.
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        cached_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=cached_plan.id,
            version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})]),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version_one, test_plan=cached_plan)
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        runner_calls: list[QATestPlan] = []

        def rediscover(url: str) -> DiscoveryResult:
            # No matching element -> deterministic recovery is NOT_FOUND.
            return DiscoveryResult(
                status=DiscoveryStatus.SUCCESS,
                url=url,
                interactive_elements=[],
            )

        def stale_then_raise(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            selector = plan.steps[0].parameters["selector"]
            return {"status": "failed", "steps": [{
                "action": "click", "status": "failed",
                "error": f"Selector '{selector}' was not found on the page."}]}

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=rediscover,
            plan_generator=generator,
            runner=stale_then_raise,
            plan_store=store,
            execution_repository=repository,
        )

        result = pipeline.run("Check")
        trace = result.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, TraceStatus.FAILED)
        self.assertIsNone(trace.error_stage)

        # Only the original stale failure is persisted.
        persisted = repository.list_for_test_step(step.id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual([execution.status for execution in persisted], [ExecutionStatus.FAILED])

        step_trace = trace.steps[0]
        self.assertIsNone(step_trace.regeneration)
        self.assertEqual(trace.totals.regenerations, 0)
        self.assertEqual(generator.calls, [])
        self.assertIsNotNone(step_trace.locator_recovery)
        self.assertEqual(
            step_trace.locator_recovery.status, RecoveryStatus.NO_MATCH
        )
        self.assertEqual(trace.totals.locator_recoveries, 1)
        self.assertEqual(len(runner_calls), 1)

        attempts = step_trace.execution_attempts
        self.assertEqual(len(attempts), 1)
        self.assertEqual(trace.totals.execution_attempts, 1)
        self.assertEqual(attempts[0].plan_version_number, 1)
        self.assertEqual(attempts[0].execution_id, persisted[0].id)
        self.assertEqual(attempts[0].status, ExecutionStatus.FAILED)
        self.assertEqual(len(persisted), len(attempts))

        current = store.find(step.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.id, version_one.id)

    def test_runner_failed_result_becomes_failed_execution(self) -> None:
        def failed_runner(plan: QATestPlan) -> dict[str, Any]:
            return {
                "status": "failed",
                "url": plan.url,
                "steps": [{"action": "click", "status": "failed", "error": "not found"}],
            }

        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=failed_runner,
        )

        executions = pipeline.run("Open the example homepage.").executions

        self.assertEqual([item.status for item in executions], [ExecutionStatus.FAILED] * 2)
        self.assertEqual([item.error for item in executions], ["not found"] * 2)

    def test_failed_step_with_continue_policy_executes_subsequent_steps(self) -> None:
        steps = [
            DomainTestStep(
                name="Step 0",
                description="Perform action 0",
                expected="Action 0 is completed",
                order=0,
                failure_policy=FailurePolicy.CONTINUE,
            ),
            self.make_step(1),
            self.make_step(2),
        ]
        case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com/",
            steps=steps,
        )
        runner_calls: list[QATestPlan] = []

        def runner(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            if len(runner_calls) == 1:
                return {
                    "status": "failed",
                    "url": plan.url,
                    "steps": [{
                        "action": "assert_title",
                        "status": "failed",
                        "error": "wrong title",
                    }],
                }
            return {"status": "passed", "url": plan.url, "steps": []}

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, self.events),
            discovery=self.discover,
            plan_generator=self.generator,
            runner=runner,
        )

        result = pipeline.run("Check the example page.")

        # FAIL + CONTINUE: the later steps are executed normally.
        self.assertEqual(len(runner_calls), 3)
        self.assertEqual(
            [execution.status for execution in result.executions],
            [ExecutionStatus.FAILED, ExecutionStatus.PASSED, ExecutionStatus.PASSED],
        )
        self.assertEqual(
            [execution.test_step_id for execution in result.executions],
            [steps[0].id, steps[1].id, steps[2].id],
        )
        # An ordinary assertion failure never enters the recovery path.
        self.assertEqual(
            sum(event[0] == "discovery" for event in self.events), 3
        )
        # The original failed step remains FAILED; nothing is blocked.
        self.assertEqual(
            result.test_run.final_execution_for_step(steps[0].id).status,
            ExecutionStatus.FAILED,
        )
        self.assertEqual(result.test_run.failed_steps, [steps[0].id])
        self.assertEqual(result.blocked_step_ids, [])
        self.assertEqual(result.test_run.blocked_steps, [])
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)

    def test_failed_step_with_block_rest_policy_blocks_subsequent_steps(self) -> None:
        steps = [
            DomainTestStep(
                name="Step 0",
                description="Perform action 0",
                expected="Action 0 is completed",
                order=0,
                failure_policy=FailurePolicy.BLOCK_REST,
            ),
            self.make_step(1),
            self.make_step(2),
        ]
        case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com/",
            steps=steps,
        )
        runner_calls: list[QATestPlan] = []

        def runner(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {
                "status": "failed",
                "url": plan.url,
                "steps": [{
                    "action": "assert_title",
                    "status": "failed",
                    "error": "wrong title",
                }],
            }

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, self.events),
            discovery=self.discover,
            plan_generator=self.generator,
            runner=runner,
        )

        result = pipeline.run("Check the example page.")

        # FAIL + BLOCK_REST: subsequent steps are never executed.
        self.assertEqual(len(runner_calls), 1)
        self.assertEqual(len(result.executions), 1)
        # The original failed step remains FAILED (it really executed).
        self.assertEqual(result.executions[0].status, ExecutionStatus.FAILED)
        self.assertEqual(result.executions[0].test_step_id, steps[0].id)
        # Subsequent steps are marked BLOCKED, in step order, with no Execution.
        self.assertEqual(result.blocked_step_ids, [steps[1].id, steps[2].id])
        self.assertEqual(
            result.test_run.blocked_steps, [steps[1].id, steps[2].id]
        )
        self.assertIsNone(
            result.test_run.final_execution_for_step(steps[1].id)
        )
        self.assertIsNone(
            result.test_run.final_execution_for_step(steps[2].id)
        )
        # FAILED and BLOCKED remain distinguishable step outcomes.
        self.assertEqual(result.test_run.failed_steps, [steps[0].id])
        self.assertTrue(
            set(result.test_run.failed_steps).isdisjoint(
                result.test_run.blocked_steps
            )
        )
        # The final TestCase status is FAILED in both policies.
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)

    def test_block_rest_policy_does_not_replace_infrastructure_abort(self) -> None:
        steps = [
            DomainTestStep(
                name="Step 0",
                description="Perform action 0",
                expected="Action 0 is completed",
                order=0,
                failure_policy=FailurePolicy.BLOCK_REST,
            ),
            self.make_step(1),
        ]
        case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com/",
            steps=steps,
        )
        runner = unittest.mock.Mock(side_effect=RuntimeError("browser crashed"))
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, self.events),
            discovery=self.discover,
            plan_generator=self.generator,
            runner=runner,
        )

        # Infrastructure failures still abort the run: no PipelineResult
        # (and therefore no TestRun with BLOCKED steps) is produced, and the
        # failure policy never converts an abort into blocked steps.
        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("Check the example page.")

        self.assertTrue(
            raised.exception.stage.startswith("execution (step 0")
        )
        self.assertEqual(runner.call_count, 1)

    def test_stale_ui_failure_without_locator_identity_stays_drift(self) -> None:
        one_step_case = DomainTestCase(
            name="Example flow",
            description="Open the example homepage.",
            base_url="https://example.com/",
            steps=[self.make_step(0)],
        )
        decomposer = _FakeDecomposer(one_step_case, self.events)
        runner_calls = 0

        def fail_once_on_missing_selector(plan: QATestPlan) -> dict[str, Any]:
            nonlocal runner_calls
            runner_calls += 1
            if runner_calls == 1:
                return {
                    "status": "failed",
                    "url": plan.url,
                    "steps": [{
                        "action": "click",
                        "status": "failed",
                        "error": (
                            "Could not click element matching selector '#old-menu': "
                            "Timeout 5000ms exceeded while waiting for locator('#old-menu')."
                        ),
                    }],
                }
            return {"status": "passed", "url": plan.url, "steps": []}

        pipeline = QATestPipeline(
            decomposer=decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=fail_once_on_missing_selector,
        )

        result = pipeline.run("Open the example homepage.")

        self.assertEqual([entry.test_plan_version.version for entry in result.test_plans], [1])
        self.assertEqual([run.status for run in result.executions], [ExecutionStatus.FAILED])
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)
        self.assertIsNone(result.trace.steps[0].locator_recovery)
        self.assertIsNone(result.trace.steps[0].regeneration)
        self.assertEqual(runner_calls, 1)
        self.assertEqual(
            [event[0] for event in self.events],
            ["decomposer", "discovery", "generator", "discovery"],
        )

    def test_assertion_failure_does_not_trigger_regeneration(self) -> None:
        one_step_case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com/",
            steps=[self.make_step(0)],
        )
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(one_step_case, self.events),
            discovery=self.discover,
            plan_generator=self.generator,
            runner=lambda plan: {
                "status": "failed",
                "url": plan.url,
                "steps": [{
                    "action": "assert_visible",
                    "status": "failed",
                    "error": "Selector '#heading' was not found on the page.",
                }],
            },
        )

        result = pipeline.run("Check the example page.")

        self.assertEqual(len(result.test_plans), 1)
        self.assertEqual(len(result.executions), 1)
        self.assertEqual(result.executions[0].status, ExecutionStatus.FAILED)
        self.assertEqual(len(self.generator.calls), 1)
        self.assertEqual(sum(event[0] == "discovery" for event in self.events), 1)

    def test_url_and_title_assertion_failures_do_not_trigger_regeneration(self) -> None:
        one_step_case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com/",
            steps=[self.make_step(0)],
        )
        for action in ("assert_url", "assert_title"):
            with self.subTest(action=action):
                events: list[tuple[Any, ...]] = []
                generator = _FakeGenerator(events)
                discovery_calls: list[str] = []

                def discovery(url: str) -> DiscoveryResult:
                    discovery_calls.append(url)
                    events.append(("discovery", url))
                    return self.discovery_result

                pipeline = QATestPipeline(
                    decomposer=_FakeDecomposer(one_step_case, events),
                    discovery=discovery,
                    plan_generator=generator,
                    runner=lambda plan: {
                        "status": "failed",
                        "url": plan.url,
                        "steps": [{
                            "action": action,
                            "status": "failed",
                            "error": "Expected value did not match.",
                        }],
                    },
                )

                result = pipeline.run("Check the example page.")

                self.assertEqual(result.executions[0].status, ExecutionStatus.FAILED)
                self.assertEqual(len(discovery_calls), 1)
                self.assertEqual(len(generator.calls), 1)
                self.assertEqual(
                    sum(event[0] == "discovery" for event in events),
                    1,
                )

    def test_second_run_uses_cached_version_without_discovery_or_generation(self) -> None:
        store = _CountingPlanStore()
        execution_repository = InMemoryExecutionRepository()
        pipeline = QATestPipeline(
            decomposer=self.decomposer,
            discovery=self.discover,
            plan_generator=self.generator,
            runner=self.run_plan,
            plan_store=store,
            execution_repository=execution_repository,
        )

        first = pipeline.run("Open the example homepage.")
        first_discoveries = sum(event[0] == "discovery" for event in self.events)
        first_generations = len(self.generator.calls)
        first_runners = len(self.runner_calls)
        first_saves = len(store.save_calls)
        second = pipeline.run("Open the example homepage.")

        self.assertEqual(len(first.test_plans), 2)
        self.assertEqual(len(second.test_plans), 2)
        self.assertEqual(
            [pair.test_plan_version.id for pair in second.test_plans],
            [store.find(step.id).id for step in sorted(self.steps, key=lambda item: item.order)],
        )
        self.assertEqual(sum(event[0] == "discovery" for event in self.events), first_discoveries)
        self.assertEqual(len(self.generator.calls), first_generations)
        self.assertEqual(len(self.runner_calls), first_runners + 2)
        self.assertEqual(first_saves, 2)
        self.assertEqual(len(store.save_calls), first_saves)
        self.assertEqual(
            execution_repository.list_for_test_case(self.test_case),
            first.executions + second.executions,
        )
        for step, pair, execution in zip(
            sorted(self.steps, key=lambda item: item.order),
            second.test_plans,
            second.executions,
        ):
            self.assertEqual(pair.test_plan.test_step_id, step.id)
            self.assertEqual(pair.test_plan_version.test_plan_id, pair.test_plan.id)
            self.assertEqual(execution.test_step_id, step.id)
            self.assertEqual(execution.test_plan_version_id, pair.test_plan_version.id)

    def test_plan_cache_trace_records_miss_then_hit_with_version_association(
        self,
    ) -> None:
        # P2-1: characterize ExecutionTrace plan_cache recording. A cache
        # MISS records hit=False with no version and full generation
        # evidence; a cache HIT records hit=True bound to the stored
        # version and performs no new generation.
        step = self.make_step(0)
        case = DomainTestCase(
            name="Flow",
            description="Check",
            base_url="https://example.com/",
            steps=[step],
        )
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        store = InMemoryPlanStore()
        repository = InMemoryExecutionRepository()
        discovery_calls: list[str] = []

        def discover(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            return DiscoveryResult(
                status=DiscoveryStatus.SUCCESS, url="https://example.com/"
            )

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(case, events),
            discovery=discover,
            plan_generator=generator,
            runner=lambda plan: {"status": "passed", "steps": []},
            plan_store=store,
            execution_repository=repository,
        )

        # --- cache MISS: nothing stored, generation happens. ---
        miss = pipeline.run("Check")
        miss_step = miss.trace.steps[0]
        self.assertIsNotNone(miss_step.plan_cache)
        self.assertFalse(miss_step.plan_cache.hit)
        self.assertIsNone(miss_step.plan_cache.version_id)
        self.assertIsNone(miss_step.plan_cache.version_number)

        generated = miss_step.plan_generation
        self.assertIsNotNone(generated)
        self.assertEqual(generated.version_number, 1)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(discovery_calls), 1)
        stored_version = store.find(step.id)
        self.assertIsNotNone(stored_version)
        self.assertEqual(generated.test_plan_version_id, stored_version.id)
        self.assertEqual(
            miss.executions[0].test_plan_version_id, stored_version.id
        )

        # --- cache HIT: stored version reused, no new generation. ---
        hit = pipeline.run("Check")
        hit_step = hit.trace.steps[0]
        self.assertIsNotNone(hit_step.plan_cache)
        self.assertTrue(hit_step.plan_cache.hit)
        self.assertEqual(hit_step.plan_cache.version_number, 1)
        self.assertEqual(hit_step.plan_cache.version_id, stored_version.id)
        # Current behavior: no plan_generation row is recorded on a HIT.
        self.assertIsNone(hit_step.plan_generation)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(discovery_calls), 1)
        self.assertEqual(
            hit.executions[0].test_plan_version_id, stored_version.id
        )
        self.assertEqual(
            hit_step.execution_attempts[0].plan_version_id, stored_version.id
        )

    def test_legacy_stale_v1_is_reported_without_regeneration(self) -> None:
        step = self.make_step(0)
        test_case = DomainTestCase(
            name="Example flow",
            description="Open the example homepage.",
            base_url="https://example.com/",
            steps=[step],
        )
        plan_store = InMemoryPlanStore()
        test_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=test_plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url="https://example.com/",
                steps=[QATestStep(action="click", parameters={"selector": "#old"})],
            ),
        )
        plan_store.save(step.id, version_one, test_plan=test_plan)
        repository = InMemoryExecutionRepository()
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        runner_calls = 0
        discovery_calls: list[str] = []

        def discovery(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            return self.discovery_result

        def runner(plan: QATestPlan) -> dict[str, Any]:
            nonlocal runner_calls
            runner_calls += 1
            if runner_calls == 1:
                return {
                    "status": "failed",
                    "url": plan.url,
                    "steps": [{
                        "action": "click",
                        "status": "failed",
                        "error": "Selector '#old' was not found on the page.",
                    }],
                }
            return {"status": "passed", "url": plan.url, "steps": []}

        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(test_case, events),
            discovery=discovery,
            plan_generator=generator,
            runner=runner,
            plan_store=plan_store,
            execution_repository=repository,
        )

        result = pipeline.run("Open the example homepage.")

        self.assertEqual(len(result.executions), 1)
        self.assertEqual([execution.status for execution in result.executions], [ExecutionStatus.FAILED])
        self.assertEqual([execution.test_step_id for execution in result.executions], [step.id])
        self.assertEqual([execution.test_plan_version_id for execution in result.executions], [version_one.id])
        failed_execution = result.executions[0]
        self.assertEqual(failed_execution.test_step_id, step.id)
        self.assertEqual(failed_execution.planned_step_index, 0)
        self.assertEqual(
            failed_execution.planned_interaction(version_one),
            version_one.qa_test_plan.steps[0],
        )
        self.assertEqual(version_one.version, 1)
        self.assertEqual(
            version_one.qa_test_plan.steps[0].parameters["selector"], "#old"
        )
        self.assertEqual(result.test_plans[0].test_plan_version.id, version_one.id)
        self.assertEqual(repository.list_for_test_step(step.id), result.executions)
        self.assertTrue(all(repository.get(item.id) is item for item in result.executions))
        self.assertEqual(plan_store.find(step.id).id, version_one.id)
        self.assertEqual(plan_store.get_version(version_one.id), version_one)
        self.assertEqual(discovery_calls, ["https://example.com/"])
        self.assertEqual(generator.calls, [])
        self.assertEqual(result.trace.steps[0].locator_recovery.status, RecoveryStatus.NO_MATCH)
        self.assertIsNone(result.trace.steps[0].regeneration)

    def test_real_decomposer_ids_allow_cache_hit_on_repeated_pipeline_run(self) -> None:
        events: list[tuple[Any, ...]] = []
        generator = _FakeGenerator(events)
        store = _CountingPlanStore()
        discovery_calls: list[str] = []
        runner_calls: list[QATestPlan] = []

        def discovery(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            return self.discovery_result

        def runner(plan: QATestPlan) -> dict[str, Any]:
            runner_calls.append(plan)
            return {"status": "passed", "url": plan.url, "steps": []}

        pipeline = QATestPipeline(
            decomposer=TestCaseDecomposer(),
            plan_generator=generator,
            discovery=discovery,
            runner=runner,
            plan_store=store,
        )
        task = "Open the homepage and verify the URL."

        first = pipeline.run(task, base_url="https://example.com/")
        second = pipeline.run(task, base_url="https://example.com/")

        self.assertEqual(first.test_case.id, second.test_case.id)
        self.assertEqual(
            [step.id for step in first.test_case.steps],
            [step.id for step in second.test_case.steps],
        )
        self.assertEqual(len(discovery_calls), len(first.test_case.steps))
        self.assertEqual(len(generator.calls), len(first.test_case.steps))
        self.assertEqual(len(store.save_calls), len(first.test_case.steps))
        self.assertEqual(len(runner_calls), 2 * len(first.test_case.steps))
        self.assertEqual(
            [pair.test_plan_version.id for pair in first.test_plans],
            [pair.test_plan_version.id for pair in second.test_plans],
        )
        for step, pair, execution in zip(
            second.test_case.steps,
            second.test_plans,
            second.executions,
        ):
            self.assertEqual(pair.test_plan.test_step_id, step.id)
            self.assertEqual(pair.test_plan_version.test_plan_id, pair.test_plan.id)
            self.assertEqual(execution.test_step_id, step.id)
            self.assertEqual(execution.test_plan_version_id, pair.test_plan_version.id)

    def test_cached_stale_plan_without_identity_stays_for_manual_review(self) -> None:
        one_step_case = DomainTestCase(
            name="Example flow",
            description="Open the example homepage.",
            base_url="https://example.com/",
            steps=[self.make_step(0)],
        )
        events: list[tuple[Any, ...]] = []
        decomposer = _FakeDecomposer(one_step_case, events)
        generator = _FakeGenerator(events)
        store = _CountingPlanStore()
        runner_versions: list[QATestPlan] = []

        def stale_once(plan: QATestPlan) -> dict[str, Any]:
            runner_versions.append(plan)
            if len(runner_versions) == 2:
                return {
                    "status": "failed",
                    "url": plan.url,
                    "steps": [{
                        "action": "click",
                        "status": "failed",
                        "error": "Selector '#old-menu' was not found on the page.",
                    }],
                }
            return {"status": "passed", "url": plan.url, "steps": []}

        def discovery(url: str) -> DiscoveryResult:
            events.append(("discovery", url))
            return self.discovery_result

        pipeline = QATestPipeline(
            decomposer=decomposer,
            discovery=discovery,
            plan_generator=generator,
            runner=stale_once,
            plan_store=store,
        )

        initial = pipeline.run("Open the example homepage.")
        self.assertEqual([pair.test_plan_version.version for pair in initial.test_plans], [1])
        self.assertEqual([version.version for _, version in store.save_calls], [1])

        drifted = pipeline.run("Open the example homepage.")
        self.assertEqual([pair.test_plan_version.version for pair in drifted.test_plans], [1])
        self.assertEqual([execution.status for execution in drifted.executions], [ExecutionStatus.FAILED])
        self.assertEqual(drifted.test_run.status, ExecutionStatus.FAILED)
        self.assertIsNone(drifted.trace.steps[0].locator_recovery)
        self.assertEqual(drifted.trace.steps[0].regeneration, None)
        self.assertEqual(store.find(one_step_case.steps[0].id).id,
                         initial.test_plans[0].test_plan_version.id)
        self.assertEqual(store.get_version(initial.test_plans[0].test_plan_version.id),
                         initial.test_plans[0].test_plan_version)
        self.assertEqual([version.version for _, version in store.save_calls], [1])
        self.assertEqual([version.version for version in generator.versions], [1])
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(runner_versions), 2)

    def test_cached_assertion_failure_does_not_discover_or_regenerate(self) -> None:
        one_step_case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com/",
            steps=[self.make_step(0)],
        )
        events: list[tuple[Any, ...]] = []
        decomposer = _FakeDecomposer(one_step_case, events)
        generator = _FakeGenerator(events)
        store = InMemoryPlanStore()
        runner = lambda plan: {
            "status": "failed",
            "url": plan.url,
            "steps": [{
                "action": "assert_visible",
                "status": "failed",
                "error": "Expected heading was not visible.",
            }],
        }
        pipeline = QATestPipeline(
            decomposer=decomposer,
            discovery=lambda url: (events.append(("discovery", url)) or self.discovery_result),
            plan_generator=generator,
            runner=runner,
            plan_store=store,
        )

        first = pipeline.run("Check the example page.")
        discovery_count = sum(event[0] == "discovery" for event in events)
        generation_count = len(generator.calls)
        second = pipeline.run("Check the example page.")

        self.assertEqual(first.executions[0].status, ExecutionStatus.FAILED)
        self.assertEqual(second.executions[0].status, ExecutionStatus.FAILED)
        self.assertEqual(sum(event[0] == "discovery" for event in events), discovery_count)
        self.assertEqual(len(generator.calls), generation_count)
        self.assertEqual([version.version for version in generator.versions], [1])

    def test_infrastructure_failure_does_not_trigger_regeneration(self) -> None:
        one_step_case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com/",
            steps=[self.make_step(0)],
        )
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(one_step_case, self.events),
            discovery=self.discover,
            plan_generator=self.generator,
            runner=lambda plan: {
                "status": "failed",
                "url": plan.url,
                "steps": [{
                    "action": "click",
                    "status": "failed",
                    "error": "Browser connection closed unexpectedly.",
                }],
            },
        )

        result = pipeline.run("Check the example page.")

        self.assertEqual(len(result.test_plans), 1)
        self.assertEqual(result.executions[0].status, ExecutionStatus.FAILED)
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)
        self.assertEqual(len(self.generator.calls), 1)
        self.assertEqual(sum(event[0] == "discovery" for event in self.events), 1)


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
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


class _FakeGenerator:
    def __init__(
        self,
        events: list[tuple[Any, ...]],
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.calls: list[tuple[DomainTestStep, DiscoveryResult]] = []
        self.generated: list[GeneratedTestPlan] = []
        self.plans: list[DomainTestPlan] = []
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
        self.calls.append((test_step, discovery_result))
        if self.error:
            raise self.error
        plan = existing_test_plan or DomainTestPlan(test_step_id=test_step.id, name=test_step.name)
        self.plans.append(plan)
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=version_number,
            qa_test_plan=QATestPlan(
                url=discovery_result.url,
                steps=[
                    QATestStep(
                        action="assert_url",
                        parameters={"expected": discovery_result.url},
                    )
                    if "url" in test_step.expected.casefold()
                    else QATestStep(action="assert_page_loaded")
                ],
            ),
        )
        self.versions.append(version)
        generated = GeneratedTestPlan(test_plan=plan, test_plan_version=version)
        self.generated.append(generated)
        return generated


class _CountingPlanStore(InMemoryPlanStore):
    def __init__(self) -> None:
        super().__init__()
        self.save_calls: list[tuple[Any, DomainTestPlanVersion]] = []

    def save(
        self,
        test_step_id: Any,
        plan_version: DomainTestPlanVersion,
        *,
        test_plan: DomainTestPlan,
    ) -> None:
        self.save_calls.append((test_step_id, plan_version))
        super().save(test_step_id, plan_version, test_plan=test_plan)


if __name__ == "__main__":
    unittest.main()
