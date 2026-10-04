import unittest
from typing import Any

from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    ExecutionStatus,
    InteractiveElement,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.pipeline import PipelineResult, PipelineStageError, QATestPipeline
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import GeneratedTestPlan


class QATestPipelineTests(unittest.TestCase):
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
                self.assertEqual(fallback.discover.call_count,
                                 0 if status == DiscoveryStatus.SUCCESS else 1)
                if status != DiscoveryStatus.SUCCESS:
                    self.assertEqual(generator.calls[0][1].navigation_paths[0].menu_item_text, "Account")

    def test_matched_stale_locator_is_repaired_deterministically_and_preserves_history(self) -> None:
        step = self.make_step(0)
        test_case = DomainTestCase(name="Flow", description="Check", base_url="https://example.com/", steps=[step])
        plan_store = InMemoryPlanStore()
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=plan.id, version=1,
            qa_test_plan=QATestPlan(url="https://example.com/", steps=[
                QATestStep(action="click", parameters={"selector": "#old", "text": "Continue"})
            ]),
        )
        plan_store.save(step.id, version_one, test_plan=plan)
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
        self.assertEqual(result.test_plans[0].test_plan_version.qa_test_plan.steps[0].parameters["selector"], "#old")
        self.assertEqual(result.test_plans[1].test_plan_version.qa_test_plan.steps[0].parameters["selector"], "#new")
        self.assertEqual([e.status for e in result.executions], [ExecutionStatus.FAILED, ExecutionStatus.PASSED])
        self.assertEqual([e.test_plan_version_id for e in repository.list_for_test_step(step.id)],
                         [version_one.id, result.test_plans[-1].test_plan_version.id])
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)
        self.assertEqual(generator.calls, [])

    def test_ambiguous_and_not_found_recovery_fall_back_to_llm(self) -> None:
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
                self.assertEqual(len(generator.calls), 1)
                self.assertEqual([p.test_plan_version.version for p in result.test_plans], [1, 2])

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
            description=f"Perform check {order}",
            expected=f"Check {order} passes",
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
        self.assertEqual(result.test_plans, self.generator.generated)
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

    def test_stale_ui_failure_rediscovers_and_generates_version_two_once(self) -> None:
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

        self.assertEqual([entry.test_plan_version.version for entry in result.test_plans], [1, 2])
        self.assertIs(result.test_plans[0].test_plan, result.test_plans[1].test_plan)
        self.assertEqual([run.status for run in result.executions], [
            ExecutionStatus.FAILED, ExecutionStatus.PASSED
        ])
        self.assertEqual(
            [run.test_plan_version_id for run in result.executions],
            [entry.test_plan_version.id for entry in result.test_plans],
        )
        self.assertEqual(
            [event[0] for event in self.events],
            ["decomposer", "discovery", "generator", "discovery", "generator"],
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

    def test_stale_v1_and_v2_attempts_are_both_saved_to_execution_repository(self) -> None:
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

        self.assertEqual(len(result.executions), 2)
        self.assertEqual(
            [execution.status for execution in result.executions],
            [ExecutionStatus.FAILED, ExecutionStatus.PASSED],
        )
        self.assertEqual(
            [execution.test_step_id for execution in result.executions],
            [step.id, step.id],
        )
        self.assertEqual(
            [execution.test_plan_version_id for execution in result.executions],
            [version_one.id, result.test_plans[-1].test_plan_version.id],
        )
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
        self.assertEqual(plan_store.find(step.id).version, 2)
        self.assertEqual(discovery_calls, ["https://example.com/"])
        self.assertEqual(len(generator.calls), 1)
        self.assertIs(generator.calls[0][0], step)
        self.assertIs(generator.plans[0], test_plan)
        self.assertEqual(generator.versions[0].version, 2)

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

    def test_cached_stale_plan_is_replaced_by_version_two_and_reused(self) -> None:
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

        repaired = pipeline.run("Open the example homepage.")
        self.assertEqual([pair.test_plan_version.version for pair in repaired.test_plans], [1, 2])
        self.assertEqual(repaired.executions[0].status, ExecutionStatus.FAILED)
        self.assertEqual(
            [execution.status for execution in repaired.executions],
            [ExecutionStatus.FAILED, ExecutionStatus.PASSED],
        )
        self.assertEqual(
            [execution.test_step_id for execution in repaired.executions],
            [one_step_case.steps[0].id, one_step_case.steps[0].id],
        )
        self.assertEqual(
            [execution.test_plan_version_id for execution in repaired.executions],
            [pair.test_plan_version.id for pair in repaired.test_plans],
        )
        self.assertEqual(repaired.test_run.executions, repaired.executions)
        self.assertEqual(repaired.test_run.status, ExecutionStatus.PASSED)
        self.assertEqual(
            repaired.test_run.final_execution_for_step(one_step_case.steps[0].id).status,
            ExecutionStatus.PASSED,
        )
        self.assertEqual(store.find(one_step_case.steps[0].id), repaired.test_plans[-1].test_plan_version)
        self.assertEqual(
            store.get_version(repaired.executions[0].test_plan_version_id),
            repaired.test_plans[0].test_plan_version,
        )
        self.assertEqual(
            store.get_version(repaired.executions[1].test_plan_version_id),
            repaired.test_plans[1].test_plan_version,
        )
        self.assertEqual([version.version for _, version in store.save_calls], [1, 2])

        calls_before_cache_hit = (len(generator.calls), sum(e[0] == "discovery" for e in events))
        reused = pipeline.run("Open the example homepage.")

        self.assertEqual([pair.test_plan_version.version for pair in reused.test_plans], [2])
        self.assertEqual(reused.test_plans[0].test_plan_version.id, store.find(one_step_case.steps[0].id).id)
        self.assertEqual(len(generator.calls), calls_before_cache_hit[0])
        self.assertEqual(sum(e[0] == "discovery" for e in events), calls_before_cache_hit[1])
        self.assertEqual(len(store.save_calls), 2)
        self.assertEqual([version.version for version in generator.versions], [1, 2])
        self.assertEqual(reused.executions[0].test_step_id, one_step_case.steps[0].id)
        self.assertEqual(
            reused.executions[0].test_plan_version_id,
            reused.test_plans[0].test_plan_version.id,
        )

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
                steps=[QATestStep(action="assert_page_loaded")],
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
