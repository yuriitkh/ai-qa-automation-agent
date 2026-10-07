import inspect
import json
import unittest
from typing import get_type_hints
from unittest.mock import patch

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.models import (
    DiscoveryResult,
    PlanVersionOrigin,
    DiscoveryStatus,
    InteractiveElement,
    QATestPlan,
    QATestStep,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.test_plan_generator import LLMTestPlanGenerator, TestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.execution_progress import (
    ExecutionEventType,
    ExecutionProgressReporter,
    ExecutionProgressStore,
    active_execution_progress,
)


class TestPlanGeneratorTests(unittest.TestCase):
    def make_test_step(self) -> DomainTestStep:
        return DomainTestStep(
            name="Check homepage",
            description="Open the homepage",
            expected="The homepage is loaded",
            order=0,
        )

    def test_generator_exposes_expected_interface(self) -> None:
        method = TestPlanGenerator.generate
        signature = inspect.signature(method)
        hints = get_type_hints(method)

        self.assertEqual(list(signature.parameters), ["self", "test_step", "discovery_result"])
        self.assertIs(hints["test_step"], DomainTestStep)
        self.assertIs(hints["discovery_result"], DiscoveryResult)
        self.assertIs(hints["return"], DomainTestPlanVersion)

    def test_generate_accepts_a_test_step_and_discovery_result(self) -> None:
        step = self.make_test_step()
        result = DiscoveryResult(
            status=DiscoveryStatus.PARTIAL,
            url="https://example.com",
            snapshot={"url": "https://example.com", "links": []},
        )

        with self.assertRaisesRegex(NotImplementedError, "no executable plan was created"):
            TestPlanGenerator().generate(step, result)

    def test_unimplemented_generation_fails_predictably(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "Test plan generation is not implemented",
        ):
            TestPlanGenerator().generate(
                self.make_test_step(),
                DiscoveryResult(status=DiscoveryStatus.PARTIAL, url="https://example.com"),
            )

    def test_existing_domain_models_remain_compatible(self) -> None:
        test_step = self.make_test_step()
        test_plan = DomainTestPlan(test_step_id=test_step.id, name="Homepage plan")
        executable = QATestPlan(
            url="https://example.com",
            steps=[QATestStep(action="navigate", parameters={"url": "https://example.com"})],
        )
        version = DomainTestPlanVersion(
            test_plan_id=test_plan.id,
            version=1,
            qa_test_plan=executable,
        )

        self.assertEqual(test_plan.test_step_id, test_step.id)
        self.assertIs(version.qa_test_plan, executable)
        self.assertNotIn("actual_result", type(test_step).model_fields)


class LLMTestPlanGeneratorTests(unittest.TestCase):
    def make_test_step(self) -> DomainTestStep:
        return DomainTestStep(
            name="Verify Boliglån page",
            description="Open Lån → Boliglån",
            expected="Boliglån page is displayed",
            order=2,
        )

    def make_discovery_result(self) -> DiscoveryResult:
        return DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://www.dnb.no/",
            title="DNB",
            snapshot={"headings": [{"selector": "h1", "text": "Boliglån"}]},
            navigation_paths=[
                {
                    "menu_button_selector": "#menu-button-open",
                    "menu_tab_text": "Privat",
                    "menu_tab_selector": "#header-navigation-menu-tab-1",
                    "menu_item_text": "Lån",
                    "menu_item_selector": '#main-menu a[role="menuitem"][href="/lan"]',
                    "submenu_text": "Boliglån",
                    "submenu_selector": 'main li a[href="/lan/boliglan"]',
                    "expected_url": "https://www.dnb.no/lan/boliglan",
                    "heading_text": "Boliglån",
                    "heading_selector": 'h1:has-text("Boliglån")',
                }
            ],
            direct_navigation_paths=[
                {
                    "strategy": "direct_nav",
                    "root_url": "https://www.finn.no/",
                    "steps": [
                        {"text": "Reise", "selector": 'a[href="/reise"]',
                         "href": "/reise", "resolved_url": "https://www.finn.no/reise/"},
                        {"text": "Restplasser", "selector": 'nav a[href="/reise/restplasser/"]',
                         "href": "/reise/restplasser/",
                         "resolved_url": "https://www.finn.no/reise/restplasser/"},
                    ],
                    "expected_url": "https://www.finn.no/reise/restplasser/",
                    "heading_text": "Restplasser",
                    "heading_selector": 'h1:has-text("Restplasser")',
                }
            ],
            warnings=["Partial page data"],
            strategies_used=["menu_navigation", "direct_navigation"],
        )

    def test_successful_generation_creates_linked_plan_version(self) -> None:
        step = self.make_test_step()
        executable = QATestPlan(
            url="https://www.dnb.no/",
            steps=[
                QATestStep(action="navigate", parameters={"url": "https://www.dnb.no/"}),
                QATestStep(action="click", parameters={"selector": 'main li a[href="/lan/boliglan"]'}),
                QATestStep(action="assert_url", parameters={"expected": "https://www.dnb.no/lan/boliglan"}),
                QATestStep(action="assert_visible", parameters={
                    "selector": 'h1:has-text("Boliglån")', "expected_text": "Boliglån"
                }),
            ],
        )
        router = _StubRouter(executable)
        created_plans: list[DomainTestPlan] = []

        def create_test_plan_model(**values: object) -> DomainTestPlan:
            plan = DomainTestPlan(**values)
            created_plans.append(plan)
            return plan

        with patch(
            "qa_agent.test_plan_generator.TestPlan",
            side_effect=create_test_plan_model,
        ) as plan_factory:
            version = LLMTestPlanGenerator(router).generate(step, self.make_discovery_result())

        self.assertIsInstance(version, DomainTestPlanVersion)
        self.assertIsNotNone(version.test_plan_id)
        self.assertEqual(version.test_plan_id, created_plans[0].id)
        self.assertEqual(plan_factory.call_args.kwargs["test_step_id"], step.id)
        self.assertEqual(version.qa_test_plan, executable)
        self.assertEqual(len(version.qa_test_plan.steps), 4)
        self.assertEqual(version.origin, PlanVersionOrigin.AI_GENERATED)

    def test_generate_with_plan_returns_the_created_plan_and_its_version(self) -> None:
        step = self.make_test_step()
        executable = QATestPlan(
            url="https://www.dnb.no/",
            steps=[QATestStep(action="assert_page_loaded")],
        )

        generated = LLMTestPlanGenerator(_StubRouter(executable)).generate_with_plan(
            step,
            self.make_discovery_result(),
        )

        self.assertEqual(generated.test_plan.test_step_id, step.id)
        self.assertEqual(generated.test_plan_version.test_plan_id, generated.test_plan.id)
        self.assertIs(generated.test_plan_version.qa_test_plan, executable)

    def test_regeneration_reuses_plan_and_creates_requested_next_version(self) -> None:
        step = self.make_test_step()
        generator = LLMTestPlanGenerator(_StubRouter(QATestPlan(
            url="https://www.dnb.no/",
            steps=[QATestStep(action="assert_page_loaded")],
        )))
        first = generator.generate_with_plan(step, self.make_discovery_result())
        second = generator.generate_with_plan(
            step,
            self.make_discovery_result(),
            existing_test_plan=first.test_plan,
            version_number=2,
        )

        self.assertIs(second.test_plan, first.test_plan)
        self.assertEqual(second.test_plan_version.test_plan_id, first.test_plan.id)
        self.assertEqual(second.test_plan_version.version, 2)
        self.assertEqual(second.test_plan_version.origin, PlanVersionOrigin.REGENERATED)

    def test_router_failure_propagates_without_masking(self) -> None:
        error = RuntimeError("router failed")
        router = _StubRouter(error=error)

        with self.assertRaisesRegex(RuntimeError, "router failed") as raised:
            LLMTestPlanGenerator(router).generate(self.make_test_step(), self.make_discovery_result())

        self.assertIs(raised.exception, error)

    def test_invalid_router_plan_gets_one_repair_and_never_creates_a_version(self) -> None:
        router = _StubRouter({"url": "", "steps": []})

        with patch("qa_agent.test_plan_generator.TestPlan") as plan_factory:
            with self.assertRaises(PlanValidationError) as raised:
                LLMTestPlanGenerator(router).generate(self.make_test_step(), self.make_discovery_result())

        plan_factory.assert_not_called()
        self.assertEqual(len(router.calls), 2)
        self.assertEqual(raised.exception.issues[0].code, "MISSING_TARGET_URL")

    def test_invalid_plan_is_repaired_once_for_multi_field_registration_step(self) -> None:
        target_url = "http://127.0.0.1:43123/register"
        step = DomainTestStep(
            name="Enter registration details",
            description="Fill first name, last name, email, and password, then create the account.",
            expected="The account confirmation is shown.",
            order=1,
        )
        discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url=target_url,
            interactive_elements=[
                InteractiveElement(kind="input", tag="input", selector="#first-name", accessible_name="First name"),
                InteractiveElement(kind="input", tag="input", selector="#last-name", accessible_name="Last name"),
                InteractiveElement(kind="input", tag="input", selector="#email", accessible_name="Email"),
                InteractiveElement(kind="input", tag="input", selector="#password", accessible_name="Password"),
                InteractiveElement(kind="button", tag="button", role="button", selector="#create-account", accessible_name="Create account"),
            ],
        )
        invalid = QATestPlan(url=target_url, steps=[
            QATestStep(action="fill", parameters={"selector": "#first-name"}),
        ])
        repaired = QATestPlan(url=target_url, steps=[
            QATestStep(action="fill", parameters={"selector": "#first-name", "value": "Ada"}),
            QATestStep(action="fill", parameters={"selector": "#last-name", "value": "Lovelace"}),
            QATestStep(action="fill", parameters={"selector": "#email", "value": "ada@example.test"}),
            QATestStep(action="fill", parameters={"selector": "#password", "value": "local-test-password"}),
            QATestStep(action="click", parameters={"selector": "#create-account"}),
            QATestStep(action="assert_text_contains", parameters={"expected_text": "Account created"}),
        ])
        router = _SequencedRouter([invalid, repaired])
        generated = LLMTestPlanGenerator(router).generate_with_plan(step, discovery)

        self.assertEqual(len(router.calls), 2)
        self.assertEqual(len(generated.test_plan_version.qa_test_plan.steps), 6)
        self.assertEqual(generated.test_plan_version.origin, PlanVersionOrigin.REPAIRED)
        self.assertIn("MISSING_INPUT_VALUE", router.calls[1]["task"])
        self.assertIn("Enter registration details", router.calls[1]["task"])

    def test_repair_lifecycle_is_reported_without_exposing_candidate_output(self) -> None:
        step = self.make_test_step()
        invalid = QATestPlan(url="https://www.dnb.no/", steps=[
            QATestStep(action="click", parameters={}),
        ])
        valid = QATestPlan(url="https://www.dnb.no/", steps=[
            QATestStep(action="assert_page_loaded"),
        ])
        router = _SequencedRouter([invalid, valid])
        store = ExecutionProgressStore()
        progress_id = store.create(step.id, "AUTOMATION")
        reporter = ExecutionProgressReporter(store, progress_id)

        with active_execution_progress(reporter):
            LLMTestPlanGenerator(router).generate(step, self.make_discovery_result())

        events = store.get(progress_id).events
        repair_events = [event for event in events if event.event_type in {
            ExecutionEventType.PLAN_REPAIR_STARTED,
            ExecutionEventType.PLAN_REPAIR_SUCCEEDED,
            ExecutionEventType.PLAN_REPAIR_FAILED,
        }]
        self.assertEqual(
            [event.event_type for event in repair_events],
            [ExecutionEventType.PLAN_REPAIR_STARTED, ExecutionEventType.PLAN_REPAIR_SUCCEEDED],
        )
        self.assertNotIn("model output", json.dumps([event.to_public_dict() for event in events]))

    def test_repair_uses_router_fallback_without_changing_provider_order(self) -> None:
        class FirstProvider(LLMProvider):
            def __init__(self):
                self.calls = 0

            @property
            def is_available(self):
                return True

            def create_test_plan(self, task, target_url, page_snapshot):
                self.calls += 1
                if self.calls == 1:
                    return QATestPlan(url=target_url, steps=[{"action": "click", "parameters": {}}])
                raise RetryableLLMError("fake repair provider outage")

        class FallbackProvider(LLMProvider):
            def __init__(self):
                self.calls = 0

            @property
            def is_available(self):
                return True

            def create_test_plan(self, task, target_url, page_snapshot):
                self.calls += 1
                return QATestPlan(url=target_url, steps=[{"action": "assert_page_loaded", "parameters": {}}])

        first = FirstProvider()
        fallback = FallbackProvider()
        router = LLMRouter([first, fallback])

        generated = LLMTestPlanGenerator(router).generate_with_plan(
            self.make_test_step(), self.make_discovery_result()
        )

        self.assertEqual(generated.test_plan_version.origin, PlanVersionOrigin.REPAIRED)
        self.assertEqual(first.calls, 2)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(router.selected_provider_name, "FallbackProvider")

    def test_discovery_context_and_single_step_task_are_passed_to_router(self) -> None:
        router = _StubRouter(QATestPlan(
            url="https://www.dnb.no/",
            steps=[QATestStep(action="assert_page_loaded")],
        ))
        step = self.make_test_step()
        result = self.make_discovery_result()

        LLMTestPlanGenerator(router).generate(step, result)

        self.assertEqual(len(router.calls), 1)
        call = router.calls[0]
        self.assertEqual(call["target_url"], result.url)
        self.assertIn("one human TestStep", call["task"])
        self.assertIn("multiple ordered executable actions", call["task"])
        self.assertIn(step.name, call["task"])
        self.assertIn(step.description, call["task"])
        self.assertIn(step.expected, call["task"])
        self.assertIn(f"TestStep order: {step.order}", call["task"])

        context = json.loads(call["page_snapshot"])
        self.assertEqual(context["url"], result.url)
        self.assertEqual(context["navigation_paths"][0]["expected_url"],
                         "https://www.dnb.no/lan/boliglan")
        self.assertEqual(context["direct_navigation_paths"][0]["strategy"], "direct_nav")
        self.assertEqual(context["headings"], result.snapshot["headings"])
        self.assertEqual(context["warnings"], result.warnings)

    def test_only_the_supplied_test_step_is_used_for_one_router_request(self) -> None:
        router = _StubRouter(QATestPlan(
            url="https://www.dnb.no/",
            steps=[QATestStep(action="assert_page_loaded")],
        ))
        step = self.make_test_step()

        LLMTestPlanGenerator(router).generate(step, self.make_discovery_result())

        self.assertEqual(len(router.calls), 1)
        self.assertEqual(router.calls[0]["task"].count("TestStep order:"), 1)
        self.assertIn("Do not generate a TestCase", router.calls[0]["task"])
        self.assertIn("multiple ordered executable actions", router.calls[0]["task"])

    def test_enabled_state_assertions_require_matching_discovered_input_selector(self) -> None:
        selector = 'input[name="my-disabled"]'
        discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://www.selenium.dev/selenium/web/web-form.html",
            interactive_elements=[
                InteractiveElement(
                    kind="input", tag="input", selector=selector,
                    accessible_name="Disabled input", name="my-disabled",
                    enabled=False,
                ),
                InteractiveElement(
                    kind="input", tag="input", selector='input[name="my-text"]',
                    accessible_name="Text input", name="my-text", enabled=True,
                ),
            ],
        )

        for action in ("assert_disabled", "assert_enabled"):
            with self.subTest(action=action):
                step = DomainTestStep(
                    name="Check Disabled input",
                    description="Verify the input labeled Disabled input",
                    expected="Disabled input is disabled",
                    order=0,
                )
                plan = QATestPlan(
                    url=discovery.url,
                    steps=[QATestStep(action=action, parameters={"selector": "input"})],
                )

                with self.assertRaisesRegex(PlanValidationError, "deterministic Discovery selector"):
                    LLMTestPlanGenerator(_StubRouter(plan)).generate(step, discovery)

                plan.steps[0].parameters["selector"] = "#my-disabled"
                with self.assertRaisesRegex(PlanValidationError, "deterministic Discovery selector"):
                    LLMTestPlanGenerator(_StubRouter(plan)).generate(step, discovery)

                plan.steps[0].parameters["selector"] = selector
                LLMTestPlanGenerator(_StubRouter(plan)).generate(step, discovery)

    def test_radio_assert_selected_requires_concrete_discovered_selector(self) -> None:
        selector = "#my-radio-2"
        discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://www.selenium.dev/selenium/web/web-form.html",
            interactive_elements=[
                InteractiveElement(
                    kind="radio", tag="input", selector=selector,
                    accessible_name="Default radio", id="my-radio-2",
                ),
                InteractiveElement(
                    kind="radio", tag="input", selector="#my-radio-1",
                    accessible_name="Checked radio", id="my-radio-1",
                ),
            ],
        )
        test_step = DomainTestStep(
            name="Select Default radio",
            description="Click the radio button labeled Default radio",
            expected="Default radio is selected",
            order=0,
        )
        plan = QATestPlan(
            url=discovery.url,
            steps=[
                QATestStep(action="click", parameters={"selector": selector}),
                QATestStep(action="assert_selected", parameters={"selector": selector}),
            ],
        )

        LLMTestPlanGenerator(_StubRouter(plan)).generate(test_step, discovery)

        plan.steps[1].parameters["selector"] = "input[type=radio]"
        with self.assertRaisesRegex(PlanValidationError, "deterministic Discovery selector"):
            LLMTestPlanGenerator(_StubRouter(plan)).generate(test_step, discovery)


class _StubRouter:
    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, str]] = []

    def create_test_plan(self, *, task: str, target_url: str, page_snapshot: str) -> object:
        self.calls.append({
            "task": task,
            "target_url": target_url,
            "page_snapshot": page_snapshot,
        })
        if self.error:
            raise self.error
        return self.result


class _SequencedRouter:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, str]] = []

    def create_test_plan(self, *, task: str, target_url: str, page_snapshot: str) -> object:
        self.calls.append({"task": task, "target_url": target_url, "page_snapshot": page_snapshot})
        return self.results.pop(0)


if __name__ == "__main__":
    unittest.main()
