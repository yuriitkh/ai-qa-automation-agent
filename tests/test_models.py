import unittest

from pydantic import ValidationError
from uuid import uuid4

from qa_agent.models import (
    Evidence,
    EvidenceType,
    Execution,
    ExecutionStatus,
    DiscoveryResult,
    DiscoveryStatus,
    ExecutionSegment,
    InteractiveElement,
    NavigationAction,
    NavigationSequence,
    Precondition as DomainPrecondition,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)


class QATestStepTests(unittest.TestCase):
    def test_all_supported_actions_are_accepted(self) -> None:
        actions = (
            "navigate",
            "assert_page_loaded",
            "assert_title",
            "assert_visible",
            "click", "check", "uncheck",
            "fill",
            "assert_hidden",
            "assert_url",
            "select_option", "assert_text_contains", "assert_checked", "assert_unchecked",
            "assert_selected", "assert_enabled", "assert_disabled",
        )

        parameters_by_action = {
            "check": {"selector": "#choice"},
            "uncheck": {"selector": "#choice"},
            "select_option": {"selector": "select", "option_label": "Two"},
            "assert_text_contains": {"expected_text": "required phrase"},
            "assert_checked": {"selector": "#choice"},
            "assert_unchecked": {"selector": "#choice"},
            "assert_selected": {"selector": "select", "expected": "Two"},
            "assert_enabled": {"selector": "#control"},
            "assert_disabled": {"selector": "#control"},
        }
        for action in actions:
            with self.subTest(action=action):
                step = QATestStep(
                    action=action,
                    parameters=parameters_by_action.get(action, {}),
                )
                self.assertEqual(step.action, action)

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            QATestStep(action="delete")

    def test_select_option_requires_a_selector_and_visible_option_label(self) -> None:
        with self.assertRaises(ValidationError):
            QATestStep(action="select_option", parameters={"selector": "select"})
        with self.assertRaises(ValidationError):
            QATestStep(action="select_option", parameters={"option_label": "Two"})

    def test_assert_selected_accepts_radio_selector_without_select_expected(self) -> None:
        step = QATestStep(
            action="assert_selected", parameters={"selector": "#default-radio"}
        )
        self.assertEqual(step.parameters, {"selector": "#default-radio"})

    def test_checkbox_state_actions_require_a_selector(self) -> None:
        for action in ("check", "uncheck", "assert_unchecked"):
            with self.subTest(action=action), self.assertRaises(ValidationError):
                QATestStep(action=action)

    def test_option_label_is_rejected_for_non_select_actions(self) -> None:
        for action, parameters in (
            ("navigate", {"url": "https://example.test"}),
            ("assert_page_loaded", {}),
            ("assert_disabled", {"selector": "#control"}),
        ):
            with self.subTest(action=action):
                QATestStep(action=action, parameters=parameters)
                with self.assertRaises(ValidationError):
                    QATestStep(
                        action=action,
                        parameters={**parameters, "option_label": None},
                    )


class DomainModelTests(unittest.TestCase):
    def test_generic_discovery_elements_navigation_and_legacy_defaults(self) -> None:
        link = InteractiveElement(
            kind="link", selector="#loans", accessible_name="Loans",
            tag="a", href="/loans",
        )
        sequence = NavigationSequence(
            actions=[NavigationAction(element=link)], destination_url="https://example.test/loans"
        )
        result = DiscoveryResult(
            status=DiscoveryStatus.PARTIAL,
            url="https://example.test/",
            interactive_elements=[link],
            navigation=[sequence],
        )
        self.assertEqual(result.interactive_elements[0].accessible_name, "Loans")
        self.assertEqual(result.navigation[0].actions[0].element.selector, "#loans")
        self.assertEqual(result.navigation[0].destination_url, "https://example.test/loans")
        legacy = DiscoveryResult(status=DiscoveryStatus.PARTIAL, url="https://example.test/")
        self.assertEqual(legacy.interactive_elements, [])
        self.assertEqual(legacy.navigation, [])

    def make_test_step(self, order: int = 0) -> DomainTestStep:
        return DomainTestStep(
            name=f"Step {order + 1}",
            description="Open the page and inspect the result.",
            expected="The page is displayed.",
            order=order,
        )

    def make_qa_test_plan(self) -> QATestPlan:
        return QATestPlan(
            url="https://example.com",
            steps=[QATestStep(action="navigate", parameters={"url": "https://example.com"})],
        )

    def test_test_case_holds_multiple_ordered_steps(self) -> None:
        case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com",
            steps=[self.make_test_step(0), self.make_test_step(1)],
        )

        self.assertEqual([step.order for step in case.steps], [0, 1])
        self.assertEqual(case.preconditions, [])

    def test_test_case_preconditions_are_ordered_and_round_trip(self) -> None:
        conditions = [
            DomainPrecondition(
                description="Registration email is unused",
                order=0,
                provided_data_keys=["registration_email"],
            ),
            DomainPrecondition(
                description="Account has sufficient balance",
                order=1,
            ),
        ]
        case = DomainTestCase(
            name="Register user",
            description="Create an account.",
            steps=[self.make_test_step()],
            preconditions=conditions,
        )

        restored = DomainTestCase.model_validate(case.model_dump())

        self.assertEqual(restored.id, case.id)
        self.assertEqual(
            [item.id for item in restored.preconditions],
            [item.id for item in conditions],
        )
        self.assertEqual(
            [item.order for item in restored.preconditions], [0, 1]
        )
        self.assertEqual(
            restored.preconditions[0].provided_data_keys,
            ["registration_email"],
        )

    def test_test_case_rejects_unordered_preconditions(self) -> None:
        with self.assertRaises(ValidationError):
            DomainTestCase(
                name="Invalid preconditions",
                description="Conditions are out of order.",
                steps=[self.make_test_step()],
                preconditions=[
                    DomainPrecondition(description="Second", order=1),
                    DomainPrecondition(description="First", order=0),
                ],
            )

    def test_legacy_test_case_exposes_one_implicit_segment(self) -> None:
        steps = [self.make_test_step(0), self.make_test_step(1)]
        case = DomainTestCase(
            name="Example flow",
            description="Check the example page.",
            base_url="https://example.com",
            steps=steps,
        )

        self.assertEqual(len(case.segments), 1)
        self.assertEqual(case.segments[0].order, 0)
        self.assertEqual(case.segments[0].base_url, case.base_url)
        self.assertEqual(case.steps, steps)
        self.assertIs(case.segments[0].steps[0], steps[0])
        self.assertIs(case.segments[0].steps[1], steps[1])

    def test_unsorted_legacy_steps_round_trip_without_reordering(self) -> None:
        steps = [self.make_test_step(1), self.make_test_step(0)]
        case = DomainTestCase(
            name="Legacy flow",
            description="Keep supplied step order.",
            base_url="https://example.com",
            steps=steps,
        )

        restored = DomainTestCase(**case.model_dump())

        self.assertEqual(restored.id, case.id)
        self.assertEqual(restored.segments[0].id, case.segments[0].id)
        self.assertEqual([step.id for step in restored.steps], [step.id for step in steps])
        self.assertEqual([step.order for step in restored.steps], [1, 0])
        self.assertEqual(restored.segments[0].base_url, case.base_url)
        self.assertTrue(restored.segments[0].is_implicit)

    def test_ordered_legacy_steps_round_trip(self) -> None:
        steps = [self.make_test_step(0), self.make_test_step(1)]
        case = DomainTestCase(
            name="Legacy flow",
            description="Round-trip an ordered legacy case.",
            steps=steps,
        )

        restored = DomainTestCase(**case.model_dump())

        self.assertEqual(restored.id, case.id)
        self.assertEqual(restored.segments[0].id, case.segments[0].id)
        self.assertEqual([step.id for step in restored.steps], [step.id for step in steps])

    def test_explicit_segments_round_trip(self) -> None:
        steps = [self.make_test_step(index) for index in range(3)]
        case = DomainTestCase(
            name="Segmented flow",
            description="Round-trip explicit groups.",
            segments=[
                ExecutionSegment(order=0, steps=steps[:2]),
                ExecutionSegment(order=1, steps=steps[2:]),
            ],
        )

        restored = DomainTestCase(**case.model_dump())

        self.assertEqual([segment.id for segment in restored.segments], [segment.id for segment in case.segments])
        self.assertEqual(
            [[step.id for step in segment.steps] for segment in restored.segments],
            [[step.id for step in segment.steps] for segment in case.segments],
        )

    def test_dual_steps_and_segments_reject_conflicting_step_content(self) -> None:
        step = self.make_test_step(0)
        case = DomainTestCase(
            name="Segmented flow",
            description="Canonical content comes from segments.",
            segments=[ExecutionSegment(order=0, steps=[step])],
        )
        dumped = case.model_dump()
        dumped["steps"][0]["expected"] = "Conflicting expectation"

        with self.assertRaises(ValidationError):
            DomainTestCase(**dumped)

    def test_explicit_segments_preserve_order_and_step_identity(self) -> None:
        steps = [self.make_test_step(index) for index in range(3)]
        case = DomainTestCase(
            name="Segmented flow",
            description="Two explicit groups.",
            segments=[
                ExecutionSegment(order=0, base_url="https://one.example", steps=steps[:2]),
                ExecutionSegment(order=1, base_url="https://two.example", steps=steps[2:]),
            ],
        )

        self.assertEqual([[step.id for step in segment.steps] for segment in case.segments], [
            [steps[0].id, steps[1].id], [steps[2].id],
        ])
        self.assertEqual([step.id for step in case.steps], [step.id for step in steps])
        self.assertIs(case.steps[2], steps[2])

    def test_explicit_segments_reject_duplicate_step_membership(self) -> None:
        step = self.make_test_step(0)
        with self.assertRaises(ValidationError):
            DomainTestCase(
                name="Invalid flow",
                description="A step appears in two groups.",
                segments=[
                    ExecutionSegment(order=0, steps=[step]),
                    ExecutionSegment(order=1, steps=[step]),
                ],
            )

    def test_explicit_segments_reject_unordered_groups_and_steps(self) -> None:
        first, second = self.make_test_step(0), self.make_test_step(1)
        with self.assertRaises(ValidationError):
            DomainTestCase(
                name="Invalid flow",
                description="Groups are out of order.",
                segments=[
                    ExecutionSegment(order=1, steps=[first]),
                    ExecutionSegment(order=0, steps=[second]),
                ],
            )

        with self.assertRaises(ValidationError):
            DomainTestCase(
                name="Invalid flow",
                description="Steps are out of order.",
                segments=[
                    ExecutionSegment(order=0, steps=[second, first]),
                ],
            )

    def test_test_step_has_description_and_expected(self) -> None:
        step = self.make_test_step()

        self.assertEqual(step.description, "Open the page and inspect the result.")
        self.assertEqual(step.expected, "The page is displayed.")

    def test_test_step_does_not_have_actual_result(self) -> None:
        step = self.make_test_step()

        self.assertNotIn("actual_result", type(step).model_fields)

    def test_test_plan_references_test_step(self) -> None:
        step = self.make_test_step()
        plan = DomainTestPlan(test_step_id=step.id, name="Browser plan")

        self.assertEqual(plan.test_step_id, step.id)

    def test_multiple_versions_can_belong_to_one_test_plan(self) -> None:
        plan = DomainTestPlan(test_step_id=self.make_test_step().id, name="Browser plan")
        versions = [
            DomainTestPlanVersion(test_plan_id=plan.id, version=1, qa_test_plan=self.make_qa_test_plan()),
            DomainTestPlanVersion(test_plan_id=plan.id, version=2, qa_test_plan=self.make_qa_test_plan()),
        ]

        self.assertEqual({version.test_plan_id for version in versions}, {plan.id})
        self.assertEqual([version.version for version in versions], [1, 2])

    def test_test_plan_version_contains_executable_qa_test_plan(self) -> None:
        executable_plan = self.make_qa_test_plan()
        version = DomainTestPlanVersion(
            test_plan_id=DomainTestPlan(test_step_id=self.make_test_step().id, name="Plan").id,
            version=1,
            qa_test_plan=executable_plan,
        )

        self.assertIs(version.qa_test_plan, executable_plan)

    def test_test_plan_version_rejects_field_mutation(self) -> None:
        version = DomainTestPlanVersion(
            test_plan_id=DomainTestPlan(test_step_id=self.make_test_step().id, name="Plan").id,
            version=1,
            qa_test_plan=self.make_qa_test_plan(),
        )

        with self.assertRaises(ValidationError):
            version.version = 2

    def test_multiple_executions_can_reference_one_version(self) -> None:
        test_step = self.make_test_step()
        version = DomainTestPlanVersion(
            test_plan_id=DomainTestPlan(test_step_id=test_step.id, name="Plan").id,
            version=1,
            qa_test_plan=self.make_qa_test_plan(),
        )
        executions = [
            Execution(test_step_id=test_step.id, test_plan_version_id=version.id, actual_result="Pass 1"),
            Execution(test_step_id=test_step.id, test_plan_version_id=version.id, actual_result="Pass 2"),
        ]

        self.assertEqual([run.test_plan_version_id for run in executions], [version.id, version.id])
        self.assertEqual([run.test_step_id for run in executions], [test_step.id, test_step.id])

    def test_each_execution_stores_its_own_actual_result(self) -> None:
        test_step = self.make_test_step()
        version_id = DomainTestPlanVersion(
            test_plan_id=DomainTestPlan(test_step_id=test_step.id, name="Plan").id,
            version=1,
            qa_test_plan=self.make_qa_test_plan(),
        ).id
        first = Execution(test_step_id=test_step.id, test_plan_version_id=version_id, actual_result="Passed")
        second = Execution(test_step_id=test_step.id, test_plan_version_id=version_id, actual_result="Failed")

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.actual_result, "Passed")
        self.assertEqual(second.actual_result, "Failed")

    def test_evidence_is_immutable_and_can_be_created(self) -> None:
        execution_id = uuid4()
        evidence = Evidence(
            execution_id=execution_id,
            type=EvidenceType.SCREENSHOT,
            path="artifacts/failure.png",
            description="Page after failed assertion",
        )

        self.assertEqual(evidence.execution_id, execution_id)
        self.assertEqual(evidence.type, EvidenceType.SCREENSHOT)
        self.assertEqual(evidence.path, "artifacts/failure.png")
        with self.assertRaises(ValidationError):
            evidence.path = "changed.png"

    def test_execution_supports_no_evidence_or_ordered_evidence(self) -> None:
        step = self.make_test_step()
        version_id = self.make_test_plan_version_id()
        without_evidence = Execution(
            test_step_id=step.id,
            test_plan_version_id=version_id,
            actual_result="Passed",
        )
        self.assertEqual(without_evidence.evidence, ())

        execution_id = uuid4()
        first = Evidence(
            execution_id=execution_id,
            type=EvidenceType.SCREENSHOT,
            path="artifacts/page.png",
        )
        second = Evidence(
            execution_id=execution_id,
            type=EvidenceType.DOM_SNAPSHOT,
            path="artifacts/page.html",
        )
        with_one = Execution(
            id=execution_id,
            test_step_id=step.id,
            test_plan_version_id=version_id,
            actual_result="Passed",
            evidence=[first],
        )
        with_multiple = Execution(
            id=execution_id,
            test_step_id=step.id,
            test_plan_version_id=version_id,
            actual_result="Passed",
            evidence=[first, second],
        )

        self.assertEqual(with_one.evidence, (first,))
        self.assertEqual(with_multiple.evidence, (first, second))
        self.assertEqual([item.id for item in with_multiple.evidence], [first.id, second.id])

    def test_execution_rejects_evidence_linked_to_another_execution(self) -> None:
        evidence = Evidence(
            execution_id=uuid4(),
            type=EvidenceType.PAGE_SOURCE,
            path="artifacts/source.html",
        )
        with self.assertRaises(ValidationError):
            Execution(
                test_step_id=self.make_test_step().id,
                test_plan_version_id=self.make_test_plan_version_id(),
                actual_result="Passed",
                evidence=[evidence],
            )

    def test_execution_status_validation(self) -> None:
        test_step = self.make_test_step()
        execution = Execution(
            test_step_id=test_step.id,
            test_plan_version_id=self.make_test_plan_version_id(),
            actual_result="Pending",
        )
        self.assertEqual(execution.status, ExecutionStatus.PENDING)

        with self.assertRaises(ValidationError):
            Execution(
                test_step_id=test_step.id,
                test_plan_version_id=self.make_test_plan_version_id(),
                status="SKIPPED",
                actual_result="Skipped",
            )

    def make_test_plan_version_id(self):
        return DomainTestPlanVersion(
            test_plan_id=DomainTestPlan(test_step_id=self.make_test_step().id, name="Plan").id,
            version=1,
            qa_test_plan=self.make_qa_test_plan(),
        ).id


if __name__ == "__main__":
    unittest.main()
