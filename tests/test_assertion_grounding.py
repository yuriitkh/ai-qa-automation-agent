import unittest

from qa_agent.assertion_grounding import (
    classify_assertions,
    validate_assertion_grounding,
)
from qa_agent.models import (
    AssertionGrounding,
    DiscoveryResult,
    DiscoveryStatus,
    InteractiveElement,
    QATestPlan,
    QATestStep,
    TestStep as DomainTestStep,
)
from qa_agent.test_plan_validation import PlanValidationError


class AssertionGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.step = DomainTestStep(
            name="Verify confirmation",
            description="Verify the account confirmation state.",
            expected="The confirmation state is displayed.",
            order=0,
        )
        self.discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS, url="https://example.test/"
        )

    @staticmethod
    def plan(action: QATestStep) -> QATestPlan:
        return QATestPlan(url="https://example.test/", steps=[action])

    def test_generic_confirmation_requirement_allows_structural_assertion(self) -> None:
        discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.test/",
            snapshot={"visible_text_elements": [
                {"selector": "#confirmation", "tag": "p", "text": "Your account was created."}
            ]},
        )
        plan = self.plan(QATestStep(
            action="assert_visible", parameters={"selector": "#confirmation"}
        ))

        entries = validate_assertion_grounding(
            plan,
            self.step,
            discovery,
            requirement_context="Register an account and verify that a confirmation state is displayed.",
        )

        self.assertEqual(entries[0].category, AssertionGrounding.REQUIREMENT_GROUNDED)

    def test_exact_required_text_is_requirement_grounded(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_text_contains", parameters={"expected_text": "Registration successful"}
        ))
        discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url="https://example.test/")

        entries = validate_assertion_grounding(
            plan,
            self.step,
            discovery,
            requirement_context='The page must display the exact text "Registration successful".',
        )

        self.assertEqual(entries[0].category, AssertionGrounding.REQUIREMENT_GROUNDED)

    def test_atomic_step_requirement_is_used_when_case_description_is_unrelated(self) -> None:
        step = DomainTestStep(
            name="Verify title",
            description="Check the page title.",
            expected="The exact title is Home.",
            order=0,
        )
        plan = self.plan(QATestStep(
            action="assert_title", parameters={"expected": "Home"}
        ))
        discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url="https://example.test/")

        entries = validate_assertion_grounding(
            plan,
            step,
            discovery,
            requirement_context="Exercise a retry after one plan generation failure.",
        )

        self.assertEqual(entries[0].category, AssertionGrounding.REQUIREMENT_GROUNDED)

    def test_exact_observed_text_is_observation_grounded(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_text_contains", parameters={"expected_text": "Registration successful"}
        ))
        discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.test/",
            snapshot={"visible_text_elements": [
                {"selector": "#notice", "tag": "p", "text": "Registration successful. Welcome."}
            ]},
        )

        entries = validate_assertion_grounding(
            plan, self.step, discovery, requirement_context="Verify a confirmation state is displayed."
        )

        self.assertEqual(entries[0].category, AssertionGrounding.OBSERVATION_GROUNDED)

    def test_example_does_not_ground_an_exact_status_value(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_text_contains", parameters={"expected_text": "Unconfirmed"}
        ))
        discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url="https://example.test/")

        with self.assertRaises(PlanValidationError) as raised:
            validate_assertion_grounding(
                plan,
                self.step,
                discovery,
                requirement_context="Verify that a confirmation state is displayed (e.g. Unconfirmed).",
            )

        self.assertEqual(raised.exception.issues[0].code, "UNGROUNDED_ASSERTION")
        self.assertNotIn("Unconfirmed", str(raised.exception))

    def test_model_authored_step_specificity_does_not_upgrade_related_generic_scenario(self) -> None:
        generated_step = DomainTestStep(
            name="Verify confirmation status",
            description="The confirmation status (e.g. Unconfirmed) is shown.",
            expected="Unconfirmed is displayed.",
            order=0,
        )
        plan = self.plan(QATestStep(
            action="assert_text_contains", parameters={"expected_text": "Unconfirmed"}
        ))
        discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url="https://example.test/")

        with self.assertRaises(PlanValidationError) as raised:
            validate_assertion_grounding(
                plan,
                generated_step,
                discovery,
                requirement_context="Register a user and verify that a confirmation state is displayed.",
            )

        self.assertEqual(raised.exception.issues[0].code, "UNGROUNDED_ASSERTION")

    def test_ai_discovery_suggestion_is_not_observation_evidence(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_text_contains", parameters={"expected_text": "Unconfirmed"}
        ))
        discovery = DiscoveryResult(
            status=DiscoveryStatus.PARTIAL,
            url="https://example.test/",
            # Models a fallback suggestion merged into typed fields. The
            # deterministic snapshot remains the only observation source.
            interactive_elements=[InteractiveElement(
                kind="text", selector="#status", text="Unconfirmed", accessible_name="Unconfirmed"
            )],
        )

        categories = classify_assertions(
            plan, self.step, discovery,
            requirement_context="Verify that a confirmation state is displayed.",
        )

        self.assertEqual(categories[0].category, AssertionGrounding.INFERRED)

    def test_invented_status_value_is_rejected_without_exposing_the_value(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_selected",
            parameters={"selector": "#status", "expected": "Pending"},
        ))
        discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url="https://example.test/")

        with self.assertRaises(PlanValidationError) as raised:
            validate_assertion_grounding(
                plan, self.step, discovery,
                requirement_context="Verify that the account status is displayed.",
            )

        self.assertEqual(raised.exception.issues[0].code, "UNGROUNDED_ASSERTION")
        self.assertNotIn("Pending", str(raised.exception))

    def test_checkbox_unchecked_is_grounded_structurally_without_invented_label(self) -> None:
        step = DomainTestStep(
            name="Verify the terms checkbox",
            description="The terms checkbox remains unchecked.",
            expected="The terms checkbox is not checked.",
            order=0,
        )
        plan = self.plan(QATestStep(
            action="assert_unchecked", parameters={"selector": "#terms"}
        ))
        entries = validate_assertion_grounding(plan, step, self.discovery)

        self.assertEqual(entries[0].category, AssertionGrounding.REQUIREMENT_GROUNDED)
        self.assertEqual(set(plan.steps[0].parameters), {"selector"})

    def test_positive_checkbox_assertion_is_not_grounded_by_a_negative_requirement(self) -> None:
        step = DomainTestStep(
            name="Verify the checkbox",
            description="The checkbox must not be checked.",
            expected="The checkbox is unchecked.",
            order=0,
        )
        plan = self.plan(QATestStep(
            action="assert_checked", parameters={"selector": "#terms"}
        ))

        self.assertEqual(
            classify_assertions(plan, step, self.discovery)[0].category,
            AssertionGrounding.UNKNOWN,
        )

    def test_negated_requirement_does_not_ground_a_forbidden_exact_value(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_text_contains", parameters={"expected_text": "Unconfirmed"}
        ))
        discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url="https://example.test/")

        with self.assertRaises(PlanValidationError):
            validate_assertion_grounding(
                plan,
                self.step,
                discovery,
                requirement_context="The status must not be Unconfirmed.",
            )


if __name__ == "__main__":
    unittest.main()
