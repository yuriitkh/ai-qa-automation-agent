import unittest

from qa_agent.assertion_grounding import classify_assertions, validate_assertion_grounding
from qa_agent.expected_result_coverage import (
    ExpectedResultCoverageStatus,
    expected_result_coverage,
    validate_expected_result_coverage,
)
from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    QATestPlan,
    QATestStep,
    TestStep as DomainTestStep,
)
from qa_agent.test_plan_validation import PlanValidationError


class ExpectedResultCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.step = DomainTestStep(
            name="Enter an invalid email and verify an error message is displayed",
            description="Submit the invalid email and verify the error message.",
            expected="An error message is displayed.",
            order=0,
        )
        self.discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.test/register",
        )

    @staticmethod
    def plan(*actions: QATestStep) -> QATestPlan:
        return QATestPlan(url="https://example.test/register", steps=list(actions) or [
            QATestStep(action="click", parameters={"selector": "#submit"}),
        ])

    def test_verification_step_without_assertion_is_not_covered(self) -> None:
        result = expected_result_coverage(
            self.step,
            self.plan(
                QATestStep(action="fill", parameters={"selector": "#email", "value": "bad"}),
                QATestStep(action="click", parameters={"selector": "#submit"}),
            ),
        )

        self.assertEqual(result.status, ExpectedResultCoverageStatus.NOT_COVERED)

    def test_relevant_visible_assertion_covers_error_message(self) -> None:
        result = expected_result_coverage(
            self.step,
            self.plan(QATestStep(
                action="assert_visible", parameters={"selector": "#error-message"}
            )),
        )

        self.assertEqual(result.status, ExpectedResultCoverageStatus.COVERED)
        self.assertEqual(result.matching_action_indexes, (0,))

    def test_irrelevant_title_assertion_does_not_cover_error_message(self) -> None:
        result = expected_result_coverage(
            self.step,
            self.plan(QATestStep(
                action="assert_title", parameters={"expected": "Home"}
            )),
        )

        self.assertEqual(result.status, ExpectedResultCoverageStatus.NOT_COVERED)

    def test_relevant_text_assertion_covers_error_message(self) -> None:
        result = expected_result_coverage(
            self.step,
            self.plan(QATestStep(
                action="assert_text_contains",
                parameters={"expected_text": "Invalid email format"},
            )),
        )

        self.assertEqual(result.status, ExpectedResultCoverageStatus.COVERED)

    def test_action_only_step_needs_no_assertion(self) -> None:
        step = DomainTestStep(
            name="Open the login page",
            description="Navigate to the login page.",
            expected="The login page is opened.",
            order=0,
        )

        result = expected_result_coverage(step, self.plan())

        self.assertEqual(
            result.status,
            ExpectedResultCoverageStatus.NO_VERIFICATION_REQUIRED,
        )

    def test_relevant_but_ungrounded_assertion_is_not_acceptable(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_text_contains",
            parameters={"expected_text": "Invalid email format"},
        ))
        result = expected_result_coverage(self.step, plan)

        self.assertEqual(result.status, ExpectedResultCoverageStatus.COVERED)
        entries = classify_assertions(plan, self.step, self.discovery)
        self.assertEqual(entries[0].category.value, "INFERRED")
        with self.assertRaises(PlanValidationError) as raised:
            validate_assertion_grounding(plan, self.step, self.discovery)
        self.assertEqual(raised.exception.issues[0].code, "UNGROUNDED_ASSERTION")

    def test_grounded_but_irrelevant_assertion_is_not_coverage(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_title", parameters={"expected": "Home"}
        ))
        discovery = self.discovery.model_copy(update={"title": "Home"})

        entries = validate_assertion_grounding(plan, self.step, discovery)
        result = expected_result_coverage(self.step, plan)

        self.assertEqual(entries[0].category.value, "OBSERVATION_GROUNDED")
        self.assertEqual(result.status, ExpectedResultCoverageStatus.NOT_COVERED)

    def test_grounded_relevant_assertion_is_acceptable(self) -> None:
        plan = self.plan(QATestStep(
            action="assert_visible", parameters={"selector": "#error-message"}
        ))

        entries = validate_assertion_grounding(plan, self.step, self.discovery)
        result = validate_expected_result_coverage(self.step, plan)

        self.assertEqual(entries[0].category.value, "REQUIREMENT_GROUNDED")
        self.assertEqual(result.status, ExpectedResultCoverageStatus.COVERED)

    def test_two_expected_results_report_partial_coverage(self) -> None:
        step = self.step.model_copy(update={
            "expected": "An error message is displayed and the submit button is disabled."
        })
        plan = self.plan(QATestStep(
            action="assert_visible", parameters={"selector": "#error-message"}
        ))

        result = expected_result_coverage(step, plan)

        self.assertEqual(result.status, ExpectedResultCoverageStatus.PARTIALLY_COVERED)

    def test_unrecognized_required_result_is_unknown_and_fails_closed(self) -> None:
        step = self.step.model_copy(update={
            "name": "Verify the account state",
            "description": "Review the account outcome.",
            "expected": "The account reaches the correct state.",
        })

        result = expected_result_coverage(step, self.plan())

        self.assertEqual(result.status, ExpectedResultCoverageStatus.UNKNOWN)
        with self.assertRaises(PlanValidationError):
            validate_expected_result_coverage(step, self.plan())


if __name__ == "__main__":
    unittest.main()
