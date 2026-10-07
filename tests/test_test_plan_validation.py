import unittest

from qa_agent.models import QATestPlan, QATestStep
from qa_agent.test_plan_validation import PlanValidationError, validate_executable_plan


class ExecutablePlanValidationTests(unittest.TestCase):
    def test_accepts_valid_single_action_plan(self):
        plan = validate_executable_plan({
            "url": "http://127.0.0.1:43123/register",
            "steps": [{"action": "navigate", "parameters": {"url": "http://127.0.0.1:43123/register"}}],
        })

        self.assertEqual(plan.steps[0].action, "navigate")

    def test_accepts_multi_action_registration_and_compound_verification(self):
        plan = QATestPlan(
            url="http://127.0.0.1:43123/register",
            steps=[
                QATestStep(action="fill", parameters={"selector": "#first-name", "value": "Ada"}),
                QATestStep(action="fill", parameters={"selector": "#last-name", "value": "Lovelace"}),
                QATestStep(action="fill", parameters={"selector": "#email", "value": "ada@example.test"}),
                QATestStep(action="click", parameters={"selector": "#create-account"}),
                QATestStep(action="assert_text_contains", parameters={"expected_text": "Account created"}),
            ],
        )

        self.assertEqual(len(validate_executable_plan(plan).steps), 5)

    def test_click_missing_locator_has_exact_safe_diagnostic(self):
        with self.assertRaises(PlanValidationError) as raised:
            validate_executable_plan(QATestPlan(
                url="http://127.0.0.1/register",
                steps=[QATestStep(action="click")],
            ))

        issue = raised.exception.issues[0]
        self.assertEqual((issue.code, issue.path), ("MISSING_LOCATOR", "steps[0].parameters.selector"))
        self.assertEqual(issue.message, "CLICK action requires a locator.")

    def test_fill_missing_value_is_rejected_without_echoing_other_parameters(self):
        private_value = "LOCAL_SECRET_SENTINEL_77"
        with self.assertRaises(PlanValidationError) as raised:
            validate_executable_plan({
                "url": "http://127.0.0.1/register",
                "steps": [{
                    "action": "fill",
                    "parameters": {"selector": "#email", "expected": private_value},
                }],
            })

        issue = raised.exception.issues[0]
        self.assertEqual(issue.code, "MISSING_INPUT_VALUE")
        self.assertEqual(issue.path, "steps[0].parameters.value")
        self.assertNotIn(private_value, str(raised.exception))

    def test_unsupported_action_and_empty_plan_are_rejected(self):
        with self.assertRaises(PlanValidationError) as unsupported:
            validate_executable_plan({
                "url": "http://127.0.0.1/register",
                "steps": [{"action": "execute_script", "parameters": {}}],
            })
        self.assertEqual(unsupported.exception.issues[0].code, "UNSUPPORTED_ACTION")

        with self.assertRaises(PlanValidationError) as empty:
            validate_executable_plan({"url": "http://127.0.0.1/register", "steps": []})
        self.assertEqual(empty.exception.issues[0].code, "EMPTY_ACTION_SEQUENCE")


if __name__ == "__main__":
    unittest.main()
