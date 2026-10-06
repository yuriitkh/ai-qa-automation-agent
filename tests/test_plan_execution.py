import unittest
from uuid import uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    ExecutionStatus,
    QATestPlan,
    QATestStep,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.plan_execution import (
    PlanExecutionClassification,
    PlanExecutionService,
)


class PlanExecutionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_step = DomainTestStep(
            name="Verify title",
            description="Verify the page title",
            expected="The expected title is visible",
            order=0,
        )
        self.plan_version = DomainTestPlanVersion(
            test_plan_id=uuid4(),
            version=7,
            qa_test_plan=QATestPlan(
                url="https://example.com",
                steps=[QATestStep(action="assert_title", parameters={"expected": "Home"})],
            ),
        )
        self.repository = InMemoryExecutionRepository()

    def execute(self, runner):
        service = PlanExecutionService(runner, self.repository)
        return service.execute(self.test_step, self.plan_version)

    def test_success_is_passed_and_saves_one_execution_for_exact_version(self) -> None:
        received_plans = []

        def runner(plan):
            received_plans.append(plan)
            return {"status": "passed", "steps": []}

        outcome = self.execute(runner)

        self.assertEqual(outcome.classification, PlanExecutionClassification.PASSED)
        self.assertIs(received_plans[0], self.plan_version.qa_test_plan)
        self.assertEqual(outcome.execution.status, ExecutionStatus.PASSED)
        self.assertEqual(outcome.execution.test_plan_version_id, self.plan_version.id)
        saved = self.repository.list_for_test_step(self.test_step.id)
        self.assertEqual(saved, [outcome.execution])

    def test_assertion_failure_is_product_failure_not_automation_drift(self) -> None:
        outcome = self.execute(lambda _: {
            "status": "failed",
            "steps": [{
                "action": "assert_title",
                "status": "failed",
                "error": "Expected title 'Home', but got 'Different'.",
            }],
        })

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.PRODUCT_FAILURE
        )
        self.assertEqual(outcome.execution.status, ExecutionStatus.FAILED)
        self.assertEqual(
            len(self.repository.list_for_test_step(self.test_step.id)), 1
        )

    def test_existing_stale_selector_failure_is_automation_drift(self) -> None:
        outcome = self.execute(lambda _: {
            "status": "failed",
            "steps": [{
                "action": "click",
                "status": "failed",
                "error": "Selector '#old' was not found on the page.",
            }],
        })

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.AUTOMATION_DRIFT
        )
        self.assertEqual(outcome.execution.test_plan_version_id, self.plan_version.id)

    def test_runner_exception_is_infrastructure_error_and_persists_failed_execution(self) -> None:
        runner_error = RuntimeError("browser launch failed")

        def runner(_):
            raise runner_error

        outcome = self.execute(runner)

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.INFRASTRUCTURE_ERROR
        )
        self.assertIs(outcome.error, runner_error)
        self.assertEqual(outcome.execution.status, ExecutionStatus.FAILED)
        self.assertEqual(outcome.execution.error, "browser launch failed")
        self.assertIsNone(outcome.execution.runner_result)
        self.assertEqual(
            self.repository.list_for_test_step(self.test_step.id), [outcome.execution]
        )

    def test_unclassified_runner_failure_is_conservatively_infrastructure_error(self) -> None:
        outcome = self.execute(lambda _: {"status": "failed", "steps": []})

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.INFRASTRUCTURE_ERROR
        )

    def test_unsupported_runner_status_is_infrastructure_error(self) -> None:
        outcome = self.execute(lambda _: {"status": "unknown", "steps": []})

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.INFRASTRUCTURE_ERROR
        )
        self.assertIn("unsupported status", outcome.execution.error or "")


if __name__ == "__main__":
    unittest.main()
