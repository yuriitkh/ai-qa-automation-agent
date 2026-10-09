import unittest
from uuid import uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    AssertionGrounding,
    AssertionGroundingEntry,
    ExecutionStatus,
    QATestPlan,
    QATestStep,
    PlanVersionOrigin,
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
            assertion_grounding=(AssertionGroundingEntry(
                step_index=0,
                category=AssertionGrounding.REQUIREMENT_GROUNDED,
            ),),
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

    def test_successful_runner_evidence_is_attached_to_its_execution(self) -> None:
        outcome = self.execute(lambda _: {
            "status": "passed",
            "steps": [{"action": "assert_title", "status": "passed"}],
            "evidence": [{
                "type": "SCREENSHOT",
                "path": "artifacts/verification.png",
                "description": "Verification screenshot.",
                "scope": "PAGE",
                "event": "VERIFICATION:assert_title:0",
            }],
        })

        self.assertEqual(len(outcome.execution.evidence), 1)
        evidence = outcome.execution.evidence[0]
        self.assertEqual(evidence.execution_id, outcome.execution.id)
        self.assertEqual(evidence.scope.value, "PAGE")
        self.assertEqual(evidence.event, "VERIFICATION:assert_title:0")

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

    def test_inferred_exact_assertion_failure_is_not_product_failure(self) -> None:
        version = self.plan_version.model_copy(update={
            "assertion_grounding": (AssertionGroundingEntry(
                step_index=0, category=AssertionGrounding.INFERRED
            ),),
        })
        outcome = PlanExecutionService(
            lambda _: {
                "status": "failed",
                "steps": [{
                    "action": "assert_title", "status": "failed",
                    "error": "Expected title SECRET_EXPECTATION, got another title.",
                }],
            },
            self.repository,
        ).execute(self.test_step, version)

        self.assertEqual(
            outcome.classification,
            PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR,
        )
        self.assertNotIn("SECRET_EXPECTATION", outcome.execution.error or "")
        self.assertIn("not grounded", (outcome.execution.error or "").casefold())

    def test_observation_grounded_assertion_failure_is_automation_drift(self) -> None:
        version = self.plan_version.model_copy(update={
            "assertion_grounding": (AssertionGroundingEntry(
                step_index=0, category=AssertionGrounding.OBSERVATION_GROUNDED
            ),),
        })
        outcome = PlanExecutionService(
            lambda _: {
                "status": "failed",
                "steps": [{"action": "assert_title", "status": "failed", "error": "Mismatch."}],
            },
            self.repository,
        ).execute(self.test_step, version)

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.AUTOMATION_DRIFT
        )

    def test_legacy_assertion_without_metadata_is_conservative(self) -> None:
        version = self.plan_version.model_copy(update={"assertion_grounding": None})
        outcome = PlanExecutionService(
            lambda _: {
                "status": "failed",
                "steps": [{"action": "assert_title", "status": "failed", "error": "Mismatch."}],
            },
            self.repository,
        ).execute(self.test_step, version)

        self.assertEqual(
            outcome.classification,
            PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR,
        )
        self.assertIn("no grounding metadata", outcome.execution.error or "")

    def test_human_edited_exact_assertion_failure_remains_product_failure(self) -> None:
        version = self.plan_version.model_copy(update={
            "origin": PlanVersionOrigin.HUMAN_EDITED,
            "assertion_grounding": None,
        })
        outcome = PlanExecutionService(
            lambda _: {
                "status": "failed",
                "steps": [{"action": "assert_title", "status": "failed", "error": "Mismatch."}],
            },
            self.repository,
        ).execute(self.test_step, version)

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.PRODUCT_FAILURE
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

    def test_generated_plan_with_missing_assertion_target_is_execution_error(self) -> None:
        outcome = self.execute(lambda _: {
            "status": "failed",
            "steps": [{
                "action": "assert_visible",
                "status": "failed",
                "error": "Selector '#submit' was not found on the page.",
            }],
        })

        self.assertEqual(
            outcome.classification,
            PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR,
        )

    def test_generated_plan_interaction_failure_is_automation_execution_error(self) -> None:
        outcome = self.execute(lambda _: {
            "status": "failed",
            "steps": [{
                "action": "click",
                "status": "failed",
                "error": "Click failed because an overlay intercepted the action.",
            }],
        })

        self.assertEqual(
            outcome.classification,
            PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR,
        )

    def test_browser_failure_returned_by_runner_is_infrastructure_error(self) -> None:
        outcome = self.execute(lambda _: {
            "status": "failed",
            "steps": [{
                "action": "click",
                "status": "failed",
                "error": "Browser has been closed unexpectedly.",
            }],
        })

        self.assertEqual(
            outcome.classification, PlanExecutionClassification.INFRASTRUCTURE_ERROR
        )

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
        self.assertEqual(outcome.execution.runner_result, {"qa_classification": "INFRASTRUCTURE_ERROR"})
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
