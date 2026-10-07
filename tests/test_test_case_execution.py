import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qa_agent.demo import seed_demo_data
from qa_agent.models import (
    ExecutionSegment,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_execution import RunUnavailableError, TestCaseExecutionService as RunCaseService


class TestCaseExecutionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "service.sqlite3"
        seed_demo_data(
            self.database,
            demo_base_url="http://127.0.0.1:8765/demo-target/registration",
        )
        self.storage = create_sqlite_storage(self.database)
        self.local_case = next(
            case for case in self.storage.test_case_repository.list()
            if case.name == "Local registration demo"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_service_executes_saved_pins_through_workflow_and_records_once(self) -> None:
        received_plans = []

        def runner(plan):
            received_plans.append(plan)
            return {"status": "passed", "url": plan.url, "steps": [], "evidence": []}

        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            evidence_directory=self.root / "evidence",
            runner_factory=lambda _directory: runner,
        )
        result = service.run(self.local_case.id, WorkflowType.REGRESSION)

        self.assertEqual(len(received_plans), 3)
        self.assertEqual(result.outcome.value, "PASSED")
        self.assertEqual(len(result.test_run.executions), 3)
        record = self.storage.run_history.get(result.test_run.id)
        self.assertIsNotNone(record)
        self.assertEqual(record.workflow_type, WorkflowType.REGRESSION)
        self.assertEqual(
            [item.test_plan_version_id for item in record.executions],
            [self.storage.plan_store.find(step.id).id for step in self.local_case.steps],
        )
        self.assertEqual(
            len(self.storage.run_history.list_for_test_case(self.local_case.id)), 1
        )

    def test_missing_saved_plan_fails_before_workflow_or_history_creation(self) -> None:
        case = DomainTestCase(
            name="Unplanned case",
            description="No saved browser plan.",
            segments=[ExecutionSegment(order=0, steps=[DomainTestStep(
                name="Check title",
                description="Open the page.",
                expected="A title is present.",
                order=0,
            )])],
        )
        self.storage.test_case_repository.save(case)
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            runner_factory=lambda _directory: self.fail("Runner must not start"),
        )

        with self.assertRaisesRegex(RunUnavailableError, "no saved plan"):
            service.run(case.id, WorkflowType.VALIDATION)
        self.assertEqual(self.storage.run_history.list_for_test_case(case.id), [])

    def test_automation_is_not_exposed_for_persisted_testcases(self) -> None:
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            runner_factory=lambda _directory: self.fail("Runner must not start"),
        )

        with self.assertRaisesRegex(RunUnavailableError, "Automation is not available"):
            service.run(self.local_case.id, WorkflowType.AUTOMATION)
        self.assertEqual(self.storage.run_history.list_for_test_case(self.local_case.id), [])

    def test_complete_plan_set_makes_validation_and_regression_available(self) -> None:
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
        )

        availability = service.workflow_availability(self.local_case.id)

        self.assertFalse(availability.automation_available)
        self.assertTrue(availability.validation_available)
        self.assertTrue(availability.regression_available)
        self.assertEqual(availability.usable_plan_count, len(self.local_case.steps))
        self.assertEqual(
            [item[2] for item in availability.plan_versions],
            [self.storage.plan_store.find(step.id).id for step in self.local_case.steps],
        )

    def test_incomplete_plan_set_keeps_pinned_workflows_unavailable(self) -> None:
        case = DomainTestCase(
            name="One missing plan",
            description="Check a page.",
            segments=[ExecutionSegment(order=0, steps=[DomainTestStep(
                name="Check title", description="Open the page.",
                expected="A title is present.", order=0,
            )])],
        )
        self.storage.test_case_repository.save(case)
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
        )

        availability = service.workflow_availability(case.id)

        self.assertFalse(availability.validation_available)
        self.assertFalse(availability.regression_available)
        self.assertEqual(availability.reason, "No complete automation version has been generated yet.")
        with self.assertRaisesRegex(RunUnavailableError, "no saved plan"):
            service.run(case.id, WorkflowType.VALIDATION)

    def test_plan_missing_required_action_input_does_not_count_as_usable_coverage(self) -> None:
        step = DomainTestStep(
            name="Submit registration",
            description="Submit the form.",
            expected="Registration completes.",
            order=0,
        )
        case = DomainTestCase(
            name="Invalid saved plan",
            description="The saved click plan has no selector.",
            segments=[ExecutionSegment(order=0, steps=[step])],
        )
        self.storage.test_case_repository.save(case)
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url="http://127.0.0.1/",
                steps=[QATestStep(action="click", parameters={})],
            ),
        )
        self.storage.plan_store.save(step.id, version, test_plan=plan)
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
        )

        availability = service.workflow_availability(case.id)
        self.assertEqual(availability.usable_plan_count, 0)
        self.assertFalse(availability.validation_available)
        self.assertFalse(availability.regression_available)
        with self.assertRaisesRegex(RunUnavailableError, "saved plans are not usable"):
            service.run(case.id, WorkflowType.VALIDATION)

    def test_automation_delegates_to_existing_workflow_for_canonical_case(self) -> None:
        class FakeAutomation:
            def __init__(self):
                self.received = []

            def run_test_case(self, test_case, run_context):
                self.received.append((test_case, run_context))
                return SimpleNamespace(test_run=SimpleNamespace(id="run-automation"))

        automation = FakeAutomation()
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            automation_workflow=automation,
        )

        result = service.run(self.local_case.id, WorkflowType.AUTOMATION)

        self.assertEqual(result.test_run.id, "run-automation")
        self.assertEqual(len(automation.received), 1)
        self.assertEqual(automation.received[0][0].id, self.local_case.id)
        self.assertEqual(self.storage.run_history.list_for_test_case(self.local_case.id), [])


if __name__ == "__main__":
    unittest.main()
