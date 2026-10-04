import tempfile
import unittest
from pathlib import Path

from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    ExecutionStatus,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.pipeline import QATestPipeline
from qa_agent.sqlite_storage import SQLiteExecutionRepository, SQLitePlanStore
from qa_agent.test_plan_generator import GeneratedTestPlan


class SQLitePipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "pipeline.sqlite3"
        self.step = DomainTestStep(
            name="Open page",
            description="Open the example page.",
            expected="The example page is loaded.",
            order=0,
        )
        self.test_case = DomainTestCase(
            name="Example",
            description="Open the example page.",
            base_url="https://example.com/",
            steps=[self.step],
        )
        self.discovery_result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.com/",
            title="Example",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_new_store_instances_reuse_plan_after_restart(self) -> None:
        decomposer = _FixedDecomposer(self.test_case)
        generator = _FakeGenerator()
        discovery_calls: list[str] = []
        runner_calls: list[QATestPlan] = []

        def discovery(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            return self.discovery_result

        def runner(plan: QATestPlan) -> dict:
            runner_calls.append(plan)
            return {"status": "passed", "url": plan.url, "steps": []}

        first_pipeline = QATestPipeline(
            decomposer=decomposer,
            plan_generator=generator,
            discovery=discovery,
            runner=runner,
            plan_store=SQLitePlanStore(self.db_path),
            execution_repository=SQLiteExecutionRepository(self.db_path),
        )
        first = first_pipeline.run("Open the example page.")

        # Fresh adapters model a new process opening the same SQLite database.
        second_pipeline = QATestPipeline(
            decomposer=decomposer,
            plan_generator=generator,
            discovery=discovery,
            runner=runner,
            plan_store=SQLitePlanStore(self.db_path),
            execution_repository=SQLiteExecutionRepository(self.db_path),
        )
        second = second_pipeline.run("Open the example page.")

        self.assertEqual(discovery_calls, ["https://example.com/"])
        self.assertEqual(generator.calls, 1)
        self.assertEqual(len(runner_calls), 2)
        self.assertEqual(first.test_plans[0].test_plan_version.id, second.test_plans[0].test_plan_version.id)
        self.assertEqual(first.test_plans[0].test_plan_version.version, 1)
        self.assertEqual(first.test_plans[0].test_plan_version.qa_test_plan, second.test_plans[0].test_plan_version.qa_test_plan)
        self.assertEqual(
            SQLiteExecutionRepository(self.db_path).list_for_test_step(self.step.id),
            first.executions + second.executions,
        )

    def test_stale_retry_persists_both_executions_and_version_two(self) -> None:
        test_plan = DomainTestPlan(test_step_id=self.step.id, name=self.step.name)
        version_one = DomainTestPlanVersion(
            test_plan_id=test_plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url="https://example.com/",
                steps=[QATestStep(action="click", parameters={"selector": "#old"})],
            ),
        )
        SQLitePlanStore(self.db_path).save(
            self.step.id,
            version_one,
            test_plan=test_plan,
        )
        generator = _FakeGenerator()
        discovery_calls: list[str] = []
        runner_calls: list[QATestPlan] = []

        def discovery(url: str) -> DiscoveryResult:
            discovery_calls.append(url)
            return self.discovery_result

        def runner(plan: QATestPlan) -> dict:
            runner_calls.append(plan)
            if len(runner_calls) == 1:
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
            decomposer=_FixedDecomposer(self.test_case),
            plan_generator=generator,
            discovery=discovery,
            runner=runner,
            plan_store=SQLitePlanStore(self.db_path),
            execution_repository=SQLiteExecutionRepository(self.db_path),
        )

        result = pipeline.run("Open the example page.")

        self.assertEqual(
            [execution.status for execution in result.executions],
            [ExecutionStatus.FAILED, ExecutionStatus.PASSED],
        )
        self.assertEqual(
            [execution.test_plan_version_id for execution in result.executions],
            [version_one.id, result.test_plans[-1].test_plan_version.id],
        )
        self.assertEqual(len(discovery_calls), 1)
        self.assertEqual(generator.calls, 1)
        self.assertEqual(generator.last_existing_plan.id, test_plan.id)
        self.assertEqual(generator.last_version_number, 2)

        reopened_plan_store = SQLitePlanStore(self.db_path)
        self.assertEqual(reopened_plan_store.find(self.step.id).version, 2)
        self.assertEqual(reopened_plan_store.get_version(version_one.id), version_one)
        self.assertEqual(
            reopened_plan_store.get_version(result.executions[1].test_plan_version_id),
            result.test_plans[-1].test_plan_version,
        )
        self.assertEqual(
            [entry.status for entry in SQLiteExecutionRepository(self.db_path).list_for_test_step(self.step.id)],
            [ExecutionStatus.FAILED, ExecutionStatus.PASSED],
        )


class _FixedDecomposer:
    def __init__(self, test_case: DomainTestCase) -> None:
        self._test_case = test_case

    def decompose(self, task: str, base_url: str | None = None) -> DomainTestCase:
        return self._test_case


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self.last_existing_plan: DomainTestPlan | None = None
        self.last_version_number = 0

    def generate_with_plan(
        self,
        test_step: DomainTestStep,
        discovery_result: DiscoveryResult,
        *,
        existing_test_plan: DomainTestPlan | None = None,
        version_number: int = 1,
    ) -> GeneratedTestPlan:
        self.calls += 1
        self.last_existing_plan = existing_test_plan
        self.last_version_number = version_number
        plan = existing_test_plan or DomainTestPlan(
            test_step_id=test_step.id,
            name=test_step.name,
        )
        qa_plan = QATestPlan(
            url=discovery_result.url,
            steps=[QATestStep(action="assert_page_loaded")],
        )
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=version_number,
            qa_test_plan=qa_plan,
        )
        return GeneratedTestPlan(test_plan=plan, test_plan_version=version)


if __name__ == "__main__":
    unittest.main()
