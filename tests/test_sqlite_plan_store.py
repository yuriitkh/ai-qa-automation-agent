import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qa_agent.models import (
    QATestPlan,
    QATestStep,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.sqlite_storage import SQLitePlanStore


class SQLitePlanStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "agent.sqlite3"
        self.step = DomainTestStep(
            name="Open destination",
            description="Open the destination page.",
            expected="The destination page is loaded.",
            order=0,
        )
        self.plan = DomainTestPlan(test_step_id=self.step.id, name=self.step.name)
        self.qa_plan = QATestPlan(
            url="https://example.com/",
            steps=[
                QATestStep(action="navigate", parameters={"url": "https://example.com/"}),
                QATestStep(action="assert_page_loaded"),
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_version(self, number: int, *, version_id=None) -> DomainTestPlanVersion:
        return DomainTestPlanVersion(
            id=version_id or uuid4(),
            test_plan_id=self.plan.id,
            version=number,
            created_at=datetime(2025, 1, number, tzinfo=timezone.utc),
            qa_test_plan=self.qa_plan,
        )

    def test_plan_version_and_executable_plan_survive_new_store_instance(self) -> None:
        version = self.make_version(1)
        SQLitePlanStore(self.db_path).save(
            self.step.id,
            version,
            test_plan=self.plan,
        )

        reopened = SQLitePlanStore(self.db_path)
        restored_version = reopened.find(self.step.id)
        restored_plan = reopened.find_test_plan(self.step.id)

        self.assertEqual(restored_version, version)
        self.assertEqual(restored_version.qa_test_plan, self.qa_plan)
        self.assertEqual(restored_plan, self.plan)
        self.assertEqual(restored_version.test_plan_id, restored_plan.id)
        self.assertEqual(restored_plan.test_step_id, self.step.id)

    def test_version_two_preserves_version_one_and_both_survive_reopen(self) -> None:
        store = SQLitePlanStore(self.db_path)
        version_one = self.make_version(1)
        version_two = self.make_version(2)
        store.save(self.step.id, version_one, test_plan=self.plan)
        store.save(self.step.id, version_two, test_plan=self.plan)

        reopened = SQLitePlanStore(self.db_path)
        restored = reopened.find(self.step.id)

        self.assertEqual(restored.version, 2)
        self.assertEqual(restored.id, version_two.id)
        self.assertEqual(restored.test_plan_id, self.plan.id)
        self.assertEqual(reopened.get_version(version_one.id), version_one)
        self.assertEqual(reopened.get_version(version_two.id), version_two)

    def test_rejects_rewriting_an_existing_version_id(self) -> None:
        store = SQLitePlanStore(self.db_path)
        version_one = self.make_version(1)
        store.save(self.step.id, version_one, test_plan=self.plan)
        changed = version_one.model_copy(update={
            "qa_test_plan": QATestPlan(
                url="https://changed.example/",
                steps=[QATestStep(action="assert_title", parameters={"expected": "Changed"})],
            )
        })

        with self.assertRaises(ValueError):
            store.save(self.step.id, changed, test_plan=self.plan)
        self.assertEqual(store.get_version(version_one.id), version_one)

    def test_rejects_invalid_relationships_downgrade_and_reassignment(self) -> None:
        store = SQLitePlanStore(self.db_path)
        version_one = self.make_version(1)
        store.save(self.step.id, version_one, test_plan=self.plan)

        with self.assertRaises(ValueError):
            store.save(self.step.id, version_one, test_plan=DomainTestPlan(
                test_step_id=self.step.id,
                name="Unrelated plan",
            ))

        with self.assertRaises(ValueError):
            store.save(self.step.id, self.make_version(1), test_plan=self.plan)

        another_step = DomainTestStep(
            name="Other step",
            description="A different step.",
            expected="It passes.",
            order=1,
        )
        with self.assertRaises(ValueError):
            store.save(another_step.id, version_one, test_plan=self.plan)

        self.assertEqual(SQLitePlanStore(self.db_path).find(self.step.id).id, version_one.id)

    def test_unknown_step_returns_none(self) -> None:
        store = SQLitePlanStore(self.db_path)
        self.assertIsNone(store.find(uuid4()))
        self.assertIsNone(store.find_test_plan(uuid4()))


if __name__ == "__main__":
    unittest.main()
