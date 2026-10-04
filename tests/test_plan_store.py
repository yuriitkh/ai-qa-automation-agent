import unittest
from uuid import uuid4

from qa_agent.models import (
    QATestPlan,
    QATestStep,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.plan_store import InMemoryPlanStore


class InMemoryPlanStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryPlanStore()
        self.step = DomainTestStep(
            name="Check page",
            description="Open the page.",
            expected="The page loads.",
            order=0,
        )
        self.plan = DomainTestPlan(test_step_id=self.step.id, name=self.step.name)
        self.version = DomainTestPlanVersion(
            test_plan_id=self.plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url="https://example.com/",
                steps=[QATestStep(action="assert_page_loaded")],
            ),
        )

    def test_save_and_find_returns_the_same_version_and_plan(self) -> None:
        self.store.save(self.step.id, self.version, test_plan=self.plan)

        self.assertEqual(self.store.find(self.step.id), self.version)
        self.assertEqual(self.store.get_version(self.version.id), self.version)
        self.assertEqual(self.store.find_test_plan(self.step.id), self.plan)

    def test_unknown_step_returns_none(self) -> None:
        self.assertIsNone(self.store.find(uuid4()))
        self.assertIsNone(self.store.get_version(uuid4()))
        self.assertIsNone(self.store.find_test_plan(uuid4()))

    def test_save_requires_test_plan_relationship(self) -> None:
        with self.assertRaises(TypeError):
            self.store.save(self.step.id, self.version)  # type: ignore[call-arg]

    def test_distinct_test_step_ids_do_not_conflict(self) -> None:
        other_step = DomainTestStep(
            name="Other check",
            description="Check another page.",
            expected="It loads.",
            order=1,
        )
        other_plan = DomainTestPlan(test_step_id=other_step.id, name=other_step.name)
        other_version = DomainTestPlanVersion(
            test_plan_id=other_plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url="https://example.org/",
                steps=[QATestStep(action="assert_page_loaded")],
            ),
        )

        self.store.save(self.step.id, self.version, test_plan=self.plan)
        self.store.save(other_step.id, other_version, test_plan=other_plan)

        self.assertEqual(self.store.find(self.step.id), self.version)
        self.assertEqual(self.store.find(other_step.id), other_version)

    def test_rejects_mismatched_plan_relationships(self) -> None:
        wrong_plan = DomainTestPlan(test_step_id=self.step.id, name="Wrong plan")

        with self.assertRaises(ValueError):
            self.store.save(self.step.id, self.version, test_plan=wrong_plan)

    def test_rejects_plan_from_another_test_step(self) -> None:
        other_step = DomainTestStep(
            name="Other check",
            description="Different step.",
            expected="It passes.",
            order=1,
        )
        other_plan = DomainTestPlan(test_step_id=other_step.id, name=other_step.name)
        other_version = DomainTestPlanVersion(
            test_plan_id=other_plan.id,
            version=1,
            qa_test_plan=self.version.qa_test_plan,
        )

        with self.assertRaises(ValueError):
            self.store.save(self.step.id, other_version, test_plan=other_plan)

    def test_rejects_replacing_current_plan_or_downgrading_version(self) -> None:
        self.store.save(self.step.id, self.version, test_plan=self.plan)
        unrelated_plan = DomainTestPlan(test_step_id=self.step.id, name="Other plan")
        unrelated_version = DomainTestPlanVersion(
            test_plan_id=unrelated_plan.id,
            version=2,
            qa_test_plan=self.version.qa_test_plan,
        )
        with self.assertRaises(ValueError):
            self.store.save(self.step.id, unrelated_version, test_plan=unrelated_plan)

        version_two = DomainTestPlanVersion(
            test_plan_id=self.plan.id,
            version=2,
            qa_test_plan=self.version.qa_test_plan,
        )
        self.store.save(self.step.id, version_two, test_plan=self.plan)
        with self.assertRaises(ValueError):
            self.store.save(self.step.id, self.version, test_plan=self.plan)

    def test_keeps_immutable_historical_versions_and_returns_current(self) -> None:
        self.store.save(self.step.id, self.version, test_plan=self.plan)
        version_two = DomainTestPlanVersion(
            test_plan_id=self.plan.id,
            version=2,
            qa_test_plan=QATestPlan(
                url="https://example.com/",
                steps=[QATestStep(action="click", parameters={"selector": "#new"})],
            ),
        )
        self.store.save(self.step.id, version_two, test_plan=self.plan)

        # A mutable nested value from a retrieved model must not rewrite storage.
        retrieved_one = self.store.get_version(self.version.id)
        retrieved_one.qa_test_plan.steps[0].parameters["selector"] = "#mutated"

        self.assertEqual(self.store.find(self.step.id), version_two)
        self.assertEqual(self.store.get_version(self.version.id), self.version)
        self.assertEqual(self.store.get_version(version_two.id), version_two)


if __name__ == "__main__":
    unittest.main()
