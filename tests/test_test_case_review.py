import sqlite3
import tempfile
import unittest
from pathlib import Path
from uuid import UUID, uuid4

from qa_agent.automation_lifecycle import plan_fingerprint
from qa_agent.drafts import DraftStatus, SQLiteDraftRepository
from qa_agent.models import (
    AssertionGrounding,
    AssertionGroundingEntry,
    PlanVersionOrigin,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.test_case_review import (
    InMemoryTestCaseReviewRepository,
    SQLiteTestCaseReviewRepository,
    TestCaseReviewService as ReviewService,
    TestCaseReviewStatus as ReviewStatus,
)


class TestCaseReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "review.sqlite3"
        self.plans = InMemoryPlanStore()
        self.repository = InMemoryTestCaseReviewRepository()
        self.review = ReviewService(self.repository, self.plans)
        self.case = DomainTestCase(
            name="Local title check",
            description="Verify that the local page displays its expected title.",
            base_url="http://127.0.0.1/local",
            steps=[DomainTestStep(
                name="Check the page title",
                description="Open the local page and inspect its title.",
                expected='The page title is "Local page".',
                order=0,
            )],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_legacy_case_defaults_to_approved_and_new_case_review_is_separate(self):
        self.assertEqual(
            self.review.status(self.case.id), ReviewStatus.APPROVED
        )
        self.review.mark_ready_for_review(self.case)
        self.assertEqual(
            self.review.status(self.case.id), ReviewStatus.READY_FOR_REVIEW
        )
        self.assertIsNone(self.review.record(self.case.id).approved_plan_fingerprint)
        self.review.approve_test_case(self.case)
        self.assertEqual(
            self.review.status(self.case.id), ReviewStatus.APPROVED
        )

    def test_validation_approval_pins_current_versions_and_plan_change_invalidates_it(self):
        step = self.case.steps[0]
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        first = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=1,
            origin=PlanVersionOrigin.AI_GENERATED,
            qa_test_plan=QATestPlan(
                url=self.case.base_url,
                steps=[QATestStep(action="assert_title", parameters={"expected": "Local page"})],
            ),
            assertion_grounding=(AssertionGroundingEntry(
                step_index=0, category=AssertionGrounding.REQUIREMENT_GROUNDED,
            ),),
        )
        self.plans.save(step.id, first, test_plan=plan)
        self.review.mark_ready_for_review(self.case)
        self.review.approve_test_case(self.case)

        pinned = self.review.approve_for_validation(self.case)
        self.assertEqual(pinned, plan_fingerprint(self.case, self.plans))
        self.assertTrue(self.review.validation_approved_for(self.case))

        second = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=2,
            origin=PlanVersionOrigin.HUMAN_EDITED,
            qa_test_plan=first.qa_test_plan,
            assertion_grounding=first.assertion_grounding,
        )
        self.plans.save(step.id, second, test_plan=plan)
        self.assertFalse(self.review.validation_approved_for(self.case))
        self.assertEqual(
            self.review.record(self.case.id).approved_plan_fingerprint, pinned
        )
        self.assertEqual(self.plans.get_version(first.id), first)

    def test_review_records_persist_outside_canonical_testcase_storage(self):
        repository = SQLiteTestCaseReviewRepository(self.database)
        review = ReviewService(repository, self.plans)
        review.mark_ready_for_review(self.case)
        review.approve_test_case(self.case)

        restored = ReviewService(
            SQLiteTestCaseReviewRepository(self.database), self.plans
        )
        self.assertEqual(restored.status(self.case.id), ReviewStatus.APPROVED)
        record = restored.record(self.case.id)
        self.assertIsNone(record.approved_plan_fingerprint)

    def test_draft_migration_adds_used_state_without_rewriting_existing_content(self):
        connection = sqlite3.connect(self.database)
        connection.execute(
            """CREATE TABLE drafts (
                draft_id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL,
                base_url TEXT, notes TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """INSERT INTO drafts VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "172b74bd-27ea-4b39-89cf-8836d5cc08d8",
                "Existing idea", "Check the local sign-in page.",
                "http://127.0.0.1/sign-in", None,
                "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()
        connection.close()

        drafts = SQLiteDraftRepository(self.database)
        draft = drafts.get(UUID("172b74bd-27ea-4b39-89cf-8836d5cc08d8"))
        self.assertEqual(draft.status, DraftStatus.ACTIVE)
        case_id = uuid4()
        self.assertTrue(drafts.mark_used(draft.id, case_id))
        restored = SQLiteDraftRepository(self.database).get(draft.id)
        self.assertEqual(restored.title, "Existing idea")
        self.assertEqual(restored.body, "Check the local sign-in page.")
        self.assertEqual(restored.status, DraftStatus.USED)
        self.assertEqual(restored.converted_test_case_id, case_id)


if __name__ == "__main__":
    unittest.main()
