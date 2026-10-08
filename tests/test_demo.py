import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qa_agent.demo import seed_demo_data
from qa_agent.models import ExecutionStatus
from qa_agent.run_history import (
    RunHistoryRecord,
    WorkflowType,
)
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_execution import TestCaseExecutionService
from qa_agent.web import LocalWebApplication


class DemoSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "demo.sqlite3"
        self.storage = create_sqlite_storage(self.database)
        self.repository = self.storage.run_history_repository
        self.application = LocalWebApplication(
            self.storage.run_history,
            test_cases=self.storage.test_case_repository,
            run_service=TestCaseExecutionService(
                self.storage.test_case_repository,
                self.storage.plan_store,
                self.storage.execution_repository,
                self.storage.run_history,
                evidence_directory=Path(self.temporary_directory.name) / "evidence",
            ),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_seed_is_idempotent_preserves_user_rows_and_feeds_ui_and_reports(self) -> None:
        user_record = RunHistoryRecord(
            run_id=uuid4(),
            test_case_id=uuid4(),
            test_case_name="User-owned run",
            test_case_description="Existing history must remain intact.",
            workflow_type=WorkflowType.AUTOMATION,
            outcome="PASSED",
            status=ExecutionStatus.PASSED,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        saved_user_record = self.repository.save(user_record)

        first = seed_demo_data(self.database)
        second = seed_demo_data(self.database)

        self.assertEqual((first.created, first.skipped), (3, 0))
        self.assertEqual((second.created, second.skipped), (0, 3))
        records = self.repository.list_recent(limit=10)
        self.assertEqual(len(records), 4)
        self.assertEqual(self.repository.get(user_record.run_id), saved_user_record)

        definitions = self.storage.test_case_repository.list()
        self.assertEqual(
            {case.name for case in definitions},
            {"User registration", "Checkout flow", "Local registration demo"},
        )

        self.assertEqual(
            {record.test_case_name for record in records if record.run_id in first.run_ids},
            {"User registration", "Checkout flow"},
        )
        self.assertEqual(len([record for record in records if record.test_case_name == "User registration"]), 2)
        registration_records = [
            record for record in records if record.test_case_name == "User registration"
        ]
        self.assertEqual(
            {record.workflow_type for record in registration_records},
            {WorkflowType.VALIDATION, WorkflowType.REGRESSION},
        )
        failed = next(record for record in registration_records if record.status == ExecutionStatus.FAILED)
        self.assertEqual(failed.outcome, "PRODUCT_FAILURE")
        self.assertEqual([step.status for step in failed.steps], [
            ExecutionStatus.PASSED, ExecutionStatus.FAILED, ExecutionStatus.BLOCKED
        ])
        self.assertEqual(failed.setup_status, "SUCCEEDED")
        self.assertTrue(failed.cleanup_succeeded)
        self.assertTrue(failed.executions)
        self.assertTrue(all(item.test_plan_version_id for item in failed.executions))
        self.assertTrue(all(not item.evidence for item in failed.executions))
        successful = next(
            record for record in records
            if record.run_id in first.run_ids and record.workflow_type == WorkflowType.VALIDATION
        )
        self.assertEqual(successful.status, ExecutionStatus.PASSED)
        self.assertEqual(successful.setup_status, "SUCCEEDED")
        self.assertTrue(successful.cleanup_succeeded)

        safe_history_json = "\n".join(record.model_dump_json() for record in records)
        self.assertNotIn("demo-only-sensitive-value-not-a-credential", safe_history_json)
        self.assertIn("[REDACTED]", safe_history_json)

        dashboard = self.application.handle("GET", "/")
        self.assertEqual(dashboard.status, 200)
        self.assertIn(b"User registration", dashboard.body)
        self.assertIn(b"Checkout flow", dashboard.body)
        self.assertIn(b"PRODUCT FAILURE", dashboard.body)

        registration_id = registration_records[0].test_case_id
        testcase_page = self.application.handle("GET", f"/test-cases/{registration_id}")
        self.assertEqual(testcase_page.status, 200)
        self.assertIn(b"Run History", testcase_page.body)
        self.assertIn(b"User registration", testcase_page.body)
        self.assertIn(b"PRODUCT FAILURE", testcase_page.body)

        cases_page = self.application.handle("GET", "/test-cases")
        self.assertIn(b"Local registration demo", cases_page.body)
        local_case = next(case for case in definitions if case.name == "Local registration demo")
        local_detail = self.application.handle("GET", f"/test-cases/{local_case.id}")
        self.assertEqual(local_detail.status, 200)
        self.assertIn(b"No runs yet.", local_detail.body)
        self.assertIn(b"Run Validation", local_detail.body)

        for record in records:
            if record.run_id not in first.run_ids:
                continue
            detail = self.application.handle("GET", f"/runs/{record.run_id}")
            json_report = self.application.handle("GET", f"/runs/{record.run_id}/report.json")
            html_report = self.application.handle("GET", f"/runs/{record.run_id}/report.html")
            self.assertEqual(detail.status, 200)
            self.assertEqual(json_report.status, 200)
            self.assertEqual(html_report.status, 200)
            parsed = json.loads(json_report.body)
            self.assertEqual(parsed["workflow_type"], record.workflow_type.value)
            self.assertIn("test_plan_version_id", json_report.body.decode("utf-8"))
            self.assertNotIn("demo-only-sensitive-value-not-a-credential", json_report.body.decode("utf-8"))
            self.assertNotIn("demo-only-sensitive-value-not-a-credential", html_report.body.decode("utf-8"))
            self.assertNotIn(b"/evidence/", html_report.body)

        failed_report = json.loads(
            self.application.handle("GET", f"/runs/{failed.run_id}/report.json").body
        )
        self.assertEqual(
            [step["status"] for step in failed_report["steps"]],
            ["PASSED", "FAILED", "BLOCKED"],
        )

    def test_seed_refuses_to_overwrite_an_edited_demo_definition(self) -> None:
        seed_demo_data(self.database)
        local_case = next(
            case for case in self.storage.test_case_repository.list()
            if case.name == "Local registration demo"
        )
        edited = local_case.model_copy(update={"description": "User-edited definition."})
        self.storage.test_case_repository.save(edited)

        with self.assertRaisesRegex(ValueError, "different content; preserving it"):
            seed_demo_data(self.database)

        self.assertEqual(
            self.storage.test_case_repository.get(local_case.id).description,
            "User-edited definition.",
        )

    def test_local_registration_fixture_exposes_common_interaction_controls(self) -> None:
        response = self.application.handle("GET", "/demo-target/registration")
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 200)
        for selector in (
            'id="email"', 'id="password"', 'id="terms"', 'type="radio"',
            'id="region"', 'id="form-error"', 'id="details-dialog"',
            'id="cookie-consent"', 'id="accept-cookies"',
        ):
            self.assertIn(selector, body)
        self.assertEqual(
            self.application.handle("GET", "/demo-target/registration/help").status,
            200,
        )


if __name__ == "__main__":
    unittest.main()
