import json
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from threading import Event

from playwright.sync_api import expect, sync_playwright

from qa_agent.demo import seed_demo_data
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.models import TestCase as DomainTestCase, TestStep as DomainTestStep
from qa_agent.execution_progress import get_active_execution_progress
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_execution import TestCaseExecutionService
from qa_agent.web import LocalWebApplication, create_http_server


class PersistedTestCaseBrowserFlowTests(unittest.TestCase):
    def test_ui_runs_saved_plans_with_local_playwright_and_persists_safe_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "ui-run.sqlite3"
            evidence_root = root / "evidence"
            storage = create_sqlite_storage(database)
            run_service = TestCaseExecutionService(
                storage.test_case_repository,
                storage.plan_store,
                storage.execution_repository,
                storage.run_history,
                evidence_directory=evidence_root,
            )
            application = LocalWebApplication(
                storage.run_history,
                evidence_root=evidence_root,
                test_cases=storage.test_case_repository,
                run_service=run_service,
            )
            server = create_http_server(application, host="127.0.0.1", port=0)
            port = server.server_address[1]
            base_url = f"http://127.0.0.1:{port}/demo-target/registration"
            seed_demo_data(database, demo_base_url=base_url)
            test_case = next(
                case for case in storage.test_case_repository.list()
                if case.name == "Local registration demo"
            )
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            connection = HTTPConnection("127.0.0.1", port, timeout=90)
            try:
                connection.request(
                    "POST",
                    f"/test-cases/{test_case.id}/run",
                    body="workflow=VALIDATION",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 303)
                progress_location = response.getheader("Location")
                response.read()
                self.assertIsNotNone(progress_location)
                progress_id = progress_location.rsplit("/", 1)[1]
                deadline = time.monotonic() + 80
                progress = None
                while time.monotonic() < deadline:
                    connection.request("GET", f"/api/progress/{progress_id}")
                    progress_response = connection.getresponse()
                    progress_body = progress_response.read().decode("utf-8")
                    self.assertEqual(progress_response.status, 200)
                    progress = json.loads(progress_body)
                    if progress["finished"]:
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(progress)
                self.assertTrue(progress["finished"], "background run did not finish in time")
                event_types = [event["type"] for event in progress["events"]]
                self.assertIn("TESTCASE_LOADED", event_types)
                self.assertIn("SETUP_SUCCEEDED", event_types)
                self.assertIn("PLAN_REUSED", event_types)
                self.assertIn("STEP_STARTED", event_types)
                self.assertIn("STEP_FAILED", event_types)
                self.assertIn("STEP_BLOCKED", event_types)
                self.assertIn("EVIDENCE_CAPTURED", event_types)
                self.assertIn("CLEANUP_SUCCEEDED", event_types)
                self.assertEqual(event_types[-1], "RUN_FINISHED")
                self.assertEqual(progress["outcome"], "AUTOMATION_EXECUTION_ERROR")
                self.assertNotIn(str(evidence_root), progress_body)
                run_location = progress["final_run_url"]

                record = storage.run_history.list_for_test_case(test_case.id)[0]
                self.assertEqual(progress["elapsed_ms"], record.duration_ms)
                self.assertEqual(record.workflow_type, WorkflowType.VALIDATION)
                self.assertEqual(record.status.value, "FAILED")
                self.assertEqual(record.outcome, "AUTOMATION_EXECUTION_ERROR")
                self.assertEqual(
                    [step.status.value for step in record.steps],
                    ["PASSED", "FAILED", "BLOCKED"],
                )

                execution_reference = next(
                    item for item in record.executions if item.evidence
                )
                detail = storage.run_history.get_detail(record.run_id)
                execution = detail.executions[execution_reference.execution_id]
                evidence_path = Path(execution.evidence[0].path).resolve()
                self.assertTrue(evidence_path.is_relative_to(evidence_root.resolve()))
                self.assertTrue(evidence_path.is_file())
                self.assertRegex(
                    evidence_path.name,
                    r"^execution-[0-9a-f]{32}-step-[0-9a-f]{32}-event-\d+-page\.png$",
                )

                connection.request("GET", run_location)
                detail_response = connection.getresponse()
                detail_body = detail_response.read().decode("utf-8")
                self.assertEqual(detail_response.status, 200)
                self.assertIn("FAILED", detail_body)
                self.assertIn("BLOCKED", detail_body)
                self.assertIn(f"/runs/{record.run_id}/evidence/", detail_body)
                self.assertNotIn(str(evidence_root), detail_body)

                connection.request("GET", f"/runs/{record.run_id}/report.json")
                json_response = connection.getresponse()
                report_body = json_response.read().decode("utf-8")
                self.assertEqual(json_response.status, 200)
                report = json.loads(report_body)
                self.assertEqual(
                    [step["status"] for step in report["steps"]],
                    ["PASSED", "FAILED", "BLOCKED"],
                )
                self.assertNotIn(str(evidence_root), report_body)
                safe_history = storage.run_history.get(record.run_id).model_dump_json()
                self.assertNotIn(str(root), safe_history)

                connection.request("GET", "/")
                dashboard_response = connection.getresponse()
                dashboard_body = dashboard_response.read().decode("utf-8")
                self.assertEqual(dashboard_response.status, 200)
                self.assertIn("Local registration demo", dashboard_body)
                self.assertIn(str(record.run_id), dashboard_body)

                connection.request("GET", f"/test-cases/{test_case.id}")
                case_response = connection.getresponse()
                case_body = case_response.read().decode("utf-8")
                self.assertEqual(case_response.status, 200)
                self.assertIn("Run History", case_body)
                self.assertIn("FAILED", case_body)

                connection.request("GET", f"/runs/{record.run_id}/report.html")
                html_report_response = connection.getresponse()
                html_report_body = html_report_response.read().decode("utf-8")
                self.assertEqual(html_report_response.status, 200)
                self.assertIn("/evidence/", html_report_body)
                self.assertNotIn(str(root), html_report_body)

                evidence_index = next(
                    index for index, reference in enumerate(execution_reference.evidence)
                    if reference.id == execution.evidence[0].id
                )
                connection.request(
                    "GET",
                    f"/runs/{record.run_id}/evidence/{execution.id}/{evidence_index}",
                )
                image_response = connection.getresponse()
                image_bytes = image_response.read()
                self.assertEqual(image_response.status, 200)
                self.assertEqual(image_response.getheader("Content-Type"), "image/png")
                self.assertTrue(image_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)


class ProgressLifecycleBrowserTests(unittest.TestCase):
    def test_finished_phase_and_execution_state_update_together_in_browser(self) -> None:
        case = DomainTestCase(
            name="Progress lifecycle",
            description="Verify the live progress terminal state.",
            steps=[DomainTestStep(
                name="Inspect page",
                description="Inspect the local page.",
                expected="The page is available.",
                order=0,
            )],
        )
        started = Event()
        release = Event()

        class BlockingRunService:
            def run(self, _test_case_id, _workflow):
                progress = get_active_execution_progress()
                progress.test_case_loaded(case)
                started.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("Test release was not signaled.")
                return type("Result", (), {"test_run": None, "outcome": "PASSED"})()

        history = RunHistoryService(InMemoryRunHistoryRepository())
        repository = InMemoryTestCaseRepository()
        repository.save(case)
        application = LocalWebApplication(
            history,
            test_cases=repository,
            run_service=BlockingRunService(),
        )
        server = create_http_server(application, host="127.0.0.1", port=0)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            response = application.handle(
                "POST",
                f"/test-cases/{case.id}/run",
                "workflow=AUTOMATION",
            )
            self.assertEqual(response.status, 303)
            progress_id = response.headers["Location"].rsplit("/", 1)[1]
            self.assertTrue(started.wait(timeout=5))

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{port}/runs/progress/{progress_id}")
                expect(page.locator("[data-progress-state]")).to_have_text(
                    "Running", timeout=5000
                )
                release.set()
                expect(page.locator("[data-progress-phase]").first).to_have_text(
                    "Finished", timeout=8000
                )
                expect(page.locator("[data-progress-state]")).to_have_text(
                    "Finished", timeout=8000
                )
                browser.close()

            snapshot = json.loads(
                application.handle("GET", f"/api/progress/{progress_id}").body
            )
            self.assertEqual(snapshot["state"], "FINISHED")
            self.assertEqual(snapshot["phase"], "Finished")
            self.assertTrue(snapshot["finished"])
            self.assertEqual(snapshot["outcome"], "PASSED")
            self.assertIsNone(snapshot["run_status"])
            self.assertNotIn("RUNNING", [step["state"] for step in snapshot["steps"]])
            self.assertEqual(snapshot["events"][-1]["type"], "RUN_FINISHED")
        finally:
            release.set()
            application.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
