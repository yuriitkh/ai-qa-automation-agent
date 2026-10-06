import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from qa_agent.demo import seed_demo_data
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
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
                run_location = response.getheader("Location")
                response.read()
                self.assertIsNotNone(run_location)

                record = storage.run_history.list_for_test_case(test_case.id)[0]
                self.assertEqual(record.workflow_type, WorkflowType.VALIDATION)
                self.assertEqual(record.status.value, "FAILED")
                self.assertEqual(record.outcome, "PRODUCT_FAILURE")
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
                self.assertRegex(evidence_path.name, r"^execution-[0-9a-f]{32}\.png$")

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


if __name__ == "__main__":
    unittest.main()
