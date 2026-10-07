import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from uuid import UUID
from unittest.mock import patch

from qa_agent.automation_lifecycle import (
    AutomationLifecycleService,
    AutomationStatus,
    SQLiteAutomationLifecycleRepository,
)
from qa_agent.drafts import Draft, SQLiteDraftRepository
from qa_agent.execution_progress import (
    AuthoringEventType,
    AuthoringProgressReporter,
    ExecutionProgressStore,
)
from qa_agent.models import (
    ExecutionSegment,
    PlanVersionOrigin,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_editing import create_manual_test_case, edit_test_case
from qa_agent.test_case_execution import TestCaseExecutionService
from qa_agent.web import LocalWebApplication, should_suppress_successful_progress_log


class ProductAuthoringControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "product.sqlite3"
        self.storage = create_sqlite_storage(self.database)
        self.drafts = SQLiteDraftRepository(self.database)
        self.lifecycle_repository = SQLiteAutomationLifecycleRepository(self.database)
        self.lifecycle = AutomationLifecycleService(self.lifecycle_repository, self.storage.plan_store)
        self.app = self.make_app()

    def tearDown(self):
        self.app.close()
        self.temp_dir.cleanup()

    def make_app(self, **kwargs):
        return LocalWebApplication(
            self.storage.run_history,
            test_cases=self.storage.test_case_repository,
            plan_store=self.storage.plan_store,
            drafts=self.drafts,
            automation_lifecycle=self.lifecycle,
            automation_lifecycle_repository=self.lifecycle_repository,
            **kwargs,
        )

    def post(self, path, fields=None):
        return self.app.handle("POST", path, urlencode(fields or {}))

    def test_draft_create_edit_delete_persists_and_stays_out_of_testcases(self):
        response = self.post("/drafts", {
            "title": "Password reset edge cases",
            "body": "Check expired links and repeated submissions.",
            "base_url": "http://127.0.0.1:8000/reset",
            "notes": "Needs a local user fixture.",
        })
        self.assertEqual(response.status, 303)
        draft_id = UUID(urlsplit(response.headers["Location"]).path.rsplit("/", 1)[1])
        self.assertEqual(self.storage.test_case_repository.list(), [])
        listed = self.app.handle("GET", "/drafts")
        self.assertIn(b"Password reset edge cases", listed.body)
        self.assertIn(b"Recent Drafts", self.app.handle("GET", "/").body)
        self.assertIn(b"Password reset edge cases", self.app.handle("GET", "/test-cases/new").body)

        response = self.post(f"/drafts/{draft_id}/update", {
            "title": "Password reset links",
            "body": "Check expired links and one-time use.",
            "base_url": "http://127.0.0.1:8000/reset",
            "notes": "Updated notes.",
        })
        self.assertEqual(response.status, 303)
        restored = SQLiteDraftRepository(self.database).get(draft_id)
        self.assertEqual(restored.title, "Password reset links")
        self.assertEqual(restored.notes, "Updated notes.")
        self.assertTrue(self.drafts.delete(draft_id))
        self.assertIsNone(self.drafts.get(draft_id))

    def test_draft_manual_conversion_keeps_draft_until_explicit_delete(self):
        draft = Draft(title="Checkout validation", body="Submit a cart with an invalid postal code.")
        self.drafts.save(draft)
        response = self.post(f"/drafts/{draft.id}/convert")
        self.assertEqual(response.status, 303)
        case_id = UUID(urlsplit(response.headers["Location"]).path.split("/")[2])
        converted = self.storage.test_case_repository.get(case_id)
        self.assertEqual(converted.description, draft.body)
        self.assertEqual(converted.steps[0].description, draft.body)
        self.assertIsNotNone(self.drafts.get(draft.id))
        self.assertEqual(self.lifecycle.status(converted), AutomationStatus.NOT_AUTOMATED)

    def test_manual_creation_works_without_an_authoring_service_and_escapes_html(self):
        response = self.post("/test-cases/manual", {
            "name": "<script>Checkout</script>",
            "description": "Scenario <img src=x onerror=alert(1)>",
            "base_url": "http://127.0.0.1:8000/demo-target/registration",
            "preconditions": "A local test account exists.\nThe cart contains one item.",
            "step_name_0": "Submit <unsafe>",
            "step_action_0": "Click the local submit button.",
            "step_expected_0": "A confirmation appears.",
            "step_name_1": "",
            "step_action_1": "",
            "step_expected_1": "",
        })
        self.assertEqual(response.status, 303)
        case_id = UUID(urlsplit(response.headers["Location"]).path.split("/")[2])
        case = self.storage.test_case_repository.get(case_id)
        self.assertEqual(len(case.preconditions), 2)
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.NOT_AUTOMATED)
        page = self.app.handle("GET", f"/test-cases/{case_id}").body.decode()
        self.assertIn("&lt;script&gt;Checkout&lt;/script&gt;", page)
        self.assertIn("&lt;unsafe&gt;", page)
        self.assertNotIn("<script>Checkout</script>", page)
        self.assertIn("Not automated", page)
        self.assertIn("View TestPlan", page)
        plan_page = self.app.handle("GET", f"/test-cases/{case_id}/plans").body.decode()
        self.assertIn("Read-only view", plan_page)
        self.assertIn("No saved TestPlan", plan_page)
        self.assertIsNone(self.app._authoring_service)

    def test_unconfigured_ai_does_not_block_manual_fallback_and_preserves_inputs(self):
        response = self.post("/test-cases/generate", {
            "name": "Local-only sign-in",
            "base_url": "http://127.0.0.1/sign-in",
            "scenario": "Reject a disabled local account.",
        })
        self.assertEqual(response.status, 503)
        page = response.body.decode()
        self.assertIn("Create manually with these details", page)
        prepared = self.app.handle("POST", "/test-cases/manual/prepare", urlencode({
            "name": "Local-only sign-in",
            "base_url": "http://127.0.0.1/sign-in",
            "description": "Reject a disabled local account.",
        }))
        manual_page = prepared.body.decode()
        self.assertIn("Reject a disabled local account.", manual_page)
        self.assertIn("http://127.0.0.1/sign-in", manual_page)
        self.assertIsNone(self.app._authoring_service)

    def test_saved_testcase_add_delete_duplicate_and_reorder_preserve_order(self):
        case = self.case_with_steps(["Alpha", "Beta", "Gamma"])
        self.storage.test_case_repository.save(case)
        editor = self.app.handle("GET", f"/test-cases/{case.id}/edit").body.decode()
        self.assertIn("Move up", editor)
        self.assertIn("Move down", editor)
        self.assertIn("Duplicate", editor)
        self.assertIn("Delete", editor)

        moved_down = self.post(f"/test-cases/{case.id}/edit", self.edit_values(case, "down:0:0"))
        self.assertEqual(moved_down.status, 303)
        case = self.storage.test_case_repository.get(case.id)
        self.assertEqual([step.name for step in case.steps], ["Beta", "Alpha", "Gamma"])

        moved_up = self.post(f"/test-cases/{case.id}/edit", self.edit_values(case, "up:0:1"))
        self.assertEqual(moved_up.status, 303)
        case = self.storage.test_case_repository.get(case.id)
        self.assertEqual([step.name for step in case.steps], ["Alpha", "Beta", "Gamma"])

        edited_values = self.edit_values(case, "save")
        edited_values["name"] = "Locally edited case"
        edited_values["description"] = "A manually updated scenario."
        edited_values["preconditions"] = "The test user is enabled."
        self.assertEqual(self.post(f"/test-cases/{case.id}/edit", edited_values).status, 303)
        case = self.storage.test_case_repository.get(case.id)
        self.assertEqual(case.name, "Locally edited case")
        self.assertEqual(case.description, "A manually updated scenario.")
        self.assertEqual(case.preconditions[0].description, "The test user is enabled.")

        duplicated = self.post(f"/test-cases/{case.id}/edit", self.edit_values(case, "duplicate:0:0"))
        self.assertEqual(duplicated.status, 303)
        case = self.storage.test_case_repository.get(case.id)
        self.assertEqual([step.name for step in case.steps][:2], ["Alpha", "Alpha (copy)"])
        self.assertNotEqual(case.steps[0].id, case.steps[1].id)

        added = self.post(f"/test-cases/{case.id}/edit", self.edit_values(case, "add:0"))
        self.assertEqual(added.status, 303)
        case = self.storage.test_case_repository.get(case.id)
        self.assertEqual(len(case.steps), 5)

        deleted = self.post(f"/test-cases/{case.id}/edit", self.edit_values(case, "delete:0:4"))
        self.assertEqual(deleted.status, 303)
        restored = self.storage.test_case_repository.get(case.id)
        self.assertEqual(len(restored.steps), 4)
        self.assertEqual([step.order for step in restored.steps], list(range(4)))
        reconstructed = create_sqlite_storage(self.database).test_case_repository.get(case.id)
        self.assertEqual([step.name for step in reconstructed.steps], [step.name for step in restored.steps])

    def test_step_reordering_stays_within_its_segment(self):
        case = DomainTestCase(
            name="Segment-safe editor", description="Keep navigation groups intact.",
            segments=[
                ExecutionSegment(order=0, base_url="http://127.0.0.1/one", steps=[self.step("One", 0)]),
                ExecutionSegment(order=1, base_url="http://127.0.0.1/two", steps=[self.step("Two", 1), self.step("Three", 2)]),
            ],
        )
        self.storage.test_case_repository.save(case)
        self.post(f"/test-cases/{case.id}/edit", self.edit_values(case, "add:1"))
        updated = self.storage.test_case_repository.get(case.id)
        self.assertEqual(len(updated.segments), 2)
        self.assertEqual(updated.segments[0].base_url, "http://127.0.0.1/one")
        self.assertEqual(updated.segments[1].base_url, "http://127.0.0.1/two")
        self.assertEqual([step.order for step in updated.steps], [0, 1, 2, 3])

    def test_ai_failure_offers_manual_and_save_draft_paths_with_scenario(self):
        progress = ExecutionProgressStore()
        progress_id = progress.create_authoring(
            name="Unavailable provider scenario", base_url="http://127.0.0.1/local",
            scenario="Verify the sign-in lockout after repeated failures.",
        )
        reporter = AuthoringProgressReporter(progress, progress_id)
        reporter.emit(AuthoringEventType.AUTHORING_REQUESTED)
        reporter.emit(AuthoringEventType.INPUT_VALIDATED)
        reporter.emit(AuthoringEventType.AUTHORING_STARTED)
        reporter.emit(AuthoringEventType.LLM_REQUEST_STARTED)
        reporter.finish_failure("AI_PROVIDER_ERROR", "The AI provider could not complete the request.")
        app = self.make_app(progress_store=progress)
        page = app.handle("GET", f"/test-cases/authoring-progress/{progress_id}").body.decode()
        self.assertIn("Try Again", page)
        self.assertIn("Create manually", page)
        self.assertIn("Save as Draft", page)

        saved = app.handle("POST", f"/test-cases/authoring-progress/{progress_id}/save-draft", "")
        draft_id = UUID(urlsplit(saved.headers["Location"]).path.rsplit("/", 1)[1])
        self.assertEqual(self.drafts.get(draft_id).body, "Verify the sign-in lockout after repeated failures.")
        manual = app.handle("POST", f"/test-cases/authoring-progress/{progress_id}/create-manually", "")
        case_id = UUID(urlsplit(manual.headers["Location"]).path.split("/")[2])
        self.assertEqual(
            self.storage.test_case_repository.get(case_id).description,
            "Verify the sign-in lockout after repeated failures.",
        )
        app.close()

    def test_automation_lifecycle_validation_gate_and_history_preservation(self):
        case = self.case_with_steps(["Assert local page", "Assert form"])
        self.storage.test_case_repository.save(case)
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.NOT_AUTOMATED)
        versions = self.save_plans(case)
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.NEEDS_VALIDATION)
        self.assertEqual(self.lifecycle.mark_validation_completed(case, passed=False), AutomationStatus.NEEDS_VALIDATION)
        self.assertEqual(self.lifecycle.mark_validation_completed(case, passed=True), AutomationStatus.AUTOMATION_READY)
        restarted_lifecycle = AutomationLifecycleService(
            SQLiteAutomationLifecycleRepository(self.database), self.storage.plan_store
        )
        self.assertEqual(restarted_lifecycle.status(case), AutomationStatus.AUTOMATION_READY)

        # A manual plan change requires validation again.
        step = case.steps[0]
        plan = self.storage.plan_store.find_test_plan(step.id)
        self.storage.plan_store.save(step.id, DomainTestPlanVersion(
            test_plan_id=plan.id, version=2, origin=PlanVersionOrigin.HUMAN_EDITED,
            qa_test_plan=QATestPlan(url="http://127.0.0.1/local", steps=[QATestStep(action="assert_page_loaded")]),
        ), test_plan=plan)
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.NEEDS_VALIDATION)
        self.assertEqual(self.lifecycle.mark_validation_completed(case, passed=True), AutomationStatus.AUTOMATION_READY)

        updated = edit_test_case(case, {"description": "The local sign-in page loads and remains usable."})
        self.storage.test_case_repository.save(updated)
        self.assertEqual(self.lifecycle.status(updated), AutomationStatus.NEEDS_UPDATE)
        self.assertEqual(self.lifecycle.mark_validation_completed(updated, passed=False), AutomationStatus.NEEDS_VALIDATION)
        self.assertEqual(self.lifecycle.mark_validation_completed(updated, passed=True), AutomationStatus.AUTOMATION_READY)
        self.assertIsNotNone(self.storage.plan_store.get_version(versions[0]))
        self.assertIsNotNone(self.storage.plan_store.get_version(self.storage.plan_store.find(step.id).id))
        removed_step_version = self.storage.plan_store.find(updated.steps[1].id).id
        removed = edit_test_case(updated, self.edit_values(updated, "delete:0:1"), "delete:0:1")
        self.storage.test_case_repository.save(removed)
        self.assertEqual(self.lifecycle.status(removed), AutomationStatus.NEEDS_UPDATE)
        self.assertIsNotNone(self.storage.plan_store.get_version(removed_step_version))

    def test_product_failure_does_not_invalidate_automation_ready(self):
        case = self.case_with_steps(["Find existing account"])
        self.storage.test_case_repository.save(case)
        self.save_plans(case)
        self.lifecycle.mark_automation_completed(case)
        self.lifecycle.mark_validation_completed(case, passed=True)
        service = TestCaseExecutionService(
            self.storage.test_case_repository, self.storage.plan_store,
            self.storage.execution_repository, self.storage.run_history,
            runner_factory=lambda _directory: lambda plan: {
                "status": "failed", "url": plan.url,
                "steps": [{"action": "assert_title", "status": "failed", "error": "Expected title was absent."}],
                "evidence": [],
            },
            automation_lifecycle=self.lifecycle,
        )
        result = service.run(case.id, WorkflowType.REGRESSION)
        self.assertEqual(result.outcome.value, "PRODUCT_FAILURE")
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.AUTOMATION_READY)

    def test_validation_run_pass_is_the_gate_to_automation_ready(self):
        case = self.case_with_steps(["Assert local page"])
        self.storage.test_case_repository.save(case)
        self.save_plans(case)
        service = TestCaseExecutionService(
            self.storage.test_case_repository, self.storage.plan_store,
            self.storage.execution_repository, self.storage.run_history,
            runner_factory=lambda _directory: lambda plan: {
                "status": "passed", "url": plan.url, "steps": [], "evidence": [],
            },
            automation_lifecycle=self.lifecycle,
        )
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.NEEDS_VALIDATION)
        result = service.run(case.id, WorkflowType.VALIDATION)
        self.assertEqual(result.outcome.value, "PASSED")
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.AUTOMATION_READY)
        edited = edit_test_case(case, {"description": "The local page remains correct after its scenario changes."})
        self.storage.test_case_repository.save(edited)
        self.assertEqual(self.lifecycle.status(edited), AutomationStatus.NEEDS_UPDATE)
        result = service.run(edited.id, WorkflowType.VALIDATION)
        self.assertEqual(result.outcome.value, "PASSED")
        self.assertEqual(self.lifecycle.status(edited), AutomationStatus.AUTOMATION_READY)

    def test_validation_does_not_mark_a_concurrently_changed_plan_ready(self):
        case = self.case_with_steps(["Assert local page"])
        self.storage.test_case_repository.save(case)
        self.save_plans(case)
        step = case.steps[0]
        plan = self.storage.plan_store.find_test_plan(step.id)
        def runner(_plan):
            self.storage.plan_store.save(step.id, DomainTestPlanVersion(
                test_plan_id=plan.id, version=2, origin=PlanVersionOrigin.HUMAN_EDITED,
                qa_test_plan=QATestPlan(url=case.base_url, steps=[QATestStep(action="assert_page_loaded")]),
            ), test_plan=plan)
            return {"status": "passed", "url": case.base_url, "steps": [], "evidence": []}

        service = TestCaseExecutionService(
            self.storage.test_case_repository, self.storage.plan_store,
            self.storage.execution_repository, self.storage.run_history,
            runner_factory=lambda _directory: runner,
            automation_lifecycle=self.lifecycle,
        )
        result = service.run(case.id, WorkflowType.VALIDATION)
        self.assertEqual(result.outcome.value, "PASSED")
        self.assertEqual(self.lifecycle.status(case), AutomationStatus.NEEDS_VALIDATION)

    def test_health_is_minimal_and_progress_success_logs_are_quiet_only(self):
        secret = "DO_NOT_EXPOSE_THIS_TEST_SECRET"
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            response = self.app.handle("GET", "/health")
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body), {"status": "ok", "service": "ai-qa-agent"})
        self.assertNotIn(secret, response.body.decode())
        self.assertTrue(should_suppress_successful_progress_log("GET", "/api/progress/abc", 200))
        self.assertTrue(should_suppress_successful_progress_log("GET", "/api/test-cases/authoring-progress/abc", 200))
        self.assertFalse(should_suppress_successful_progress_log("GET", "/api/test-cases/authoring-progress/abc", 404))
        self.assertFalse(should_suppress_successful_progress_log("GET", "/api/progress/abc", 500))
        self.assertFalse(should_suppress_successful_progress_log("GET", "/test-cases/abc", 200))

    def test_windows_launcher_scripts_have_safe_local_ownership_rules(self):
        start = (Path(__file__).parents[1] / "scripts" / "start-ai-qa-agent.ps1").read_text()
        stop = (Path(__file__).parents[1] / "scripts" / "stop-ai-qa-agent.ps1").read_text()
        shortcuts = (Path(__file__).parents[1] / "scripts" / "install-windows-shortcuts.ps1").read_text()
        self.assertLess(start.index(".venv\\Scripts\\python.exe"), start.index(".venv-original\\Scripts\\python.exe"))
        self.assertIn("/health", start)
        self.assertIn("Port 8000 is in use by another process", start)
        self.assertIn("--launcher-instance", start)
        self.assertIn("System.Threading.Mutex", start)
        self.assertIn("launcher-owned server is still starting", start)
        self.assertIn("did not become healthy", start)
        self.assertLess(start.index("$health.service -eq 'ai-qa-agent'"), start.index("Start-Process -FilePath $python"))
        self.assertIn("Get-CimInstance Win32_Process", stop)
        self.assertIn("$actualPython.Equals($expectedPython", stop)
        self.assertIn("--launcher-instance", stop)
        self.assertLess(stop.index("if (-not $owned)"), stop.index("Stop-Process -Id $processId"))
        self.assertIn("stale AI QA Agent PID file", stop)
        self.assertIn("malformed stale PID file was removed", stop)
        self.assertNotIn("Get-NetTCPConnection", stop)
        self.assertIn("WScript.Shell", shortcuts)
        self.assertIn("AI QA Agent", shortcuts)
        self.assertIn("Stop AI QA Agent", shortcuts)

    def case_with_steps(self, names):
        return DomainTestCase(
            name="Local product case", description="Verify a behavior on a local page.",
            base_url="http://127.0.0.1/local",
            segments=[ExecutionSegment(order=0, is_implicit=True, base_url="http://127.0.0.1/local", steps=[
                self.step(name, index) for index, name in enumerate(names)
            ])],
        )

    @staticmethod
    def step(name, order):
        return DomainTestStep(name=name, description=f"Perform {name.lower()}.", expected="The expected result is shown.", order=order)

    def save_plans(self, case):
        version_ids = []
        for step in case.steps:
            plan = DomainTestPlan(test_step_id=step.id, name=step.name)
            version = DomainTestPlanVersion(
                test_plan_id=plan.id, version=1, origin=PlanVersionOrigin.AI_GENERATED,
                qa_test_plan=QATestPlan(url=case.base_url, steps=[QATestStep(action="assert_title", parameters={"expected": "Local page"})]),
            )
            self.storage.plan_store.save(step.id, version, test_plan=plan)
            version_ids.append(version.id)
        return version_ids

    @staticmethod
    def edit_values(case, operation):
        fields = {
            "name": case.name,
            "description": case.description,
            "base_url": case.base_url or "",
            "preconditions": "\n".join(item.description for item in case.preconditions),
            "operation": operation,
        }
        for segment_index, segment in enumerate(case.segments):
            for step_index, step in enumerate(segment.steps):
                prefix = f"step_{segment_index}_{step_index}_"
                fields[prefix + "name"] = step.name
                fields[prefix + "action"] = step.description
                fields[prefix + "expected"] = step.expected
        return fields


if __name__ == "__main__":
    unittest.main()
