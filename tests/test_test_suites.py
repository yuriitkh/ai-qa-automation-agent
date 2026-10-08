import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID, uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.reporting import RunReportGenerator
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.test_suites import (
    InMemoryTestSuiteRepository,
    SQLiteTestSuiteRepository,
    TestSuiteService as SuiteService,
)
from qa_agent.web import LocalWebApplication


def _case(name: str) -> DomainTestCase:
    return DomainTestCase(
        name=name,
        description="Local saved definition.",
        base_url="http://127.0.0.1:8000/demo-target/registration",
        steps=[DomainTestStep(name="Open", description="Open local page.", expected="It loads.", order=0)],
    )


class TestSuiteRepositoryTests(unittest.TestCase):
    def test_sqlite_suite_and_order_survive_repository_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "suites.sqlite3"
            first, second = uuid4(), uuid4()
            repository = SQLiteTestSuiteRepository(database)
            suite = repository.create("Smoke", "Local smoke cases.")
            repository.add_member(suite.id, first)
            repository.add_member(suite.id, second)
            repository.move_member(suite.id, second, -1)
            updated = repository.update(suite.id, "Smoke checks", "Updated locally.")

            reopened = SQLiteTestSuiteRepository(database)
            self.assertEqual(reopened.get(suite.id).name, "Smoke checks")
            self.assertEqual(reopened.get(suite.id).description, "Updated locally.")
            self.assertEqual(reopened.members(suite.id), [second, first])
            self.assertEqual(reopened.list()[0].id, suite.id)
            self.assertGreaterEqual(updated.updated_at, suite.created_at)

    def test_migration_is_idempotent_and_remove_compacts_order(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "suites.sqlite3"
            repository = SQLiteTestSuiteRepository(database)
            suite = repository.create("Regression", "")
            ids = [uuid4() for _ in range(3)]
            for case_id in ids:
                repository.add_member(suite.id, case_id)
            repository.remove_member(suite.id, ids[1])
            reopened = SQLiteTestSuiteRepository(database)
            self.assertEqual(reopened.members(suite.id), [ids[0], ids[2]])
            reopened.add_member(suite.id, uuid4())
            self.assertEqual(len(reopened.members(suite.id)), 3)

    def test_testcase_can_belong_to_multiple_suites_and_duplicate_membership_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = InMemoryTestCaseRepository()
            case = _case("Registration")
            cases.save(case)
            repository = SQLiteTestSuiteRepository(Path(directory) / "suites.sqlite3")
            service = SuiteService(repository, cases)
            smoke = service.create("Smoke", "")
            authentication = service.create("Authentication", "")
            service.add_member(smoke.id, case.id)
            service.add_member(smoke.id, case.id)
            service.add_member(authentication.id, case.id)

            self.assertEqual(service.member_ids(smoke.id), [case.id])
            self.assertEqual(service.member_ids(authentication.id), [case.id])

    def test_invalid_or_stale_testcase_membership_is_handled_safely(self):
        cases = InMemoryTestCaseRepository()
        valid = _case("Existing")
        cases.save(valid)
        repository = InMemoryTestSuiteRepository()
        service = SuiteService(repository, cases)
        suite = service.create("Smoke", "")
        with self.assertRaisesRegex(KeyError, "TestCase not found"):
            service.add_member(suite.id, uuid4())
        repository.add_member(suite.id, uuid4())  # stale legacy/reference row
        repository.add_member(suite.id, valid.id)
        self.assertEqual(service.member_ids(suite.id), [valid.id])
        self.assertEqual([item.id for item in service.members(suite.id)], [valid.id])

    def test_reorder_uses_explicit_member_order(self):
        repository = InMemoryTestSuiteRepository()
        suite = repository.create("Order", "")
        ids = [uuid4() for _ in range(3)]
        for case_id in ids:
            repository.add_member(suite.id, case_id)
        repository.move_member(suite.id, ids[2], -1)
        self.assertEqual(repository.members(suite.id), [ids[0], ids[2], ids[1]])
        with self.assertRaisesRegex(KeyError, "not in this Test Suite"):
            repository.move_member(suite.id, uuid4(), 1)

    def test_suite_search_page_is_bounded_and_uses_name_filter(self):
        repository = InMemoryTestSuiteRepository()
        for name in ("Checkout smoke", "Checkout regression", "Settings"):
            repository.create(name, "")
        page, total = repository.list_page(search="checkout", offset=1, limit=1)

        self.assertEqual(total, 2)
        self.assertEqual(len(page), 1)
        self.assertEqual(page[0].name, "Checkout smoke")


class TestSuiteWebTests(unittest.TestCase):
    def setUp(self):
        self.cases = InMemoryTestCaseRepository()
        self.case = _case("Search <local>")
        self.cases.save(self.case)
        self.plans = InMemoryPlanStore()
        plan = DomainTestPlan(test_step_id=self.case.steps[0].id, name="Open")
        self.plans.save(
            self.case.steps[0].id,
            DomainTestPlanVersion(
                test_plan_id=plan.id,
                version=1,
                qa_test_plan=QATestPlan(
                    url="http://127.0.0.1:8000/demo-target/registration",
                    steps=[QATestStep(action="navigate", parameters={"url": "http://127.0.0.1:8000/demo-target/registration"})],
                ),
            ),
            test_plan=plan,
        )
        self.suite_repo = InMemoryTestSuiteRepository()
        self.suites = SuiteService(self.suite_repo, self.cases)
        history = RunHistoryService(InMemoryRunHistoryRepository(), InMemoryExecutionRepository())
        self.app = LocalWebApplication(
            history,
            RunReportGenerator(),
            test_cases=self.cases,
            plan_store=self.plans,
            test_suites=self.suites,
        )

    def test_suite_list_and_central_testcase_export_shortcuts_render(self):
        list_html = self.app.handle("GET", "/test-cases").body.decode("utf-8")
        self.assertNotIn('name="test_case_id"', list_html)
        self.assertNotIn("Select visible", list_html)
        self.assertNotIn("Export selected", list_html)
        self.assertIn("Needs validation", list_html)
        self.assertIn("Latest status:", list_html)
        self.assertIn(f'href="/test-cases/{self.case.id}/edit"', list_html)
        self.assertIn(f'href="/export?mode=testcases&amp;case={self.case.public_id}"', list_html)
        self.assertIn(f'href="/test-cases/{self.case.id}"', list_html)
        self.assertNotIn('class="export-selection-controls"', list_html)
        self.assertIn(".field{display:grid;gap:.42rem}", list_html)
        self.assertIn(".test-case-list{min-width:700px", list_html)
        case_html = self.app.handle("GET", f"/test-cases/{self.case.id}").body.decode("utf-8")
        self.assertIn("Export &rarr;", case_html)
        self.assertIn(f"/export?mode=testcases&amp;case={self.case.public_id}", case_html)
        self.assertNotIn("Python Playwright</a>", case_html)
        suites_html = self.app.handle("GET", "/test-suites").body.decode("utf-8")
        self.assertIn("Create Test Suite", suites_html)
        self.assertIn("/export?mode=testcases&amp;case=", list_html)

    def test_suite_list_shows_description_counts_readiness_and_open_action(self):
        suite = self.suites.create("Smoke <suite>", "Short <description> for local checks.")
        self.suites.add_member(suite.id, self.case.id)
        html = self.app.handle("GET", "/test-suites").body.decode("utf-8")

        self.assertIn(f'href="/test-suites/{suite.id}"', html)
        self.assertIn("Smoke &lt;suite&gt;", html)
        self.assertIn("Short &lt;description&gt; for local checks.", html)
        self.assertIn("1 TestCase", html)
        self.assertIn("0 of 1 Automation ready", html)
        self.assertIn(f'>Open</a>', html)
        self.assertIn(f'href="/export?mode=suites&amp;suite={suite.id}"', html)
        self.assertIn('class="button primary" type="submit">Create suite</button>', html)

    def test_suite_members_render_as_compact_accessible_rows_with_lifecycle(self):
        suite = self.suites.create("Compact rows", "")
        self.suites.add_member(suite.id, self.case.id)
        self.cases.save(_case("Other available case"))
        html = self.app.handle("GET", f"/test-suites/{suite.id}").body.decode("utf-8")

        self.assertIn('class="suite-members"', html)
        self.assertIn('class="suite-member-main"', html)
        self.assertIn('class="suite-member-actions"', html)
        self.assertIn("Export suite &rarr;", html)
        self.assertNotIn("Python ZIP", html)
        self.assertIn('class="suite-form"', html)
        self.assertIn("Needs validation", html)
        self.assertIn("1.", html)
        self.assertRegex(html, r'aria-label="Move TC-[0-9]+ up"')
        self.assertRegex(html, r'aria-label="Move TC-[0-9]+ down"')
        self.assertRegex(html, r'aria-label="Remove TC-[0-9]+"')
        self.assertIn('class="button danger-button"', html)
        self.assertIn('class="button primary" type="submit">Save changes</button>', html)
        self.assertIn(".suite-member-actions{display:flex", html)
        self.assertIn('class="inline-form"', html)
        self.assertIn("@media(max-width:640px){.suite-member", html)

    def test_bulk_selection_returns_zip_and_rejects_invalid_ids(self):
        body = urlencode([("test_case_id", str(self.case.id)), ("target", "python")])
        response = self.app.handle("POST", "/test-cases/export", body)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
            self.assertIn("export-manifest.json", archive.namelist())
            self.assertTrue(any(item.endswith(".py") for item in archive.namelist()))
        bad = urlencode([("test_case_id", str(uuid4())), ("target", "python")])
        invalid = self.app.handle("POST", "/test-cases/export", bad)
        self.assertEqual(invalid.status, 409)
        self.assertIn(b"TestCase not found", invalid.body)

    def test_bulk_form_accepts_repeated_case_ids_and_preserves_security(self):
        second = _case("Second")
        self.cases.save(second)
        plan = DomainTestPlan(test_step_id=second.steps[0].id, name="Open")
        self.plans.save(second.steps[0].id, DomainTestPlanVersion(
            test_plan_id=plan.id, version=1,
            qa_test_plan=QATestPlan(url="http://127.0.0.1:8000/local", steps=[QATestStep(action="assert_page_loaded")]),
        ), test_plan=plan)
        body = urlencode([
            ("test_case_id", str(self.case.id)),
            ("test_case_id", str(second.id)),
            ("target", "portable"),
        ])
        response = self.app.handle("POST", "/test-cases/export", body)
        self.assertEqual(response.status, 200)
        with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
            names = archive.namelist()
            self.assertEqual(len([name for name in names if name.endswith(".testplan.json")]), 2)
            self.assertTrue(all(".." not in name.split("/") for name in names))

    def test_create_edit_add_remove_order_and_escaped_suite_names(self):
        created = self.app.handle("POST", "/test-suites", urlencode({"name": "Smoke <script>", "description": "<tag>"}))
        self.assertEqual(created.status, 303)
        suite_id = created.headers["Location"].rsplit("/", 1)[1]
        self.assertIn(suite_id, created.headers["Location"])
        page = self.app.handle("GET", created.headers["Location"]).body.decode("utf-8")
        self.assertIn("Smoke &lt;script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;tag&gt;", page)

        added = self.app.handle("POST", f"/test-suites/{suite_id}/members/add", urlencode({"test_case_id": str(self.case.id)}))
        self.assertEqual(added.status, 303)
        self.assertEqual(self.suites.member_ids(UUID(suite_id)), [self.case.id])
        second = _case("Second member")
        self.cases.save(second)
        self.assertEqual(
            self.app.handle("POST", f"/test-suites/{suite_id}/members/add", urlencode({"test_case_id": str(second.id)})).status,
            303,
        )
        move_down = self.app.handle(
            "POST", f"/test-suites/{suite_id}/members/move-down",
            urlencode({"test_case_id": str(self.case.id)}),
        )
        self.assertEqual(move_down.status, 303)
        self.assertEqual(self.suites.member_ids(UUID(suite_id)), [second.id, self.case.id])
        move_up = self.app.handle(
            "POST", f"/test-suites/{suite_id}/members/move-up",
            urlencode({"test_case_id": str(self.case.id)}),
        )
        self.assertEqual(move_up.status, 303)
        self.assertEqual(self.suites.member_ids(UUID(suite_id)), [self.case.id, second.id])
        update = self.app.handle("POST", f"/test-suites/{suite_id}/update", urlencode({"name": "Updated", "description": "Safe"}))
        self.assertEqual(update.status, 303)
        self.assertEqual(self.suites.get(UUID(suite_id)).name, "Updated")
        remove_second = self.app.handle(
            "POST", f"/test-suites/{suite_id}/members/remove",
            urlencode({"test_case_id": str(second.id)}),
        )
        self.assertEqual(remove_second.status, 303)
        self.assertEqual(self.suites.member_ids(UUID(suite_id)), [self.case.id])
        remove = self.app.handle("POST", f"/test-suites/{suite_id}/members/remove", urlencode({"test_case_id": str(self.case.id)}))
        self.assertEqual(remove.status, 303)
        self.assertEqual(self.suites.member_ids(UUID(suite_id)), [])

    def test_suite_export_uses_member_order_and_zip_target(self):
        suite = self.suites.create("Smoke", "Local")
        self.suites.add_member(suite.id, self.case.id)
        response = self.app.handle("GET", f"/test-suites/{suite.id}/export?format=typescript")
        self.assertEqual(response.status, 200)
        with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
            manifest = archive.read("export-manifest.json").decode("utf-8")
            self.assertIn('"language": "typescript"', manifest)
            self.assertIn('"suite": "Smoke"', manifest)
            self.assertTrue(any(name.endswith(".spec.ts") for name in archive.namelist()))

    def test_missing_automation_has_actionable_suite_export_response(self):
        empty = _case("No plan <script>alert</script>")
        empty = empty.model_copy(update={
            "segments": [empty.segments[0].model_copy(update={
                "steps": [empty.steps[0].model_copy(update={"name": "Dismiss banner with Accept all"})]
            })]
        })
        self.cases.save(empty)
        suite = self.suites.create("Needs automation", "")
        self.suites.add_member(suite.id, empty.id)
        response = self.app.handle("GET", f"/test-suites/{suite.id}/export?format=python")
        self.assertEqual(response.status, 409)
        self.assertIn(b"Suite export needs attention", response.body)
        self.assertIn(b"1 TestCase is not fully automated", response.body)
        self.assertIn((empty.public_id or "").encode(), response.body)
        self.assertIn(b"Step 1", response.body)
        self.assertIn("Step 1 \u2014 Dismiss banner with Accept all".encode("utf-8"), response.body)
        self.assertIn(b"No executable plan is saved.", response.body)
        self.assertIn(f'href="/test-cases/{empty.id}"'.encode(), response.body)
        self.assertIn(f'href="/test-suites/{suite.id}"'.encode(), response.body)
        self.assertIn(b"No plan &lt;script&gt;alert&lt;/script&gt;", response.body)
        self.assertNotIn(b"No plan <script>alert</script>", response.body)


if __name__ == "__main__":
    unittest.main()
