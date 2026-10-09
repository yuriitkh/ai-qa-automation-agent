import io
import json
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from urllib.parse import urlencode
from uuid import uuid4

from qa_agent.automation_lifecycle import (
    AutomationLifecycleService,
    AutomationStatus,
    InMemoryAutomationLifecycleRepository,
)
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
from qa_agent.test_suites import InMemoryTestSuiteRepository, TestSuiteService as SuiteService
from qa_agent.web import LocalWebApplication


def _case(name: str) -> DomainTestCase:
    return DomainTestCase(
        name=name,
        description="Local definition for export workspace tests.",
        base_url="http://127.0.0.1:8000/demo-target/registration",
        steps=[DomainTestStep(
            name="Open local page",
            description="Open the local test page.",
            expected="The page loads.",
            order=0,
        )],
    )


class ExportWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.cases = InMemoryTestCaseRepository()
        self.plans = InMemoryPlanStore()
        self.lifecycle = AutomationLifecycleService(InMemoryAutomationLifecycleRepository(), self.plans)
        self.suite_repo = InMemoryTestSuiteRepository()
        self.suites = SuiteService(self.suite_repo, self.cases)
        history = RunHistoryService(InMemoryRunHistoryRepository(), InMemoryExecutionRepository())
        self.app = LocalWebApplication(
            history,
            RunReportGenerator(),
            test_cases=self.cases,
            plan_store=self.plans,
            test_suites=self.suites,
            automation_lifecycle=self.lifecycle,
        )
        self.ready_case = self.add_ready_case("Alpha registration")

    def add_case(self, name: str) -> DomainTestCase:
        case = _case(name)
        self.cases.save(case)
        return case

    def add_ready_case(self, name: str) -> DomainTestCase:
        case = self.add_case(name)
        for step in case.steps:
            plan = DomainTestPlan(test_step_id=step.id, name=step.name)
            self.plans.save(
                step.id,
                DomainTestPlanVersion(
                    test_plan_id=plan.id,
                    version=1,
                    qa_test_plan=QATestPlan(
                        url=case.base_url,
                        steps=[
                            QATestStep(action="navigate", parameters={"url": case.base_url}),
                            QATestStep(action="assert_page_loaded"),
                        ],
                    ),
                ),
                test_plan=plan,
            )
        self.lifecycle.mark_automation_completed(case)
        self.lifecycle.mark_validation_completed(case, True)
        return case

    def add_blocked_case(self, name: str) -> DomainTestCase:
        case = self.add_case(name)
        for step in case.steps:
            plan = DomainTestPlan(test_step_id=step.id, name=step.name)
            self.plans.save(
                step.id,
                DomainTestPlanVersion(
                    test_plan_id=plan.id,
                    version=1,
                    qa_test_plan=QATestPlan(
                        url=case.base_url,
                        steps=[QATestStep(action="navigate", parameters={"url": case.base_url})],
                    ),
                ),
                test_plan=plan,
            )
        self.lifecycle.mark_automation_completed(case)
        return case

    @staticmethod
    def _form(pairs):
        return urlencode(pairs)

    def _submit(self, pairs):
        return self.app.handle("POST", "/export", self._form(pairs))

    def test_export_route_has_top_level_navigation_and_source_modes(self):
        response = self.app.handle("GET", "/export")
        html = response.body.decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn('href="/export" aria-current="page">Export</a>', html)
        self.assertIn('aria-label="Export source"', html)
        self.assertIn(">TestCases</a>", html)
        self.assertIn(">Test Suites</a>", html)

    def test_testcase_search_matches_public_id_and_name(self):
        other = self.add_ready_case("Delta settings")
        by_name = self.app.handle("GET", "/export?q=Alpha+registration").body.decode("utf-8")
        self.assertIn("Alpha registration", by_name)
        self.assertNotIn("Delta settings", by_name)
        by_public_id = self.app.handle("GET", f"/export?q={self.ready_case.public_id}").body.decode("utf-8")
        self.assertIn("Alpha registration", by_public_id)
        self.assertNotEqual(other.public_id, self.ready_case.public_id)

    def test_suite_search_and_testcase_filter_by_suite(self):
        other = self.add_ready_case("Other case")
        suite = self.suites.create("Checkout suite", "")
        self.suites.create("Settings suite", "")
        self.suites.add_member(suite.id, self.ready_case.id)
        suite_html = self.app.handle("GET", "/export?mode=suites&q=Checkout").body.decode("utf-8")
        self.assertIn("Checkout suite", suite_html)
        self.assertNotIn("Settings suite", suite_html)
        case_html = self.app.handle("GET", f"/export?test_suite={suite.id}").body.decode("utf-8")
        self.assertIn("Alpha registration", case_html)
        self.assertNotIn("Other case", case_html)
        self.assertIn(f'value="{suite.id}" selected', case_html)
        self.assertNotEqual(other.id, self.ready_case.id)

    def test_suite_readiness_filters_find_ready_and_blocked_members(self):
        blocked = self.add_blocked_case("Needs validation case")
        ready_suite = self.suites.create("Ready suite", "")
        blocked_suite = self.suites.create("Blocked suite", "")
        self.suites.add_member(ready_suite.id, self.ready_case.id)
        self.suites.add_member(blocked_suite.id, blocked.id)

        ready_html = self.app.handle("GET", "/export?mode=suites&readiness=READY").body.decode("utf-8")
        self.assertIn("Ready suite", ready_html)
        self.assertNotIn("Blocked suite", ready_html)
        blocked_html = self.app.handle("GET", "/export?mode=suites&readiness=BLOCKED").body.decode("utf-8")
        self.assertIn("Blocked suite", blocked_html)
        self.assertNotIn("Ready suite", blocked_html)

    def test_lifecycle_filter_and_existing_date_filters(self):
        blocked = self.add_blocked_case("Needs validation case")
        ready_html = self.app.handle("GET", "/export?lifecycle=AUTOMATION_READY").body.decode("utf-8")
        self.assertIn("Alpha registration", ready_html)
        self.assertNotIn("Needs validation case", ready_html)
        blocked_html = self.app.handle("GET", "/export?lifecycle=NEEDS_VALIDATION").body.decode("utf-8")
        self.assertIn("Needs validation case", blocked_html)
        self.assertNotIn("Alpha registration", blocked_html)

        now = datetime.now(timezone.utc)
        self.cases._timestamps[self.ready_case.id] = (now - timedelta(days=2), now - timedelta(days=2))
        self.cases._timestamps[blocked.id] = (now, now)
        date_html = self.app.handle("GET", f"/export?created_from={now.date().isoformat()}").body.decode("utf-8")
        self.assertIn("Needs validation case", date_html)
        self.assertNotIn("Alpha registration", date_html)
        updated_html = self.app.handle("GET", f"/export?updated_from={now.date().isoformat()}").body.decode("utf-8")
        self.assertIn("Needs validation case", updated_html)
        self.assertNotIn("Alpha registration", updated_html)
        created_to_html = self.app.handle(
            "GET", f"/export?created_to={(now.date() - timedelta(days=1)).isoformat()}"
        ).body.decode("utf-8")
        self.assertIn("Alpha registration", created_to_html)
        self.assertNotIn("Needs validation case", created_to_html)

    def test_testcase_catalog_uses_bounded_pages(self):
        for index in range(101):
            self.add_case(f"Paged case {index:03d}")
        first_page = self.app.handle("GET", "/export").body.decode("utf-8")
        self.assertEqual(first_page.count('type="checkbox" name="case"'), 100)
        self.assertIn('aria-label="TestCase pages"', first_page)
        self.assertIn("page=2", first_page)
        second_page = self.app.handle("GET", "/export?page=2").body.decode("utf-8")
        self.assertEqual(second_page.count('type="checkbox" name="case"'), 2)
        self.assertIn("Previous", second_page)

    def test_selected_testcase_and_suite_are_preselected(self):
        case_html = self.app.handle(
            "GET", f"/export?mode=testcases&case={self.ready_case.public_id}"
        ).body.decode("utf-8")
        self.assertIn(f'name="case" value="{self.ready_case.public_id}" aria-label=', case_html)
        self.assertIn(" checked", case_html)
        self.assertIn("1 direct TestCases", case_html)
        suite = self.suites.create("Preselected suite", "")
        self.suites.add_member(suite.id, self.ready_case.id)
        suite_html = self.app.handle("GET", f"/export?mode=suites&suite={suite.id}").body.decode("utf-8")
        self.assertIn(f'name="suite" value="{suite.id}" aria-label=', suite_html)
        self.assertIn(" checked", suite_html)
        self.assertIn("Preselected suite", suite_html)

    def test_require_all_multiselect_produces_portable_json_bundle(self):
        second = self.add_ready_case("Beta profile")
        response = self._submit([
            ("case", self.ready_case.public_id),
            ("case", second.public_id),
            ("target", "portable"),
            ("policy", "require_all"),
        ])
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
            manifest = json.loads(archive.read("export-manifest.json"))
            self.assertEqual(len(manifest["test_cases"]), 2)
            self.assertEqual(len([name for name in archive.namelist() if name.endswith(".testplan.json")]), 2)

    def test_portable_single_case_is_json_and_playwright_formats_are_zips(self):
        portable = self._submit([
            ("case", self.ready_case.public_id), ("target", "portable"), ("policy", "require_all"),
        ])
        self.assertEqual(portable.status, 200)
        self.assertTrue(portable.content_type.startswith("application/json"))
        self.assertEqual(json.loads(portable.body)["test_case"]["public_id"], self.ready_case.public_id)

        for target, suffix in (("python", ".py"), ("typescript", ".spec.ts"), ("csharp", "Tests.cs")):
            with self.subTest(target=target):
                response = self._submit([
                    ("case", self.ready_case.public_id), ("target", target), ("policy", "require_all"),
                ])
                self.assertEqual(response.status, 200)
                self.assertEqual(response.content_type, "application/zip")
                with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
                    self.assertIn("export-manifest.json", archive.namelist())
                    self.assertTrue(any(name.endswith(suffix) for name in archive.namelist()))

    def test_suite_and_direct_selection_deduplicate_by_testcase_identity(self):
        first_suite = self.suites.create("Shared membership", "")
        second_suite = self.suites.create("Another shared membership", "")
        self.suites.add_member(first_suite.id, self.ready_case.id)
        self.suites.add_member(second_suite.id, self.ready_case.id)
        response = self._submit([
            ("case", self.ready_case.public_id),
            ("suite", str(first_suite.id)),
            ("suite", str(second_suite.id)),
            ("target", "portable"),
            ("policy", "require_all"),
        ])
        self.assertEqual(response.status, 200)
        self.assertTrue(response.content_type.startswith("application/json"))
        self.assertEqual(json.loads(response.body)["test_case"]["public_id"], self.ready_case.public_id)

    def test_blocked_reason_and_require_all_policy_stop_download(self):
        blocked = self.add_blocked_case("Needs validation")
        page = self.app.handle("GET", f"/export?case={blocked.public_id}").body.decode("utf-8")
        self.assertIn("Blocked", page)
        self.assertIn("Saved automation needs Validation before export.", page)
        response = self._submit([
            ("case", self.ready_case.public_id), ("case", blocked.public_id),
            ("target", "python"), ("policy", "require_all"),
        ])
        self.assertEqual(response.status, 409)
        self.assertNotIn("Content-Disposition", response.headers)
        self.assertIn(b"Require-all export stopped", response.body)
        self.assertIn(b"Saved automation needs Validation before export.", response.body)

    def test_ready_only_exports_ready_cases_and_explains_omission(self):
        blocked = self.add_blocked_case("Blocked case")
        page = self.app.handle(
            "GET", f"/export?case={self.ready_case.public_id}&case={blocked.public_id}"
        ).body.decode("utf-8")
        self.assertIn("Ready-only export excludes blocked cases listed above", page)
        response = self._submit([
            ("case", self.ready_case.public_id), ("case", blocked.public_id),
            ("target", "python"), ("policy", "ready_only"),
        ])
        self.assertEqual(response.status, 200)
        with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
            manifest = json.loads(archive.read("export-manifest.json"))
            self.assertEqual([item["public_id"] for item in manifest["test_cases"]], [self.ready_case.public_id])

    def test_testcase_and_suite_export_shortcuts_open_the_workspace(self):
        case_list = self.app.handle("GET", "/test-cases").body.decode("utf-8")
        self.assertNotIn('name="test_case_id"', case_list)
        self.assertNotIn("Export selected", case_list)
        self.assertIn(f"/export?mode=testcases&amp;case={self.ready_case.public_id}", case_list)
        case_detail = self.app.handle("GET", f"/test-cases/{self.ready_case.id}").body.decode("utf-8")
        self.assertIn("Export &rarr;", case_detail)
        self.assertNotIn("Python Playwright</a>", case_detail)

        suite = self.suites.create("Shortcut suite", "")
        self.suites.add_member(suite.id, self.ready_case.id)
        suite_detail = self.app.handle("GET", f"/test-suites/{suite.id}").body.decode("utf-8")
        self.assertIn("Export suite &rarr;", suite_detail)
        self.assertIn(f"/export?mode=suites&amp;suite={suite.id}", suite_detail)
        self.assertNotIn("Portable JSON ZIP", suite_detail)

    def test_central_download_uses_existing_exporter_and_never_calls_llm(self):
        exporter = self.app._testplan_exports
        spy = Mock(wraps=exporter)
        self.app._testplan_exports = spy
        provider_router = Mock()
        self.app._provider_router = provider_router
        response = self._submit([
            ("case", self.ready_case.public_id), ("target", "python"), ("policy", "require_all"),
        ])
        self.assertEqual(response.status, 200)
        spy.bulk_zip.assert_called_once_with([self.ready_case.id], "python")
        provider_router.assert_not_called()

    def test_export_keeps_public_ids_and_does_not_include_credentials(self):
        response = self._submit([
            ("case", self.ready_case.public_id), ("target", "portable"), ("policy", "require_all"),
        ])
        exported = response.body.decode("utf-8")
        self.assertIn(self.ready_case.public_id, exported)
        self.assertNotIn(str(self.ready_case.id), exported)
        self.assertNotIn("api_key", exported.casefold())
        self.assertNotIn("secret", exported.casefold())


if __name__ == "__main__":
    unittest.main()
