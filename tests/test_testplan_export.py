import io
import json
import unittest
import zipfile
from datetime import datetime, timezone
from uuid import uuid4

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
from qa_agent.execution_semantics import (
    ACTION_TIMEOUT_MS,
    ASSERTION_TIMEOUT_MS,
    NAVIGATION_TIMEOUT_MS,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.testplan_export import (
    ACTION_EXPORT_HANDLERS,
    ExportableTestCase,
    TestPlanExportError as ExportError,
    TestPlanExportService as ExportService,
    _slug,
    csharp_source,
    portable_json,
    portable_zip,
    project_zip,
    python_source,
    safe_basename,
    typescript_source,
)


def _all_actions() -> list[QATestStep]:
    return [
        QATestStep(action="navigate", parameters={"url": "http://127.0.0.1:8000/local"}),
        QATestStep(action="assert_page_loaded"),
        QATestStep(action="assert_title", parameters={"expected": 'Title "quoted" ☃'}),
        QATestStep(action="assert_visible", parameters={"selector": "main > h1", "expected_text": "Ready\\nNow"}),
        QATestStep(action="click", parameters={"selector": "button[type='submit']"}),
        QATestStep(action="fill", parameters={"selector": "input[name='query']", "value": "line 1\nline 2 \"quoted\" ☃"}),
        QATestStep(action="assert_hidden", parameters={"selector": "#hidden"}),
        QATestStep(action="assert_url", parameters={"expected": "http://127.0.0.1:8000/result"}),
        QATestStep(action="select_option", parameters={"selector": "select#kind", "option_label": "Web"}),
        QATestStep(action="assert_text_contains", parameters={"selector": "main", "expected_text": "text\\npart"}),
        QATestStep(action="assert_checked", parameters={"selector": "input#accepted"}),
        QATestStep(action="assert_selected", parameters={"selector": "select#kind", "expected": "Web"}),
        QATestStep(action="assert_selected", parameters={"selector": "input[type=radio]"}),
        QATestStep(action="assert_enabled", parameters={"selector": "button#save"}),
        QATestStep(action="assert_disabled", parameters={"selector": "button#blocked"}),
    ]


def _make_case(name: str = "Search filter") -> DomainTestCase:
    return DomainTestCase(
        name=name,
        description="A saved case with punctuation <safe> and Unicode 雪.",
        base_url="http://127.0.0.1:8000/local",
        segments=[
            ExecutionSegment(order=0, base_url="http://127.0.0.1:8000/one", steps=[
                DomainTestStep(name="Search", description="Search locally.", expected="A result appears.", order=0),
            ]),
            ExecutionSegment(order=1, base_url="http://127.0.0.1:8000/two", steps=[
                DomainTestStep(name="Confirm", description="Confirm locally.", expected="The result is exact.", order=1),
            ]),
        ],
    )


class TestPlanExportTests(unittest.TestCase):
    def setUp(self):
        self.cases = InMemoryTestCaseRepository()
        self.plan_store = InMemoryPlanStore()
        self.case = _make_case()
        self.cases.save(self.case)
        self._save_plan(self.case.steps[0], _all_actions())
        self._save_plan(self.case.steps[1], [
            QATestStep(action="navigate", parameters={"url": "http://127.0.0.1:8000/second"}),
            QATestStep(action="assert_title", parameters={"expected": "Second"}),
        ])
        self.service = ExportService(self.cases, self.plan_store)

    def _save_plan(self, step: DomainTestStep, actions, *, version: int = 1, origin=PlanVersionOrigin.AI_GENERATED):
        plan = self.plan_store.find_test_plan(step.id)
        if plan is None:
            plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        saved = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=version,
            created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            origin=origin,
            qa_test_plan=QATestPlan(url="http://127.0.0.1:8000/local", steps=actions),
        )
        self.plan_store.save(step.id, saved, test_plan=plan)
        return saved

    def test_portable_json_contains_schema_and_exact_saved_current_version(self):
        old = self.plan_store.find(self.case.steps[0].id)
        latest = self._save_plan(
            self.case.steps[0],
            [QATestStep(action="navigate", parameters={"url": "http://127.0.0.1:8000/v2"})],
            version=2,
            origin=PlanVersionOrigin.REPAIRED,
        )

        body, filename = self.service.portable_json(self.case.id)
        payload = json.loads(body)
        exported = payload["test_case"]["segments"][0]["steps"][0]["testplan_version"]
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["source_application"], "AI QA Agent")
        self.assertEqual(payload["test_case"]["public_id"], "TC-0001")
        self.assertEqual(exported["version"], 2)
        self.assertEqual(exported["provenance"], "REPAIRED")
        self.assertEqual(exported["plan"], latest.qa_test_plan.model_dump(mode="json"))
        self.assertNotEqual(exported["version"], old.version)
        self.assertEqual(filename, "tc-0001_search-filter.testplan.json")
        self.assertNotIn(self.case.description, body)  # authoring scenario is not an exported prompt
        self.assertNotIn("api_key", body.casefold())
        self.assertNotIn("C:\\Users\\", body)

    def test_local_paths_and_credential_bearing_urls_are_rejected(self):
        step = self.case.steps[0]
        plan = self.plan_store.find_test_plan(step.id)
        for version_number, unsafe_value in enumerate((
            "C:\\Users\\person\\secret.txt",
            "/home/person/private.txt",
            "https://user:password@127.0.0.1/private",
            "https://127.0.0.1/local?api_key=private",
        ), start=2):
            with self.subTest(value_kind="path" if ":\\" in unsafe_value or unsafe_value.startswith("/") else "url"):
                unsafe_plan = QATestPlan(
                    url="http://127.0.0.1:8000/local",
                    steps=[QATestStep(action="fill", parameters={"selector": "#query", "value": unsafe_value})]
                    if not unsafe_value.startswith("http")
                    else [QATestStep(action="navigate", parameters={"url": unsafe_value})],
                )
                self.plan_store.save(
                    step.id,
                    DomainTestPlanVersion(test_plan_id=plan.id, version=version_number, qa_test_plan=unsafe_plan),
                    test_plan=plan,
                )
                with self.assertRaisesRegex(ExportError, "local path or credential-bearing URL") as caught:
                    self.service.portable_json(self.case.id)
                self.assertNotIn(unsafe_value, str(caught.exception))

    def test_portable_json_serialization_is_deterministic_and_human_readable(self):
        exportable = self.service.get(self.case.id)
        first = portable_json(exportable)
        self.assertEqual(first, portable_json(exportable))
        self.assertTrue(first.endswith("\n"))
        self.assertIn("\n  \"schema_version\": 1", first)
        self.assertEqual(json.loads(first)["test_case"]["segments"][1]["order"], 1)

    def test_python_export_maps_all_canonical_actions_and_compiles(self):
        source = python_source(self.service.get(self.case.id))
        compile(source, "exported_test.py", "exec")
        for fragment in (
            "page.goto(", "page.wait_for_load_state(", "to_have_title(",
            "to_have_text(re.compile(", f".click(timeout={ACTION_TIMEOUT_MS})", ".fill(", "to_be_hidden(",
            "to_have_url(", "select_option(label=",
            "to_be_checked(", "option:checked", "to_be_enabled(", "to_be_disabled(",
        ):
            self.assertIn(fragment, source)
        self.assertIn("page.locator(\"button[type='submit']\")", source)
        self.assertNotIn("qa_agent", source)
        self.assertNotIn("import requests", source)

    def test_typescript_export_maps_actions_and_uses_playwright_test(self):
        source = typescript_source(self.service.get(self.case.id))
        for fragment in (
            "import { test, expect }", "await page.goto(", "waitForLoadState(",
            "toHaveTitle(", "toHaveText(", f".click({{ timeout: {ACTION_TIMEOUT_MS} }})", ".fill(",
            "toBeHidden(", "toHaveURL(", "selectOption(", "toHaveText(new RegExp(",
            "toBeChecked(", "option:checked", "toBeEnabled(", "toBeDisabled(",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("qa_agent", source)

    def test_csharp_export_maps_actions_and_uses_nunit_playwright(self):
        source = csharp_source(self.service.get(self.case.id))
        for fragment in (
            "Microsoft.Playwright.NUnit", "[Test]", "await page.GotoAsync(",
            "WaitForLoadStateAsync(", "ToHaveTitleAsync(", "InnerTextAsync()",
            ".ClickAsync(", ".FillAsync(", "ToBeHiddenAsync(", "ToHaveURLAsync(",
            "SelectOptionAsync(", "GetByText(", "option:checked",
            "ToBeCheckedAsync(", "ToBeEnabledAsync(",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("qa_agent", source)

    def test_click_and_timeout_semantics_match_all_export_targets(self):
        exportable = self.service.get(self.case.id)
        py = python_source(exportable)
        ts = typescript_source(exportable)
        cs = csharp_source(exportable)

        self.assertIn(f".click(timeout={ACTION_TIMEOUT_MS})", py)
        self.assertIn(f".click({{ timeout: {ACTION_TIMEOUT_MS} }})", ts)
        self.assertIn(f".ClickAsync(new() {{ Timeout = {ACTION_TIMEOUT_MS} }})", cs)
        self.assertNotIn("expect_navigation", py)
        self.assertNotIn("wait_for_load_state('load', timeout=10000)", py)
        self.assertNotIn("waitForLoadState('load', { timeout: 10000 })", ts)
        self.assertNotIn("Timeout = 10000", cs)
        self.assertIn(f"timeout={ASSERTION_TIMEOUT_MS}", py)
        self.assertIn(f"timeout: {ASSERTION_TIMEOUT_MS}", ts)
        self.assertIn(f"Timeout = {ASSERTION_TIMEOUT_MS}", cs)
        self.assertIn(f"timeout={NAVIGATION_TIMEOUT_MS}", py)
        self.assertIn(f"timeout: {NAVIGATION_TIMEOUT_MS}", ts)
        self.assertIn(f"Timeout = {NAVIGATION_TIMEOUT_MS}", cs)

    def test_text_semantics_remain_exact_or_contains_across_targets(self):
        exportable = self.service.get(self.case.id)
        py = python_source(exportable)
        ts = typescript_source(exportable)
        cs = csharp_source(exportable)

        self.assertIn("to_have_js_property('innerText'", py)
        self.assertIn("toHaveJSProperty('innerText'", ts)
        self.assertIn("ToHaveJSPropertyAsync(\"innerText\"", cs)
        self.assertIn("re.compile(", py)
        self.assertIn("new RegExp(", ts)
        self.assertIn("to_have_text(re.compile(", py)
        self.assertIn("toHaveText(new RegExp(", ts)
        self.assertIn("ToHaveTextAsync(new Regex(", cs)
        self.assertIn("exact=False", py)
        self.assertIn("exact: false", ts)
        self.assertIn("Exact = false", cs)

    def test_locators_and_special_characters_are_escaped_without_strategy_changes(self):
        source = python_source(self.service.get(self.case.id))
        self.assertIn("button[type='submit']", source)
        self.assertIn(r'line 1\nline 2 \"quoted\"', source)
        self.assertIn("☃", source)
        ts = typescript_source(self.service.get(self.case.id))
        cs = csharp_source(self.service.get(self.case.id))
        self.assertIn("\"input[name='query']\"", ts)
        self.assertIn(r'\"quoted\"', cs)

    def test_segment_and_step_order_are_preserved(self):
        py = python_source(self.service.get(self.case.id))
        self.assertLess(py.index("/local"), py.index("/second"))
        self.assertEqual(py.count("page = page.context.new_page()"), 2)
        ts = typescript_source(self.service.get(self.case.id))
        self.assertEqual(ts.count("page = await page.context().newPage();"), 2)
        cs = csharp_source(self.service.get(self.case.id))
        self.assertIn("var page = await Page.Context.NewPageAsync();", cs)
        self.assertIn("page = await page.Context.NewPageAsync();", cs)
        json_segments = json.loads(portable_json(self.service.get(self.case.id)))["test_case"]["segments"]
        self.assertEqual([item["order"] for item in json_segments], [0, 1])
        self.assertEqual([item["steps"][0]["order"] for item in json_segments], [0, 1])

    def test_case_without_saved_plan_is_rejected_for_all_exports(self):
        no_plan = _make_case("Without automation")
        self.cases.save(no_plan)
        with self.assertRaisesRegex(ExportError, "Automation required"):
            self.service.portable_json(no_plan.id)
        with self.assertRaisesRegex(ExportError, "Automation required"):
            self.service.source(no_plan.id, "python")

    def test_unsupported_action_fails_with_clear_safe_message(self):
        step = self.case.steps[0]
        plan = self.plan_store.find_test_plan(step.id)
        invalid = QATestPlan.model_construct(
            url="http://127.0.0.1:8000/local",
            steps=[QATestStep.model_construct(action="future-action", parameters={})],
        )
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=2,
            origin=PlanVersionOrigin.REPAIRED,
            qa_test_plan=invalid,
        )
        self.plan_store.save(step.id, version, test_plan=plan)
        with self.assertRaisesRegex(ExportError, "unsupported action"):
            self.service.source(self.case.id, "typescript")

    def test_unmapped_action_parameters_fail_instead_of_being_dropped(self):
        step = self.case.steps[0]
        plan = self.plan_store.find_test_plan(step.id)
        unsafe = QATestPlan(
            url="http://127.0.0.1:8000/local",
            steps=[QATestStep(action="fill", parameters={"selector": "#user", "value": "ok", "mystery": "RAW_PRIVATE_VALUE"})],
        )
        version = DomainTestPlanVersion(test_plan_id=plan.id, version=2, qa_test_plan=unsafe)
        self.plan_store.save(step.id, version, test_plan=plan)
        with self.assertRaisesRegex(ExportError, "unsupported parameters") as caught:
            self.service.source(self.case.id, "python")
        self.assertNotIn("RAW_PRIVATE_VALUE", str(caught.exception))
        with self.assertRaisesRegex(ExportError, "unsupported parameters"):
            self.service.portable_json(self.case.id)

    def test_exporter_support_contract_matches_every_canonical_action(self):
        self.assertEqual(set(ACTION_EXPORT_HANDLERS), set(QATestStep.ACTION_PARAMETER_FIELDS))

    def test_case_metadata_with_local_paths_or_url_credentials_is_rejected(self):
        step = self.case.steps[0]
        self._save_plan(step, _all_actions(), version=2)
        updated = self.cases.get(self.case.id)
        updated.base_url = "https://example.test/register?access_token=private"
        self.cases.save(updated)
        with self.assertRaisesRegex(ExportError, "local path or credential-bearing URL") as caught:
            self.service.portable_json(self.case.id)
        self.assertNotIn("private", str(caught.exception))

    def test_filename_sanitization_and_collision_handling(self):
        hostile = self.case.model_copy(update={"name": "../../CON\\checkout? <unsafe>"})
        basename = safe_basename(hostile)
        self.assertNotIn("..", basename)
        self.assertNotIn("/", basename)
        self.assertNotIn("\\", basename)
        self.assertTrue(basename.startswith("tc-0001_"))
        self.assertNotEqual(_slug("CON"), "con")
        exportable = self.service.get(self.case.id)
        duplicate = ExportableTestCase(hostile, exportable.plans)
        archive_bytes = portable_zip([exportable, duplicate])
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertFalse(any(name.startswith("/") or ".." in name.split("/") for name in names))

    def test_bulk_zip_contains_portable_json_and_safe_manifest(self):
        second = _make_case("Second case")
        self.cases.save(second)
        self._save_plan(second.steps[0], [QATestStep(action="navigate", parameters={"url": "http://127.0.0.1:8000/second"})])
        self._save_plan(second.steps[1], [QATestStep(action="assert_page_loaded")])
        content, filename = self.service.bulk_zip([second.id, self.case.id], "portable")
        self.assertEqual(filename, "ai-qa-portable-testplans.zip")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.assertIn("README.md", archive.namelist())
            self.assertIn("export-manifest.json", archive.namelist())
            self.assertEqual(len([name for name in archive.namelist() if name.endswith(".testplan.json")]), 2)
            manifest = json.loads(archive.read("export-manifest.json"))
            self.assertEqual(len(manifest["test_cases"]), 2)
            for name in archive.namelist():
                self.assertFalse(name.startswith("/"))
                self.assertNotIn("..", name.split("/"))

    def test_project_zips_have_manifests_and_external_dependencies_only(self):
        exportable = self.service.get(self.case.id)
        cases = {
            "python": ("python/", "requirements.txt"),
            "typescript": ("typescript/", "package.json"),
            "csharp": ("csharp/", "csharp/AIQAAgent.Export.csproj"),
        }
        for language, (directory, dependency_file) in cases.items():
            with self.subTest(language=language):
                content = project_zip(language, [exportable], suite_name="Smoke")
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    names = archive.namelist()
                    self.assertIn("export-manifest.json", names)
                    self.assertIn(dependency_file, names)
                    self.assertTrue(any(name.startswith(directory) for name in names))
                    manifest = json.loads(archive.read("export-manifest.json"))
                    self.assertEqual(manifest["language"], language)
                    self.assertEqual(manifest["suite"], "Smoke")
                    self.assertEqual(manifest["test_cases"][0]["plan_versions"][0]["version"], 1)
                    self.assertNotIn(b"qa_agent", b"".join(archive.read(name) for name in names if name.endswith((".py", ".ts", ".cs"))))
                    self.assertNotIn("C:\\Users\\", " ".join(archive.namelist()))

    def test_export_service_has_no_router_or_llm_dependency(self):
        self.assertFalse(hasattr(self.service, "_router"))
        self.assertFalse(hasattr(self.service, "_llm_router"))
        first = self.service.source(self.case.id, "python")[0]
        self.assertIn("page.goto", first)
        self.assertEqual(first, self.service.source(self.case.id, "python")[0])


if __name__ == "__main__":
    unittest.main()
