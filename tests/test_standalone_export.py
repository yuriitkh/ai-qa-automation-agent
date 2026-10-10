"""Offline M4 regressions, with generated projects run outside the repository."""

import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from qa_agent.automation_lifecycle import (
    AutomationLifecycleService, AutomationStatus, InMemoryAutomationLifecycleRepository,
)
from qa_agent.cli import main
from qa_agent.models import (
    ExecutionSegment, FailurePolicy, ManualRecoverySource, QATestPlan, QATestStep,
    TestCase as Case, TestPlan as Plan, TestPlanVersion as Version, TestStep as Step,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.test_case_review import InMemoryTestCaseReviewRepository, TestCaseReviewService as Review
from qa_agent.testplan_export import (
    ExportableTestCase, TestPlanExportError as ExportError, TestPlanExportService as Exports,
    _zip_info, csharp_source, portable_json, project_zip, python_source, safe_basename, typescript_source,
)


SOURCE_URL = "http://127.0.0.1:1/form"


def action(name, **parameters):
    return QATestStep(action=name, parameters=parameters)


def make_export(actions=None, *, multi_step=False, store=None, cases=None):
    store = store or InMemoryPlanStore()
    cases = cases or InMemoryTestCaseRepository()
    steps = [Step(name="Open page", description="Open the local page.", expected="The page loads.", order=0)]
    if multi_step:
        steps.append(Step(name="Confirm state", description="Check retained input.", expected="The page loads.", order=1))
    case = Case(name="Standalone registration", description="Local fixture only.", base_url=SOURCE_URL, steps=steps)
    cases.save(case)
    sequences = actions or [[action("navigate", url=SOURCE_URL), action("assert_page_loaded")]]
    for step, sequence in zip(steps, sequences, strict=True):
        plan = Plan(test_step_id=step.id, name=step.name)
        store.save(step.id, Version(test_plan_id=plan.id, version=1, qa_test_plan=QATestPlan(url=SOURCE_URL, steps=sequence)), test_plan=plan)
    lifecycle = AutomationLifecycleService(InMemoryAutomationLifecycleRepository(), store)
    review = Review(InMemoryTestCaseReviewRepository(), store)
    service = Exports(cases, store, lifecycle=lifecycle, review=review)
    return case, store, cases, lifecycle, review, service


def approve(case, lifecycle, review):
    review.approve_for_validation(case)
    lifecycle.mark_automation_completed(case)
    assert lifecycle.mark_validation_completed(case, True) == AutomationStatus.AUTOMATION_READY


def unpack(data, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(directory)
    return directory


def clean_environment():
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("PYTEST_") or key in {"PYTHONPATH", "BASE_URL", "BROWSER", "HEADLESS", "TIMEOUT_MS", "ACTION_TIMEOUT_MS", "ASSERTION_TIMEOUT_MS", "NAVIGATION_TIMEOUT_MS"}:
            environment.pop(key, None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def run_pytest(directory, *arguments, environment=None):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *arguments], cwd=directory,
                          env=environment or clean_environment(), capture_output=True, text=True, timeout=60)


def test_python_project_discovery_is_independent(tmp_path, local_target):
    case, _, _, lifecycle, review, service = make_export()
    approve(case, lifecycle, review)
    data, _ = service.bulk_zip([case.id], "python")
    directory = unpack(data, tmp_path / "exported")
    result = run_pytest(directory, "--collect-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout
    assert (directory / "python/tests/test_tc_0001_standalone_registration.py").exists()
    assert "pytest-playwright" not in (directory / "requirements.txt").read_text()
    for path in directory.rglob("*.py"):
        ast.parse(path.read_text())
        assert "qa_agent" not in path.read_text()
    workflow = (directory / ".github/workflows/tests.yml").read_text()
    assert "python -m pytest -q" in workflow and "--with-deps chromium" in workflow
    assert "continue-on-error" not in workflow
    environment = clean_environment()
    environment["BASE_URL"] = local_target[0]
    executed = run_pytest(directory, environment=environment)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "1 passed" in executed.stdout


@pytest.fixture
def local_target():
    hits = []
    html = b'''<!doctype html><title>Local fixture</title><main>
    <input id="name"><button id="save" onclick="document.querySelector('#message').innerText='Ready\\nNow'; document.cookie='fixture=yes'">Save</button>
    <div id="message">Waiting</div><div id="hidden" hidden>Hidden</div>
    <input id="accepted" type="checkbox"><input id="choice" type="radio">
    <select id="kind"><option value="other">Other</option><option value="web">Web</option></select>
    <button id="blocked" disabled>Blocked</button></main>
    <script>if(document.cookie.includes('fixture=yes')) document.querySelector('#message').innerText='Retained cookie';</script>'''
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def execute_project(tmp_path, local_target, sequences, *, multi_step=False, case_change=None):
    case, _, cases, _, _, service = make_export(sequences, multi_step=multi_step)
    if case_change:
        case_change(case)
        cases.save(case)
    directory = unpack(project_zip("python", [service.get(case.id)]), tmp_path / "project")
    environment = clean_environment()
    environment.update(BASE_URL=local_target[0], HEADLESS="true", ASSERTION_TIMEOUT_MS="200", ACTION_TIMEOUT_MS="1000")
    return run_pytest(directory, environment=environment)


def test_python_runtime_all_actions_and_state_across_steps(tmp_path, local_target):
    sequences = [
        [action("navigate", url=SOURCE_URL), action("assert_page_loaded"), action("assert_title", expected="Local fixture"),
         action("fill", selector="#name", value='Alice "quoted"'), action("click", selector="#save")],
        [action("assert_value", selector="#name", expected='Alice "quoted"'),
         action("assert_visible", selector="#message", expected_text="Ready\\nNow"),
         action("assert_text_contains", selector="main", expected_text="Ready\\nNow"),
         action("assert_hidden", selector="#hidden"), action("check", selector="#accepted"),
         action("assert_checked", selector="#accepted"), action("uncheck", selector="#accepted"),
         action("assert_unchecked", selector="#accepted"), action("select_option", selector="#kind", option_label="Web"),
         action("assert_selected", selector="#kind", expected="Web"), action("assert_selected", selector="#kind", expected="web"),
         action("check", selector="#choice"), action("assert_selected", selector="#choice"),
         action("assert_enabled", selector="#save"), action("assert_disabled", selector="#blocked"),
         action("assert_url", expected=SOURCE_URL)],
    ]
    result = execute_project(tmp_path, local_target, sequences, multi_step=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_python_initializes_segment_and_shares_context(tmp_path, local_target):
    sequences = [[action("click", selector="#save")], [action("assert_visible", selector="#message", expected_text="Retained cookie")]]
    def split(case):
        case.segments = [ExecutionSegment(order=i, base_url=SOURCE_URL, steps=[step]) for i, step in enumerate(case.steps)]
    result = execute_project(tmp_path, local_target, sequences, multi_step=True, case_change=split)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bad_action", [
    action("assert_title", expected="Wrong title"),
    action("assert_value", selector="#name", expected="Wrong value"),
    action("assert_visible", selector="#message", expected_text="Wrong text"),
    action("assert_text_contains", selector="main", expected_text="Missing text"),
    action("assert_selected", selector="#kind", expected="Unavailable"),
    action("assert_selected", selector="#choice", expected="invalid-radio-expectation"),
    action("assert_selected", selector="#kind"),
])
def test_python_incorrect_assertions_never_pass(tmp_path, local_target, bad_action):
    result = execute_project(tmp_path, local_target, [[action("navigate", url=SOURCE_URL), bad_action]])
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed" in result.stdout and "1 passed" not in result.stdout


@pytest.mark.parametrize("policy,continued", [(FailurePolicy.CONTINUE, True), (FailurePolicy.BLOCK_REST, False)])
def test_python_failure_policy_retains_failure(tmp_path, local_target, policy, continued):
    sequences = [[action("navigate", url=SOURCE_URL), action("assert_title", expected="Wrong")],
                 [action("navigate", url=SOURCE_URL.replace("/form", "/continued")), action("assert_page_loaded")]]
    def change(case):
        case.steps[0].failure_policy = policy
    result = execute_project(tmp_path, local_target, sequences, multi_step=True, case_change=change)
    assert result.returncode == 1, result.stdout + result.stderr
    assert ("/continued" in local_target[1]) == continued


@pytest.mark.parametrize("language", ["python", "typescript", "csharp"])
def test_projects_have_stable_bytes_and_review_status(language):
    case, _, _, _, _, service = make_export()
    snapshot = service.get(case.id)
    first = project_zip(language, [snapshot])
    assert first == project_zip(language, [snapshot])
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert "REVIEW_ONLY" in archive.read("README.md").decode()
        manifest = json.loads(archive.read("export-manifest.json"))
        assert manifest["standalone_execution"] == "NOT_TESTED"
        assert manifest["test_cases"][0]["verification"] == "REVIEW_ONLY"


def test_typescript_structure_and_pinned_dependency():
    from importlib.metadata import version
    case, _, _, _, _, service = make_export()
    with zipfile.ZipFile(io.BytesIO(project_zip("typescript", [service.get(case.id)]))) as archive:
        package = json.loads(archive.read("package.json"))
        assert package["devDependencies"] == {"@playwright/test": version("playwright")}
        assert package["scripts"]["test"] == "playwright test"
        json.loads(archive.read("tsconfig.json"))
        config = archive.read("playwright.config.ts").decode()
        assert "./typescript/tests" in config and "HEADLESS" in config and "BROWSER" in config
        source = archive.read(next(name for name in archive.namelist() if name.endswith(".spec.ts"))).decode()
        assert "import { test, expect } from '@playwright/test'" in source
        assert "if (errors.length) throw" in source


def test_typescript_all_actions_parse_with_installed_node(tmp_path):
    import playwright
    from test_testplan_export import _all_actions
    node = shutil.which("node") or str(Path(playwright.__file__).parent / "driver" / ("node.exe" if os.name == "nt" else "node"))
    if not Path(node).is_file():
        pytest.skip("NOT TESTED: Node runtime unavailable for TypeScript syntax validation")
    case, _, _, _, _, service = make_export([_all_actions()])
    directory = unpack(project_zip("typescript", [service.get(case.id)]), tmp_path / "typescript")
    for path in directory.rglob("*.ts"):
        # Node's --check parses TS as JavaScript. Its installed type stripper
        # plus an unlinked VM module validates syntax without loading imports.
        check = "const {stripTypeScriptTypes}=require('node:module'); const {SourceTextModule}=require('node:vm'); new SourceTextModule(stripTypeScriptTypes(require('node:fs').readFileSync(process.argv[1], 'utf8')));"
        result = subprocess.run([node, "--experimental-vm-modules", "-e", check, str(path)], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr


def test_csharp_project_preserves_nunit_and_unique_classes():
    case, _, _, _, _, service = make_export()
    snapshot = service.get(case.id)
    duplicate = ExportableTestCase(case.model_copy(update={"name": "Standalone-registration"}), snapshot.plans)
    with zipfile.ZipFile(io.BytesIO(project_zip("csharp", [snapshot, duplicate]))) as archive:
        project = ET.fromstring(archive.read("csharp/AIQAAgent.Export.csproj"))
        packages = {item.attrib["Include"]: item.attrib["Version"] for item in project.findall(".//PackageReference")}
        assert packages == {"Microsoft.NET.Test.Sdk": "17.10.0", "Microsoft.Playwright.NUnit": "1.44.0", "NUnit": "3.14.0", "NUnit3TestAdapter": "4.5.0"}
        assert project.findtext(".//IsTestProject") == "true"
        assert b"csharp/AIQAAgent.Export.csproj" in archive.read("AIQAAgent.Export.sln")
        ET.fromstring(archive.read("csharp/export.runsettings"))
        classes = []
        for filename in archive.namelist():
            if filename.endswith(".cs"):
                source = archive.read(filename).decode()
                classes += re.findall(r"public class (\w+) : PageTest", source)
                assert "[TestFixture]" in source and "[Test]" in source
                assert 'if (errors.Count > 0) Assert.Fail' in source
        assert len(classes) == len(set(classes)) == 2


def test_long_csharp_class_names_remain_unique_after_collision_suffix():
    case, _, _, _, _, service = make_export()
    snapshot = service.get(case.id)
    long_case = case.model_copy(update={"name": "x" * 100})
    with zipfile.ZipFile(io.BytesIO(project_zip("csharp", [ExportableTestCase(long_case, snapshot.plans)] * 2))) as archive:
        sources = [archive.read(name).decode() for name in archive.namelist() if name.endswith(".cs")]
    classes = [re.search(r"public class (\w+)", source)[1] for source in sources]
    assert len(set(classes)) == 2


@pytest.mark.parametrize("language,generator", [("python", python_source), ("typescript", typescript_source), ("csharp", csharp_source)])
def test_timeout_configuration_never_rewrites_selector_or_data_literals(language, generator):
    # These strings look like emitted timing arguments and must stay literal.
    selector = '[data-note="timeout=5000 timeout: 5000 Timeout = 5000"]'
    value = "timeout=5000 timeout: 5000 Timeout = 5000"
    case, _, _, _, _, service = make_export([[action("fill", selector=selector, value=value)]])
    source = generator(service.get(case.id))
    if language == "python":
        tree = ast.parse(source)
        fill = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fill")
        assert ast.literal_eval(fill.args[0]) == value
        locator = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "locator")
        assert ast.literal_eval(locator.args[0]) == selector
    else:
        assert value in source
        assert 'timeout=5000 timeout: 5000 Timeout = 5000' in source


@pytest.mark.parametrize("state", ["unapproved", "unvalidated", "failed", "stale-definition", "stale-plan", "manual-recovery"])
def test_verified_export_rejects_ineligible_states(state):
    case, store, cases, lifecycle, review, service = make_export()
    approve(case, lifecycle, review)
    if state == "unapproved":
        review.mark_ready_for_review(case)
    elif state == "unvalidated":
        lifecycle.mark_automation_completed(case)
    elif state == "failed":
        lifecycle.mark_validation_completed(case, False)
    elif state == "stale-definition":
        case.name = "Changed definition"
        cases.save(case)
    else:
        old = store.find(case.steps[0].id)
        changes = {"id": uuid4(), "version": 2}
        if state == "manual-recovery":
            changes["manual_recovery"] = ManualRecoverySource(operation_id=uuid4())
        store.save(case.steps[0].id, old.model_copy(update=changes), test_plan=store.find_test_plan(case.steps[0].id))
    with pytest.raises(ExportError, match="approved, Browser Validated"):
        service.bulk_zip([case.id], "python")
    source, _ = service.source(case.id, "python")
    assert "for review" in source


def test_verified_manifest_pins_current_version_without_claiming_execution_pass():
    case, store, _, lifecycle, review, service = make_export()
    approve(case, lifecycle, review)
    with zipfile.ZipFile(io.BytesIO(service.bulk_zip([case.id], "python")[0])) as archive:
        manifest = json.loads(archive.read("export-manifest.json"))
        assert manifest["test_cases"][0]["verification"] == "APPROVED_AND_BROWSER_VALIDATED"
        assert manifest["standalone_execution"] == "NOT_TESTED"
        snapshot = service.get_verified(case.id)
    old = store.find(case.steps[0].id)
    store.save(case.steps[0].id, old.model_copy(update={"id": uuid4(), "version": 2}), test_plan=store.find_test_plan(case.steps[0].id))
    assert snapshot.plans[0].version.version == 1
    assert json.loads(portable_json(snapshot))["test_case"]["segments"][0]["steps"][0]["testplan_version"]["version"] == 1
    with pytest.raises(ExportError):
        service.get_verified(case.id)


def test_plan_change_during_readiness_cannot_verify_old_snapshot():
    case, store, _, lifecycle, review, service = make_export()
    approve(case, lifecycle, review)
    def replace_plan(_):
        old = store.find(case.steps[0].id)
        store.save(case.steps[0].id, old.model_copy(update={"id": uuid4(), "version": 2}), test_plan=store.find_test_plan(case.steps[0].id))
        return AutomationStatus.AUTOMATION_READY
    with patch.object(lifecycle, "status", side_effect=replace_plan), pytest.raises(ExportError):
        service.get_verified(case.id)


@pytest.mark.parametrize("selector,value", [
    ('input[type="password"]', "dummy-private-password"),
    ("#api_key", "dummy-private-key"),
    ("#query", "Bearer dummy-private-token"),
    ("#query", "password=dummy-private-password"),
    ("#query", "https://user:dummy-private-password@127.0.0.1/form"),
    ("#query", "C:\\Users\\person\\private.txt"),
    ("#query", "/home/person/private.txt"),
])
def test_secret_values_fail_all_renderers_without_echo(selector, value):
    case, _, _, _, _, service = make_export([[action("fill", selector=selector, value=value)]])
    # Public low-level renderers must enforce safety too.
    snapshot = service.get(case.id) if not (":\\" in value or value.startswith("/home") or value.startswith("https")) else None
    generators = [python_source, typescript_source, csharp_source, portable_json]
    for generator in generators:
        with pytest.raises(ExportError) as caught:
            generator(snapshot or service.get(case.id))
        assert value not in str(caught.value)


def test_registered_credentials_and_metadata_are_blocked(monkeypatch):
    value = "dummy-registered-private-value"
    monkeypatch.setenv("FIXTURE_API_KEY", value)
    case, _, cases, _, _, service = make_export([[action("fill", selector="#query", value=value)]])
    with pytest.raises(ExportError, match="sensitive data"):
        service.source(case.id, "python")
    case.name = value
    cases.save(case)
    with pytest.raises(ExportError):
        service.bulk_zip([case.id], "portable")


@pytest.mark.parametrize("path", ["../private", "/absolute", "C:/private", "folder\\..\\private", "folder\\file", "bad\x00path"])
def test_zip_paths_reject_traversal_and_windows_paths(path):
    with pytest.raises(ExportError):
        _zip_info(path)


def test_hostile_public_id_and_normalized_names_cannot_escape_archive():
    case, _, _, _, _, service = make_export()
    snapshot = service.get(case.id)
    hostile = case.model_copy(update={"public_id": "../../C:\\private", "name": "../../CON\\private"})
    basename = safe_basename(hostile)
    assert all(value not in basename for value in ("..", "/", "\\", ":"))
    for language in ("python", "typescript", "csharp"):
        # Names themselves contain private paths and must fail before packaging.
        with pytest.raises(ExportError):
            project_zip(language, [ExportableTestCase(hostile, snapshot.plans)])


def test_unsupported_actions_and_parameters_fail_direct_renderers():
    case, _, _, _, _, service = make_export()
    snapshot = service.get(case.id)
    for sequence in ([QATestStep.model_construct(action="future-action", parameters={})],
                     [action("click", selector="#save", timeout="123")]):
        plan = QATestPlan.model_construct(url=SOURCE_URL, steps=sequence)
        from dataclasses import replace
        saved = replace(snapshot.plans[0], version=snapshot.plans[0].version.model_copy(update={"qa_test_plan": plan}))
        for language in ("python", "typescript", "csharp"):
            with pytest.raises(ExportError):
                project_zip(language, [ExportableTestCase(case, (saved,))])


def cli_fixture(tmp_path, *, validated):
    database = tmp_path / "source.sqlite3"
    storage = create_sqlite_storage(database)
    case, _, _, _, _, _ = make_export(store=storage.plan_store, cases=storage.test_case_repository)
    lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
    review = Review(storage.test_case_review_repository, storage.plan_store)
    if validated:
        approve(case, lifecycle, review)
    return database, case


def test_cli_exports_verified_project_from_read_only_snapshot(tmp_path, capsys):
    database, case = cli_fixture(tmp_path, validated=True)
    before = hashlib.sha256(database.read_bytes()).digest()
    output = tmp_path / "standalone.zip"
    with patch("qa_agent.cli.build_pipeline", side_effect=AssertionError("No LLM pipeline allowed")):
        assert main(["export", "--database", str(database), "--test-case", case.public_id,
                     "--format", "python", "--output", str(output)]) == 0
    assert hashlib.sha256(database.read_bytes()).digest() == before
    with zipfile.ZipFile(output) as archive:
        assert "pytest.ini" in archive.namelist()
        assert json.loads(archive.read("export-manifest.json"))["test_cases"][0]["verification"] == "APPROVED_AND_BROWSER_VALIDATED"
    assert "Exported" in capsys.readouterr().out


def test_cli_unvalidated_source_review_and_no_overwrite(tmp_path, capsys):
    database, case = cli_fixture(tmp_path, validated=False)
    output = tmp_path / "source.py"
    arguments = ["export", "--database", str(database), "--test-case", str(case.id), "--format", "python", "--output", str(output)]
    assert main(arguments) == 1
    assert not output.exists()
    assert main([*arguments, "--source-review"]) == 0
    first = output.read_bytes()
    assert main([*arguments, "--source-review"]) == 1
    assert output.read_bytes() == first
    ast.parse(first.decode())


def test_cli_export_usage_does_not_build_pipeline():
    with patch("qa_agent.cli.build_pipeline", side_effect=AssertionError("No pipeline")):
        assert main(["export"]) == 2


@pytest.mark.parametrize("language", ["python", "typescript", "csharp"])
@pytest.mark.parametrize("state", ["ready", "unapproved", "unvalidated"])
def test_all_web_project_downloads_enforce_shared_eligibility(language, state):
    from urllib.parse import urlencode
    from qa_agent.execution_repository import InMemoryExecutionRepository
    from qa_agent.reporting import RunReportGenerator
    from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
    from qa_agent.test_suites import InMemoryTestSuiteRepository, TestSuiteService
    from qa_agent.web import LocalWebApplication
    case, store, cases, lifecycle, review, _ = make_export()
    approve(case, lifecycle, review)
    if state == "unapproved":
        review.mark_ready_for_review(case)
    elif state == "unvalidated":
        lifecycle.mark_automation_completed(case)
    suites = TestSuiteService(InMemoryTestSuiteRepository(), cases)
    suite = suites.create("Standalone smoke", "Local fixture")
    suites.add_member(suite.id, case.id)
    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository(), InMemoryExecutionRepository()), RunReportGenerator(),
        test_cases=cases, plan_store=store, automation_lifecycle=lifecycle, test_case_review=review, test_suites=suites,
    )
    responses = [
        app.handle("GET", f"/test-cases/{case.id}/export/{language}-zip"),
        app.handle("POST", "/test-cases/export", urlencode({"test_case_id": str(case.id), "target": language})),
        app.handle("GET", f"/test-suites/{suite.id}/export?format={language}"),
        app.handle("POST", "/export", urlencode({"case": case.public_id, "target": language, "policy": "require_all"})),
    ]
    for response in responses:
        assert response.status == (200 if state == "ready" else 409), response.body.decode(errors="replace")[:300]
        if state == "ready":
            assert response.content_type == "application/zip"
            assert ".zip" in response.headers["Content-Disposition"]
            with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
                assert json.loads(archive.read("export-manifest.json"))["test_cases"][0]["verification"] == "APPROVED_AND_BROWSER_VALIDATED"
    # Historical single-file routes remain an explicitly labeled review path.
    source = app.handle("GET", f"/test-cases/{case.id}/export/{language}")
    assert source.status == 200 and b"for review" in source.body


def test_insufficient_assertion_coverage_cannot_export_verified_project():
    case, _, cases, lifecycle, review, service = make_export()
    case.steps[0].expected = "The name field contains Alice."
    cases.save(case)
    lifecycle.mark_automation_completed(case)
    assert lifecycle.mark_validation_completed(case, True) == AutomationStatus.NEEDS_VALIDATION
    with pytest.raises(ValueError, match="cover each expected result"):
        review.approve_for_validation(case)
    with pytest.raises(ExportError):
        service.bulk_zip([case.id], "python")


def test_url_override_and_timeout_configuration(monkeypatch):
    case, _, _, _, _, service = make_export()
    namespace = {}
    exec(python_source(service.get(case.id)), namespace)
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:8765")
    assert namespace["export_url"](SOURCE_URL + "?q=yes#part", SOURCE_URL) == "http://127.0.0.1:8765/form?q=yes#part"
    assert namespace["export_url"]("http://localhost:1234/other", SOURCE_URL) == "http://localhost:1234/other"
    monkeypatch.setenv("TIMEOUT_MS", "222")
    assert namespace["export_timeout"]("ACTION_TIMEOUT_MS", 5000) == 222
    monkeypatch.setenv("ACTION_TIMEOUT_MS", "333")
    assert namespace["export_timeout"]("ACTION_TIMEOUT_MS", 5000) == 333
    monkeypatch.setenv("ACTION_TIMEOUT_MS", "0")
    with pytest.raises(ValueError):
        namespace["export_timeout"]("ACTION_TIMEOUT_MS", 5000)
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:8765/private")
    with pytest.raises(ValueError):
        namespace["export_url"](SOURCE_URL, SOURCE_URL)
