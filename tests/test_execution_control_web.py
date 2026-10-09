"""Real Chromium integration against loopback servers and local fake providers."""
from contextlib import contextmanager
from threading import Event, Thread

import pytest

from playwright.sync_api import expect, sync_playwright, Page
from unittest.mock import patch

from qa_agent.cookie_consent import current_cookie_consent_policy
from qa_agent.evidence_policy import current_evidence_policy
from qa_agent.execution_preferences import SQLiteExecutionPreferences
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.browser_runner import BrowserRunner
from qa_agent.evidence_policy import EvidencePolicy, EvidenceMode
from qa_agent.models import QATestPlan
from qa_agent.llm.router import LLMRouter
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.run_history import RunHistoryService, InMemoryRunHistoryRepository, WorkflowType
from qa_agent.test_case_authoring import TestCaseAuthoringService as AuthoringService
from qa_agent.test_case_execution import TestCaseExecutionService as ExecutionService
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication, create_http_server, WebResponse

from test_background_authoring import AuthoringProvider
from test_execution_control import eventually
from test_suite_runs import _case, _plan, SuiteRunTestHarness


@contextmanager
def local_server(app):
    server = create_http_server(app, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown(); server.server_close(); thread.join(3)


def case_application(preferences, runner=None):
    cases, plans, executions = InMemoryTestCaseRepository(), InMemoryPlanStore(), InMemoryExecutionRepository()
    case = _case(); cases.save(case)
    plan, version = _plan(case); plans.save(case.steps[0].id, version, test_plan=plan)
    history = RunHistoryService(InMemoryRunHistoryRepository(), executions, plan_store=plans)
    execution = ExecutionService(cases, plans, executions, history, runner_factory=lambda _: runner or (lambda _: {"status": "passed", "steps": []}))
    app = LocalWebApplication(history, test_cases=cases, plan_store=plans, run_service=execution, execution_preferences=preferences)
    return app, case


def test_browser_preferences_save_queue_retry_refresh_restart_and_mobile(tmp_path):
    entered, release = Event(), Event()
    class SlowPreferences(SQLiteExecutionPreferences):
        def update(self, *args):
            entered.set(); release.wait(3)
            return super().update(*args)
    database = tmp_path / "preferences.sqlite3"
    preferences = SlowPreferences(database)
    app, case = case_application(preferences)
    original = app._test_cases.get(case.id).model_dump_json()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            with local_server(app) as origin:
                page.goto(f"{origin}/test-cases/{case.id}")
                assert page.locator('[data-execution-preferences]').count() == 1
                assert page.get_by_text('Evidence settings', exact=True).count() == 1
                assert page.get_by_text('Cookie consent', exact=True).count() == 1
                assert page.get_by_role('button', name='Check readiness', exact=True).count() == 1
                assert page.evaluate("(() => {const ids=[...document.querySelectorAll('[id]')].map(n=>n.id); return ids.length===new Set(ids).size;})()")
                page.get_by_text('Evidence settings', exact=True).click()
                page.get_by_text('Cookie consent', exact=True).click()
                page.locator('select[name="evidence_mode"]').select_option('EVERY_STEP')
                assert entered.wait(2)
                expect(page.locator('[data-preferences-status]')).to_have_text('Saving…')
                expect(page.get_by_role('button', name='Run Validation', exact=True)).to_be_disabled()
                page.locator('select[name="cookie_policy"]').select_option('LEAVE_UNCHANGED')
                page.locator('select[name="screenshot_mode"]').select_option('ELEMENT_AND_PAGE')
                release.set()
                expect(page.locator('[data-preferences-status]')).to_have_text('Saved')
                assert preferences.get(case.id).revision == 3
                page.reload()
                for field, value in [('cookie_policy','LEAVE_UNCHANGED'), ('evidence_mode','EVERY_STEP'), ('screenshot_mode','ELEMENT_AND_PAGE')]:
                    expect(page.locator(f'select[name="{field}"]')).to_have_value(value)
                page.get_by_text('Evidence settings', exact=True).click()
                failed = []
                def fail_once(route):
                    if not failed:
                        failed.append(True)
                        route.fulfill(status=503, content_type='application/json', body='{"error":"Local save unavailable"}')
                    else:
                        route.continue_()
                page.route('**/preferences', fail_once)
                page.locator('select[name="evidence_mode"]').select_option('EVERY_VERIFICATION')
                expect(page.locator('[data-preferences-status]')).to_have_text('Save failed — Retry')
                expect(page.get_by_role('button', name='Run Regression', exact=True)).to_be_disabled()
                page.get_by_role('button', name='Retry', exact=True).click()
                expect(page.locator('[data-preferences-status]')).to_have_text('Saved')
                expect(page.get_by_role('button', name='Run Regression', exact=True)).to_be_enabled()
                page.set_viewport_size({'width':375,'height':850})
                expect(page.get_by_role('button', name='Run Validation', exact=True)).to_be_visible()
                expect(page.get_by_role('button', name='Run Regression', exact=True)).to_be_visible()
                assert app._test_cases.get(case.id).model_dump_json() == original
                assert app._run_history.list_recent() == []
            # A fresh app and connection read exactly the same saved record.
            restarted = LocalWebApplication(app._run_history, test_cases=app._test_cases, plan_store=app._plan_store, execution_preferences=SQLiteExecutionPreferences(database))
            with local_server(restarted) as origin:
                page.goto(f"{origin}/test-cases/{case.id}")
                expect(page.locator('select[name="evidence_mode"]')).to_have_value('EVERY_VERIFICATION')
                expect(page.locator('select[name="cookie_policy"]')).to_have_value('LEAVE_UNCHANGED')
                expect(page.locator('select[name="screenshot_mode"]')).to_have_value('ELEMENT_AND_PAGE')
        finally:
            release.set(); browser.close()


@pytest.mark.parametrize("conflict", [False, True])
def test_browser_preferences_reconcile_responses_without_losing_pending_edits(tmp_path, conflict):
    preferences = SQLiteExecutionPreferences(tmp_path / "preferences.sqlite3")
    app, case = case_application(preferences)
    with local_server(app) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page, other_tab = browser.new_page(), browser.new_page()
        try:
            # Hold the first actual HTTP response after the server has handled
            # it, so edits made while it is in flight have a fixed ordering.
            page.add_init_script("""
                const originalFetch = window.fetch.bind(window);
                window.fetch = async (...args) => {
                    const response = await originalFetch(...args);
                    if (String(args[0]).endsWith('/preferences') && !window.preferenceResponseHeld) {
                        window.preferenceResponseHeld = true;
                        await new Promise(resolve => { window.releasePreferenceResponse = resolve; });
                    }
                    return response;
                };
            """)
            url = f"{origin}/test-cases/{case.id}"
            page.goto(url)
            other_tab.goto(url)
            page.get_by_text("Evidence settings", exact=True).click()
            page.get_by_text("Cookie consent", exact=True).click()
            if conflict:
                other_tab.get_by_text("Cookie consent", exact=True).click()
                other_tab.locator('select[name="cookie_policy"]').select_option("LEAVE_UNCHANGED")
                expect(other_tab.locator('[data-preferences-status]')).to_have_text("Saved")
            page.locator('select[name="evidence_mode"]').select_option("EVERY_STEP")
            page.wait_for_function("typeof window.releasePreferenceResponse === 'function'")
            expect(page.locator('[data-preferences-status]')).to_have_text("Saving…")
            page.locator('select[name="evidence_mode"]').select_option("EVERY_VERIFICATION")
            page.locator('select[name="screenshot_mode"]').select_option("ELEMENT_AND_PAGE")
            page.evaluate("window.releasePreferenceResponse()")
            if conflict:
                expect(page.locator('[data-preferences-status]')).to_have_text("Save failed — Retry")
                expect(page.locator('select[name="cookie_policy"]')).to_have_value("LEAVE_UNCHANGED")
                expect(page.locator('select[name="evidence_mode"]')).to_have_value("EVERY_VERIFICATION")
                expect(page.locator('select[name="screenshot_mode"]')).to_have_value("ELEMENT_AND_PAGE")
                assert page.locator('[data-execution-preferences]').get_attribute("data-revision") == "1"
                expect(page.get_by_role("button", name="Run Regression", exact=True)).to_be_disabled()
                page.get_by_role("button", name="Retry", exact=True).click()
            expect(page.locator('[data-preferences-status]')).to_have_text("Saved")
            saved = preferences.get(case.id).model_dump(mode="json")
            assert saved["evidence_mode"] == "EVERY_VERIFICATION"
            assert saved["screenshot_mode"] == "ELEMENT_AND_PAGE"
            assert saved["cookie_policy"] == ("LEAVE_UNCHANGED" if conflict else "AUTO_HANDLE")
            for field in ("cookie_policy", "evidence_mode", "screenshot_mode"):
                expect(page.locator(f'select[name="{field}"]')).to_have_value(saved[field])
            assert page.locator('[data-execution-preferences]').get_attribute("data-revision") == str(saved["revision"])
            expect(page.get_by_role("button", name="Run Regression", exact=True)).to_be_enabled()
            assert app._run_history.list_recent() == []
        finally:
            browser.close()


def test_browser_authoring_stop_reopen_and_late_response(tmp_path):
    entered, release = Event(), Event()
    provider = AuthoringProvider(entered=entered, release=release)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), test_cases=InMemoryTestCaseRepository(), authoring_service=AuthoringService(LLMRouter([provider])))
    with local_server(app) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            job = app._background_authoring.start(name='Check title', base_url=f'{origin}/local', scenario='Check the page title.')
            assert entered.wait(2)
            url = f'{origin}/test-cases/authoring-progress/{job}'
            page.goto(url)
            expect(page.get_by_role('button', name='Stop', exact=True)).to_be_visible()
            page.get_by_role('button', name='Stop', exact=True).click()
            expect(page.locator('[data-authoring-phase]')).to_have_text('Stopped by user')
            expect(page.locator('[data-stop-form]')).to_be_hidden()
            page.close(); page = browser.new_page(); page.goto(url)
            assert page.get_by_role('button', name='Stop', exact=True).count() == 0
            expect(page.locator('[data-authoring-result-message]')).to_have_text('Stopped by user')
            release.set()
            assert app._progress_store.get_authoring(job).state.value == 'CANCELLED'
            assert not app._draft_store._drafts and app._test_cases.list() == []
        finally:
            release.set(); browser.close()


def test_browser_run_stop_waits_for_action_and_owner_cleanup_and_freezes_policy(tmp_path):
    entered, release_action, cleanup_entered, release_cleanup = Event(), Event(), Event(), Event()
    policies = []
    class Runner:
        @contextmanager
        def open_test_case_session(self, case):
            try:
                yield self
            finally:
                cleanup_entered.set(); release_cleanup.wait(5)
        def run_plan(self, segment_order, plan):
            policies.append((current_cookie_consent_policy().value, current_evidence_policy().mode.value))
            entered.set(); release_action.wait(5)
            policies.append((current_cookie_consent_policy().value, current_evidence_policy().mode.value))
            return {'status':'passed', 'steps':[]}
        def __call__(self, plan):
            raise AssertionError('The bound session must execute the Run')
    preferences = SQLiteExecutionPreferences(tmp_path / 'prefs.sqlite3')
    app, case = case_application(preferences, Runner())
    with local_server(app) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            job = app._background_runs.start(case.id, WorkflowType.REGRESSION)
            assert entered.wait(2)
            url = f'{origin}/runs/progress/{job}'
            page.goto(url); page.get_by_role('button',name='Stop',exact=True).click()
            expect(page.locator('[data-progress-phase]')).to_contain_text('Stopping…')
            expect(page.get_by_role('button',name='Stopping…',exact=True)).to_be_disabled()
            assert app._progress_store.get(job).finished_at is None
            page.reload()
            expect(page.get_by_role('button',name='Stopping…',exact=True)).to_be_disabled()
            preferences.update(case.id, 'cookie_policy', 'LEAVE_UNCHANGED', 0)
            preferences.update(case.id, 'evidence_mode', 'EVERY_STEP', 1)
            release_action.set(); assert cleanup_entered.wait(2)
            page.reload()
            expect(page.locator('[data-progress-phase]')).to_contain_text('Stopping…')
            assert app._progress_store.get(job).finished_at is None
            release_cleanup.set()
            expect(page.locator('[data-progress-phase]')).to_have_text('Stopped by user')
            expect(page.locator('[data-stop-form]')).to_be_hidden()
            record = app._run_history.get(app._progress_store.get(job).final_run_id)
            assert record.outcome == 'CANCELLED'
            assert record.evidence_policy.mode.value == 'FAILURES_ONLY'
            assert record.cookie_consent.policy.value == 'AUTO_HANDLE'
            assert policies[:2] == [('AUTO_HANDLE','FAILURES_ONLY')]*2
            response = page.request.post(f'{origin}/test-cases/{case.id}/run', form={'workflow':'REGRESSION'}, max_redirects=0)
            assert response.status == 303
            next_job = response.headers['location'].rsplit('/',1)[-1]
            finished = eventually(lambda: app._progress_store.get(next_job), lambda snapshot: snapshot.finished_at is not None)
            assert finished.outcome == 'PASSED'
            assert policies[2:] == [('LEAVE_UNCHANGED','EVERY_STEP')]*2
            assert app._run_history.get(finished.final_run_id).evidence_policy.mode.value == 'EVERY_STEP'
        finally:
            release_action.set(); release_cleanup.set(); browser.close()


def test_browser_suite_stop_acknowledges_and_terminal_control_disappears():
    harness = SuiteRunTestHarness(); harness.setUp()
    entered, release = Event(), Event()
    def runner(plan):
        entered.set(); release.wait(5)
        return {'status':'passed','steps':[]}
    service = harness.make_service(runner)
    app = LocalWebApplication(harness.history, test_cases=harness.cases, suite_run_service=service)
    with local_server(app) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            run = service.start(harness.suite.id).run
            assert entered.wait(2)
            url=f'{origin}/suite-runs/{run.public_id}'
            page.goto(url); page.get_by_role('button',name='Stop',exact=True).click()
            expect(page.get_by_role('button',name='Stopping…',exact=True)).to_be_disabled()
            page.reload()
            expect(page.get_by_role('button',name='Stopping…',exact=True)).to_be_disabled()
            release.set()
            expect(page.locator('[data-suite-run-live]')).to_have_text('Stopped by user.')
            expect(page.locator('[data-stop-form]')).to_be_hidden()
            page.reload()
            assert page.get_by_role('button',name='Stop',exact=True).count() == 0
        finally:
            release.set(); browser.close()


def test_real_playwright_action_stop_retains_screenshot_and_closes_session(tmp_path):
    entered = Event()
    app, case = case_application(SQLiteExecutionPreferences(tmp_path / 'preferences.sqlite3'), BrowserRunner(tmp_path / 'evidence', headless=True))
    original_handle = app.handle
    def handle(method, target, body=None, **kwargs):
        if method == 'GET' and target == '/local-fixture':
            return WebResponse.html(200, '<!doctype html><title>Welcome</title><p id="notice">Welcome</p>')
        return original_handle(method, target, body, **kwargs)
    app.handle = handle
    original_locator = Page.locator
    def locator(page, selector, **kwargs):
        if selector == '#never-present':
            entered.set()
        return original_locator(page, selector, **kwargs)
    with local_server(app) as origin, patch.object(Page, 'locator', locator), sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Fixture plan is installed before execution; its immutable identity
            # must survive Stop unchanged. All navigation stays on loopback.
            test_plan, version = _plan(case)
            version = version.model_copy(update={'qa_test_plan': QATestPlan(url=f'{origin}/local-fixture', steps=[
                {'action':'navigate','parameters':{'url':f'{origin}/local-fixture'}},
                {'action':'assert_title','parameters':{'expected':'Welcome'}},
                {'action':'click','parameters':{'selector':'#never-present'}},
            ])})
            app._plan_store = InMemoryPlanStore()
            app._run_service._plan_store = app._plan_store
            app._run_history._plan_store = app._plan_store
            app._plan_store.save(case.steps[0].id, version, test_plan=test_plan)
            job = app._background_runs.start(case.id, WorkflowType.REGRESSION, evidence_policy=EvidencePolicy(mode=EvidenceMode.EVERY_VERIFICATION))
            assert entered.wait(4)
            page.goto(f'{origin}/runs/progress/{job}')
            page.get_by_role('button',name='Stop',exact=True).click()
            expect(page.locator('[data-progress-phase]')).to_have_text('Stopped by user', timeout=15000)
            snapshot = app._progress_store.get(job)
            record = app._run_history.get(snapshot.final_run_id)
            assert record.outcome == 'CANCELLED' and record.executions[0].classification == 'CANCELLED'
            execution = app._run_history.get_detail(record.run_id).executions[record.executions[0].execution_id]
            assert [item['status'] for item in execution.runner_result['steps']] == ['passed','passed','cancelled']
            assert execution.evidence and all(__import__('pathlib').Path(item.path).is_file() for item in execution.evidence)
            assert execution.test_plan_version_id == version.id and app._plan_store.find(case.steps[0].id).id == version.id
            expect(page.locator('[data-stop-form]')).to_be_hidden()
        finally:
            browser.close()
