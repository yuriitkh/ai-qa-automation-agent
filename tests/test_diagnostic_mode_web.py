"""Diagnostic settings and polling in Chromium against loopback only."""
from playwright.sync_api import expect, sync_playwright
import pytest

from qa_agent.execution_progress import ExecutionProgressReporter, active_execution_progress
from qa_agent.llm.router import LLMRouter
from qa_agent.llm_usage import llm_usage_scope
from qa_agent.models import TestCase as Case
from qa_agent.reliability import AutomationReliabilitySupervisor, SQLiteReliabilityRepository
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.web import LocalWebApplication
from tests.test_diagnostic_mode import input_step, candidate
from tests.test_automation_reliability import LocalProvider
from tests.test_execution_control_web import local_server
from tests.test_generation_reliability import observed_registration


def test_browser_settings_restart_progress_failure_and_export_links(tmp_path):
    database = tmp_path / 'diagnostic-settings.sqlite3'
    supervisor = AutomationReliabilitySupervisor(SQLiteReliabilityRepository(database))
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), reliability=supervisor)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            with local_server(app) as origin:
                page.goto(f'{origin}/settings/diagnostics')
                page.locator('select[name="level"]').select_option('TRACE')
                page.get_by_role('button', name='Save Diagnostic Mode').click()
                expect(page.locator('p').filter(has_text='Current effective level')).to_have_text('Current effective level: TRACE')
                step = input_step()
                case = Case(name='Local diagnostic fixture', description='Local input.', steps=[step])
                progress_id = app.progress_store.create(case.id, 'AUTOMATION')
                reporter = ExecutionProgressReporter(app.progress_store, progress_id)
                reporter.test_case_loaded(case)
                gen = LLMTestPlanGenerator(LLMRouter([LocalProvider([candidate(expected='private-value')])]), supervisor)
                with active_execution_progress(reporter), llm_usage_scope(related_test_case_id=case.id), pytest.raises(PlanValidationError):
                    gen.generate_with_plan(step, observed_registration())
                page.goto(f'{origin}/runs/progress/{progress_id}')
                expect(page.locator('.progress-step .progress-failure')).to_have_text('Failed gate: assertion_grounding · Failure reason: FILL_ASSERT_VALUE_MISMATCH')
                page.get_by_role('link', name='Generation decisions — Step 1: Verify input').click()
                expect(page.get_by_text('Provider response: SUCCESS · Plan decision: REJECTED', exact=True)).to_be_visible()
                expect(page.get_by_text('Failed gate: assertion_grounding · Failure reason: FILL_ASSERT_VALUE_MISMATCH', exact=True)).to_be_visible()
                assert page.get_by_role('link', name='Export Diagnostics (JSON)').count() == 1
                assert page.get_by_role('link', name='Export Diagnostics (ZIP)').count() == 1
                assert 'private-value' not in page.content()
                assert page.get_by_text('Diagnostic chronology', exact=True).count() == 1
        finally:
            browser.close()
            app.close()
    restarted = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()),
        reliability=AutomationReliabilitySupervisor(SQLiteReliabilityRepository(database)))
    try:
        assert b'Current effective level: <strong>TRACE</strong>' in restarted.handle('GET', '/settings/diagnostics').body
    finally:
        restarted.close()
