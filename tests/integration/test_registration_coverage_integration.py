"""Production Discovery and browser assertions against a loopback registration page."""
import json
from unittest.mock import patch

import pytest
from playwright.sync_api import sync_playwright

from qa_agent.browser_discovery import capture_current_page_discovery
from qa_agent.automation_lifecycle import AutomationLifecycleService, AutomationStatus
from qa_agent.expected_result_coverage import expected_result_coverage
from qa_agent.browser_runner import BrowserRunner
from qa_agent.llm.router import LLMRouter
from qa_agent.models import ExecutionStatus, QATestPlan, TestCase as Case, TestStep as Step
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet, StepPlanSelection
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.pipeline import QATestPipeline
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.web import WebResponse, _local_demo_page, _local_demo_javascript
from qa_agent.workflows import RegressionWorkflow, ValidationWorkflow, WorkflowOutcome
from tests.integration.test_test_case_quality_integration import local_server
from tests.test_registration_coverage import RegistrationProvider, assertion, registration_plan, registration_step


class RegistrationTarget:
    def __init__(self, *, reject_input=False):
        self.markup = _local_demo_page()
        self.javascript = _local_demo_javascript() + "\nerror.textContent = 'hidden-fixture-private-content';"
        if reject_input:
            self.javascript += "\nform.addEventListener('input', () => { error.hidden = false; });"

    def close(self):
        pass

    def handle(self, method, target, body=None, headers=None):
        if target == '/assets/demo-registration.js':
            return WebResponse(200, 'application/javascript', self.javascript.encode())
        return WebResponse.html(200, self.markup)


@pytest.mark.parametrize("reject_input,require_error_absence,classification", [
    (False, True, "PASSED"), (True, True, "PRODUCT_FAILURE"),
    (True, False, "AUTOMATION_EXECUTION_ERROR"),
])
def test_observed_hidden_error_coverage_preserves_requirement_and_runtime_classification(tmp_path, reject_input, require_error_absence, classification):
    with local_server(RegistrationTarget(reject_input=reject_input)) as origin:
        url = origin + '/demo-target/registration'
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url)
                page.locator('#password').fill('private-input-value')
                # The reject fixture makes input surface an error. Restore
                # initial state to verify hidden identity capture in both cases.
                page.locator('#form-error').evaluate('(element) => { element.hidden = true; }')
                discovery = capture_current_page_discovery(page)
                state = next(item for item in discovery.snapshot['state_elements'] if item['selector'] == '#form-error')
                assert state == {'tag': 'p', 'role': 'alert', 'selector': '#form-error', 'visible': False,
                                 'form_selector': '#registration-form'}
                assert not any(item.selector == '#form-error' for item in discovery.interactive_elements)
                assert not any(item['selector'] == '#form-error' for item in discovery.snapshot['visible_text_elements'])
                snapshot = json.dumps(discovery.snapshot)
                assert 'hidden-fixture-private-content' not in snapshot and 'private-input-value' not in snapshot
                assert page.locator('#form-error').is_hidden() and page.locator('#password').input_value() == 'private-input-value'
            finally:
                browser.close()

        storage = create_sqlite_storage(tmp_path / 'registration.sqlite3')
        step = registration_step()
        scenario = (step.description + ' ' + step.expected if require_error_absence else
                    'Fill the registration fields and verify the confirmation state is displayed.')
        case = Case(name='User Registration and Confirmation Display', description=scenario, base_url=url, steps=[step])
        plan = registration_plan(assertion(), url=url)
        plan = QATestPlan(url=url, steps=[{"action": "navigate", "parameters": {"url": url}}, *plan.steps])
        provider = RegistrationProvider([plan])
        generator = LLMTestPlanGenerator(LLMRouter([provider]))
        runner = BrowserRunner(tmp_path / 'evidence', headless=True)
        pipeline = QATestPipeline(None, generator, discovery=lambda _: discovery, runner=runner,
            plan_store=storage.plan_store, execution_repository=storage.execution_repository, run_history=storage.run_history)
        with patch('qa_agent.browser_runner.ASSERTION_TIMEOUT_MS', 300):
            result = pipeline.run_test_case(case)
        execution = result.executions[0]
        assert execution.status == (ExecutionStatus.FAILED if reject_input else ExecutionStatus.PASSED)
        assert execution.runner_result['qa_classification'] == classification
        version = storage.plan_store.find(step.id)
        frozen = version.model_dump_json()
        assert provider.calls[0][2].find('state_elements') >= 0
        assert len(provider.calls) == 1
        assert set(generator.supervisor.repository.list_records()[0].attempts[0].quality_gates.values()) == {'PASSED'}
        assert len(storage.run_history.list_for_test_case(case.id)) == 1
        if not reject_input:
            pins = PlanVersionSet(selections=(StepPlanSelection(step.id, version.id),))
            regression = RegressionWorkflow(PinnedExecutionService(storage.plan_store, PlanExecutionService(runner, storage.execution_repository)),
                                            run_history=storage.run_history)
            with patch.object(generator.supervisor, 'generate', side_effect=AssertionError('Regression must never generate')):
                saved = regression.run(case, pins)
            assert saved.test_run.status == ExecutionStatus.PASSED
            assert len(provider.calls) == 1 and storage.plan_store.get_version(version.id).model_dump_json() == frozen


def test_discovered_semantic_subject_survives_sqlite_validation_and_no_ai_regression(tmp_path):
    target = RegistrationTarget()
    target.javascript += "\ndocument.querySelector('#page-status').textContent = 'Registration successful';"
    with local_server(target) as origin:
        url = origin + '/demo-target/registration'
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url)
                discovery = capture_current_page_discovery(page)
            finally:
                browser.close()
        step = Step(name='Verify registration confirmation', description='Verify the registration confirmation message.',
                    expected='A confirmation message is displayed.', order=0)
        case = Case(name='Observed registration confirmation', description=step.description + ' ' + step.expected,
                    base_url=url, steps=[step])
        plan = QATestPlan(url=url, steps=[
            {'action': 'navigate', 'parameters': {'url': url}},
            {'action': 'assert_visible', 'parameters': {'selector': '#page-status'}},
        ])
        provider = RegistrationProvider([plan])
        generator = LLMTestPlanGenerator(LLMRouter([provider]))
        storage = create_sqlite_storage(tmp_path / 'semantic-coverage.sqlite3')
        runner = BrowserRunner(tmp_path / 'evidence')
        result = QATestPipeline(None, generator, discovery=lambda _: discovery, runner=runner,
            plan_store=storage.plan_store, execution_repository=storage.execution_repository,
            run_history=storage.run_history).run_test_case(case)
        assert result.test_run.status == ExecutionStatus.PASSED
        version = storage.plan_store.find(step.id)
        frozen = version.model_dump_json()
        history = storage.run_history.list_for_test_case(case.id)[0]
        assert expected_result_coverage(step, version).is_sufficient
        assert 'Registration successful' not in frozen
        pins = PlanVersionSet.from_mapping({step.id: version.id})
        executor = PinnedExecutionService(storage.plan_store, PlanExecutionService(runner, storage.execution_repository))
        lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
        lifecycle.mark_automation_completed(case)
        with patch.object(generator.supervisor, 'generate', side_effect=AssertionError('Saved workflows must never generate')):
            validation = ValidationWorkflow(executor, run_history=storage.run_history).run(case, pins)
            assert validation.outcome == WorkflowOutcome.PASSED
            assert lifecycle.mark_validation_completed(case, passed=True) == AutomationStatus.AUTOMATION_READY
            regression = RegressionWorkflow(executor, run_history=storage.run_history).run(case, pins)
            assert regression.outcome == WorkflowOutcome.PASSED
        assert len(provider.calls) == 1
        assert storage.plan_store.get_version(version.id).model_dump_json() == frozen
        assert history in storage.run_history.list_for_test_case(case.id)
