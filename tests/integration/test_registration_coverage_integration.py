"""Production Discovery and browser assertions against a loopback registration page."""
import json
from unittest.mock import patch

import pytest
from playwright.sync_api import sync_playwright

from qa_agent.browser_discovery import capture_current_page_discovery
from qa_agent.browser_runner import BrowserRunner
from qa_agent.llm.router import LLMRouter
from qa_agent.models import ExecutionStatus, QATestPlan, TestCase as Case
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet, StepPlanSelection
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.pipeline import QATestPipeline
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.web import WebResponse, _local_demo_page
from qa_agent.workflows import RegressionWorkflow
from tests.integration.test_test_case_quality_integration import local_server
from tests.test_registration_coverage import RegistrationProvider, assertion, registration_plan, registration_step


class RegistrationTarget:
    def __init__(self, *, reject_input=False):
        markup, _, script = _local_demo_page().partition('<script>')
        javascript, _, tail = script.partition('</script>')
        # Serve the existing fixture's script from the same origin under the
        # local server's CSP, retaining its DOM rather than inventing selectors.
        self.markup = markup + '<script src="/registration.js"></script>' + tail
        self.javascript = javascript + "\nerror.textContent = 'hidden-fixture-private-content';"
        if reject_input:
            self.javascript += "\nform.addEventListener('input', () => { error.hidden = false; });"

    def close(self):
        pass

    def handle(self, method, target, body=None, headers=None):
        if target == '/registration.js':
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
                assert state == {'tag': 'p', 'role': 'alert', 'selector': '#form-error', 'visible': False}
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
