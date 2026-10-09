"""M2.1: real loopback DOM/CSP and execution, deterministic fake generation."""
import json
from unittest.mock import patch

import pytest
from playwright.sync_api import sync_playwright

from qa_agent.automation_lifecycle import AutomationLifecycleService, AutomationStatus, definition_fingerprint
from qa_agent.browser_discovery import capture_current_page_discovery
from qa_agent.browser_runner import BrowserRunner
from qa_agent.cookie_consent import CookieConsentPolicy, cookie_consent_scope
from qa_agent.expected_result_coverage import expected_result_coverage
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan, TestCase as Case, TestStep as Step
from qa_agent.pipeline import PipelineStageError, QATestPipeline
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.web import WebResponse
from qa_agent.workflows import RegressionWorkflow, ValidationWorkflow
from tests.integration.test_registration_coverage_integration import RegistrationTarget
from tests.integration.test_test_case_quality_integration import local_server
from tests.test_registration_coverage import RegistrationProvider
from tests.test_generation_reliability import action


def discovery_at(url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url)
            page.locator('#password').fill('private-discovery-value')
            discovery = capture_current_page_discovery(page)
            assert 'private-discovery-value' not in json.dumps(discovery.snapshot)
            return discovery
        finally:
            browser.close()


def pipeline_for(tmp_path, case, discovery, candidate):
    storage = create_sqlite_storage(tmp_path / 'isolated.sqlite3')
    storage.test_case_repository.save(case)
    provider = RegistrationProvider([candidate])
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    runner = BrowserRunner(tmp_path / 'evidence', headless=True)
    pipeline = QATestPipeline(None, generator, discovery=lambda _: discovery, runner=runner,
        plan_store=storage.plan_store, execution_repository=storage.execution_repository,
        run_history=storage.run_history)
    return pipeline, generator, provider, storage, runner


@pytest.mark.parametrize('reject_input', [False, True])
def test_accepted_values_wording_uses_same_form_and_preserves_real_product_failure(tmp_path, reject_input):
    with local_server(RegistrationTarget(reject_input=reject_input)) as origin:
        url = origin + '/demo-target/registration'
        discovery = discovery_at(url)
        assert discovery.snapshot['forms'][0]['selector'] == '#registration-form'
        assert discovery.snapshot['forms'][0]['required_field_count'] == 0
        error = next(r for r in discovery.snapshot['state_elements'] if r['selector'] == '#form-error')
        assert error['form_selector'] == '#registration-form'
        step = Step(name='Enter valid unique test data into all required fields',
                    description='Fill the registration email and password.',
                    expected='Entered values are accepted without validation errors.', order=0)
        case = Case(name='Input acceptance', description=step.description + ' ' + step.expected,
                    base_url=url, steps=[step])
        candidate = QATestPlan(url=url, steps=[action('navigate', url=url),
            action('fill', '#email', value='local@example.test'),
            action('fill', '#password', value='fixture-only-data'), action('assert_hidden', '#form-error')])
        pipeline, _, provider, storage, _ = pipeline_for(tmp_path, case, discovery, candidate)
        frozen_case = case.model_dump_json()  # Includes the new test DB's assigned public ID.
        with patch('qa_agent.browser_runner.ASSERTION_TIMEOUT_MS', 200):
            result = pipeline.run_test_case(case)
        assert result.executions[0].runner_result['qa_classification'] == ('PRODUCT_FAILURE' if reject_input else 'PASSED')
        assert len(provider.calls) == 1
        assert case.model_dump_json() == frozen_case
        assert definition_fingerprint(storage.test_case_repository.get(case.id)) == definition_fingerprint(case)
        assert len(storage.run_history.list_for_test_case(case.id)) == 1


def test_actual_form_discovery_generation_validation_and_no_ai_regression(tmp_path):
    with local_server(RegistrationTarget()) as origin, cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
        url = origin + '/demo-target/registration'
        discovery = discovery_at(url)
        form = discovery.snapshot['forms'][0]
        assert form['tag'] == 'form' and form['accessible_name'] == 'Registration'
        assert 'hidden-fixture-private-content' not in json.dumps(discovery.snapshot)
        step = Step(name='Open Registration Page', description='Navigate to the registration page.',
                    expected='The registration form is displayed', order=0)
        case = Case(name='Actual form visibility', description=step.description + ' ' + step.expected,
                    base_url=url, steps=[step])
        candidate = QATestPlan(url=url, steps=[action('navigate', url=url), action('assert_visible', '#registration-form')])
        pipeline, generator, provider, storage, runner = pipeline_for(tmp_path, case, discovery, candidate)
        result = pipeline.run_test_case(case)
        assert result.test_run.status.value == 'PASSED'
        version = storage.plan_store.find(step.id)
        frozen = version.model_dump_json()
        assert expected_result_coverage(step, version).is_sufficient
        lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
        lifecycle.mark_automation_completed(case)
        pins = PlanVersionSet.from_mapping({step.id: version.id})
        executor = PinnedExecutionService(storage.plan_store, PlanExecutionService(runner, storage.execution_repository))
        with patch.object(generator.supervisor, 'generate', side_effect=AssertionError('Saved workflows must not use AI')):
            validation = ValidationWorkflow(executor, run_history=storage.run_history).run(case, pins)
            assert validation.outcome.value == 'PASSED'
            assert lifecycle.mark_validation_completed(case, passed=True) == AutomationStatus.AUTOMATION_READY
            regression = RegressionWorkflow(executor, run_history=storage.run_history).run(case, pins)
            assert regression.outcome.value == 'PASSED'
        assert len(provider.calls) == 1
        assert storage.plan_store.get_version(version.id).model_dump_json() == frozen
        assert len(storage.run_history.list_for_test_case(case.id)) == 3


def test_shipped_fixture_remains_unsupported_for_errors_on_all_required_fields(tmp_path):
    with local_server(RegistrationTarget()) as origin:
        url = origin + '/demo-target/registration'
        discovery = discovery_at(url)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url)
                page.locator('#email').fill('local@example.test')
                page.locator('#create-account').click()
                assert page.locator('#password').input_value() == ''
                assert page.locator('#registration-success').is_visible()
            finally:
                browser.close()
        step = Step(name='Submit empty registration form', description='Leave required fields empty and submit.',
                    expected='Submission is blocked and validation errors are displayed for required fields.', order=0)
        case = Case(name='Required field validation', description=step.description + ' ' + step.expected, base_url=url, steps=[step])
        candidate = QATestPlan(url=url, steps=[action('navigate', url=url), action('click', '#create-account'),
            action('assert_visible', '#form-error'), action('assert_hidden', '#registration-success')])
        pipeline, generator, provider, storage, _ = pipeline_for(tmp_path, case, discovery, candidate)
        with pytest.raises(PipelineStageError) as caught:
            pipeline.run_test_case(case)
        assert isinstance(caught.value.__cause__, PlanValidationError)
        assert caught.value.__cause__.issues[0].code == 'EXPECTED_RESULT_NOT_COVERED'
        assert expected_result_coverage(step, candidate, discovery=discovery).status.value == 'UNKNOWN'
        assert len(provider.calls) == 1
        assert storage.plan_store.find(step.id) is None
        assert not storage.execution_repository.list_for_test_case(case)
        assert not storage.run_history.list_for_test_case(case.id)
        assert generator.supervisor.repository.list_records()[0].outcome == 'NEEDS_ATTENTION'


class RequiredFieldsTarget:
    """Independent fixture; the shipped demo's behavior is untouched."""
    def __init__(self, *, native=False, missing_password_error=False):
        self.markup = '''<!doctype html><html><body><form id="registration-form" aria-label="Registration" NO_VALIDATE>
            <label>Email<input id="email" type="email" required aria-errormessage="email-error"></label>
            <label>Password<input id="password" type="password" required aria-errormessage="password-error"></label>
            <span id="email-error" hidden>Email is required.</span><span id="password-error" hidden>Password is required.</span>
            <p id="form-error" role="alert" hidden>Complete the required fields.</p><p id="registration-success" role="status" hidden>Registration submitted.</p>
            <button id="submit" type="submit">Submit</button></form><script src="/fixture.js" defer></script></body></html>'''.replace('NO_VALIDATE', '' if native else 'novalidate')
        if native:
            self.markup = self.markup.replace(' aria-errormessage="email-error"', '').replace(' aria-errormessage="password-error"', '')
        self.javascript = '''document.querySelector('form').addEventListener('submit', event => {
            event.preventDefault();
            for (const name of ['email', 'password']) {
                document.getElementById(name + '-error').hidden = document.getElementById(name).value.length > 0;
            }
            document.getElementById('form-error').hidden = false;
            document.getElementById('registration-success').hidden = true;
        });'''
        if missing_password_error:
            self.javascript += "document.querySelector('form').addEventListener('submit', () => { document.getElementById('password-error').hidden = true; });"

    def close(self):
        pass

    def handle(self, method, target, body=None, headers=None):
        if target == '/fixture.js':
            return WebResponse(200, 'application/javascript', self.javascript.encode())
        return WebResponse.html(200, self.markup)


@pytest.mark.parametrize('missing_password_error', [False, True])
def test_grounded_compound_required_errors_detects_missing_runtime_error(tmp_path, missing_password_error):
    with local_server(RequiredFieldsTarget(missing_password_error=missing_password_error)) as origin:
        discovery = discovery_at(origin)
        assert discovery.snapshot['forms'][0]['required_field_count'] == 2
        step = Step(name='Submit empty registration form', description='Submit with all required fields empty.',
                    expected='Submission is blocked and validation errors are displayed for all required fields.', order=0)
        case = Case(name='Per-field validation', description=step.description + ' ' + step.expected, base_url=origin, steps=[step])
        candidate = QATestPlan(url=origin, steps=[action('navigate', url=origin), action('click', '#submit'),
            action('assert_visible', '#form-error'), action('assert_visible', '#email-error'),
            action('assert_visible', '#password-error'), action('assert_hidden', '#registration-success')])
        pipeline, generator, provider, _, _ = pipeline_for(tmp_path, case, discovery, candidate)
        with patch('qa_agent.browser_runner.ASSERTION_TIMEOUT_MS', 200):
            result = pipeline.run_test_case(case)
        assert result.executions[0].runner_result['qa_classification'] == ('PRODUCT_FAILURE' if missing_password_error else 'PASSED')
        assert len(provider.calls) == 1
        assert set(generator.supervisor.repository.list_records()[0].attempts[0].quality_gates.values()) == {'PASSED'}


def test_browser_native_validation_is_not_invented_as_dom_error_evidence(tmp_path):
    with local_server(RequiredFieldsTarget(native=True)) as origin:
        discovery = discovery_at(origin)
        assert discovery.snapshot['forms'][0]['no_validate'] is False
        assert discovery.snapshot['forms'][0]['required_field_count'] == 2
        controls = [r for r in discovery.snapshot['interactive_elements'] if r.get('required')]
        assert all(not r['error_selectors'] for r in controls)
        step = Step(name='Submit empty registration form', description='Submit with required fields empty.',
                    expected='Submission is blocked and validation errors are displayed for required fields.', order=0)
        case = Case(name='Native validity', description=step.description + ' ' + step.expected, base_url=origin, steps=[step])
        candidate = QATestPlan(url=origin, steps=[action('navigate', url=origin), action('click', '#submit'),
            action('assert_visible', '#form-error'), action('assert_hidden', '#registration-success')])
        pipeline, _, provider, _, _ = pipeline_for(tmp_path, case, discovery, candidate)
        with pytest.raises(PipelineStageError) as caught:
            pipeline.run_test_case(case)
        assert 'Browser-native' in caught.value.__cause__.issues[0].message
        assert len(provider.calls) == 1
