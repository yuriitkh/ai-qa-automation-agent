"""M2.2 real loopback browser execution; fake generation and isolated storage."""
from unittest.mock import patch

import pytest
from playwright.sync_api import sync_playwright

from qa_agent.browser_runner import BrowserRunner
from qa_agent.cookie_consent import CookieConsentPolicy, cookie_consent_scope
from qa_agent.models import TestCase as Case, TestStep as Step, QATestPlan
from qa_agent.pinned_execution import PlanVersionSet, PinnedExecutionService
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.workflows import ValidationWorkflow, RegressionWorkflow
from qa_agent.automation_lifecycle import AutomationLifecycleService, AutomationStatus
from tests.integration.test_generation_reliability_integration import discovery_at, pipeline_for
from tests.integration.test_registration_coverage_integration import RegistrationTarget
from tests.integration.test_test_case_quality_integration import local_server
from tests.test_generation_reliability import action


def test_initial_fill_plan_establishes_configured_page_and_retains_state_and_immutable_versions(tmp_path):
    with local_server(RegistrationTarget()) as origin, cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
        url = origin + '/demo-target/registration'
        discovery = discovery_at(url)
        step = Step(name='Input invalid email', description="Enter 'invalid-email' into the email field.",
                    expected='Email field contains the invalid email.', order=0)
        case = Case(name='Invalid input', description=step.description + ' ' + step.expected, base_url=url, steps=[step])
        candidate = QATestPlan(url=url, steps=[action('fill', '#email', value='invalid-email'),
                           action('assert_value', '#email', expected='invalid-email')])
        pipeline, generator, provider, storage, runner = pipeline_for(tmp_path, case, discovery, candidate)
        assert pipeline.run_test_case(case).test_run.status.value == 'PASSED'
        version = storage.plan_store.find(step.id)
        frozen = version.model_dump_json()
        old_run = storage.run_history.list_for_test_case(case.id)[0].model_dump_json()
        lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
        lifecycle.mark_automation_completed(case)
        executor = PinnedExecutionService(storage.plan_store, PlanExecutionService(runner, storage.execution_repository))
        pins = PlanVersionSet.from_mapping({step.id: version.id})
        with patch.object(generator.supervisor, 'generate', side_effect=AssertionError('No AI allowed')):
            assert ValidationWorkflow(executor, run_history=storage.run_history).run(case, pins).outcome.value == 'PASSED'
            assert lifecycle.mark_validation_completed(case, passed=True) == AutomationStatus.AUTOMATION_READY
            assert RegressionWorkflow(executor, run_history=storage.run_history).run(case, pins).outcome.value == 'PASSED'
        assert storage.plan_store.get_version(version.id).model_dump_json() == frozen
        assert old_run in [run.model_dump_json() for run in storage.run_history.list_for_test_case(case.id)]
        assert len(provider.calls) == 1


def test_initialized_segment_does_not_navigate_again_between_related_steps(tmp_path):
    with local_server(RegistrationTarget()) as origin, cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
        url = origin + '/demo-target/registration'
        steps = [Step(name='Fill email', description='Enter invalid email.', expected='Input is entered.', order=0),
                 Step(name='Verify form', description='Verify registration.', expected='The registration form is displayed.', order=1)]
        case = Case(name='State preservation', description='Prepare and verify registration.', base_url=url, steps=steps)
        with BrowserRunner(tmp_path / 'evidence', headless=True).open_test_case_session(case) as session:
            first = QATestPlan(url=url, steps=[action('fill', '#email', value='invalid-email')])
            assert session.run_plan(0, first)['status'] == 'passed'
            page = session._pages[0]
            with patch.object(page, 'goto', side_effect=AssertionError('Do not reset a valid segment')):
                second = QATestPlan(url=url, steps=[action('assert_visible', '#registration-form')])
                assert session.run_plan(0, second)['status'] == 'passed'
                assert page.locator('#email').input_value() == 'invalid-email'
                assert session.capture_discovery(0).snapshot['forms'][0]['selector'] == '#registration-form'


def test_input_value_assertion_failure_is_product_failure_and_redacts_input(tmp_path):
    with local_server(RegistrationTarget()) as origin, sync_playwright() as playwright, cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
        url = origin + '/demo-target/registration'
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url)
            page.locator('#email').evaluate("element => element.addEventListener('input', () => { element.value = ''; })")
            candidate = QATestPlan(url=url, steps=[action('fill', '#email', value='private-input-value'),
                                  action('assert_value', '#email', expected='private-input-value')])
            from qa_agent.browser_runner import _run_plan_on_page
            from qa_agent.browser_discovery import capture_current_page_discovery
            step = Step(name='Fill email', description="Enter 'private-input-value' into the email field.",
                        expected='Email field contains the input.', order=0)
            case = Case(name='Rejected input', description=step.description, base_url=url, steps=[step])
            _, generator, _, storage, _ = pipeline_for(tmp_path, case, capture_current_page_discovery(page), candidate)
            generated = generator.generate_with_plan(step, capture_current_page_discovery(page))
            with patch('qa_agent.browser_runner.ASSERTION_TIMEOUT_MS', 200):
                result = _run_plan_on_page(candidate, page)
            assert result['status'] == 'failed' and result['steps'][-1]['action'] == 'assert_value'
            assert 'private-input-value' not in str(result)
            outcome = PlanExecutionService(lambda _: result, storage.execution_repository).execute(step, generated.test_plan_version)
            assert outcome.classification.value == 'PRODUCT_FAILURE'
        finally:
            browser.close()


def test_missing_configured_url_does_not_infer_target_from_generated_plan(tmp_path):
    step = Step(name='Fill email', description='Enter email.', expected='Input is entered.', order=0)
    case = Case(name='Missing target', description='No browser target established.', steps=[step])
    candidate = QATestPlan(url='http://127.0.0.1:1/invented', steps=[action('fill', '#email', value='fixture')])
    with BrowserRunner(tmp_path / 'evidence', headless=True).open_test_case_session(case) as session:
        with pytest.raises(ValueError, match='configured segment URL'):
            session.run_plan(0, candidate)
        assert not session._pages


def test_discovery_establishes_new_configured_segment_without_resetting_previous_segment(tmp_path):
    from qa_agent.models import ExecutionSegment
    with local_server(RegistrationTarget()) as origin, cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
        url = origin + '/demo-target/registration'
        steps = [Step(name=f'Input {i}', description='Enter email.', expected='Input is entered.', order=i) for i in range(2)]
        case = Case(name='Two segments', description='Independent configured pages.', segments=[
            ExecutionSegment(order=i, base_url=url, steps=[step]) for i, step in enumerate(steps)])
        with BrowserRunner(tmp_path / 'evidence', headless=True).open_test_case_session(case) as session:
            assert session.run_plan(0, QATestPlan(url=url, steps=[action('fill', '#email', value='private-first-segment')]))['status'] == 'passed'
            discovery = session.capture_discovery(1)
            assert discovery.status.value == 'PARTIAL' and discovery.snapshot['forms'][0]['selector'] == '#registration-form'
            assert '#email' in {item.selector for item in discovery.interactive_elements}
            assert session._pages[1].locator('#email').input_value() == ''
            assert session._pages[0].locator('#email').input_value() == 'private-first-segment'
            assert 'private-first-segment' not in discovery.model_dump_json()
