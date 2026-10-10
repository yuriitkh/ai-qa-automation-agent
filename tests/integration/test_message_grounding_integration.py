"""Unmodified local demo: real Discovery, fake generation, Chromium execution."""
import pytest
from playwright.sync_api import expect, sync_playwright

from qa_agent.browser_discovery import capture_current_page_discovery
from qa_agent.browser_runner import BrowserRunner
from qa_agent.models import QATestPlan
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.web import WebResponse, _local_demo_javascript, _local_demo_page
from tests.integration.test_test_case_quality_integration import local_server
from tests.test_message_grounding_regressions import CLARIFIED, MESSAGE, ORIGINAL, empty_submission, generator_for, tc02_step


class UnchangedRegistrationDemo:
    def handle(self, method, target, body=None, headers=None):
        if target == '/assets/demo-registration.js':
            return WebResponse(200, 'application/javascript', _local_demo_javascript().encode())
        if target == '/demo-target/registration':
            return WebResponse.html(200, _local_demo_page())
        return WebResponse.html(404, 'Not found')

    def close(self):
        pass


def test_tc02_empty_form_message_discovery_generation_and_browser_execution(tmp_path):
    with local_server(UnchangedRegistrationDemo()) as origin:
        url = origin + '/demo-target/registration'
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url)
                assert page.locator('#email').input_value() == page.locator('#password').input_value() == ''
                expect(page.locator('#form-error')).to_be_hidden()
                discovery = capture_current_page_discovery(page)
                error = next(item for item in discovery.snapshot['state_elements'] if item['selector'] == '#form-error')
                assert error['visible'] is False and 'text' not in error
                form = next(item for item in discovery.snapshot['forms'] if item['selector'] == '#registration-form')
                assert form['required_field_count'] == 0

                candidate = empty_submission(url=url)
                rejected, _ = generator_for(candidate)
                with pytest.raises(PlanValidationError) as caught:
                    rejected.generate_with_plan(tc02_step(), discovery, requirement_context=ORIGINAL)
                assert caught.value.issues[0].reason_code == 'OUTPUT_VALUE_ONLY_IN_STEP'
                generator, provider = generator_for(candidate)
                generated = generator.generate_with_plan(tc02_step(), discovery, requirement_context=CLARIFIED)
                assert set(generator.supervisor.repository.list_records()[0].attempts[0].quality_gates.values()) == {'PASSED'}
                assert len(provider.calls) == 1

                # Actual product check, independently of generation acceptance.
                page.locator('#create-account').click()
                expect(page.locator('#form-error')).to_be_visible()
                expect(page.locator('#form-error')).to_have_text(MESSAGE)
                expect(page.locator('#registration-success')).to_be_hidden()
                assert page.locator('#email').get_attribute('aria-invalid') == 'true'
                assert page.locator('#password').get_attribute('aria-invalid') is None
                assert page.locator('#email').input_value() == page.locator('#password').input_value() == ''
            finally:
                browser.close()

        # Run the generated exact-message plan in a second, isolated browser.
        plan = generated.test_plan_version.qa_test_plan
        executable = QATestPlan(url=url, steps=[{'action': 'navigate', 'parameters': {'url': url}}, *plan.steps])
        result = BrowserRunner(tmp_path / 'evidence', headless=True)(executable)
        assert result['status'] == 'passed' and len(result['steps']) == 5
        assert all(item['status'] == 'passed' for item in result['steps'])
