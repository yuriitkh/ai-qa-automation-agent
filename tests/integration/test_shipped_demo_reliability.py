"""Browser checks of the actual application route, headers and segment state."""
import json
from unittest.mock import patch

from playwright.sync_api import expect, sync_playwright

from qa_agent.automation_lifecycle import AutomationLifecycleService, AutomationStatus
from qa_agent.browser_discovery import MAX_SNAPSHOT_CHARS, capture_current_page_discovery
from qa_agent.browser_runner import BrowserRunner
from qa_agent.cookie_consent import CookieConsentPolicy, cookie_consent_scope
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan
from qa_agent.pipeline import QATestPipeline
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.workflows import RegressionWorkflow, ValidationWorkflow

from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication
from tests.integration.test_test_case_quality_integration import local_server
from tests.test_generation_boundaries import scenario
from tests.test_registration_coverage import RegistrationProvider


def demo_application(tmp_path):
    return LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()),
        test_cases=InMemoryTestCaseRepository(), evidence_root=tmp_path)


def test_shipped_registration_script_runs_under_actual_csp(tmp_path):
    with local_server(demo_application(tmp_path)) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            response = page.goto(origin + "/demo-target/registration")
            assert response.headers["content-security-policy"] == "default-src 'self'; style-src 'unsafe-inline'; img-src 'self'; object-src 'none'"
            assert page.locator("script:not([src])").count() == 0
            script = page.request.get(origin + "/assets/demo-registration.js")
            assert script.status == 200 and script.headers["content-type"].startswith("application/javascript")
            page.locator("#change-status").click()
            expect(page.locator("#page-status")).to_have_text("Page state changed.", timeout=2000)
        finally:
            browser.close()


def test_actual_registration_invalid_reset_cookie_dialog_and_selection_controls(tmp_path):
    with local_server(demo_application(tmp_path)) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(origin + "/demo-target/registration")
            page.locator("#accept-cookies").click()
            expect(page.locator("#cookie-consent")).to_be_hidden()
            page.locator("#email").fill("invalid")
            page.locator("#password").fill("local-private-fixture")
            page.locator("#create-account").click()
            expect(page.locator("#form-error")).to_have_text("Enter a valid email address.")
            expect(page.locator("#email")).to_have_attribute("aria-invalid", "true")
            expect(page.locator("#registration-success")).to_be_hidden()
            page.locator("#email").fill("local@example.test")
            page.locator("#create-account").click()
            expect(page.locator("#form-error")).to_be_hidden()
            expect(page.locator("#registration-success")).to_be_visible()
            page.locator("#terms").check()
            page.locator("#plan-pro").click()
            page.locator("#region").select_option(label="North")
            expect(page.locator("#terms")).to_be_checked()
            expect(page.locator("#plan-pro")).to_be_checked()
            expect(page.locator("#region")).to_have_value("north")
            discovery = capture_current_page_discovery(page)
            serialized = json.dumps(discovery.snapshot, separators=(",", ":"))
            assert len(serialized) <= MAX_SNAPSHOT_CHARS
            assert "local-private-fixture" not in serialized and "local@example.test" not in serialized
            controls = {item.selector: item for item in discovery.interactive_elements}
            assert controls["#email"].input_type == "email"
            assert controls["#email"].has_value is True and controls["#password"].has_value is True
            assert controls["#create-account"].button_type == "submit"
            assert controls["#region"].option_labels == ("Choose a region", "North", "South")
            page.evaluate("""() => {
                const label = document.createElement('label'); label.id = 'notes-label';
                label.textContent = 'Private notes';
                const area = document.createElement('textarea'); area.id = 'private-notes';
                area.textContent = 'sensitive-default-notes'; label.append(area); document.body.append(label);
            }""")
            safe_discovery = capture_current_page_discovery(page)
            assert 'sensitive-default-notes' not in safe_discovery.model_dump_json()
            notes = next(item for item in safe_discovery.interactive_elements if item.selector == '#private-notes')
            assert notes.accessible_name == 'Private notes' and notes.text == ''
            page.locator('#notes-label').evaluate('(element) => element.remove()')
            page.locator("#reset-form").click()
            expect(page.locator("#email")).to_have_value("")
            expect(page.locator("#password")).to_have_value("")
            expect(page.locator("#email")).not_to_have_attribute("aria-invalid", "true")
            expect(page.locator("#terms")).not_to_be_checked()
            expect(page.locator("#registration-success")).to_be_hidden()
            page.locator("#open-details").click()
            expect(page.locator("#details-dialog")).to_be_visible()
            page.locator("#close-details").click()
            expect(page.locator("#details-dialog")).to_be_hidden()
            page.locator("#open-details").click()
            page.locator("#confirm-details").click()
            expect(page.locator("#dialog-result")).to_be_visible()
            page.locator("#change-status").click()
            page.locator("#reset-demo").click()
            expect(page.locator("#page-status")).to_have_text("Ready")
            expect(page.locator("#cookie-consent")).to_be_visible()
        finally:
            browser.close()


def test_generated_atomic_invalid_email_steps_preserve_state_and_pinned_workflows(tmp_path):
    storage = create_sqlite_storage(tmp_path / "atomic.sqlite3")
    app = demo_application(tmp_path / "served-evidence")
    with local_server(app) as origin:
        case, _, plans = scenario()
        url = origin + "/demo-target/registration"
        case.base_url = url
        case.description = "Enter invalid email and valid password. Submit the registration form. Verify that an error message is displayed and successful confirmation is hidden."
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url)
                initial_discovery = capture_current_page_discovery(page)
            finally:
                browser.close()
        plans = [QATestPlan(url=url, steps=plan.steps) for plan in plans]
        plans[0] = QATestPlan(url=url, steps=[{"action": "navigate", "parameters": {"url": url}}, *plans[0].steps])
        provider = RegistrationProvider(plans)
        generator = LLMTestPlanGenerator(LLMRouter([provider]))
        runner = BrowserRunner(tmp_path / "evidence", headless=True)
        with cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
            result = QATestPipeline(None, generator, discovery=lambda _: initial_discovery, runner=runner,
                plan_store=storage.plan_store, execution_repository=storage.execution_repository, run_history=storage.run_history).run_test_case(case)
            assert result.test_run.status.value == "PASSED"
            assert len(provider.calls) == 4
            assert "browser_state_preserved\": true" in provider.calls[1][0]
            submit_snapshot = json.loads(provider.calls[1][2])
            assert "current_page" in submit_snapshot["strategies_used"]
            assert "fixture-password" not in provider.calls[1][2]
            versions = [storage.plan_store.find(step.id) for step in case.steps]
            frozen = [version.model_dump_json() for version in versions]
            history = storage.run_history.list_for_test_case(case.id)[0]
            assert [[action.action for action in version.qa_test_plan.steps] for version in versions] == [
                ["navigate", "fill", "fill"], ["click"], ["assert_visible"], ["assert_hidden"]]
            pins = PlanVersionSet.from_mapping({step.id: version.id for step, version in zip(case.steps, versions)})
            executor = PinnedExecutionService(storage.plan_store, PlanExecutionService(runner, storage.execution_repository))
            lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
            lifecycle.mark_automation_completed(case)
            with patch.object(generator.supervisor, "generate", side_effect=AssertionError("Pinned workflows must not generate")):
                validation = ValidationWorkflow(executor, run_history=storage.run_history).run(case, pins)
                assert validation.outcome.value == "PASSED"
                assert lifecycle.mark_validation_completed(case, passed=True) == AutomationStatus.AUTOMATION_READY
                regression = RegressionWorkflow(executor, run_history=storage.run_history).run(case, pins)
                assert regression.outcome.value == "PASSED"
            assert len(provider.calls) == 4
            assert [storage.plan_store.get_version(version.id).model_dump_json() for version in versions] == frozen
            assert history in storage.run_history.list_for_test_case(case.id)


def test_duplicate_visible_text_does_not_fail_valid_substring_assertion(tmp_path):
    with local_server(demo_application(tmp_path)) as origin, cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
        url = origin + "/demo-target/registration"
        with BrowserRunner(headless=True).open_session() as session:
            page = session.new_page()
            session.run_plan(page, QATestPlan(url=url, steps=[{"action": "navigate", "parameters": {"url": url}}]))
            page.evaluate("() => { const p = document.createElement('p'); p.textContent = 'Registration demo'; document.body.append(p); }")
            plan = QATestPlan(url=url, steps=[{"action": "assert_text_contains", "parameters": {"expected_text": "Registration demo"}}])
            assert session.run_plan(page, plan)["status"] == "passed"
            page.evaluate("() => { document.querySelector('h1').hidden = true; document.body.lastElementChild.hidden = true; }")
            with patch("qa_agent.browser_runner.ASSERTION_TIMEOUT_MS", 250):
                assert session.run_plan(page, plan)["status"] == "failed"
