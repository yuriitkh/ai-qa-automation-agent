"""Real Chromium editor and cookie-state checks against loopback fixtures."""
from copy import deepcopy
from contextlib import contextmanager
from threading import Thread
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from playwright.sync_api import expect, sync_playwright

from qa_agent.browser_discovery import capture_current_page_discovery
from qa_agent.browser_runner import BrowserRunner
from qa_agent.cookie_consent import CookieConsentPolicy, cookie_consent_scope, handle_cookie_consent
from qa_agent.drafts import Draft
from qa_agent.models import QATestPlan, TestCase as Case, TestStep as Step, DiscoveryResult, DiscoveryStatus, ExecutionStatus
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.pipeline import QATestPipeline
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_case_execution import WorkflowAvailability
from qa_agent.web import WebResponse, create_http_server
from tests.test_test_case_quality_editing import application, case_fixture


@contextmanager
def local_server(app):
    server = create_http_server(app, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown(); server.server_close(); thread.join(3)


def assert_step_action_layout(page):
    actions = page.locator('.structured-step .step-actions').first
    assert actions.locator('button').all_text_contents() == [
        "Insert Step before", "Insert Step after", "Move up", "Move down", "Delete"]
    delete = actions.get_by_role('button', name='Delete Step 1', exact=True)
    assert 'danger-button' in delete.get_attribute('class').split()
    assert delete.evaluate("button => getComputedStyle(button).color") == 'rgb(141, 37, 25)'
    assert delete.evaluate("button => getComputedStyle(button).backgroundColor") == 'rgb(255, 240, 237)'
    for width in (1280, 390):
        page.set_viewport_size({"width": width, "height": 900})
        boxes = [actions.locator('button').nth(i).bounding_box() for i in range(5)]
        assert max(box['height'] for box in boxes) - min(box['height'] for box in boxes) < 2
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert all(box['x'] >= 0 and box['x'] + box['width'] <= width for box in boxes)
        if width == 1280:
            assert max(box['y'] for box in boxes) - min(box['y'] for box in boxes) < 2
    page.set_viewport_size({"width": 1280, "height": 900})


def test_editor_review_insert_move_delete_cancel_save_conflict_and_mobile(tmp_path):
    app, case, repo = application()
    original = deepcopy(case)
    app._run_service = SimpleNamespace(workflow_availability=lambda _: WorkflowAvailability(True, False, False, 0, 6, ()))
    with local_server(app) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        dialogs = {"accept": True, "messages": []}
        def answer_dialog(dialog):
            dialogs["messages"].append(dialog.message)
            dialog.accept() if dialogs["accept"] else dialog.dismiss()
        page.on("dialog", answer_dialog)
        try:
            page.goto(f"{origin}/test-cases/{case.id}")
            assert page.locator('body').inner_text().count('0 / 6 steps have executable plans.') == 1
            expect(page.get_by_role('link', name='View TestPlan', exact=True)).to_have_count(1)
            assert page.locator('select[name="cookie_policy"]').count() == 1
            assert page.locator('select[name="evidence_mode"]').count() == 1
            page.screenshot(path=str(tmp_path / 'testcase-desktop.png'), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(tmp_path / 'testcase-mobile.png'), full_page=True)
            page.set_viewport_size({"width": 1280, "height": 900})
            page.goto(f"{origin}/test-cases/{case.id}/edit")
            expect(page.locator('[data-structured-steps] fieldset')).to_have_count(6)
            expect(page.get_by_label("Description", exact=True).first).to_have_value(case.steps[0].description)
            assert page.locator('input[name$=".name"]').count() == 0
            assert_step_action_layout(page)
            expect(page.get_by_role('button', name='Regenerate TestCase with AI', exact=True)).to_have_count(0)
            dialogs["accept"] = False
            page.get_by_role('button', name='Delete Step 1', exact=True).focus()
            page.keyboard.press('Enter')
            expect(page.locator('[data-structured-steps] fieldset')).to_have_count(6)
            assert dialogs["messages"][-1].startswith('Delete this step?')
            assert repo.get(case.id) == original
            dialogs["accept"] = True
            page.get_by_role("button", name="Insert Step before 1", exact=True).focus()
            page.keyboard.press('Enter')
            expect(page.get_by_label("Description", exact=True).first).to_be_focused()
            page.get_by_label("Description", exact=True).first.fill("Відкрити налаштування облікового запису.\n" + "Довгий опис. " * 90)
            page.get_by_label("Expected Result", exact=True).first.fill("Налаштування доступні.")
            page.get_by_role("button", name="Move down Step 1", exact=True).focus()
            page.keyboard.press('Space')
            page.get_by_role("button", name="Delete Step 3", exact=True).click()
            expect(page.locator('[data-structured-steps] fieldset')).to_have_count(6)
            assert repo.get(case.id) == original
            page.get_by_role("link", name="Cancel unsaved changes", exact=True).click()
            expect(page).to_have_url(f"{origin}/test-cases/{case.id}")
            assert repo.get(case.id) == original

            page.goto(f"{origin}/test-cases/{case.id}/edit")
            other = browser.new_page()
            other.goto(page.url)
            page.get_by_role("button", name="Insert Step after 3", exact=True).click()
            page.get_by_label("Description", exact=True).nth(3).fill("Перевірити налаштування профілю.")
            page.get_by_label("Expected Result", exact=True).nth(3).fill("Налаштування профілю видимі.")
            page.get_by_label("Summary", exact=True).fill("Edited local account checks")
            page.get_by_role("button", name="Save TestCase", exact=True).click()
            expect(page).to_have_url(f"{origin}/test-cases/{case.id}")
            saved = repo.get(case.id)
            assert len(saved.steps) == 7 and saved.segments[1].id == original.segments[1].id
            assert saved.steps[0].id == original.steps[0].id
            assert [s.order for s in saved.steps] == list(range(7))
            other.get_by_label("Summary", exact=True).fill("Unsaved old-tab Summary")
            other.get_by_label("Description", exact=True).first.fill("Retained description from old tab.")
            other.get_by_role("button", name="Save TestCase", exact=True).click()
            expect(other.get_by_role("alert")).to_contain_text("changed elsewhere")
            expect(other.get_by_label("Summary", exact=True)).to_have_value("Unsaved old-tab Summary")
            expect(other.get_by_label("Description", exact=True).first).to_have_value("Retained description from old tab.")
            assert repo.get(case.id) == saved

            from qa_agent.test_case_authoring import TestCaseDraft
            token = app._draft_store.put(TestCaseDraft(test_case=case_fixture()))
            page.goto(f"{origin}/test-cases/review/{token}")
            assert_step_action_layout(page)
            expect(page.get_by_role('button', name='Regenerate TestCase with AI', exact=True)).to_be_visible()
            dialogs["accept"] = False
            page.get_by_role('button', name='Delete Step 1', exact=True).click()
            expect(page.locator('[data-structured-steps] fieldset')).to_have_count(6)
            dialogs["accept"] = True
            page.get_by_role("button", name="Insert Step before 1", exact=True).click()
            page.get_by_label("Description", exact=True).first.fill("Open the local preferences page.")
            page.get_by_label("Expected Result", exact=True).first.fill("Preferences are visible.")
            page.get_by_role("button", name="Save TestCase", exact=True).click()
            expect(page.get_by_role("button", name="Approve TestCase", exact=True)).to_be_visible()
            page.goto(f"{origin}/test-cases/{case.id}/edit")
            page.screenshot(path=str(tmp_path / "editor-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            actions = page.locator('[data-structured-editor] > .actions .button')
            heights = [actions.nth(i).bounding_box()["height"] for i in range(actions.count())]
            assert max(heights) - min(heights) < 2
            page.screenshot(path=str(tmp_path / "editor-mobile.png"), full_page=True)
        finally:
            browser.close(); app.close()


def test_draft_selection_preserves_explicit_summary_and_copies_unedited_title():
    app, _, _ = application()
    draft = Draft(title="Local account preferences", body="Open the preferences and inspect defaults.", base_url="http://127.0.0.1/settings")
    app._drafts.save(draft)
    with local_server(app) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(f"{origin}/test-cases/new")
            page.locator('[data-draft-select]').first.click()
            expect(page.locator('#case-name')).to_have_value(draft.title)
            other = Draft(title="Other local account settings", body="Inspect local settings.", base_url="http://127.0.0.1/other")
            app._drafts.save(other)
            page.goto(f"{origin}/test-cases/new?draft_id={draft.id}")
            page.locator(f'[data-draft-select][data-draft-id="{other.id}"]').click()
            expect(page.locator('#case-name')).to_have_value(other.title)
            page.locator('#case-name').fill("My explicitly edited Summary")
            page.locator('[data-draft-select]').first.click()
            expect(page.locator('#case-name')).to_have_value("My explicitly edited Summary")
            page.locator('#case-name').fill("")
            page.locator(f'[data-draft-select][data-draft-id="{draft.id}"]').click()
            expect(page.locator('#case-name')).to_have_value(draft.title)
        finally:
            browser.close(); app.close()


class ConsentFixture:
    def __init__(self, *, absent=False, uncertain=False): self.absent, self.uncertain = absent, uncertain
    def close(self): pass
    def handle(self, method, target, body=None, headers=None):
        if target == '/consent.js':
            return WebResponse(200, 'application/javascript', b'''
              if (document.cookie.includes('consent=stored')) document.getElementById('cookie-banner')?.remove();
              ['accept-cookies', 'reject-cookies'].forEach(id => document.getElementById(id)?.addEventListener('click', () => {
                document.cookie = 'consent=stored; path=/'; document.getElementById('cookie-banner').remove();
              }));
              window.visits = (window.visits || 0) + 1;
            ''')
        buttons = '<button id="reject-cookies">Reject optional cookies</button><button id="accept-cookies">Accept all cookies</button><button id="settings">Cookie settings</button>'
        markup = '' if self.absent else f'<div id="cookie-banner" role="dialog" aria-label="Cookie consent">Cookie consent choices{buttons}</div>'
        if self.uncertain: markup += '<div role="dialog" id="other-consent">Cookie consent <button>Accept all cookies</button></div>'
        return WebResponse.html(200, '<!doctype html><html><head><title>Consent fixture</title></head><body>' + markup + '<button id="main-action">Continue</button><script src="/consent.js"></script></body></html>')


def test_cookie_journey_discovery_uses_same_page_context_and_preserves_choices():
    with local_server(ConsentFixture()) as origin:
        case = Case(name="Cookie consent journey", description="Inspect consent and revisit the local page.", base_url=origin,
                    steps=[Step(name="Open page", description="Open page.", expected="The page is loaded.", order=0)])
        runner = BrowserRunner(headless=True)
        with cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED), runner.open_test_case_session(case) as session:
            navigate = QATestPlan(url=origin, steps=[{"action": "navigate", "parameters": {"url": origin}}, {"action": "assert_page_loaded"}])
            assert session.run_plan(0, navigate)["status"] == "passed"
            page = session._pages[0]
            context = page.context
            snapshot = session.capture_discovery(0)
            assert any(item.selector == "#cookie-banner" for item in snapshot.interactive_elements)
            assert page.evaluate("window.visits") == 1
            assert session.run_plan(0, QATestPlan(url=origin, steps=[{"action": "assert_visible", "parameters": {"selector": "#cookie-banner"}}]))["status"] == "passed"
            assert session.run_plan(0, QATestPlan(url=origin, steps=[{"action": "click", "parameters": {"selector": "#reject-cookies"}}]))["status"] == "passed"
            after = session.capture_discovery(0)
            assert not any(item.selector == "#cookie-banner" for item in after.interactive_elements)
            assert session._pages[0] is page and page.context is context
            assert page.evaluate("window.visits") == 1 and context.cookies()[0]["value"] == "stored"
            assert session.run_plan(0, QATestPlan(url=origin, steps=[{"action": "assert_enabled", "parameters": {"selector": "#main-action"}}]))["status"] == "passed"
            assert session.run_plan(0, navigate)["status"] == "passed"
            assert page.locator('#cookie-banner').count() == 0
            assert page.context is context


def test_generation_pipeline_observes_each_live_step_without_a_fresh_probe():
    with local_server(ConsentFixture()) as origin:
        requirements = [
            ("Open local homepage.", "The page is loaded."),
            ("Inspect cookie banner.", "Cookie banner is visible on the page."),
            ("Click Reject optional cookies.", "Reject optional cookies clicked."),
            ("Inspect main action.", "The main action is enabled."),
            ("Open local homepage again.", 'The page title is "Consent fixture".'),
        ]
        case = Case(name="Consent across five steps", description="Inspect consent options, reject optional cookies, and revisit the page.", base_url=origin,
            steps=[Step(name=description, description=description, expected=expected, order=i) for i, (description, expected) in enumerate(requirements)])
        class Provider(LLMProvider):
            def __init__(self): self.calls = 0
            def create_test_plan(self, task, target_url, page_snapshot):
                candidates = [
                    [{"action": "navigate", "parameters": {"url": origin}}, {"action": "assert_page_loaded"}],
                    [{"action": "assert_visible", "parameters": {"selector": "#cookie-banner"}}],
                    [{"action": "click", "parameters": {"selector": "#reject-cookies"}}],
                    [{"action": "assert_enabled", "parameters": {"selector": "#main-action"}}],
                    [{"action": "navigate", "parameters": {"url": origin}}, {"action": "assert_title", "parameters": {"expected": "Consent fixture"}}],
                ]
                actions = candidates[self.calls]; self.calls += 1
                return QATestPlan(url=target_url, steps=actions)
        provider = Provider()
        probes, observations = [], []
        def initial_probe(url):
            probes.append(url)
            return DiscoveryResult(status=DiscoveryStatus.PARTIAL, url=url)
        def observe(page):
            observations.append((id(page), id(page.context), page.locator('#cookie-banner').count(), page.context.cookies()))
            return capture_current_page_discovery(page)
        pipeline = QATestPipeline(None, LLMTestPlanGenerator(LLMRouter([provider])), discovery=initial_probe, runner=BrowserRunner(headless=True))
        with cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED), patch('qa_agent.browser_discovery.capture_current_page_discovery', side_effect=observe):
            result = pipeline.run_test_case(case)
        assert provider.calls == 5 and len(probes) == 1
        assert len(observations) == 4
        assert len({(row[0], row[1]) for row in observations}) == 1
        assert [row[2] for row in observations] == [1, 1, 0, 0]
        assert observations[-1][3][0]["value"] == "stored"
        assert all(execution.status == ExecutionStatus.PASSED for execution in result.executions)


@pytest.mark.parametrize("absent,uncertain,stored,status", [(False, False, False, "HANDLED"),
    (True, False, False, "NO_BANNER"), (False, True, False, "REQUIRES_ATTENTION"), (False, False, True, "NO_BANNER")])
def test_default_consent_policy_visible_absent_uncertain_and_stored(absent, uncertain, stored, status):
    with local_server(ConsentFixture(absent=absent, uncertain=uncertain)) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        try:
            if stored: context.add_cookies([{"name": "consent", "value": "stored", "url": origin}])
            page = context.new_page(); page.goto(origin)
            before = capture_current_page_discovery(page)
            with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE): record = handle_cookie_consent(page)
            assert record.status.value == status
            if status == "HANDLED":
                assert len([item for item in before.interactive_elements if item.tag == "button"]) >= 4
                assert page.locator('#cookie-banner').count() == 0
            if uncertain: assert page.locator('#cookie-banner').count() == 1
            if absent or stored: assert not any(item.selector == "#cookie-banner" for item in before.interactive_elements)
        finally: browser.close()
