"""Native editor smoke: rejected candidate -> edit -> validate -> save -> review."""
from playwright.sync_api import expect, sync_playwright
from urllib.parse import urlencode

from tests.test_execution_control_web import local_server
from tests.test_manual_recovery import recovery


def test_browser_recovery_edit_validate_save_and_review_without_provider_requests(recovery):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            with local_server(recovery.app) as origin:
                page.goto(f'{origin}/settings/reliability/{recovery.record.id}')
                page.get_by_role('link', name='Edit rejected candidate', exact=True).click()
                expect(page.get_by_text('AI-generated rejected candidate', exact=False)).to_be_visible()
                page.locator('input[name="action.1.param.expected"]').fill('declared')
                page.get_by_role('button', name='Add action', exact=True).click()
                added = page.locator('[data-action-card]').nth(2)
                added.locator('[data-action-type]').select_option('assert_hidden')
                added.locator('[data-action-param="selector"]').fill('#form-error')
                added.locator('[data-action-move="up"]').click()
                page.locator('[data-action-card]').nth(1).get_by_role('button', name='Delete', exact=True).click()
                expect(page.locator('[data-action-card]')).to_have_count(2)
                data = page.locator('[data-automation-form]').evaluate('form => Object.fromEntries(new FormData(form))')
                assert data.get('_csrf') == recovery.app._csrf_token
                assert recovery.app._valid_control_request(urlencode(data), {'Host': origin.removeprefix('http://'), 'Origin': origin})
                page.get_by_role('button', name='Validate locally', exact=True).click()
                expect(page.get_by_text('All mandatory Quality Gates passed.', exact=False)).to_be_visible()
                assert recovery.plans.find(recovery.step.id) is None
                page.get_by_role('button', name='Save validated manual version', exact=True).click()
                expect(page.get_by_text('Automation saved.', exact=True)).to_be_visible()
                assert recovery.plans.find(recovery.step.id).manual_recovery.operation_id == recovery.record.id
                assert not recovery.app._test_case_review.validation_approved_for(recovery.case)
                page.locator('.success-state').get_by_role('link', name='Back to TestCase', exact=True).click()
                expect(page.get_by_role('button', name='Approve for Validation', exact=True)).to_be_visible()
                assert len(recovery.provider.calls) == 1
        finally:
            browser.close()
