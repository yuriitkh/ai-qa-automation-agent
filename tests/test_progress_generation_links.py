"""Generation links use in-memory progress and local Chromium only."""
from html import escape
from uuid import uuid4

import pytest
from playwright.sync_api import expect, sync_playwright

from qa_agent.execution_progress import (
    ExecutionEventType, ExecutionProgressReporter, ExecutionProgressStore, ProgressStep,
)
from qa_agent.models import TestCase as Case, TestStep as Step
from qa_agent.reliability import AutomationReliabilitySupervisor
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.web import LocalWebApplication
from tests.test_execution_control_web import local_server


@pytest.fixture
def progress():
    case = Case(name='Registration', description='Register.', steps=[
        Step(name='Open registration page', description='Open.', expected='Form visible.', order=0),
        Step(name='Enter registration data', description='Enter.', expected='Data entered.', order=1),
    ])
    store = ExecutionProgressStore()
    progress_id = store.create(case.id, 'AUTOMATION')
    reporter = ExecutionProgressReporter(store, progress_id)
    reporter.test_case_loaded(case)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()),
                              progress_store=store, reliability=AutomationReliabilitySupervisor())
    try:
        yield app, store, reporter, case, progress_id
    finally:
        app.close()


def emit(store, progress_id, operation, **metadata):
    store.append_for(progress_id, ExecutionEventType.AUTOMATION_PREPARATION_STARTED,
                     reliability_operation_id=operation, **metadata)


def payload(app, store, progress_id):
    return app._progress_payload(store.get(progress_id))


def all_links(data):
    return [*data['generation_decisions'],
            *(link for step in data['steps'] for link in step['generation_decisions'])]


@pytest.mark.parametrize('finished', [False, True])
def test_per_step_labels_urls_and_no_duplicate_operations(progress, finished):
    app, store, reporter, case, progress_id = progress
    operations = [uuid4(), uuid4()]
    # Events arrive in a different order from TestSteps, with repeated IDs.
    for index in (1, 0, 1, 0):
        reporter.emit(ExecutionEventType.AUTOMATION_PREPARATION_STARTED,
                      step=case.steps[index], reliability_operation_id=operations[index])
    if finished:
        reporter.finish(outcome='AUTOMATION_GENERATION_ERROR')
    data = payload(app, store, progress_id)
    html = app.handle('GET', f'/runs/progress/{progress_id}').body.decode()
    rows = html.split('<li class="progress-step">')[1:]
    for index, operation in enumerate(operations):
        label = f'Generation decisions — Step {index + 1}: {case.steps[index].name}'
        url = f'/settings/reliability/{operation}'
        assert data['steps'][index]['generation_decisions'] == [{'label': label, 'url': url}]
        assert html.count(f'href="{url}"') == 1
        assert label in rows[index].split('</li>')[0]
    assert data['generation_decisions'] == []
    assert len(all_links(data)) == 2
    assert not any('/settings/reliability/' in action['url'] for action in data['actions'])


def test_ids_override_stale_order_and_late_event_supplies_missing_association(progress):
    app, store, _, case, progress_id = progress
    operation = uuid4()
    emit(store, progress_id, operation)
    emit(store, progress_id, operation, step_order=0, step_name='Stale name')
    emit(store, progress_id, operation, step_id=case.steps[1].id, step_order=0, step_name='Stale name')
    data = payload(app, store, progress_id)
    assert data['steps'][0]['generation_decisions'] == []
    assert data['steps'][1]['generation_decisions'][0]['label'].endswith('Step 2: Enter registration data')
    assert data['generation_decisions'] == [] and len(all_links(data)) == 1


def test_distinct_operations_for_one_step_are_all_preserved(progress):
    app, store, _, case, progress_id = progress
    operations = [uuid4(), uuid4()]
    for operation in operations:
        for _ in range(2):
            emit(store, progress_id, operation, step_id=case.steps[0].id)
    data = payload(app, store, progress_id)
    assert [link['url'] for link in data['steps'][0]['generation_decisions']] == [
        f'/settings/reliability/{operation}' for operation in operations
    ]
    assert len(all_links(data)) == 2


@pytest.mark.parametrize('metadata,location,label', [
    ({'step_order': 1}, 1, 'Step 2: Enter registration data'),
    ({}, None, 'Step ?: TestStep unavailable'),
    ({'step_order': 7, 'step_name': 'Removed step'}, None, 'Step 8: Removed step'),
    ({'step_order': -1}, None, 'Step ?: TestStep unavailable'),
])
def test_missing_step_information_has_explicit_fallback(progress, metadata, location, label):
    app, store, _, _, progress_id = progress
    operation = uuid4()
    emit(store, progress_id, operation, **metadata)
    data = payload(app, store, progress_id)
    links = data['generation_decisions'] if location is None else data['steps'][location]['generation_decisions']
    assert links == [{'label': 'Generation decisions — ' + label,
                      'url': f'/settings/reliability/{operation}'}]
    html = app.handle('GET', f'/runs/progress/{progress_id}').body.decode()
    assert html.count(f'href="/settings/reliability/{operation}"') == 1


def test_unknown_id_is_not_attached_to_an_unrelated_step_with_same_order(progress):
    app, store, _, _, progress_id = progress
    emit(store, progress_id, uuid4(), step_id=uuid4(), step_order=0, step_name='Removed step')
    data = payload(app, store, progress_id)
    assert all(step['generation_decisions'] == [] for step in data['steps'])
    assert data['generation_decisions'][0]['label'].endswith('Step 1: Removed step')


def test_ambiguous_order_does_not_guess_step_identity(progress):
    app, store, _, case, progress_id = progress
    store.register_test_case(progress_id, case.id, case.name, [
        ProgressStep(id=step.id, order=0, name=step.name) for step in case.steps
    ])
    emit(store, progress_id, uuid4(), step_order=0)
    data = payload(app, store, progress_id)
    assert all(step['generation_decisions'] == [] for step in data['steps'])
    assert len(data['generation_decisions']) == 1


@pytest.mark.parametrize('associated', [False, True])
def test_long_and_html_step_names_are_escaped_without_losing_link(progress, associated):
    app, store, _, case, progress_id = progress
    name = '<img src=x onerror="window.injected=1"> & \'quoted\' ' + 'x' * 300
    if associated:
        store.register_test_case(progress_id, case.id, case.name, [
            ProgressStep(id=case.steps[0].id, order=0, name=name),
        ])
    operation = uuid4()
    emit(store, progress_id, operation, step_id=case.steps[0].id if associated else uuid4(),
         step_order=0, step_name=name)
    data = payload(app, store, progress_id)
    assert all_links(data)[0]['label'] == f'Generation decisions — Step 1: {name}'
    html = app.handle('GET', f'/runs/progress/{progress_id}').body.decode()
    assert escape(name) in html and '<img src=x' not in html
    assert html.count(f'href="/settings/reliability/{operation}"') == 1


def test_empty_step_name_gets_a_readable_label(progress):
    app, store, _, case, progress_id = progress
    store.register_test_case(progress_id, case.id, case.name, [
        ProgressStep(id=case.steps[0].id, order=0, name='   '),
    ])
    emit(store, progress_id, uuid4(), step_id=case.steps[0].id)
    assert all_links(payload(app, store, progress_id))[0]['label'].endswith('Step 1: Unnamed TestStep')


def test_no_reliability_service_does_not_offer_unavailable_links(progress):
    app, store, _, _, progress_id = progress
    emit(store, progress_id, uuid4())
    app._reliability = None
    assert all_links(payload(app, store, progress_id)) == []


def test_live_polling_moves_fallback_link_without_duplicates_and_wraps_long_names(progress):
    app, store, reporter, case, progress_id = progress
    name = '<img src=x onerror="window.injected=1"> & ' + 'x' * 300
    store.register_test_case(progress_id, case.id, case.name, [
        ProgressStep(id=case.steps[0].id, order=0, name=name),
        ProgressStep(id=case.steps[1].id, order=1, name=case.steps[1].name),
    ])
    known, missing = uuid4(), uuid4()
    emit(store, progress_id, known, step_id=case.steps[1].id)
    emit(store, progress_id, missing)
    with local_server(app) as origin, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 360, 'height': 800})
        try:
            page.goto(f'{origin}/runs/progress/{progress_id}')
            fallback = page.locator('[data-progress-unassigned-decisions] a')
            expect(fallback).to_have_text('Generation decisions — Step ?: TestStep unavailable')
            expect(page.locator('.progress-step').nth(1).get_by_role('link', name='Generation decisions — Step 2: Enter registration data')).to_have_attribute('href', f'/settings/reliability/{known}')
            emit(store, progress_id, missing, step_id=case.steps[0].id)
            emit(store, progress_id, missing, step_id=case.steps[0].id)
            link = page.locator('.progress-step').nth(0).locator(f'a[href="/settings/reliability/{missing}"]')
            expect(link).to_have_text(f'Generation decisions — Step 1: {name}')
            expect(fallback).to_have_count(0)
            reporter.finish(outcome='AUTOMATION_GENERATION_ERROR')
            expect(page.locator('[data-progress-result]')).to_be_visible()
            expect(page.locator('a[href^="/settings/reliability/"]')).to_have_count(2)
            assert page.locator('#progress-steps img').count() == 0
            assert page.evaluate('window.injected') is None
            assert link.evaluate('(node) => getComputedStyle(node).overflowWrap') == 'anywhere'
            assert page.locator('#progress-steps').evaluate('(node) => node.scrollWidth <= node.clientWidth')
        finally:
            browser.close()
