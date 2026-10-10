"""M3 local recovery uses synthetic candidates and isolated plan/history stores."""
from datetime import datetime, timedelta, timezone
import io
import json
from types import SimpleNamespace
from urllib.parse import urlencode
from uuid import uuid4
import zipfile

import pytest

from qa_agent.automation_lifecycle import AutomationStatus, AutomationLifecycleService, InMemoryAutomationLifecycleRepository
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.execution_progress import ExecutionProgressReporter, active_execution_progress
from qa_agent.llm.router import LLMRouter
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm_usage import llm_usage_scope
from qa_agent.manual_recovery import ManualRecoveryStore, MAX_RECOVERY_BYTES, MAX_RECOVERY_ENTRIES, RecoveryUnavailable
from qa_agent.models import TestCase as Case, TestPlan as Plan, TestPlanVersion as Version, PlanVersionOrigin
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.reliability import AutomationReliabilitySupervisor, ReliabilityRecord, ReliabilitySettings
from qa_agent.redaction import register_secret
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService, WorkflowType
from qa_agent.sqlite_storage import SQLitePlanStore
from qa_agent.test_case_execution import TestCaseExecutionService as ExecutionService, RunUnavailableError
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.test_case_review import TestCaseReviewService as ReviewService, InMemoryTestCaseReviewRepository
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.web import LocalWebApplication
from tests.test_automation_reliability import LocalProvider
from tests.test_diagnostic_mode import input_step, candidate
from tests.test_generation_reliability import action, observed_registration, plan


@pytest.fixture(params=['memory', 'sqlite'])
def recovery(request, tmp_path):
    step = input_step()
    case = Case(name='Recovery <fixture>', description=step.description,
                base_url=observed_registration().url, steps=[step])
    cases = InMemoryTestCaseRepository()
    cases.save(case)
    plans = SQLitePlanStore(tmp_path / 'plans.sqlite3') if request.param == 'sqlite' else InMemoryPlanStore()
    supervisor = AutomationReliabilitySupervisor()
    supervisor.repository.save_settings(ReliabilitySettings(diagnostic_level='DEBUG'))
    executions = InMemoryExecutionRepository()
    history = RunHistoryService(InMemoryRunHistoryRepository(), executions, plans)
    lifecycle = AutomationLifecycleService(InMemoryAutomationLifecycleRepository(), plans)
    review = ReviewService(InMemoryTestCaseReviewRepository(), plans)
    execution = ExecutionService(cases, plans, executions, history, automation_lifecycle=lifecycle, test_case_review=review,
                                 runner_factory=lambda _: lambda plan: {'status': 'failed', 'steps': []})
    app = LocalWebApplication(history, test_cases=cases, plan_store=plans, reliability=supervisor, run_service=execution,
                              automation_lifecycle=lifecycle, test_case_review=review,
                              recovery_discovery=lambda url: observed_registration())
    provider = LocalProvider([candidate(expected='wrong')])
    generator = LLMTestPlanGenerator(LLMRouter([provider]), supervisor)
    progress_id = app.progress_store.create(case.id, 'AUTOMATION')
    reporter = ExecutionProgressReporter(app.progress_store, progress_id)
    reporter.test_case_loaded(case)
    with active_execution_progress(reporter), llm_usage_scope(related_test_case_id=case.id), pytest.raises(PlanValidationError):
        generator.generate_with_plan(step, observed_registration(), requirement_context=case.description)
    record = supervisor.repository.list_records()[0]
    base = f'/test-cases/{case.id}/automation/steps/{step.id}/recovery/{record.id}'
    def forbidden(*args, **kwargs):
        raise AssertionError('Recovery must never invoke generation')
    provider.create_test_plan = forbidden
    try:
        yield SimpleNamespace(app=app, case=case, step=step, cases=cases, plans=plans, supervisor=supervisor, database=tmp_path / 'plans.sqlite3',
                              provider=provider, record=record, base=base, progress_id=progress_id)
    finally:
        app.close()


def form(fixture, *, proposed=None, expected_version=0, revision=None):
    entry = fixture.app._recovery_store.get(fixture.record.id, fixture.case.id, fixture.step.id)
    values = {'_csrf': fixture.app._csrf_token, 'recovery_revision': str((entry.revision if entry else 0) if revision is None else revision),
              'expected_version': str(expected_version), 'plan_url': observed_registration().url}
    proposed = proposed or candidate()
    rows = proposed.steps if hasattr(proposed, 'steps') else proposed
    for index, item in enumerate(rows):
        item = item.model_dump() if hasattr(item, 'model_dump') else item
        values[f'action.{index}.type'] = item['action']
        values.update({f'action.{index}.param.{key}': value for key, value in item['parameters'].items()})
    return values


def post(fixture, operation='validate', **kwargs):
    return fixture.app.handle('POST', fixture.base + '/' + operation, urlencode(form(fixture, **kwargs)))


def test_rejected_candidate_is_editable_and_bound_to_safe_step_links(recovery):
    page = recovery.app.handle('GET', recovery.base)
    assert page.status == 200 and page.headers['Cache-Control'] == 'no-store'
    assert b'AI-generated rejected candidate' in page.body
    assert b'value="wrong"' in page.body and b'FILL_ASSERT_VALUE_MISMATCH' in page.body
    assert b'Original requirements' in page.body and b'&lt;fixture&gt;' in page.body
    decisions = recovery.app.handle('GET', f'/settings/reliability/{recovery.record.id}').body.decode()
    assert decisions.count('>Edit rejected candidate</a>') == 1 and recovery.base in decisions
    progress = recovery.app.handle('GET', f'/runs/progress/{recovery.progress_id}').body.decode()
    assert progress.count(f'href="{recovery.base}"') == 1
    assert len(recovery.provider.calls) == 1 and recovery.plans.find(recovery.step.id) is None


def test_validate_then_save_immutable_manual_version_requires_explicit_review(recovery):
    preview = post(recovery)
    assert preview.status == 200 and b'All mandatory Quality Gates passed' in preview.body
    assert recovery.plans.find(recovery.step.id) is None
    saved = post(recovery, 'save')
    assert saved.status == 303
    version = recovery.plans.find(recovery.step.id)
    assert version.origin == PlanVersionOrigin.HUMAN_EDITED and version.manual_recovery.operation_id == recovery.record.id
    assert version.manual_recovery.previous_version_id is None and version.assertion_grounding
    assert recovery.app._automation_lifecycle.status(recovery.case) == AutomationStatus.NEEDS_VALIDATION
    assert not recovery.app._test_case_review.validation_approved_for(recovery.case)
    completed = recovery.app._recovery_store.get(recovery.record.id, recovery.case.id, recovery.step.id)
    assert completed.candidate is None and completed.draft is None
    if isinstance(recovery.plans, SQLitePlanStore):
        assert SQLitePlanStore(recovery.database).get_version(version.id) == version
    assert len(recovery.provider.calls) == 1


@pytest.mark.parametrize('proposed,reason', [
    ([action('evaluate', '#email', value='bad')], 'Choose a supported action'),
    (candidate(expected='wrong'), 'FILL_ASSERT_VALUE_MISMATCH'),
    (candidate(value='undeclared', expected='undeclared'), 'VALUE_NOT_DECLARED_IN_REQUIREMENT'),
    (plan(action('fill', '#invented', value='declared'), action('assert_value', '#invented', expected='declared')), 'ACTION_TARGET_MISMATCH'),
    (plan(action('fill', '#email', value='declared'), action('assert_value', '#email', expected='declared'), action('click', '#invented')), 'DISCOVERY_SELECTOR_MISMATCH'),
    (plan(action('fill', '#email', value='declared'), action('assert_value', '#email', expected='declared'), action('click', '#submit')), 'TESTSTEP_BOUNDARY_VIOLATION'),
    (plan(action('fill', '#email', value='declared')), 'EXPECTED_RESULT_NOT_COVERED'),
])
def test_edits_cannot_bypass_any_mandatory_gate(recovery, proposed, reason):
    response = post(recovery, 'save', proposed=proposed)
    assert response.status == 400 and reason in response.body.decode()
    assert recovery.plans.find(recovery.step.id) is None
    assert recovery.app._automation_lifecycle.status(recovery.case) != AutomationStatus.AUTOMATION_READY
    assert len(recovery.provider.calls) == 1


def test_action_add_remove_reorder_and_edit_are_revalidated(recovery):
    first = plan(action('assert_hidden', '#form-error'), action('fill', '#email', value='declared'), action('assert_value', '#email', expected='declared'))
    assert post(recovery, proposed=first).status == 200
    reordered = plan(action('assert_value', '#email', expected='declared'), action('fill', '#email', value='declared'))
    assert b'MISSING_PRECEDING_FILL' in post(recovery, proposed=reordered).body
    assert post(recovery, 'save', proposed=candidate()).status == 303
    assert [item.action for item in recovery.plans.find(recovery.step.id).qa_test_plan.steps] == ['fill', 'assert_value']


def test_approved_previous_version_is_never_overwritten(recovery):
    old_plan = Plan(test_step_id=recovery.step.id, name=recovery.step.name)
    edited, grounding = LLMTestPlanGenerator._validate_generated_plan(candidate(), observed_registration(), recovery.step, requirement_context=recovery.case.description)
    old = Version(test_plan_id=old_plan.id, version=1, qa_test_plan=edited, assertion_grounding=grounding)
    recovery.plans.save(recovery.step.id, old, test_plan=old_plan)
    recovery.app._test_case_review.approve_for_validation(recovery.case)
    old_approval = recovery.app._test_case_review.record(recovery.case.id).approved_plan_fingerprint
    assert post(recovery, 'save', expected_version=1).status == 303
    current = recovery.plans.find(recovery.step.id)
    assert current.version == 2 and current.manual_recovery.previous_version_id == old.id
    assert recovery.plans.get_version(old.id) == old
    assert recovery.app._test_case_review.record(recovery.case.id).approved_plan_fingerprint == old_approval
    assert not recovery.app._test_case_review.validation_approved_for(recovery.case)
    assert post(recovery, 'save', expected_version=1).status == 409


def test_historical_candidate_is_empty_and_requires_real_discovery(recovery):
    recovery.app._recovery_store.discard(recovery.record.id)
    html = recovery.app.handle('GET', recovery.base).body.decode()
    assert 'Empty manual recovery draft' in html and 'historical candidate is unavailable' in html
    assert 'data-action-card' not in html
    assert 'Create manual recovery draft' in recovery.app.handle('GET', f'/settings/reliability/{recovery.record.id}').body.decode()
    assert b'deterministic Browser Discovery is required' in post(recovery).body
    assert post(recovery, 'discover').status == 303
    assert post(recovery, 'save').status == 303
    assert len(recovery.provider.calls) == 1


@pytest.mark.parametrize('tamper', ['case', 'step', 'operation', 'accepted', 'missing-csrf', 'stale-revision', 'duplicate-field', 'url'])
def test_identifiers_csrf_revision_and_url_tampering_are_rejected(recovery, tamper):
    values = form(recovery)
    target = recovery.base
    if tamper == 'case':
        target = target.replace(str(recovery.case.id), str(uuid4()))
    elif tamper == 'step':
        target = target.replace(str(recovery.step.id), str(uuid4()))
    elif tamper == 'operation':
        other = ReliabilityRecord(test_step_id=recovery.step.id, test_case_id=uuid4(), settings=ReliabilitySettings(), started_at=datetime.now(timezone.utc), outcome='NEEDS_ATTENTION')
        recovery.supervisor.repository.save(other)
        target = target.replace(str(recovery.record.id), str(other.id))
    elif tamper == 'accepted':
        recovery.record.outcome = 'READY_FOR_REVIEW'
        recovery.supervisor.repository.save(recovery.record)
    elif tamper == 'missing-csrf':
        values.pop('_csrf')
    elif tamper == 'stale-revision':
        values['recovery_revision'] = '999'
    elif tamper == 'url':
        values['plan_url'] = 'javascript:alert(1)'
    body = urlencode(values) + ('&expected_version=0' if tamper == 'duplicate-field' else '')
    response = recovery.app.handle('POST', target + '/save', body)
    assert response.status in {400, 403, 404, 409}
    assert recovery.plans.find(recovery.step.id) is None


def test_changed_original_requirements_and_generated_wording_cannot_authorize_values(recovery):
    changed = recovery.case.model_copy(deep=True)
    changed.description = 'Enter an email address.'
    recovery.cases.save(changed)
    assert post(recovery, 'save').status == 409
    assert recovery.plans.find(recovery.step.id) is None


@pytest.mark.parametrize('format', ['json', 'zip'])
def test_private_candidate_is_never_in_diagnostic_exports(recovery, format):
    assert post(recovery, proposed=candidate(expected='private-edited-mismatch')).status == 400
    response = recovery.app.handle('GET', f'/settings/reliability/{recovery.record.id}/diagnostics.{format}')
    assert response.status == 200
    raw = response.body
    if format == 'zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            raw = archive.read('diagnostics.json') + archive.read('summary.txt')
    assert b'private-edited-mismatch' not in raw and b'"wrong"' not in raw
    assert recovery.plans.find(recovery.step.id) is None


def test_secret_and_password_values_removed_and_html_escaped(recovery):
    secret = 'synthetic-configured-key-M3'
    register_secret(secret)
    entry = recovery.app._recovery_store.get(recovery.record.id, recovery.case.id, recovery.step.id)
    recovery.app._recovery_store.capture(recovery.record, plan(action('fill', '#password', value='private-password'),
        action('assert_value', '#password', expected='private-password'), action('fill', '#email', value=secret)))
    safe = recovery.app._recovery_store.get(recovery.record.id, recovery.case.id, recovery.step.id)
    raw = json.dumps(safe.candidate)
    assert 'private-password' not in raw and secret not in raw
    response = post(recovery, proposed=candidate(value=secret, expected=secret))
    assert response.status == 400 and secret.encode() not in response.body
    response = post(recovery, proposed=candidate(value='<script>alert(1)</script>', expected='<script>alert(1)</script>'))
    assert response.status == 400 and b'<script>alert(1)</script>' not in response.body and b'&lt;script&gt;alert(1)&lt;/script&gt;' in response.body


def test_discard_and_storage_failure_do_not_mutate_plans_or_cases(recovery, monkeypatch):
    original = recovery.cases.get(recovery.case.id).model_dump_json()
    def unavailable(*args, **kwargs):
        raise OSError('private storage failure')
    monkeypatch.setattr(recovery.app._recovery_store, 'update', unavailable)
    response = post(recovery, 'save')
    assert response.status == 503 and b'private storage failure' not in response.body
    assert recovery.plans.find(recovery.step.id) is None and recovery.cases.get(recovery.case.id).model_dump_json() == original
    assert post(recovery, 'discard').status == 303
    assert recovery.app._recovery_store.get(recovery.record.id, recovery.case.id, recovery.step.id) is None


def test_unapproved_recovery_cannot_execute_and_failed_browser_validation_is_not_ready(recovery):
    assert post(recovery, 'save').status == 303
    calls = []
    execution = ExecutionService(recovery.cases, recovery.plans, InMemoryExecutionRepository(), recovery.app._run_history,
        automation_lifecycle=recovery.app._automation_lifecycle, test_case_review=recovery.app._test_case_review,
        runner_factory=lambda _: lambda plan: calls.append(plan) or {'status': 'failed', 'steps': []})
    assert not execution.workflow_availability(recovery.case.id).validation_available
    with pytest.raises(RunUnavailableError):
        execution.run(recovery.case.id, WorkflowType.VALIDATION)
    assert calls == []
    recovery.app._test_case_review.approve_for_validation(recovery.case)
    assert execution.workflow_availability(recovery.case.id).validation_available
    outcome = execution.run(recovery.case.id, WorkflowType.VALIDATION)
    assert len(calls) == 1 and outcome.test_run.status.value == 'FAILED'
    assert recovery.app._automation_lifecycle.status(recovery.case) == AutomationStatus.NEEDS_VALIDATION


def test_recovery_store_limits_expiry_and_cross_binding():
    now = datetime.now(timezone.utc)
    store = ManualRecoveryStore(clock=lambda: now)
    step = input_step()
    case = Case(name='Fixture', description=step.description, steps=[step])
    operations = [uuid4() for _ in range(MAX_RECOVERY_ENTRIES + 2)]
    for operation in operations:
        store.empty(operation, case, step)
        now += timedelta(seconds=1)
    assert store.get(operations[0], case.id, step.id) is None
    assert store.get(operations[-1], uuid4(), step.id) is None
    assert store.get(operations[-1], case.id, uuid4()) is None
    entry = store.get(operations[-1], case.id, step.id)
    with pytest.raises(RecoveryUnavailable):
        store.update(entry, entry.revision, draft={'steps': [], 'url': 'x' * (MAX_RECOVERY_BYTES + 1)})
    assert store.get(entry.operation_id, case.id, step.id).revision == 0
    now += timedelta(hours=2)
    assert store.get(entry.operation_id, case.id, step.id) is None


def test_generation_capture_is_opt_in_and_normal_success_is_unchanged():
    step = input_step()
    supervisor = AutomationReliabilitySupervisor()
    generator = LLMTestPlanGenerator(LLMRouter([LocalProvider([candidate()])]), supervisor)
    accepted = generator.generate_with_plan(step, observed_registration())
    assert accepted.test_plan_version.manual_recovery is None and supervisor.recovery_store is None
    supervisor.recovery_store = ManualRecoveryStore()
    generator = LLMTestPlanGenerator(LLMRouter([LocalProvider([candidate()])]), supervisor)
    with llm_usage_scope(related_test_case_id=uuid4()):
        generator.generate_with_plan(step, observed_registration())
    assert supervisor.recovery_store._entries == {}


def test_saved_manual_version_edits_cannot_bypass_recovery_gates(recovery):
    assert post(recovery, 'save').status == 303
    first = recovery.plans.find(recovery.step.id)
    # The older schema-only editor endpoint must not erase recovery provenance.
    response = recovery.app.handle('POST', f'/test-cases/{recovery.case.id}/automation/steps/{recovery.step.id}/save',
                                   urlencode(form(recovery, expected_version=1)))
    assert response.status == 409 and recovery.plans.find(recovery.step.id) == first
    editor = recovery.app.handle('GET', f'/test-cases/{recovery.case.id}/automation/edit').body.decode()
    assert recovery.base + '/save' in editor
    assert post(recovery, 'save', expected_version=1, proposed=candidate(expected='another-mismatch')).status == 400
    assert recovery.plans.find(recovery.step.id) == first
    assert post(recovery, 'save', expected_version=1).status == 303
    second = recovery.plans.find(recovery.step.id)
    assert second.version == 2 and second.manual_recovery.previous_version_id == first.id
    assert recovery.plans.get_version(first.id) == first


def test_existing_other_case_and_other_operation_cannot_access_candidate(recovery):
    other = recovery.case.model_copy(update={'id': uuid4(), 'public_id': None})
    recovery.cases.save(other)
    path = recovery.base.replace(str(recovery.case.id), str(other.id))
    assert recovery.app.handle('GET', path).status == 404
    record = ReliabilityRecord(test_step_id=recovery.step.id, test_case_id=recovery.case.id,
        settings=ReliabilitySettings(), started_at=datetime.now(timezone.utc), outcome='NEEDS_ATTENTION')
    recovery.supervisor.repository.save(record)
    recovery.app._recovery_store.prepare(record, recovery.step, observed_registration(), recovery.case.description, None)
    recovery.app._recovery_store.capture(record, candidate(expected='second-operation-only'))
    other_path = recovery.base.replace(str(recovery.record.id), str(record.id))
    assert b'second-operation-only' in recovery.app.handle('GET', other_path).body
    assert b'second-operation-only' not in recovery.app.handle('GET', recovery.base).body


def test_missing_authoritative_source_is_filled_from_original_case_not_generated_step(recovery):
    entry = recovery.app._recovery_store.get(recovery.record.id, recovery.case.id, recovery.step.id)
    entry.requirement_context = ''
    recovery.app._recovery_store._put(entry)
    original = recovery.case.model_copy(deep=True)
    original.description = 'Enter an email address.'
    recovery.cases.save(original)
    response = post(recovery, 'save')
    assert response.status == 400 and b'VALUE_NOT_DECLARED_IN_REQUIREMENT' in response.body
    assert recovery.plans.find(recovery.step.id) is None


def test_draft_is_not_an_exportable_plan_and_overlarge_action_lists_are_rejected(recovery):
    assert post(recovery).status == 200
    exported = recovery.app.handle('GET', f'/test-cases/{recovery.case.id}/export/json')
    assert exported.status != 200
    assert recovery.plans.find(recovery.step.id) is None
    entry = recovery.app._recovery_store.get(recovery.record.id, recovery.case.id, recovery.step.id)
    with pytest.raises(RecoveryUnavailable):
        recovery.app._recovery_store.update(entry, entry.revision, draft={'url': observed_registration().url, 'steps': [{}] * 65})


def test_recovery_capture_failure_never_changes_generation_rejection(recovery, monkeypatch):
    def unavailable(*args):
        raise OSError('private capture failure')
    monkeypatch.setattr(recovery.app._recovery_store, 'capture', unavailable)
    generator = LLMTestPlanGenerator(LLMRouter([LocalProvider([candidate(expected='mismatch')])]), recovery.supervisor)
    with llm_usage_scope(related_test_case_id=recovery.case.id), pytest.raises(PlanValidationError) as caught:
        generator.generate_with_plan(recovery.step, observed_registration(), requirement_context=recovery.case.description)
    assert caught.value.issues[0].reason_code == 'FILL_ASSERT_VALUE_MISMATCH'
    assert recovery.plans.find(recovery.step.id) is None


def test_original_case_boundaries_are_used_when_generation_had_no_step_context(recovery):
    from qa_agent.models import TestStep as Step
    later = Step(name='Verify success', description='Verify the registration success is visible.',
                 expected='The registration success is visible.', order=1)
    case = Case(id=recovery.case.id, public_id=recovery.case.public_id, name=recovery.case.name,
                description=recovery.case.description, base_url=recovery.case.base_url,
                steps=[recovery.step, later])
    recovery.cases.save(case)
    proposed = plan(action('fill', '#email', value='declared'), action('assert_value', '#email', expected='declared'),
                    action('assert_visible', '#registration-success'))
    response = post(recovery, 'save', proposed=proposed)
    assert response.status == 400 and b'TESTSTEP_BOUNDARY_VIOLATION' in response.body
    assert recovery.plans.find(recovery.step.id) is None


def test_incomplete_original_requirements_stay_needs_attention(recovery):
    recovery.app._recovery_store.discard(recovery.record.id)
    updated = recovery.case.model_copy(deep=True)
    updated.description = 'TBD'
    recovery.cases.save(updated)
    assert recovery.app.handle('GET', recovery.base).status == 200
    assert post(recovery, 'discover').status == 303
    result = post(recovery, 'save')
    assert result.status == 400 and b'Needs Attention' in result.body and b'original TestCase requirements are incomplete' in result.body
    assert recovery.plans.find(recovery.step.id) is None


@pytest.mark.parametrize('level', ['OFF', 'NORMAL', 'DEBUG', 'TRACE'])
def test_schema_rejected_json_is_recoverable_at_every_diagnostic_level(recovery, level):
    from qa_agent.candidate_diagnostics import attach_provider_candidate_summary
    class InvalidProvider(LocalProvider):
        def create_test_plan(self, *args):
            failure = RetryableLLMError('Invalid output', category='INVALID_RESPONSE')
            attach_provider_candidate_summary(failure, json.dumps({'url': 42, 'steps': [{'action': 'fill', 'parameters': {'selector': ['#email'], 'value': 'editable-private-value'}}]}))
            raise failure
    recovery.supervisor.repository.update_diagnostic_level(level)
    generator = LLMTestPlanGenerator(LLMRouter([InvalidProvider([])]), recovery.supervisor)
    with llm_usage_scope(related_test_case_id=recovery.case.id), pytest.raises(RetryableLLMError):
        generator.generate_with_plan(recovery.step, observed_registration(), requirement_context=recovery.case.description)
    record = recovery.supervisor.repository.list_records()[0]
    private = recovery.app._recovery_store.get(record.id, recovery.case.id, recovery.step.id)
    assert private.candidate['url'] == 42
    assert private.candidate['steps'][0]['parameters']['selector'] == ['#email']
    assert 'editable-private-value' not in record.model_dump_json()
    path = recovery.base.replace(str(recovery.record.id), str(record.id))
    response = recovery.app.handle('GET', path)
    assert response.status == 200 and b'editable-private-value' in response.body


def test_superseded_draft_cannot_commit_a_validated_older_revision():
    store = ManualRecoveryStore()
    step = input_step()
    case = Case(name='Revision fixture', description=step.description, steps=[step])
    entry = store.empty(uuid4(), case, step)
    updated = store.update(entry, entry.revision, draft={'url': observed_registration().url, 'steps': []})
    calls = []
    with pytest.raises(RecoveryUnavailable):
        store.commit(entry, lambda: calls.append(True))
    assert calls == [] and store.get(updated.operation_id, case.id, step.id).revision == updated.revision


@pytest.mark.parametrize('private', ['api_key=unregistered-private-key', 'Bearer synthetic-private-token', 'sk-privatecredential123456'])
def test_unregistered_credential_syntax_is_not_retained_or_reflected(recovery, private):
    recovery.app._recovery_store.capture(recovery.record, candidate(value=private, expected=private))
    entry = recovery.app._recovery_store.get(recovery.record.id, recovery.case.id, recovery.step.id)
    assert private not in json.dumps(entry.candidate)
    result = post(recovery, 'save', proposed=candidate(value=private, expected=private))
    assert result.status == 400 and private.encode() not in result.body
    assert recovery.plans.find(recovery.step.id) is None
