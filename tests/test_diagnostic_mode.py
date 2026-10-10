"""M2.3 diagnostics: fake providers, synthetic browser states and temporary SQLite."""
from datetime import datetime, timedelta, timezone
import io
import json
import sqlite3
from types import SimpleNamespace
from urllib.parse import urlencode
from uuid import uuid4
import zipfile

import pytest

from qa_agent.diagnostic_mode import (
    DiagnosticLevel, DiagnosticEvent, MAX_EVENTS, MAX_OPTIONAL_BYTES, MAX_RETAINED_OPERATIONS,
    bound_optional, failure_reasons, structural_metadata,
    diagnostic_scope, current_diagnostic_level,
)
from qa_agent.diagnostic_export import diagnostic_report, diagnostic_download, encode_report, MAX_REPORT_BYTES
from qa_agent.execution_diagnostics import ActionFailureDiagnostic
from qa_agent.execution_progress import ExecutionEventType, ExecutionProgressReporter, ExecutionProgressStore, active_execution_progress
from qa_agent.execution_trace import ExecutionTraceRecorder, active_trace_recorder
from qa_agent.input_value_assertions import input_value_grounding_reason
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.llm.usage_metadata import capture_openai_usage, capture_response_http_status, mark_provider_request
from qa_agent.llm_usage import llm_usage_scope
from qa_agent.models import ExecutionStatus, TestStep as Step
from qa_agent.redaction import register_secret
from qa_agent.reliability import (
    AutomationReliabilitySupervisor, InMemoryReliabilityRepository, SQLiteReliabilityRepository,
    ReliabilitySettings, ReliabilityRecord, ReliabilityOperation,
)
from qa_agent.run_history import RunHistoryService, InMemoryRunHistoryRepository, RunHistoryRecord, HistoryExecutionReference, WorkflowType
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError, PlanValidationIssue
from qa_agent.web import LocalWebApplication
from tests.test_automation_reliability import LocalProvider
from tests.test_generation_reliability import action, observed_registration, plan


def input_step(description='Enter "declared" into the email field.', expected='Email field contains "declared".'):
    return Step(name='Verify input', description=description, expected=expected, order=0)


def candidate(value='declared', expected='declared'):
    return plan(action('fill', '#email', value=value), action('assert_value', '#email', expected=expected))


def generator(level='NORMAL', candidates=None, providers=None, repository=None):
    repo = repository or InMemoryReliabilityRepository()
    repo.save_settings(ReliabilitySettings(diagnostic_level=level))
    supervisor = AutomationReliabilitySupervisor(repo)
    return LLMTestPlanGenerator(LLMRouter(providers or [LocalProvider(candidates or [candidate()])]), supervisor)


@pytest.mark.parametrize('level', list(DiagnosticLevel))
@pytest.mark.parametrize('rejected', [False, True])
def test_levels_preserve_decisions_and_essential_audit(level, rejected):
    gen = generator(level, [candidate(expected='private-mismatch') if rejected else candidate()])
    if rejected:
        with pytest.raises(PlanValidationError) as caught:
            gen.generate_with_plan(input_step(), observed_registration())
        assert caught.value.issues[0].code == 'UNGROUNDED_ASSERTION'
        assert caught.value.issues[0].reason_code == 'FILL_ASSERT_VALUE_MISMATCH'
    else:
        gen.generate_with_plan(input_step(), observed_registration())
    record = gen.supervisor.repository.list_records()[0]
    assert record.diagnostic_level == level
    attempt = record.attempts[0]
    assert attempt.provider_status == 'SUCCESS' and attempt.duration_ms is not None
    assert attempt.quality_gates == ({'schema_and_actions': 'PASSED', 'locator_identity': 'PASSED',
        'assertion_grounding': 'FAILED', 'expected_result_coverage': 'NOT_RUN', 'step_boundaries': 'NOT_RUN'} if rejected else dict.fromkeys(attempt.quality_gates, 'PASSED'))
    assert record.outcome == ('NEEDS_ATTENTION' if rejected else 'READY_FOR_REVIEW')
    assert record.decisions and (record.candidate_version_id is None) == rejected
    assert bool(record.diagnostic_events) == (level != DiagnosticLevel.OFF)
    assert bool(record.effective_provider_order) == (level in {DiagnosticLevel.DEBUG, DiagnosticLevel.TRACE})
    assert bool(attempt.candidate_diagnostics) == (rejected and level in {DiagnosticLevel.DEBUG, DiagnosticLevel.TRACE})
    assert any(event.kind == 'QUALITY_GATE' for event in record.diagnostic_events) == (level == DiagnosticLevel.TRACE)
    if rejected:
        assert attempt.failure_reasons[0]['reason_code'] == 'FILL_ASSERT_VALUE_MISMATCH'
    assert 'private-mismatch' not in record.model_dump_json()


@pytest.mark.parametrize('level', list(DiagnosticLevel))
def test_sqlite_settings_persist_across_restart_without_overwriting_reliability_settings(tmp_path, level):
    path = tmp_path / 'settings.sqlite3'
    repo = SQLiteReliabilityRepository(path)
    repo.save_settings(ReliabilitySettings(provider_fallback=False, max_total_attempts=1))
    repo.update_diagnostic_level(level)
    settings = SQLiteReliabilityRepository(path).settings()
    assert settings.diagnostic_level == level and not settings.provider_fallback and settings.max_total_attempts == 1


@pytest.mark.parametrize('value', ['debug', 'BOGUS', '', 7, None])
def test_invalid_configuration_falls_back_safely_without_rewriting_saved_settings(tmp_path, value):
    repo = SQLiteReliabilityRepository(tmp_path / 'settings.sqlite3')
    raw = json.dumps({'diagnostic_level': value})
    with sqlite3.connect(repo.path) as connection:
        connection.execute('UPDATE automation_reliability_settings SET settings_json=?', (raw,))
    supervisor = AutomationReliabilitySupervisor(repo)
    settings, fallback = supervisor.effective_settings()
    assert settings == ReliabilitySettings() and fallback
    with sqlite3.connect(repo.path) as connection:
        assert connection.execute('SELECT settings_json FROM automation_reliability_settings').fetchone()[0] == raw


@pytest.mark.parametrize('description,context,value,expected,mutation,reason', [
    ('Enter "declared" into email.', None, 'declared', 'other', None, 'FILL_ASSERT_VALUE_MISMATCH'),
    ('Enter "declared" into email.', None, 'declared', 'declared', 'no_fill', 'MISSING_PRECEDING_FILL'),
    ('Enter "declared" into email.', None, 'declared', 'declared', 'unknown_control', 'UNSUPPORTED_INPUT_CONTROL'),
    ('Enter an email address.', None, 'declared', 'declared', None, 'VALUE_NOT_DECLARED_IN_REQUIREMENT'),
    ('Enter "declared" into email or password.', None, 'declared', 'declared', None, 'AMBIGUOUS_INPUT_BINDING'),
    ('Enter "declared" into email. Enter "other" into email.', None, 'declared', 'declared', None, 'CONFLICTING_REQUIREMENT_VALUES'),
    ('Enter "declared" into email. Never use "declared" in email.', None, 'declared', 'declared', None, 'PROHIBITED_INPUT_VALUE'),
    ('Enter "declared" into email.', 'Enter an email address.', 'declared', 'declared', None, 'VALUE_NOT_DECLARED_IN_REQUIREMENT'),
    ('Enter "other" into email.', 'Enter "declared" into email.', 'declared', 'declared', None, 'CONFLICTING_REQUIREMENT_VALUES'),
    ('Enter "declared" into email.', 'Enter "declared" into email. Never use "declared" in email.', 'declared', 'declared', None, 'PROHIBITED_INPUT_VALUE'),
])
def test_grounding_reasons_follow_the_same_decision_branches(description, context, value, expected, mutation, reason):
    proposed = candidate(value, expected)
    index = 1
    if mutation == 'no_fill':
        proposed.steps = proposed.steps[1:]
        index = 0
    if mutation == 'unknown_control':
        for item in proposed.steps:
            item.parameters['selector'] = '#absent'
    actual = input_value_grounding_reason(proposed, index, observed_registration(), (description,), requirement_context=context)
    assert actual == reason
    assert value not in actual and expected not in actual


@pytest.mark.parametrize('error,reason,known', [
    (RuntimeError('private payload rate limit password'), 'UNKNOWN', False),
    (RetryableLLMError('rate limit'), 'UNKNOWN', False),
    (RetryableLLMError('private', http_status=429), 'PROVIDER_RATE_LIMITED', True),
    (RetryableLLMError('private', category='RATE_LIMIT', http_status=429), 'PROVIDER_RATE_LIMITED', True),
    (RetryableLLMError('private', category='PROVIDER_UNAVAILABLE'), 'PROVIDER_UNAVAILABLE', True),
    (RetryableLLMError('private', category='INVALID_RESPONSE'), 'INVALID_STRUCTURED_OUTPUT', True),
    (RetryableLLMError('private', category='INVALID_RESPONSE', provider_error_code='output_token_limit'), 'OUTPUT_TRUNCATED', True),
])
def test_typed_failures_are_distinguished_and_unknown_strings_are_not_interpreted(error, reason, known):
    assert failure_reasons(error) == [{'reason_code': reason, 'root_cause_known': known}]


@pytest.mark.parametrize('expected,reason', [
    ('Email field contains "declared".', 'EXPECTED_RESULT_NOT_COVERED'),
    ('Data is durably persisted correctly.', 'UNSUPPORTED_EXPECTED_RESULT'),
])
def test_coverage_rejection_reasons_preserve_public_code(expected, reason):
    gen = generator('DEBUG', [plan(action('fill', '#email', value='declared'))])
    with pytest.raises(PlanValidationError) as caught:
        gen.generate_with_plan(input_step(expected=expected), observed_registration())
    assert caught.value.issues[0].code == 'EXPECTED_RESULT_NOT_COVERED'
    assert caught.value.issues[0].reason_code == reason


@pytest.mark.parametrize('target,action_name,reason', [
    ('#submit', 'assert_value', 'UNSUPPORTED_INPUT_CONTROL'),
    ('#email', 'assert_unchecked', 'ASSERTION_TARGET_MISMATCH'),
])
def test_wrong_assertion_targets_get_deterministic_reasons(target, action_name, reason):
    gen = generator('DEBUG', [plan(action(action_name, target, **({'expected': 'declared'} if action_name == 'assert_value' else {})))])
    with pytest.raises(PlanValidationError) as caught:
        gen.generate_with_plan(input_step(), observed_registration())
    assert caught.value.issues[0].code == 'ACTION_TARGET_MISMATCH'
    assert caught.value.issues[0].reason_code == reason


@pytest.mark.parametrize('changes,reason,known', [
    ({'page_open': False}, 'BROWSER_PRECONDITION_FAILED', True),
    ({'target_count': 0}, 'SELECTOR_NOT_FOUND', True),
    ({}, 'BROWSER_ACTION_FAILED', False),
    ({'target_count': 1, 'target_visible': False}, 'BROWSER_ACTION_FAILED', False),
])
def test_browser_failure_reason_does_not_guess_the_underlying_product_cause(changes, reason, known):
    diagnostic = ActionFailureDiagnostic(action_index=0, action='click', exception_category='ACTION_ERROR', exception_type='Other', **changes)
    assert diagnostic.reason_code == reason and diagnostic.root_cause_known == known


def test_routing_skips_fallback_provider_usage_and_correlation_are_separate():
    class ResponseProvider(LocalProvider):
        def create_test_plan(self, *args):
            mark_provider_request()
            capture_response_http_status(200)
            capture_openai_usage({'choices': [{'finish_reason': 'stop', 'message': {'content': 'private-response'}}],
                                  'usage': {'prompt_tokens': 12, 'completion_tokens': 7}})
            return super().create_test_plan(*args)
    providers = [LocalProvider([], name='Skipped', available=False),
                 LocalProvider([RetryableLLMError('private', category='RATE_LIMIT', http_status=429)], name='Limited'),
                 ResponseProvider([candidate()], name='Selected')]
    gen = generator('TRACE', providers=providers)
    trace = ExecutionTraceRecorder('Local synthetic run')
    step = input_step()
    trace.begin_step(step)
    case_id = uuid4()
    with active_trace_recorder(trace), llm_usage_scope(related_test_case_id=case_id):
        gen.generate_with_plan(step, observed_registration())
    record = gen.supervisor.repository.list_records()[0]
    assert record.test_case_id == case_id and record.test_step_id == step.id
    assert len(record.attempts) == 2 and [item.provider_status for item in record.attempts] == ['FAILED', 'SUCCESS']
    assert record.attempts[0].failure_reasons[0]['reason_code'] == 'PROVIDER_RATE_LIMITED'
    assert record.attempts[0].provider_metadata['http_status'] == 429
    assert record.attempts[1].input_tokens == 12 and record.attempts[1].output_tokens == 7
    assert record.attempts[1].provider_metadata['finish_reason'] == 'stop'
    assert record.attempts[1].provider_metadata['http_status'] == 200
    assert record.attempts[1].provider_metadata['response_chars'] == len('private-response')
    assert record.attempts[1].request_sent is True
    skipped = next(event for event in record.diagnostic_events if event.kind == 'PROVIDER_SKIPPED')
    assert skipped.metadata['request_sent'] is False
    assert any(decision.action == 'PROVIDER_FALLBACK' for decision in record.decisions)
    assert all(attempt.reliability_operation_id == record.id for attempt in trace._steps[0].provider_attempts)
    assert [attempt.attempt_index for attempt in trace._steps[0].provider_attempts] == [None, 1, 2]
    assert 'private-response' not in record.model_dump_json()
    assert all('request_sent' not in event.metadata for event in record.diagnostic_events if event.kind == 'PROVIDER_REQUEST_STARTED')


@pytest.mark.parametrize('level', ['NORMAL', 'DEBUG', 'TRACE'])
def test_response_structure_and_content_are_separated_by_level(level):
    class ResponseProvider(LocalProvider):
        def create_test_plan(self, *args):
            capture_openai_usage(SimpleNamespace(choices=[SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='sensitive payload'))]))
            raise RetryableLLMError('sensitive payload', category='INVALID_RESPONSE')
    gen = generator(level, providers=[ResponseProvider([])])
    with pytest.raises(RetryableLLMError):
        gen.generate_with_plan(input_step(), observed_registration())
    attempt = gen.supervisor.repository.list_records()[0].attempts[0]
    assert bool(attempt.provider_metadata) == (level != 'NORMAL')
    assert ('response_chars' in attempt.provider_metadata) == (level == 'TRACE')
    if level != 'NORMAL':
        assert attempt.failure_reasons[0]['reason_code'] == 'OUTPUT_TRUNCATED'
    assert 'sensitive payload' not in attempt.model_dump_json()


@pytest.mark.parametrize('failure', ['collector', 'candidate', 'retention', 'storage'])
def test_diagnostic_failures_do_not_interrupt_gates_or_generation(monkeypatch, failure):
    gen = generator('TRACE', [candidate(expected='other') if failure == 'candidate' else candidate()])
    def unavailable(*args, **kwargs):
        raise OSError('private local path')
    if failure == 'collector':
        monkeypatch.setattr('qa_agent.diagnostic_mode.structural_metadata', unavailable)
    elif failure == 'candidate':
        monkeypatch.setattr('qa_agent.candidate_diagnostics.rejected_candidate_diagnostics', unavailable)
    elif failure == 'retention':
        monkeypatch.setattr(gen.supervisor.repository.primary, 'prune_diagnostics', unavailable)
    else:
        monkeypatch.setattr(gen.supervisor.repository.primary, 'save', unavailable)
    if failure == 'candidate':
        with pytest.raises(PlanValidationError) as caught:
            gen.generate_with_plan(input_step(), observed_registration())
        assert caught.value.issues[0].code == 'UNGROUNDED_ASSERTION'
    else:
        result = gen.generate_with_plan(input_step(), observed_registration())
        assert result.test_plan_version.qa_test_plan == candidate()
    record = gen.supervisor.repository.list_records()[0]
    assert record.diagnostic_storage_fallback == (failure == 'storage')
    assert record.outcome == ('NEEDS_ATTENTION' if failure == 'candidate' else 'READY_FOR_REVIEW')


def test_optional_candidate_collection_cannot_replace_a_provider_schema_failure(monkeypatch):
    original = RetryableLLMError('private schema failure', category='INVALID_RESPONSE', provider_error_code='invalid_schema_response')
    gen = generator('DEBUG', providers=[LocalProvider([original])])
    def unavailable(*args, **kwargs):
        raise OSError('private diagnostic failure')
    monkeypatch.setattr('qa_agent.candidate_diagnostics.rejected_candidate_diagnostics', unavailable)
    with pytest.raises(RetryableLLMError) as caught:
        gen.generate_with_plan(input_step(), observed_registration())
    assert caught.value is original
    saved = gen.supervisor.repository.list_records()[0]
    assert saved.outcome == 'NEEDS_ATTENTION'
    assert saved.attempts[0].provider_status == 'FAILED'
    assert saved.attempts[0].failure_reasons[0]['reason_code'] == 'INVALID_STRUCTURED_OUTPUT'
    assert saved.attempts[0].candidate_diagnostics is None


@pytest.fixture
def app():
    supervisor = AutomationReliabilitySupervisor()
    application = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), reliability=supervisor)
    try:
        yield application
    finally:
        application.close()


@pytest.mark.parametrize('level', list(DiagnosticLevel))
def test_web_settings_save_requires_csrf_and_has_effective_level(app, level):
    response = app.handle('POST', '/settings/diagnostics', urlencode({'level': level.value}), headers={'X-QA-CSRF': app._csrf_token})
    assert response.status == 303
    html = app.handle('GET', '/settings/diagnostics').body.decode()
    assert f'Current effective level: <strong>{level.value}</strong>' in html
    assert 'Diagnostic Mode' in app.handle('GET', '/settings/reliability').body.decode()
    assert app.handle('POST', '/settings/diagnostics', 'level=TRACE').status == 403
    assert app._reliability.repository.settings().diagnostic_level == level


@pytest.mark.parametrize('body', ['level=bad', 'level=DEBUG&level=TRACE', 'level=TRACE&payload=secret', ''])
def test_invalid_web_settings_do_not_change_effective_mode(app, body):
    assert app.handle('POST', '/settings/diagnostics', body, headers={'X-QA-CSRF': app._csrf_token}).status == (403 if body == 'level=DEBUG&level=TRACE' else 400)
    assert app._reliability.repository.settings().diagnostic_level == DiagnosticLevel.NORMAL


def test_settings_storage_failure_is_visible_and_is_not_reported_as_saved(app, monkeypatch):
    def unavailable(*args):
        raise OSError('private local path')
    monkeypatch.setattr(app._reliability.repository.primary, 'update_diagnostic_level', unavailable)
    response = app.handle('POST', '/settings/diagnostics', 'level=TRACE', headers={'X-QA-CSRF': app._csrf_token})
    assert response.status == 503 and 'private local path' not in response.body.decode()
    assert app._reliability.repository.settings().diagnostic_level == DiagnosticLevel.NORMAL
    monkeypatch.setattr(app._reliability.repository.primary, 'settings', unavailable)
    response = app.handle('GET', '/settings/diagnostics')
    assert response.status == 200 and b'safe effective fallback is NORMAL' in response.body


def test_metadata_and_validation_paths_drop_untrusted_sensitive_fields():
    secret = 'private-value-DO-NOT-RECORD'
    register_secret(secret)
    metadata = structural_metadata({'provider': secret, 'model': '<script>fixture</script>', 'http_status': 429,
        'prompt': secret, 'response': secret, 'html': '<input value="secret">', 'cookies': secret,
        'Authorization': secret, 'finish_reason': secret, 'provider_error_code': secret, 'duration_ms': True})
    assert secret not in json.dumps(metadata) and 'duration_ms' not in metadata
    error = PlanValidationError([PlanValidationIssue(code='UNGROUNDED_ASSERTION', path=secret, message=secret)])
    assert failure_reasons(error) == [{'code': 'UNGROUNDED_ASSERTION', 'path': 'plan', 'reason_code': 'UNKNOWN', 'root_cause_known': False}]


def record(step_id=None, case_id=None, version_id=None, started=None):
    return ReliabilityRecord(test_step_id=step_id or uuid4(), test_case_id=case_id,
        candidate_version_id=version_id, settings=ReliabilitySettings(diagnostic_level='DEBUG'),
        diagnostic_level='DEBUG', started_at=started or datetime.now(timezone.utc), outcome='READY_FOR_REVIEW')


def run(case_id, references):
    return RunHistoryRecord(run_id=uuid4(), test_case_id=case_id, test_case_name='Private case name',
        test_case_description='Private requirement value', workflow_type=WorkflowType.VALIDATION,
        status=ExecutionStatus.FAILED, outcome='AUTOMATION_EXECUTION_ERROR',
        started_at=datetime.now(timezone.utc), executions=references)


def reference(step_id, version_id):
    return HistoryExecutionReference(execution_id=uuid4(), test_step_id=step_id,
        test_plan_version_id=version_id, status=ExecutionStatus.FAILED,
        started_at=datetime.now(timezone.utc), safe_error='password=private',
        safe_actual_result={'input_value': 'private'}, classification='AUTOMATION_EXECUTION_ERROR',
        action_failure=ActionFailureDiagnostic(action_index=0, action='fill', selector_identity='#email',
            exception_category='TIMEOUT', exception_type='TimeoutError', target_count=0))


@pytest.mark.parametrize('format', ['json', 'zip'])
def test_operation_and_run_exports_use_exact_step_version_correlation(app, format):
    case_id, first, second, version_a, version_b = [uuid4() for _ in range(5)]
    records = [record(first, case_id, version_a), record(second, case_id, version_b),
               record(first, uuid4(), version_a), record(first, case_id, uuid4())]
    for item in records:
        app._reliability.repository.save(item)
    saved = run(case_id, [reference(first, version_a), reference(second, version_b)])
    app._run_history._repository.save(saved)
    for scope, identity, expected_ops, executions in [('settings/reliability', records[0].id, [records[0].id], 1),
                                                     ('runs', saved.run_id, [records[0].id, records[1].id], 2)]:
        response = app.handle('GET', f'/{scope}/{identity}/diagnostics.{format}')
        assert response.status == 200 and response.headers['Cache-Control'] == 'no-store'
        if format == 'zip':
            with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
                assert archive.namelist() == ['diagnostics.json', 'summary.txt']
                raw = archive.read('diagnostics.json')
                assert b'No product PASS is inferred' in archive.read('summary.txt')
        else:
            raw = response.body
        report = json.loads(raw)
        assert set(op['operation_id'] for op in report['operations']) == set(map(str, expected_ops))
        assert len(report['runs'][0]['executions']) == executions
        assert 'Private case name' not in raw.decode() and 'private' not in raw.decode()
        assert report['runs'][0]['executions'][0]['action_failure']['reason_code'] == 'SELECTOR_NOT_FOUND'
        assert report['runs'][0]['executions'][0]['evidence_availability'] == 'UNAVAILABLE'
        assert report['runs'][0]['executions'][0]['reliability_operation_ids'] == [str(records[0].id)]


def test_rejected_operation_cannot_acquire_an_unrelated_successful_execution(app):
    case_id, step_id = uuid4(), uuid4()
    rejected = record(step_id, case_id)
    rejected.outcome = 'NEEDS_ATTENTION'
    app._reliability.repository.save(rejected)
    app._run_history._repository.save(run(case_id, [reference(step_id, uuid4())]))
    report = json.loads(app.handle('GET', f'/settings/reliability/{rejected.id}/diagnostics.json').body)
    assert report['runs'] == []
    assert any('No correlated Browser Execution' in limitation for limitation in report['limitations'])


@pytest.mark.parametrize('path', [
    '/settings/reliability/../../qa_agent.db/diagnostics.zip',
    '/settings/reliability/%2e%2e%2fqa_agent.db/diagnostics.json',
    '/runs/not-a-uuid/diagnostics.zip', '/runs/00000000-0000-0000-0000-000000000000/diagnostics.json',
])
def test_export_traversal_and_missing_record_requests_do_not_read_files(app, path):
    assert app.handle('GET', path).status == 404


def test_export_filename_is_uuid_only_and_archive_has_fixed_members():
    report = diagnostic_report([])
    with pytest.raises(ValueError):
        diagnostic_download(report, '../private', 'zip')
    with pytest.raises(ValueError):
        diagnostic_download(report, uuid4(), '../../private')
    assert len(encode_report(report)) <= MAX_REPORT_BYTES


def test_export_and_generation_page_sanitize_legacy_candidate_metadata_and_escape_html(app):
    item = record()
    secret = 'private-form-value-SYNTHETIC'
    register_secret(secret)
    from qa_agent.reliability import ReliabilityAttempt, ReliabilityDecision
    item.decisions = [ReliabilityDecision(timestamp=item.started_at, action='STOP', reason=secret, category='INVALID_RESPONSE')]
    item.attempts = [ReliabilityAttempt(index=1, action='INITIAL_GENERATION', reason=secret,
        provider=f'<script>{secret}</script>', model='<img src=x onerror=alert(1)>', started_at=item.started_at,
        status='REJECTED', candidate_diagnostics={'actions': [{'action': 'fill', 'parameters': {'value': secret}}],
            'raw_response': secret, 'validation_issues': [{'code': 'UNGROUNDED_ASSERTION', 'path': secret}],
            'failed_gate': 'assertion_grounding'}, quality_gates={'assertion_grounding': 'FAILED'})]
    app._reliability.repository.save(item)
    html = app.handle('GET', f'/settings/reliability/{item.id}').body.decode()
    assert '<img src=x' not in html and '&lt;img src=x' in html
    assert secret not in html and 'Export Diagnostics' in html and 'Failure reason: UNKNOWN' in html
    raw = app.handle('GET', f'/settings/reliability/{item.id}/diagnostics.json').body.decode()
    assert secret not in raw and 'raw_response' not in raw


@pytest.mark.parametrize('sqlite', [False, True])
def test_optional_retention_is_bounded_without_deleting_audit_or_history(tmp_path, sqlite):
    repo = SQLiteReliabilityRepository(tmp_path / 'retention.sqlite3') if sqlite else InMemoryReliabilityRepository()
    now = datetime.now(timezone.utc)
    legacy = record(started=now - timedelta(days=20))
    legacy.diagnostic_level = None
    legacy.diagnostic_events = [DiagnosticEvent(timestamp=now, kind='DISCOVERY', metadata={'control_count': 2})]
    repo.save(legacy)
    original = repo.get(legacy.id).model_dump_json()
    for index in range(MAX_RETAINED_OPERATIONS + 3):
        item = record(started=now - timedelta(seconds=index))
        item.diagnostic_events = [DiagnosticEvent(timestamp=now, kind='DISCOVERY', metadata={'control_count': 2})]
        repo.save(item)
    old = record(started=now - timedelta(days=8))
    repo.save(old)
    repo.prune_diagnostics(now)
    records = repo.list_records()
    assert len(records) == MAX_RETAINED_OPERATIONS + 5
    assert sum(bool(item.diagnostic_events) for item in records if item.diagnostic_level is not None) <= MAX_RETAINED_OPERATIONS
    assert repo.get(old.id).diagnostics_expired
    assert repo.get(legacy.id).model_dump_json() == original
    assert all(item.outcome == 'READY_FOR_REVIEW' for item in records)


def test_per_operation_event_and_byte_budgets_preserve_essential_outcomes():
    item = record()
    item.diagnostic_events = [DiagnosticEvent(timestamp=item.started_at, kind='PROVIDER_RESPONSE',
        metadata={'provider': 'x' * 180, 'model': 'y' * 180, 'response_chars': 999}) for _ in range(MAX_EVENTS + 100)]
    bounded = bound_optional(item)
    assert len(bounded.diagnostic_events) <= MAX_EVENTS
    assert len(json.dumps({'events': [event.model_dump(mode='json') for event in bounded.diagnostic_events],
                           'candidates': [], 'providers': [], 'routing': []}).encode()) <= MAX_OPTIONAL_BYTES
    assert bounded.outcome == item.outcome and bounded.test_step_id == item.test_step_id


def test_report_budget_reports_omissions_instead_of_unbounded_export(monkeypatch):
    items = [record() for _ in range(8)]
    for item in items:
        item.diagnostic_events = [DiagnosticEvent(timestamp=item.started_at, kind='DISCOVERY', metadata={'provider': 'x' * 180}) for _ in range(100)]
    monkeypatch.setattr('qa_agent.diagnostic_export.MAX_REPORT_BYTES', 4096)
    raw = encode_report(diagnostic_report(items))
    report = json.loads(raw)
    assert len(raw) <= 4096 and report['report_truncated'] and report['operations_omitted'] > 0


def test_historical_record_without_m23_fields_has_unknown_level_and_causes():
    item = record()
    data = item.model_dump(mode='json')
    for key in ('diagnostic_level', 'diagnostic_events', 'diagnostic_events_omitted', 'diagnostics_expired', 'diagnostic_storage_fallback'):
        data.pop(key)
    historical = ReliabilityRecord.model_validate(data)
    report = diagnostic_report([historical])
    assert report['operations'][0]['effective_level'] == 'UNKNOWN_HISTORICAL'
    assert report['operations'][0]['optional_metadata_unavailable'] is True


def test_active_operation_keeps_level_while_settings_change():
    repo = InMemoryReliabilityRepository()
    class ChangesSettings(LocalProvider):
        def create_test_plan(self, *args):
            repo.update_diagnostic_level('OFF')
            return super().create_test_plan(*args)
    gen = generator('TRACE', providers=[ChangesSettings([candidate(), candidate()])], repository=repo)
    gen.generate_with_plan(input_step(), observed_registration())
    first = gen.supervisor.repository.list_records()[0]
    assert first.diagnostic_level == DiagnosticLevel.TRACE and any(event.kind == 'QUALITY_GATE' for event in first.diagnostic_events)
    gen.generate_with_plan(input_step(), observed_registration())
    second = gen.supervisor.repository.list_records()[0]
    assert second.diagnostic_level == DiagnosticLevel.OFF and second.diagnostic_events == []


def test_parent_run_policy_is_inherited_and_context_is_restored():
    assert current_diagnostic_level() is None
    gen = generator('DEBUG')
    with diagnostic_scope('TRACE'):
        gen.generate_with_plan(input_step(), observed_registration())
    assert current_diagnostic_level() is None
    assert gen.supervisor.repository.list_records()[0].diagnostic_level == DiagnosticLevel.TRACE
    with diagnostic_scope('invalid'):
        assert current_diagnostic_level() == DiagnosticLevel.NORMAL
    assert current_diagnostic_level() is None


def test_progress_export_and_failure_reason_use_the_corresponding_step(app):
    from qa_agent.models import TestCase as Case
    first, second = input_step(), input_step()
    second.order = 1
    second.name = 'Second input'
    case = Case(name='Synthetic', description='Synthetic input.', steps=[first, second])
    progress_id = app.progress_store.create(case.id, 'AUTOMATION')
    reporter = ExecutionProgressReporter(app.progress_store, progress_id)
    reporter.test_case_loaded(case)
    gen = LLMTestPlanGenerator(LLMRouter([LocalProvider([candidate(), candidate(expected='private mismatch')])]), app._reliability)
    with active_execution_progress(reporter), llm_usage_scope(related_test_case_id=case.id):
        gen.generate_with_plan(first, observed_registration())
        with pytest.raises(PlanValidationError):
            gen.generate_with_plan(second, observed_registration())
    data = app._progress_payload(app.progress_store.get(progress_id))
    assert data['steps'][0]['generation_failures'] == []
    assert data['steps'][1]['generation_failures'][0]['reason_codes'] == ['FILL_ASSERT_VALUE_MISMATCH']
    assert data['steps'][1]['generation_failures'][0]['failed_gate'] == 'assertion_grounding'
    html = app.handle('GET', f'/runs/progress/{progress_id}').body.decode()
    assert html.index('Second input') < html.index('Failure reason: FILL_ASSERT_VALUE_MISMATCH')
    assert 'private mismatch' not in html
    raw = app.handle('GET', f'/api/progress/{progress_id}/diagnostics.json')
    assert raw.status == 200
    report = json.loads(raw.body)
    assert {op['test_step_id'] for op in report['operations']} == {str(first.id), str(second.id)}
    assert len(report['operations']) == 2 and report['runs'] == []


def test_normal_reliability_form_preserves_diagnostic_selection(app):
    app._reliability.repository.update_diagnostic_level('TRACE')
    response = app.handle('POST', '/settings/reliability', urlencode(dict(
        additional_retries='off', provider_fallback='on', automatic_plan_repair='off', max_total_attempts='2')))
    assert response.status == 303
    assert app._reliability.repository.settings().diagnostic_level == DiagnosticLevel.TRACE


def test_export_storage_failures_return_a_safe_error_without_provider_invocation(app, monkeypatch):
    def unavailable(*args):
        raise OSError('private password or local path')
    monkeypatch.setattr(app._run_history, 'get', unavailable)
    response = app.handle('GET', f'/runs/{uuid4()}/diagnostics.json')
    assert response.status == 503 and b'No provider request was made' in response.body
    assert b'private' not in response.body
