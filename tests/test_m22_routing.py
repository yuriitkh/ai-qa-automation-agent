"""Saved provider ordering, safe local skips, CLI parity and fallback."""
import json
from unittest.mock import patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from qa_agent.cli import build_pipeline
from qa_agent.execution_progress import ExecutionProgressReporter, ExecutionProgressStore
from qa_agent.execution_trace import ExecutionTraceRecorder, RequestKind, ProviderAttemptOutcome, active_trace_recorder
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.provider_settings import ProviderSettingsRepository, ProviderSettingsService
from qa_agent.reliability import SQLiteReliabilityRepository, AutomationReliabilitySupervisor, ReliabilitySettings
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.web import LocalWebApplication
from tests.test_provider_settings import MemorySecrets
from tests.test_automation_reliability import LocalProvider
from tests.test_generation_reliability import action, observed_registration, plan
from tests.test_m22_generation import input_step


@pytest.mark.parametrize('fallback', [False, True])
def test_web_saved_priority_is_actual_generation_order_and_unavailable_is_not_a_request(tmp_path, fallback):
    candidate = plan(action('fill', '#email', value='invalid-email'), action('assert_value', '#email', expected='invalid-email'))
    calls = []
    def factory(name, key, model, timeout):
        results = [RetryableLLMError('fixture', category='PROVIDER_UNAVAILABLE')] if fallback and name == 'openrouter' else [candidate]
        provider = LocalProvider(results, name=name, available=bool(key))
        provider.model = model
        original = provider.create_test_plan
        def request(*args):
            calls.append(name)
            return original(*args)
        provider.create_test_plan = request
        provider.create_structured_output = lambda *args: calls.append(name) or '{"ok":true}'
        return provider
    database = tmp_path / 'settings.sqlite3'
    environment = {'LLM_PROVIDER_ORDER': 'openai,gemini,openrouter,groq',
                   'OPENROUTER_API_KEY': 'fake-key', 'GROQ_API_KEY': 'fake-groq-key'}
    settings = ProviderSettingsService(ProviderSettingsRepository(database), MemorySecrets(),
                                       environment=environment, provider_factory=factory)
    router = settings.create_router()
    supervisor = AutomationReliabilitySupervisor(SQLiteReliabilityRepository(database))
    supervisor.repository.save_settings(ReliabilitySettings(diagnostic_level='DEBUG'))
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()),
                              provider_settings=settings, provider_router=router, reliability=supervisor)
    try:
        for provider, operation, model in [('openrouter', 'move_up', ''), ('openrouter', 'move_up', ''),
                                          ('groq', 'move_up', ''), ('groq', 'move_up', ''),
                                          ('openrouter', 'save_model', 'openai/gpt-6-luna')]:
            response = app.handle('POST', '/settings/providers', urlencode(dict(provider_id=provider, operation=operation, model=model)))
            assert response.status == 303
        assert [p.name for p in router._providers] == ['openrouter', 'groq', 'openai', 'gemini']
        assert router._providers[0].model == 'openai/gpt-6-luna'
        generator = LLMTestPlanGenerator(router, supervisor)
        recorder = ExecutionTraceRecorder(task='fixture')
        step = input_step()
        recorder.begin_step(step)
        with active_trace_recorder(recorder):
            generator.generate_with_plan(step, observed_registration())
        record = supervisor.repository.list_records()[0]
        assert [p['provider'] for p in record.effective_provider_order] == ['openrouter', 'groq', 'openai', 'gemini']
        assert [p['available'] for p in record.effective_provider_order] == [True, True, False, False]
        assert calls == (['openrouter', 'groq'] if fallback else ['openrouter'])
        assert len(record.attempts) == (2 if fallback else 1)
        # Authoring/structured requests use the same refreshed chain.
        calls.clear()
        assert router.create_structured_output('fixture', {'type': 'object'}, 'fixture') == '{"ok":true}'
        assert calls == ['openrouter']
        # Rebuilding settings retains the selected model and saved order.
        reconstructed = ProviderSettingsService(ProviderSettingsRepository(database), MemorySecrets(),
                                               environment=environment, provider_factory=factory)
        assert [p.name for p in reconstructed.create_router()._providers] == ['openrouter', 'groq', 'openai', 'gemini']
    finally:
        app.close()


def test_cli_uses_shared_saved_models_priorities_and_enabled_flags(tmp_path):
    database = tmp_path / 'cli.sqlite3'
    settings = ProviderSettingsService(ProviderSettingsRepository(database), MemorySecrets(), environment={})
    settings.update('openrouter', model='user-selected-model')
    settings.move('openrouter', -1)
    settings.move('openrouter', -1)
    settings.update('gemini', enabled=False)
    with patch.dict('os.environ', {'LLM_PROVIDER_ORDER': 'openai,gemini,openrouter,groq'}, clear=True), \
         patch('qa_agent.cli.create_default_secret_store', return_value=MemorySecrets()):
        pipeline = build_pipeline(database_path=database, evidence_directory=tmp_path / 'evidence')
    providers = pipeline._plan_generator._router._providers
    from qa_agent.llm.router import LLMRouter
    assert [LLMRouter._provider_id(p) for p in providers] == ['openrouter', 'openai', 'groq']
    assert providers[0].model == 'user-selected-model'


def test_progress_numbers_only_real_requests_and_labels_unavailable_locally():
    from types import SimpleNamespace
    store = ExecutionProgressStore()
    # Capture through the public reporter with a small recording-store spy.
    recorded = []
    store.set_provider_diagnostics = lambda progress_id, rows: recorded.extend(rows)
    reporter = ExecutionProgressReporter(store, 'fixture')
    attempts = [SimpleNamespace(provider_name=name, request_kind=RequestKind.TEST_PLAN, outcome=outcome,
        model='fixture', duration_ms=None, error_class=None) for name, outcome in
        [('openai', ProviderAttemptOutcome.UNAVAILABLE), ('openrouter', ProviderAttemptOutcome.SUCCESS)]]
    reporter.capture_provider_diagnostics(SimpleNamespace(steps=[SimpleNamespace(test_step_id=uuid4(), order=0, provider_attempts=attempts)]))
    assert recorded[0]['request_sent'] is False and recorded[0]['attempt_number'] is None
    assert recorded[1]['request_sent'] is True and recorded[1]['attempt_number'] == 1
    html = LocalWebApplication._progress_diagnostics_html([], recorded)
    assert 'Skipped locally (no API request)' in html
