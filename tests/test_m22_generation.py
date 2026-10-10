"""Offline M2.2 reproductions through the full generation-gate boundary."""
from types import SimpleNamespace
import json
from unittest.mock import patch

import pytest

from qa_agent.llm.openai_compatible import OpenAICompatibleProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan, TestStep as Step
from qa_agent.reliability import ReliabilitySettings
from qa_agent.diagnostic_mode import DiagnosticLevel
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from tests.test_generation_reliability import URL, action, observed_registration, generate, reject, plan
from tests.test_registration_coverage import RegistrationProvider


def diagnostic_generator(*args):
    generator = LLMTestPlanGenerator(*args)
    settings = generator.supervisor.repository.settings()
    generator.supervisor.repository.save_settings(settings.model_copy(update={'diagnostic_level': DiagnosticLevel.DEBUG}))
    return generator


def input_step():
    return Step(name='Input invalid email', order=0,
        description="Enter an invalid email address such as 'invalid-email' into the email field",
        expected='Email field contains the invalid email.')


def test_entered_data_without_errors_is_not_a_separate_acceptance_obligation():
    step = Step(name='Fill required fields with valid unique data', description='Fill the registration fields.',
                expected='Entered data is accepted without validation errors.', order=0)
    generate(step, plan(action('fill', '#email', value='local@example.test'),
                        action('assert_hidden', '#form-error')), observed_registration())


def test_fill_only_step_cannot_submit_even_without_future_step_context():
    step = Step(name='Fill required fields', description='Enter email.', expected='The input is entered.', order=0)
    candidate = plan(action('fill', '#email', value='local@example.test'), action('click', '#submit'))
    reject(step, candidate, observed_registration(), 'TESTSTEP_BOUNDARY_VIOLATION')


def test_explicit_input_literal_can_verify_the_actual_input_value():
    candidate = {'url': URL, 'steps': [action('fill', '#email', value='invalid-email'),
                    action('assert_value', '#email', expected='invalid-email')]}
    provider = RegistrationProvider([candidate])
    result = LLMTestPlanGenerator(LLMRouter([provider])).generate_with_plan(input_step(), observed_registration())
    assert result.test_plan_version.qa_test_plan.steps[-1].action == 'assert_value'


def test_input_visibility_is_not_input_value_verification():
    reject(input_step(), plan(action('fill', '#email', value='invalid-email'),
                             action('assert_visible', '#email')), observed_registration())


def test_rejected_candidate_has_value_free_bounded_gate_diagnostics():
    candidate = plan(action('fill', '#password', value='private-password-value'),
                     action('assert_unchecked', '#form-error'))
    provider = RegistrationProvider([candidate])
    generator = diagnostic_generator(LLMRouter([provider]))
    with pytest.raises(PlanValidationError):
        generator.generate_with_plan(input_step(), observed_registration())
    record = generator.supervisor.repository.list_records()[0]
    diagnostics = record.attempts[0].candidate_diagnostics
    assert diagnostics['validation_issues'][0]['code'] == 'ACTION_TARGET_MISMATCH'
    assert diagnostics['actions'][1]['assertion'] is True
    assert diagnostics['actions'][1]['selector_hash']
    assert 'private-password-value' not in record.model_dump_json()
    assert '#form-error' not in record.model_dump_json()
    assert record.candidate_version_id is None


@pytest.mark.parametrize('finish,content,expected_code', [
    ('length', '{"steps":[]}', 'output_token_limit'),
    ('length', '{"steps":', 'output_token_limit'),
    ('stop', None, 'missing_structured_response'),
    ('stop', '{"steps":', 'invalid_json'),
])
def test_adapter_diagnoses_truncation_and_missing_json(finish, content, expected_code):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
        message=SimpleNamespace(content=content))], usage=None)
    with patch('qa_agent.llm.openai_compatible.OpenAI') as client:
        client.return_value.chat.completions.create.return_value = response
        provider = OpenAICompatibleProvider('openrouter', 'UNUSED', 'fake-model', api_key='fake-test-key')
        with pytest.raises(RetryableLLMError) as caught:
            provider.create_structured_output('fixture', {'type': 'object'}, 'fixture')
    assert caught.value.category == 'INVALID_RESPONSE'
    assert caught.value.provider_error_code == expected_code


@pytest.mark.parametrize('assertion', [action('assert_page_loaded'), action('assert_visible', '#registration-form')])
def test_specific_displayed_page_requires_relevant_discovery_identity(assertion):
    step = Step(name='Open registration page', description='Navigate to registration.',
                expected='The registration page is displayed.', order=0)
    candidate = plan(action('navigate', url=URL), assertion)
    if assertion['action'] == 'assert_page_loaded':
        reject(step, candidate, observed_registration())
    else:
        generate(step, candidate, observed_registration())


@pytest.mark.parametrize('change,code', [
    ('invented_value', 'UNGROUNDED_ASSERTION'), ('other_input', 'UNGROUNDED_ASSERTION'),
    ('no_fill', 'UNGROUNDED_ASSERTION'), ('text_check', 'UNGROUNDED_ASSERTION'),
    ('checkbox', 'ACTION_TARGET_MISMATCH'), ('unobserved', 'ACTION_TARGET_MISMATCH'),
    ('changed_after_fill', 'UNGROUNDED_ASSERTION'), ('negated_input', 'UNGROUNDED_ASSERTION'),
    ('ambiguous_inputs', 'UNGROUNDED_ASSERTION'),
])
def test_input_literal_grounding_cannot_justify_other_values_outputs_or_controls(change, code):
    step, discovery = input_step(), observed_registration()
    actions = [action('fill', '#email', value='invalid-email'), action('assert_value', '#email', expected='invalid-email')]
    if change == 'invented_value':
        actions[-1]['parameters']['expected'] = 'invented@example.test'
    elif change == 'other_input':
        actions = [action('fill', '#password', value='invalid-email'), action('assert_value', '#password', expected='invalid-email')]
    elif change == 'no_fill':
        actions = actions[1:]
    elif change == 'text_check':
        actions[-1] = action('assert_text_contains', '#email', expected_text='invalid-email')
    elif change in {'checkbox', 'unobserved'}:
        actions[-1]['parameters']['selector'] = '#submit' if change == 'checkbox' else '#invented'
    elif change == 'changed_after_fill':
        actions.insert(1, action('fill', '#email', value='different'))
    elif change == 'negated_input':
        step.description = "Never enter 'invalid-email' into the email field."
    else:
        step.description = "Enter 'invalid-email' into the password and 'something-else' into the email field."
    reject(step, plan(*actions), discovery, code)


def test_explicit_atomic_input_instruction_survives_unrelated_case_context_and_periods():
    step = input_step()
    step.description = "Enter 'test.user@example.test' into the email field."
    candidate = plan(action('fill', '#email', value='test.user@example.test'),
                     action('assert_value', '#email', expected='test.user@example.test'))
    result = LLMTestPlanGenerator(LLMRouter([RegistrationProvider([candidate])])).generate_with_plan(
        step, observed_registration(), requirement_context='Exercise a retry after generation failure.')
    assert result.test_plan_version.assertion_grounding[0].category.value == 'REQUIREMENT_GROUNDED'


@pytest.mark.parametrize('context', ["Enter 'different' into the email field.", "Never enter 'invalid-email' into the email field."])
def test_atomic_input_example_cannot_override_explicit_original_input_requirement(context):
    candidate = plan(action('fill', '#email', value='invalid-email'), action('assert_value', '#email', expected='invalid-email'))
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator(LLMRouter([RegistrationProvider([candidate])])).generate_with_plan(
            input_step(), observed_registration(), requirement_context=context)
    assert caught.value.issues[0].code == 'UNGROUNDED_ASSERTION'


@pytest.mark.parametrize('value', ['', ' '])
def test_blank_exact_input_value_cannot_bypass_grounding(value):
    reject(input_step(), plan(action('fill', '#email', value=value),
                             action('assert_value', '#email', expected=value)),
           observed_registration(), 'UNGROUNDED_ASSERTION')


@pytest.mark.parametrize('value', ['', ' '])
def test_explicit_blank_input_literal_has_requirement_grounding(value):
    step = Step(name='Fill email', description=f"Enter '{value}' into the email field.",
                expected='Email field contains the declared input.', order=0)
    candidate = plan(action('fill', '#email', value=value), action('assert_value', '#email', expected=value))
    result, _ = generate(step, candidate, observed_registration())
    assert result.test_plan_version.assertion_grounding[0].category.value == 'REQUIREMENT_GROUNDED'


def test_expected_empty_literal_cannot_be_replaced_with_a_nonempty_input_example():
    step = input_step()
    step.expected = "Email field contains ''."
    reject(step, plan(action('fill', '#email', value='invalid-email'),
                      action('assert_value', '#email', expected='invalid-email')),
           observed_registration())


def test_rejected_diagnostics_retain_initial_and_targeted_repair_without_approved_version():
    step = Step(name='Fill required fields', description='Enter email.',
                expected='Entered data is accepted without validation errors.', order=0)
    candidates = [plan(action('fill', '#email', value='private-test-input')),
                  plan(action('fill', '#email', value='private-test-input'), action('assert_unchecked', '#form-error'))]
    generator = diagnostic_generator(LLMRouter([RegistrationProvider(candidates)]))
    generator.supervisor.repository.save_settings(ReliabilitySettings(automatic_plan_repair=True, diagnostic_level='DEBUG'))
    with pytest.raises(PlanValidationError):
        generator.generate_with_plan(step, observed_registration())
    record = generator.supervisor.repository.list_records()[0]
    assert [a.action for a in record.attempts] == ['INITIAL_GENERATION', 'TARGETED_REPAIR']
    assert all(a.candidate_diagnostics for a in record.attempts)
    assert record.attempts[0].candidate_diagnostics['coverage'] == 'NOT_COVERED'
    assert record.attempts[0].candidate_diagnostics['requirement_clauses'][0]['covered'] is False
    assert record.candidate_version_id is None
    assert 'private-test-input' not in record.model_dump_json()


def test_diagnostics_never_copy_parameters_urls_values_messages_or_untrusted_paths():
    from qa_agent.candidate_diagnostics import rejected_candidate_diagnostics
    from qa_agent.test_plan_validation import PlanValidationIssue
    step = input_step()
    step.expected = 'A confirmation message "private-expected-secret" is displayed.'
    sensitive = 'http://user:private-credential@localhost/path?token=private-token'
    raw = {'url': sensitive, 'steps': [action('fill', '[data-secret="private-selector-secret"]',
           value='private-password'), action('private-unknown-action', expected=sensitive)] * 100}
    error = PlanValidationError([PlanValidationIssue(code='INVALID_PARAMETERS', path=sensitive,
                                                   message='private-provider-body')])
    diagnostics = rejected_candidate_diagnostics(raw, step, observed_registration(), error)
    serialized = json.dumps(diagnostics)
    assert not any(secret in serialized for secret in ['private-', 'http://', '[data-secret'])
    assert len(diagnostics['actions']) == 64 and diagnostics['actions_omitted'] == 136
    assert len(serialized) < 12000
    assert diagnostics['validation_issues'][0]['path'] == 'plan'


@pytest.mark.parametrize('content', ['', 'not JSON', '{"url":"fixture","steps":[]}', '[]'])
def test_adapter_schema_and_missing_output_errors_are_safe(content):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content=content))], usage=None)
    with patch('qa_agent.llm.openai_compatible.OpenAI') as client:
        client.return_value.chat.completions.create.return_value = response
        provider = OpenAICompatibleProvider('openrouter', 'UNUSED', 'fake-model', api_key='fake-key')
        with pytest.raises(RetryableLLMError) as caught:
            provider.create_test_plan('fixture', URL, '{}')
    assert caught.value.category == 'INVALID_RESPONSE'
    assert caught.value.provider_error_code in {'missing_structured_response', 'invalid_json', 'invalid_schema_response'}


@pytest.mark.parametrize('description', ['Enter email; do not submit the form.', 'Fill email without clicking submit.'])
def test_negated_submission_is_not_permission_to_submit(description):
    step = Step(name='Fill required fields', description=description, expected='The input is entered.', order=0)
    reject(step, plan(action('fill', '#email', value='fixture'), action('click', '#submit')),
           observed_registration(), 'TESTSTEP_BOUNDARY_VIOLATION')


@pytest.mark.parametrize('failure', ['length', 'response_format'])
def test_structured_failures_fall_back_without_retrying_or_saving_raw_response(failure):
    from tests.test_automation_reliability import LocalProvider
    step = input_step()
    candidate = plan(action('fill', '#email', value='invalid-email'), action('assert_value', '#email', expected='invalid-email'))
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',
        message=SimpleNamespace(content='private-raw-LLM-response'))], usage=None)
    class Unsupported(RuntimeError):
        status_code = 400
        body = {'error': {'message': 'response_format is unsupported private-provider-body',
                          'param': 'response_format', 'code': 'unsupported_parameter'}}
    first = OpenAICompatibleProvider('openrouter', 'UNUSED', 'fake-model', api_key='fake-key')
    second = LocalProvider([candidate], name='fallback')
    with patch('qa_agent.llm.openai_compatible.OpenAI') as client:
        if failure == 'length':
            client.return_value.chat.completions.create.return_value = response
        else:
            client.return_value.chat.completions.create.side_effect = Unsupported('private-provider-body')
        generator = diagnostic_generator(LLMRouter([first, second]))
        generator.generate_with_plan(step, observed_registration())
        client.return_value.chat.completions.create.assert_called_once()
    record = generator.supervisor.repository.list_records()[0]
    assert len(record.attempts) == 2 and record.attempts[1].action == 'PROVIDER_FALLBACK'
    assert record.attempts[0].error_category == ('INVALID_RESPONSE' if failure == 'length' else 'SCHEMA_ERROR')
    assert record.attempts[0].structured_response_code == ('output_token_limit' if failure == 'length' else None)
    assert 'private-' not in record.model_dump_json()
    assert record.attempts[0].candidate_diagnostics['coverage_evaluated'] is False if failure == 'length' else True


@pytest.mark.parametrize('adapter', ['openrouter', 'gemini', 'groq'])
def test_schema_rejection_inside_real_adapter_retains_safe_action_and_parameter_diagnostics(adapter):
    content = json.dumps({'url': URL, 'steps': [action('fill', '#email', value='private-input'),
                         action('assert_value', '[data-password="private-selector"]')]})
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=content))], usage=None)
    from qa_agent.llm.gemini import GeminiProvider
    from qa_agent.llm.groq import GroqProvider
    with patch('qa_agent.llm.openai_compatible.OpenAI') as client, \
         patch('qa_agent.llm.gemini.genai.Client') as gemini_client, \
         patch('qa_agent.llm.groq.httpx.post') as groq_post:
        client.return_value.chat.completions.create.return_value = response
        gemini_client.return_value.interactions.create.return_value = SimpleNamespace(output_text=content)
        groq_post.return_value = SimpleNamespace(is_error=False, json=lambda: {'choices': [{'message': {'content': content}}]})
        provider = (OpenAICompatibleProvider('openrouter', 'UNUSED', 'fake-model', api_key='fake-key') if adapter == 'openrouter'
                    else GeminiProvider(api_key='fake-key', model='fake-model') if adapter == 'gemini'
                    else GroqProvider(api_key='fake-key', model='fake-model'))
        generator = diagnostic_generator(LLMRouter([provider]))
        with pytest.raises(RetryableLLMError):
            generator.generate_with_plan(input_step(), observed_registration())
    record = generator.supervisor.repository.list_records()[0]
    diagnostics = record.attempts[0].candidate_diagnostics
    assert diagnostics['actions'][1]['action'] == 'assert_value'
    assert diagnostics['actions'][1]['selector_hash']
    assert diagnostics['validation_issues'] == [{'code': 'MISSING_EXPECTED_VALUE', 'path': 'steps[1].parameters.expected'}]
    assert diagnostics['failed_gate'] == 'provider_response_schema'
    assert diagnostics['coverage_evaluated'] is False and record.candidate_version_id is None
    assert 'private-' not in record.model_dump_json()


def test_adapter_summary_metadata_cannot_inject_secrets_into_persistence():
    error = RetryableLLMError('fixture', category='INVALID_RESPONSE', provider_error_code='invalid_schema_response')
    error.rejected_actions = {'actions': [{'action': 'private-action', 'selector_hash': 'private-secret',
                                          'value': 'private-input', 'index': 'private-id'}],
        'actions_omitted': 'private-count', 'private-extra': 'private-data',
        'validation_issues': [{'code': 'INVALID_PARAMETERS', 'path': 'private-path', 'message': 'private-detail'}]}
    generator = LLMTestPlanGenerator(LLMRouter([RegistrationProvider([error])]))
    with pytest.raises(RetryableLLMError):
        generator.generate_with_plan(input_step(), observed_registration())
    assert 'private-' not in generator.supervisor.repository.list_records()[0].model_dump_json()


def test_unknown_compound_requirement_retains_hashed_uninterpreted_clause_identity():
    from qa_agent.candidate_diagnostics import rejected_candidate_diagnostics
    from qa_agent.test_plan_validation import PlanValidationIssue
    step = input_step()
    step.expected = 'The registration form is displayed and private-data is persisted correctly.'
    diagnostics = rejected_candidate_diagnostics(plan(action('assert_visible', '#registration-form')), step,
        observed_registration(), PlanValidationError([PlanValidationIssue(code='EXPECTED_RESULT_NOT_COVERED', path='steps', message='fixture')]))
    assert diagnostics['coverage'] == 'UNKNOWN'
    assert diagnostics['requirement_clauses'][1]['kind'] == 'UNKNOWN'
    assert diagnostics['requirement_clauses'][1]['clause_hash']
    assert 'private-data' not in json.dumps(diagnostics)


def test_error_message_field_contains_remains_a_valid_text_assertion():
    step = Step(name='Verify error', description='Verify the validation message.',
                expected="The validation error field contains 'Email invalid'.", order=0)
    generate(step, plan(action('assert_text_contains', '#form-error', expected_text='Email invalid')),
             observed_registration())


def test_input_literal_containing_error_word_is_still_an_input_value_assertion():
    step = Step(name='Fill email', description="Enter 'error-value' into the email field.",
                expected="Email field contains 'error-value'.", order=0)
    generate(step, plan(action('fill', '#email', value='error-value'),
                        action('assert_value', '#email', expected='error-value')), observed_registration())


def test_new_input_action_has_strict_schema_and_all_editor_export_contracts():
    from qa_agent.llm.json_schema import qa_test_plan_schema
    from qa_agent.automation_editor import ACTION_EDITOR_FIELDS
    from qa_agent.testplan_export import ACTION_EXPORT_HANDLERS
    from qa_agent.models import QATestStep
    variant = next(v for v in qa_test_plan_schema()['properties']['steps']['items']['anyOf']
                   if v['properties']['action']['enum'] == ['assert_value'])
    assert variant['properties']['parameters']['properties']['expected']['type'] == 'string'
    assert ACTION_EDITOR_FIELDS['assert_value'] == ('selector', 'expected')
    assertion = QATestStep(action='assert_value', parameters={'selector': '#email', 'expected': 'fixture'})
    snippets = ['\n'.join(emit(assertion.action, assertion.parameters, 0)) for emit in ACTION_EXPORT_HANDLERS['assert_value']]
    assert 'to_have_value' in snippets[0] and 'toHaveValue' in snippets[1] and 'ToHaveValueAsync' in snippets[2]


def test_rejected_diagnostics_persist_and_render_without_changing_historical_operation(tmp_path):
    import sqlite3
    from datetime import datetime, timezone
    from qa_agent.reliability import SQLiteReliabilityRepository, ReliabilityRecord, AutomationReliabilitySupervisor
    from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
    from qa_agent.web import LocalWebApplication
    repo = SQLiteReliabilityRepository(tmp_path / 'diagnostics.sqlite3')
    historical = ReliabilityRecord(test_step_id=input_step().id, settings=ReliabilitySettings(),
        started_at=datetime.now(timezone.utc), outcome='NEEDS_ATTENTION')
    data = historical.model_dump(mode='json')
    data.pop('effective_provider_order')  # Pre-M2.2 serialized record shape.
    original = json.dumps(data)
    with sqlite3.connect(repo.path) as connection:
        connection.execute('INSERT INTO automation_reliability_operations VALUES (?, ?, ?)',
                           (str(historical.id), historical.started_at.isoformat(), original))
    candidate = plan(action('fill', '#email', value='private-input'), action('assert_unchecked', '#form-error'))
    supervisor = AutomationReliabilitySupervisor(repo)
    generator = diagnostic_generator(LLMRouter([RegistrationProvider([candidate])]), supervisor)
    with pytest.raises(PlanValidationError):
        generator.generate_with_plan(input_step(), observed_registration())
    record = next(r for r in repo.list_records() if r.id != historical.id)
    restarted = SQLiteReliabilityRepository(repo.path)
    assert restarted.get(record.id).attempts[0].candidate_diagnostics['actions'][1]['assertion']
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), reliability=supervisor)
    try:
        html = app.handle('GET', f'/settings/reliability/{record.id}').body.decode()
        assert 'Rejected candidate diagnostics' in html and 'ACTION_TARGET_MISMATCH' in html
        assert 'Effective provider order' in html and 'selector_hash' in html
        assert 'private-input' not in html and '#form-error' not in html
    finally:
        app.close()
    with sqlite3.connect(repo.path) as connection:
        assert connection.execute('SELECT record_json FROM automation_reliability_operations WHERE id = ?',
                                  (str(historical.id),)).fetchone()[0] == original
