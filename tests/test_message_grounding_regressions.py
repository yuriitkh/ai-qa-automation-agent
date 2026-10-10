"""TC-02 reproductions using supplied requirements and synthetic DOM evidence.

These candidates are fixtures, not reconstructions of the unavailable provider
response for operation 86b0d9c5-48ec-4cff-8a7a-c31250b05900.
"""
import json

import pytest

from qa_agent.assertion_grounding import validate_assertion_grounding
from qa_agent.diagnostic_mode import failure_reasons
from qa_agent.expected_result_coverage import _expectation_kind, _result_clauses, expected_result_coverage
from qa_agent.llm.router import LLMRouter
from qa_agent.models import DiscoveryResult, DiscoveryStatus, InteractiveElement, QATestPlan, TestStep as Step
from qa_agent.reliability import ReliabilitySettings
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError, PlanValidationIssue
from tests.test_registration_coverage import RegistrationProvider


URL = 'http://127.0.0.1:8000/demo-target/registration'
MESSAGE = 'Enter a valid email address.'
ORIGINAL = (
    'Open the registration page. Leave all required fields empty and submit the form. '
    'Verify that registration is blocked, required fields show validation errors, '
    'and no successful registration confirmation appears.'
)
CLARIFIED = (
    "Open the registration page. Leave Email and Password empty and submit the form. "
    "Verify that the message 'Enter a valid email address.' is displayed. "
    "Verify that the successful registration confirmation is hidden."
)


def tc02_step(message=MESSAGE):
    return Step(name='Leave fields Email and Password empty and submit the form',
                description="Leave fields 'Email' and 'Password' empty and submit the form.",
                expected=f"The message '{message}' is displayed.", order=1)


def demo_discovery(url=URL):
    return DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=url,
        interactive_elements=[
            InteractiveElement(selector='#email', tag='input', kind='input', input_type='email', accessible_name='Email'),
            InteractiveElement(selector='#password', tag='input', kind='input', input_type='password', accessible_name='Password'),
            InteractiveElement(selector='#create-account', tag='button', kind='button', button_type='submit', accessible_name='Create account'),
        ], snapshot={'state_elements': [
            {'selector': '#form-error', 'tag': 'p', 'role': 'alert', 'visible': False, 'form_selector': '#registration-form'},
            {'selector': '#registration-success', 'tag': 'p', 'role': 'status', 'visible': False, 'form_selector': '#registration-form'},
        ], 'forms': [{'selector': '#registration-form', 'tag': 'form', 'required_field_count': 0}]})


def empty_submission(message=MESSAGE, selector='#form-error', action='assert_visible', url=URL):
    return QATestPlan(url=url, steps=[
        {'action': 'fill', 'parameters': {'selector': '#email', 'value': ''}},
        {'action': 'fill', 'parameters': {'selector': '#password', 'value': ''}},
        {'action': 'click', 'parameters': {'selector': '#create-account'}},
        {'action': action, 'parameters': {'selector': selector, 'expected_text': message}},
    ])


def generator_for(candidate):
    provider = RegistrationProvider([candidate])
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    generator.supervisor.repository.save_settings(ReliabilitySettings(diagnostic_level='DEBUG'))
    return generator, provider


@pytest.mark.parametrize('action', ['assert_visible', 'assert_text_contains'])
def test_tc02_literal_is_grounded_and_covered_without_broader_context(action):
    plan, step, discovery = empty_submission(action=action), tc02_step(), demo_discovery()
    _, entries = LLMTestPlanGenerator._validate_generated_plan(plan, discovery, step)
    assert entries[0].category.value == 'REQUIREMENT_GROUNDED'
    assert _expectation_kind(step.expected) == 'visible'
    assert expected_result_coverage(step, plan, discovery=discovery).is_sufficient


def test_related_original_cannot_be_upgraded_by_atomic_message_and_reports_provenance():
    proposed = empty_submission()
    generator, provider = generator_for(proposed)
    with pytest.raises(PlanValidationError) as caught:
        generator.generate_with_plan(tc02_step(), demo_discovery(), requirement_context=ORIGINAL)
    issue = caught.value.issues[0]
    assert issue.code == 'UNGROUNDED_ASSERTION' and issue.path == 'steps[3].parameters'
    assert failure_reasons(caught.value)[0]['reason_code'] == 'OUTPUT_VALUE_ONLY_IN_STEP'
    assert failure_reasons(caught.value)[0]['root_cause_known'] is True
    record = generator.supervisor.repository.list_records()[0]
    assert record.outcome == 'NEEDS_ATTENTION' and record.candidate_version_id is None
    assert len(provider.calls) == 1 and record.attempts[0].provider_status == 'SUCCESS'
    assert record.attempts[0].quality_gates == {
        'schema_and_actions': 'PASSED', 'locator_identity': 'PASSED', 'assertion_grounding': 'FAILED',
        'expected_result_coverage': 'NOT_RUN', 'step_boundaries': 'NOT_RUN',
    }
    diagnostic = json.dumps(record.attempts[0].candidate_diagnostics)
    assert MESSAGE not in diagnostic and '#email' not in diagnostic and '#form-error' not in diagnostic
    assert MESSAGE not in str(caught.value)


@pytest.mark.parametrize('action', ['assert_visible', 'assert_text_contains'])
def test_clarified_original_passes_all_generation_gates_without_product_pass(action):
    proposed = empty_submission(action=action)
    generator, provider = generator_for(proposed)
    frozen = proposed.model_dump_json()
    generated = generator.generate_with_plan(tc02_step(), demo_discovery(), requirement_context=CLARIFIED)
    assert generated.test_plan_version.qa_test_plan.model_dump_json() == frozen
    record = generator.supervisor.repository.list_records()[0]
    assert set(record.attempts[0].quality_gates.values()) == {'PASSED'}
    assert len(provider.calls) == 1
    task = provider.calls[0][0]
    assert 'an exact output literal found only in an elaborated TestStep is not independent evidence' in task
    assert 'never on Email or Password inputs' in task


@pytest.mark.parametrize('selector', ['#email', '#password'])
@pytest.mark.parametrize('action', ['assert_visible', 'assert_text_contains'])
def test_exact_message_cannot_be_verified_on_input_identity(selector, action):
    plan = empty_submission(selector=selector, action=action)
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(plan, demo_discovery(), tc02_step(), requirement_context=CLARIFIED)
    assert caught.value.issues[0].code == 'UNGROUNDED_ASSERTION'
    assert caught.value.issues[0].reason_code == 'ASSERTION_TARGET_MISMATCH'
    assert not expected_result_coverage(tc02_step(), plan, discovery=demo_discovery()).is_sufficient


@pytest.mark.parametrize('message', [
    'Enter a valid email address.', 'Email and Password are required.', 'Email but also password is missing.',
    'No errors are hidden; the URL address is disabled and selected.', 'Registration submitted.',
])
@pytest.mark.parametrize('quotes', [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’')])
def test_quoted_message_words_are_not_state_instructions_or_clause_boundaries(message, quotes):
    expected = f'The message {quotes[0]}{message}{quotes[1]} is displayed.'
    assert _expectation_kind(expected) == 'visible'
    assert len(_result_clauses(expected)) == 1
    step = tc02_step().model_copy(update={'expected': expected})
    assert expected_result_coverage(step, empty_submission(message), discovery=demo_discovery()).is_sufficient
    compound = expected + ' and the submit button is disabled.'
    assert len(_result_clauses(compound)) == 2
    assert not expected_result_coverage(step.model_copy(update={'expected': compound}), empty_submission(message),
                                        discovery=demo_discovery()).is_sufficient


@pytest.mark.parametrize('wrong', ['Email is required.', 'Enter a valid email.', 'Registration submitted.'])
def test_conflicting_but_authorized_message_cannot_cover_exact_expected_result(wrong):
    # Both literals are allowed sources; coverage must still require the current result.
    context = CLARIFIED + f" An additional message may say '{wrong}'."
    plan = empty_submission(wrong)
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(plan, demo_discovery(), tc02_step(), requirement_context=context)
    assert caught.value.issues[0].code == 'EXPECTED_RESULT_NOT_COVERED'


def test_structural_visibility_cannot_replace_required_exact_message():
    plan = empty_submission()
    plan.steps[-1].parameters.pop('expected_text')
    assert not expected_result_coverage(tc02_step(), plan, discovery=demo_discovery()).is_sufficient


@pytest.mark.parametrize('literal,message', [
    ('"O\'Reilly\'s email address and password."', "O'Reilly's email address and password."),
    ("'Say \"address\" and continue.'", 'Say "address" and continue.'),
])
def test_opposite_quote_characters_inside_message_remain_literal(literal, message):
    expected = f'The message {literal} is displayed.'
    assert _expectation_kind(expected) == 'visible' and len(_result_clauses(expected)) == 1
    assert expected_result_coverage(tc02_step().model_copy(update={'expected': expected}),
                                    empty_submission(message), discovery=demo_discovery()).is_sufficient


def test_invented_message_reports_absence_of_grounding_without_leaking_value():
    generator, _ = generator_for(empty_submission('invented-private-output'))
    with pytest.raises(PlanValidationError) as caught:
        generator.generate_with_plan(tc02_step(), demo_discovery(), requirement_context=CLARIFIED)
    assert caught.value.issues[0].reason_code == 'OUTPUT_VALUE_NOT_GROUNDED'
    assert 'invented-private-output' not in json.dumps(failure_reasons(caught.value))


def test_message_observed_elsewhere_does_not_ground_empty_error_container():
    discovery = demo_discovery()
    discovery.snapshot['visible_text_elements'] = [{'selector': '#help', 'tag': 'p', 'text': MESSAGE}]
    with pytest.raises(PlanValidationError) as caught:
        validate_assertion_grounding(empty_submission(), tc02_step(), discovery, requirement_context=ORIGINAL)
    assert caught.value.issues[0].reason_code == 'OUTPUT_VALUE_ONLY_IN_STEP'
    discovery.snapshot['visible_text_elements'].append({'selector': '#form-error', 'tag': 'p', 'text': MESSAGE})
    discovery.snapshot['state_elements'][0]['visible'] = True
    assert validate_assertion_grounding(empty_submission(), tc02_step(), discovery,
                                       requirement_context=ORIGINAL)[0].category.value == 'OBSERVATION_GROUNDED'


@pytest.mark.parametrize('error', [
    ValueError('unsupported exact assertion: guessed cause'),
    PlanValidationError([PlanValidationIssue(code='UNGROUNDED_ASSERTION', path='steps[2].parameters', message='')]),
    PlanValidationError([PlanValidationIssue(code='UNGROUNDED_ASSERTION', path='steps[2].parameters', message='', reason_code='UNKNOWN')]),
])
def test_missing_typed_evidence_remains_unknown(error):
    reason = failure_reasons(error)[0]
    assert reason['reason_code'] == 'UNKNOWN' and reason['root_cause_known'] is False


def test_generic_scenario_cannot_authorize_ai_discovery_message():
    discovery = demo_discovery().model_copy(update={'status': DiscoveryStatus.PARTIAL,
        'interactive_elements': [InteractiveElement(kind='text', tag='p', selector='#form-error', text=MESSAGE)]})
    with pytest.raises(PlanValidationError):
        validate_assertion_grounding(empty_submission(), tc02_step(), discovery, requirement_context=ORIGINAL)


def test_required_field_outcome_remains_unsupported_for_demo():
    step = tc02_step().model_copy(update={'expected': 'Required fields show validation errors.'})
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(empty_submission(), demo_discovery(), step, requirement_context=CLARIFIED)
    assert caught.value.issues[0].code == 'EXPECTED_RESULT_NOT_COVERED'
