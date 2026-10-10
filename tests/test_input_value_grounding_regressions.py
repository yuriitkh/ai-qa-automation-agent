"""TC-0045 reproductions: local provider fixtures and synthetic deterministic DOM only."""
import json

import pytest

from qa_agent.assertion_grounding import classify_assertions
from qa_agent.candidate_diagnostics import rejected_candidate_diagnostics
from qa_agent.expected_result_coverage import (
    ExpectedResultCoverageStatus, _expectation_kind, assertion_subject_matches, expected_result_coverage,
)
from qa_agent.llm.router import LLMRouter
from qa_agent.models import TestStep as Step
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError, PlanValidationIssue
from tests.test_generation_reliability import action, observed_registration, plan
from tests.test_registration_coverage import RegistrationProvider


ORIGINAL = (
    'Open the registration page. Enter an invalid email address such as invalid-email '
    'and valid data in the other required fields. Attempt to submit. Verify that the '
    'invalid email is rejected and successful registration is not confirmed.'
)
PRESENCE = 'The entered data is present in the form fields.'


def input_step(description='Enter an invalid email address and valid data in the other required fields.',
               expected=PRESENCE):
    return Step(name='Enter registration data', description=description, expected=expected, order=1)


def candidate(value='invalid-email'):
    return plan(action('fill', '#email', value=value), action('assert_value', '#email', expected=value))


def run_generation(step, proposed, context=None):
    generator = LLMTestPlanGenerator(LLMRouter([RegistrationProvider([proposed])]))
    frozen = proposed.model_dump_json()
    result = generator.generate_with_plan(step, observed_registration(), requirement_context=context)
    record = generator.supervisor.repository.list_records()[0]
    assert set(record.attempts[0].quality_gates.values()) == {'PASSED'}
    assert result.test_plan_version.qa_test_plan.model_dump_json() == frozen
    return result


def reject_generation(step, proposed, context=None, code='UNGROUNDED_ASSERTION'):
    generator = LLMTestPlanGenerator(LLMRouter([RegistrationProvider([proposed])]))
    with pytest.raises(PlanValidationError) as caught:
        generator.generate_with_plan(step, observed_registration(), requirement_context=context)
    assert caught.value.issues[0].code == code
    record = generator.supervisor.repository.list_records()[0]
    assert record.candidate_version_id is None and record.outcome == 'NEEDS_ATTENTION'
    return record.attempts[0]


def test_tc0045_bare_example_survives_later_output_negation():
    result = run_generation(input_step(), candidate(), ORIGINAL)
    assert result.test_plan_version.assertion_grounding[0].category.value == 'REQUIREMENT_GROUNDED'


@pytest.mark.parametrize('marker', ['such as', 'for example', 'e.g.'])
@pytest.mark.parametrize('value', ['invalid-email', 'test.user+tag@example.test'])
def test_bare_example_markers_preserve_exact_single_token(marker, value):
    run_generation(input_step(), candidate(value), f'Enter an email address {marker} {value}.')


@pytest.mark.parametrize('literal,value', [
    ("'test.user@example.test'", 'test.user@example.test'),
    ('"o\'reilly@example.test"', "o'reilly@example.test"),
    ("'a\"b@example.test'", 'a"b@example.test'),
    ('"not confirmed; password. Never enter anything else"', 'not confirmed; password. Never enter anything else'),
    ("''", ''), ("' '", ' '), ("'value.'", 'value.'),
])
def test_quoted_values_are_preserved_and_not_parsed_as_instructions(literal, value):
    step = input_step(f'Enter {literal} into the email field.', f'Email field contains {literal}.')
    run_generation(step, candidate(value))


@pytest.mark.parametrize('quoted', [False, True])
@pytest.mark.parametrize('separator', ['. ', '; ', '\n', ' and verify '])
def test_output_negation_does_not_negate_the_input_instruction(quoted, separator):
    value = '"invalid-email"' if quoted else 'invalid-email'
    context = f'Enter an invalid email such as {value}{separator}successful registration is not confirmed.'
    run_generation(input_step(), candidate(), context)


@pytest.mark.parametrize('context', [
    'Enter an invalid email address.',
    'Enter an invalid email address such as invalid-email.',
    'Enter an email address such as "invalid-email".',
    'Enter "declared-password" into the password field.',
    'The user must enter "invalid-email" into the email field.',
    'Use "invalid-email" for the email field.',
    'Email field must contain "invalid-email".',
])
def test_atomic_elaboration_cannot_supply_or_replace_original_input_data(context):
    reject_generation(input_step('Enter "invented-value" into the email field.'), candidate('invented-value'), context)


def test_atomic_conflict_is_rejected_even_when_the_candidate_matches_the_original():
    reject_generation(input_step('Enter "different" into the email field.'), candidate(),
                      'Enter an email such as invalid-email.')


@pytest.mark.parametrize('separator', ['. ', '; ', '\n', ' but '])
@pytest.mark.parametrize('restriction', [
    'Never enter "invalid-email" into the email field.',
    'Do not use "invalid-email" in the email field.',
    'Email must not contain "invalid-email".',
])
def test_later_original_restrictions_are_checked_before_acceptance(separator, restriction):
    reject_generation(input_step(), candidate(), 'Enter an email such as invalid-email' + separator + restriction)


def test_later_atomic_restriction_cannot_be_overridden_by_matching_original():
    reject_generation(input_step('Enter "invalid-email" into email; never enter "invalid-email" into email.'),
                      candidate(), 'Enter an email such as invalid-email.')


def test_a_different_forbidden_value_does_not_negate_the_declared_value():
    run_generation(input_step(), candidate(),
                   'Enter an email such as invalid-email. Never enter "other-value" into the email field.')


@pytest.mark.parametrize('context', [
    'Enter "invalid-email" into email. Enter "different" into email.',
    'Enter "invalid-email" or "different" into email.',
    'Enter an email such as invalid-email or different.',
    'Enter an email such as invalid-email and different.',
    'Enter "invalid-email" or different into email.',
    'Enter an email such as invalid email.',
    'Enter an email such as invalid-email$extra.',
    'Enter "invalid-email" into password and "different" into email.',
    'Enter an email such as invalid-email; enter an email such as "unterminated.',
])
def test_ambiguous_or_conflicting_original_declarations_fail_closed(context):
    reject_generation(input_step(), candidate(), context)


def test_ambiguous_atomic_binding_is_not_rescued_by_a_matching_original():
    reject_generation(input_step('Enter "invalid-email" into email and password.'), candidate(),
                      'Enter an email such as invalid-email.')


@pytest.mark.parametrize('context', [None, 'Enter an email such as invalid-email.'])
def test_uninterpreted_atomic_prohibition_is_not_ignored(context):
    reject_generation(input_step('Enter "invalid-email" into email. Never enter invalid-email into email.'),
                      candidate(), context)


def test_unsupported_atomic_binding_cannot_contradict_the_original():
    reject_generation(input_step('The user must enter "different" into the email field.'), candidate(),
                      'Enter an email such as invalid-email.')


@pytest.mark.parametrize('description,context', [
    ('Enter an invalid email address.', None),
    ('Enter "invented" into email.', 'Enter an invalid email address.'),
    ('Enter an invalid email address.', 'Open the "invented" page. Enter an invalid email address.'),
    ('Email input contains "invented".', None),
    ('Enter data into the form such as invented.', None),
])
def test_values_cannot_establish_their_own_truth(description, context):
    reject_generation(input_step(description), candidate('invented'), context)


@pytest.mark.parametrize('mutation', ['mismatch', 'missing_fill', 'changed_fill', 'other_control'])
def test_original_example_does_not_bypass_fill_and_control_binding(mutation):
    proposed = candidate()
    if mutation == 'mismatch':
        proposed.steps[-1].parameters['expected'] = 'different'
    elif mutation == 'missing_fill':
        proposed.steps = proposed.steps[1:]
    elif mutation == 'changed_fill':
        proposed = plan(*proposed.steps[:1], action('fill', '#email', value='different'), proposed.steps[-1])
    else:
        for item in proposed.steps:
            item.parameters['selector'] = '#password'
    reject_generation(input_step(), proposed, ORIGINAL)


@pytest.mark.parametrize('assertion', ['assert_text_contains', 'assert_visible'])
def test_input_examples_never_ground_product_output(assertion):
    step = input_step(expected='A confirmation message is displayed.')
    proposed = plan(action('fill', '#email', value='invalid-email'),
                    action(assertion, '#registration-success', expected_text='invalid-email'))
    reject_generation(step, proposed, ORIGINAL)


def test_action_only_presence_diagnostics_have_no_mandatory_form_clause():
    step = input_step()
    error = PlanValidationError([PlanValidationIssue(code='UNGROUNDED_ASSERTION', path='steps[1].parameters', message='fixture')])
    diagnostics = rejected_candidate_diagnostics(candidate('private-value'), step, observed_registration(), error)
    assert _expectation_kind(PRESENCE) is None
    assert diagnostics['coverage'] == 'NO_VERIFICATION_REQUIRED'
    assert diagnostics['requirement_clauses'] == [] and diagnostics['clauses_omitted'] == 0
    assert diagnostics['uninterpreted_requirement'] is False
    assert diagnostics['requirement_hash']
    assert 'private-value' not in json.dumps(diagnostics) and '#email' not in json.dumps(diagnostics)


@pytest.mark.parametrize('assertion', ['input', 'form'])
def test_explicit_collective_retention_requires_more_than_one_assertion(assertion):
    step = input_step('Verify entered data in the form fields.', PRESENCE)
    proposed = candidate() if assertion == 'input' else plan(action('assert_visible', '#registration-form'))
    attempt = reject_generation(step, proposed, ORIGINAL, code='EXPECTED_RESULT_NOT_COVERED')
    assert attempt.candidate_diagnostics['coverage'] == 'UNKNOWN'
    assert attempt.candidate_diagnostics['uninterpreted_requirement'] is True
    assert all(row['kind'] != 'form_visible' and not row['covered']
               for row in attempt.candidate_diagnostics['requirement_clauses'])


@pytest.mark.parametrize('expected', [
    'All form fields contain "invalid-email".',
    'Every required field contains "invalid-email".',
    'Each input contains "invalid-email".',
    'Email field contains "invalid-email" in all fields.',
    'All required fields are visible.',
])
def test_collective_field_wording_cannot_bind_to_a_single_control(expected):
    step = input_step('Verify all field values.', expected)
    attempt = reject_generation(step, candidate(), ORIGINAL, code='EXPECTED_RESULT_NOT_COVERED')
    assert attempt.candidate_diagnostics['coverage'] == 'UNKNOWN'


def test_two_explicit_field_requirements_need_both_field_assertions():
    context = 'Enter "invalid-email" into email. Enter "valid-password" into password.'
    step = input_step(context, 'Email field contains "invalid-email" and Password field contains "valid-password".')
    proposed = plan(action('fill', '#email', value='invalid-email'),
                    action('fill', '#password', value='valid-password'),
                    action('assert_value', '#email', expected='invalid-email'))
    attempt = reject_generation(step, proposed, context, code='EXPECTED_RESULT_NOT_COVERED')
    assert attempt.candidate_diagnostics['coverage'] == 'PARTIALLY_COVERED'
    assert [row['covered'] for row in attempt.candidate_diagnostics['requirement_clauses']] == [True, False]
    proposed = plan(*proposed.steps, action('assert_value', '#password', expected='valid-password'))
    run_generation(step, proposed, context)


@pytest.mark.parametrize('selector,status', [
    ('#registration-form', ExpectedResultCoverageStatus.COVERED),
    ('#registration-heading', ExpectedResultCoverageStatus.NOT_COVERED),
])
def test_genuine_form_visibility_still_requires_the_discovered_form(selector, status):
    step = input_step('Verify the registration form.', 'The registration form is present.')
    proposed = plan(action('assert_visible', selector))
    assert _expectation_kind(step.expected) == 'form_visible'
    assert expected_result_coverage(step, proposed, discovery=observed_registration()).status == status
    assert bool(assertion_subject_matches(step, proposed, discovery=observed_registration())[0]) == (status == ExpectedResultCoverageStatus.COVERED)


def test_missing_genuine_form_assertion_still_fails_coverage():
    step = input_step('Verify the registration form.', 'The registration form is displayed.')
    assert expected_result_coverage(step, candidate(), discovery=observed_registration()).status == ExpectedResultCoverageStatus.NOT_COVERED


def test_unrelated_case_summary_retains_the_explicit_atomic_requirement():
    result = run_generation(input_step('Enter "test.user@example.test" into the email field.'),
                            candidate('test.user@example.test'), 'Exercise a retry after generation failure.')
    assert result.test_plan_version.assertion_grounding[0].category.value == 'REQUIREMENT_GROUNDED'


def test_related_generic_case_cannot_gain_literal_specificity_from_atomic_elaboration():
    reject_generation(input_step('Enter "invented-value" into the email field.'), candidate('invented-value'),
                      'Validate the registration workflow with invalid email.')


def test_observation_or_generated_fill_without_declared_data_cannot_ground_an_input_value():
    proposed = candidate()
    discovery = observed_registration()
    discovery.snapshot['visible_text_elements'] = [{'text': 'invalid-email', 'selector': '#example'}]
    entries = classify_assertions(proposed, input_step(), discovery)
    assert entries[0].category.value == 'INFERRED'


def two_input_candidate(second_value='synthetic-password', second_expected=None):
    """Synthetic values and selectors; historical diagnostics do not retain them."""
    return plan(action('fill', '#email', value='invalid-email'),
                action('fill', '#password', value=second_value),
                action('assert_value', '#email', expected='invalid-email'),
                action('assert_value', '#password', expected=second_value if second_expected is None else second_expected),
                action('assert_hidden', '#registration-success'))


@pytest.mark.parametrize('description', [
    'Enter an invalid email address and valid data in the other required fields.',
    'Enter an invalid email address. Enter "synthetic-password" into the password field.',
])
def test_tc0046_second_value_cannot_be_grounded_by_its_matching_fill_or_elaboration(description):
    step = input_step(description, 'The entered details are present in the form.')
    proposed = two_input_candidate()
    entries = classify_assertions(proposed, step, observed_registration(), requirement_context=ORIGINAL)
    assert [(item.step_index, item.category.value) for item in entries[:2]] == [
        (2, 'REQUIREMENT_GROUNDED'), (3, 'INFERRED'),
    ]
    attempt = reject_generation(step, proposed, ORIGINAL)
    assert attempt.quality_gates['assertion_grounding'] == 'FAILED'
    assert attempt.quality_gates['expected_result_coverage'] == 'NOT_RUN'
    assert attempt.candidate_diagnostics['validation_issues'] == [
        {'code': 'UNGROUNDED_ASSERTION', 'path': 'steps[3].parameters'},
    ]
    assert attempt.candidate_diagnostics['coverage'] == 'NO_VERIFICATION_REQUIRED'
    assert attempt.candidate_diagnostics['requirement_clauses'] == []
    assert 'synthetic-password' not in json.dumps(attempt.candidate_diagnostics)


def test_two_original_input_declarations_allow_both_exact_matching_assertions():
    context = ORIGINAL + ' Enter "synthetic-password" into the password field.'
    run_generation(input_step(expected='The entered details are present in the form.'),
                   two_input_candidate(), context)


@pytest.mark.parametrize('context,expected', [
    (ORIGINAL + ' Enter "synthetic-password" into password.', 'different-password'),
    (ORIGINAL + ' Enter "synthetic-password" into password. Never use "synthetic-password" in password.', 'synthetic-password'),
    (ORIGINAL + ' Enter "synthetic-password" into password. Enter "conflicting-password" into password.', 'synthetic-password'),
])
def test_second_input_keeps_equality_restrictions_and_original_conflict_protections(context, expected):
    attempt = reject_generation(input_step(expected='The entered details are present in the form.'),
                                two_input_candidate(second_expected=expected), context)
    assert attempt.candidate_diagnostics['validation_issues'] == [
        {'code': 'UNGROUNDED_ASSERTION', 'path': 'steps[3].parameters'},
    ]
