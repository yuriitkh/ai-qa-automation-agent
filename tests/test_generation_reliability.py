"""M2.1 generation regressions. All provider responses and DOM evidence are local."""
import json

import pytest

from qa_agent.expected_result_coverage import ExpectedResultCoverageStatus, expected_result_coverage
from qa_agent.browser_discovery import MAX_SELECTOR_CHARS, MAX_SNAPSHOT_CHARS, _build_snapshot
from qa_agent.llm.router import LLMRouter
from qa_agent.models import DiscoveryResult, DiscoveryStatus, QATestPlan, TestStep as Step
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from tests.test_registration_coverage import RegistrationProvider

URL = "http://127.0.0.1:9876/registration"


def observed_registration(*, required=False):
    controls = [
        {"kind": "input", "tag": "input", "selector": selector,
         "input_type": kind, "accessible_name": label, "form_selector": "#registration-form",
         "visible": True, "required": required,
         "error_selectors": [error] if required else []}
        for selector, kind, label, error in (
            ("#email", "email", "Email", "#email-error"),
            ("#password", "password", "Password", "#password-error"),
        )
    ]
    controls.append({"kind": "button", "tag": "button", "selector": "#submit",
        "button_type": "submit", "form_selector": "#registration-form", "visible": True})
    states = [
        {"tag": "p", "role": "alert", "selector": "#form-error", "visible": False,
         "form_selector": "#registration-form"},
        {"tag": "p", "role": "status", "selector": "#registration-success", "visible": False,
         "form_selector": "#registration-form"},
        {"tag": "p", "role": "alert", "selector": "#search-error", "visible": False,
         "form_selector": "#search-form"},
    ]
    if required:
        states += [{"tag": "span", "role": "", "selector": f"#{name}-error", "visible": False,
                    "form_selector": "#registration-form", "error_for": [f"#{name}"]}
                   for name in ("email", "password")]
    return DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=URL, snapshot={
        "forms": [{"tag": "form", "selector": "#registration-form", "id": "registration-form",
                   "accessible_name": "Registration", "visible": True,
                   "required_field_count": 2 if required else 0, "no_validate": True}],
        "interactive_elements": controls, "state_elements": states,
        "headings": [{"tag": "h1", "selector": "#registration-heading", "text": "Registration"}],
    })


def action(name, selector=None, **parameters):
    if selector is not None:
        parameters["selector"] = selector
    return {"action": name, "parameters": parameters}


def plan(*actions):
    return QATestPlan(url=URL, steps=list(actions))


def generate(step, candidate, discovery):
    provider = RegistrationProvider([candidate])
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    frozen = candidate.model_dump_json()
    result = generator.generate_with_plan(step, discovery)
    assert len(provider.calls) == 1
    assert candidate.model_dump_json() == frozen == result.test_plan_version.qa_test_plan.model_dump_json()
    gates = generator.supervisor.repository.list_records()[0].attempts[0].quality_gates
    assert set(gates.values()) == {"PASSED"}
    assert expected_result_coverage(step, result.test_plan_version).is_sufficient
    return result, provider


def reject(step, candidate, discovery, code="EXPECTED_RESULT_NOT_COVERED"):
    provider = RegistrationProvider([candidate])
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    with pytest.raises(PlanValidationError) as caught:
        generator.generate_with_plan(step, discovery)
    assert caught.value.issues[0].code == code
    assert len(provider.calls) == 1
    record = generator.supervisor.repository.list_records()[0]
    assert record.outcome == "NEEDS_ATTENTION" and record.candidate_version_id is None
    return caught.value


@pytest.mark.parametrize("expected", [
    "All fields accept the input without error.",
    "Entered values are accepted without validation errors.",
])
def test_grounded_input_acceptance_without_errors_has_consistent_generation_coverage(expected):
    step = Step(name="Enter valid unique test data into all required fields",
                description="Fill the registration fields.", expected=expected, order=0)
    candidate = plan(action("fill", "#email", value="local@example.test"),
                     action("fill", "#password", value="local-fixture-data"),
                     action("assert_hidden", "#form-error"))
    _, provider = generate(step, candidate, observed_registration())
    assert "input acceptance" in provider.calls[0][0].lower()


@pytest.mark.parametrize("mutation", ["no_assertion", "wrong_form", "before_inputs", "no_inputs"])
def test_input_acceptance_cannot_be_proved_by_arbitrary_absence(mutation):
    step = Step(name="Enter registration data", description="Fill the registration fields.",
                expected="Entered values are accepted without validation errors.", order=0)
    inputs = [action("fill", "#email", value="local@example.test"), action("fill", "#password", value="local-data")]
    check = action("assert_hidden", "#search-error" if mutation == "wrong_form" else "#form-error")
    actions = inputs if mutation == "no_assertion" else [check, *inputs] if mutation == "before_inputs" else [check] if mutation == "no_inputs" else [*inputs, check]
    reject(step, plan(*actions), observed_registration())


def test_acceptance_without_errors_does_not_erase_an_independent_outcome():
    step = Step(name="Enter registration data", description="Fill the fields.",
                expected="Entered values are accepted without error and data is persisted correctly.", order=0)
    candidate = plan(action("fill", "#email", value="local@example.test"), action("assert_hidden", "#form-error"))
    reject(step, candidate, observed_registration())
    assert expected_result_coverage(step, candidate, discovery=observed_registration()).status == ExpectedResultCoverageStatus.UNKNOWN


@pytest.mark.parametrize("selector", ["#registration-form", "#opaque-form"])
def test_registration_form_visibility_uses_actual_structural_identity(selector):
    discovery = observed_registration()
    discovery.snapshot["forms"][0].update(selector=selector, id=selector[1:])
    step = Step(name="Open Registration Page", description="Navigate to the registration page.",
                expected="The registration form is displayed", order=0)
    candidate = plan(action("navigate", url=URL), action("assert_visible", selector))
    _, provider = generate(step, candidate, discovery)
    assert "form" in provider.calls[0][2]


@pytest.mark.parametrize("assertion", [action("assert_page_loaded"), action("assert_visible", "#registration-heading")])
def test_navigation_or_registration_heading_does_not_prove_form_visibility(assertion):
    step = Step(name="Open Registration Page", description="Navigate to the registration page.",
                expected="The registration form is displayed", order=0)
    reject(step, plan(action("navigate", url=URL), assertion), observed_registration())


def test_unobserved_or_ai_suggested_form_identity_is_still_rejected():
    discovery = observed_registration().model_copy(update={"status": DiscoveryStatus.PARTIAL})
    discovery.snapshot["forms"] = []
    for collection in ("interactive_elements", "state_elements"):
        for record in discovery.snapshot[collection]:
            record.pop("form_selector", None)
    step = Step(name="Open Registration Page", description="Navigate to registration.",
                expected="The registration form is displayed", order=0)
    reject(step, plan(action("assert_visible", "#registration-form")), discovery, "DISCOVERY_SELECTOR_MISMATCH")


def submit_step(expected="Submission is blocked and validation errors are displayed for required fields."):
    return Step(name="Submit empty registration form", description="Leave the required fields empty and submit.",
                expected=expected, order=0)


def blocked_plan(*field_errors):
    return plan(action("click", "#submit"), action("assert_visible", "#form-error"),
                *(action("assert_visible", selector) for selector in field_errors),
                action("assert_hidden", "#registration-success"))


def test_compound_submission_blocked_and_all_required_errors_are_verifiable_when_grounded():
    generate(submit_step(), blocked_plan("#email-error", "#password-error"), observed_registration(required=True))


@pytest.mark.parametrize("mutation", ["global_only", "missing_field", "missing_success", "missing_submit", "before_submit"])
def test_compound_required_field_checks_cannot_accept_partial_or_pre_action_evidence(mutation):
    candidate = blocked_plan("#email-error", "#password-error")
    actions = candidate.model_dump()["steps"]
    if mutation == "global_only":
        actions = [a for a in actions if a["parameters"].get("selector") not in {"#email-error", "#password-error"}]
    elif mutation == "missing_field":
        actions = [a for a in actions if a["parameters"].get("selector") != "#password-error"]
    elif mutation == "missing_success":
        actions = [a for a in actions if a["action"] != "assert_hidden"]
    elif mutation == "missing_submit":
        actions = actions[1:]
    else:
        actions = [*actions[1:], actions[0]]
    reject(submit_step(), plan(*actions), observed_registration(required=True))


@pytest.mark.parametrize("mode", ["fixture_without_required_fields", "native_only", "truncated_fields"])
def test_missing_required_field_verification_capability_remains_unknown(mode):
    discovery = observed_registration(required=mode != "fixture_without_required_fields")
    if mode == "native_only":
        discovery.snapshot["forms"][0]["no_validate"] = False
        for control in discovery.snapshot["interactive_elements"]:
            control["error_selectors"] = []
    elif mode == "truncated_fields":
        discovery.snapshot["forms"][0]["required_field_count"] = 3
    candidate = blocked_plan()
    error = reject(submit_step(), candidate, discovery)
    assert expected_result_coverage(submit_step(), candidate, discovery=discovery).status == ExpectedResultCoverageStatus.UNKNOWN
    assert "required" in error.issues[0].message.lower()


def test_visible_error_alone_or_hidden_success_alone_cannot_prove_blocked_submission():
    step = submit_step("Form submission is blocked.")
    for check in (action("assert_visible", "#form-error"), action("assert_hidden", "#registration-success")):
        reject(step, plan(action("click", "#submit"), check), observed_registration())
    generate(step, blocked_plan(), observed_registration())


def test_required_error_check_cannot_use_an_unbound_alternative_selector():
    discovery = observed_registration(required=True)
    discovery.snapshot['interactive_elements'][0]['error_selectors'].append('#registration-success')
    candidate = blocked_plan('#registration-success', '#password-error')
    reject(submit_step(), candidate, discovery)


@pytest.mark.parametrize('expected', [
    'The account is accepted without error.',
    'Entered values are accepted by the server without error.',
    'A confirmation is displayed without error.',
])
def test_input_acceptance_normalization_does_not_approve_other_outcomes(expected):
    step = Step(name='Fill registration fields', description='Fill the registration fields.', expected=expected, order=0)
    candidate = plan(action('fill', '#email', value='local@example.test'), action('assert_hidden', '#form-error'))
    reject(step, candidate, observed_registration())


@pytest.mark.parametrize('expected', [
    'Submission is blocked. Validation errors are displayed for all required fields.',
    'Submission is blocked; all required fields display validation errors.',
])
def test_compound_without_and_keeps_both_required_errors_and_blocked_submission(expected):
    discovery = observed_registration(required=True)
    complete = blocked_plan('#email-error', '#password-error')
    generate(submit_step(expected), complete, discovery)
    partial = plan(*(a for a in complete.steps if a.action != 'assert_hidden'))
    reject(submit_step(expected), partial, discovery)


@pytest.mark.parametrize('expected', [
    'Validation errors are logged for all required fields.',
    'Form submission is blocked by the server transaction.',
])
def test_dom_assertions_do_not_prove_logging_or_transaction_outcomes(expected):
    step = submit_step(expected)
    candidate = blocked_plan('#email-error', '#password-error')
    reject(step, candidate, observed_registration(required=True))
    assert expected_result_coverage(step, candidate, discovery=observed_registration(required=True)).status == ExpectedResultCoverageStatus.UNKNOWN


def test_form_and_error_relationships_are_bounded_and_do_not_copy_values_or_aggregate_text():
    raw = observed_registration(required=True).snapshot
    raw['forms'][0].update(text='private aggregate input content', value='private-input-value')
    raw['interactive_elements'][0]['value'] = 'private-input-value'
    raw['state_elements'][0]['text'] = 'private-hidden-error-content'
    raw['forms'] *= 20
    snapshot = _build_snapshot(raw)
    data = json.loads(snapshot)
    assert len(snapshot) <= MAX_SNAPSHOT_CHARS and len(data['forms']) <= 8
    assert 'private-' not in snapshot
    assert data['forms'][0]['required_field_count'] == 2
    assert data['interactive_elements'][0]['error_selectors'] == ['#email-error']
    assert next(r for r in data['state_elements'] if r['selector'] == '#email-error')['error_for'] == ['#email']


def test_overlong_or_malformed_relationships_are_dropped_without_inventing_selectors():
    raw = observed_registration(required=True).snapshot
    long_selector = '#' + 'x' * MAX_SELECTOR_CHARS
    raw['forms'][0]['selector'] = long_selector
    raw['interactive_elements'][0]['form_selector'] = long_selector
    raw['interactive_elements'][0]['error_selectors'] = [long_selector]
    raw['state_elements'][0]['error_for'] = None
    raw['state_elements'][0]['form_selector'] = long_selector
    data = json.loads(_build_snapshot(raw))
    assert not data['forms']
    assert 'form_selector' not in data['interactive_elements'][0]
    assert not data['interactive_elements'][0]['error_selectors']
    assert 'form_selector' not in data['state_elements'][0]


def test_generic_negative_validation_after_input_still_requires_the_same_form():
    step = Step(name='Fill registration fields', description='Fill the fields.',
                expected='No validation errors are displayed.', order=0)
    candidate = plan(action('fill', '#email', value='local@example.test'), action('assert_hidden', '#search-error'))
    reject(step, candidate, observed_registration())


@pytest.mark.parametrize('include_password_error', [False, True])
def test_known_field_errors_all_need_absence_checks_after_input(include_password_error):
    discovery = observed_registration(required=True)
    step = Step(name='Fill registration fields', description='Fill the fields.',
                expected='Entered values are accepted without error.', order=0)
    checks = [action('assert_hidden', '#form-error'), action('assert_hidden', '#email-error')]
    if include_password_error:
        checks.append(action('assert_hidden', '#password-error'))
    candidate = plan(action('fill', '#email', value='local@example.test'),
                     action('fill', '#password', value='fixture-data'), *checks)
    if include_password_error:
        generate(step, candidate, discovery)
    else:
        reject(step, candidate, discovery)


def test_checkbox_input_can_verify_acceptance_without_an_artificial_text_fill():
    discovery = observed_registration()
    discovery.snapshot['interactive_elements'].append({'kind': 'checkbox', 'tag': 'input',
        'input_type': 'checkbox', 'selector': '#terms', 'form_selector': '#registration-form'})
    step = Step(name='Check required terms', description='Set the terms checkbox.',
                expected='All fields accept the input without error.', order=0)
    generate(step, plan(action('check', '#terms'), action('assert_hidden', '#form-error')), discovery)


@pytest.mark.parametrize('value', ['', 'non-empty-data'])
def test_empty_required_field_scenario_can_clear_fields_but_cannot_populate_them(value):
    discovery = observed_registration(required=True)
    candidate = plan(action('fill', '#password', value=value), *blocked_plan('#email-error', '#password-error').steps)
    if value:
        reject(submit_step(), candidate, discovery, 'TESTSTEP_BOUNDARY_VIOLATION')
    else:
        generate(submit_step(), candidate, discovery)
