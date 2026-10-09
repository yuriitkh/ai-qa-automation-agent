"""Local regressions for registration input checks and negative error coverage."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from qa_agent.assertion_grounding import classify_assertions
from qa_agent.browser_discovery import MAX_SNAPSHOT_CHARS, _build_snapshot
from qa_agent.execution_progress import ExecutionProgressReporter, ExecutionProgressStore, active_execution_progress
from qa_agent.expected_result_coverage import ExpectedResultCoverageError, ExpectedResultCoverageStatus, expected_result_coverage, validate_expected_result_coverage
from qa_agent.automation_lifecycle import AutomationLifecycleService, AutomationStatus
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.llm.usage_metadata import capture_openai_usage
from qa_agent.llm_usage import LLMUsageRepository, LLMUsageService
from qa_agent.models import DiscoveryResult, DiscoveryStatus, InteractiveElement, PlanVersionOrigin, QATestPlan, TestCase as Case, TestPlan as Plan, TestPlanVersion as Version, TestStep as Step
from qa_agent.pipeline import PipelineStageError, QATestPipeline
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.reliability import AutomationReliabilitySupervisor, InMemoryReliabilityRepository, ReliabilitySettings
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.test_case_review import TestCaseReviewService as CaseReview
from qa_agent.workflows import RegressionWorkflow, ValidationWorkflow, WorkflowOutcome


URL = "http://127.0.0.1:8000/demo-target/registration"


def registration_step(expected="All fields accept the input without error."):
    return Step(name="Fill registration fields", order=0,
        description="Fill all required fields in the registration form with valid information.", expected=expected)


def registration_discovery(url=URL):
    return DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=url,
        interactive_elements=[InteractiveElement(tag="input", kind="input", selector=selector)
                              for selector in ("#email", "#password")],
        snapshot={"state_elements": [{"selector": "#form-error", "tag": "p", "role": "alert", "visible": False}]})


def registration_plan(*assertions, url=URL):
    return QATestPlan(url=url, steps=[
        {"action": "fill", "parameters": {"selector": "#email", "value": "local@example.test"}},
        {"action": "fill", "parameters": {"selector": "#password", "value": "local-fixture-password"}},
        *assertions,
    ])


def assertion(action="assert_hidden", selector="#form-error"):
    return {"action": action, "parameters": {"selector": selector}}


@pytest.mark.parametrize("outcome", ["persistence", "confirmation"])
def test_full_generation_gates_reject_false_positive_expected_results(outcome):
    if outcome == "persistence":
        step = Step(name="Click Save", description="Click the Save button.",
                    expected="Data is persisted correctly.", order=0)
        plan = QATestPlan(url=URL, steps=[{"action": "click", "parameters": {"selector": "#save"}}])
        discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=URL,
            interactive_elements=[InteractiveElement(kind="button", tag="button", selector="#save")])
        context = step.description + " " + step.expected
    else:
        step = Step(name="Fill registration form", description="Fill the registration form with valid information.",
                    expected="A confirmation message is displayed.", order=0)
        plan = registration_plan(assertion("assert_visible", "#registration-form"))
        discovery = registration_discovery().model_copy(update={"snapshot": {
            "visible_text_elements": [{"selector": "#registration-form", "tag": "form", "text": "Registration form"}]
        }})
        context = "Fill the registration form and verify that a confirmation message is displayed."
    frozen = plan.model_dump_json()
    provider = RegistrationProvider([plan])
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    with pytest.raises(PlanValidationError) as caught:
        generator.generate_with_plan(step, discovery, requirement_context=context)
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"
    expected_status = ExpectedResultCoverageStatus.UNKNOWN if outcome == "persistence" else ExpectedResultCoverageStatus.NOT_COVERED
    assert expected_result_coverage(step, plan, discovery=discovery).status == expected_status
    assert len(provider.calls) == 1 and plan.model_dump_json() == frozen
    record = generator.supervisor.repository.list_records()[0]
    assert record.outcome == "NEEDS_ATTENTION" and record.candidate_version_id is None
    assert record.attempts[0].quality_gates["expected_result_coverage"] == "FAILED"


@pytest.mark.parametrize("name,description,expected,action", [
    ("Open login", "Navigate to login.", "The login page is opened.", {"action": "navigate", "parameters": {"url": URL}}),
    ("Enter email", "Enter email into the form.", "The email is entered.", {"action": "fill", "parameters": {"selector": "#email", "value": "local@example.test"}}),
    ("Click Save", "Click the Save button.", "The button is clicked.", {"action": "click", "parameters": {"selector": "#save"}}),
    ("Register", "Perform registration.", "The registration action is processed.", {"action": "click", "parameters": {"selector": "#save"}}),
    ("Click Save", "Click the Save button.", "Done.", {"action": "click", "parameters": {"selector": "#save"}}),
])
def test_full_generation_gates_preserve_explicit_action_only_results(name, description, expected, action):
    step = Step(name=name, description=description, expected=expected, order=0)
    plan = QATestPlan(url=URL, steps=[action])
    discovery = registration_discovery().model_copy(update={"snapshot": {"buttons": [{"selector": "#save"}]}})
    candidate, _ = LLMTestPlanGenerator._validate_generated_plan(plan, discovery, step)
    assert expected_result_coverage(step, candidate).status == ExpectedResultCoverageStatus.NO_VERIFICATION_REQUIRED


def confirmation_discovery():
    return registration_discovery().model_copy(update={"snapshot": {"visible_text_elements": [
        {"selector": "#registration-form", "tag": "form", "text": "Registration form"},
        {"selector": "#node-17", "tag": "p", "role": "status", "text": "Registration successful. Welcome."},
    ]}})


def confirmation_step():
    return Step(name="Fill registration form", description="Fill the registration form with valid information.",
                expected="A confirmation message is displayed.", order=0)


def test_full_generation_gates_accept_observed_semantic_subject_without_rewriting_plan():
    step, discovery = confirmation_step(), confirmation_discovery()
    plan = registration_plan(assertion("assert_visible", "#node-17"))
    frozen = plan.model_dump_json()
    generator = LLMTestPlanGenerator(LLMRouter([RegistrationProvider([plan])]))
    generated = generator.generate_with_plan(step, discovery)
    version = generated.test_plan_version
    assert version.qa_test_plan.model_dump_json() == frozen
    assert expected_result_coverage(step, version).status == ExpectedResultCoverageStatus.COVERED
    assert version.assertion_grounding[0].category.value == "REQUIREMENT_GROUNDED"
    assert version.assertion_grounding[0].covered_expectation_indexes == (0,)
    assert "Registration successful" not in version.model_dump_json()
    # Pure plan coverage cannot assume what an opaque selector represents.
    assert not expected_result_coverage(step, plan).is_sufficient
    changed_step = step.model_copy(update={"expected": "An error message is displayed."})
    changed_plan = version.model_copy(update={"qa_test_plan": registration_plan(assertion("assert_visible", "#registration-form"))})
    assert not expected_result_coverage(changed_step, version).is_sufficient
    assert not expected_result_coverage(step, changed_plan).is_sufficient


@pytest.mark.parametrize("text", ["Account created successfully", "Registration successful"])
@pytest.mark.parametrize("source", ["requirement", "discovery"])
def test_semantic_confirmation_text_remains_valid_when_required_or_observed(text, source):
    discovery = registration_discovery()
    context = None
    if source == "requirement":
        context = f'Fill the registration form and verify the exact confirmation text "{text}".'
    else:
        discovery.snapshot["visible_text_elements"] = [{"selector": "#node-17", "tag": "p", "text": text}]
    plan = registration_plan({"action": "assert_text_contains", "parameters": {"expected_text": text}})
    candidate, _ = LLMTestPlanGenerator._validate_generated_plan(plan, discovery, confirmation_step(), requirement_context=context)
    assert candidate == plan


@pytest.mark.parametrize("included", ["both", "confirmation", "absence"])
def test_full_generation_gates_preserve_compound_discovered_subjects(included):
    step = registration_step("A confirmation message is displayed and no errors are displayed.")
    discovery = confirmation_discovery()
    discovery.snapshot["state_elements"] = registration_discovery().snapshot["state_elements"]
    checks = []
    if included in {"both", "confirmation"}:
        checks.append(assertion("assert_visible", "#node-17"))
    if included in {"both", "absence"}:
        checks.append(assertion())
    generator = LLMTestPlanGenerator(LLMRouter([RegistrationProvider([registration_plan(*checks)])]))
    if included == "both":
        version = generator.generate_with_plan(step, discovery).test_plan_version
        assert expected_result_coverage(step, version).status == ExpectedResultCoverageStatus.COVERED
    else:
        with pytest.raises(PlanValidationError) as caught:
            generator.generate_with_plan(step, discovery)
        assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"
        assert expected_result_coverage(step, registration_plan(*checks), discovery=discovery).status == ExpectedResultCoverageStatus.PARTIALLY_COVERED


def test_semantically_named_but_unobserved_confirmation_selector_is_still_rejected():
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(
            registration_plan(assertion("assert_visible", "#invented-confirmation")), confirmation_discovery(), confirmation_step())
    assert caught.value.issues[0].code == "DISCOVERY_SELECTOR_MISMATCH"


@pytest.mark.parametrize("expected", ["The checkbox is unchecked.", "The checkbox is not checked."])
def test_negative_checkbox_state_uses_observed_subject_without_negation_token_overlap(expected):
    step = Step(name="Verify checkbox state", description="Verify the checkbox remains clear.", expected=expected, order=0)
    discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=URL,
        interactive_elements=[InteractiveElement(kind="checkbox", tag="input", selector="#node-17", accessible_name="Accept terms")])
    plan = QATestPlan(url=URL, steps=[assertion("assert_unchecked", "#node-17")])
    generator = LLMTestPlanGenerator(LLMRouter([RegistrationProvider([plan])]))
    version = generator.generate_with_plan(step, discovery).test_plan_version
    assert expected_result_coverage(step, version).is_sufficient
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(
            QATestPlan(url=URL, steps=[assertion("assert_checked", "#node-17")]), discovery, step)
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"


@pytest.mark.parametrize("parameters", [
    {"selector": "#registration-form"},
    {"selector": "#form-message"},
])
def test_other_observed_nodes_and_generic_message_tokens_cannot_supply_confirmation(parameters):
    discovery = confirmation_discovery()
    discovery.snapshot["visible_text_elements"].append({"selector": "#form-message", "tag": "p", "text": "Fill the registration form."})
    plan = registration_plan({"action": "assert_visible", "parameters": parameters})
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(plan, discovery, confirmation_step())
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"
    assert classify_assertions(plan, confirmation_step(), discovery)[0].category.value == "UNKNOWN"


def test_text_assertion_cannot_borrow_other_text_from_the_same_element():
    discovery = confirmation_discovery()
    discovery.snapshot["visible_text_elements"][0]["text"] = "Registration form. Confirmation message."
    plan = registration_plan({"action": "assert_text_contains", "parameters": {
        "selector": "#registration-form", "expected_text": "Registration form"}})
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(plan, discovery, confirmation_step())
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"


def test_form_visibility_cannot_borrow_a_child_confirmation_from_aggregate_text():
    discovery = confirmation_discovery()
    discovery.snapshot["visible_text_elements"][0].update({
        "text": "Registration form. Registration successful.",
        "accessible_name": "Registration form. Registration successful.",
    })
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(
            registration_plan(assertion("assert_visible", "#registration-form")), discovery, confirmation_step())
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"


@pytest.mark.parametrize("role", ["alert", "ALERT"])
def test_alert_role_alone_is_not_an_error_subject_for_negative_coverage(role):
    discovery = registration_discovery()
    discovery.snapshot["state_elements"].append({"selector": "#node-17", "tag": "p", "role": role, "visible": False})
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(registration_plan(assertion(selector="#node-17")), discovery, registration_step())
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"


def test_ai_suggestion_and_generic_status_role_cannot_establish_confirmation_subject():
    plan = registration_plan(assertion("assert_visible", "#node-17"))
    for discovery in (
        registration_discovery().model_copy(update={"snapshot": {"state_elements": [
            {"selector": "#node-17", "tag": "p", "role": "status", "visible": False}]}}),
        registration_discovery().model_copy(update={"status": DiscoveryStatus.PARTIAL,
            "interactive_elements": [*registration_discovery().interactive_elements,
                InteractiveElement(kind="text", selector="#node-17", text="Confirmation message")]}),
    ):
        with pytest.raises(PlanValidationError) as caught:
            LLMTestPlanGenerator._validate_generated_plan(plan, discovery, confirmation_step())
        assert caught.value.issues[0].code == (
            "DISCOVERY_SELECTOR_MISMATCH" if discovery.status == DiscoveryStatus.PARTIAL else "EXPECTED_RESULT_NOT_COVERED")
        assert expected_result_coverage(confirmation_step(), plan, discovery=discovery).status == ExpectedResultCoverageStatus.NOT_COVERED


def test_expected_outcome_subject_cannot_be_replaced_by_its_account_modifier():
    step = confirmation_step().model_copy(update={"expected": "The account confirmation is displayed."})
    discovery = confirmation_discovery()
    discovery.snapshot["visible_text_elements"].append({"selector": "#account-form", "tag": "form", "text": "Account profile form"})
    plan = registration_plan(assertion("assert_visible", "#account-form"))
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(plan, discovery, step)
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"


def test_page_load_wording_requires_a_supported_load_assertion():
    step = Step(name="Open local page", description="Open the local test page.", expected="The page loads.", order=0)
    navigate = {"action": "navigate", "parameters": {"url": URL}}
    missing = QATestPlan(url=URL, steps=[navigate])
    assert expected_result_coverage(step, missing).status == ExpectedResultCoverageStatus.NOT_COVERED
    complete = QATestPlan(url=URL, steps=[navigate, {"action": "assert_page_loaded"}])
    LLMTestPlanGenerator._validate_generated_plan(complete, registration_discovery(), step)


def test_page_load_assertion_does_not_verify_an_unsupported_server_load_outcome():
    step = Step(name="Click Save", description="Click Save.", expected="The server load is low.", order=0)
    plan = QATestPlan(url=URL, steps=[{"action": "assert_page_loaded"}])
    assert expected_result_coverage(step, plan).status == ExpectedResultCoverageStatus.UNKNOWN
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(plan, registration_discovery(), step)
    assert caught.value.issues[0].code == "EXPECTED_RESULT_NOT_COVERED"


def test_unsupported_compound_outcome_is_unknown_even_when_an_action_is_completed():
    step = Step(name="Click Save", description="Click Save.",
                expected="The button is clicked and data is persisted correctly.", order=0)
    assert expected_result_coverage(step, QATestPlan(url=URL, steps=[assertion("assert_visible", "#save")])).status == ExpectedResultCoverageStatus.UNKNOWN


@pytest.mark.parametrize("expected,action", [
    ("Data is persisted correctly.", {"action": "click", "parameters": {"selector": "#save"}}),
    ("A confirmation message is displayed.", assertion("assert_visible", "#registration-form")),
])
def test_saved_false_positive_plans_cannot_approve_validate_or_become_ready(tmp_path, expected, action):
    storage = create_sqlite_storage(tmp_path / "coverage.sqlite3")
    step = Step(name="Click Save", description="Click the registration form Save button.", expected=expected, order=0)
    case = Case(name="Registration coverage", description=step.description, base_url=URL, steps=[step])
    plan = Plan(test_step_id=step.id, name=step.name)
    version = Version(test_plan_id=plan.id, version=1, origin=PlanVersionOrigin.HUMAN_EDITED,
                      qa_test_plan=QATestPlan(url=URL, steps=[action]))
    storage.plan_store.save(step.id, version, test_plan=plan)
    frozen = version.model_dump_json()
    review = CaseReview(storage.test_case_review_repository, storage.plan_store)
    review.approve_test_case(case)
    with pytest.raises(ValueError, match="cover each expected result"):
        review.approve_for_validation(case)
    lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
    lifecycle.mark_automation_completed(case)
    assert lifecycle.mark_validation_completed(case, passed=True) == AutomationStatus.NEEDS_VALIDATION
    pins = PlanVersionSet.from_mapping({step.id: version.id})
    executions = []
    executor = PinnedExecutionService(storage.plan_store, PlanExecutionService(
        lambda candidate: executions.append(candidate) or {"status": "passed", "steps": []}, storage.execution_repository))
    with pytest.raises(ExpectedResultCoverageError):
        ValidationWorkflow(executor).run(case, pins)
    assert not executions
    # Old pins remain runnable without AI, but uncovered success is technical,
    # never PASS or a fabricated product defect.
    regression = RegressionWorkflow(executor, run_history=storage.run_history).run(case, pins)
    assert regression.outcome == WorkflowOutcome.AUTOMATION_EXECUTION_ERROR
    assert len(executions) == 1
    assert storage.plan_store.get_version(version.id).model_dump_json() == frozen


class RegistrationProvider(LLMProvider):
    model = "local-model"

    def __init__(self, candidates, *, name="OpenRouter", available=True):
        self.candidates, self.name, self.available, self.calls = list(candidates), name, available, []

    @property
    def is_available(self):
        return self.available

    def create_test_plan(self, task, target_url, page_snapshot):
        self.calls.append((task, target_url, page_snapshot))
        candidate = self.candidates.pop(0)
        if isinstance(candidate, Exception):
            raise candidate
        capture_openai_usage({"usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})
        return candidate


@pytest.mark.parametrize("expected", [
    "All fields accept the input without error.",
    "All fields accept the input without errors.",
    "Form submission is processed without client-side errors.",
    "No validation errors are displayed.",
])
def test_negative_error_expectation_requires_post_action_absence_check(expected):
    step = registration_step(expected)
    valid = registration_plan(assertion())
    assert validate_expected_result_coverage(step, valid).status == ExpectedResultCoverageStatus.COVERED
    for invalid in (
        registration_plan(),
        registration_plan(assertion("assert_visible")),
        registration_plan(assertion("assert_hidden", "#email")),
        registration_plan(assertion("assert_enabled", "#email")),
        QATestPlan(url=URL, steps=[assertion(), *registration_plan().steps]),
    ):
        assert expected_result_coverage(step, invalid).status == ExpectedResultCoverageStatus.NOT_COVERED
        with pytest.raises(PlanValidationError, match="after the input actions"):
            validate_expected_result_coverage(step, invalid)


def test_positive_error_expectation_keeps_its_existing_meaning_and_compound_requires_both():
    positive = registration_step("An error message is displayed.")
    assert validate_expected_result_coverage(positive, registration_plan(assertion("assert_visible"))).is_sufficient
    assert not expected_result_coverage(positive, registration_plan(assertion())).is_sufficient
    compound = registration_step("All fields accept input without error and the submit button is enabled.")
    assert expected_result_coverage(compound, registration_plan(assertion())).status == ExpectedResultCoverageStatus.PARTIALLY_COVERED
    assert validate_expected_result_coverage(compound, registration_plan(assertion(), assertion("assert_enabled", "#submit"))).is_sufficient


@pytest.mark.parametrize("expected", ["The confirmation is displayed without error.", "Without error, the confirmation is displayed."])
def test_absence_language_does_not_erase_another_state_in_the_same_clause(expected):
    step = registration_step(expected)
    assert expected_result_coverage(step, registration_plan(assertion())).status == ExpectedResultCoverageStatus.PARTIALLY_COVERED
    assert validate_expected_result_coverage(step, registration_plan(assertion(), assertion("assert_visible", "#confirmation"))).is_sufficient


def test_error_absence_grounding_requires_authoritative_requirement_and_never_invents_exact_text():
    plan = registration_plan(assertion())
    step = registration_step()
    discovery = registration_discovery()
    assert classify_assertions(plan, step, discovery)[0].category.value == "REQUIREMENT_GROUNDED"
    assert classify_assertions(plan, step, discovery,
        requirement_context="Fill the registration fields without error.")[0].category.value == "REQUIREMENT_GROUNDED"
    # A generated step cannot upgrade a related scenario into a new requirement.
    assert classify_assertions(plan, step, discovery,
        requirement_context="Fill the registration fields and verify the confirmation state is displayed.")[0].category.value == "UNKNOWN"
    invented = registration_plan({"action": "assert_text_contains", "parameters": {"selector": "#form-error", "expected_text": "All fields accepted successfully"}})
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(invented, discovery, step)
    assert caught.value.issues[0].code == "UNGROUNDED_ASSERTION"


@pytest.mark.parametrize("selector,code", [("#invented-error", "DISCOVERY_SELECTOR_MISMATCH"), ("#form-error", "EXPECTED_RESULT_NOT_COVERED")])
def test_generation_rejects_unobserved_identity_and_opposite_assertion(selector, code):
    provider = RegistrationProvider([registration_plan(assertion("assert_visible", selector))])
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    with pytest.raises(PlanValidationError) as caught:
        generator.generate_with_plan(registration_step(), registration_discovery())
    assert caught.value.issues[0].code == code
    assert provider.calls and caught.value.provider_response_succeeded
    assert generator.supervisor.repository.list_records()[0].candidate_version_id is None


def test_coverage_rejection_is_generation_error_without_plan_run_or_product_failure(tmp_path):
    storage = create_sqlite_storage(tmp_path / "rejected.sqlite3")
    step = registration_step()
    case = Case(name="User Registration and Confirmation Display", description=step.description, base_url=URL, steps=[step])
    missing = registration_plan()
    frozen = deepcopy(missing)
    unavailable = RegistrationProvider([], name="OpenAI", available=False)
    failing = RegistrationProvider([RetryableLLMError("local outage", category="PROVIDER_UNAVAILABLE")], name="First configured provider")
    fallback = RegistrationProvider([missing])
    usage_repo = LLMUsageRepository(storage.database_path)
    supervisor = AutomationReliabilitySupervisor(storage.reliability_repository)
    generator = LLMTestPlanGenerator(LLMRouter([unavailable, failing, fallback], usage_recorder=LLMUsageService(usage_repo)), supervisor)
    pipeline = QATestPipeline(None, generator, discovery=lambda _: registration_discovery(),
        runner=lambda _: pytest.fail("A rejected candidate must never execute"), plan_store=storage.plan_store,
        execution_repository=storage.execution_repository, run_history=storage.run_history)
    progress = ExecutionProgressStore()
    progress_id = progress.create(case.id, "AUTOMATION")
    with active_execution_progress(ExecutionProgressReporter(progress, progress_id)), pytest.raises(PipelineStageError) as caught:
        pipeline.run_test_case(case)
    assert isinstance(caught.value.__cause__, PlanValidationError)
    failure = progress.get(progress_id).automation_generation_failure
    assert failure.failure_category == "AUTOMATION_GENERATION_ERROR"
    assert "Provider responded successfully, but Expected Result Coverage" in failure.safe_reason
    assert "successful actions alone" in failure.safe_reason
    assert storage.plan_store.find(step.id) is None and not storage.plan_store.list_versions(step.id)
    assert not storage.execution_repository.list_for_test_step(step.id)
    assert not storage.run_history.list_for_test_case(case.id)
    assert missing == frozen  # Validation never fabricates an assertion.
    assert unavailable.calls == [] and len(failing.calls) == len(fallback.calls) == 1
    record = supervisor.repository.list_records()[0]
    assert record.outcome == "NEEDS_ATTENTION" and record.final_category == "EXPECTED_RESULT_NOT_COVERED"
    assert record.candidate_version_id is None and not record.repaired
    assert [a.status for a in record.attempts] == ["REJECTED", "REJECTED"]
    assert record.attempts[0].error_category == "PROVIDER_UNAVAILABLE"
    assert record.attempts[-1].quality_gates == {
        "schema_and_actions": "PASSED", "locator_identity": "PASSED",
        "assertion_grounding": "PASSED", "expected_result_coverage": "FAILED", "step_boundaries": "NOT_RUN"}
    usage = usage_repo.list_records()
    assert [r.request_status for r in usage] == ["FAILED", "SUCCESS"]
    assert usage[-1].fallback_used and usage[-1].input_tokens == 11 and usage[-1].output_tokens == 7
    attempts = [attempt for s in caught.value.trace.steps for attempt in s.provider_attempts]
    assert [(a.provider_name, a.outcome.value) for a in attempts] == [
        ("OpenAI", "UNAVAILABLE"), ("First configured provider", "RETRYABLE_ERROR"), ("OpenRouter", "SUCCESS")]


@pytest.mark.parametrize("corrected", [True, False])
def test_existing_opt_in_correction_is_bounded_preserves_actions_and_requires_review(corrected):
    missing = registration_plan()
    candidate = registration_plan(assertion()) if corrected else missing
    provider = RegistrationProvider([missing, candidate, registration_plan(assertion())])
    repository = InMemoryReliabilityRepository()
    repository.save_settings(ReliabilitySettings(automatic_plan_repair=True, max_total_attempts=3))
    supervisor = AutomationReliabilitySupervisor(repository)
    generator = LLMTestPlanGenerator(LLMRouter([provider]), supervisor)
    step = registration_step()
    original = step.model_dump_json()
    if corrected:
        result = generator.generate_with_plan(step, registration_discovery())
        version = result.test_plan_version
        assert version.origin.value == "REPAIRED" and supervisor.requires_review(version.id)
        assert version.qa_test_plan.steps[:-1] == missing.steps
    else:
        with pytest.raises(PlanValidationError), patch("qa_agent.test_plan_generator.TestPlan") as factory:
            generator.generate_with_plan(step, registration_discovery())
        factory.assert_not_called()
    assert len(provider.calls) == 2 and step.model_dump_json() == original
    assert "EXPECTED_RESULT_NOT_COVERED" in provider.calls[-1][0]
    assert "do not invent a selector" in provider.calls[-1][0]
    record = repository.list_records()[0]
    assert record.attempts[0].quality_gates["expected_result_coverage"] == "FAILED"
    assert record.outcome == ("READY_FOR_REVIEW" if corrected else "NEEDS_ATTENTION")


def test_hidden_state_identity_normalization_is_bounded_and_contains_no_hidden_values():
    data = {"url": URL, "state_elements": [
        {"tag": "p", "role": "alert", "selector": "#form-error", "visible": False,
         "text": "hidden private content", "value": "private input"},
        {"tag": "input", "role": "input", "selector": "#private"},
        {"tag": "p", "role": ["alert"], "selector": "#malformed"},
        {"tag": "p", "role": "alert", "selector": "x" * 181},
    ]}
    snapshot = json.loads(_build_snapshot(data))
    assert snapshot["state_elements"] == [{"tag": "p", "role": "alert", "selector": "#form-error", "visible": False}]
    data["state_elements"] = [{"tag": "p", "role": "alert", "selector": f"#error-{i}"} for i in range(100)]
    assert len(json.loads(_build_snapshot(data))["state_elements"]) == 16
    data["visible_text_elements"] = [{"tag": "p", "selector": "#" + "x" * 170, "text": "a" * 120, "visible": True}] * 16
    assert len(_build_snapshot(data)) <= MAX_SNAPSHOT_CHARS
