"""Local regressions for registration input checks and negative error coverage."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from qa_agent.assertion_grounding import classify_assertions
from qa_agent.browser_discovery import MAX_SNAPSHOT_CHARS, _build_snapshot
from qa_agent.execution_progress import ExecutionProgressReporter, ExecutionProgressStore, active_execution_progress
from qa_agent.expected_result_coverage import ExpectedResultCoverageStatus, expected_result_coverage, validate_expected_result_coverage
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.llm.usage_metadata import capture_openai_usage
from qa_agent.llm_usage import LLMUsageRepository, LLMUsageService
from qa_agent.models import DiscoveryResult, DiscoveryStatus, InteractiveElement, QATestPlan, TestCase as Case, TestStep as Step
from qa_agent.pipeline import PipelineStageError, QATestPipeline
from qa_agent.reliability import AutomationReliabilitySupervisor, InMemoryReliabilityRepository, ReliabilitySettings
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError


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
        "assertion_grounding": "PASSED", "expected_result_coverage": "FAILED"}
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
