"""Deterministic generation gates for related atomic TestSteps."""
import json
import pytest

from qa_agent.llm.router import LLMRouter
from qa_agent.models import DiscoveryResult, DiscoveryStatus, InteractiveElement, QATestPlan, TestCase as Case, TestStep as Step
from qa_agent.pipeline import PipelineStageError, QATestPipeline
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.generation_context import StepGenerationContext
from tests.test_registration_coverage import RegistrationProvider, assertion

URL = "http://127.0.0.1:8000/demo-target/registration"


def scenario():
    steps = [
        Step(name="Enter registration data", description="Fill an invalid email and valid password.", expected="The input is entered.", order=0),
        Step(name="Submit registration form", description="Click Create account.", expected="The button is clicked.", order=1),
        Step(name="Verify invalid email rejection", description="Verify the validation error.", expected="An error message is displayed.", order=2),
        Step(name="Verify success is absent", description="Verify registration has no success confirmation.", expected="The success confirmation is hidden.", order=3),
    ]
    case = Case(name="Invalid email validation", description="Enter invalid data, submit, verify rejection and absence of success.", base_url=URL, steps=steps)
    discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=URL, interactive_elements=[
        InteractiveElement(kind="input", tag="input", selector="#email", accessible_name="Email"),
        InteractiveElement(kind="input", tag="input", selector="#password", accessible_name="Password"),
        InteractiveElement(kind="button", tag="button", selector="#create-account", accessible_name="Create account"),
    ], snapshot={"state_elements": [
        {"selector": "#form-error", "tag": "p", "role": "alert", "visible": False},
        {"selector": "#registration-success", "tag": "p", "role": "status", "visible": False},
    ]})
    actions = [
        [{"action": "fill", "parameters": {"selector": "#email", "value": "invalid"}},
         {"action": "fill", "parameters": {"selector": "#password", "value": "fixture-password"}}],
        [{"action": "click", "parameters": {"selector": "#create-account"}}],
        [assertion("assert_visible", "#form-error")],
        [assertion("assert_hidden", "#registration-success")],
    ]
    return case, discovery, [QATestPlan(url=URL, steps=items) for items in actions]


def test_full_generation_rejects_premature_submission_and_future_assertions():
    case, discovery, plans = scenario()
    plans[0] = QATestPlan(url=URL, steps=[action for plan in plans for action in plan.steps])
    provider = RegistrationProvider(plans)
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    executed = []
    pipeline = QATestPipeline(None, generator, discovery=lambda _: discovery,
        runner=lambda plan: executed.append(plan) or {"status": "passed", "steps": []})
    with pytest.raises(PipelineStageError) as caught:
        pipeline.run_test_case(case)
    assert isinstance(caught.value.__cause__, PlanValidationError)
    assert caught.value.__cause__.issues[0].code == "TESTSTEP_BOUNDARY_VIOLATION"
    assert len(provider.calls) == 1 and not executed
    assert pipeline._plan_store.find(case.steps[0].id) is None
    record = generator.supervisor.repository.list_records()[0]
    assert record.outcome == "NEEDS_ATTENTION"
    assert record.attempts[0].quality_gates["step_boundaries"] == "FAILED"


@pytest.mark.parametrize("extra", ["submit", "error", "success"])
def test_future_actions_are_rejected_individually(extra):
    case, discovery, plans = scenario()
    check = plans[{"submit": 1, "error": 2, "success": 3}[extra]].steps
    plan = QATestPlan(url=URL, steps=plans[0].steps + check)
    context = StepGenerationContext(0, remaining_steps=tuple(case.steps[1:]))
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator(LLMRouter([RegistrationProvider([plan])])).generate_with_plan(case.steps[0], discovery, step_context=context)
    assert caught.value.issues[0].code == "TESTSTEP_BOUNDARY_VIOLATION"


@pytest.mark.parametrize("extra", ["fill", "navigate"])
def test_submit_step_rejects_repeated_completed_setup(extra):
    case, discovery, plans = scenario()
    discovery.strategies_used = ["current_page"]
    context = StepGenerationContext(0, previous_steps=(case.steps[0],), remaining_steps=tuple(case.steps[2:]),
        completed_actions=(("fill", "#email"), ("fill", "#password")), state_preserved=True)
    action = plans[0].steps[0] if extra == "fill" else {"action": "navigate", "parameters": {"url": URL}}
    candidate = QATestPlan(url=URL, steps=[action, *plans[1].steps])
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(candidate, discovery, case.steps[1], step_context=context)
    assert caught.value.issues[0].code == "TESTSTEP_BOUNDARY_VIOLATION"


def test_generation_context_preserves_minimal_steps_without_exposing_previous_input_values():
    case, discovery, plans = scenario()
    provider = RegistrationProvider(plans)
    result = QATestPipeline(None, LLMTestPlanGenerator(LLMRouter([provider])), discovery=lambda _: discovery,
        runner=lambda plan: {"status": "passed", "steps": []}).run_test_case(case)
    assert result.test_run.status.value == "PASSED"
    assert len(provider.calls) == 4
    submit_task = provider.calls[1][0]
    context = json.loads(submit_task.split("ExecutionSegment context (only current TestStep is in scope): ")[1].splitlines()[0])
    assert context["previous_steps"][0]["order"] == 0
    assert [item["order"] for item in context["remaining_steps"]] == [2, 3]
    assert context["completed_action_targets"] == [["fill", "#email"], ["fill", "#password"]]
    assert "fixture-password" not in submit_task
    assert "assert_selected additionally requires expected" in submit_task


def test_explicit_navigation_refilling_and_combined_current_actions_remain_supported():
    case, discovery, plans = scenario()
    context = StepGenerationContext(0, previous_steps=(case.steps[0],), completed_actions=(("fill", "#email"),), state_preserved=True)
    discovery.strategies_used = ["current_page"]
    step = case.steps[1].model_copy(update={"description": "Reload the page, fill the email again and click Create account."})
    plan = QATestPlan(url=URL, steps=[{"action": "navigate", "parameters": {"url": URL}}, *plans[0].steps, *plans[1].steps])
    candidate, _ = LLMTestPlanGenerator._validate_generated_plan(plan, discovery, step, step_context=context)
    assert candidate == plan


def test_setup_refilling_is_allowed_when_current_discovery_observes_the_control_is_empty():
    case, discovery, plans = scenario()
    discovery.interactive_elements[0].has_value = False
    context = StepGenerationContext(0, completed_actions=(("fill", "#email"),), state_preserved=True)
    plan = QATestPlan(url=URL, steps=[plans[0].steps[0], *plans[1].steps])
    candidate, _ = LLMTestPlanGenerator._validate_generated_plan(plan, discovery, case.steps[1], step_context=context)
    assert candidate == plan


@pytest.mark.parametrize("action,selector,code", [
    ("assert_checked", "#email", "ACTION_TARGET_MISMATCH"),
    ("assert_checked", "#terms-paragraph", "ACTION_TARGET_MISMATCH"),
    ("assert_selected", "#region", "MISSING_EXPECTED_VALUE"),
    ("assert_visible", "#invented", "DISCOVERY_SELECTOR_MISMATCH"),
])
def test_canonical_gate_rejections_remain_strict(action, selector, code):
    case, discovery, _ = scenario()
    discovery.interactive_elements.append(InteractiveElement(kind="select", tag="select", selector="#region"))
    discovery.snapshot["visible_text_elements"] = [{"selector": "#terms-paragraph", "tag": "p", "text": "Terms checkbox"}]
    plan = QATestPlan(url=URL, steps=[assertion(action, selector)])
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator._validate_generated_plan(plan, discovery, case.steps[0])
    assert caught.value.issues[0].code == code


@pytest.mark.parametrize("expected", ["North", "Invented region"])
def test_select_assertion_requires_the_requested_option_not_merely_an_available_label(expected):
    _, discovery, _ = scenario()
    discovery.status = DiscoveryStatus.PARTIAL
    control = InteractiveElement(kind="select", tag="select", selector="#region", accessible_name="Region", option_labels=("North", "South"))
    discovery.interactive_elements = [control]
    discovery.snapshot = {"interactive_elements": [control.model_dump(mode="json")]}
    step = Step(name="Verify region selection", description="Verify North is selected for the region.", expected="The region is selected.", order=0)
    plan = QATestPlan(url=URL, steps=[{"action": "assert_selected", "parameters": {"selector": "#region", "expected": expected}}])
    if expected == "North":
        candidate, _ = LLMTestPlanGenerator._validate_generated_plan(plan, discovery, step)
        assert candidate == plan
    else:
        with pytest.raises(PlanValidationError) as caught:
            LLMTestPlanGenerator._validate_generated_plan(plan, discovery, step)
        assert caught.value.issues[0].code == "UNGROUNDED_ASSERTION"


def test_ai_fallback_control_cannot_establish_a_selector_or_checkbox_type():
    _, discovery, _ = scenario()
    discovery.status = DiscoveryStatus.PARTIAL
    discovery.interactive_elements = [InteractiveElement(kind="checkbox", selector="#invented-checkbox", accessible_name="Terms")]
    step = Step(name="Click Terms", description="Click the Terms checkbox.", expected="The button is clicked.", order=0)
    plan = QATestPlan(url=URL, steps=[{"action": "click", "parameters": {"selector": "#invented-checkbox"}}])
    provider = RegistrationProvider([plan])
    with pytest.raises(PlanValidationError) as caught:
        LLMTestPlanGenerator(LLMRouter([provider])).generate(step, discovery)
    assert caught.value.issues[0].code in {"DISCOVERY_SELECTOR_MISMATCH", "ACTION_TARGET_MISMATCH"}
