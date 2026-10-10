import json
import re
from dataclasses import dataclass

from qa_agent.execution_control import check_cancelled
from qa_agent.llm.router import LLMRouter
from qa_agent.models import (
    AssertionGroundingEntry,
    DiscoveryResult,
    DiscoveryStatus,
    PlanVersionOrigin,
    QATestPlan,
    QATestStep,
    TestPlan,
    TestPlanVersion,
    TestStep,
)
from qa_agent.test_plan_validation import (
    PlanValidationError,
    PlanValidationIssue,
    validate_executable_plan,
)
from qa_agent.llm_usage import (
    OP_GENERATE_AUTOMATION_PLAN,
    OP_REPAIR_AUTOMATION_PLAN,
    llm_usage_scope,
)
from qa_agent.assertion_grounding import validate_assertion_grounding
from qa_agent.expected_result_coverage import validate_expected_result_coverage
from qa_agent.generation_context import StepGenerationContext, observed_controls, validate_step_boundaries
from qa_agent.reliability import AutomationReliabilitySupervisor, current_reliability_operation


@dataclass(frozen=True)
class GeneratedTestPlan:
    """The created TestPlan and its version, linked without copying either."""

    test_plan: TestPlan
    test_plan_version: TestPlanVersion

    def __post_init__(self) -> None:
        if self.test_plan_version.test_plan_id != self.test_plan.id:
            raise ValueError("TestPlanVersion must reference the supplied TestPlan.")


class TestPlanGenerator:
    """Boundary for creating executable plan versions from atomic test steps.

    The default implementation is intentionally a placeholder until a concrete
    generation strategy is selected and connected.
    """

    def generate(
        self,
        test_step: TestStep,
        discovery_result: DiscoveryResult,
    ) -> TestPlanVersion:
        """Generate a versioned executable plan for ``test_step``.

        Args:
            test_step: The atomic behavior or check to implement.
            discovery_result: Typed browser discovery data available to a future
                deterministic or LLM-backed generation strategy.

        """
        return self.generate_with_plan(test_step, discovery_result).test_plan_version

    def generate_with_plan(
        self,
        test_step: TestStep,
        discovery_result: DiscoveryResult,
        *,
        existing_test_plan: TestPlan | None = None,
        version_number: int = 1,
        requirement_context: str | None = None,
        step_context: StepGenerationContext | None = None,
    ) -> GeneratedTestPlan:
        """Generate a version while retaining its owning TestPlan object."""
        raise NotImplementedError(
            "Test plan generation is not implemented; no executable plan was created."
        )


class LLMTestPlanGenerator(TestPlanGenerator):
    """Generate an executable plan version through the configured LLM Router."""

    def __init__(self, router: LLMRouter, supervisor: AutomationReliabilitySupervisor | None = None) -> None:
        self._router = router
        self.supervisor = supervisor or AutomationReliabilitySupervisor()

    def generate_with_plan(
        self,
        test_step: TestStep,
        discovery_result: DiscoveryResult,
        *,
        existing_test_plan: TestPlan | None = None,
        version_number: int = 1,
        requirement_context: str | None = None,
        step_context: StepGenerationContext | None = None,
    ) -> GeneratedTestPlan:
        task = self._build_task_context(
            test_step, discovery_result, requirement_context=requirement_context, step_context=step_context
        )
        page_snapshot = self._build_discovery_context(discovery_result)
        previous_candidate = None
        repairing = False
        def request(action, error):
            nonlocal previous_candidate, repairing
            operation = current_reliability_operation()
            if operation and operation.supervisor.recovery_store is not None and not operation.record.attempts:
                try:
                    operation.supervisor.recovery_store.prepare(operation.record, test_step, discovery_result, requirement_context, step_context)
                except Exception:
                    pass
            repairing = error is not None
            request_task = self._build_repair_task(task, error.issues) if isinstance(error, PlanValidationError) else (
                task + "\nReturn a complete valid JSON plan matching the supplied schema. Preserve all original actions, requirements, assertions and locator identities. Do not invent values or selectors."
                if error is not None else task
            )
            with llm_usage_scope(operation_type=OP_REPAIR_AUTOMATION_PLAN if error else OP_GENERATE_AUTOMATION_PLAN):
                call = lambda: self._router.create_test_plan(task=request_task, target_url=discovery_result.url, page_snapshot=page_snapshot)
                if isinstance(self._router, LLMRouter) and type(self._router).create_test_plan is LLMRouter.create_test_plan:
                    candidate = call()
                else:
                    operation = current_reliability_operation()
                    candidate = operation.invoke(None, call, action=operation.request_action, reason=operation.request_reason)
                return candidate

        def validate(value):
            nonlocal previous_candidate
            original = previous_candidate
            previous_candidate = value
            try:
                validated = self._validate_generated_plan(value, discovery_result, test_step, requirement_context=requirement_context, step_context=step_context)
                if repairing and original is not None:
                    self._validate_repair_preserves_actions(original, value)
            except PlanValidationError as error:
                from qa_agent.diagnostic_mode import DiagnosticLevel, enabled
                operation = current_reliability_operation()
                if operation:
                    operation.capture_recovery(value)
                if operation and enabled(operation.record.settings.diagnostic_level, DiagnosticLevel.DEBUG):
                    try:
                        from qa_agent.candidate_diagnostics import rejected_candidate_diagnostics
                        error.candidate_diagnostics = rejected_candidate_diagnostics(value, test_step, discovery_result, error)
                    except Exception:
                        pass
                raise
            return validated

        (executable_plan, grounding), operation_id, repaired = self.supervisor.generate(
            test_step, request,
            validate,
        )
        if existing_test_plan is not None:
            if existing_test_plan.test_step_id != test_step.id:
                raise ValueError("Existing TestPlan belongs to a different TestStep.")
            test_plan = existing_test_plan
        else:
            test_plan = TestPlan(
                test_step_id=test_step.id,
                name=test_step.name,
            )
        version = TestPlanVersion(
            test_plan_id=test_plan.id,
            version=version_number,
            origin=(PlanVersionOrigin.REPAIRED if repaired else (
                PlanVersionOrigin.REGENERATED
                if existing_test_plan is not None
                else PlanVersionOrigin.AI_GENERATED
            )),
            qa_test_plan=executable_plan,
            assertion_grounding=grounding,
        )
        self.supervisor.attach_candidate(operation_id, version.id)
        return GeneratedTestPlan(test_plan=test_plan, test_plan_version=version)

    @staticmethod
    def _validate_repair_preserves_actions(previous, candidate):
        """A structural repair may add missing coverage, but cannot remove valid actions."""
        original = previous.model_dump() if isinstance(previous, QATestPlan) else previous
        if not isinstance(original, dict) or not isinstance(original.get("steps"), list):
            return
        repaired = validate_executable_plan(candidate)
        remaining = iter(repaired.steps)
        for action in original["steps"]:
            try:
                valid = validate_executable_plan({"url": repaired.url, "steps": [action]}).steps[0]
            except PlanValidationError:
                continue
            if not any(item == valid for item in remaining):
                raise PlanValidationError([PlanValidationIssue(
                    code="REPAIR_CHANGED_SEMANTICS", path="steps",
                    message="A structural repair must preserve existing valid actions, values, assertions and their order.",
                )])

    @staticmethod
    def _validate_generated_plan(
        value: object,
        discovery_result: DiscoveryResult,
        test_step: TestStep,
        *,
        requirement_context: str | None = None,
        step_context: StepGenerationContext | None = None,
    ) -> tuple[QATestPlan, tuple[AssertionGroundingEntry, ...]]:
        import time
        from qa_agent.diagnostic_mode import DiagnosticLevel, record_diagnostic, discovery_metadata, enabled
        def gate(name, call):
            started = time.perf_counter()
            try:
                result = call()
            except Exception:
                record_diagnostic('QUALITY_GATE', minimum=DiagnosticLevel.TRACE,
                                  stage=name, status='FAILED', duration_ms=int((time.perf_counter() - started) * 1000))
                raise
            record_diagnostic('QUALITY_GATE', minimum=DiagnosticLevel.TRACE,
                              stage=name, status='PASSED', duration_ms=int((time.perf_counter() - started) * 1000))
            return result
        operation = current_reliability_operation()
        if operation and enabled(operation.record.settings.diagnostic_level, DiagnosticLevel.DEBUG):
            record_diagnostic('DISCOVERY', minimum=DiagnosticLevel.DEBUG, **discovery_metadata(discovery_result))
        check_cancelled()
        executable_plan = gate('schema_and_actions', lambda: validate_executable_plan(value))
        check_cancelled()
        gate('locator_identity', lambda: LLMTestPlanGenerator._validate_discovery_capabilities(
            executable_plan, discovery_result, test_step))
        check_cancelled()
        gate('locator_identity', lambda: LLMTestPlanGenerator._validate_locator_identity(executable_plan, discovery_result))
        check_cancelled()
        grounding = gate('assertion_grounding', lambda: validate_assertion_grounding(
            executable_plan,
            test_step,
            discovery_result,
            requirement_context=requirement_context,
        ))
        check_cancelled()
        coverage = gate('expected_result_coverage', lambda: validate_expected_result_coverage(test_step, executable_plan, discovery=discovery_result))
        record_diagnostic('COVERAGE', minimum=DiagnosticLevel.DEBUG, coverage=coverage.status.value)
        check_cancelled()
        gate('step_boundaries', lambda: validate_step_boundaries(executable_plan, test_step, discovery_result, step_context))
        return executable_plan, grounding

    @staticmethod
    def _validate_locator_identity(plan: QATestPlan, discovery: DiscoveryResult) -> None:
        selectors = set(observed_controls(discovery))

        def collect(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, str) and (key == "selector" or key.endswith("_selector")):
                        selectors.add(child)
                    else:
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(discovery.snapshot)
        if discovery.status == DiscoveryStatus.SUCCESS:
            for field in (discovery.navigation_paths, discovery.direct_navigation_paths, discovery.navigation):
                collect([item.model_dump(mode="python") for item in field])
        for index, action in enumerate(plan.steps):
            selector = action.parameters.get("selector")
            if selector is not None and selector not in selectors:
                raise PlanValidationError([PlanValidationIssue(
                    code="DISCOVERY_SELECTOR_MISMATCH", path=f"steps[{index}].parameters.selector",
                    message="The control selector is not established by deterministic Discovery. Review its identity before generation continues.",
                )])

    @staticmethod
    def _build_repair_task(
        original_task: str,
        issues: tuple[PlanValidationIssue, ...],
    ) -> str:
        issue_lines = "\n".join(
            f"- {issue.code} at {issue.path}: {issue.message}"
            for issue in issues
        )
        return (
            f"{original_task}\n\n"
            "The previous plan failed executable-plan validation. Regenerate a "
            "corrected plan for the same TestStep, preserving every requested "
            "action and verification. Every verification-oriented TestStep must "
            "include a semantically relevant assertion for its expected result; "
            "an unrelated assertion does not count. Do not add artificial "
            "assertions for action-only steps. Fix each listed issue. A human TestStep "
            "may require multiple ordered executable actions. Use only actions "
            "and selectors supported by the supplied schema and page snapshot. "
            "Do not add exact text, status, value, label, ID, URL, or count unless "
            "the original requirement explicitly requires it or deterministic page "
            "evidence shows it. Keep generic state checks structural; examples are "
            "illustrative and are not requirements.\n"
            f"Safe validation issues:\n{issue_lines}"
        )

    @staticmethod
    def _build_task_context(
        test_step: TestStep,
        discovery_result: DiscoveryResult,
        *,
        requirement_context: str | None = None,
        step_context: StepGenerationContext | None = None,
    ) -> str:
        context = {}
        if step_context is not None:
            describe = lambda item: {"order": item.order, "name": item.name, "description": item.description, "expected": item.expected}
            context = {
                "segment_order": step_context.segment_order,
                "previous_steps": [describe(item) for item in step_context.previous_steps[-12:]],
                "remaining_steps": [describe(item) for item in step_context.remaining_steps[:12]],
                "completed_action_targets": list(step_context.completed_actions[-64:]),
                "browser_state_preserved": step_context.state_preserved,
            }
        return (
            "Generate an executable Playwright-oriented QATestPlan for this "
            "one human TestStep. Do not generate a TestCase or add unrelated "
            "checks. A human TestStep may require multiple ordered executable "
            "actions, such as filling several fields. Submit only when this TestStep requests submission. "
            "The original TestCase is background context, never permission to implement future steps. "
            "Preserve the active segment's browser state. Do not repeat navigation or completed form filling "
            "unless the current step requests it or observed state establishes that setup is necessary. "
            "Navigation/setup is allowed when needed and not already satisfied. "
            "Preserve every requested action and verification from the TestStep. "
            "Every verification-oriented TestStep must include at least one "
            "assertion that checks the expected result itself; an unrelated page "
            "assertion does not count. Action-only steps do not need artificial "
            "assertions. Keep assertions relevant to the expected behavior and "
            "check them after the requested actions. For an expected result without "
            "errors, fill alone is insufficient: use assert_hidden on an observed "
            "error container when Discovery establishes its selector. Hidden alert/status "
            "identities appear in state_elements. If suitable evidence is missing, "
            "do not invent selectors or replace the expected result. "
            "For input acceptance without validation errors, perform the requested input actions "
            "and then check the observed error container of that same form; an unrelated hidden "
            "error or an absence check before input does not prove acceptance. This verifies the "
            "observable input/error state, not persistence or server-side acceptance. "
            "When Discovery links field-specific errors to those inputs, check each linked error's absence too. "
            "When the expected state is a displayed form, assert_visible must target the actual "
            "form identity from forms, not merely its heading, title, or assert_page_loaded. "
            "For blocked submission, verify both a visible observed validation error and a hidden "
            "observed success confirmation for the submitted form after clicking its submit control. "
            "For errors on all required fields, use each required control's observed error_selectors "
            "and assert_visible for every required field after submission. A global error alone is "
            "insufficient. forms.required_field_count detects incomplete Discovery; form_selector "
            "and aria-errormessage relationships bind the controls and errors. Browser-native "
            "validation bubbles are not DOM elements and have no supported selector assertion here. "
            "If the form has no observed required fields or no grounded per-field error identities, "
            "the required-field outcome is unsupported; stop for Needs Attention instead of inventing errors. "
            "Ground exact assertion values in the requirement or deterministic "
            "page evidence. A related original TestCase requirement is authoritative: "
            "an exact output literal found only in an elaborated TestStep is not independent evidence. "
            "If that literal is not required by the original scenario or observed at the assertion target, "
            "it needs human requirement clarification; do not invent a replacement message or "
            "drop an exact-message requirement in favor of a structural check. "
            "Quoted message content is literal text, not a URL, input instruction, or state instruction. "
            "Verify displayed messages with assert_visible and expected_text or assert_text_contains "
            "on the observed message container; never on Email or Password inputs. "
            "Treat deterministic interactive_elements "
            "as authoritative: when a requested control matches an accessible_name, "
            "use its exact selector unchanged. Use select_option for selects, "
            "check or uncheck to set checkbox state, assert_checked or "
            "assert_unchecked to verify checkbox state, click to select a radio, "
            "assert_selected for radio/select state, "
            "assert_enabled/assert_disabled for enabled state, and "
            "assert_text_contains for substring requirements. Use assert_value with selector and expected "
            "to verify an input/textarea's DOM value; text/visibility assertions do not verify input values. "
            "A quoted literal chosen from an explicit input instruction (including an example) may "
            "be filled and then verified on that same observed input. An example never establishes "
            "a product message, confirmation or other output. Do not invent exact "
            "text, statuses, values, IDs, labels, URLs, or counts that are not "
            "explicitly required or present in deterministic observed page evidence. "
            "Prefer a structural assertion when the requirement asks for a generic "
            "state or status. Examples introduced by 'e.g.', 'for example', or "
            "'such as' are illustrative, never mandatory unless separately required. "
            "Exact-value assertions require explicit requirement or observation "
            "grounding; when uncertain, do not assert the value.\n"
            "Canonical actions and required parameters (case-sensitive): "
            f"{json.dumps(QATestStep.ACTION_PARAMETER_FIELDS)}\n"
            "assert_selected additionally requires expected for a discovered select, but not for a radio. "
            "For a select, use an observed option_label to select and a requirement-stated expected label to verify. "
            "An available option alone does not establish that it should be selected. "
            "Checkbox assertions require a discovered checkbox; never infer one from appearance. "
            "Use interactive_elements input_type/button_type and option_labels to distinguish controls. "
            "For compound expected results, cover every clause with its relevant assertion. "
            "For a specific displayed page, verify a relevant Discovery-grounded form/control identity; "
            "navigation and assert_page_loaded alone do not establish that page's expected content. "
            "Use the smallest action sequence satisfying this step; avoid duplicate actions and "
            "unrelated or precautionary checkbox checks. Never assert_unchecked on an error container. "
            "For unsupported meaningful outcomes, stop for Needs Attention; action completion cannot prove persistence. "
            "Assertion subjects must match the expected state using the asserted value or that exact Discovery identity, "
            "never action-context overlap or unrelated parent/form text.\n"
            f"ExecutionSegment context (only current TestStep is in scope): {json.dumps(context, ensure_ascii=False)}\n"
            f"Original TestCase requirement: {requirement_context or '(not supplied)'}\n"
            f"TestStep order: {test_step.order}\n"
            f"TestStep name: {test_step.name}\n"
            f"TestStep description: {test_step.description}\n"
            f"Expected result: {test_step.expected}\n"
            f"Discovery status: {discovery_result.status.value}\n"
            f"Discovery warnings: {json.dumps(discovery_result.warnings, ensure_ascii=False)}"
        )

    @staticmethod
    def _build_discovery_context(discovery_result: DiscoveryResult) -> str:
        snapshot = dict(discovery_result.snapshot)
        # SUCCESS typed fields are deterministic. Fallback results may merge
        # AI suggestions into these fields; retain snapshot evidence instead.
        successful = discovery_result.status == DiscoveryStatus.SUCCESS
        snapshot.update(
            {
                "url": discovery_result.url,
                "title": discovery_result.title,
                "navigation_paths": [
                    path.model_dump(mode="json")
                    for path in discovery_result.navigation_paths
                ] if successful else snapshot.get("navigation_paths", []),
                "direct_navigation_paths": [
                    path.model_dump(mode="json")
                    for path in discovery_result.direct_navigation_paths
                ] if successful else snapshot.get("direct_navigation_paths", []),
                "interactive_elements": [element.model_dump(mode="json", exclude_defaults=True)
                    for element in list(observed_controls(discovery_result).values())[:32]],
                "navigation": [sequence.model_dump(mode="json") for sequence in discovery_result.navigation]
                    if successful else snapshot.get("navigation", []),
                "warnings": discovery_result.warnings,
                "strategies_used": discovery_result.strategies_used,
            }
        )
        return json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _validate_discovery_capabilities(
        plan: QATestPlan, discovery: DiscoveryResult, test_step: TestStep | str
    ) -> None:
        elements = observed_controls(discovery)
        intent = (
            test_step.casefold()
            if isinstance(test_step, str)
            else f"{test_step.name} {test_step.description} {test_step.expected}".casefold()
        )
        for index, step in enumerate(plan.steps):
            path = f"steps[{index}].parameters.selector"
            selector = step.parameters.get("selector")
            element = elements.get(selector) if isinstance(selector, str) else None
            relevant_actions = {
                "checkbox": {"click", "check", "uncheck", "assert_checked", "assert_unchecked"},
                "radio": {"click", "assert_selected"},
                "select": {"select_option", "assert_selected"},
                "input": {"assert_enabled", "assert_disabled"},
            }
            for discovered in elements.values():
                label = discovered.accessible_name.casefold()
                actions = next((allowed for kind, allowed in relevant_actions.items()
                                if kind in (discovered.kind + " " + discovered.tag + " " + discovered.role).casefold()), set())
                if (
                    label
                    and _intent_mentions_label(intent, label)
                    and step.action in actions
                    and selector != discovered.selector
                ):
                    raise PlanValidationError([PlanValidationIssue(
                        code="DISCOVERY_SELECTOR_MISMATCH",
                        path=path,
                        message="The action must use the deterministic Discovery selector for the requested control.",
                    )])
            if element is None and step.action in {"select_option", "assert_selected", "check", "uncheck", "assert_checked", "assert_unchecked", "assert_value"}:
                raise PlanValidationError([PlanValidationIssue(code="ACTION_TARGET_MISMATCH", path=path,
                    message="This action requires a control type established by deterministic Discovery.",
                    reason_code='UNSUPPORTED_INPUT_CONTROL' if step.action == 'assert_value' else
                                'ASSERTION_TARGET_MISMATCH' if step.action.startswith('assert_') else None)])
            if element is None:
                continue
            kinds = {element.kind.casefold(), element.tag.casefold(), element.role.casefold(), element.input_type.casefold()}
            if step.action == "assert_value" and (element.tag not in {"input", "textarea"}
                    or element.input_type in {"checkbox", "radio", "button", "submit", "hidden", "file"}):
                raise PlanValidationError([PlanValidationIssue(
                    code="ACTION_TARGET_MISMATCH", path=path,
                    message="ASSERT_VALUE must target a discovered text input or textarea.",
                    reason_code='UNSUPPORTED_INPUT_CONTROL',
                )])
            if step.action == "select_option" and "select" not in kinds:
                raise PlanValidationError([PlanValidationIssue(
                    code="ACTION_TARGET_MISMATCH",
                    path=path,
                    message="SELECT_OPTION must target a discovered select control.",
                )])
            if step.action == "assert_selected":
                if "radio" in kinds:
                    if step.parameters.get("expected") is not None:
                        raise PlanValidationError([PlanValidationIssue(
                            code="INVALID_PARAMETER",
                            path=f"steps[{index}].parameters.expected",
                            message="A radio selection assertion does not take an expected option value.",
                        )])
                elif "select" in kinds:
                    if not isinstance(step.parameters.get("expected"), str):
                        raise PlanValidationError([PlanValidationIssue(
                            code="MISSING_EXPECTED_VALUE",
                            path=f"steps[{index}].parameters.expected",
                            message="A select assertion requires an expected option value.",
                        )])
                else:
                    raise PlanValidationError([PlanValidationIssue(
                        code="ACTION_TARGET_MISMATCH",
                        path=path,
                        message="ASSERT_SELECTED does not target a radio or select control supported by Discovery.",
                        reason_code='ASSERTION_TARGET_MISMATCH',
                    )])
            if step.action in {"check", "uncheck", "assert_checked", "assert_unchecked"} and "checkbox" not in kinds:
                raise PlanValidationError([PlanValidationIssue(
                    code="ACTION_TARGET_MISMATCH",
                    path=path,
                    message=f"{step.action.upper()} must target a discovered checkbox control.",
                    reason_code='ASSERTION_TARGET_MISMATCH' if step.action.startswith('assert_') else None,
                )])


def _intent_mentions_label(intent: str, label: str) -> bool:
    """Match a control name as a phrase, not as a substring of another word."""
    words = [word for word in label.split() if word]
    if not words:
        return False
    phrase = r"\s+".join(re.escape(word) for word in words)
    return re.search(rf"(?<!\w){phrase}(?!\w)", intent, re.IGNORECASE) is not None
