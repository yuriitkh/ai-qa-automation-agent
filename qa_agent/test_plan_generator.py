import json
from dataclasses import dataclass

from qa_agent.llm.router import LLMRouter
from qa_agent.execution_progress import ExecutionEventType, emit_progress_event
from qa_agent.models import (
    DiscoveryResult,
    PlanVersionOrigin,
    QATestPlan,
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
    ) -> GeneratedTestPlan:
        """Generate a version while retaining its owning TestPlan object."""
        raise NotImplementedError(
            "Test plan generation is not implemented; no executable plan was created."
        )


class LLMTestPlanGenerator(TestPlanGenerator):
    """Generate an executable plan version through the configured LLM Router."""

    def __init__(self, router: LLMRouter) -> None:
        self._router = router

    def generate_with_plan(
        self,
        test_step: TestStep,
        discovery_result: DiscoveryResult,
        *,
        existing_test_plan: TestPlan | None = None,
        version_number: int = 1,
    ) -> GeneratedTestPlan:
        task = self._build_task_context(test_step, discovery_result)
        page_snapshot = self._build_discovery_context(discovery_result)
        with llm_usage_scope(operation_type=OP_GENERATE_AUTOMATION_PLAN):
            router_result = self._router.create_test_plan(
                task=task,
                target_url=discovery_result.url,
                page_snapshot=page_snapshot,
            )

        # Validate at this boundary as well, so a nonconforming Router
        # implementation cannot create a version from invalid plan data.
        repaired = False
        try:
            executable_plan = self._validate_generated_plan(
                router_result, discovery_result, test_step
            )
        except PlanValidationError as initial_error:
            repaired = True
            emit_progress_event(
                ExecutionEventType.PLAN_REPAIR_STARTED,
                step=test_step,
                message="Generated automation requires correction. Repairing once.",
            )
            repair_task = self._build_repair_task(task, initial_error.issues)
            try:
                with llm_usage_scope(operation_type=OP_REPAIR_AUTOMATION_PLAN):
                    repaired_result = self._router.create_test_plan(
                        task=repair_task,
                        target_url=discovery_result.url,
                        page_snapshot=page_snapshot,
                    )
            except Exception:
                emit_progress_event(
                    ExecutionEventType.PLAN_REPAIR_FAILED,
                    step=test_step,
                    message="Automation repair did not produce a validated plan.",
                )
                raise
            try:
                executable_plan = self._validate_generated_plan(
                    repaired_result, discovery_result, test_step
                )
            except PlanValidationError:
                emit_progress_event(
                    ExecutionEventType.PLAN_REPAIR_FAILED,
                    step=test_step,
                    message="Generated automation still failed validation.",
                )
                raise
            emit_progress_event(
                ExecutionEventType.PLAN_REPAIR_SUCCEEDED,
                step=test_step,
                message="Automation validated after repair.",
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
        )
        return GeneratedTestPlan(test_plan=test_plan, test_plan_version=version)

    @staticmethod
    def _validate_generated_plan(
        value: object,
        discovery_result: DiscoveryResult,
        test_step: TestStep,
    ) -> QATestPlan:
        executable_plan = validate_executable_plan(value)
        LLMTestPlanGenerator._validate_discovery_capabilities(
            executable_plan, discovery_result, test_step
        )
        return executable_plan

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
            "action and verification. Fix each listed issue. A human TestStep "
            "may require multiple ordered executable actions. Use only actions "
            "and selectors supported by the supplied schema and page snapshot.\n"
            f"Safe validation issues:\n{issue_lines}"
        )

    @staticmethod
    def _build_task_context(
        test_step: TestStep,
        discovery_result: DiscoveryResult,
    ) -> str:
        return (
            "Generate an executable Playwright-oriented QATestPlan for this "
            "one human TestStep. Do not generate a TestCase or add unrelated "
            "checks. A human TestStep may require multiple ordered executable "
            "actions, such as filling several fields before submitting a form. "
            "Preserve every requested action and verification from the TestStep. "
            "Treat deterministic interactive_elements "
            "as authoritative: when a requested control matches an accessible_name, "
            "use its exact selector unchanged. Use select_option for selects, "
            "assert_checked for checkbox/radio, assert_selected for select state, "
            "assert_enabled/assert_disabled for enabled state, and "
            "assert_text_contains for substring requirements.\n"
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
        # The typed fields are authoritative, while retaining all other data
        # from the existing structured snapshot unchanged.
        snapshot.update(
            {
                "url": discovery_result.url,
                "title": discovery_result.title,
                "navigation_paths": [
                    path.model_dump(mode="json")
                    for path in discovery_result.navigation_paths
                ],
                "direct_navigation_paths": [
                    path.model_dump(mode="json")
                    for path in discovery_result.direct_navigation_paths
                ],
                "interactive_elements": [element.model_dump(mode="json") for element in discovery_result.interactive_elements],
                "navigation": [sequence.model_dump(mode="json") for sequence in discovery_result.navigation],
                "warnings": discovery_result.warnings,
                "strategies_used": discovery_result.strategies_used,
            }
        )
        return json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _validate_discovery_capabilities(
        plan: QATestPlan, discovery: DiscoveryResult, test_step: TestStep | str
    ) -> None:
        elements = {element.selector: element for element in discovery.interactive_elements}
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
                "checkbox": {"click", "assert_checked"},
                "radio": {"click", "assert_selected"},
                "select": {"select_option", "assert_selected"},
                "input": {"assert_enabled", "assert_disabled"},
            }
            for discovered in discovery.interactive_elements:
                label = discovered.accessible_name.casefold()
                actions = next((allowed for kind, allowed in relevant_actions.items()
                                if kind in (discovered.kind + " " + discovered.tag + " " + discovered.role).casefold()), set())
                if label and label in intent and step.action in actions and selector != discovered.selector:
                    raise PlanValidationError([PlanValidationIssue(
                        code="DISCOVERY_SELECTOR_MISMATCH",
                        path=path,
                        message="The action must use the deterministic Discovery selector for the requested control.",
                    )])
            if element is None:
                continue
            kind = (element.kind + " " + element.tag + " " + element.role).casefold()
            if step.action == "select_option" and "select" not in kind:
                raise PlanValidationError([PlanValidationIssue(
                    code="ACTION_TARGET_MISMATCH",
                    path=path,
                    message="SELECT_OPTION must target a discovered select control.",
                )])
            if step.action == "assert_selected":
                if "radio" in kind:
                    if step.parameters.get("expected") is not None:
                        raise PlanValidationError([PlanValidationIssue(
                            code="INVALID_PARAMETER",
                            path=f"steps[{index}].parameters.expected",
                            message="A radio selection assertion does not take an expected option value.",
                        )])
                elif "select" in kind:
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
                    )])
            if step.action == "assert_checked" and "checkbox" not in kind:
                raise PlanValidationError([PlanValidationIssue(
                    code="ACTION_TARGET_MISMATCH",
                    path=path,
                    message="ASSERT_CHECKED must target a discovered checkbox control.",
                )])
