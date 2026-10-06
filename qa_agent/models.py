from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QATestStep(BaseModel):
    ACTION_PARAMETER_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "navigate": ("url",),
        "assert_page_loaded": (),
        "assert_title": ("expected",),
        "assert_visible": ("selector", "expected_text"),
        "click": ("selector",),
        "fill": ("selector", "value"),
        "assert_hidden": ("selector",),
        "assert_url": ("expected",),
        "select_option": ("selector", "option_label"),
        "assert_text_contains": ("expected_text",),
        "assert_checked": ("selector",),
        "assert_selected": ("selector",),
        "assert_enabled": ("selector",),
        "assert_disabled": ("selector",),
    }

    action: Literal[
        "navigate",
        "assert_page_loaded",
        "assert_title",
        "assert_visible",
        "click",
        "fill",
        "assert_hidden",
        "assert_url",
        "select_option",
        "assert_text_contains",
        "assert_checked",
        "assert_selected",
        "assert_enabled",
        "assert_disabled",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capability_parameters(self) -> "QATestStep":
        selector = self.parameters.get("selector")
        if self.action in {"select_option", "assert_selected", "assert_checked", "assert_enabled", "assert_disabled"}:
            if not isinstance(selector, str) or not selector.strip():
                raise ValueError(f"{self.action} requires a selector.")
        if self.action == "select_option" and not isinstance(self.parameters.get("option_label"), str):
            raise ValueError("select_option requires option_label (the visible option label).")
        if self.action != "select_option" and "option_label" in self.parameters:
            raise ValueError("option_label is only valid for select_option.")
        if self.action == "assert_text_contains" and not isinstance(self.parameters.get("expected_text"), str):
            raise ValueError("assert_text_contains requires expected_text.")
        if self.action == "assert_selected":
            expected = self.parameters.get("expected")
            if expected is not None and not isinstance(expected, str):
                raise ValueError("assert_selected expected must be a string or null.")
        return self


class QATestPlan(BaseModel):
    url: str = Field(min_length=1)
    steps: list[QATestStep] = Field(min_length=1)


class DiscoveryStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class NavigationPath(BaseModel):
    """DNB-style navigation path with the existing snapshot field names."""

    menu_button_selector: str = ""
    menu_tab_text: str = ""
    menu_tab_selector: str = ""
    menu_item_text: str = ""
    menu_item_selector: str = ""
    submenu_text: str = ""
    submenu_selector: str = ""
    expected_url: str = ""
    heading_text: str = ""
    heading_selector: str = ""


class DirectNavigationStep(BaseModel):
    text: str
    selector: str
    href: str
    resolved_url: str


class DirectNavigationPath(BaseModel):
    """FINN-style direct navigation path, preserving the current JSON shape."""

    strategy: Literal["direct_nav"] = "direct_nav"
    root_url: str
    steps: list[DirectNavigationStep] = Field(min_length=2, max_length=2)
    expected_url: str
    heading_text: str
    heading_selector: str


class InteractiveElement(BaseModel):
    """Small browser-independent identity record for a discovered control."""

    kind: str
    selector: str
    text: str = ""
    accessible_name: str = ""
    tag: str = ""
    role: str = ""
    id: str = ""
    name: str = ""
    href: str = ""
    visible: bool = True
    enabled: bool = True


class NavigationAction(BaseModel):
    """One ordered interaction in a discovered navigation sequence."""

    element: InteractiveElement


class NavigationSequence(BaseModel):
    actions: list[NavigationAction] = Field(min_length=1)
    destination_url: str = ""


class DiscoveryResult(BaseModel):
    status: DiscoveryStatus
    url: str
    title: str = ""
    snapshot: dict[str, Any] = Field(default_factory=dict)
    navigation_paths: list[NavigationPath] = Field(default_factory=list)
    direct_navigation_paths: list[DirectNavigationPath] = Field(default_factory=list)
    interactive_elements: list[InteractiveElement] = Field(default_factory=list)
    navigation: list[NavigationSequence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    strategies_used: list[str] = Field(default_factory=list)


class AIDiscoveryResult(BaseModel):
    """Structured suggestions from AI; never contains executable browser code."""

    navigation_paths: list[NavigationPath] = Field(default_factory=list)
    direct_navigation_paths: list[DirectNavigationPath] = Field(default_factory=list)
    interactive_elements: list[InteractiveElement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FailurePolicy(str, Enum):
    """Explicit policy for what happens after a TestStep's final outcome is FAILED.

    CONTINUE keeps the historical behavior: later steps still execute.
    BLOCK_REST stops the TestCase: later steps become BLOCKED and never run.
    """

    CONTINUE = "CONTINUE"
    BLOCK_REST = "BLOCK_REST"


class TestStep(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    order: int = Field(ge=0)
    # What the pipeline does when this step's final outcome is FAILED.
    # The default preserves the historical continue-after-failure behavior
    # and is intentionally excluded from the decomposer's uuid5 step identity.
    failure_policy: FailurePolicy = FailurePolicy.CONTINUE


class TestCase(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    base_url: str | None = None
    steps: list[TestStep] = Field(min_length=1)


class TestPlan(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    test_step_id: UUID
    name: str = Field(min_length=1)


class TestPlanVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    test_plan_id: UUID
    version: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    qa_test_plan: QATestPlan


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    # Step-level outcome only: the step never executed because an earlier
    # step's BLOCK_REST failure policy prevented continuation. A blocked
    # step carries no Execution; FAILED always means "executed and failed".
    BLOCKED = "BLOCKED"


class EvidenceType(str, Enum):
    SCREENSHOT = "SCREENSHOT"
    PAGE_SOURCE = "PAGE_SOURCE"
    DOM_SNAPSHOT = "DOM_SNAPSHOT"


class Evidence(BaseModel):
    """Infrastructure-independent reference to evidence from one execution."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    execution_id: UUID
    type: EvidenceType
    path: str = Field(min_length=1)
    description: str | None = None
    timestamp: datetime | None = None


class Execution(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    test_step_id: UUID
    test_plan_version_id: UUID
    # Zero-based index into the immutable TestPlanVersion.qa_test_plan.steps.
    # None preserves compatibility with executions created before this link existed.
    planned_step_index: int | None = Field(default=None, ge=0)
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    actual_result: Any
    error: str | None = None
    runner_result: dict[str, Any] | None = None
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def validate_evidence_execution(self) -> "Execution":
        if any(item.execution_id != self.id for item in self.evidence):
            raise ValueError("Execution evidence must reference its owning Execution.")
        return self

    def planned_interaction(self, plan_version: TestPlanVersion) -> QATestStep | None:
        """Resolve this execution's interaction from its exact plan version."""
        if plan_version.id != self.test_plan_version_id:
            raise ValueError("TestPlanVersion does not belong to this Execution.")
        if self.planned_step_index is None:
            return None
        if self.planned_step_index >= len(plan_version.qa_test_plan.steps):
            raise ValueError("Execution planned_step_index is outside its TestPlanVersion.")
        return plan_version.qa_test_plan.steps[self.planned_step_index]


class TestRun(BaseModel):
    """Domain summary and ordered execution history for one TestCase run."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    test_case_id: UUID
    test_step_ids: list[UUID]
    test_step_names: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None
    executions: list[Execution] = Field(default_factory=list)
    # Steps that never executed because an earlier step's BLOCK_REST failure
    # policy prevented continuation. Blocked steps never carry an Execution.
    blocked_step_ids: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_execution_steps(self) -> "TestRun":
        if len(set(self.test_step_ids)) != len(self.test_step_ids):
            raise ValueError("TestRun test_step_ids must be unique.")
        if self.test_step_names and len(self.test_step_names) != len(self.test_step_ids):
            raise ValueError("TestRun test_step_names must align with test_step_ids.")
        known_step_ids = set(self.test_step_ids)
        if any(execution.test_step_id not in known_step_ids for execution in self.executions):
            raise ValueError("TestRun contains an execution for an unrelated TestStep.")
        if len(set(self.blocked_step_ids)) != len(self.blocked_step_ids):
            raise ValueError("TestRun blocked_step_ids must be unique.")
        if any(blocked not in known_step_ids for blocked in self.blocked_step_ids):
            raise ValueError("TestRun blocked_step_ids must name known TestSteps.")
        if any(execution.test_step_id in set(self.blocked_step_ids)
               for execution in self.executions):
            raise ValueError("A blocked TestStep must not have an Execution.")
        return self

    @classmethod
    def from_test_case(
        cls,
        test_case: TestCase,
        executions: list[Execution],
        blocked_step_ids: list[UUID] | None = None,
    ) -> "TestRun":
        ordered_steps = sorted(test_case.steps, key=lambda step: step.order)
        if executions:
            started_at = min(execution.started_at for execution in executions)
            finished_times = [
                execution.finished_at
                for execution in executions
                if execution.finished_at is not None
            ]
            finished_at = (
                max(finished_times)
                if len(finished_times) == len(executions)
                else None
            )
        else:
            started_at = datetime.now(timezone.utc)
            finished_at = None
        return cls(
            test_case_id=test_case.id,
            test_step_ids=[step.id for step in ordered_steps],
            test_step_names=[step.name for step in ordered_steps],
            started_at=started_at,
            finished_at=finished_at,
            executions=executions,
            blocked_step_ids=list(blocked_step_ids or []),
        )

    @property
    def final_executions(self) -> list[Execution]:
        latest_by_step: dict[UUID, Execution] = {}
        for execution in self.executions:
            latest_by_step[execution.test_step_id] = execution
        return [
            latest_by_step[step_id]
            for step_id in self.test_step_ids
            if step_id in latest_by_step
        ]

    def final_execution_for_step(self, test_step_id: UUID) -> Execution | None:
        if test_step_id not in self.test_step_ids:
            return None
        return next(
            (
                execution
                for execution in reversed(self.executions)
                if execution.test_step_id == test_step_id
            ),
            None,
        )

    @property
    def passed_steps(self) -> list[UUID]:
        return [
            execution.test_step_id
            for execution in self.final_executions
            if execution.status == ExecutionStatus.PASSED
        ]

    @property
    def blocked_steps(self) -> list[UUID]:
        """Blocked step ids in execution order; these steps never ran."""
        blocked = set(self.blocked_step_ids)
        return [step_id for step_id in self.test_step_ids if step_id in blocked]

    @property
    def failed_steps(self) -> list[UUID]:
        # Blocked steps are excluded: FAILED means "executed and failed",
        # BLOCKED means "never executed because of an earlier failure".
        blocked = set(self.blocked_step_ids)
        return [
            step_id
            for step_id in self.test_step_ids
            if step_id not in blocked
            and (
                (execution := self.final_execution_for_step(step_id)) is None
                or execution.status != ExecutionStatus.PASSED
            )
        ]

    @property
    def status(self) -> ExecutionStatus:
        # Blocked steps also fail the run: a TestCase that did not execute
        # to completion has not passed.
        if self.failed_steps or self.blocked_step_ids:
            return ExecutionStatus.FAILED
        return ExecutionStatus.PASSED

    @property
    def overall_status(self) -> ExecutionStatus:
        """Status of the complete TestCase, based on each step's latest attempt."""
        return self.status
