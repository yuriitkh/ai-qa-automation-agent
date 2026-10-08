from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from qa_agent.evidence_policy import EvidenceScope
from qa_agent.run_context import RunContext


class QATestStep(BaseModel):
    ACTION_PARAMETER_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "navigate": ("url",),
        "assert_page_loaded": (),
        "assert_title": ("expected",),
        "assert_visible": ("selector",),
        "click": ("selector",),
        "check": ("selector",),
        "uncheck": ("selector",),
        "fill": ("selector", "value"),
        "assert_hidden": ("selector",),
        "assert_url": ("expected",),
        "select_option": ("selector", "option_label"),
        "assert_text_contains": ("expected_text",),
        "assert_checked": ("selector",),
        "assert_unchecked": ("selector",),
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
        "check",
        "uncheck",
        "fill",
        "assert_hidden",
        "assert_url",
        "select_option",
        "assert_text_contains",
        "assert_checked",
        "assert_unchecked",
        "assert_selected",
        "assert_enabled",
        "assert_disabled",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capability_parameters(self) -> "QATestStep":
        selector = self.parameters.get("selector")
        if self.action in {
            "select_option", "assert_selected", "assert_checked", "assert_unchecked",
            "check", "uncheck", "assert_enabled", "assert_disabled",
        }:
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
    label: str = ""
    placeholder: str = ""
    test_id: str = ""
    tag: str = ""
    role: str = ""
    id: str = ""
    name: str = ""
    href: str = ""
    dialog_identity: str = ""
    visible: bool = True
    enabled: bool = True


class LocatorIdentityEntry(BaseModel):
    """Value-free identity evidence captured for one planned interaction."""

    model_config = ConfigDict(frozen=True)

    step_index: int = Field(ge=0)
    accessible_name: str | None = Field(default=None, max_length=120)
    label: str | None = Field(default=None, max_length=120)
    placeholder: str | None = Field(default=None, max_length=120)
    visible_text: str | None = Field(default=None, max_length=120)
    test_id: str | None = Field(default=None, max_length=180)
    element_id: str | None = Field(default=None, max_length=180)
    name: str | None = Field(default=None, max_length=180)
    href: str | None = Field(default=None, max_length=500)
    dialog_identity: str | None = Field(default=None, max_length=120)
    tag: str | None = Field(default=None, max_length=40)
    role: str | None = Field(default=None, max_length=60)


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


class ExecutionSegment(BaseModel):
    """Domain grouping for ordered TestSteps within a TestCase.

    Segment boundaries are descriptive only at this stage; they do not define
    browser, discovery, or execution lifecycles.
    """

    id: UUID = Field(default_factory=uuid4)
    order: int = Field(ge=0)
    base_url: str | None = None
    is_implicit: bool = False
    steps: list[TestStep] = Field(min_length=1)


class Precondition(BaseModel):
    """A declarative condition that must hold before a TestCase is run.

    ``provided_data_keys`` names RunContext values that must be available once
    this condition has been established. It describes a contract, not setup.
    """

    id: UUID = Field(default_factory=uuid4)
    description: str = Field(min_length=1)
    order: int = Field(ge=0)
    provided_data_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provided_data_keys(self) -> "Precondition":
        if any(not key or key != key.strip() for key in self.provided_data_keys):
            raise ValueError("Precondition data keys must be non-empty trimmed strings.")
        if len(set(self.provided_data_keys)) != len(self.provided_data_keys):
            raise ValueError("Precondition data keys must be unique.")
        return self


class TestCase(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    public_id: str | None = Field(default=None, pattern=r"^TC-\d{4,}$")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    base_url: str | None = None
    preconditions: list[Precondition] = Field(default_factory=list)
    segments: list[ExecutionSegment] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_steps(cls, value: Any) -> Any:
        """Accept legacy flat ``steps=`` input as one implicit segment."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        flat_steps = data.pop("steps", None)
        segments = data.get("segments")
        if segments is None:
            if flat_steps is None:
                raise ValueError("TestCase requires steps or segments.")
            data["segments"] = [{
                "order": 0,
                "base_url": data.get("base_url"),
                "is_implicit": True,
                "steps": flat_steps,
            }]
        elif flat_steps is not None:
            # model_dump() includes the compatibility property. Accept it only
            # when it exactly matches the canonical segmented representation.
            flattened = [
                step
                for segment in segments
                for step in (segment.get("steps", []) if isinstance(segment, dict) else segment.steps)
            ]
            flat_models = [step if isinstance(step, TestStep) else TestStep.model_validate(step) for step in flat_steps]
            segment_models = [step if isinstance(step, TestStep) else TestStep.model_validate(step) for step in flattened]
            if flat_models != segment_models:
                raise ValueError("TestCase steps must match the flattened segment steps.")
            segment_orders = [segment.order if isinstance(segment, ExecutionSegment) else segment.get("order") for segment in segments]
            step_orders = [step.order if isinstance(step, TestStep) else step.get("order") for step in flattened]
            step_ids = [step.id if isinstance(step, TestStep) else step.get("id") for step in flattened]
            is_implicit = [segment.is_implicit if isinstance(segment, ExecutionSegment) else segment.get("is_implicit", False) for segment in segments]
            if segment_orders != sorted(segment_orders) or len(set(segment_orders)) != len(segment_orders):
                raise ValueError("ExecutionSegments must have unique, ordered order values.")
            if any(is_implicit) and (len(segments) != 1 or is_implicit != [True]):
                raise ValueError("Only one implicit ExecutionSegment is allowed.")
            if not any(is_implicit) and step_orders != sorted(step_orders):
                raise ValueError("ExecutionSegment steps must be ordered by order.")
            if len(set(step_ids)) != len(step_ids):
                raise ValueError("A TestStep cannot belong to multiple segments.")
        else:
            # Explicit segment input must already have deterministic ordering
            # and must not repeat a TestStep across groups.
            segment_orders = [segment.order if isinstance(segment, ExecutionSegment) else segment.get("order") for segment in segments]
            if segment_orders != sorted(segment_orders) or len(set(segment_orders)) != len(segment_orders):
                raise ValueError("ExecutionSegments must have unique, ordered order values.")
            explicit_steps = [
                step
                for segment in segments
                for step in (segment.steps if isinstance(segment, ExecutionSegment) else segment.get("steps", []))
            ]
            step_orders = [step.order if isinstance(step, TestStep) else step.get("order") for step in explicit_steps]
            step_ids = [step.id if isinstance(step, TestStep) else step.get("id") for step in explicit_steps]
            if step_orders != sorted(step_orders):
                raise ValueError("ExecutionSegment steps must be ordered by order.")
            if len(set(step_ids)) != len(step_ids):
                raise ValueError("A TestStep cannot belong to multiple segments.")
        return data

    @model_validator(mode="after")
    def validate_segments(self) -> "TestCase":
        segments = self.segments
        segment_orders = [segment.order for segment in segments]
        if segment_orders != sorted(segment_orders) or len(set(segment_orders)) != len(segment_orders):
            raise ValueError("ExecutionSegments must be ordered by order.")
        seen_step_ids: set[UUID] = set()
        for segment in segments:
            for step in segment.steps:
                if step.id in seen_step_ids:
                    raise ValueError("A TestStep cannot belong to multiple segments.")
                seen_step_ids.add(step.id)
        precondition_orders = [item.order for item in self.preconditions]
        if precondition_orders != sorted(precondition_orders) or len(set(precondition_orders)) != len(precondition_orders):
            raise ValueError("TestCase Preconditions must have unique, ordered order values.")
        return self

    @computed_field
    @property
    def steps(self) -> list[TestStep]:
        """Backward-compatible flat view, ordered by segment then step."""
        return [
            step
            for segment in self.segments
            for step in segment.steps
        ]


class TestPlan(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    test_step_id: UUID
    name: str = Field(min_length=1)


class PlanVersionOrigin(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN_EDITED = "HUMAN_EDITED"
    REGENERATED = "REGENERATED"
    REPAIRED = "REPAIRED"


class AssertionGrounding(str, Enum):
    """Deterministic provenance for an assertion in one immutable plan version."""

    REQUIREMENT_GROUNDED = "REQUIREMENT_GROUNDED"
    OBSERVATION_GROUNDED = "OBSERVATION_GROUNDED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class AssertionGroundingEntry(BaseModel):
    """Value-free grounding metadata keyed to a plan action index."""

    model_config = ConfigDict(frozen=True)

    step_index: int = Field(ge=0)
    category: AssertionGrounding


class TestPlanVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    test_plan_id: UUID
    version: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    origin: PlanVersionOrigin | None = None
    qa_test_plan: QATestPlan
    # Optional for compatibility with saved plans created before grounding.
    # Metadata records only categories and indexes; it never stores page text.
    assertion_grounding: tuple[AssertionGroundingEntry, ...] | None = None
    # Locator identity contains control labels/attributes, never entered values.
    # It is separate from QATestPlan so the canonical executable/export format
    # remains unchanged. Missing metadata is valid for legacy and edited plans.
    locator_identity: tuple[LocatorIdentityEntry, ...] | None = None

    @model_validator(mode="after")
    def validate_assertion_grounding_indexes(self) -> "TestPlanVersion":
        if self.assertion_grounding is None:
            return self
        indexes = [entry.step_index for entry in self.assertion_grounding]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Assertion grounding action indexes must be unique.")
        for index in indexes:
            if index >= len(self.qa_test_plan.steps):
                raise ValueError("Assertion grounding index is outside the saved plan.")
            if not self.qa_test_plan.steps[index].action.startswith("assert_"):
                raise ValueError("Assertion grounding metadata must refer to assertion actions.")
        return self

    @model_validator(mode="after")
    def validate_locator_identity_indexes(self) -> "TestPlanVersion":
        if self.locator_identity is None:
            return self
        indexes = [entry.step_index for entry in self.locator_identity]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Locator identity indexes must be unique.")
        for index in indexes:
            if index >= len(self.qa_test_plan.steps):
                raise ValueError("Locator identity index is outside the saved plan.")
            if self.qa_test_plan.steps[index].action not in {"click", "check", "uncheck", "fill"}:
                raise ValueError("Locator identity metadata must refer to supported interactions.")
        return self


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
    scope: EvidenceScope | None = None
    event: str | None = None


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
    run_context: RunContext = Field(default_factory=RunContext)
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
        run_context: RunContext | None = None,
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
            run_context=run_context if run_context is not None else RunContext(),
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
