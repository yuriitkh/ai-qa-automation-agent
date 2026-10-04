from typing import Protocol
from copy import deepcopy
from uuid import UUID

from qa_agent.models import TestPlan, TestPlanVersion


def _validate_plan_store_write(
    test_step_id: UUID,
    plan_version: TestPlanVersion,
    test_plan: TestPlan,
    *,
    current_plan_id: UUID | None,
    current_version_number: int | None,
    current_version_id: UUID | None,
) -> None:
    """Apply the shared relationship and monotonic-version invariants."""
    if test_plan.id != plan_version.test_plan_id:
        raise ValueError("TestPlanVersion does not reference the supplied TestPlan.")
    if test_plan.test_step_id != test_step_id:
        raise ValueError("TestPlan does not belong to the supplied TestStep.")
    if current_plan_id is not None and current_plan_id != test_plan.id:
        raise ValueError("A TestStep cannot be reassigned to a different TestPlan.")
    if (
        current_version_number is not None
        and plan_version.version < current_version_number
    ):
        raise ValueError("Cannot replace a cached TestPlanVersion with an older version.")
    if (
        current_version_number == plan_version.version
        and current_version_id is not None
        and current_version_id != plan_version.id
    ):
        raise ValueError("A TestPlan version number cannot identify two versions.")


class PlanStore(Protocol):
    """Minimal lookup/save contract for a TestStep's current plan version."""

    def save(
        self,
        test_step_id: UUID,
        plan_version: TestPlanVersion,
        *,
        test_plan: TestPlan,
    ) -> None: ...

    def find(self, test_step_id: UUID) -> TestPlanVersion | None: ...

    def get_version(self, version_id: UUID) -> TestPlanVersion | None: ...

    def find_test_plan(self, test_step_id: UUID) -> TestPlan | None: ...


class InMemoryPlanStore:
    """Process-local store keyed by stable ``TestStep.id`` values."""

    def __init__(self) -> None:
        self._versions: dict[UUID, TestPlanVersion] = {}
        self._version_history: dict[UUID, TestPlanVersion] = {}
        self._test_plans: dict[UUID, TestPlan] = {}

    def save(
        self,
        test_step_id: UUID,
        plan_version: TestPlanVersion,
        *,
        test_plan: TestPlan,
    ) -> None:
        current_plan = self._test_plans.get(test_step_id)
        current_version = self._versions.get(test_step_id)
        _validate_plan_store_write(
            test_step_id,
            plan_version,
            test_plan,
            current_plan_id=current_plan.id if current_plan is not None else None,
            current_version_number=current_version.version if current_version is not None else None,
            current_version_id=current_version.id if current_version is not None else None,
        )

        existing = self._version_history.get(plan_version.id)
        if existing is not None and existing != plan_version:
            raise ValueError("A TestPlanVersion ID cannot be reused for different content.")
        for existing_version in self._version_history.values():
            if (
                existing_version.test_plan_id == plan_version.test_plan_id
                and existing_version.version == plan_version.version
                and existing_version.id != plan_version.id
            ):
                raise ValueError("A TestPlan version number cannot identify two versions.")

        stored_version = deepcopy(plan_version)
        self._test_plans[test_step_id] = test_plan
        self._versions[test_step_id] = stored_version
        self._version_history[plan_version.id] = stored_version

    def find(self, test_step_id: UUID) -> TestPlanVersion | None:
        version = self._versions.get(test_step_id)
        return deepcopy(version) if version is not None else None

    def get_version(self, version_id: UUID) -> TestPlanVersion | None:
        version = self._version_history.get(version_id)
        return deepcopy(version) if version is not None else None

    def find_test_plan(self, test_step_id: UUID) -> TestPlan | None:
        return self._test_plans.get(test_step_id)
