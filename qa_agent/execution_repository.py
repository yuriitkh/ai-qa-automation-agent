from typing import Protocol
from uuid import UUID

from qa_agent.models import Execution, TestCase


class ExecutionRepository(Protocol):
    """Storage contract for completed execution attempts."""

    def save(self, execution: Execution) -> None: ...

    def get(self, execution_id: UUID) -> Execution | None: ...

    def list_for_test_step(self, test_step_id: UUID) -> list[Execution]: ...

    def list_for_test_case(self, test_case: TestCase) -> list[Execution]: ...


class InMemoryExecutionRepository:
    """Insertion-ordered process-local storage for execution records."""

    def __init__(self) -> None:
        self._executions: dict[UUID, Execution] = {}

    def save(self, execution: Execution) -> None:
        if execution.id in self._executions:
            raise ValueError(f"Execution {execution.id} already exists.")
        self._executions[execution.id] = execution

    def get(self, execution_id: UUID) -> Execution | None:
        return self._executions.get(execution_id)

    def list_for_test_step(self, test_step_id: UUID) -> list[Execution]:
        return [
            execution
            for execution in self._executions.values()
            if execution.test_step_id == test_step_id
        ]

    def list_for_test_case(self, test_case: TestCase) -> list[Execution]:
        test_step_ids = {step.id for step in test_case.steps}
        return [
            execution
            for execution in self._executions.values()
            if execution.test_step_id in test_step_ids
        ]
