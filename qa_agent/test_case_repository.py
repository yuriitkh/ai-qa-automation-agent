"""Repositories for canonical, persisted TestCase definitions."""

from copy import deepcopy
from typing import Protocol
from uuid import UUID

from qa_agent.models import TestCase


class TestCaseRepository(Protocol):
    def save(self, test_case: TestCase) -> None: ...

    def get(self, test_case_id: UUID) -> TestCase | None: ...

    def list(self) -> list[TestCase]: ...


class InMemoryTestCaseRepository:
    """Process-local TestCase storage with copy-on-read/write semantics."""

    def __init__(self) -> None:
        self._test_cases: dict[UUID, TestCase] = {}

    def save(self, test_case: TestCase) -> None:
        self._test_cases[test_case.id] = deepcopy(test_case)

    def get(self, test_case_id: UUID) -> TestCase | None:
        test_case = self._test_cases.get(test_case_id)
        return deepcopy(test_case) if test_case is not None else None

    def list(self) -> list[TestCase]:
        return [
            deepcopy(test_case)
            for test_case in sorted(
                self._test_cases.values(), key=lambda item: (item.name.casefold(), str(item.id))
            )
        ]
