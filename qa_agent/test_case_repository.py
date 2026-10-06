"""Repositories for canonical, persisted TestCase definitions."""

from copy import deepcopy
from typing import Protocol
from uuid import UUID

from qa_agent.models import TestCase
from qa_agent.public_ids import format_test_case_public_id, parse_test_case_public_id


class TestCaseRepository(Protocol):
    def save(self, test_case: TestCase) -> None: ...

    def get(self, test_case_id: UUID) -> TestCase | None: ...

    def get_by_public_id(self, public_id: str) -> TestCase | None: ...

    def list(self) -> list[TestCase]: ...


class InMemoryTestCaseRepository:
    """Process-local TestCase storage with copy-on-read/write semantics."""

    def __init__(self) -> None:
        self._test_cases: dict[UUID, TestCase] = {}
        self._next_public_id = 1

    def save(self, test_case: TestCase) -> None:
        existing = self._test_cases.get(test_case.id)
        public_id = existing.public_id if existing is not None else test_case.public_id
        sequence = parse_test_case_public_id(public_id)
        if public_id is None:
            public_id = format_test_case_public_id(self._next_public_id)
            sequence = self._next_public_id
        if any(
            item.id != test_case.id and item.public_id == public_id
            for item in self._test_cases.values()
        ):
            raise ValueError(f"TestCase public ID {public_id} is already in use.")
        test_case.public_id = public_id
        self._next_public_id = max(self._next_public_id, (sequence or 0) + 1)
        self._test_cases[test_case.id] = deepcopy(test_case)

    def get(self, test_case_id: UUID) -> TestCase | None:
        test_case = self._test_cases.get(test_case_id)
        return deepcopy(test_case) if test_case is not None else None

    def get_by_public_id(self, public_id: str) -> TestCase | None:
        test_case = next(
            (item for item in self._test_cases.values() if item.public_id == public_id),
            None,
        )
        return deepcopy(test_case) if test_case is not None else None

    def list(self) -> list[TestCase]:
        return [
            deepcopy(test_case)
            for test_case in sorted(
                self._test_cases.values(), key=lambda item: (item.name.casefold(), str(item.id))
            )
        ]
