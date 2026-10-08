"""Repositories for canonical, persisted TestCase definitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Protocol
from uuid import UUID

from qa_agent.models import TestCase
from qa_agent.public_ids import format_test_case_public_id, parse_test_case_public_id


@dataclass(frozen=True)
class TestCaseCatalogEntry:
    test_case: TestCase
    created_at: datetime
    updated_at: datetime


class TestCaseRepository(Protocol):
    def save(self, test_case: TestCase) -> None: ...

    def get(self, test_case_id: UUID) -> TestCase | None: ...

    def get_by_public_id(self, public_id: str) -> TestCase | None: ...

    def list(self) -> list[TestCase]: ...

    def list_export_catalog(
        self,
        *,
        search: str = "",
        created_from: str = "",
        created_before: str = "",
        updated_from: str = "",
        updated_before: str = "",
        test_case_ids: list[UUID] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[TestCaseCatalogEntry], int]: ...


class InMemoryTestCaseRepository:
    """Process-local TestCase storage with copy-on-read/write semantics."""

    def __init__(self) -> None:
        self._test_cases: dict[UUID, TestCase] = {}
        self._timestamps: dict[UUID, tuple[datetime, datetime]] = {}
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
        now = datetime.now(timezone.utc)
        created_at = self._timestamps.get(test_case.id, (now, now))[0]
        self._timestamps[test_case.id] = (created_at, now)

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

    def list_export_catalog(
        self,
        *,
        search: str = "",
        created_from: str = "",
        created_before: str = "",
        updated_from: str = "",
        updated_before: str = "",
        test_case_ids: list[UUID] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[TestCaseCatalogEntry], int]:
        selected_ids = set(test_case_ids) if test_case_ids is not None else None
        needle = search.casefold().strip()
        entries: list[TestCaseCatalogEntry] = []
        for test_case in self.list():
            timestamps = self._timestamps.get(test_case.id)
            if timestamps is None:
                continue
            created_at, updated_at = timestamps
            if selected_ids is not None and test_case.id not in selected_ids:
                continue
            if needle and needle not in test_case.name.casefold() and needle not in (test_case.public_id or "").casefold():
                continue
            if not _timestamp_matches(created_at, created_from, created_before):
                continue
            if not _timestamp_matches(updated_at, updated_from, updated_before):
                continue
            entries.append(TestCaseCatalogEntry(test_case, created_at, updated_at))
        total = len(entries)
        return entries[offset:offset + limit], total


def _timestamp_matches(value: datetime, starts_on: str, ends_before: str) -> bool:
    if starts_on and value.date() < date.fromisoformat(starts_on):
        return False
    if ends_before and value.date() >= date.fromisoformat(ends_before):
        return False
    return True
