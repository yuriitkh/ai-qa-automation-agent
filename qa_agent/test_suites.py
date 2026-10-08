"""Lightweight persisted Test Suite organization for saved TestCases."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from qa_agent.models import TestCase
from qa_agent.test_case_repository import TestCaseRepository


@dataclass(frozen=True)
class TestSuite:
    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


class TestSuiteRepository(Protocol):
    def create(self, name: str, description: str) -> TestSuite: ...
    def get(self, suite_id: UUID) -> TestSuite | None: ...
    def list(self) -> list[TestSuite]: ...
    def list_page(self, *, search: str = "", offset: int = 0, limit: int = 100) -> tuple[list[TestSuite], int]: ...
    def update(self, suite_id: UUID, name: str, description: str) -> TestSuite | None: ...
    def members(self, suite_id: UUID) -> list[UUID]: ...
    def add_member(self, suite_id: UUID, test_case_id: UUID) -> None: ...
    def remove_member(self, suite_id: UUID, test_case_id: UUID) -> None: ...
    def move_member(self, suite_id: UUID, test_case_id: UUID, direction: int) -> None: ...


class SQLiteTestSuiteRepository:
    """Idempotently migrate and persist suites in the application's SQLite DB."""

    def __init__(self, database_path: str | Path) -> None:
        raw_path = str(database_path)
        self._database_path = raw_path if raw_path == ":memory:" else str(Path(raw_path).expanduser())
        self._memory_connection: sqlite3.Connection | None = None
        if self._database_path != ":memory:":
            Path(self._database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(":memory:")
            self._memory_connection.row_factory = sqlite3.Row
            self._memory_connection.execute("PRAGMA foreign_keys = ON")
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS test_suites (
                    suite_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS test_suite_members (
                    suite_id TEXT NOT NULL REFERENCES test_suites(suite_id) ON DELETE CASCADE,
                    test_case_id TEXT NOT NULL,
                    order_index INTEGER NOT NULL CHECK (order_index >= 0),
                    PRIMARY KEY (suite_id, test_case_id),
                    UNIQUE (suite_id, order_index)
                );
                CREATE INDEX IF NOT EXISTS idx_test_suite_members_order
                    ON test_suite_members(suite_id, order_index);
                """
            )

    @contextmanager
    def _connection(self):
        if self._memory_connection is not None:
            with self._memory_connection:
                yield self._memory_connection
            return
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, name: str, description: str) -> TestSuite:
        now = datetime.now(timezone.utc)
        suite = TestSuite(uuid4(), name, description, now, now)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO test_suites VALUES (?, ?, ?, ?, ?)",
                (str(suite.id), suite.name, suite.description, now.isoformat(), now.isoformat()),
            )
        return suite

    def get(self, suite_id: UUID) -> TestSuite | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM test_suites WHERE suite_id = ?", (str(suite_id),)).fetchone()
        return self._to_suite(row) if row else None

    def list(self) -> list[TestSuite]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM test_suites ORDER BY name COLLATE NOCASE, suite_id").fetchall()
        return [self._to_suite(row) for row in rows]

    def list_page(
        self, *, search: str = "", offset: int = 0, limit: int = 100
    ) -> tuple[list[TestSuite], int]:
        conditions = " WHERE instr(lower(name), lower(?)) > 0" if search.strip() else ""
        parameters: list[object] = [search.strip()] if conditions else []
        safe_limit = min(max(int(limit), 1), 500)
        safe_offset = max(int(offset), 0)
        with self._connection() as connection:
            total = int(connection.execute(
                "SELECT COUNT(*) FROM test_suites" + conditions, parameters
            ).fetchone()[0])
            rows = connection.execute(
                "SELECT * FROM test_suites" + conditions
                + " ORDER BY name COLLATE NOCASE, suite_id LIMIT ? OFFSET ?",
                [*parameters, safe_limit, safe_offset],
            ).fetchall()
        return [self._to_suite(row) for row in rows], total

    def update(self, suite_id: UUID, name: str, description: str) -> TestSuite | None:
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE test_suites SET name = ?, description = ?, updated_at = ? WHERE suite_id = ?",
                (name, description, now.isoformat(), str(suite_id)),
            )
            row = connection.execute("SELECT * FROM test_suites WHERE suite_id = ?", (str(suite_id),)).fetchone()
        return self._to_suite(row) if cursor.rowcount and row else None

    def members(self, suite_id: UUID) -> list[UUID]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT test_case_id FROM test_suite_members WHERE suite_id = ? ORDER BY order_index",
                (str(suite_id),),
            ).fetchall()
        return [UUID(row["test_case_id"]) for row in rows]

    def add_member(self, suite_id: UUID, test_case_id: UUID) -> None:
        with self._connection() as connection:
            exists = connection.execute("SELECT 1 FROM test_suites WHERE suite_id = ?", (str(suite_id),)).fetchone()
            if exists is None:
                raise KeyError("Test Suite not found.")
            current = connection.execute(
                "SELECT 1 FROM test_suite_members WHERE suite_id = ? AND test_case_id = ?",
                (str(suite_id), str(test_case_id)),
            ).fetchone()
            if current is not None:
                return
            order_index = connection.execute(
                "SELECT COALESCE(MAX(order_index) + 1, 0) FROM test_suite_members WHERE suite_id = ?",
                (str(suite_id),),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO test_suite_members(suite_id, test_case_id, order_index) VALUES (?, ?, ?)",
                (str(suite_id), str(test_case_id), int(order_index)),
            )

    def remove_member(self, suite_id: UUID, test_case_id: UUID) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM test_suite_members WHERE suite_id = ? AND test_case_id = ?",
                (str(suite_id), str(test_case_id)),
            )
            self._normalize_order(connection, suite_id)

    def move_member(self, suite_id: UUID, test_case_id: UUID, direction: int) -> None:
        if direction not in {-1, 1}:
            raise ValueError("Direction must be -1 or 1.")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT test_case_id FROM test_suite_members WHERE suite_id = ? ORDER BY order_index",
                (str(suite_id),),
            ).fetchall()
            items = [row["test_case_id"] for row in rows]
            member_id = str(test_case_id)
            if member_id not in items:
                raise KeyError("TestCase is not in this Test Suite.")
            index = items.index(member_id)
            target = min(max(index + direction, 0), len(items) - 1)
            items[index], items[target] = items[target], items[index]
            self._write_order(connection, suite_id, items)

    @staticmethod
    def _normalize_order(connection: sqlite3.Connection, suite_id: UUID) -> None:
        rows = connection.execute(
            "SELECT test_case_id FROM test_suite_members WHERE suite_id = ? ORDER BY order_index",
            (str(suite_id),),
        ).fetchall()
        SQLiteTestSuiteRepository._write_order(connection, suite_id, [row["test_case_id"] for row in rows])

    @staticmethod
    def _write_order(connection: sqlite3.Connection, suite_id: UUID, ids: list[str]) -> None:
        maximum = connection.execute(
            "SELECT COALESCE(MAX(order_index), -1) FROM test_suite_members WHERE suite_id = ?",
            (str(suite_id),),
        ).fetchone()[0]
        offset = int(maximum) + len(ids) + 1
        connection.execute(
            "UPDATE test_suite_members SET order_index = order_index + ? WHERE suite_id = ?",
            (offset, str(suite_id)),
        )
        for index, case_id in enumerate(ids):
            connection.execute(
                "UPDATE test_suite_members SET order_index = ? WHERE suite_id = ? AND test_case_id = ?",
                (index, str(suite_id), case_id),
            )

    @staticmethod
    def _to_suite(row: sqlite3.Row) -> TestSuite:
        return TestSuite(
            UUID(row["suite_id"]), row["name"], row["description"],
            datetime.fromisoformat(row["created_at"]), datetime.fromisoformat(row["updated_at"]),
        )


class InMemoryTestSuiteRepository:
    def __init__(self) -> None:
        self._suites: dict[UUID, TestSuite] = {}
        self._members: dict[UUID, list[UUID]] = {}

    def create(self, name: str, description: str) -> TestSuite:
        now = datetime.now(timezone.utc)
        suite = TestSuite(uuid4(), name, description, now, now)
        self._suites[suite.id] = suite
        self._members[suite.id] = []
        return suite

    def get(self, suite_id: UUID) -> TestSuite | None:
        return self._suites.get(suite_id)

    def list(self) -> list[TestSuite]:
        return sorted(self._suites.values(), key=lambda item: (item.name.casefold(), str(item.id)))

    def list_page(
        self, *, search: str = "", offset: int = 0, limit: int = 100
    ) -> tuple[list[TestSuite], int]:
        needle = search.casefold().strip()
        suites = [
            suite for suite in self.list()
            if not needle or needle in suite.name.casefold()
        ]
        return suites[offset:offset + limit], len(suites)

    def update(self, suite_id: UUID, name: str, description: str) -> TestSuite | None:
        old = self._suites.get(suite_id)
        if old is None:
            return None
        suite = TestSuite(old.id, name, description, old.created_at, datetime.now(timezone.utc))
        self._suites[suite_id] = suite
        return suite

    def members(self, suite_id: UUID) -> list[UUID]:
        return list(self._members.get(suite_id, []))

    def add_member(self, suite_id: UUID, test_case_id: UUID) -> None:
        if suite_id not in self._members:
            raise KeyError("Test Suite not found.")
        if test_case_id not in self._members[suite_id]:
            self._members[suite_id].append(test_case_id)

    def remove_member(self, suite_id: UUID, test_case_id: UUID) -> None:
        if suite_id in self._members and test_case_id in self._members[suite_id]:
            self._members[suite_id].remove(test_case_id)

    def move_member(self, suite_id: UUID, test_case_id: UUID, direction: int) -> None:
        if direction not in {-1, 1}:
            raise ValueError("Direction must be -1 or 1.")
        items = self._members.get(suite_id)
        if items is None or test_case_id not in items:
            raise KeyError("TestCase is not in this Test Suite.")
        index = items.index(test_case_id)
        target = min(max(index + direction, 0), len(items) - 1)
        items[index], items[target] = items[target], items[index]


class TestSuiteService:
    def __init__(self, repository: TestSuiteRepository, test_cases: TestCaseRepository | None) -> None:
        self._repository = repository
        self._test_cases = test_cases

    def create(self, name: str, description: str) -> TestSuite:
        return self._repository.create(_clean_name(name), _clean_description(description))

    def update(self, suite_id: UUID, name: str, description: str) -> TestSuite:
        suite = self._repository.update(suite_id, _clean_name(name), _clean_description(description))
        if suite is None:
            raise KeyError("Test Suite not found.")
        return suite

    def get(self, suite_id: UUID) -> TestSuite | None:
        return self._repository.get(suite_id)

    def list(self) -> list[TestSuite]:
        return self._repository.list()

    def list_page(
        self, *, search: str = "", offset: int = 0, limit: int = 100
    ) -> tuple[list[TestSuite], int]:
        list_page = getattr(self._repository, "list_page", None)
        if list_page is not None:
            return list_page(search=search, offset=offset, limit=limit)
        needle = search.casefold().strip()
        suites = [suite for suite in self._repository.list() if not needle or needle in suite.name.casefold()]
        return suites[offset:offset + limit], len(suites)

    def member_ids(self, suite_id: UUID) -> list[UUID]:
        if self._test_cases is None:
            return []
        available_ids = {case.id for case in self._test_cases.list()}
        return [case_id for case_id in self._repository.members(suite_id) if case_id in available_ids]

    def all_member_ids(self, suite_id: UUID) -> list[UUID]:
        """Return persisted membership in order, including stale references."""
        return self._repository.members(suite_id)

    def members(self, suite_id: UUID) -> list[TestCase]:
        if self._test_cases is None:
            return []
        by_id = {case.id: case for case in self._test_cases.list()}
        return [by_id[case_id] for case_id in self._repository.members(suite_id) if case_id in by_id]

    def add_member(self, suite_id: UUID, test_case_id: UUID) -> None:
        if self._repository.get(suite_id) is None:
            raise KeyError("Test Suite not found.")
        if not self._case_exists(test_case_id):
            raise KeyError("TestCase not found.")
        self._repository.add_member(suite_id, test_case_id)

    def remove_member(self, suite_id: UUID, test_case_id: UUID) -> None:
        self._repository.remove_member(suite_id, test_case_id)

    def move_member(self, suite_id: UUID, test_case_id: UUID, direction: int) -> None:
        self._repository.move_member(suite_id, test_case_id, direction)

    def _case_exists(self, case_id: UUID) -> bool:
        return self._test_cases is not None and self._test_cases.get(case_id) is not None


def _clean_name(name: str) -> str:
    value = name.strip()
    if not value or len(value) > 120:
        raise ValueError("Suite name must contain 1 to 120 characters.")
    return value


def _clean_description(description: str) -> str:
    value = description.strip()
    if len(value) > 1000:
        raise ValueError("Suite description must be 1000 characters or fewer.")
    return value

