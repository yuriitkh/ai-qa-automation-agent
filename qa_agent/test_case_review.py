"""Persistent TestCase review state, independent of automation lifecycle."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID

from qa_agent.automation_lifecycle import plan_fingerprint, has_complete_plans
from qa_agent.expected_result_coverage import has_sufficient_test_case_coverage
from qa_agent.models import TestCase
from qa_agent.plan_store import PlanStore


class TestCaseReviewStatus(str, Enum):
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    APPROVED = "APPROVED"


@dataclass(frozen=True)
class TestCaseReviewRecord:
    test_case_id: UUID
    status: TestCaseReviewStatus
    approved_plan_fingerprint: str | None
    updated_at: datetime


class TestCaseReviewRepository(Protocol):
    def get(self, test_case_id: UUID) -> TestCaseReviewRecord | None: ...
    def save(self, record: TestCaseReviewRecord) -> None: ...


class SQLiteTestCaseReviewRepository:
    """Additive metadata table; canonical TestCase JSON is left untouched."""

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path).expanduser()
        self._memory_connection = None
        if str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS test_case_review (
                    test_case_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    approved_plan_fingerprint TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )

    @contextmanager
    def _connection(self):
        if self._memory_connection is not None:
            with self._memory_connection as connection:
                yield connection
            return
        connection = sqlite3.connect(str(self._path))
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, test_case_id: UUID) -> TestCaseReviewRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM test_case_review WHERE test_case_id=?",
                (str(test_case_id),),
            ).fetchone()
        if row is None:
            return None
        return TestCaseReviewRecord(
            UUID(row["test_case_id"]),
            TestCaseReviewStatus(row["status"]),
            row["approved_plan_fingerprint"],
            datetime.fromisoformat(row["updated_at"]),
        )

    def save(self, record: TestCaseReviewRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO test_case_review
                    (test_case_id, status, approved_plan_fingerprint, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(test_case_id) DO UPDATE SET
                        status=excluded.status,
                        approved_plan_fingerprint=excluded.approved_plan_fingerprint,
                        updated_at=excluded.updated_at""",
                (str(record.test_case_id), record.status.value,
                 record.approved_plan_fingerprint, record.updated_at.isoformat()),
            )


class InMemoryTestCaseReviewRepository:
    def __init__(self) -> None:
        self._records: dict[UUID, TestCaseReviewRecord] = {}
        self._lock = RLock()

    def get(self, test_case_id: UUID) -> TestCaseReviewRecord | None:
        with self._lock:
            return self._records.get(test_case_id)

    def save(self, record: TestCaseReviewRecord) -> None:
        with self._lock:
            self._records[record.test_case_id] = record


class TestCaseReviewService:
    def __init__(self, repository: TestCaseReviewRepository, plan_store: PlanStore) -> None:
        self._repository = repository
        self._plan_store = plan_store

    def record(self, test_case_id: UUID) -> TestCaseReviewRecord | None:
        return self._repository.get(test_case_id)

    def status(self, test_case_id: UUID) -> TestCaseReviewStatus:
        record = self.record(test_case_id)
        # TestCases persisted before review metadata existed remain usable.
        return record.status if record is not None else TestCaseReviewStatus.APPROVED

    def mark_ready_for_review(self, test_case: TestCase) -> None:
        self._save(test_case.id, TestCaseReviewStatus.READY_FOR_REVIEW, None)

    def approve_test_case(self, test_case: TestCase) -> None:
        self._save(test_case.id, TestCaseReviewStatus.APPROVED, None)

    def approve_for_validation(self, test_case: TestCase) -> str:
        if self.status(test_case.id) != TestCaseReviewStatus.APPROVED:
            raise ValueError("Approve the TestCase before approving its automation for Validation.")
        if not has_complete_plans(test_case, self._plan_store):
            raise ValueError("Generate complete automation before approving it for Validation.")
        if not has_sufficient_test_case_coverage(test_case, self._plan_store):
            raise ValueError("Automation must cover each expected result before Validation.")
        fingerprint = plan_fingerprint(test_case, self._plan_store)
        self._save(test_case.id, TestCaseReviewStatus.APPROVED, fingerprint)
        return fingerprint

    def validation_approved_for(self, test_case: TestCase) -> bool:
        record = self.record(test_case.id)
        if record is None:
            return True
        return (
            record.status == TestCaseReviewStatus.APPROVED
            and record.approved_plan_fingerprint is not None
            and record.approved_plan_fingerprint == plan_fingerprint(test_case, self._plan_store)
        )

    def plan_fingerprint(self, test_case: TestCase) -> str | None:
        record = self.record(test_case.id)
        return record.approved_plan_fingerprint if record else None

    def _save(
        self,
        test_case_id: UUID,
        status: TestCaseReviewStatus,
        approved_plan_fingerprint: str | None,
    ) -> None:
        self._repository.save(TestCaseReviewRecord(
            test_case_id=test_case_id,
            status=status,
            approved_plan_fingerprint=approved_plan_fingerprint,
            updated_at=datetime.now(timezone.utc),
        ))
