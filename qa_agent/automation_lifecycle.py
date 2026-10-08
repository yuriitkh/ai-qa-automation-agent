"""Persistent status for the relationship between TestCases and saved plans."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID

from qa_agent.models import TestCase
from qa_agent.expected_result_coverage import has_sufficient_test_case_coverage
from qa_agent.plan_store import PlanStore
from qa_agent.test_plan_validation import validate_executable_plan


class AutomationStatus(str, Enum):
    NOT_AUTOMATED = "NOT_AUTOMATED"
    AUTOMATION_READY = "AUTOMATION_READY"
    NEEDS_UPDATE = "NEEDS_UPDATE"
    NEEDS_VALIDATION = "NEEDS_VALIDATION"
    AUTOMATION_FAILED = "AUTOMATION_FAILED"


@dataclass(frozen=True)
class AutomationLifecycleRecord:
    test_case_id: UUID
    state: AutomationStatus
    definition_fingerprint: str
    plan_fingerprint: str
    updated_at: datetime


class AutomationLifecycleRepository(Protocol):
    def get(self, test_case_id: UUID) -> AutomationLifecycleRecord | None: ...
    def save(self, record: AutomationLifecycleRecord) -> None: ...


class SQLiteAutomationLifecycleRepository:
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
                """CREATE TABLE IF NOT EXISTS test_case_automation_lifecycle (
                    test_case_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    definition_fingerprint TEXT NOT NULL,
                    plan_fingerprint TEXT NOT NULL,
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

    def get(self, test_case_id: UUID) -> AutomationLifecycleRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM test_case_automation_lifecycle WHERE test_case_id=?",
                (str(test_case_id),),
            ).fetchone()
        if row is None:
            return None
        return AutomationLifecycleRecord(
            UUID(row["test_case_id"]), AutomationStatus(row["state"]),
            row["definition_fingerprint"], row["plan_fingerprint"],
            datetime.fromisoformat(row["updated_at"]),
        )

    def save(self, record: AutomationLifecycleRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO test_case_automation_lifecycle
                    (test_case_id, state, definition_fingerprint, plan_fingerprint, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(test_case_id) DO UPDATE SET state=excluded.state,
                        definition_fingerprint=excluded.definition_fingerprint,
                        plan_fingerprint=excluded.plan_fingerprint,
                        updated_at=excluded.updated_at""",
                (str(record.test_case_id), record.state.value,
                 record.definition_fingerprint, record.plan_fingerprint,
                 record.updated_at.isoformat()),
            )


class InMemoryAutomationLifecycleRepository:
    def __init__(self) -> None:
        self._records: dict[UUID, AutomationLifecycleRecord] = {}
        self._lock = RLock()

    def get(self, test_case_id: UUID) -> AutomationLifecycleRecord | None:
        with self._lock:
            return self._records.get(test_case_id)

    def save(self, record: AutomationLifecycleRecord) -> None:
        with self._lock:
            self._records[record.test_case_id] = record


class AutomationLifecycleService:
    def __init__(self, repository: AutomationLifecycleRepository, plan_store: PlanStore) -> None:
        self._repository = repository
        self._plan_store = plan_store

    def status(self, test_case: TestCase) -> AutomationStatus:
        record = self._repository.get(test_case.id)
        current_definition = definition_fingerprint(test_case)
        current_plans = plan_fingerprint(test_case, self._plan_store)
        complete = has_complete_plans(test_case, self._plan_store)
        coverage_sufficient = (
            complete and has_sufficient_test_case_coverage(test_case, self._plan_store)
        )
        if record is None:
            return AutomationStatus.NEEDS_VALIDATION if complete else AutomationStatus.NOT_AUTOMATED
        if record.state == AutomationStatus.NOT_AUTOMATED:
            return AutomationStatus.NEEDS_VALIDATION if complete else AutomationStatus.NOT_AUTOMATED
        if current_definition != record.definition_fingerprint:
            return AutomationStatus.NEEDS_UPDATE
        if record.state == AutomationStatus.NEEDS_UPDATE:
            return AutomationStatus.NEEDS_UPDATE
        if record.state == AutomationStatus.AUTOMATION_READY:
            if not complete:
                return AutomationStatus.NEEDS_UPDATE
            if not coverage_sufficient:
                return AutomationStatus.NEEDS_VALIDATION
            if current_plans != record.plan_fingerprint:
                return AutomationStatus.NEEDS_VALIDATION
        if record.state == AutomationStatus.NEEDS_VALIDATION and not complete:
            return AutomationStatus.NEEDS_UPDATE
        return record.state

    def mark_automation_completed(self, test_case: TestCase) -> AutomationStatus:
        status = (
            AutomationStatus.NEEDS_VALIDATION
            if has_complete_plans(test_case, self._plan_store)
            else AutomationStatus.AUTOMATION_FAILED
        )
        self._save(test_case, status)
        return status

    def mark_automation_failed(self, test_case: TestCase) -> AutomationStatus:
        current = self.status(test_case)
        # A failed attempt does not invalidate an already validated plan set.
        if current == AutomationStatus.AUTOMATION_READY:
            return current
        self._save(test_case, AutomationStatus.AUTOMATION_FAILED)
        return AutomationStatus.AUTOMATION_FAILED

    def mark_validation_completed(
        self,
        test_case: TestCase,
        passed: bool,
        *,
        validated_plan_fingerprint: str | None = None,
    ) -> AutomationStatus:
        current = self.status(test_case)
        if not has_complete_plans(test_case, self._plan_store):
            return AutomationStatus.NEEDS_UPDATE if current != AutomationStatus.NOT_AUTOMATED else current
        coverage_sufficient = has_sufficient_test_case_coverage(
            test_case, self._plan_store
        )
        current_plan_fingerprint = plan_fingerprint(test_case, self._plan_store)
        expected_fingerprint = validated_plan_fingerprint or current_plan_fingerprint
        still_current = current_plan_fingerprint == expected_fingerprint
        status = (
            AutomationStatus.AUTOMATION_READY
            if passed and still_current and coverage_sufficient
            else AutomationStatus.NEEDS_VALIDATION
        )
        # Store the exact pinned versions that Validation exercised. If a plan
        # changes concurrently, status() observes the fingerprint mismatch.
        stored_fingerprint = expected_fingerprint if status == AutomationStatus.AUTOMATION_READY else current_plan_fingerprint
        self._save(test_case, status, plan_fingerprint_value=stored_fingerprint)
        return status

    def mark_test_case_changed(self, test_case: TestCase) -> None:
        record = self._repository.get(test_case.id)
        if record is None:
            return
        if record.state != AutomationStatus.NOT_AUTOMATED:
            self._repository.save(AutomationLifecycleRecord(
                test_case_id=test_case.id,
                state=AutomationStatus.NEEDS_UPDATE,
                definition_fingerprint=record.definition_fingerprint,
                plan_fingerprint=record.plan_fingerprint,
                updated_at=datetime.now(timezone.utc),
            ))

    def _save(
        self,
        test_case: TestCase,
        status: AutomationStatus,
        *,
        plan_fingerprint_value: str | None = None,
    ) -> None:
        self._repository.save(AutomationLifecycleRecord(
            test_case_id=test_case.id,
            state=status,
            definition_fingerprint=definition_fingerprint(test_case),
            plan_fingerprint=(
                plan_fingerprint_value
                if plan_fingerprint_value is not None
                else plan_fingerprint(test_case, self._plan_store)
            ),
            updated_at=datetime.now(timezone.utc),
        ))


def definition_fingerprint(test_case: TestCase) -> str:
    definition = test_case.model_dump(mode="json", exclude={"public_id", "steps"})
    return hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_fingerprint(test_case: TestCase, plan_store: PlanStore) -> str:
    versions = []
    for step in test_case.steps:
        version = plan_store.find(step.id)
        versions.append((str(step.id), str(version.id) if version else ""))
    return hashlib.sha256(json.dumps(versions, separators=(",", ":")).encode()).hexdigest()


def plan_fingerprint_for_versions(test_case: TestCase, versions: dict[UUID, UUID]) -> str:
    selected = [(str(step.id), str(versions.get(step.id, ""))) for step in test_case.steps]
    return hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()


def has_complete_plans(test_case: TestCase, plan_store: PlanStore) -> bool:
    if not test_case.steps:
        return False
    for step in test_case.steps:
        version = plan_store.find(step.id)
        plan = plan_store.find_test_plan(step.id)
        if version is None or plan is None or version.test_plan_id != plan.id or plan.test_step_id != step.id:
            return False
        try:
            validate_executable_plan(version.qa_test_plan)
        except (TypeError, ValueError):
            return False
    return True
