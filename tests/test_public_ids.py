import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qa_agent.models import ExecutionStatus, TestCase as DomainTestCase, TestStep as DomainTestStep
from qa_agent.public_ids import format_run_public_id, format_test_case_public_id
from qa_agent.run_history import (
    InMemoryRunHistoryRepository,
    RunHistoryRecord,
    WorkflowType,
)
from qa_agent.sqlite_storage import SQLiteRunHistoryRepository, SQLiteTestCaseRepository
from qa_agent.test_case_repository import InMemoryTestCaseRepository


def _case() -> DomainTestCase:
    return DomainTestCase(
        name="Registration",
        description="Create and verify a registration.",
        steps=[DomainTestStep(
            name="Open registration",
            description="Open the registration page.",
            expected="The form is visible.",
            order=0,
        )],
    )


def _record(*, run_id=None, test_case_id=None) -> RunHistoryRecord:
    return RunHistoryRecord(
        run_id=run_id or uuid4(),
        test_case_id=test_case_id or uuid4(),
        test_case_name="Registration",
        test_case_description="Create and verify a registration.",
        workflow_type=WorkflowType.VALIDATION,
        outcome="PASSED",
        status=ExecutionStatus.PASSED,
        started_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )


def _set_counter(database: Path, namespace: str, value: int) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE public_id_counters SET next_value = ? WHERE namespace = ?",
            (value, namespace),
        )
        connection.commit()
    finally:
        connection.close()


class PublicIdTests(unittest.TestCase):
    def test_formatters_expand_beyond_minimum_width(self) -> None:
        self.assertEqual(format_test_case_public_id(1), "TC-0001")
        self.assertEqual(format_test_case_public_id(9999), "TC-9999")
        self.assertEqual(format_test_case_public_id(10000), "TC-10000")
        self.assertEqual(format_run_public_id(1), "RUN-000001")
        self.assertEqual(format_run_public_id(999999), "RUN-999999")
        self.assertEqual(format_run_public_id(1000000), "RUN-1000000")

    def test_in_memory_repositories_assign_stable_monotonic_ids(self) -> None:
        cases = InMemoryTestCaseRepository()
        first_case = _case()
        second_case = _case()
        first_case_id = first_case.id
        cases.save(first_case)
        cases.save(second_case)
        cases.save(first_case)

        self.assertEqual(first_case.id, first_case_id)
        self.assertEqual(first_case.public_id, "TC-0001")
        self.assertEqual(second_case.public_id, "TC-0002")
        self.assertEqual(cases.get_by_public_id("TC-0001").id, first_case_id)

        runs = InMemoryRunHistoryRepository()
        first_record = _record(test_case_id=first_case.id)
        second_record = _record(test_case_id=second_case.id)
        first_run_id = first_record.run_id
        saved_first = runs.save(first_record)
        saved_second = runs.save(second_record)

        self.assertEqual(saved_first.run_id, first_run_id)
        self.assertEqual(saved_first.public_id, "RUN-000001")
        self.assertEqual(saved_second.public_id, "RUN-000002")
        self.assertEqual(runs.get_by_public_id("RUN-000001").run_id, first_run_id)

    def test_repositories_reject_duplicate_supplied_public_ids(self) -> None:
        cases = InMemoryTestCaseRepository()
        first_case = _case()
        first_case.public_id = "TC-0009"
        cases.save(first_case)
        second_case = _case()
        second_case.public_id = "TC-0009"
        with self.assertRaises(ValueError):
            cases.save(second_case)

        runs = InMemoryRunHistoryRepository()
        runs.save(_record().model_copy(update={"public_id": "RUN-000009"}))
        with self.assertRaises(ValueError):
            runs.save(_record().model_copy(update={"public_id": "RUN-000009"}))

    def test_sqlite_sequences_expand_at_boundaries_and_survive_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ids.sqlite3"
            cases = SQLiteTestCaseRepository(database)
            _set_counter(database, "test_case", 9999)
            first_case, second_case = _case(), _case()
            original_case_id = first_case.id
            cases.save(first_case)
            cases.save(second_case)
            self.assertEqual(first_case.public_id, "TC-9999")
            self.assertEqual(second_case.public_id, "TC-10000")
            reopened_cases = SQLiteTestCaseRepository(database)
            self.assertEqual(reopened_cases.get(first_case.id).public_id, "TC-9999")
            self.assertEqual(reopened_cases.get(second_case.id).public_id, "TC-10000")
            self.assertEqual(reopened_cases.get(first_case.id).id, original_case_id)

            runs = SQLiteRunHistoryRepository(database)
            _set_counter(database, "run", 999999)
            first_record = runs.save(_record(test_case_id=first_case.id))
            second_record = runs.save(_record(test_case_id=first_case.id))
            self.assertEqual(first_record.public_id, "RUN-999999")
            self.assertEqual(second_record.public_id, "RUN-1000000")
            reopened_runs = SQLiteRunHistoryRepository(database)
            self.assertEqual(reopened_runs.get(first_record.run_id), first_record)
            self.assertEqual(reopened_runs.get_by_public_id("RUN-1000000"), second_record)

    def test_legacy_rows_are_migrated_idempotently_without_changing_uuids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.sqlite3"
            cases = (_case(), _case())
            case_json = []
            for case in cases:
                payload = json.loads(case.model_dump_json(exclude={"steps"}))
                payload.pop("public_id", None)
                case_json.append(json.dumps(payload))

            run_record = _record(test_case_id=cases[0].id)
            run_payload = json.loads(run_record.model_dump_json())
            run_payload.pop("public_id", None)
            run_payload.pop("test_case_public_id", None)

            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "CREATE TABLE test_cases (test_case_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                    "definition_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
                )
                for index, case in enumerate(cases):
                    timestamp = f"2026-01-0{index + 1}T00:00:00+00:00"
                    connection.execute(
                        "INSERT INTO test_cases VALUES (?, ?, ?, ?, ?)",
                        (str(case.id), case.name, case_json[index], timestamp, timestamp),
                    )
                connection.execute(
                    "CREATE TABLE run_history (insertion_order INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "run_id TEXT NOT NULL UNIQUE, test_case_id TEXT NOT NULL, started_at TEXT NOT NULL, "
                    "record_json TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO run_history(run_id, test_case_id, started_at, record_json) VALUES (?, ?, ?, ?)",
                    (str(run_record.run_id), str(run_record.test_case_id),
                     run_record.started_at.isoformat(), json.dumps(run_payload)),
                )
                connection.commit()
            finally:
                connection.close()

            cases_repository = SQLiteTestCaseRepository(database)
            migrated_cases = cases_repository.list()
            self.assertEqual(cases_repository.get(cases[0].id).public_id, "TC-0001")
            self.assertEqual(cases_repository.get(cases[1].id).public_id, "TC-0002")
            self.assertEqual({item.id for item in migrated_cases}, {item.id for item in cases})
            history_repository = SQLiteRunHistoryRepository(database)
            migrated_run = history_repository.get(run_record.run_id)
            self.assertEqual(migrated_run.public_id, "RUN-000001")
            self.assertEqual(migrated_run.test_case_public_id, "TC-0001")
            self.assertEqual(migrated_run.test_case_id, cases[0].id)

            self.assertEqual(SQLiteTestCaseRepository(database).list(), migrated_cases)
            self.assertEqual(SQLiteRunHistoryRepository(database).get(run_record.run_id), migrated_run)
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM test_cases").fetchone()[0], 2)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM run_history").fetchone()[0], 1)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
