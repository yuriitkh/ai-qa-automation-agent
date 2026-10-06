"""File-backed SQLite implementations of plan and execution storage contracts."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

from qa_agent.execution_repository import ExecutionRepository
from qa_agent.models import (
    Execution,
    ExecutionStatus,
    Evidence,
    EvidenceType,
    QATestPlan,
    TestCase,
    TestPlan,
    TestPlanVersion,
)
from qa_agent.plan_store import PlanStore, _validate_plan_store_write
from qa_agent.run_history import RunHistoryRecord


class _SQLiteStorage:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(self._db_path)
            self._memory_connection.row_factory = sqlite3.Row

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            with self._memory_connection:
                yield self._memory_connection
            return

        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()


class SQLitePlanStore(_SQLiteStorage):
    """Persist immutable TestPlanVersion history and each TestStep's current version."""

    def __init__(self, db_path: str | Path) -> None:
        super().__init__(db_path)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_test_plans (
                    test_step_id TEXT PRIMARY KEY,
                    test_plan_id TEXT NOT NULL,
                    test_plan_name TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL CHECK (version_number >= 1),
                    created_at TEXT NOT NULL,
                    qa_test_plan_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS test_plan_versions (
                    version_id TEXT PRIMARY KEY,
                    test_step_id TEXT NOT NULL,
                    test_plan_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL CHECK (version_number >= 1),
                    created_at TEXT NOT NULL,
                    qa_test_plan_json TEXT NOT NULL,
                    UNIQUE (test_plan_id, version_number)
                )
                """
            )
            # Migrate the one current version from databases created by the
            # previous schema. Earlier versions cannot be reconstructed.
            connection.execute(
                """
                INSERT OR IGNORE INTO test_plan_versions (
                    version_id, test_step_id, test_plan_id, version_number,
                    created_at, qa_test_plan_json
                )
                SELECT version_id, test_step_id, test_plan_id, version_number,
                       created_at, qa_test_plan_json
                FROM cached_test_plans
                """
            )

    def save(
        self,
        test_step_id: UUID,
        plan_version: TestPlanVersion,
        *,
        test_plan: TestPlan,
    ) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """
                SELECT test_plan_id, version_id, version_number
                FROM cached_test_plans WHERE test_step_id = ?
                """,
                (str(test_step_id),),
            ).fetchone()
            _validate_plan_store_write(
                test_step_id,
                plan_version,
                test_plan,
                current_plan_id=UUID(current["test_plan_id"]) if current else None,
                current_version_number=int(current["version_number"]) if current else None,
                current_version_id=UUID(current["version_id"]) if current else None,
            )
            by_id = connection.execute(
                "SELECT * FROM test_plan_versions WHERE version_id = ?",
                (str(plan_version.id),),
            ).fetchone()
            if by_id is not None:
                expected = (
                    str(test_step_id), str(test_plan.id), plan_version.version,
                    plan_version.created_at.isoformat(), plan_version.qa_test_plan.model_dump_json(),
                )
                actual = (
                    by_id["test_step_id"], by_id["test_plan_id"],
                    int(by_id["version_number"]), by_id["created_at"],
                    by_id["qa_test_plan_json"],
                )
                if actual != expected:
                    raise ValueError("A TestPlanVersion ID cannot be reused for different content.")
            else:
                number = connection.execute(
                    "SELECT version_id FROM test_plan_versions "
                    "WHERE test_plan_id = ? AND version_number = ?",
                    (str(test_plan.id), plan_version.version),
                ).fetchone()
                if number is not None:
                    raise ValueError("A TestPlan version number cannot identify two versions.")
                connection.execute(
                    """
                    INSERT INTO test_plan_versions (
                        version_id, test_step_id, test_plan_id, version_number,
                        created_at, qa_test_plan_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(plan_version.id), str(test_step_id), str(test_plan.id),
                        plan_version.version, plan_version.created_at.isoformat(),
                        plan_version.qa_test_plan.model_dump_json(),
                    ),
                )
            connection.execute(
                """
                INSERT INTO cached_test_plans (
                    test_step_id, test_plan_id, test_plan_name, version_id,
                    version_number, created_at, qa_test_plan_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(test_step_id) DO UPDATE SET
                    test_plan_id = excluded.test_plan_id,
                    test_plan_name = excluded.test_plan_name,
                    version_id = excluded.version_id,
                    version_number = excluded.version_number,
                    created_at = excluded.created_at,
                    qa_test_plan_json = excluded.qa_test_plan_json
                """,
                (
                    str(test_step_id),
                    str(test_plan.id),
                    test_plan.name,
                    str(plan_version.id),
                    plan_version.version,
                    plan_version.created_at.isoformat(),
                    plan_version.qa_test_plan.model_dump_json(),
                ),
            )

    def find(self, test_step_id: UUID) -> TestPlanVersion | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM cached_test_plans WHERE test_step_id = ?",
                (str(test_step_id),),
            ).fetchone()
        if row is None:
            return None
        return self._to_version(row)

    def get_version(self, version_id: UUID) -> TestPlanVersion | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM test_plan_versions WHERE version_id = ?",
                (str(version_id),),
            ).fetchone()
        return self._to_version(row) if row is not None else None

    def find_test_plan(self, test_step_id: UUID) -> TestPlan | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT test_plan_id, test_step_id, test_plan_name "
                "FROM cached_test_plans WHERE test_step_id = ?",
                (str(test_step_id),),
            ).fetchone()
        if row is None:
            return None
        return TestPlan(
            id=UUID(row["test_plan_id"]),
            test_step_id=UUID(row["test_step_id"]),
            name=row["test_plan_name"],
        )

    @staticmethod
    def _to_version(row: sqlite3.Row) -> TestPlanVersion:
        return TestPlanVersion(
            id=UUID(row["version_id"]),
            test_plan_id=UUID(row["test_plan_id"]),
            version=int(row["version_number"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            qa_test_plan=QATestPlan.model_validate_json(row["qa_test_plan_json"]),
        )


class SQLiteTestCaseRepository(_SQLiteStorage):
    """Persist canonical TestCase definitions independently of run history."""

    def __init__(self, db_path: str | Path) -> None:
        super().__init__(db_path)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS test_cases (
                    test_case_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    definition_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, test_case: TestCase) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO test_cases (
                    test_case_id, name, definition_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(test_case_id) DO UPDATE SET
                    name = excluded.name,
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(test_case.id),
                    test_case.name,
                    test_case.model_dump_json(exclude={"steps"}),
                    now,
                    now,
                ),
            )

    def get(self, test_case_id: UUID) -> TestCase | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT definition_json FROM test_cases WHERE test_case_id = ?",
                (str(test_case_id),),
            ).fetchone()
        return TestCase.model_validate_json(row["definition_json"]) if row else None

    def list(self) -> list[TestCase]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT definition_json FROM test_cases "
                "ORDER BY name COLLATE NOCASE, test_case_id"
            ).fetchall()
        return [TestCase.model_validate_json(row["definition_json"]) for row in rows]

    def find_test_plan(self, test_step_id: UUID) -> TestPlan | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT test_plan_id, test_plan_name FROM cached_test_plans "
                "WHERE test_step_id = ?",
                (str(test_step_id),),
            ).fetchone()
        if row is None:
            return None
        return TestPlan(
            id=UUID(row["test_plan_id"]),
            test_step_id=test_step_id,
            name=row["test_plan_name"],
        )


class SQLiteExecutionRepository(_SQLiteStorage):
    """Persist execution records with an explicit insertion sequence."""

    def __init__(self, db_path: str | Path) -> None:
        super().__init__(db_path)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    insertion_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL UNIQUE,
                    test_step_id TEXT NOT NULL,
                    test_plan_version_id TEXT NOT NULL,
                    planned_step_index INTEGER,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    actual_result_json TEXT NOT NULL,
                    error TEXT,
                    runner_result_json TEXT
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(executions)").fetchall()
            }
            if "planned_step_index" not in columns:
                connection.execute(
                    "ALTER TABLE executions ADD COLUMN planned_step_index INTEGER"
                )

    def save(self, execution: Execution) -> None:
        actual_result_json = json.dumps(execution.actual_result, ensure_ascii=False)
        runner_result_json = (
            json.dumps(execution.runner_result, ensure_ascii=False)
            if execution.runner_result is not None
            else None
        )
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO executions (
                        execution_id, test_step_id, test_plan_version_id,
                        planned_step_index, status,
                        started_at, finished_at, actual_result_json, error,
                        runner_result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(execution.id),
                        str(execution.test_step_id),
                        str(execution.test_plan_version_id),
                        execution.planned_step_index,
                        execution.status.value,
                        execution.started_at.isoformat(),
                        execution.finished_at.isoformat() if execution.finished_at else None,
                        actual_result_json,
                        execution.error,
                        runner_result_json,
                    ),
                )
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed: executions.execution_id" not in str(error):
                raise
            raise ValueError(f"Execution {execution.id} already exists.") from error

    def get(self, execution_id: UUID) -> Execution | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM executions WHERE execution_id = ?",
                (str(execution_id),),
            ).fetchone()
        return self._to_execution(row) if row is not None else None

    def list_for_test_step(self, test_step_id: UUID) -> list[Execution]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM executions WHERE test_step_id = ? "
                "ORDER BY insertion_order ASC",
                (str(test_step_id),),
            ).fetchall()
        return [self._to_execution(row) for row in rows]

    def list_for_test_case(self, test_case: TestCase) -> list[Execution]:
        test_step_ids = [str(step.id) for step in test_case.steps]
        placeholders = ", ".join("?" for _ in test_step_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM executions WHERE test_step_id IN ({placeholders}) "
                "ORDER BY insertion_order ASC",
                test_step_ids,
            ).fetchall()
        return [self._to_execution(row) for row in rows]

    @staticmethod
    def _to_execution(row: sqlite3.Row) -> Execution:
        runner_result = (
            json.loads(row["runner_result_json"])
            if row["runner_result_json"] is not None
            else None
        )
        evidence = []
        if isinstance(runner_result, dict):
            for item in runner_result.get("evidence", []):
                if (
                    isinstance(item, dict)
                    and item.get("type") == EvidenceType.SCREENSHOT.value
                    and isinstance(item.get("path"), str)
                    and item["path"]
                ):
                    evidence.append(Evidence(
                        id=UUID(item["evidence_id"]) if item.get("evidence_id") else uuid4(),
                        execution_id=UUID(row["execution_id"]),
                        type=EvidenceType.SCREENSHOT,
                        path=item["path"],
                        description=item.get("description"),
                    ))
        return Execution(
            id=UUID(row["execution_id"]),
            test_step_id=UUID(row["test_step_id"]),
            test_plan_version_id=UUID(row["test_plan_version_id"]),
            planned_step_index=row["planned_step_index"],
            status=ExecutionStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=(
                datetime.fromisoformat(row["finished_at"])
                if row["finished_at"] is not None
                else None
            ),
            actual_result=json.loads(row["actual_result_json"]),
            error=row["error"],
            runner_result=runner_result,
            evidence=tuple(evidence),
        )


class SQLiteRunHistoryRepository(_SQLiteStorage):
    """Persist safe run snapshots and references to canonical Executions."""

    def __init__(self, db_path: str | Path) -> None:
        super().__init__(db_path)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_history (
                    insertion_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    test_case_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_history_test_case_order "
                "ON run_history(test_case_id, insertion_order DESC)"
            )

    def save(self, record: RunHistoryRecord) -> None:
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO run_history (
                        run_id, test_case_id, started_at, record_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(record.run_id),
                        str(record.test_case_id),
                        record.started_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed: run_history.run_id" not in str(error):
                raise
            raise ValueError(f"Run {record.run_id} already exists in history.") from error

    def get(self, run_id: UUID) -> RunHistoryRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT record_json FROM run_history WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return RunHistoryRecord.model_validate_json(row["record_json"]) if row else None

    def list_recent(self, limit: int = 50) -> list[RunHistoryRecord]:
        _validate_run_history_limit(limit)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT record_json FROM run_history "
                "ORDER BY insertion_order DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [RunHistoryRecord.model_validate_json(row["record_json"]) for row in rows]

    def list_for_test_case(
        self, test_case_id: UUID, limit: int = 50
    ) -> list[RunHistoryRecord]:
        _validate_run_history_limit(limit)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT record_json FROM run_history WHERE test_case_id = ? "
                "ORDER BY insertion_order DESC LIMIT ?",
                (str(test_case_id), limit),
            ).fetchall()
        return [RunHistoryRecord.model_validate_json(row["record_json"]) for row in rows]


def _validate_run_history_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Run history limit must be a non-negative integer.")
