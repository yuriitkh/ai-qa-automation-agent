"""File-backed SQLite implementations of plan and execution storage contracts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

from qa_agent.execution_repository import ExecutionRepository
from qa_agent.automation_lifecycle import (
    AutomationStatus,
    definition_fingerprint,
)
from qa_agent.models import (
    Execution,
    ExecutionStatus,
    Evidence,
    EvidenceType,
    QATestPlan,
    PlanVersionOrigin,
    TestCase,
    TestPlan,
    TestPlanVersion,
)
from qa_agent.plan_store import PlanStore, _validate_plan_store_write
from qa_agent.run_history import RunHistoryRecord
from qa_agent.public_ids import (
    format_run_public_id,
    format_test_case_public_id,
    parse_run_public_id,
    parse_test_case_public_id,
)
from qa_agent.test_case_repository import TestCaseCatalogEntry


def _format_public_id(prefix: str, sequence: int) -> str:
    return (
        format_test_case_public_id(sequence)
        if prefix == "TC-"
        else format_run_public_id(sequence)
    )


def _parse_public_id(prefix: str, value: str | None) -> int | None:
    return (
        parse_test_case_public_id(value)
        if prefix == "TC-"
        else parse_run_public_id(value)
    )


def _migrate_public_ids(
    connection: sqlite3.Connection,
    *,
    namespace: str,
    table: str,
    key_column: str,
    json_column: str,
    order_by: str,
    prefix: str,
) -> None:
    """Add stable IDs to legacy rows and keep the JSON snapshot in sync."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS public_id_counters ("
        "namespace TEXT PRIMARY KEY, next_value INTEGER NOT NULL CHECK (next_value >= 1))"
    )
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if "public_id" not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN public_id TEXT")

    rows = connection.execute(
        f"SELECT {key_column}, public_id, {json_column} FROM {table} ORDER BY {order_by}"
    ).fetchall()
    parsed_rows: list[tuple[str, str | None, dict, str | None]] = []
    reserved: dict[str, int] = {}
    for row in rows:
        key = row[key_column]
        stored_id = row["public_id"]
        payload = json.loads(row[json_column])
        if not isinstance(payload, dict):
            raise ValueError(f"Stored {table} data must be a JSON object.")
        candidate_ids = (stored_id, payload.get("public_id"))
        chosen = None
        for candidate in candidate_ids:
            sequence = _parse_public_id(prefix, candidate)
            if sequence is not None and candidate not in reserved:
                chosen = candidate
                reserved[candidate] = sequence
                break
        parsed_rows.append((key, stored_id, payload, chosen))

    next_sequence = max(reserved.values(), default=0) + 1
    max_sequence = max(reserved.values(), default=0)
    for key, stored_id, payload, chosen in parsed_rows:
        if chosen is None:
            while True:
                candidate = _format_public_id(prefix, next_sequence)
                next_sequence += 1
                if candidate not in reserved:
                    chosen = candidate
                    reserved[candidate] = next_sequence - 1
                    max_sequence = max(max_sequence, next_sequence - 1)
                    break
        if stored_id != chosen or payload.get("public_id") != chosen:
            payload["public_id"] = chosen
            connection.execute(
                f"UPDATE {table} SET public_id = ?, {json_column} = ? WHERE {key_column} = ?",
                (chosen, json.dumps(payload, ensure_ascii=False), key),
            )
        else:
            max_sequence = max(max_sequence, reserved[chosen])

    counter = connection.execute(
        "SELECT next_value FROM public_id_counters WHERE namespace = ?",
        (namespace,),
    ).fetchone()
    next_value = max_sequence + 1
    if counter is not None:
        next_value = max(next_value, int(counter["next_value"]))
    connection.execute(
        "INSERT INTO public_id_counters(namespace, next_value) VALUES (?, ?) "
        "ON CONFLICT(namespace) DO UPDATE SET next_value = MAX(next_value, excluded.next_value)",
        (namespace, next_value),
    )
    index_name = f"idx_{table}_public_id"
    connection.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table}(public_id)"
    )


def _allocate_public_id(
    connection: sqlite3.Connection,
    *,
    namespace: str,
    table: str,
    prefix: str,
) -> str:
    counter = connection.execute(
        "SELECT next_value FROM public_id_counters WHERE namespace = ?",
        (namespace,),
    ).fetchone()
    existing = connection.execute(f"SELECT public_id FROM {table}").fetchall()
    maximum = max(
        (_parse_public_id(prefix, row["public_id"]) or 0 for row in existing),
        default=0,
    )
    sequence = max(maximum + 1, int(counter["next_value"]) if counter else 1)
    connection.execute(
        "INSERT INTO public_id_counters(namespace, next_value) VALUES (?, ?) "
        "ON CONFLICT(namespace) DO UPDATE SET next_value = excluded.next_value",
        (namespace, sequence + 1),
    )
    return _format_public_id(prefix, sequence)


def _advance_public_id_counter(
    connection: sqlite3.Connection,
    *,
    namespace: str,
    sequence: int,
) -> None:
    connection.execute(
        "INSERT INTO public_id_counters(namespace, next_value) VALUES (?, ?) "
        "ON CONFLICT(namespace) DO UPDATE SET next_value = MAX(next_value, excluded.next_value)",
        (namespace, sequence + 1),
    )


def _grounding_json(plan_version: TestPlanVersion) -> str | None:
    if plan_version.assertion_grounding is None:
        return None
    return json.dumps(
        [entry.model_dump(mode="json") for entry in plan_version.assertion_grounding],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _locator_identity_json(plan_version: TestPlanVersion) -> str | None:
    if plan_version.locator_identity is None:
        return None
    return json.dumps(
        [entry.model_dump(mode="json") for entry in plan_version.locator_identity],
        ensure_ascii=False,
        separators=(",", ":"),
    )


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
                    origin TEXT,
                    qa_test_plan_json TEXT NOT NULL,
                    assertion_grounding_json TEXT,
                    locator_identity_json TEXT,
                    UNIQUE (test_plan_id, version_number)
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(test_plan_versions)").fetchall()
            }
            if "origin" not in columns:
                connection.execute("ALTER TABLE test_plan_versions ADD COLUMN origin TEXT")
            if "assertion_grounding_json" not in columns:
                connection.execute(
                    "ALTER TABLE test_plan_versions ADD COLUMN assertion_grounding_json TEXT"
                )
            if "locator_identity_json" not in columns:
                connection.execute(
                    "ALTER TABLE test_plan_versions ADD COLUMN locator_identity_json TEXT"
                )
            # Migrate the one current version from databases created by the
            # previous schema. Earlier versions cannot be reconstructed.
            connection.execute(
                """
                INSERT OR IGNORE INTO test_plan_versions (
                    version_id, test_step_id, test_plan_id, version_number,
                    created_at, origin, qa_test_plan_json
                )
                SELECT version_id, test_step_id, test_plan_id, version_number,
                       created_at, NULL, qa_test_plan_json
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
                    plan_version.created_at.isoformat(),
                    plan_version.origin.value if plan_version.origin is not None else None,
                    plan_version.qa_test_plan.model_dump_json(),
                    _grounding_json(plan_version),
                    _locator_identity_json(plan_version),
                )
                actual = (
                    by_id["test_step_id"], by_id["test_plan_id"],
                    int(by_id["version_number"]), by_id["created_at"],
                    by_id["origin"],
                    by_id["qa_test_plan_json"],
                    by_id["assertion_grounding_json"],
                    by_id["locator_identity_json"],
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
                        created_at, origin, qa_test_plan_json, assertion_grounding_json,
                        locator_identity_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(plan_version.id), str(test_step_id), str(test_plan.id),
                        plan_version.version, plan_version.created_at.isoformat(),
                        plan_version.origin.value if plan_version.origin is not None else None,
                        plan_version.qa_test_plan.model_dump_json(),
                        _grounding_json(plan_version),
                        _locator_identity_json(plan_version),
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
                "SELECT version_id FROM cached_test_plans WHERE test_step_id = ?",
                (str(test_step_id),),
            ).fetchone()
        if row is None:
            return None
        return self.get_version(UUID(row["version_id"]))

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

    def list_versions(self, test_step_id: UUID) -> tuple[TestPlanVersion, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM test_plan_versions WHERE test_step_id = ? "
                "ORDER BY version_number DESC, created_at DESC",
                (str(test_step_id),),
            ).fetchall()
        return tuple(self._to_version(row) for row in rows)

    @staticmethod
    def _to_version(row: sqlite3.Row) -> TestPlanVersion:
        return TestPlanVersion(
            id=UUID(row["version_id"]),
            test_plan_id=UUID(row["test_plan_id"]),
            version=int(row["version_number"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            origin=(PlanVersionOrigin(row["origin"]) if row["origin"] else None),
            qa_test_plan=QATestPlan.model_validate_json(row["qa_test_plan_json"]),
            assertion_grounding=(
                json.loads(row["assertion_grounding_json"])
                if row["assertion_grounding_json"] else None
            ),
            locator_identity=(
                json.loads(row["locator_identity_json"])
                if row["locator_identity_json"] else None
            ),
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
                    public_id TEXT,
                    name TEXT NOT NULL,
                    definition_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS test_case_automation_lifecycle (
                    test_case_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    definition_fingerprint TEXT NOT NULL,
                    plan_fingerprint TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            _migrate_public_ids(
                connection,
                namespace="test_case",
                table="test_cases",
                key_column="test_case_id",
                json_column="definition_json",
                order_by="created_at, test_case_id",
                prefix="TC-",
            )

    def save(self, test_case: TestCase) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT public_id, definition_json FROM test_cases WHERE test_case_id = ?",
                (str(test_case.id),),
            ).fetchone()
            public_id = existing["public_id"] if existing is not None else test_case.public_id
            sequence = parse_test_case_public_id(public_id)
            if public_id is None:
                public_id = _allocate_public_id(
                    connection,
                    namespace="test_case",
                    table="test_cases",
                    prefix="TC-",
                )
                sequence = parse_test_case_public_id(public_id)
            elif sequence is None:
                raise ValueError("TestCase public ID must use the TC-0001 format.")
            duplicate = connection.execute(
                "SELECT test_case_id FROM test_cases WHERE public_id = ? AND test_case_id != ?",
                (public_id, str(test_case.id)),
            ).fetchone()
            if duplicate is not None:
                raise ValueError(f"TestCase public ID {public_id} is already in use.")
            _advance_public_id_counter(
                connection, namespace="test_case", sequence=sequence or 0
            )
            stored_case = test_case.model_copy(update={"public_id": public_id})
            stored_json = stored_case.model_dump_json(exclude={"steps"})
            if existing is not None:
                old_case = TestCase.model_validate_json(existing["definition_json"])
                old_json = old_case.model_dump_json(exclude={"steps"})
                if old_json != stored_json:
                    lifecycle = connection.execute(
                        "SELECT state FROM test_case_automation_lifecycle WHERE test_case_id=?",
                        (str(test_case.id),),
                    ).fetchone()
                    old_steps = [str(step.id) for step in old_case.steps]
                    plan_table = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cached_test_plans'"
                    ).fetchone()
                    had_saved_plans = False
                    if plan_table is not None and old_steps:
                        placeholders = ",".join("?" for _ in old_steps)
                        had_saved_plans = connection.execute(
                            f"SELECT 1 FROM cached_test_plans WHERE test_step_id IN ({placeholders}) LIMIT 1",
                            old_steps,
                        ).fetchone() is not None
                    if lifecycle is not None and lifecycle["state"] != AutomationStatus.NOT_AUTOMATED.value:
                        connection.execute(
                            "UPDATE test_case_automation_lifecycle SET state=?, updated_at=? WHERE test_case_id=?",
                            (AutomationStatus.NEEDS_UPDATE.value, now, str(test_case.id)),
                        )
                    elif lifecycle is None and had_saved_plans:
                        # Older databases may contain plans without lifecycle metadata.
                        # Record the old definition hash so this edit is visibly stale.
                        connection.execute(
                            """INSERT INTO test_case_automation_lifecycle
                                (test_case_id, state, definition_fingerprint, plan_fingerprint, updated_at)
                                VALUES (?, ?, ?, '', ?)""",
                            (str(test_case.id), AutomationStatus.NEEDS_UPDATE.value,
                             definition_fingerprint(old_case), now),
                        )
            connection.execute(
                """
                INSERT INTO test_cases (
                    test_case_id, public_id, name, definition_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(test_case_id) DO UPDATE SET
                    name = excluded.name,
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(test_case.id),
                    public_id,
                    test_case.name,
                    stored_json,
                    now,
                    now,
                ),
            )
            test_case.public_id = public_id

    def get(self, test_case_id: UUID) -> TestCase | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT definition_json FROM test_cases WHERE test_case_id = ?",
                (str(test_case_id),),
            ).fetchone()
        return TestCase.model_validate_json(row["definition_json"]) if row else None

    def get_by_public_id(self, public_id: str) -> TestCase | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT definition_json FROM test_cases WHERE public_id = ?",
                (public_id,),
            ).fetchone()
        return TestCase.model_validate_json(row["definition_json"]) if row else None

    def list(self) -> list[TestCase]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT definition_json FROM test_cases "
                "ORDER BY name COLLATE NOCASE, test_case_id"
            ).fetchall()
        return [TestCase.model_validate_json(row["definition_json"]) for row in rows]

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
        if test_case_ids is not None and not test_case_ids:
            return [], 0
        conditions = []
        parameters: list[object] = []
        if search.strip():
            conditions.append(
                "(instr(lower(name), lower(?)) > 0 "
                "OR instr(upper(COALESCE(public_id, '')), upper(?)) > 0)"
            )
            parameters.extend([search.strip(), search.strip()])
        for column, lower, upper in (
            ("created_at", created_from, created_before),
            ("updated_at", updated_from, updated_before),
        ):
            if lower:
                conditions.append(f"{column} >= ?")
                parameters.append(lower)
            if upper:
                conditions.append(f"{column} < ?")
                parameters.append(upper)
        if test_case_ids is not None:
            conditions.append("test_case_id IN (" + ",".join("?" for _ in test_case_ids) + ")")
            parameters.extend(str(item) for item in test_case_ids)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        safe_limit = min(max(int(limit), 1), 500)
        safe_offset = max(int(offset), 0)
        with self._connection() as connection:
            total = int(connection.execute(
                "SELECT COUNT(*) FROM test_cases" + where, parameters
            ).fetchone()[0])
            rows = connection.execute(
                "SELECT definition_json, created_at, updated_at FROM test_cases"
                + where
                + " ORDER BY name COLLATE NOCASE, test_case_id LIMIT ? OFFSET ?",
                [*parameters, safe_limit, safe_offset],
            ).fetchall()
        entries = [
            TestCaseCatalogEntry(
                TestCase.model_validate_json(row["definition_json"]),
                datetime.fromisoformat(row["created_at"]),
                datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]
        return entries, total

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
                    record_json TEXT NOT NULL,
                    public_id TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_history_test_case_order "
                "ON run_history(test_case_id, insertion_order DESC)"
            )
            _migrate_public_ids(
                connection,
                namespace="run",
                table="run_history",
                key_column="run_id",
                json_column="record_json",
                order_by="insertion_order, run_id",
                prefix="RUN-",
            )
            testcase_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(test_cases)").fetchall()
            }
            if "public_id" in testcase_columns:
                rows = connection.execute(
                    "SELECT run_id, test_case_id, record_json FROM run_history"
                ).fetchall()
                for row in rows:
                    payload = json.loads(row["record_json"])
                    if payload.get("test_case_public_id") is not None:
                        continue
                    case_row = connection.execute(
                        "SELECT public_id FROM test_cases WHERE test_case_id = ?",
                        (row["test_case_id"],),
                    ).fetchone()
                    if case_row is not None and case_row["public_id"]:
                        payload["test_case_public_id"] = case_row["public_id"]
                        connection.execute(
                            "UPDATE run_history SET record_json = ? WHERE run_id = ?",
                            (json.dumps(payload, ensure_ascii=False), row["run_id"]),
                        )

    def save(self, record: RunHistoryRecord) -> RunHistoryRecord:
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                public_id = record.public_id
                sequence = parse_run_public_id(public_id)
                if public_id is None:
                    public_id = _allocate_public_id(
                        connection,
                        namespace="run",
                        table="run_history",
                        prefix="RUN-",
                    )
                    sequence = parse_run_public_id(public_id)
                elif sequence is None:
                    raise ValueError("Run public ID must use the RUN-000001 format.")
                duplicate = connection.execute(
                    "SELECT run_id FROM run_history WHERE public_id = ?",
                    (public_id,),
                ).fetchone()
                if duplicate is not None:
                    raise ValueError(f"Run public ID {public_id} already exists.")
                _advance_public_id_counter(
                    connection, namespace="run", sequence=sequence or 0
                )
                test_case_public_id = record.test_case_public_id
                testcase_table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'test_cases'"
                ).fetchone()
                testcase_columns = (
                    {
                        row["name"]
                        for row in connection.execute("PRAGMA table_info(test_cases)").fetchall()
                    }
                    if testcase_table is not None else set()
                )
                if test_case_public_id is None and "public_id" in testcase_columns:
                    case_row = connection.execute(
                        "SELECT public_id FROM test_cases WHERE test_case_id = ?",
                        (str(record.test_case_id),),
                    ).fetchone()
                    if case_row is not None:
                        test_case_public_id = case_row["public_id"]
                stored_record = record.model_copy(update={
                    "public_id": public_id,
                    "test_case_public_id": test_case_public_id,
                })
                connection.execute(
                    """
                    INSERT INTO run_history (
                        run_id, test_case_id, started_at, record_json, public_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.run_id),
                        str(record.test_case_id),
                        record.started_at.isoformat(),
                        stored_record.model_dump_json(),
                        public_id,
                    ),
                )
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed: run_history.run_id" not in str(error):
                raise
            raise ValueError(f"Run {record.run_id} already exists in history.") from error
        return stored_record

    def get(self, run_id: UUID) -> RunHistoryRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT record_json FROM run_history WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return RunHistoryRecord.model_validate_json(row["record_json"]) if row else None

    def get_by_public_id(self, public_id: str) -> RunHistoryRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT record_json FROM run_history WHERE public_id = ?",
                (public_id,),
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
