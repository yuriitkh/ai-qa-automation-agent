"""Additive SQLite persistence for durable Suite Runs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterator
from uuid import UUID

from qa_agent.suite_runs import (
    SuiteRun,
    SuiteRunItemStatus,
    SuiteRunRepository,
    SuiteRunStatus,
)


class SQLiteSuiteRunRepository(SuiteRunRepository):
    """Persist Suite Run state in new tables without changing existing rows."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        self._memory_connection: sqlite3.Connection | None = None
        self._lock = Lock()
        if self._database_path != ":memory:":
            Path(self._database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS suite_runs (
                    insertion_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_run_id TEXT NOT NULL UNIQUE,
                    public_id TEXT NOT NULL UNIQUE,
                    suite_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_suite_runs_recent
                    ON suite_runs(insertion_order DESC);
                CREATE TABLE IF NOT EXISTS suite_run_public_id_counters (
                    namespace TEXT PRIMARY KEY,
                    next_value INTEGER NOT NULL CHECK (next_value >= 1)
                );
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            with self._memory_connection:
                yield self._memory_connection
            return
        connection = sqlite3.connect(self._database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, run: SuiteRun) -> SuiteRun:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT public_id FROM suite_runs WHERE suite_run_id = ?",
                (str(run.id),),
            ).fetchone()
            if existing is None:
                public_id = run.public_id or self._allocate_public_id(connection)
                stored = run.model_copy(update={"public_id": public_id})
                connection.execute(
                    "INSERT INTO suite_runs(suite_run_id, public_id, suite_id, status, run_json) VALUES (?, ?, ?, ?, ?)",
                    (str(stored.id), public_id, str(stored.suite_id), stored.status.value, stored.model_dump_json()),
                )
            else:
                public_id = run.public_id or existing["public_id"]
                stored = run.model_copy(update={"public_id": public_id})
                connection.execute(
                    "UPDATE suite_runs SET public_id = ?, suite_id = ?, status = ?, run_json = ? WHERE suite_run_id = ?",
                    (public_id, str(stored.suite_id), stored.status.value, stored.model_dump_json(), str(stored.id)),
                )
        return stored

    @staticmethod
    def _allocate_public_id(connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT next_value FROM suite_run_public_id_counters WHERE namespace = 'suite_run'"
        ).fetchone()
        if row is None:
            sequence = int(connection.execute(
                "SELECT COUNT(*) + 1 FROM suite_runs"
            ).fetchone()[0])
            connection.execute(
                "INSERT INTO suite_run_public_id_counters(namespace, next_value) VALUES ('suite_run', ?)",
                (sequence + 1,),
            )
        else:
            sequence = int(row["next_value"])
            connection.execute(
                "UPDATE suite_run_public_id_counters SET next_value = ? WHERE namespace = 'suite_run'",
                (sequence + 1,),
            )
        return f"SUITE-RUN-{sequence:06d}"

    def get(self, run_id: UUID) -> SuiteRun | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT run_json FROM suite_runs WHERE suite_run_id = ?",
                (str(run_id),),
            ).fetchone()
        return SuiteRun.model_validate_json(row["run_json"]) if row else None

    def get_by_public_id(self, public_id: str) -> SuiteRun | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT run_json FROM suite_runs WHERE public_id = ?",
                (public_id,),
            ).fetchone()
        return SuiteRun.model_validate_json(row["run_json"]) if row else None

    def list_recent(self, limit: int = 50) -> list[SuiteRun]:
        safe_limit = min(max(int(limit), 0), 500)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT run_json FROM suite_runs ORDER BY insertion_order DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [SuiteRun.model_validate_json(row["run_json"]) for row in rows]

    def interrupt_incomplete(self) -> int:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT run_json FROM suite_runs WHERE status IN (?, ?, ?)",
                (SuiteRunStatus.QUEUED.value, SuiteRunStatus.RUNNING.value, SuiteRunStatus.CANCELLATION_REQUESTED.value),
            ).fetchall()
        changed = 0
        for row in rows:
            run = SuiteRun.model_validate_json(row["run_json"])
            if run.status == SuiteRunStatus.CANCELLATION_REQUESTED:
                from qa_agent.suite_runs import recover_cancelled_suite
                self.save(recover_cancelled_suite(run))
                changed += 1
                continue
            now = datetime.now(timezone.utc)
            items = []
            for item in run.items:
                if item.status == SuiteRunItemStatus.RUNNING:
                    items.append(item.model_copy(update={
                        "status": SuiteRunItemStatus.INTERRUPTED,
                        "finished_at": now,
                    }))
                elif item.status in {SuiteRunItemStatus.QUEUED, SuiteRunItemStatus.RETRYING}:
                    items.append(item.model_copy(update={"status": SuiteRunItemStatus.NOT_RUN}))
                else:
                    items.append(item)
            self.save(run.model_copy(update={
                "status": SuiteRunStatus.INTERRUPTED,
                "finished_at": now,
                "items": items,
            }))
            changed += 1
        return changed
