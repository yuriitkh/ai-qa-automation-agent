"""Shared SQLite repository wiring for the CLI and local Web UI."""

import os
from dataclasses import dataclass
from pathlib import Path

from qa_agent.execution_repository import ExecutionRepository
from qa_agent.plan_store import PlanStore
from qa_agent.run_history import RunHistoryRepository, RunHistoryService
from qa_agent.sqlite_storage import (
    SQLiteExecutionRepository,
    SQLitePlanStore,
    SQLiteRunHistoryRepository,
)


@dataclass(frozen=True)
class SQLiteApplicationStorage:
    database_path: Path
    plan_store: PlanStore
    execution_repository: ExecutionRepository
    run_history_repository: RunHistoryRepository
    run_history: RunHistoryService


def default_database_path() -> Path:
    configured = os.environ.get("QA_AGENT_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".qa_agent" / "qa_agent.sqlite3"


def create_sqlite_storage(
    database_path: str | Path | None = None,
) -> SQLiteApplicationStorage:
    path = Path(database_path).expanduser() if database_path is not None else default_database_path()
    plan_store = SQLitePlanStore(path)
    execution_repository = SQLiteExecutionRepository(path)
    run_history_repository = SQLiteRunHistoryRepository(path)
    run_history = RunHistoryService(run_history_repository, execution_repository)
    return SQLiteApplicationStorage(
        database_path=path,
        plan_store=plan_store,
        execution_repository=execution_repository,
        run_history_repository=run_history_repository,
        run_history=run_history,
    )
