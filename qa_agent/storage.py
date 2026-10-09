"""Shared SQLite repository wiring for the CLI and local Web UI."""

import os
from dataclasses import dataclass
from pathlib import Path

from qa_agent.execution_repository import ExecutionRepository
from qa_agent.automation_lifecycle import (
    AutomationLifecycleRepository,
    SQLiteAutomationLifecycleRepository,
)
from qa_agent.drafts import DraftRepository, SQLiteDraftRepository
from qa_agent.plan_store import PlanStore
from qa_agent.run_history import RunHistoryRepository, RunHistoryService
from qa_agent.sqlite_storage import (
    SQLiteExecutionRepository,
    SQLitePlanStore,
    SQLiteRunHistoryRepository,
    SQLiteTestCaseRepository,
)
from qa_agent.test_case_repository import TestCaseRepository
from qa_agent.llm_usage import LLMUsageRepository
from qa_agent.reliability import SQLiteReliabilityRepository
from qa_agent.test_case_review import (
    SQLiteTestCaseReviewRepository,
    TestCaseReviewRepository,
)


@dataclass(frozen=True)
class SQLiteApplicationStorage:
    database_path: Path
    plan_store: PlanStore
    execution_repository: ExecutionRepository
    run_history_repository: RunHistoryRepository
    run_history: RunHistoryService
    test_case_repository: TestCaseRepository
    llm_usage_repository: LLMUsageRepository
    draft_repository: DraftRepository
    automation_lifecycle_repository: AutomationLifecycleRepository
    test_case_review_repository: TestCaseReviewRepository
    reliability_repository: SQLiteReliabilityRepository


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
    test_case_repository = SQLiteTestCaseRepository(path)
    draft_repository = SQLiteDraftRepository(path)
    automation_lifecycle_repository = SQLiteAutomationLifecycleRepository(path)
    test_case_review_repository = SQLiteTestCaseReviewRepository(path)
    run_history_repository = SQLiteRunHistoryRepository(path)
    llm_usage_repository = LLMUsageRepository(path)
    reliability_repository = SQLiteReliabilityRepository(path)
    run_history = RunHistoryService(
        run_history_repository, execution_repository, plan_store=plan_store
    )
    return SQLiteApplicationStorage(
        database_path=path,
        plan_store=plan_store,
        execution_repository=execution_repository,
        run_history_repository=run_history_repository,
        run_history=run_history,
        test_case_repository=test_case_repository,
        llm_usage_repository=llm_usage_repository,
        draft_repository=draft_repository,
        automation_lifecycle_repository=automation_lifecycle_repository,
        test_case_review_repository=test_case_review_repository,
        reliability_repository=reliability_repository,
    )
