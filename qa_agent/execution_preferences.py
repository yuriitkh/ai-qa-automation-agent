"""Versioned per-TestCase preferences; never part of approved plan identity."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from threading import RLock
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.cookie_consent import CookieConsentPolicy, DEFAULT_COOKIE_CONSENT_POLICY
from qa_agent.evidence_policy import EvidenceMode, EvidencePolicy, ScreenshotMode


class ExecutionPreferences(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    revision: int = Field(default=0, ge=0)
    cookie_policy: CookieConsentPolicy = DEFAULT_COOKIE_CONSENT_POLICY
    evidence_mode: EvidenceMode = EvidenceMode.FAILURES_ONLY
    screenshot_mode: ScreenshotMode = ScreenshotMode.PAGE

    @property
    def evidence_policy(self):
        return EvidencePolicy(mode=self.evidence_mode, screenshot_mode=self.screenshot_mode)


class PreferencesConflict(ValueError):
    def __init__(self, current):
        super().__init__("Preferences changed elsewhere. Review the latest values and retry.")
        self.current = current


def updated_preferences(current, field, value, revision):
    if field not in {"cookie_policy", "evidence_mode", "screenshot_mode"}:
        raise ValueError("Choose one supported execution preference.")
    # Validate even stale requests rather than storing arbitrary fields.
    candidate = ExecutionPreferences.model_validate({**current.model_dump(), field: value})
    if type(revision) is not int or revision < 0:
        raise ValueError("A valid preferences revision is required.")
    if revision != current.revision:
        raise PreferencesConflict(current)
    return candidate.model_copy(update={"revision": current.revision + 1})


class InMemoryExecutionPreferences:
    def __init__(self):
        self._values = {}
        self._lock = RLock()

    def get(self, test_case_id: UUID):
        with self._lock:
            return self._values.get(test_case_id, ExecutionPreferences())

    def update(self, test_case_id, field, value, revision):
        with self._lock:
            result = updated_preferences(self.get(test_case_id), field, value, revision)
            self._values[test_case_id] = result
            return result


class SQLiteExecutionPreferences:
    def __init__(self, database_path):
        self.database_path = str(database_path)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS test_case_execution_preferences (
                test_case_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                cookie_policy TEXT NOT NULL, evidence_mode TEXT NOT NULL,
                screenshot_mode TEXT NOT NULL)""")

    @staticmethod
    def _get(connection, test_case_id):
        row = connection.execute(
            "SELECT revision, cookie_policy, evidence_mode, screenshot_mode FROM test_case_execution_preferences WHERE test_case_id = ?",
            (str(test_case_id),),
        ).fetchone()
        return ExecutionPreferences.model_validate(dict(row)) if row else ExecutionPreferences()

    def get(self, test_case_id):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            return self._get(connection, test_case_id)

    def update(self, test_case_id, field, value, revision):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            result = updated_preferences(self._get(connection, test_case_id), field, value, revision)
            connection.execute("""INSERT INTO test_case_execution_preferences
                (test_case_id, revision, cookie_policy, evidence_mode, screenshot_mode)
                VALUES (?, ?, ?, ?, ?) ON CONFLICT(test_case_id) DO UPDATE SET
                revision=excluded.revision, cookie_policy=excluded.cookie_policy,
                evidence_mode=excluded.evidence_mode, screenshot_mode=excluded.screenshot_mode""",
                (str(test_case_id), result.revision, result.cookie_policy.value,
                 result.evidence_mode.value, result.screenshot_mode.value),
            )
            return result
