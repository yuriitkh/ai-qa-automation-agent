"""Persistent, lightweight user Drafts kept separate from TestCases."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Draft(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=6000)
    base_url: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=6000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DraftRepository(Protocol):
    def save(self, draft: Draft) -> None: ...
    def get(self, draft_id: UUID) -> Draft | None: ...
    def list(self, limit: int = 100) -> list[Draft]: ...
    def delete(self, draft_id: UUID) -> bool: ...


class SQLiteDraftRepository:
    """SQLite persistence with an additive, idempotent table migration."""

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
                """CREATE TABLE IF NOT EXISTS drafts (
                    draft_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    base_url TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
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

    def save(self, draft: Draft) -> None:
        updated = draft.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO drafts
                    (draft_id, title, body, base_url, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(draft_id) DO UPDATE SET
                        title=excluded.title, body=excluded.body,
                        base_url=excluded.base_url, notes=excluded.notes,
                        updated_at=excluded.updated_at""",
                (str(updated.id), updated.title, updated.body, updated.base_url,
                 updated.notes, updated.created_at.isoformat(), updated.updated_at.isoformat()),
            )

    def get(self, draft_id: UUID) -> Draft | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM drafts WHERE draft_id = ?", (str(draft_id),)
            ).fetchone()
        return _draft_from_row(row) if row is not None else None

    def list(self, limit: int = 100) -> list[Draft]:
        if limit < 1:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM drafts ORDER BY updated_at DESC, draft_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [_draft_from_row(row) for row in rows]

    def delete(self, draft_id: UUID) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM drafts WHERE draft_id = ?", (str(draft_id),)
            )
            return cursor.rowcount > 0


class InMemoryDraftRepository:
    def __init__(self) -> None:
        self._drafts: dict[UUID, Draft] = {}

    def save(self, draft: Draft) -> None:
        updated = draft.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        self._drafts[draft.id] = updated

    def get(self, draft_id: UUID) -> Draft | None:
        return self._drafts.get(draft_id)

    def list(self, limit: int = 100) -> list[Draft]:
        return sorted(self._drafts.values(), key=lambda item: item.updated_at, reverse=True)[:limit]

    def delete(self, draft_id: UUID) -> bool:
        return self._drafts.pop(draft_id, None) is not None


def _draft_from_row(row: sqlite3.Row) -> Draft:
    return Draft(
        id=UUID(row["draft_id"]),
        title=row["title"],
        body=row["body"],
        base_url=row["base_url"],
        notes=row["notes"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
