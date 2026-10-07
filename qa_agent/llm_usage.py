"""Safe LLM-attempt persistence, estimates, and local usage analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import logging
import re
import sqlite3
import unicodedata
from threading import RLock
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterator, Mapping
from uuid import UUID, uuid4

from qa_agent.llm.usage_metadata import ProviderTokenUsage
from qa_agent.public_ids import parse_test_case_public_id
from qa_agent.redaction import redact_secrets

logger = logging.getLogger(__name__)

OP_AUTHOR_TESTCASE = "AUTHOR_TESTCASE"
OP_GENERATE_AUTOMATION_PLAN = "GENERATE_AUTOMATION_PLAN"
OP_REPAIR_AUTOMATION_PLAN = "REPAIR_AUTOMATION_PLAN"
OP_DISCOVERY = "DISCOVERY"
OP_OTHER = "OTHER"
OPERATION_TYPES = {
    OP_AUTHOR_TESTCASE,
    OP_GENERATE_AUTOMATION_PLAN,
    OP_REPAIR_AUTOMATION_PLAN,
    OP_DISCOVERY,
    OP_OTHER,
}

REQUEST_SUCCESS = "SUCCESS"
REQUEST_FAILED = "FAILED"
REQUEST_STATUSES = {REQUEST_SUCCESS, REQUEST_FAILED}
ERROR_CATEGORIES = {
    "RATE_LIMIT",
    "TIMEOUT",
    "AUTH_ERROR",
    "MODEL_NOT_FOUND",
    "INVALID_REQUEST",
    "SCHEMA_ERROR",
    "INVALID_RESPONSE",
    "PROVIDER_UNAVAILABLE",
    "INVALID_OUTPUT",
    "OTHER_PROVIDER_ERROR",
    "OTHER",
}


@dataclass(frozen=True)
class LLMUsageContext:
    operation_type: str | None = None
    related_test_case_id: str | None = None
    related_test_case_public_id: str | None = None
    related_workflow_id: str | None = None


_CURRENT_USAGE_CONTEXT: ContextVar[LLMUsageContext | None] = ContextVar(
    "llm_usage_context", default=None
)


@contextmanager
def llm_usage_scope(
    *,
    operation_type: str | None = None,
    related_test_case_id: str | UUID | None = None,
    related_test_case_public_id: str | None = None,
    related_workflow_id: str | None = None,
) -> Iterator[LLMUsageContext]:
    """Set safe correlation metadata for LLM calls made inside this scope."""
    parent = _CURRENT_USAGE_CONTEXT.get() or LLMUsageContext()
    context = LLMUsageContext(
        operation_type=operation_type or parent.operation_type,
        related_test_case_id=(
            str(related_test_case_id)
            if related_test_case_id is not None
            else parent.related_test_case_id
        ),
        related_test_case_public_id=(
            related_test_case_public_id
            if related_test_case_public_id is not None
            else parent.related_test_case_public_id
        ),
        related_workflow_id=(
            related_workflow_id
            if related_workflow_id is not None
            else parent.related_workflow_id
        ),
    )
    token: Token[LLMUsageContext | None] = _CURRENT_USAGE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_USAGE_CONTEXT.reset(token)


def current_llm_usage_context(default_operation: str) -> LLMUsageContext:
    current = _CURRENT_USAGE_CONTEXT.get() or LLMUsageContext()
    operation = current.operation_type or default_operation
    return LLMUsageContext(
        operation_type=operation if operation in OPERATION_TYPES else OP_OTHER,
        related_test_case_id=current.related_test_case_id,
        related_test_case_public_id=current.related_test_case_public_id,
        related_workflow_id=current.related_workflow_id,
    )


@dataclass(frozen=True)
class LLMPricing:
    provider_id: str
    model: str
    input_cost_per_1m_tokens: Decimal
    output_cost_per_1m_tokens: Decimal
    source: str
    note: str = ""
    effective_date: date | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if self.input_cost_per_1m_tokens < 0 or self.output_cost_per_1m_tokens < 0:
            raise ValueError("Model token prices must be nonnegative.")


# Keep only rates with a verified source and a deliberate local update. No
# default rates are shipped because prices vary by model, provider, and date.
LLM_PRICING: dict[tuple[str, str], LLMPricing] = {}


class LLMPricingCatalog:
    def __init__(
        self,
        pricing: Mapping[tuple[str, str], LLMPricing] | None = None,
    ) -> None:
        rates = LLM_PRICING if pricing is None else pricing
        self._pricing = {
            (provider.casefold(), model.casefold()): rate
            for (provider, model), rate in rates.items()
        }

    def find(self, provider_id: str, model: str) -> LLMPricing | None:
        return self._pricing.get((provider_id.casefold(), model.casefold()))

    def estimate(
        self,
        provider_id: str,
        model: str,
        usage: ProviderTokenUsage | None,
    ) -> float | None:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            return None
        pricing = self.find(provider_id, model)
        if pricing is None or not pricing.verified or not pricing.source.strip():
            return None
        amount = (
            Decimal(usage.input_tokens) * pricing.input_cost_per_1m_tokens
            + Decimal(usage.output_tokens) * pricing.output_cost_per_1m_tokens
        ) / Decimal(1_000_000)
        return float(amount.quantize(Decimal("0.000000000001")))


@dataclass(frozen=True)
class LLMUsageRecord:
    id: str
    operation_id: str
    started_at: datetime
    finished_at: datetime
    provider_id: str
    provider_name: str
    model: str
    operation_type: str
    request_status: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    fallback_used: bool
    fallback_from_provider: str | None = None
    error_category: str | None = None
    estimated_cost_usd: float | None = None
    related_test_case_id: str | None = None
    related_test_case_public_id: str | None = None
    related_workflow_id: str | None = None


class LLMUsageRepository:
    """SQLite repository for metadata-only LLM provider attempts."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        self._memory_connection: sqlite3.Connection | None = None
        self._memory_lock = RLock()
        if self._database_path != ":memory:":
            Path(self._database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(
                self._database_path, check_same_thread=False
            )
            self._memory_connection.row_factory = sqlite3.Row
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    request_status TEXT NOT NULL CHECK(request_status IN ('SUCCESS','FAILED')),
                    input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
                    output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
                    total_tokens INTEGER CHECK(total_tokens IS NULL OR total_tokens >= 0),
                    latency_ms INTEGER NOT NULL CHECK(latency_ms >= 0),
                    fallback_used INTEGER NOT NULL CHECK(fallback_used IN (0,1)),
                    fallback_from_provider TEXT,
                    error_category TEXT,
                    estimated_cost_usd REAL CHECK(estimated_cost_usd IS NULL OR estimated_cost_usd >= 0),
                    related_test_case_id TEXT,
                    related_test_case_public_id TEXT,
                    related_workflow_id TEXT
                )
                """
            )
            for name, column in (
                ("idx_llm_usage_started_at", "started_at"),
                ("idx_llm_usage_provider", "provider_id"),
                ("idx_llm_usage_model", "model"),
                ("idx_llm_usage_operation", "operation_type"),
                ("idx_llm_usage_test_case", "related_test_case_id"),
                ("idx_llm_usage_operation_id", "operation_id"),
            ):
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS {name} ON llm_usage({column})"
                )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            with self._memory_lock:
                with self._memory_connection:
                    yield self._memory_connection
            return
        connection = sqlite3.connect(self._database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, record: LLMUsageRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO llm_usage (
                    id, operation_id, started_at, finished_at, provider_id,
                    provider_name, model, operation_type, request_status,
                    input_tokens, output_tokens, total_tokens, latency_ms,
                    fallback_used, fallback_from_provider, error_category,
                    estimated_cost_usd, related_test_case_id,
                    related_test_case_public_id, related_workflow_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.operation_id,
                    record.started_at.astimezone(timezone.utc).isoformat(),
                    record.finished_at.astimezone(timezone.utc).isoformat(),
                    record.provider_id,
                    record.provider_name,
                    record.model,
                    record.operation_type,
                    record.request_status,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.latency_ms,
                    int(record.fallback_used),
                    record.fallback_from_provider,
                    record.error_category,
                    record.estimated_cost_usd,
                    record.related_test_case_id,
                    record.related_test_case_public_id,
                    record.related_workflow_id,
                ),
            )

    def list_records(
        self,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        test_case_id: str | UUID | None = None,
    ) -> list[LLMUsageRecord]:
        where: list[str] = []
        values: list[object] = []
        if start_at is not None:
            where.append("started_at >= ?")
            values.append(start_at.astimezone(timezone.utc).isoformat())
        if end_at is not None:
            where.append("started_at < ?")
            values.append(end_at.astimezone(timezone.utc).isoformat())
        if test_case_id is not None:
            where.append("related_test_case_id = ?")
            values.append(str(test_case_id))
        query = "SELECT * FROM llm_usage"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY started_at, id"
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [_record_from_row(row) for row in rows]

    def associate_workflow(
        self,
        workflow_id: str,
        test_case_id: str | UUID,
        public_id: str | None,
    ) -> int:
        safe_public_id = (
            public_id if parse_test_case_public_id(public_id) is not None else None
        )
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE llm_usage SET related_test_case_id = ?, "
                "related_test_case_public_id = ? WHERE related_workflow_id = ?",
                (str(test_case_id), safe_public_id, workflow_id),
            )
            return cursor.rowcount


class LLMUsageService:
    """Best-effort writer and analytics facade around the usage repository."""

    def __init__(
        self,
        repository: LLMUsageRepository,
        pricing: LLMPricingCatalog | None = None,
    ) -> None:
        self.repository = repository
        self.pricing = pricing or LLMPricingCatalog()

    def record_attempt(
        self,
        *,
        operation_id: str,
        operation_type: str,
        provider_id: str,
        provider_name: str,
        model: str | None,
        started_at: datetime,
        finished_at: datetime,
        latency_ms: int,
        request_status: str,
        usage: ProviderTokenUsage | None,
        fallback_from_provider: str | None,
        error_category: str | None,
        related_test_case_id: str | None,
        related_test_case_public_id: str | None,
        related_workflow_id: str | None,
    ) -> None:
        safe_provider_id = _safe_identifier(provider_id, fallback="unknown")
        safe_provider_name = _safe_display(provider_name, fallback="Configured provider")
        safe_model = _safe_model(model)
        normalized_usage = usage if isinstance(usage, ProviderTokenUsage) else None
        category = error_category if error_category in ERROR_CATEGORIES else None
        status = request_status if request_status in REQUEST_STATUSES else REQUEST_FAILED
        operation = operation_type if operation_type in OPERATION_TYPES else OP_OTHER
        fallback_name = (
            _safe_display(fallback_from_provider, fallback="Configured provider")
            if fallback_from_provider else None
        )
        test_case = _safe_uuid(related_test_case_id)
        public_id = (
            related_test_case_public_id
            if parse_test_case_public_id(related_test_case_public_id) is not None
            else None
        )
        workflow_id = _safe_workflow_id(related_workflow_id)
        estimated = self.pricing.estimate(safe_provider_id, safe_model, normalized_usage)
        record = LLMUsageRecord(
            id=str(uuid4()),
            operation_id=str(uuid4()) if not _safe_operation_id(operation_id) else operation_id,
            started_at=started_at,
            finished_at=finished_at,
            provider_id=safe_provider_id,
            provider_name=safe_provider_name,
            model=safe_model,
            operation_type=operation,
            request_status=status,
            input_tokens=normalized_usage.input_tokens if normalized_usage else None,
            output_tokens=normalized_usage.output_tokens if normalized_usage else None,
            total_tokens=normalized_usage.total_tokens if normalized_usage else None,
            latency_ms=max(0, int(latency_ms)),
            fallback_used=fallback_from_provider is not None,
            fallback_from_provider=fallback_name,
            error_category=category,
            estimated_cost_usd=estimated,
            related_test_case_id=test_case,
            related_test_case_public_id=public_id,
            related_workflow_id=workflow_id,
        )
        self.repository.record(record)

    def associate_workflow(
        self,
        workflow_id: str | None,
        test_case_id: str | UUID,
        public_id: str | None,
    ) -> None:
        if not workflow_id:
            return
        try:
            self.repository.associate_workflow(workflow_id, test_case_id, public_id)
        except Exception as error:
            logger.warning(
                "LLM usage association failed (%s)", type(error).__name__
            )

    def analytics(
        self,
        window: str = "30d",
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        window = window if window in {"today", "7d", "30d", "all"} else "30d"
        current = (now or datetime.now().astimezone()).astimezone()
        start_at = _window_start(window, current)
        records = self.repository.list_records(start_at=start_at)
        return aggregate_usage(records, window=window)

    def test_case_summary(self, test_case_id: str | UUID) -> dict[str, object] | None:
        records = self.repository.list_records(test_case_id=test_case_id)
        if not records:
            return None
        aggregate = _aggregate_records(records)
        by_operation = _group(records, lambda item: item.operation_type)
        by_request = _group(records, lambda item: item.operation_id)
        return {
            **aggregate,
            "operations": len(by_request),
            "fallback_operations": sum(
                any(item.fallback_used for item in group)
                for group in by_request.values()
            ),
            "providers": sorted({item.provider_name for item in records}),
            "by_operation": [
                {"operation_type": key, **_aggregate_records(group)}
                for key, group in sorted(by_operation.items())
            ],
        }


def aggregate_usage(
    records: list[LLMUsageRecord],
    *,
    window: str = "30d",
) -> dict[str, object]:
    by_provider = _group(records, lambda item: item.provider_id)
    by_model = _group(records, lambda item: (item.provider_id, item.model))
    by_operation = _group(records, lambda item: item.operation_type)
    operations = _group(records, lambda item: item.operation_id)
    fallback_operations = sum(
        any(item.fallback_used for item in group) for group in operations.values()
    )
    return {
        "window": window,
        "requests": len(records),
        "operations": len(operations),
        "successful_requests": sum(item.request_status == REQUEST_SUCCESS for item in records),
        "failed_requests": sum(item.request_status == REQUEST_FAILED for item in records),
        "fallbacks": fallback_operations,
        "fallback_requests": sum(item.fallback_used for item in records),
        "fallback_rate": (
            fallback_operations / len(operations) if operations else None
        ),
        **_aggregate_records(records),
        "by_provider": [
            {
                "provider_id": key,
                "provider_name": group[0].provider_name,
                **_aggregate_records(group),
            }
            for key, group in sorted(by_provider.items())
        ],
        "by_model": [
            {
                "provider_id": key[0], "provider_name": group[0].provider_name,
                "model": key[1], **_aggregate_records(group),
            }
            for key, group in sorted(by_model.items())
        ],
        "by_operation": [
            {"operation_type": key, **_aggregate_records(group)}
            for key, group in sorted(by_operation.items())
        ],
    }


def _aggregate_records(records: list[LLMUsageRecord]) -> dict[str, object]:
    successes = sum(item.request_status == REQUEST_SUCCESS for item in records)
    count = len(records)
    usage_records = sum(item.total_tokens is not None for item in records)
    priced_records = sum(item.estimated_cost_usd is not None for item in records)
    costs = [item.estimated_cost_usd for item in records if item.estimated_cost_usd is not None]
    latencies = [item.latency_ms for item in records if item.latency_ms is not None]
    return {
        "requests": count,
        "successful_requests": successes,
        "failed_requests": count - successes,
        "success_rate": successes / count if count else None,
        "fallback_requests": sum(item.fallback_used for item in records),
        "input_tokens": _sum_known(item.input_tokens for item in records),
        "output_tokens": _sum_known(item.output_tokens for item in records),
        "total_tokens": _sum_known(item.total_tokens for item in records),
        "usage_known_requests": usage_records,
        "usage_is_partial": bool(records) and usage_records < count,
        "estimated_cost_usd": sum(costs) if costs else None,
        "cost_known_requests": priced_records,
        "cost_is_partial": bool(records) and priced_records < count,
        "average_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "limited_sample": count < 5,
    }


def _sum_known(values: Iterator[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _group(records, key_function):
    groups: dict[object, list[LLMUsageRecord]] = {}
    for record in records:
        groups.setdefault(key_function(record), []).append(record)
    return groups


def _window_start(window: str, now: datetime) -> datetime | None:
    if window == "today":
        midnight = datetime.combine(now.date(), time.min, tzinfo=now.tzinfo)
        return midnight.astimezone(timezone.utc)
    if window == "7d":
        return now - timedelta(days=7)
    if window == "30d":
        return now - timedelta(days=30)
    return None


def _record_from_row(row: sqlite3.Row) -> LLMUsageRecord:
    return LLMUsageRecord(
        id=row["id"],
        operation_id=row["operation_id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]),
        provider_id=row["provider_id"],
        provider_name=row["provider_name"],
        model=row["model"],
        operation_type=row["operation_type"],
        request_status=row["request_status"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        latency_ms=row["latency_ms"],
        fallback_used=bool(row["fallback_used"]),
        fallback_from_provider=row["fallback_from_provider"],
        error_category=row["error_category"],
        estimated_cost_usd=row["estimated_cost_usd"],
        related_test_case_id=row["related_test_case_id"],
        related_test_case_public_id=row["related_test_case_public_id"],
        related_workflow_id=row["related_workflow_id"],
    )


def _safe_identifier(value: str, *, fallback: str) -> str:
    clean = redact_secrets(str(value)).strip().casefold()
    if "[redacted]" in clean or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", clean):
        return fallback
    return clean


def _safe_display(value: str, *, fallback: str) -> str:
    clean = " ".join(redact_secrets(str(value)).split()).strip()
    if (
        "[redacted]" in clean or not clean or len(clean) > 80
        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in clean)
    ):
        return fallback
    return clean


def _safe_model(value: str | None) -> str:
    if not value:
        return "unknown"
    clean = " ".join(redact_secrets(str(value)).split()).strip()
    if (
        "[redacted]" in clean or not clean or len(clean) > 200
        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in clean)
    ):
        return "unknown"
    return clean


def _safe_uuid(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def _safe_operation_id(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _safe_workflow_id(value: str | None) -> str | None:
    if not value:
        return None
    clean = redact_secrets(str(value)).strip()
    if "[redacted]" in clean or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", clean):
        return None
    return clean
