"""Safe, machine-readable provider failure metadata."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping


PROVIDER_FAILURE_CATEGORIES = frozenset({
    "AUTH_ERROR",
    "MODEL_NOT_FOUND",
    "INVALID_REQUEST",
    "RATE_LIMIT",
    "TIMEOUT",
    "SCHEMA_ERROR",
    "INVALID_RESPONSE",
    "PROVIDER_UNAVAILABLE",
    "OTHER_PROVIDER_ERROR",
})

SAFE_FAILURE_DETAILS = {
    "AUTH_ERROR": "Authentication failed",
    "MODEL_NOT_FOUND": "Model unavailable",
    "INVALID_REQUEST": "Invalid request",
    "RATE_LIMIT": "Rate limited",
    "TIMEOUT": "Request timed out",
    "SCHEMA_ERROR": "Structured output rejected",
    "INVALID_RESPONSE": "Invalid structured response",
    "PROVIDER_UNAVAILABLE": "Provider unavailable",
    "OTHER_PROVIDER_ERROR": "Provider request failed",
}
SAFE_DETAIL_MESSAGES = frozenset({
    *SAFE_FAILURE_DETAILS.values(),
    "Response did not match the TestCase schema",
    "Response did not contain a complete TestCase structure",
    "Structured output is not supported",
})


@dataclass(frozen=True)
class ProviderFailureDetail:
    """Allowlisted provider diagnostics; never contains provider response text."""

    provider_name: str
    category: str
    http_status: int | None = None
    safe_detail: str | None = None
    retry_after_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.category not in PROVIDER_FAILURE_CATEGORIES:
            object.__setattr__(self, "category", "OTHER_PROVIDER_ERROR")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            object.__setattr__(self, "http_status", None)
        if self.safe_detail not in SAFE_DETAIL_MESSAGES:
            object.__setattr__(self, "safe_detail", SAFE_FAILURE_DETAILS[self.category])
        if self.retry_after_seconds is not None and self.retry_after_seconds <= 0:
            object.__setattr__(self, "retry_after_seconds", None)

    @property
    def message(self) -> str:
        return self.safe_detail or SAFE_FAILURE_DETAILS[self.category]

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize only safe, typed diagnostic values."""
        return {
            "provider": self.provider_name,
            "category": self.category,
            "http_status": self.http_status,
            "message": self.message,
            "retry_after_seconds": self.retry_after_seconds,
        }


class RetryableLLMError(Exception):
    """A provider failure that allows trying the next configured provider."""

    def __init__(
        self,
        message: str,
        *,
        category: str | None = None,
        http_status: int | None = None,
        safe_detail: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = (
            category if category in PROVIDER_FAILURE_CATEGORIES else None
        )
        self.http_status = http_status
        self.safe_detail = safe_detail
        self.retry_after_seconds = retry_after_seconds


class NonRetryableLLMError(Exception):
    """A provider failure that should stop routing immediately."""


class AllProvidersFailedError(RuntimeError):
    """A provider chain failed, retaining only safe per-provider diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        all_rate_limited: bool,
        all_timed_out: bool = False,
        attempts: tuple[ProviderFailureDetail, ...] = (),
    ) -> None:
        super().__init__(message)
        self.all_rate_limited = all_rate_limited
        self.all_timed_out = all_timed_out
        self.attempts = attempts


def provider_http_failure(
    provider_name: str,
    status: int,
    *,
    payload: Any = None,
    headers: Mapping[str, Any] | None = None,
) -> RetryableLLMError:
    """Build a safe retryable failure from status and allowlisted error hints.

    Provider-supplied text is inspected only to choose a category and is never
    returned, logged, or persisted. The displayed detail always comes from the
    local allowlist above.
    """
    searchable = _safe_classification_text(payload)
    if status in {401, 403}:
        category = "AUTH_ERROR"
    elif status == 408:
        category = "TIMEOUT"
    elif status == 429:
        category = "RATE_LIMIT"
    elif status == 404 and (
        "model_not_found" in searchable
        or ("model" in searchable and any(word in searchable for word in (
            "not found", "does not exist", "unknown", "unavailable",
        )))
    ):
        category = "MODEL_NOT_FOUND"
    elif status in {400, 422} and any(word in searchable for word in (
        "json_schema", "json schema", "response_format", "structured output",
        "schema is invalid", "schema validation", "schema not supported",
    )):
        category = "SCHEMA_ERROR"
    elif status in {400, 404, 422}:
        category = "INVALID_REQUEST"
    elif status >= 500:
        category = "PROVIDER_UNAVAILABLE"
    else:
        category = "OTHER_PROVIDER_ERROR"

    retry_after = parse_retry_after(headers)
    status_note = f" HTTP {status}." if status else "."
    return RetryableLLMError(
        f"{provider_name} request failed.{status_note}",
        category=category,
        http_status=status,
        safe_detail=SAFE_FAILURE_DETAILS[category],
        retry_after_seconds=retry_after if category == "RATE_LIMIT" else None,
    )


def parse_retry_after(headers: Mapping[str, Any] | None) -> int | None:
    """Read numeric Retry-After metadata without interpreting provider text."""
    if not headers:
        return None
    raw = next(
        (value for key, value in headers.items() if str(key).casefold() == "retry-after"),
        None,
    )
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (
                retry_at.astimezone(timezone.utc) - datetime.now(timezone.utc)
            ).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 7 * 24 * 60 * 60:
        return None
    return max(1, math.ceil(seconds))


def _safe_classification_text(payload: Any) -> str:
    """Extract only fields used for category matching; discard them afterward."""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error", payload)
    if not isinstance(error, dict):
        return ""
    parts = []
    for key in ("code", "type", "message"):
        value = error.get(key)
        if isinstance(value, str):
            parts.append(value.casefold()[:2000])
    return " ".join(parts)


def category_for_error(error: BaseException) -> str:
    """Classify a failure from typed metadata or its HTTP status, safely."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        category = getattr(current, "category", None)
        if category in PROVIDER_FAILURE_CATEGORIES:
            return category
        status = getattr(current, "http_status", None)
        if status is None:
            status = getattr(current, "status_code", None)
        if status is None:
            response = getattr(current, "response", None)
            status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return provider_http_failure(
                "Provider", status, payload=_error_payload(current)
            ).category or "OTHER_PROVIDER_ERROR"
        class_name = type(current).__name__.casefold()
        if (
            isinstance(current, TimeoutError)
            or "timeout" in class_name
            or "timed out" in str(current).casefold()
            or "time out" in str(current).casefold()
        ):
            return "TIMEOUT"
        message = str(current).casefold()
        status_match = re.search(r"\bhttp\s+(\d{3})\b", message)
        if status_match:
            return provider_http_failure(
                "Provider", int(status_match.group(1)), payload=None
            ).category or "OTHER_PROVIDER_ERROR"
        if any(marker in message for marker in (
            "rate limit", "rate limited", "too many requests",
        )):
            return "RATE_LIMIT"
        if any(marker in message for marker in (
            "schema rejected", "schema error", "structured output rejected",
        )):
            return "SCHEMA_ERROR"
        if any(marker in message for marker in (
            "invalid response", "invalid structured", "json parse",
        )):
            return "INVALID_RESPONSE"
        current = current.__cause__ or current.__context__
    return "OTHER_PROVIDER_ERROR"


def failure_detail_for(provider_name: str, error: BaseException) -> ProviderFailureDetail:
    attempts = getattr(error, "attempts", ())
    if isinstance(attempts, tuple) and attempts:
        latest = attempts[-1]
        if isinstance(latest, ProviderFailureDetail):
            return ProviderFailureDetail(
                provider_name=provider_name or latest.provider_name,
                category=latest.category,
                http_status=latest.http_status,
                safe_detail=latest.safe_detail,
                retry_after_seconds=latest.retry_after_seconds,
            )
    category = category_for_error(error)
    current: BaseException | None = error
    seen: set[int] = set()
    status: int | None = None
    retry_after: int | None = None
    safe_detail: str | None = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = status or getattr(current, "http_status", None) or getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        status = status or getattr(response, "status_code", None)
        retry_after = retry_after or getattr(current, "retry_after_seconds", None)
        safe_detail = safe_detail or getattr(current, "safe_detail", None)
        current = current.__cause__ or current.__context__
    return ProviderFailureDetail(
        provider_name=provider_name,
        category=category,
        http_status=status if isinstance(status, int) else None,
        safe_detail=safe_detail,
        retry_after_seconds=retry_after if isinstance(retry_after, int) else None,
    )


def _error_payload(error: BaseException) -> Any:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(error, "response", None)
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            payload = json_method()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None
    return None
