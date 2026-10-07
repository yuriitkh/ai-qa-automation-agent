import logging
import re
import time
import unicodedata
from typing import Callable
from datetime import datetime, timezone
from uuid import uuid4

from ..execution_trace import (
    ProviderAttemptOutcome,
    RequestKind,
    elapsed_ms,
    get_active_trace_recorder,
    record_safely,
)
from ..models import AIDiscoveryResult, QATestPlan
from ..redaction import safe_failure_reason
from ..redaction import redact_secrets
from ..llm_usage import (
    OP_AUTHOR_TESTCASE,
    OP_DISCOVERY,
    OP_GENERATE_AUTOMATION_PLAN,
    OP_OTHER,
    LLMUsageContext,
    current_llm_usage_context,
)
from .base import LLMProvider
from .errors import AllProvidersFailedError, NonRetryableLLMError, RetryableLLMError
from .usage_metadata import ProviderTokenUsage, capture_provider_usage

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, providers: list[LLMProvider], *, usage_recorder=None) -> None:
        self._providers = providers
        self._selected_provider_name: str | None = None
        self._usage_recorder = usage_recorder

    def replace_providers(self, providers: list[LLMProvider]) -> None:
        """Atomically replace the chain used by subsequent requests."""
        self._providers = list(providers)

    @staticmethod
    def _provider_name(provider: LLMProvider) -> str:
        configured_name = getattr(provider, "name", None)
        if isinstance(configured_name, str) and configured_name:
            return configured_name
        return type(provider).__name__

    @classmethod
    def _provider_display_name(cls, provider: LLMProvider) -> str:
        configured = getattr(provider, "display_name", None)
        if isinstance(configured, str) and configured.strip():
            return _safe_provider_display_name(configured)
        return _safe_provider_display_name(cls._provider_name(provider))

    @classmethod
    def _provider_observability_name(cls, provider: LLMProvider) -> str:
        if getattr(provider, "is_custom", False):
            return cls._provider_display_name(provider)
        return cls._provider_name(provider)

    @staticmethod
    def _provider_id(provider: LLMProvider) -> str:
        configured_name = getattr(provider, "name", None)
        raw = configured_name if isinstance(configured_name, str) and configured_name else type(provider).__name__
        provider_id = {
            "groqprovider": "groq",
            "geminiprovider": "gemini",
            "openai-compatible": "openai",
        }.get(raw.casefold(), raw)
        return provider_id.casefold()

    def _report_priority(self, providers: list[LLMProvider] | None = None) -> None:
        print("LLM provider priority:")
        for index, provider in enumerate(providers if providers is not None else self._providers, 1):
            name = self._provider_observability_name(provider)
            state = "AVAILABLE" if provider.is_available else "SKIPPED - API key not configured"
            print(f"{index}. {name} [{state}]")

    def _record_provider_attempt(
        self,
        provider: LLMProvider,
        request_kind: RequestKind,
        outcome: ProviderAttemptOutcome,
        *,
        error: Exception | None = None,
        started: float | None = None,
        is_selected: bool = False,
    ) -> None:
        """Record one provider attempt in the active execution trace, if any."""
        recorder = get_active_trace_recorder()
        if recorder is None:
            return
        model = getattr(provider, "model", None)
        if not isinstance(model, str) or not model:
            model = getattr(provider, "_model", None)
        http_status = (
            getattr(error, "status_code", None) if error is not None else None
        )
        record_safely(
            recorder,
            "record_provider_attempt",
            self._provider_observability_name(provider),
            request_kind,
            outcome,
            model=model if isinstance(model, str) else None,
            error=error,
            http_status=http_status if isinstance(http_status, int) else None,
            duration_ms=elapsed_ms(started) if started is not None else None,
            is_selected=is_selected,
        )

    def _invoke_provider_attempt(
        self,
        provider: LLMProvider,
        operation,
        *,
        operation_id: str,
        usage_context: LLMUsageContext,
        fallback_from_provider: str | None,
    ):
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        usage: ProviderTokenUsage | None = None
        try:
            with capture_provider_usage() as captured:
                try:
                    result = operation()
                finally:
                    usage = captured.usage
        except Exception as error:
            self._record_usage_attempt(
                provider,
                operation_id,
                usage_context,
                started_at,
                started,
                usage,
                "FAILED",
                error,
                fallback_from_provider,
            )
            raise
        self._record_usage_attempt(
            provider,
            operation_id,
            usage_context,
            started_at,
            started,
            usage,
            "SUCCESS",
            None,
            fallback_from_provider,
        )
        return result

    def _record_usage_attempt(
        self,
        provider: LLMProvider,
        operation_id: str,
        usage_context: LLMUsageContext,
        started_at: datetime,
        started: float,
        usage: ProviderTokenUsage | None,
        request_status: str,
        error: Exception | None,
        fallback_from_provider: str | None,
    ) -> None:
        if self._usage_recorder is None:
            return
        name = self._provider_name(provider)
        model = getattr(provider, "model", None)
        if not isinstance(model, str) or not model:
            model = getattr(provider, "_model", None)
        try:
            self._usage_recorder.record_attempt(
                operation_id=operation_id,
                operation_type=usage_context.operation_type or OP_OTHER,
                provider_id=self._provider_id(provider),
                provider_name=self._provider_display_name(provider),
                model=model if isinstance(model, str) else None,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=elapsed_ms(started),
                request_status=request_status,
                usage=usage,
                fallback_from_provider=fallback_from_provider,
                error_category=_telemetry_error_category(error) if error else None,
                related_test_case_id=usage_context.related_test_case_id,
                related_test_case_public_id=usage_context.related_test_case_public_id,
                related_workflow_id=usage_context.related_workflow_id,
            )
        except Exception as telemetry_error:
            logger.warning(
                "LLM usage telemetry write failed (%s)",
                type(telemetry_error).__name__,
            )

    @property
    def selected_provider_name(self) -> str | None:
        return self._selected_provider_name

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        providers = self._providers
        if not providers:
            raise RuntimeError("No LLM providers are configured.")

        self._selected_provider_name = None
        self._report_priority(providers)
        failures: list[str] = []
        unavailable: list[str] = []
        operation_id = str(uuid4())
        usage_context = current_llm_usage_context(OP_GENERATE_AUTOMATION_PLAN)
        fallback_from_provider: str | None = None
        for index, provider in enumerate(providers):
            provider_name = (
                self._provider_observability_name(provider)
                if getattr(provider, "is_custom", False)
                else type(provider).__name__
            )
            if not provider.is_available:
                unavailable.append(provider_name)
                print(
                    f"LLM ROUTER: Skipping {provider_name}; "
                    "provider is not configured."
                )
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_PLAN,
                    ProviderAttemptOutcome.UNAVAILABLE,
                )
                continue

            started = time.perf_counter()
            try:
                plan = self._invoke_provider_attempt(
                    provider,
                    lambda: provider.create_test_plan(task, target_url, page_snapshot),
                    operation_id=operation_id,
                    usage_context=usage_context,
                    fallback_from_provider=fallback_from_provider,
                )
            except RetryableLLMError as error:
                reason = safe_failure_reason(error)
                failures.append(f"{provider_name}: {reason}")
                fallback_from_provider = self._provider_display_name(provider)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_PLAN,
                    ProviderAttemptOutcome.RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                if index + 1 < len(providers):
                    print(
                        f"LLM ROUTER: {provider_name} had a retryable failure "
                        f"({reason}); trying the next provider."
                    )
            except NonRetryableLLMError as error:
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_PLAN,
                    ProviderAttemptOutcome.NON_RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                raise
            except Exception as error:
                # P2-2 (Choice D): an unclassified failure stays unclassified
                # — no fallback, no wrapping, the SAME exception propagates —
                # but the attempted provider must be observable in the trace.
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_PLAN,
                    ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                    error=error,
                    started=started,
                )
                raise
            else:
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_PLAN,
                    ProviderAttemptOutcome.SUCCESS,
                    started=started,
                    is_selected=True,
                )
                self._selected_provider_name = self._provider_name(provider)
                print(f"Selected provider: {self._selected_provider_name}")
                return plan

        if not failures:
            providers = ", ".join(unavailable)
            raise RuntimeError(
                "No configured LLM providers are available. "
                f"Unavailable providers: {providers}."
            )

        raise RuntimeError(
            "All LLM providers failed: " + "; ".join(failures)
        )

    def create_discovery(self, task: str, target_url: str, page_snapshot: str) -> AIDiscoveryResult:
        """Route a structured Discovery request through configured providers."""
        providers = self._providers
        if not providers:
            raise RuntimeError("No LLM providers are configured.")
        self._report_priority(providers)
        failures = []
        unavailable = []
        operation_id = str(uuid4())
        usage_context = current_llm_usage_context(OP_DISCOVERY)
        fallback_from_provider: str | None = None
        for index, provider in enumerate(providers):
            name = (
                self._provider_observability_name(provider)
                if getattr(provider, "is_custom", False)
                else type(provider).__name__
            )
            if not provider.is_available:
                unavailable.append(name)
                self._record_provider_attempt(
                    provider,
                    RequestKind.DISCOVERY,
                    ProviderAttemptOutcome.UNAVAILABLE,
                )
                continue
            started = time.perf_counter()
            try:
                result = self._invoke_provider_attempt(
                    provider,
                    lambda: AIDiscoveryResult.model_validate(
                        provider.create_discovery(task, target_url, page_snapshot)
                    ),
                    operation_id=operation_id,
                    usage_context=usage_context,
                    fallback_from_provider=fallback_from_provider,
                )
            except RetryableLLMError as error:
                reason = safe_failure_reason(error)
                failures.append(f"{name}: {reason}")
                fallback_from_provider = self._provider_display_name(provider)
                self._record_provider_attempt(
                    provider,
                    RequestKind.DISCOVERY,
                    ProviderAttemptOutcome.RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                if index + 1 < len(providers):
                    print(
                        f"LLM ROUTER: {name} request failed ({reason}); "
                        "trying the next provider."
                    )
            except NonRetryableLLMError as error:
                self._record_provider_attempt(
                    provider,
                    RequestKind.DISCOVERY,
                    ProviderAttemptOutcome.NON_RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                raise
            except NotImplementedError as error:
                raise RuntimeError(f"{name} does not support Discovery output.") from error
            except Exception as error:
                # P2-2 (Choice D): same contract as create_test_plan — record
                # the attempted provider, then re-raise the original
                # exception unchanged with no fallback to the next provider.
                self._record_provider_attempt(
                    provider,
                    RequestKind.DISCOVERY,
                    ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                    error=error,
                    started=started,
                )
                raise
            else:
                self._record_provider_attempt(
                    provider,
                    RequestKind.DISCOVERY,
                    ProviderAttemptOutcome.SUCCESS,
                    started=started,
                    is_selected=True,
                )
                self._selected_provider_name = self._provider_name(provider)
                print(f"Selected provider: {self._selected_provider_name}")
                return result
        if failures:
            raise RuntimeError("All LLM providers failed: " + "; ".join(failures))
        raise RuntimeError("No configured LLM providers are available. Unavailable providers: " + ", ".join(unavailable))

    def create_structured_output(
        self,
        prompt: str,
        schema: dict,
        schema_name: str,
        *,
        progress_callback: Callable[..., None] | None = None,
    ) -> str:
        """Route schema-constrained output through the configured provider chain."""
        providers = self._providers
        if not providers:
            raise RuntimeError("No LLM providers are configured.")
        self._selected_provider_name = None
        self._report_priority(providers)
        failures: list[str] = []
        rate_limited_failures = 0
        timed_out_failures = 0
        unavailable: list[str] = []
        last_retryable_failure: tuple[str, str] | None = None
        operation_id = str(uuid4())
        usage_context = current_llm_usage_context(OP_AUTHOR_TESTCASE)
        if usage_context.operation_type != OP_AUTHOR_TESTCASE:
            progress_callback = None
        fallback_from_provider: str | None = None
        for index, provider in enumerate(providers):
            name = self._provider_name(provider)
            if not provider.is_available:
                unavailable.append(self._provider_observability_name(provider))
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.UNAVAILABLE,
                )
                continue
            display_name = self._provider_display_name(provider)
            if last_retryable_failure is not None:
                failed_name, category = last_retryable_failure
                _notify_authoring_progress(
                    progress_callback,
                    "fallback",
                    failed_name,
                    next_provider=display_name,
                    category=category,
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    logger.info("LLM authoring fallback: %s", display_name)
                last_retryable_failure = None
            _notify_authoring_progress(
                progress_callback, "selected", display_name
            )
            if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                logger.info("LLM authoring provider selected: %s", display_name)
            started = time.perf_counter()

            def invoke_structured_output() -> str:
                value = provider.create_structured_output(prompt, schema, schema_name)
                if not isinstance(value, str):
                    raise TypeError("provider structured output must be text")
                return value

            try:
                output = self._invoke_provider_attempt(
                    provider,
                    invoke_structured_output,
                    operation_id=operation_id,
                    usage_context=usage_context,
                    fallback_from_provider=fallback_from_provider,
                )
            except RetryableLLMError as error:
                reason = safe_failure_reason(error)
                failures.append(f"{self._provider_observability_name(provider)}: {reason}")
                fallback_from_provider = display_name
                if _is_rate_limit_failure(error):
                    rate_limited_failures += 1
                if _is_timeout_failure(error):
                    timed_out_failures += 1
                category = _provider_failure_category(error)
                _notify_authoring_progress(
                    progress_callback, "failed", display_name, category=category
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    logger.warning("LLM authoring provider failed: %s [%s]", display_name, category)
                last_retryable_failure = (display_name, category)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                if index + 1 < len(providers):
                    print(
                        "AI provider request failed; trying another configured provider."
                    )
            except NonRetryableLLMError as error:
                category = _provider_failure_category(error)
                _notify_authoring_progress(
                    progress_callback, "failed", display_name, category=category
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    logger.warning("LLM authoring provider failed: %s [%s]", display_name, category)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.NON_RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                raise
            except NotImplementedError as error:
                _notify_authoring_progress(
                    progress_callback,
                    "failed",
                    display_name,
                    category="PROVIDER_ERROR",
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    logger.warning("LLM authoring provider failed: %s [PROVIDER_ERROR]", display_name)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                    error=error,
                    started=started,
                )
                raise RuntimeError(
                    f"{self._provider_observability_name(provider)} does not support structured output."
                ) from error
            except Exception as error:
                category = _provider_failure_category(error)
                _notify_authoring_progress(
                    progress_callback, "failed", display_name, category=category
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    logger.warning("LLM authoring provider failed: %s [%s]", display_name, category)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                    error=error,
                    started=started,
                )
                raise
            else:
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.SUCCESS,
                    started=started,
                    is_selected=True,
                )
                self._selected_provider_name = name
                print(f"Selected provider: {self._selected_provider_name}")
                _notify_authoring_progress(
                    progress_callback, "completed", display_name
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    logger.info("LLM authoring completed: %s", display_name)
                return output
        if failures:
            raise AllProvidersFailedError(
                "All LLM providers failed: " + "; ".join(failures),
                all_rate_limited=(rate_limited_failures == len(failures)),
                all_timed_out=(timed_out_failures == len(failures)),
            )
        raise RuntimeError(
            "No configured LLM providers are available. Unavailable providers: "
            + ", ".join(unavailable)
        )


def _is_rate_limit_failure(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return True
    message = str(error).casefold()
    return any(marker in message for marker in (
        "http 429",
        "429 rate",
        "rate limit",
        "rate limited",
        "too many requests",
    ))


def _is_timeout_failure(error: Exception) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or "timeout" in type(current).__name__.casefold():
            return True
        if "timed out" in str(current).casefold():
            return True
        current = current.__cause__ or current.__context__
    return False


def _provider_failure_category(error: Exception) -> str:
    if _is_rate_limit_failure(error):
        return "RATE_LIMIT"
    if _is_timeout_failure(error):
        return "TIMEOUT"
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is None:
        match = re.search(r"HTTP\s+(\d{3})", str(error), re.IGNORECASE)
        status = int(match.group(1)) if match else None
    if status in {401, 403}:
        return "AUTHENTICATION"
    return "PROVIDER_ERROR"


def _safe_provider_display_name(value: str) -> str:
    known = {
        "groq": "Groq",
        "groqprovider": "Groq",
        "gemini": "Gemini",
        "geminiprovider": "Gemini",
        "openai": "OpenAI",
        "openai-compatible": "OpenAI",
        "openrouter": "OpenRouter",
    }
    normalized = " ".join(redact_secrets(str(value)).split()).strip()
    if normalized.casefold() in known:
        return known[normalized.casefold()]
    if (
        normalized and len(normalized) <= 80
        and not any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in normalized)
    ):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,59}", normalized):
            return normalized.replace("_", " ").replace("-", " ").title()
        return normalized
    return "Configured provider"


def _telemetry_error_category(error: Exception) -> str:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).casefold()
        if isinstance(current, TimeoutError) or "timeout" in type(current).__name__.casefold() or "timed out" in message:
            return "TIMEOUT"
        if any(marker in message for marker in ("http 429", "rate limit", "too many requests")):
            return "RATE_LIMIT"
        status = getattr(current, "status_code", None)
        if status is None:
            status = getattr(getattr(current, "response", None), "status_code", None)
        if status is None:
            match = re.search(r"HTTP\s+(\d{3})", message, re.IGNORECASE)
            status = int(match.group(1)) if match else None
        if status in {401, 403}:
            return "AUTH_ERROR"
        if status in {408, 429}:
            return "TIMEOUT" if status == 408 else "RATE_LIMIT"
        if isinstance(status, int) and status >= 500:
            return "PROVIDER_UNAVAILABLE"
        if any(marker in message for marker in ("invalid", "schema", "parse", "json")):
            return "INVALID_OUTPUT"
        if "unavailable" in message or "not configured" in message:
            return "PROVIDER_UNAVAILABLE"
        current = current.__cause__ or current.__context__
    return "OTHER"


def _notify_authoring_progress(
    callback: Callable[..., None] | None,
    event: str,
    provider: str,
    *,
    next_provider: str | None = None,
    category: str | None = None,
) -> None:
    if callback is None:
        return
    try:
        callback(
            event,
            provider,
            next_provider=next_provider,
            category=category,
        )
    except Exception as error:
        logger.warning(
            "Could not record safe authoring provider progress (%s)",
            type(error).__name__,
        )
