import logging
import re
import time
import unicodedata
from typing import Any, Callable
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
from .errors import (
    AllProvidersFailedError,
    NonRetryableLLMError,
    ProviderFailureDetail,
    RetryableLLMError,
    category_for_error,
    failure_detail_for,
)
from .usage_metadata import ProviderTokenUsage, capture_provider_usage
from ..reliability import current_reliability_operation, classify_failure, safe_reason, ReliabilityStopped

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
            self._capture_reliability_usage(provider, usage)
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
        self._capture_reliability_usage(provider, usage)
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

    def _capture_reliability_usage(self, provider, usage) -> None:
        operation = current_reliability_operation()
        if operation is not None:
            try:
                pricing = getattr(self._usage_recorder, "pricing", None)
                model = getattr(provider, "model", None) or getattr(provider, "_model", None)
                cost = pricing.estimate(self._provider_id(provider), model, usage) if pricing is not None and isinstance(model, str) else None
            except Exception:
                cost = None
            operation.capture_usage(usage, cost)

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
        operation = current_reliability_operation()
        if operation is not None:
            return self._supervised_test_plan(task, target_url, page_snapshot, operation)
        providers = self._providers
        if not providers:
            raise RuntimeError("No LLM providers are configured.")

        self._selected_provider_name = None
        self._report_priority(providers)
        failures: list[ProviderFailureDetail] = []
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
                failure = failure_detail_for(
                    self._provider_display_name(provider), error
                )
                failures.append(failure)
                fallback_from_provider = self._provider_display_name(provider)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_PLAN,
                    ProviderAttemptOutcome.RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                if index + 1 < len(providers):
                    print("LLM ROUTER: provider request failed; trying the next provider.")
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
            "All LLM providers failed: " + "; ".join(
                _failure_summary(item) for item in failures
            )
        )

    def _supervised_test_plan(self, task, target_url, page_snapshot, operation):
        """Use this router's configured order with the generation recovery budget."""
        providers = []
        self._selected_provider_name = None
        for provider in list(self._providers):
            if provider.is_available:
                providers.append(provider)
            else:
                self._record_provider_attempt(provider, RequestKind.TEST_PLAN, ProviderAttemptOutcome.UNAVAILABLE)
        if not providers:
            raise ReliabilityStopped("NO_PROVIDER")
        index = providers.index(operation.last_provider) if operation.request_action == "TARGETED_REPAIR" and operation.last_provider in providers else 0
        action, reason = operation.request_action, operation.request_reason
        fallback_from = None
        self._selected_provider_name = None
        usage_context = current_llm_usage_context(OP_GENERATE_AUTOMATION_PLAN)
        while index < len(providers):
            provider = providers[index]
            started = time.perf_counter()
            try:
                result = operation.invoke(
                    provider,
                    lambda: self._invoke_provider_attempt(
                        provider, lambda: provider.create_test_plan(task, target_url, page_snapshot),
                        operation_id=str(operation.record.id), usage_context=usage_context,
                        fallback_from_provider=fallback_from,
                    ),
                    action=action, reason=reason,
                )
            except Exception as error:
                self._record_provider_attempt(
                    provider, RequestKind.TEST_PLAN,
                    ProviderAttemptOutcome.NON_RETRYABLE_ERROR if isinstance(error, NonRetryableLLMError) else ProviderAttemptOutcome.RETRYABLE_ERROR if isinstance(error, RetryableLLMError) else ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                    error=error, started=started,
                )
                action = operation.provider_recovery(error, has_fallback=index + 1 < len(providers))
                if action == "STOP":
                    raise
                reason = safe_reason(classify_failure(error))
                if action == "PROVIDER_FALLBACK":
                    fallback_from = self._provider_display_name(provider)
                    index += 1
                continue
            self._record_provider_attempt(provider, RequestKind.TEST_PLAN, ProviderAttemptOutcome.SUCCESS, started=started, is_selected=True)
            self._selected_provider_name = self._provider_name(provider)
            return result
        raise ReliabilityStopped("NO_FALLBACK")

    def create_discovery(self, task: str, target_url: str, page_snapshot: str) -> AIDiscoveryResult:
        """Route a structured Discovery request through configured providers."""
        providers = self._providers
        if not providers:
            raise RuntimeError("No LLM providers are configured.")
        self._report_priority(providers)
        failures: list[ProviderFailureDetail] = []
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
                failure = failure_detail_for(
                    self._provider_display_name(provider), error
                )
                failures.append(failure)
                fallback_from_provider = self._provider_display_name(provider)
                self._record_provider_attempt(
                    provider,
                    RequestKind.DISCOVERY,
                    ProviderAttemptOutcome.RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                if index + 1 < len(providers):
                    print("LLM ROUTER: provider request failed; trying the next provider.")
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
            raise RuntimeError("All LLM providers failed: " + "; ".join(
                _failure_summary(item) for item in failures
            ))
        raise RuntimeError("No configured LLM providers are available. Unavailable providers: " + ", ".join(unavailable))

    def create_structured_output(
        self,
        prompt: str,
        schema: dict,
        schema_name: str,
        *,
        progress_callback: Callable[..., None] | None = None,
        response_validator: Callable[[str], Any] | None = None,
    ) -> str:
        """Route schema-constrained output through the configured provider chain."""
        providers = self._providers
        if not providers:
            raise RuntimeError("No LLM providers are configured.")
        self._selected_provider_name = None
        self._report_priority(providers)
        failures: list[ProviderFailureDetail] = []
        rate_limited_failures = 0
        timed_out_failures = 0
        unavailable: list[str] = []
        last_retryable_failure: ProviderFailureDetail | None = None
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
                _notify_authoring_progress(
                    progress_callback,
                    "fallback",
                    last_retryable_failure.provider_name,
                    next_provider=display_name,
                    category=last_retryable_failure.category,
                    http_status=last_retryable_failure.http_status,
                    safe_detail=last_retryable_failure.message,
                    retry_after_seconds=last_retryable_failure.retry_after_seconds,
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    logger.info(
                        "LLM authoring fallback: %s -> %s",
                        last_retryable_failure.provider_name,
                        display_name,
                    )
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
                    raise RetryableLLMError(
                        "Provider returned an invalid structured response.",
                        category="INVALID_RESPONSE",
                        safe_detail="Invalid structured response",
                    )
                if response_validator is not None:
                    response_validator(value)
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
                failure = failure_detail_for(display_name, error)
                failures.append(failure)
                fallback_from_provider = display_name
                if failure.category == "RATE_LIMIT":
                    rate_limited_failures += 1
                if failure.category == "TIMEOUT":
                    timed_out_failures += 1
                _notify_authoring_progress(
                    progress_callback, "failed", display_name,
                    category=failure.category,
                    http_status=failure.http_status,
                    safe_detail=failure.message,
                    retry_after_seconds=failure.retry_after_seconds,
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    _log_authoring_provider_failure(failure)
                last_retryable_failure = failure
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
                failure = failure_detail_for(display_name, error)
                _notify_authoring_progress(
                    progress_callback, "failed", display_name,
                    category=failure.category,
                    http_status=failure.http_status,
                    safe_detail=failure.message,
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    _log_authoring_provider_failure(failure)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.NON_RETRYABLE_ERROR,
                    error=error,
                    started=started,
                )
                raise
            except NotImplementedError as error:
                failure = ProviderFailureDetail(
                    display_name,
                    "OTHER_PROVIDER_ERROR",
                    safe_detail="Structured output is not supported",
                )
                _notify_authoring_progress(
                    progress_callback,
                    "failed",
                    display_name,
                    category=failure.category,
                    safe_detail=failure.message,
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    _log_authoring_provider_failure(failure)
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
                failure = failure_detail_for(display_name, error)
                _notify_authoring_progress(
                    progress_callback, "failed", display_name,
                    category=failure.category,
                    http_status=failure.http_status,
                    safe_detail=failure.message,
                )
                if usage_context.operation_type == OP_AUTHOR_TESTCASE:
                    _log_authoring_provider_failure(failure)
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
                "All configured LLM providers failed.",
                all_rate_limited=(rate_limited_failures == len(failures)),
                all_timed_out=(timed_out_failures == len(failures)),
                attempts=tuple(failures),
            )
        raise RuntimeError(
            "No configured LLM providers are available. Unavailable providers: "
            + ", ".join(unavailable)
        )


def _is_rate_limit_failure(error: Exception) -> bool:
    return category_for_error(error) == "RATE_LIMIT"


def _is_timeout_failure(error: Exception) -> bool:
    return category_for_error(error) == "TIMEOUT"


def _provider_failure_category(error: Exception) -> str:
    return category_for_error(error)


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
    return category_for_error(error)


def _failure_summary(failure: ProviderFailureDetail) -> str:
    status = f" HTTP {failure.http_status}" if failure.http_status else ""
    return f"{failure.provider_name} [{failure.category}]{status}"


def _log_authoring_provider_failure(failure: ProviderFailureDetail) -> None:
    status = f" HTTP {failure.http_status}" if failure.http_status else ""
    retry = (
        f" retry-after={failure.retry_after_seconds}s"
        if failure.retry_after_seconds is not None else ""
    )
    logger.warning(
        "LLM authoring provider failed: %s [%s]%s%s",
        failure.provider_name, failure.category, status, retry,
    )


def _notify_authoring_progress(
    callback: Callable[..., None] | None,
    event: str,
    provider: str,
    *,
    next_provider: str | None = None,
    category: str | None = None,
    http_status: int | None = None,
    safe_detail: str | None = None,
    retry_after_seconds: int | None = None,
) -> None:
    if callback is None:
        return
    try:
        callback(
            event,
            provider,
            next_provider=next_provider,
            category=category,
            http_status=http_status,
            safe_detail=safe_detail,
            retry_after_seconds=retry_after_seconds,
        )
    except Exception as error:
        logger.warning(
            "Could not record safe authoring provider progress (%s)",
            type(error).__name__,
        )
