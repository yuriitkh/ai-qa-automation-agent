import logging
import re
import time
from typing import Callable

from ..execution_trace import (
    ProviderAttemptOutcome,
    RequestKind,
    elapsed_ms,
    get_active_trace_recorder,
    record_safely,
)
from ..models import AIDiscoveryResult, QATestPlan
from ..redaction import safe_failure_reason
from .base import LLMProvider
from .errors import AllProvidersFailedError, NonRetryableLLMError, RetryableLLMError

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, providers: list[LLMProvider]) -> None:
        self._providers = providers
        self._selected_provider_name: str | None = None

    def replace_providers(self, providers: list[LLMProvider]) -> None:
        """Atomically replace the chain used by subsequent requests."""
        self._providers = list(providers)

    @staticmethod
    def _provider_name(provider: LLMProvider) -> str:
        configured_name = getattr(provider, "name", None)
        if isinstance(configured_name, str) and configured_name:
            return configured_name
        return type(provider).__name__

    def _report_priority(self, providers: list[LLMProvider] | None = None) -> None:
        print("LLM provider priority:")
        for index, provider in enumerate(providers if providers is not None else self._providers, 1):
            name = self._provider_name(provider)
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
            self._provider_name(provider),
            request_kind,
            outcome,
            model=model if isinstance(model, str) else None,
            error=error,
            http_status=http_status if isinstance(http_status, int) else None,
            duration_ms=elapsed_ms(started) if started is not None else None,
            is_selected=is_selected,
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
        for index, provider in enumerate(providers):
            provider_name = type(provider).__name__
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
                plan = provider.create_test_plan(task, target_url, page_snapshot)
            except RetryableLLMError as error:
                reason = safe_failure_reason(error)
                failures.append(f"{provider_name}: {reason}")
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
        for index, provider in enumerate(providers):
            name = type(provider).__name__
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
                result = provider.create_discovery(task, target_url, page_snapshot)
            except RetryableLLMError as error:
                reason = safe_failure_reason(error)
                failures.append(f"{name}: {reason}")
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
                return AIDiscoveryResult.model_validate(result)
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
        for index, provider in enumerate(providers):
            name = self._provider_name(provider)
            if not provider.is_available:
                unavailable.append(name)
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.UNAVAILABLE,
                )
                continue
            display_name = _safe_provider_display_name(name)
            if last_retryable_failure is not None:
                failed_name, category = last_retryable_failure
                _notify_authoring_progress(
                    progress_callback,
                    "fallback",
                    failed_name,
                    next_provider=display_name,
                    category=category,
                )
                logger.info(
                    "LLM authoring fallback: %s",
                    display_name,
                )
                last_retryable_failure = None
            _notify_authoring_progress(
                progress_callback, "selected", display_name
            )
            logger.info("LLM authoring provider selected: %s", display_name)
            started = time.perf_counter()
            try:
                output = provider.create_structured_output(prompt, schema, schema_name)
            except RetryableLLMError as error:
                reason = safe_failure_reason(error)
                failures.append(f"{name}: {reason}")
                if _is_rate_limit_failure(error):
                    rate_limited_failures += 1
                if _is_timeout_failure(error):
                    timed_out_failures += 1
                category = _provider_failure_category(error)
                _notify_authoring_progress(
                    progress_callback, "failed", display_name, category=category
                )
                logger.warning(
                    "LLM authoring provider failed: %s [%s]",
                    display_name,
                    category,
                )
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
                logger.warning(
                    "LLM authoring provider failed: %s [%s]",
                    display_name,
                    category,
                )
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
                logger.warning(
                    "LLM authoring provider failed: %s [PROVIDER_ERROR]",
                    display_name,
                )
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                    error=error,
                    started=started,
                )
                raise RuntimeError(
                    f"{name} does not support structured output."
                ) from error
            except Exception as error:
                category = _provider_failure_category(error)
                _notify_authoring_progress(
                    progress_callback, "failed", display_name, category=category
                )
                logger.warning(
                    "LLM authoring provider failed: %s [%s]",
                    display_name,
                    category,
                )
                self._record_provider_attempt(
                    provider,
                    RequestKind.TEST_CASE_AUTHORING,
                    ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                    error=error,
                    started=started,
                )
                raise
            else:
                if not isinstance(output, str):
                    error = TypeError("provider structured output must be text")
                    self._record_provider_attempt(
                        provider,
                        RequestKind.TEST_CASE_AUTHORING,
                        ProviderAttemptOutcome.UNCLASSIFIED_ERROR,
                        error=error,
                        started=started,
                    )
                    raise error
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
    return {
        "groq": "Groq",
        "gemini": "Gemini",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
    }.get(value.casefold(), "Configured provider")


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
