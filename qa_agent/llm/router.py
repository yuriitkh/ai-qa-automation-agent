import time

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
from .errors import NonRetryableLLMError, RetryableLLMError


class LLMRouter:
    def __init__(self, providers: list[LLMProvider]) -> None:
        self._providers = providers
        self._selected_provider_name: str | None = None

    @staticmethod
    def _provider_name(provider: LLMProvider) -> str:
        configured_name = getattr(provider, "name", None)
        if isinstance(configured_name, str) and configured_name:
            return configured_name
        return type(provider).__name__

    def _report_priority(self) -> None:
        print("LLM provider priority:")
        for index, provider in enumerate(self._providers, 1):
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
        if not self._providers:
            raise RuntimeError("No LLM providers are configured.")

        self._selected_provider_name = None
        self._report_priority()
        failures: list[str] = []
        unavailable: list[str] = []
        for index, provider in enumerate(self._providers):
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
                if index + 1 < len(self._providers):
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
        if not self._providers:
            raise RuntimeError("No LLM providers are configured.")
        self._report_priority()
        failures = []
        unavailable = []
        for index, provider in enumerate(self._providers):
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
                if index + 1 < len(self._providers):
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
