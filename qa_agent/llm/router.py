import os

from ..models import AIDiscoveryResult, QATestPlan
from .base import LLMProvider
from .errors import NonRetryableLLMError, RetryableLLMError


def _safe_failure_reason(error: Exception) -> str:
    reason = " ".join(str(error).split())
    for variable, secret in os.environ.items():
        if variable.endswith(("_API_KEY", "_TOKEN", "_SECRET")) and secret:
            reason = reason.replace(secret, "[REDACTED]")
    return (reason or type(error).__name__)[:300]


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
                continue

            try:
                plan = provider.create_test_plan(task, target_url, page_snapshot)
                self._selected_provider_name = self._provider_name(provider)
                print(f"Selected provider: {self._selected_provider_name}")
                return plan
            except RetryableLLMError as error:
                reason = _safe_failure_reason(error)
                failures.append(f"{provider_name}: {reason}")
                if index + 1 < len(self._providers):
                    print(
                        f"LLM ROUTER: {provider_name} had a retryable failure "
                        f"({reason}); trying the next provider."
                    )
            except NonRetryableLLMError:
                raise

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
                continue
            try:
                result = provider.create_discovery(task, target_url, page_snapshot)
                self._selected_provider_name = self._provider_name(provider)
                print(f"Selected provider: {self._selected_provider_name}")
                return AIDiscoveryResult.model_validate(result)
            except RetryableLLMError as error:
                failures.append(f"{name}: {_safe_failure_reason(error)}")
                if index + 1 < len(self._providers):
                    print(f"LLM ROUTER: {name} request failed ({_safe_failure_reason(error)}); trying the next provider.")
            except NonRetryableLLMError:
                raise
            except NotImplementedError as error:
                raise RuntimeError(f"{name} does not support Discovery output.") from error
        if failures:
            raise RuntimeError("All LLM providers failed: " + "; ".join(failures))
        raise RuntimeError("No configured LLM providers are available. Unavailable providers: " + ", ".join(unavailable))
