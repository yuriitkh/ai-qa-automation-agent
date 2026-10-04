import os

from ..models import AIDiscoveryResult, QATestPlan
from .base import LLMProvider
from .errors import NonRetryableLLMError, RetryableLLMError


def _safe_failure_reason(error: Exception) -> str:
    reason = " ".join(str(error).split())
    for variable in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        secret = os.environ.get(variable)
        if secret:
            reason = reason.replace(secret, "[REDACTED]")
    return (reason or type(error).__name__)[:300]


class LLMRouter:
    def __init__(self, providers: list[LLMProvider]) -> None:
        self._providers = providers
        self._selected_provider_name: str | None = None

    @property
    def selected_provider_name(self) -> str | None:
        return self._selected_provider_name

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        if not self._providers:
            raise RuntimeError("No LLM providers are configured.")

        self._selected_provider_name = None
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
                self._selected_provider_name = provider_name
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
        failures = []
        unavailable = []
        for index, provider in enumerate(self._providers):
            name = type(provider).__name__
            if not provider.is_available:
                unavailable.append(name)
                continue
            try:
                result = provider.create_discovery(task, target_url, page_snapshot)
                self._selected_provider_name = name
                return AIDiscoveryResult.model_validate(result)
            except RetryableLLMError as error:
                failures.append(f"{name}: {_safe_failure_reason(error)}")
            except NonRetryableLLMError:
                raise
            except NotImplementedError as error:
                raise RuntimeError(f"{name} does not support Discovery output.") from error
        if failures:
            raise RuntimeError("All LLM providers failed: " + "; ".join(failures))
        raise RuntimeError("No configured LLM providers are available. Unavailable providers: " + ", ".join(unavailable))
