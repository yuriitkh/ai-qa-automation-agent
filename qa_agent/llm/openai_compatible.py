"""Provider for chat-completions APIs that follow the OpenAI SDK interface."""
import os
import json
from typing import Any

import httpx
from openai import OpenAI

from qa_agent.execution_control import current_cancellation, check_cancelled
from ..models import AIDiscoveryResult, QATestPlan
from .base import LLMProvider
from .errors import (
    NonRetryableLLMError,
    RetryableLLMError,
    provider_http_failure,
)
from .json_schema import normalize_strict_json_schema, qa_test_plan_schema
from .usage_metadata import capture_openai_usage, mark_provider_request
from ..reliability import current_reliability_operation, provider_timeout


class OpenAICompatibleProvider(LLMProvider):
    DEFAULT_MAX_OUTPUT_TOKENS = 4096

    def __init__(
        self,
        name: str,
        api_key_env: str,
        model: str,
        base_url: str | None = None,
        max_output_tokens: int | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        *,
        display_name: str | None = None,
        allow_missing_api_key: bool = False,
    ):
        self.name = name
        self.display_name = display_name
        self.api_key_env = api_key_env
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._allow_missing_api_key = allow_missing_api_key
        self.timeout_seconds = timeout_seconds or 30.0
        if max_output_tokens is None:
            configured_limit = os.getenv("LLM_MAX_OUTPUT_TOKENS")
            try:
                max_output_tokens = (
                    int(configured_limit)
                    if configured_limit is not None
                    else self.DEFAULT_MAX_OUTPUT_TOKENS
                )
            except ValueError as error:
                raise ValueError("LLM_MAX_OUTPUT_TOKENS must be a positive integer.") from error
        if max_output_tokens <= 0:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be a positive integer.")
        self.max_output_tokens = max_output_tokens

    @property
    def is_available(self) -> bool:
        if self._allow_missing_api_key:
            return bool(self.base_url and self.model)
        return bool(self._resolved_api_key())

    def _resolved_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key.strip()
        return os.environ.get(self.api_key_env, "").strip()

    def _generate_json(self, prompt: str, schema: dict[str, Any], schema_name: str) -> str:
        check_cancelled()
        api_key = self._resolved_api_key()
        if not api_key:
            if self._allow_missing_api_key:
                # The SDK requires a nonempty constructor argument even for
                # local endpoints that intentionally do not authenticate.
                api_key = "not-required"
            else:
                raise NonRetryableLLMError(f"{self.name}: {self.api_key_env} is not configured.")
        try:
            limits = {"max_retries": 0} if current_reliability_operation() is not None or current_cancellation() is not None else {}
            client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=provider_timeout(self.timeout_seconds), **limits)
            mark_provider_request()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                response_format={"type": "json_schema", "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": normalize_strict_json_schema(schema),
                }},
            )
            capture_openai_usage(response)
            choice = response.choices[0] if response.choices else None
            if choice is not None and getattr(choice, 'finish_reason', None) == 'length':
                raise RetryableLLMError(f"{self.name}: output token limit reached.", category='INVALID_RESPONSE',
                    safe_detail='Output token limit reached', provider_error_code='output_token_limit')
            content = getattr(getattr(choice, 'message', None), 'content', None)
            if not isinstance(content, str) or not content.strip():
                raise RetryableLLMError(f"{self.name}: missing structured response.", category='INVALID_RESPONSE',
                    safe_detail='Missing structured response', provider_error_code='missing_structured_response')
            try:
                json.loads(content)
            except (ValueError, TypeError):
                raise RetryableLLMError(f"{self.name}: invalid JSON response.", category='INVALID_RESPONSE',
                    safe_detail='Invalid structured response', provider_error_code='invalid_json') from None
            return content
        except (RetryableLLMError, NonRetryableLLMError):
            raise
        except (httpx.TimeoutException, TimeoutError) as error:
            raise RetryableLLMError(
                f"{self.name}: request timed out.", category="TIMEOUT",
                safe_detail="Request timed out",
            ) from error
        except httpx.RequestError as error:
            raise RetryableLLMError(
                f"{self.name}: transport failed ({type(error).__name__}).",
                category="PROVIDER_UNAVAILABLE",
                safe_detail="Provider unavailable",
            ) from error
        except Exception as error:
            status = getattr(error, "status_code", None)
            response = getattr(error, "response", None)
            if status is None:
                status = getattr(response, "status_code", None)
            if isinstance(status, int):
                body = getattr(error, "body", None)
                if not isinstance(body, dict) and response is not None:
                    try:
                        candidate = response.json()
                    except Exception:
                        candidate = None
                    body = candidate if isinstance(candidate, dict) else None
                headers = getattr(response, "headers", None)
                failure = provider_http_failure(
                    self.name, status, payload=body, headers=headers
                )
                raise RetryableLLMError(
                    f"{self.name}: request failed (HTTP {status}).",
                    category=failure.category,
                    http_status=failure.http_status,
                    safe_detail=failure.safe_detail,
                    retry_after_seconds=failure.retry_after_seconds,
                    provider_error_code=failure.provider_error_code,
                    provider_error_type=failure.provider_error_type,
                    provider_error_field=failure.provider_error_field,
                ) from error
            if error.__class__.__module__.startswith("openai"):
                category = (
                    "PROVIDER_UNAVAILABLE"
                    if "connection" in error.__class__.__name__.casefold()
                    or "timeout" in error.__class__.__name__.casefold()
                    else "OTHER_PROVIDER_ERROR"
                )
                raise RetryableLLMError(
                    f"{self.name}: request failed ({error.__class__.__name__}).",
                    category=category,
                    safe_detail=(
                        "Provider unavailable" if category == "PROVIDER_UNAVAILABLE"
                        else "Provider request failed"
                    ),
                ) from error
            raise RetryableLLMError(
                f"{self.name}: invalid response ({error.__class__.__name__}).",
                category="INVALID_RESPONSE",
                safe_detail="Invalid structured response",
            ) from error

    def create_structured_output(
        self, prompt: str, schema: dict[str, Any], schema_name: str
    ) -> str:
        return self._generate_json(prompt, schema, schema_name)

    def create_test_plan(self, task: str, target_url: str, page_snapshot: str) -> QATestPlan:
        prompt = (
            "Create a structured browser QA test plan. Return only JSON matching the schema. "
            "Use the supplied target URL exactly. Use only these actions: navigate, "
            "assert_page_loaded, assert_title, assert_visible, click, check, uncheck, fill, assert_hidden, "
            "assert_url, select_option, assert_text_contains, assert_checked, assert_unchecked, assert_selected, "
            "assert_enabled, assert_disabled, assert_value. Use only selectors and URLs present in the task "
            "or page snapshot; never invent or rewrite selectors, parameter names, or URLs. "
            "Use expected for assert_title and assert_url; expected_text for text assertions; "
            "selector for element references; value for fill; option_label for select_option. "
            "assert_value uses selector and expected to check an input/textarea DOM value, not its text. "
            "Use check and uncheck to set checkbox state; use assert_checked and "
            "assert_unchecked to verify checkbox state. "
            "A human TestStep may require multiple ordered executable actions, such as filling "
            "several fields, submitting a form, and checking the result. Preserve all requested "
            "actions and verifications only when the current TestStep requests them. "
            "Respect previous and remaining segment steps; do not submit during preparation or repeat completed setup. "
            "assert_selected requires expected for a discovered select and no expected value for a radio. "
            "Checkbox actions require discovered checkbox identity, never a generic input. "
            "assert_visible requires selector and may include expected_text when requested. "
            "Use interactive_elements and state_elements as well as visible text for exact control identities. "
            "For navigation-menu tasks, follow supplied navigation_paths records in order. "
            "For direct navigation, follow supplied direct_navigation_paths steps in order. "
            "If those records are absent, do not invent a path. Preserve Unicode exactly. "
            f"Target URL: {target_url}\nPage snapshot (JSON): {page_snapshot}\nTask: {task}"
        )
        raw = None
        try:
            raw = self._generate_json(prompt, qa_test_plan_schema(), "qa_test_plan")
            return QATestPlan.model_validate_json(raw)
        except (RetryableLLMError, NonRetryableLLMError):
            raise
        except Exception as error:
            failure = RetryableLLMError(
                f"{self.name}: returned an invalid QA test plan.",
                category="INVALID_RESPONSE", safe_detail="Invalid structured response",
                provider_error_code='invalid_schema_response',
            )
            if raw is not None:
                from qa_agent.candidate_diagnostics import attach_provider_candidate_summary
                attach_provider_candidate_summary(failure, raw)
            raise failure from None

    def create_discovery(self, task: str, target_url: str, page_snapshot: str) -> AIDiscoveryResult:
        prompt = (
            "Return grounded structured navigation paths and interactive elements only. "
            f"URL: {target_url}\nTask: {task}\nPage info: {page_snapshot}"
        )
        try:
            raw = self._generate_json(prompt, AIDiscoveryResult.model_json_schema(), "ai_discovery_result")
            return AIDiscoveryResult.model_validate_json(raw)
        except (RetryableLLMError, NonRetryableLLMError):
            raise
        except Exception as error:
            raise RetryableLLMError(f"{self.name}: returned invalid Discovery data.", category='INVALID_RESPONSE',
                provider_error_code='invalid_schema_response') from error

