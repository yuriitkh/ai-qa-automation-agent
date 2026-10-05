"""Provider for chat-completions APIs that follow the OpenAI SDK interface."""
import os
from typing import Any

import httpx
from openai import OpenAI

from ..models import AIDiscoveryResult, QATestPlan
from .base import LLMProvider
from .errors import NonRetryableLLMError, RetryableLLMError
from .json_schema import normalize_strict_json_schema, qa_test_plan_schema


class OpenAICompatibleProvider(LLMProvider):
    DEFAULT_MAX_OUTPUT_TOKENS = 4096

    def __init__(
        self,
        name: str,
        api_key_env: str,
        model: str,
        base_url: str | None = None,
        max_output_tokens: int | None = None,
    ):
        self.name = name
        self.api_key_env = api_key_env
        self.model = model
        self.base_url = base_url
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
        return bool(os.environ.get(self.api_key_env, "").strip())

    def _generate_json(self, prompt: str, schema: dict[str, Any], schema_name: str) -> str:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise NonRetryableLLMError(f"{self.name}: {self.api_key_env} is not configured.")
        try:
            client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=30.0)
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
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty response")
            return content
        except (httpx.TimeoutException, TimeoutError) as error:
            raise RetryableLLMError(f"{self.name}: request timed out.") from error
        except httpx.RequestError as error:
            raise RetryableLLMError(f"{self.name}: transport failed ({type(error).__name__}).") from error
        except Exception as error:
            # Authentication, rate limits, unsupported schemas, and API failures
            # may be handled by another configured provider.
            status = getattr(error, "status_code", None)
            if status is not None or error.__class__.__module__.startswith("openai"):
                detail = f"HTTP {status}" if status else type(error).__name__
                raise RetryableLLMError(f"{self.name}: request failed ({detail}).") from error
            raise RetryableLLMError(f"{self.name}: invalid response ({type(error).__name__}).") from error

    def create_test_plan(self, task: str, target_url: str, page_snapshot: str) -> QATestPlan:
        prompt = (
            "Create a structured browser QA test plan. Return only JSON matching the schema. "
            "Use the supplied target URL exactly. Use only these actions: navigate, "
            "assert_page_loaded, assert_title, assert_visible, click, fill, assert_hidden, "
            "assert_url, select_option, assert_text_contains, assert_checked, assert_selected, "
            "assert_enabled, assert_disabled. Use only selectors and URLs present in the task "
            "or page snapshot; never invent or rewrite selectors, parameter names, or URLs. "
            "Use expected for assert_title and assert_url; expected_text for text assertions; "
            "selector for element references; value for fill; option_label for select_option. "
            "For navigation-menu tasks, follow supplied navigation_paths records in order. "
            "For direct navigation, follow supplied direct_navigation_paths steps in order. "
            "If those records are absent, do not invent a path. Preserve Unicode exactly. "
            f"Target URL: {target_url}\nPage snapshot (JSON): {page_snapshot}\nTask: {task}"
        )
        try:
            return QATestPlan.model_validate_json(
                self._generate_json(prompt, qa_test_plan_schema(), "qa_test_plan")
            )
        except (RetryableLLMError, NonRetryableLLMError):
            raise
        except Exception as error:
            raise RetryableLLMError(f"{self.name}: returned an invalid QA test plan.") from error

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
            raise RetryableLLMError(f"{self.name}: returned invalid Discovery data.") from error

