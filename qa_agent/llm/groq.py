import json
import os
from typing import Any

import httpx

from ..models import AIDiscoveryResult, QATestPlan
from .base import LLMProvider
from .errors import (
    NonRetryableLLMError,
    RetryableLLMError,
    provider_http_failure,
)
from .json_schema import normalize_strict_json_schema, qa_test_plan_schema
from .usage_metadata import capture_openai_usage


class GroqProvider(LLMProvider):
    _model = "openai/gpt-oss-20b"
    _endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model or os.getenv("GROQ_MODEL", self._model)
        self._timeout_seconds = timeout_seconds or 30.0
        self._max_output_tokens = max_output_tokens or 4096

    def _resolved_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key.strip()
        return os.environ.get("GROQ_API_KEY", "").strip()

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        """Return strict variants with an explicit common parameter shape."""
        return qa_test_plan_schema()

    @property
    def is_available(self) -> bool:
        return bool(self._resolved_api_key())

    def _structured_request_payload(
        self,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build a Groq JSON-object request with the schema in the prompt.

        The configured Groq model returns ``json_validate_failed`` for the
        Chat Completions ``json_schema`` response format, including the minimal
        connection schema. JSON-object mode is the supported structured mode
        used here; the normalized schema guides generation and callers still
        apply their canonical validators to the returned JSON.
        """
        schema_text = json.dumps(
            normalize_strict_json_schema(schema),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        message_content = (
            f"{prompt}\n\n"
            f"Output contract: {schema_name}. Return exactly one JSON object "
            "conforming to this JSON Schema:\n"
            f"{schema_text}"
        )
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message_content}],
            "response_format": {"type": "json_object"},
        }

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        api_key = self._resolved_api_key()
        if not api_key:
            raise NonRetryableLLMError("GROQ_API_KEY is not set.")

        prompt = (
            "Create a structured browser QA test plan for this task. "
            "Output exactly one JSON object matching the supplied schema. "
            "The JSON root MUST be an object, never an array. The top-level "
            "fields MUST be exactly url and steps. Every item in steps MUST be "
            "one object containing action and parameters together; never split, "
            "reorder, or emit these fields as separate array items. Each step "
            "object MUST contain only action and parameters. Output no Markdown, "
            "code fences, comments, or surrounding text. "
            "Use the supplied target URL exactly; do not infer another URL. "
            "Return that URL and ordered executable test steps. "
            "Allowed actions are ONLY: navigate, assert_page_loaded, "
            "assert_title, assert_visible, click, fill, assert_hidden, and "
            "assert_url, select_option, assert_text_contains, assert_checked, "
            "assert_selected, assert_enabled, and assert_disabled. "
            "Never invent, rename, or substitute action names. Every parameters "
            "object MUST contain exactly these five common keys: url, expected, "
            "selector, expected_text, and value. Include option_label only for "
            "select_option, where it is required; omit it for every other action. "
            "For the five common keys, set fields unused by an action to JSON null. "
            "navigate uses {url: target URL}; assert_page_loaded uses all five common "
            "keys set to null. assert_title uses {expected: expected page title}. "
            'The assert_title parameter name MUST be exactly "expected"; '
            "never use expected_title. assert_visible requires selector and may "
            "include expected_text when the requested verification names visible text. "
            "click uses parameters {selector: CSS selector}. "
            "fill uses {selector: CSS selector, value: text to fill}. "
            "select_option uses {selector, option_label}; option_label is the visible option text. "
            "For assert_text_contains use expected_text and optionally selector. "
            "assert_checked verifies checkbox state; assert_selected verifies radio state "
            "using selector with expected null, or select state using selector and expected "
            "option label or value. assert_enabled and assert_disabled use selector. "
            "One human TestStep may require multiple ordered executable actions, including "
            "filling several fields and then submitting. Preserve all requested actions "
            "and verifications. "
            "assert_hidden uses parameters {selector: CSS selector}. "
            "assert_url uses parameters {expected: expected current URL}. "
            'The assert_url parameter name MUST be exactly "expected". '
            "Use only a URL supported by the task or browser context; never "
            "invent an expected URL. "
            "Never use a parameter that does not match its action. Parameter names are "
            "case-sensitive. Never invent, rename, normalize, or silently correct "
            "action or parameter names. Do not use text instead of expected_text "
            "or locator instead of selector. Assertions MUST be based only on "
            "elements explicitly present in the supplied page snapshot. For every "
            "selector parameter in click, fill, assert_hidden, or assert_visible, "
            "use an exact selector listed in visible_text_elements, headings, "
            "links, or buttons. Use "
            "visible_text_elements to ground ordinary visible-text assertions. "
            "Use each selector exactly as provided in the PAGE SNAPSHOT. Do not "
            "modify selectors. Preserve Unicode characters exactly; do not "
            "convert them into literal escape sequences. "
            "Never invent selectors or page elements from prior knowledge. If the "
            "requested text or element is absent from the snapshot, do not invent "
            "a selector or assertion. "
            "For navigation-menu tasks, use only the supplied navigation_paths "
            "records. For each requested combination, navigate to the task's root "
            "URL, click its menu_button_selector, click menu_tab_selector when "
            "present, click menu_item_selector, then click submenu_selector, "
            "assert_url with that record's expected_url, and assert_visible with "
            "its heading_selector and heading_text. Repeat this sequence for at "
            "least three distinct records when the task requests three combinations. "
            "For direct-link navigation, use only supplied direct_navigation_paths records. "
            "For each direct record, navigate to root_url, click the steps selectors in array order, "
            "assert_url with expected_url, and then assert_visible with heading_selector and heading_text. "
            "Use only the supplied selectors and URLs; never invent or infer a missing step. "
            "If direct_navigation_paths is empty, do not invent a path. "
            "If a cookie banner is present, click its discovered strictly-necessary "
            "consent selector before opening the menu. Never select menu items by "
            "their order, numeric position, nth-child, or index. "
            f"Target URL: {target_url}\n"
            f"Page snapshot (JSON): {page_snapshot}\n"
            f"Task: {task}"
        )

        request_payload = self._structured_request_payload(
            prompt, self._response_schema(), "qa_test_plan", 8192
        )
        try:
            response = httpx.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_payload,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise RetryableLLMError(
                "Groq request timed out.", category="TIMEOUT",
                safe_detail="Request timed out",
            ) from error
        except httpx.RequestError as error:
            raise RetryableLLMError(
                "Groq transport request failed.",
                category="PROVIDER_UNAVAILABLE",
                safe_detail="Provider unavailable",
            ) from error

        if response.is_error:
            raise provider_http_failure(
                "Groq", response.status_code,
                payload=_response_payload(response), headers=response.headers,
            )

        try:
            response_data = response.json()
            capture_openai_usage(response_data)
            output_text = response_data["choices"][0]["message"]["content"]
            plan = QATestPlan.model_validate_json(output_text)
            return plan
        except Exception as error:
            raise RetryableLLMError(
                "Groq returned an invalid QA test plan.",
                category="INVALID_RESPONSE",
                safe_detail="Invalid structured response",
            ) from error

    def create_discovery(self, task: str, target_url: str, page_snapshot: str) -> AIDiscoveryResult:
        api_key = self._resolved_api_key()
        if not api_key:
            raise NonRetryableLLMError("GROQ_API_KEY is not set.")
        schema = AIDiscoveryResult.model_json_schema()
        prompt = (
            "Return grounded structured discovery candidates only. Never return code, "
            "instructions to execute, or perform browser actions. "
            f"URL: {target_url}\nTask: {task}\nPage info: {page_snapshot}"
        )
        payload = self._structured_request_payload(
            prompt, schema, "ai_discovery_result", self._max_output_tokens
        )
        try:
            response = httpx.post(self._endpoint,
                headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=self._timeout_seconds)
        except httpx.TimeoutException as error:
            raise RetryableLLMError(
                "Groq request timed out.", category="TIMEOUT",
                safe_detail="Request timed out",
            ) from error
        except httpx.RequestError as error:
            raise RetryableLLMError(
                "Groq transport request failed.", category="PROVIDER_UNAVAILABLE",
                safe_detail="Provider unavailable",
            ) from error
        if response.is_error:
            raise provider_http_failure(
                "Groq", response.status_code,
                payload=_response_payload(response), headers=response.headers,
            )
        try:
            response_data = response.json()
            capture_openai_usage(response_data)
            output = response_data["choices"][0]["message"]["content"]
            return AIDiscoveryResult.model_validate_json(output)
        except Exception as error:
            raise RetryableLLMError(
                "Groq returned invalid Discovery data.",
                category="INVALID_RESPONSE",
                safe_detail="Invalid structured response",
            ) from error

    def create_structured_output(
        self, prompt: str, schema: dict[str, Any], schema_name: str
    ) -> str:
        api_key = self._resolved_api_key()
        if not api_key:
            raise NonRetryableLLMError("GROQ_API_KEY is not set.")
        payload = self._structured_request_payload(
            prompt, schema, schema_name, self._max_output_tokens
        )
        try:
            response = httpx.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise RetryableLLMError(
                "Groq request timed out.", category="TIMEOUT",
                safe_detail="Request timed out",
            ) from error
        except httpx.RequestError as error:
            raise RetryableLLMError(
                "Groq transport request failed.", category="PROVIDER_UNAVAILABLE",
                safe_detail="Provider unavailable",
            ) from error
        if response.is_error:
            raise provider_http_failure(
                "Groq", response.status_code,
                payload=_response_payload(response), headers=response.headers,
            )
        try:
            response_data = response.json()
            capture_openai_usage(response_data)
            output = response_data["choices"][0]["message"]["content"]
            parsed_output = json.loads(output)
            if not isinstance(parsed_output, dict):
                raise ValueError("structured output root must be an object")
            return output
        except json.JSONDecodeError as error:
            raise RetryableLLMError(
                "Groq returned invalid JSON.",
                category="INVALID_RESPONSE",
                safe_detail="Invalid JSON response",
            ) from error
        except Exception as error:
            raise RetryableLLMError(
                "Groq returned an invalid structured-output response.",
                category="INVALID_RESPONSE",
                safe_detail="Invalid structured response",
            ) from error


def _response_payload(response: httpx.Response) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None
