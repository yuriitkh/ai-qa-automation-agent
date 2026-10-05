import json
import os
from typing import Any

import httpx

from ..models import AIDiscoveryResult, QATestPlan
from .base import LLMProvider
from .errors import NonRetryableLLMError, RetryableLLMError
from .json_schema import qa_test_plan_schema


class GroqProvider(LLMProvider):
    _model = "openai/gpt-oss-20b"
    _endpoint = "https://api.groq.com/openai/v1/chat/completions"

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        """Return strict variants with an explicit common parameter shape."""
        return qa_test_plan_schema()

    @property
    def is_available(self) -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise NonRetryableLLMError("GROQ_API_KEY is not set.")

        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "qa_test_plan",
                "strict": True,
                "schema": self._response_schema(),
            },
        }
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
            "never use expected_title. assert_visible uses parameters "
            "{selector: CSS selector, expected_text: expected visible text}. "
            "click uses parameters {selector: CSS selector}. "
            "fill uses {selector: CSS selector, value: text to fill}. "
            "select_option uses {selector, option_label}; option_label is the visible option text. "
            "For assert_text_contains use expected_text and optionally selector. "
            "assert_checked verifies checkbox state; assert_selected verifies radio state "
            "using selector with expected null, or select state using selector and expected "
            "option label or value. assert_enabled and assert_disabled use selector. "
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

        request_payload = {
            "model": os.getenv("GROQ_MODEL", self._model),
            "max_completion_tokens": 8192,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "response_format": response_format,
        }
        debug_request = os.environ.get("GROQ_DEBUG_REQUEST", "").lower() in {
            "1", "true", "yes",
        }
        if debug_request:
            print("GROQ DEBUG REQUEST (credentials excluded):")
            print(json.dumps(request_payload, ensure_ascii=False, indent=2))

        try:
            response = httpx.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_payload,
                timeout=30.0,
            )
        except httpx.TimeoutException as error:
            raise RetryableLLMError("Groq request timed out.") from error
        except httpx.RequestError as error:
            raise RetryableLLMError(
                f"Groq transport request failed: {type(error).__name__}."
            ) from error

        if response.status_code in (408, 429) or response.status_code >= 500:
            raise RetryableLLMError(
                f"Groq request failed with HTTP {response.status_code}."
            )
        if response.status_code in (401, 403):
            raise RetryableLLMError(
                f"Groq request failed with HTTP {response.status_code}."
            )
        if response.is_error:
            raise NonRetryableLLMError(
                f"Groq request failed with HTTP {response.status_code}."
            )

        try:
            response_data = response.json()
            output_text = response_data["choices"][0]["message"]["content"]
            plan = QATestPlan.model_validate_json(output_text)
            return plan
        except Exception as error:
            raise NonRetryableLLMError(
                f"Groq returned an invalid QA test plan: "
                f"{type(error).__name__}: {error}"
            ) from error

    def create_discovery(self, task: str, target_url: str, page_snapshot: str) -> AIDiscoveryResult:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise NonRetryableLLMError("GROQ_API_KEY is not set.")
        schema = AIDiscoveryResult.model_json_schema()
        payload = {"model": os.getenv("GROQ_MODEL", self._model), "max_completion_tokens": 4096,
                   "messages": [{"role": "user", "content":
                       "Return grounded structured discovery candidates only. Never return code, "
                       "instructions to execute, or perform browser actions. "
                       f"URL: {target_url}\nTask: {task}\nPage info: {page_snapshot}"}],
                   "response_format": {"type": "json_schema", "json_schema":
                       {"name": "ai_discovery_result", "strict": True, "schema": schema}}}
        try:
            response = httpx.post(self._endpoint,
                headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30.0)
        except httpx.TimeoutException as error:
            raise RetryableLLMError("Groq request timed out.") from error
        except httpx.RequestError as error:
            raise RetryableLLMError(f"Groq transport failure: {type(error).__name__}.") from error
        if response.status_code in (408, 429) or response.status_code >= 500:
            raise RetryableLLMError(f"Groq request failed with HTTP {response.status_code}.")
        if response.status_code in (401, 403):
            raise RetryableLLMError(f"Groq request failed with HTTP {response.status_code}.")
        if response.is_error:
            raise NonRetryableLLMError(f"Groq request failed with HTTP {response.status_code}.")
        try:
            output = response.json()["choices"][0]["message"]["content"]
            return AIDiscoveryResult.model_validate_json(output)
        except Exception as error:
            raise NonRetryableLLMError(f"Groq returned invalid Discovery data: {error}") from error
