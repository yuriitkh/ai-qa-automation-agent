import os

import httpx
from google import genai
from google.genai import types

from ..models import AIDiscoveryResult, QATestPlan
from .base import LLMProvider
from .errors import NonRetryableLLMError, RetryableLLMError


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        self._api_key = os.environ.get("GEMINI_API_KEY")
        self._client = (
            genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                ),
            )
            if self._api_key
            else None
        )

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        if self._client is None:
            raise NonRetryableLLMError(
                "Gemini provider is unavailable: GEMINI_API_KEY is not configured."
            )

        try:
            response_format = {
                "type": "text",
                "mime_type": "application/json",
                "schema": QATestPlan.model_json_schema(),
            }
            interaction = self._client.interactions.create(
                model="gemini-3.8-flash",
                input=(
                    "Create a structured browser QA test plan for this task. "
                    "Use the supplied target URL exactly; do not infer another URL. "
                    "Return that URL and ordered executable test steps. "
                    "Allowed actions are ONLY: navigate, assert_page_loaded, "
                    "assert_title, assert_visible, click, fill, assert_hidden, "
                    "and assert_url. "
                    "Never invent, rename, or substitute action names. Use this "
                    "exact parameter contract: "
                    "navigate uses parameters {url: target URL}; "
                    "assert_page_loaded uses parameters {}; "
                    "assert_title uses parameters {expected: expected page title}. "
                    'The assert_title parameter name MUST be exactly "expected"; '
                    "never use expected_title. assert_visible uses parameters "
                    "{selector: CSS selector, expected_text: expected visible text}. "
                    "click uses parameters {selector: CSS selector}; "
                    "fill uses parameters {selector: CSS selector, value: text to fill}; "
                    "assert_hidden uses parameters {selector: CSS selector}. "
                    "assert_url uses parameters {expected: expected current URL}. "
                    'The assert_url parameter name MUST be exactly "expected". '
                    "Use only a URL supported by the task or browser context; "
                    "never invent an expected URL. "
                    "Parameter names are case-sensitive. Never invent, rename, "
                    "normalize, or silently correct action or parameter names. "
                    "Do not use text instead of expected_text or locator instead "
                    "of selector. Output only the parameters defined above. "
                    "Assertions MUST be based only on elements explicitly present "
                    "in the supplied page snapshot. For every selector parameter "
                    "in click, fill, assert_hidden, or assert_visible, use an exact "
                    "selector listed in visible_text_elements, headings, links, or "
                    "buttons. "
                    "Use visible_text_elements to ground ordinary visible-text "
                    "assertions. Use each selector exactly as provided in the "
                    "PAGE SNAPSHOT. Do not modify selectors. Preserve Unicode "
                    "characters exactly; do not convert them into literal escape "
                    "sequences. "
                    "Never invent selectors or page elements from prior "
                    "knowledge. If the requested text or element is absent from the "
                    "snapshot, do not invent a selector or assertion. "
                    "For navigation-menu tasks, use only the supplied "
                    "navigation_paths records. For each requested combination, "
                    "navigate to the task's root URL, click its menu_button_selector, "
                    "click menu_tab_selector when present, click menu_item_selector, "
                    "then click submenu_selector, assert_url with that record's "
                    "expected_url, and assert_visible with its heading_selector and "
                    "heading_text. Repeat this sequence for at least three distinct "
                    "records when the task requests three combinations. For direct-link "
                    "navigation, use only supplied direct_navigation_paths records. "
                    "For each direct record, navigate to root_url, click the steps "
                    "selectors in array order, assert_url with expected_url, and then "
                    "assert_visible with heading_selector and heading_text. Use only "
                    "the supplied selectors and URLs; never invent or infer a missing "
                    "step. If direct_navigation_paths is empty, do not invent a path. "
                    "If a cookie "
                    "banner is present, click its discovered strictly-necessary "
                    "consent selector before opening the menu. Never select menu "
                    "items by their order, numeric position, nth-child, or index. "
                    f"Target URL: {target_url}\n"
                    f"Page snapshot (JSON): {page_snapshot}\n"
                    f"Task: {task}"
                ),
                response_format=response_format,
            )
            return QATestPlan.model_validate_json(interaction.output_text)
        except (httpx.TimeoutException, TimeoutError) as error:
            raise RetryableLLMError(f"Gemini request timed out: {error}") from error
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code is None:
                status_code = getattr(error, "code", None)
            if status_code is None:
                response = getattr(error, "response", None)
                if response is None:
                    response = getattr(error, "raw_response", None)
                status_code = getattr(response, "status_code", None)

            if status_code == 408 or status_code == 429 or (
                isinstance(status_code, int) and status_code >= 500
            ):
                raise RetryableLLMError(
                    f"Gemini request failed with HTTP {status_code}: {error}"
                ) from error

            raise NonRetryableLLMError(
                f"Gemini request or response was not retryable: "
                f"{type(error).__name__}: {error}"
            ) from error

    def create_discovery(self, task: str, target_url: str, page_snapshot: str) -> AIDiscoveryResult:
        if self._client is None:
            raise NonRetryableLLMError("Gemini provider is unavailable.")
        try:
            response = self._client.interactions.create(
                model="gemini-3.8-flash",
                input=("Return only structured candidate navigation paths and interactive elements "
                       "grounded in the supplied page information. Never return code or actions. "
                       f"URL: {target_url}\nTask: {task}\nPage info: {page_snapshot}"),
                response_format={"type": "text", "mime_type": "application/json",
                                 "schema": AIDiscoveryResult.model_json_schema()},
            )
            return AIDiscoveryResult.model_validate_json(response.output_text)
        except (httpx.TimeoutException, TimeoutError) as error:
            raise RetryableLLMError(f"Gemini request timed out: {error}") from error
        except Exception as error:
            raise NonRetryableLLMError(f"Gemini Discovery response was invalid: {error}") from error
