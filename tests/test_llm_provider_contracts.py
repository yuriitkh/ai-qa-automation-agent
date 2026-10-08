import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import httpx

from qa_agent.llm.gemini import GeminiProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.groq import GroqProvider
from qa_agent.llm.json_schema import qa_test_plan_schema
from qa_agent.models import QATestPlan


class LLMProviderContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = "Check https://example.com"
        self.target_url = "https://example.com"
        self.snapshot = '{"url":"https://example.com","headings":[]}'
        self.plan = QATestPlan(
            url=self.target_url,
            steps=[
                {"action": "navigate", "parameters": {"url": self.target_url}},
                {"action": "assert_page_loaded", "parameters": {}},
                {"action": "assert_title", "parameters": {"expected": "Example"}},
                {
                    "action": "assert_visible",
                    "parameters": {"selector": "h1", "expected_text": "Example"},
                },
                {"action": "click", "parameters": {"selector": "#submit"}},
                {"action": "check", "parameters": {"selector": "#terms"}},
                {"action": "uncheck", "parameters": {"selector": "#terms"}},
                {
                    "action": "fill",
                    "parameters": {"selector": "#name", "value": "Ada"},
                },
                {"action": "assert_hidden", "parameters": {"selector": "#notice"}},
                {"action": "assert_unchecked", "parameters": {"selector": "#terms"}},
                {"action": "assert_url", "parameters": {"expected": "https://example.com/lan"}},
                {"action": "assert_disabled", "parameters": {"selector": "input"}},
                {"action": "select_option", "parameters": {
                    "selector": "select", "option_label": "Two"
                }},
            ],
        )

    def _assert_prompt_contract(self, prompt: str) -> None:
        for action in (
            "navigate",
            "assert_page_loaded",
            "assert_title",
            "assert_visible",
            "click",
            "check",
            "uncheck",
            "fill",
            "assert_hidden",
            "assert_url",
        ):
            self.assertIn(action, prompt)

        for parameter_contract in (
            "{url: target URL}",
            "{expected: expected page title}",
            "assert_visible requires selector and may include expected_text",
            "click uses parameters {selector: CSS selector}",
            "{selector: CSS selector, value: text to fill}",
            "assert_hidden uses parameters {selector: CSS selector}",
            "assert_url uses parameters {expected: expected current URL}",
        ):
            self.assertIn(parameter_contract, prompt)

        self.assertIn("Never invent, rename, normalize, or silently correct", prompt)
        self.assertIn("check and uncheck", prompt)
        self.assertIn("assert_checked", prompt)
        self.assertIn("assert_unchecked", prompt)
        self.assertIn(
            "For every selector parameter in click, check, uncheck, fill, assert_hidden, or assert_visible",
            prompt,
        )
        self.assertIn("Do not use text instead of expected_text", prompt)
        self.assertIn("locator instead of selector", prompt)
        self.assertIn("navigation_paths records", prompt)
        self.assertIn("direct_navigation_paths", prompt)
        self.assertIn("steps selectors in array order", prompt)
        self.assertIn("If direct_navigation_paths is empty, do not invent a path", prompt)
        self.assertIn("menu_button_selector", prompt)
        self.assertIn("expected_url", prompt)
        self.assertIn("heading_selector", prompt)
        self.assertIn("at least three distinct records", prompt)
        self.assertIn("Never select menu items by their order", prompt)
        self.assertIn('assert_url parameter name MUST be exactly "expected"', prompt)
        self.assertIn("assert_selected", prompt)
        self.assertIn("radio", prompt)
        self.assertIn("never invent an expected URL", prompt)
        self.assertIn("multiple ordered executable actions", prompt)

    def test_gemini_prompt_and_plan_cover_all_actions(self) -> None:
        client = MagicMock()
        client.interactions.create.return_value.output_text = self.plan.model_dump_json()
        provider = GeminiProvider()
        provider._client = client

        plan = provider.create_test_plan(self.task, self.target_url, self.snapshot)

        self.assertEqual(plan, self.plan)
        call = client.interactions.create.call_args.kwargs
        self._assert_prompt_contract(call["input"])
        self.assertIn("assert_page_loaded uses parameters {}", call["input"])
        self.assertEqual(
            call["response_format"]["schema"], qa_test_plan_schema()
        )

    def test_gemini_interactions_retry_config_disables_sdk_retries(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            provider = GeminiProvider()

        self.assertTrue(provider.is_available)
        client = provider._client
        self.assertEqual(client._api_client._http_options.retry_options.attempts, 0)
        interactions = client.interactions
        retry_config = interactions.sdk_configuration.retry_config
        self.assertEqual(retry_config.max_retries, 0)

    def test_gemini_transient_http_statuses_are_retryable(self) -> None:
        for status_code in (408, 429, 500, 503):
            error = RuntimeError(f"HTTP {status_code}")
            error.status_code = status_code
            client = MagicMock()
            client.interactions.create.side_effect = error
            provider = GeminiProvider()
            provider._client = client

            with self.subTest(status_code=status_code), self.assertRaises(RetryableLLMError):
                provider.create_test_plan(self.task, self.target_url, self.snapshot)

    def test_gemini_429_uses_one_sdk_request_without_sleep(self) -> None:
        requests: list[httpx.Request] = []

        def return_rate_limit(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                429,
                json={
                    "error": {
                        "code": 429,
                        "message": "Rate limit exceeded",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                },
                request=request,
            )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            provider = GeminiProvider()

        client = provider._client
        api_client = client._api_client
        original_http_client = api_client._httpx_client
        mock_http_client = httpx.Client(
            transport=httpx.MockTransport(return_rate_limit)
        )
        api_client._httpx_client = mock_http_client
        try:
            with (
                patch("google.genai._gaos.utils.retries.time.sleep") as sdk_sleep,
                self.assertRaises(RetryableLLMError),
            ):
                provider.create_test_plan(self.task, self.target_url, self.snapshot)
        finally:
            mock_http_client.close()
            original_http_client.close()

        self.assertEqual(len(requests), 1)
        sdk_sleep.assert_not_called()

    def test_gemini_non_transient_http_status_is_retryable(self) -> None:
        error = RuntimeError("HTTP 400")
        error.status_code = 400
        provider = GeminiProvider()
        provider._client = MagicMock()
        provider._client.interactions.create.side_effect = error

        with self.assertRaises(RetryableLLMError):
            provider.create_test_plan(self.task, self.target_url, self.snapshot)

    def test_gemini_discovery_http_429_is_retryable(self) -> None:
        error = RuntimeError("HTTP 429")
        error.status_code = 429
        provider = GeminiProvider()
        provider._client = MagicMock()
        provider._client.interactions.create.side_effect = error

        with self.assertRaises(RetryableLLMError):
            provider.create_discovery(self.task, self.target_url, self.snapshot)

    def test_gemini_generic_structured_output_uses_supplied_schema(self) -> None:
        client = MagicMock()
        client.interactions.create.return_value.output_text = '{"ok":true}'
        provider = GeminiProvider()
        provider._client = client
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                  "required": ["ok"], "additionalProperties": False}

        result = provider.create_structured_output("authoring prompt", schema, "authoring")

        self.assertEqual(result, '{"ok":true}')
        call = client.interactions.create.call_args.kwargs
        self.assertEqual(call["input"], "authoring prompt")
        self.assertEqual(call["response_format"]["schema"], schema)
        self.assertEqual(call["response_format"]["mime_type"], "application/json")

    def test_groq_generic_structured_output_uses_strict_schema(self) -> None:
        response = MagicMock()
        response.is_error = False
        response.json.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                  "required": ["ok"], "additionalProperties": False}

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
            patch("qa_agent.llm.groq.httpx.post", return_value=response) as post,
        ):
            result = GroqProvider().create_structured_output(
                "authoring prompt", schema, "test_case_authoring"
            )

        self.assertEqual(result, '{"ok":true}')
        payload = post.call_args.kwargs["json"]
        self.assertIn("authoring prompt", payload["messages"][0]["content"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        prompt_schema = json.loads(
            payload["messages"][0]["content"].rsplit(
                "conforming to this JSON Schema:\n", 1
            )[-1]
        )
        self.assertEqual(prompt_schema, schema)

    def test_groq_prompt_and_schema_cover_all_actions(self) -> None:
        response = MagicMock()
        response.status_code = 200
        response.is_error = False
        response.json.return_value = {
            "choices": [
                {"message": {"content": self.plan.model_dump_json()}}
            ]
        }

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
            patch("qa_agent.llm.groq.httpx.post", return_value=response) as post,
            redirect_stdout(io.StringIO()),
        ):
            plan = GroqProvider().create_test_plan(
                self.task, self.target_url, self.snapshot
            )

        self.assertEqual(plan, self.plan)
        request = post.call_args.kwargs["json"]
        self._assert_prompt_contract(request["messages"][0]["content"])
        self.assertEqual(request["response_format"], {"type": "json_object"})
        schema = json.loads(
            request["messages"][0]["content"].rsplit(
                "conforming to this JSON Schema:\n", 1
            )[-1]
        )
        self.assertEqual(schema, GroqProvider._response_schema())
        schema_text = json.dumps(schema)
        for unsupported in ("$ref", "$defs", "oneOf"):
            self.assertNotIn(unsupported, schema_text)
        variants = schema["properties"]["steps"]["items"]["anyOf"]
        by_action = {
            variant["properties"]["action"]["enum"][0]: variant["properties"]["parameters"]
            for variant in variants
        }
        common_fields = ["url", "expected", "selector", "expected_text", "value"]
        self.assertEqual(by_action["navigate"]["required"], common_fields)
        self.assertEqual(by_action["assert_page_loaded"]["required"], common_fields)
        self.assertEqual(by_action["assert_disabled"]["required"], common_fields)
        for action, parameters_schema in by_action.items():
            self.assertEqual(
                list(parameters_schema["properties"]),
                common_fields + (["option_label"] if action == "select_option" else []),
            )
            if action != "select_option":
                self.assertNotIn("option_label", parameters_schema["properties"])
        self.assertEqual(
            by_action["select_option"]["required"], common_fields + ["option_label"]
        )
        self.assertEqual(by_action["select_option"]["properties"]["option_label"]["type"], "string")

        def assert_groq_strict_objects(node: object) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIn("properties", node)
                    self.assertTrue(node["properties"])
                    self.assertIn("required", node)
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                    self.assertIs(node.get("additionalProperties"), False)
                for value in node.values():
                    assert_groq_strict_objects(value)
            elif isinstance(node, list):
                for value in node:
                    assert_groq_strict_objects(value)

        assert_groq_strict_objects(schema)
        self.assertEqual(schema["required"], ["url", "steps"])
        self.assertFalse(schema["additionalProperties"])
        prompt = request["messages"][0]["content"]
        self.assertIn("Output exactly one JSON object", prompt)
        self.assertIn("The JSON root MUST be an object, never an array", prompt)
        self.assertIn("fields MUST be exactly url and steps", prompt)
        self.assertIn("one object containing action and parameters together", prompt)
        self.assertIn("Output no Markdown", prompt)
        self.assertIn("assert_page_loaded uses all five common keys set to null", prompt)
        self.assertIn("Include option_label only for select_option", prompt)
        self.assertIn("set fields unused by an action to JSON null", prompt)

    def test_groq_rejects_root_array_without_rewriting_it(self) -> None:
        array_response = json.dumps([self.plan.model_dump()])
        response = MagicMock()
        response.status_code = 200
        response.is_error = False
        response.json.return_value = {
            "choices": [{"message": {"content": array_response}}]
        }

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
            patch("qa_agent.llm.groq.httpx.post", return_value=response),
            redirect_stdout(io.StringIO()),
            self.assertRaises(RetryableLLMError),
        ):
            GroqProvider().create_test_plan(
                self.task, self.target_url, self.snapshot
            )

    def test_groq_transient_http_statuses_are_retryable(self) -> None:
        for status_code in (408, 429, 500, 503):
            response = MagicMock()
            response.status_code = status_code
            with (
                self.subTest(status_code=status_code),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
                patch("qa_agent.llm.groq.httpx.post", return_value=response),
                self.assertRaises(RetryableLLMError),
            ):
                GroqProvider().create_test_plan(
                    self.task, self.target_url, self.snapshot
                )

    def test_groq_timeout_is_retryable(self) -> None:
        import httpx

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
            patch(
                "qa_agent.llm.groq.httpx.post",
                side_effect=httpx.TimeoutException("timed out"),
            ),
            self.assertRaises(RetryableLLMError),
        ):
            GroqProvider().create_test_plan(
                self.task, self.target_url, self.snapshot
            )

    def test_provider_does_not_silently_rewrite_wrong_parameter_names(self) -> None:
        invalid_parameters_plan = {
            "url": self.target_url,
            "steps": [
                {
                    "action": "assert_title",
                    "parameters": {"expected_title": "Example"},
                },
                {
                    "action": "assert_visible",
                    "parameters": {"locator": "h1", "text": "Example"},
                },
                {
                    "action": "assert_url",
                    "parameters": {"expected_url": "https://example.com/lan"},
                },
            ],
        }
        response_text = json.dumps(invalid_parameters_plan)

        gemini_client = MagicMock()
        gemini_client.interactions.create.return_value.output_text = response_text
        gemini = GeminiProvider()
        gemini._client = gemini_client
        gemini_plan = gemini.create_test_plan(
            self.task, self.target_url, self.snapshot
        )

        groq_response = MagicMock()
        groq_response.status_code = 200
        groq_response.is_error = False
        groq_response.json.return_value = {
            "choices": [{"message": {"content": response_text}}]
        }
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
            patch("qa_agent.llm.groq.httpx.post", return_value=groq_response),
            redirect_stdout(io.StringIO()),
        ):
            groq_plan = GroqProvider().create_test_plan(
                self.task, self.target_url, self.snapshot
            )

        for plan in (gemini_plan, groq_plan):
            title_parameters = plan.steps[0].parameters
            visible_parameters = plan.steps[1].parameters
            url_parameters = plan.steps[2].parameters
            self.assertEqual(title_parameters, {"expected_title": "Example"})
            self.assertNotIn("expected", title_parameters)
            self.assertEqual(visible_parameters, {"locator": "h1", "text": "Example"})
            self.assertNotIn("selector", visible_parameters)
            self.assertNotIn("expected_text", visible_parameters)
            self.assertEqual(
                url_parameters, {"expected_url": "https://example.com/lan"}
            )
            self.assertNotIn("expected", url_parameters)

    def test_openai_compatible_wraps_sdk_failures_into_llm_errors(self) -> None:
        # P2-3 characterization: OpenAICompatibleProvider converts underlying
        # SDK/client failures into the project's typed LLM exceptions. Typed
        # errors survive create_test_plan's re-raise untouched, the original
        # exception is preserved only via __cause__, the exposed message
        # carries just the HTTP status or type name (raw reason redacted), and
        # a missing API key is classified non-retryable without leaking it.
        import openai

        from qa_agent.llm.errors import NonRetryableLLMError
        from qa_agent.llm.openai_compatible import OpenAICompatibleProvider

        secret = "sk-unit-test-secret-value"
        request = httpx.Request(
            "POST", "https://api.example.com/v1/chat/completions"
        )
        response_429 = httpx.Response(429, request=request, json={})
        provider = OpenAICompatibleProvider(
            "openai-compat", "TEST_COMPAT_API_KEY", "test-model"
        )

        sdk_connection_error = openai.APIConnectionError(
            message=f"connection reset by key={secret}", request=request
        )
        sdk_rate_limit_error = openai.RateLimitError(
            f"rate limited key={secret}", response=response_429, body=None
        )
        timeout_error = httpx.TimeoutException(f"slow network {secret}")
        transport_error = httpx.ConnectError(f"connection refused {secret}")
        unexpected_error = RuntimeError(f"unexpected client bug {secret}")

        cases = [
            (
                "sdk_connection_error",
                sdk_connection_error,
                "openai-compat: request failed (APIConnectionError).",
            ),
            (
                "sdk_rate_limit_error",
                sdk_rate_limit_error,
                "openai-compat: request failed (HTTP 429).",
            ),
            (
                "timeout",
                timeout_error,
                "openai-compat: request timed out.",
            ),
            (
                "transport",
                transport_error,
                "openai-compat: transport failed (ConnectError).",
            ),
            (
                "non_sdk_exception",
                unexpected_error,
                "openai-compat: invalid response (RuntimeError).",
            ),
        ]

        for label, underlying, expected_message in cases:
            with self.subTest(label):
                with (
                    patch.dict(
                        os.environ, {"TEST_COMPAT_API_KEY": secret}, clear=True
                    ),
                    patch(
                        "qa_agent.llm.openai_compatible.OpenAI"
                    ) as openai_client,
                ):
                    openai_client.return_value.chat.completions.create.side_effect = (
                        underlying
                    )
                    with self.assertRaises(RetryableLLMError) as raised:
                        provider.create_test_plan(
                            self.task, self.target_url, self.snapshot
                        )

                error = raised.exception
                self.assertEqual(str(error), expected_message)
                # Underlying exception preserved only through the cause chain.
                self.assertIs(error.__cause__, underlying)
                # Raw failure reason (and secret) never reaches the message.
                self.assertNotIn(secret, str(error))
                self.assertNotIn(str(underlying), str(error))

        # Invalid payload from a healthy client is also a typed retryable error.
        with (
            patch.dict(os.environ, {"TEST_COMPAT_API_KEY": secret}, clear=True),
            patch("qa_agent.llm.openai_compatible.OpenAI") as openai_client,
        ):
            bad_response = MagicMock()
            bad_response.choices[0].message.content = "not valid json"
            openai_client.return_value.chat.completions.create.return_value = (
                bad_response
            )
            with self.assertRaises(RetryableLLMError) as raised:
                provider.create_test_plan(
                    self.task, self.target_url, self.snapshot
                )
        self.assertEqual(
            str(raised.exception),
            "openai-compat: returned an invalid QA test plan.",
        )
        self.assertIsNotNone(raised.exception.__cause__)

        # Missing API key: non-retryable, env var named but never echoed.
        with patch.dict(os.environ, {"TEST_COMPAT_API_KEY": "  "}, clear=True):
            with self.assertRaises(NonRetryableLLMError) as raised:
                provider.create_test_plan(
                    self.task, self.target_url, self.snapshot
                )
        self.assertEqual(
            str(raised.exception),
            "openai-compat: TEST_COMPAT_API_KEY is not configured.",
        )
        self.assertNotIn(secret, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
