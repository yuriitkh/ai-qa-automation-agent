from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from qa_agent.llm.errors import AllProvidersFailedError, RetryableLLMError, failure_detail_for
from qa_agent.llm.groq import GroqProvider
from qa_agent.llm.json_schema import normalize_strict_json_schema
from qa_agent.llm.openai_compatible import OpenAICompatibleProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.llm_usage import (
    LLMUsageRepository,
    LLMUsageService,
    OP_AUTHOR_TESTCASE,
    REQUEST_FAILED,
    llm_usage_scope,
)
from qa_agent.provider_settings import ProviderSettingsRepository, ProviderSettingsService
from qa_agent.test_case_authoring import (
    TestCaseAuthoringError,
    TestCaseAuthoringService,
    _AuthoringResponse,
)


_AUTHORING_RESPONSE = json.dumps({
    "name": "Local registration",
    "preconditions": [],
    "segments": [{
        "steps": [{
            "name": "Open registration",
            "description": "Open the registration page",
            "expected": "The registration form is visible",
        }],
    }],
})
_PLAN_RESPONSE = json.dumps({
    "url": "https://example.com",
    "steps": [{"action": "navigate", "parameters": {
        "url": "https://example.com", "expected": None, "selector": None,
        "expected_text": None, "value": None,
    }}],
})
_DISCOVERY_RESPONSE = json.dumps({
    "navigation_paths": [],
    "direct_navigation_paths": [],
    "interactive_elements": [],
    "warnings": [],
})
_PROMPT_SCHEMA_MARKER = "conforming to this JSON Schema:\n"


def _schema_from_groq_prompt(payload: dict) -> dict:
    content = payload["messages"][0]["content"]
    _, schema_text = content.rsplit(_PROMPT_SCHEMA_MARKER, 1)
    return json.loads(schema_text)


def _schema_issues(value):
    issues = []
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            if not set(properties).issubset(value.get("required", [])):
                issues.append("declared properties are not required")
            if value.get("additionalProperties") is not False:
                issues.append("object is not closed")
        for child in value.values():
            issues.extend(_schema_issues(child))
    elif isinstance(value, list):
        for child in value:
            issues.extend(_schema_issues(child))
    return issues


def _fake_groq_post(
    captured: list[dict], *, reject: bool = False, response_content: str | None = None
):
    def post(url, *, headers, json, timeout):
        captured.append({"url": url, "headers": headers, "payload": json, "timeout": timeout})
        if "max_completion_tokens" in json:
            return httpx.Response(400, json={"error": {
                "code": "unsupported_parameter",
                "type": "invalid_request_error",
                "param": "max_completion_tokens",
                "message": "Raw provider body must never be logged.",
            }})
        if "max_tokens" not in json:
            return httpx.Response(400, json={"error": {
                "code": "missing_parameter",
                "type": "invalid_request_error",
                "param": "max_tokens",
                "message": "Raw provider body must never be logged.",
            }})
        if reject:
            return httpx.Response(400, json={"error": {
                "code": "invalid_request",
                "type": "invalid_request_error",
                "param": "messages",
                "message": "private prompt and fake-key must not escape",
            }})

        request_format = json["response_format"]
        if request_format != {"type": "json_object"}:
            return httpx.Response(400, json={"error": {
                "code": "json_validate_failed",
                "type": "invalid_request_error",
                "param": "response_format.type",
                "message": "json_schema response format is unsupported for this model.",
            }})

        try:
            response_schema = _schema_from_groq_prompt(json)
        except (IndexError, KeyError, TypeError, ValueError):
            response_schema = None
        if response_schema is None or _schema_issues(response_schema):
            return httpx.Response(400, json={"error": {
                "code": "invalid_schema_instruction",
                "type": "invalid_request_error",
                "param": "messages",
                "message": "The prompt did not contain a closed required schema.",
            }})

        properties = response_schema.get("properties", {})
        content = response_content or (
            '{"ok":true}' if set(properties) == {"ok"}
            else _PLAN_RESPONSE if set(properties) == {"url", "steps"}
            else _DISCOVERY_RESPONSE if "navigation_paths" in properties
            else _AUTHORING_RESPONSE
        )
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 31, "completion_tokens": 17, "total_tokens": 48},
        })

    return post


class _LegacyGroqProvider(GroqProvider):
    """Reproduce the pre-fix Groq output-token request field."""

    def _structured_request_payload(self, prompt, schema, schema_name, max_tokens):
        payload = super()._structured_request_payload(prompt, schema, schema_name, max_tokens)
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        return payload


class _LegacyJsonSchemaModeGroqProvider(GroqProvider):
    """Reproduce the rejected Chat Completions JSON Schema response mode."""

    def _structured_request_payload(self, prompt, schema, schema_name, max_tokens):
        payload = super()._structured_request_payload(prompt, schema, schema_name, max_tokens)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": normalize_strict_json_schema(schema),
            },
        }
        return payload


class _MemorySecrets:
    def __init__(self, values):
        self.values = dict(values)

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        self.values.pop(name, None)


def test_pre_fix_groq_token_field_is_rejected_by_fake_groq():
    captured = []
    provider = _LegacyGroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")

    with (
        patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured)),
        pytest.raises(RetryableLLMError) as failure,
    ):
        provider.create_structured_output(
            "private prompt", {"type": "object", "properties": {}, "required": [],
                               "additionalProperties": False}, "legacy_test"
        )

    assert failure.value.category == "INVALID_REQUEST"
    assert failure.value.http_status == 400
    detail = failure_detail_for("Groq", failure.value)
    assert detail.provider_error_field == "max_completion_tokens"
    assert "private prompt" not in str(failure.value)
    assert "Raw provider body" not in str(failure.value)
    assert captured[0]["payload"]["max_completion_tokens"] == 4096


def test_json_schema_response_format_is_rejected_by_fake_groq_model():
    captured = []
    provider = _LegacyJsonSchemaModeGroqProvider(
        api_key="fake-groq-key", model="openai/gpt-oss-20b"
    )

    with (
        patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured)),
        pytest.raises(RetryableLLMError) as failure,
    ):
        provider.create_structured_output(
            "private prompt", {"type": "object", "properties": {"ok": {"type": "boolean"}},
                               "required": ["ok"], "additionalProperties": False}, "legacy_strict"
        )

    detail = failure_detail_for("Groq", failure.value)
    assert (detail.http_status, detail.provider_error_code) == (400, "json_validate_failed")
    assert detail.provider_error_field == "response_format.type"
    assert detail.category == "SCHEMA_ERROR"
    assert "private prompt" not in str(failure.value)


def test_groq_connection_and_authoring_capability_use_valid_json_schema_requests():
    captured = []
    settings = ProviderSettingsService(
        ProviderSettingsRepository(":memory:"),
        _MemorySecrets({"groq": "fake-groq-key"}),
        environment={"LLM_PROVIDER_ORDER": "groq"},
    )

    with patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured)):
        connection = settings.test_connection("groq")
        capability = settings.test_authoring_capability("groq")

    assert connection.status == "connected"
    assert capability.status == "passed"
    assert [call["payload"]["response_format"] for call in captured] == [
        {"type": "json_object"}, {"type": "json_object"},
    ]
    for call in captured:
        payload = call["payload"]
        assert payload["model"] == "openai/gpt-oss-20b"
        assert "max_tokens" in payload
        assert "max_completion_tokens" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        assert "json_schema" not in payload["response_format"]
        assert not _schema_issues(_schema_from_groq_prompt(payload))

    connection_schema = _schema_from_groq_prompt(captured[0]["payload"])
    assert connection_schema == {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    capability_schema = _schema_from_groq_prompt(captured[1]["payload"])
    original = _AuthoringResponse.model_json_schema()
    assert capability_schema == normalize_strict_json_schema(original)
    assert "name" in capability_schema["required"]
    assert capability_schema["properties"]["name"]["anyOf"][-1] == {"type": "null"}


def test_real_testcase_authoring_uses_the_groq_compatibility_path_and_telemetry():
    captured = []
    usage = LLMUsageService(LLMUsageRepository(":memory:"))
    provider = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")
    service = TestCaseAuthoringService(LLMRouter([provider], usage_recorder=usage))

    with patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured)):
        draft = service.generate(
            "Local registration", "Verify the registration form.", "https://example.com"
        )

    assert draft is not None
    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["response_format"] == {"type": "json_object"}
    assert "Output contract: test_case_authoring." in payload["messages"][0]["content"]
    assert _schema_from_groq_prompt(payload) == normalize_strict_json_schema(
        _AuthoringResponse.model_json_schema()
    )
    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload
    records = usage.repository.list_records()
    assert len(records) == 1
    record = records[0]
    assert (record.provider_name, record.model, record.operation_type, record.request_status) == (
        "Groq", "openai/gpt-oss-20b", "AUTHOR_TESTCASE", "SUCCESS",
    )
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (31, 17, 48)


def test_groq_schema_compatibility_keeps_canonical_authoring_validation_strict():
    captured = []
    provider = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")
    service = TestCaseAuthoringService(LLMRouter([provider]))
    whitespace_step = json.dumps({
        "name": "Local registration",
        "preconditions": [],
        "segments": [{"steps": [{
            "name": " ",
            "description": "Open the registration page",
            "expected": "The registration form is visible",
        }]}],
    })

    with (
        patch(
            "qa_agent.llm.groq.httpx.post",
            side_effect=_fake_groq_post(captured, response_content=whitespace_step),
        ),
        pytest.raises(TestCaseAuthoringError) as failure,
    ):
        service.generate(
            "Local registration", "Verify the registration form.", "https://example.com"
        )

    assert failure.value.category == "AI_OUTPUT_VALIDATION_ERROR"
    assert captured
    sent_schema = _schema_from_groq_prompt(captured[0]["payload"])
    assert captured[0]["payload"]["response_format"] == {"type": "json_object"}
    assert not _schema_issues(sent_schema)


def test_groq_plan_generation_uses_same_compatible_payload_builder():
    captured = []
    provider = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")

    with patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured)):
        plan = provider.create_test_plan("Check example", "https://example.com", "{}")

    assert plan.url == "https://example.com"
    payload = captured[0]["payload"]
    assert payload["max_tokens"] == 8192
    assert "max_completion_tokens" not in payload
    assert payload["response_format"] == {"type": "json_object"}
    assert _schema_from_groq_prompt(payload) == normalize_strict_json_schema(
        GroqProvider._response_schema()
    )


def test_groq_discovery_uses_same_compatible_payload_builder():
    captured = []
    provider = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")

    with patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured)):
        result = provider.create_discovery("Check example", "https://example.com", "{}")

    assert result.navigation_paths == []
    payload = captured[0]["payload"]
    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload
    assert payload["response_format"] == {"type": "json_object"}


def test_genuinely_bad_groq_request_stays_retryable_and_does_not_log_raw_response(caplog):
    captured = []
    provider = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")

    with (
        patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured, reject=True)),
        caplog.at_level(logging.DEBUG),
        pytest.raises(RetryableLLMError) as failure,
    ):
        provider.create_structured_output(
            "private prompt", {"type": "object", "properties": {"ok": {"type": "boolean"}},
                               "required": ["ok"], "additionalProperties": False}, "bad_request"
        )

    detail = failure_detail_for("Groq", failure.value)
    assert (detail.category, detail.http_status) == ("INVALID_REQUEST", 400)
    assert detail.provider_error_field == "messages"
    assert "private prompt" not in caplog.text
    assert "fake-key" not in caplog.text
    assert "private prompt" not in str(failure.value)
    assert "Raw provider body" not in str(failure.value)


def test_failed_groq_attempt_telemetry_keeps_safe_status_without_fabricated_usage():
    captured = []
    usage = LLMUsageService(LLMUsageRepository(":memory:"))
    provider = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")
    router = LLMRouter([provider], usage_recorder=usage)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}

    with (
        patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured, reject=True)),
        llm_usage_scope(operation_type=OP_AUTHOR_TESTCASE),
        pytest.raises(AllProvidersFailedError),
    ):
        router.create_structured_output("private prompt", schema, "test_case_authoring")

    records = usage.repository.list_records()
    assert len(records) == 1
    record = records[0]
    assert (record.provider_name, record.model, record.operation_type, record.request_status) == (
        "Groq", "openai/gpt-oss-20b", OP_AUTHOR_TESTCASE, REQUEST_FAILED,
    )
    assert record.error_category == "INVALID_REQUEST"
    assert isinstance(record.latency_ms, int)
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.total_tokens is None


def test_groq_failure_falls_back_to_openrouter_without_changing_openrouter_request():
    captured = []
    groq = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")
    openrouter = OpenAICompatibleProvider(
        "openrouter", "OPENROUTER_API_KEY", "vendor/test-model",
        "https://router.invalid/v1", api_key="fake-openrouter-key",
    )
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        usage=None,
    )
    router = LLMRouter([groq, openrouter])
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}

    with (
        patch("qa_agent.llm.groq.httpx.post", side_effect=_fake_groq_post(captured, reject=True)),
        patch("qa_agent.llm.openai_compatible.OpenAI", return_value=client) as openai,
    ):
        result = router.create_structured_output("safe prompt", schema, "test")

    assert result == '{"ok":true}'
    assert router.selected_provider_name == "openrouter"
    groq_payload = captured[0]["payload"]
    assert groq_payload["max_tokens"] == 4096
    assert groq_payload["response_format"] == {"type": "json_object"}
    compatible_payload = client.chat.completions.create.call_args.kwargs
    assert compatible_payload["model"] == "vendor/test-model"
    assert compatible_payload["max_tokens"] == 4096
    assert compatible_payload["response_format"]["json_schema"]["strict"] is True
    openai.assert_called_once()


@pytest.mark.parametrize("invalid_json", ["[]", "not valid JSON"])
def test_groq_invalid_json_object_falls_back_to_openrouter(invalid_json):
    captured = []
    groq = GroqProvider(api_key="fake-groq-key", model="openai/gpt-oss-20b")
    openrouter = OpenAICompatibleProvider(
        "openrouter", "OPENROUTER_API_KEY", "vendor/test-model",
        "https://router.invalid/v1", api_key="fake-openrouter-key",
    )
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        usage=None,
    )
    router = LLMRouter([groq, openrouter])
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}

    with (
        patch(
            "qa_agent.llm.groq.httpx.post",
            side_effect=_fake_groq_post(captured, response_content=invalid_json),
        ),
        patch("qa_agent.llm.openai_compatible.OpenAI", return_value=client) as openai,
    ):
        result = router.create_structured_output("safe prompt", schema, "test")

    assert result == '{"ok":true}'
    assert router.selected_provider_name == "openrouter"
    assert captured[0]["payload"]["response_format"] == {"type": "json_object"}
    compatible_payload = client.chat.completions.create.call_args.kwargs
    assert compatible_payload["response_format"]["json_schema"]["strict"] is True
    openai.assert_called_once()


def test_custom_openai_compatible_provider_retains_its_existing_strict_schema_shape():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        usage=None,
    )
    custom = OpenAICompatibleProvider(
        "custom:local", "CUSTOM_LOCAL_API_KEY", "local-model",
        "https://local.invalid/v1", api_key="fake-custom-key",
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}

    with patch("qa_agent.llm.openai_compatible.OpenAI", return_value=client):
        result = custom.create_structured_output("local prompt", schema, "custom_check")

    assert result == '{"ok":true}'
    request = client.chat.completions.create.call_args.kwargs
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == normalize_strict_json_schema(schema)
