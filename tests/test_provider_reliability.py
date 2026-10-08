from __future__ import annotations

import json
import json as jsonlib
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import (
    AllProvidersFailedError,
    RetryableLLMError,
    category_for_error,
    failure_detail_for,
    provider_http_failure,
)
from qa_agent.llm.groq import GroqProvider
from qa_agent.llm.json_schema import normalize_strict_json_schema
from qa_agent.llm.openai_compatible import OpenAICompatibleProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.llm_usage import OP_AUTHOR_TESTCASE, llm_usage_scope
from qa_agent.provider_settings import ProviderSettingsRepository, ProviderSettingsService
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_authoring import _AuthoringResponse, _parse_authoring_response
from qa_agent.web import LocalWebApplication


_CAPABILITY_RESPONSE = json.dumps({
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


class MemorySecrets:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        self.values.pop(name, None)


def _strict_schema_issues(value):
    issues = []
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            required = set(value.get("required", []))
            missing = set(properties) - required
            if missing:
                issues.append(f"missing required fields: {sorted(missing)}")
            if value.get("additionalProperties") is not False:
                issues.append("object permits additional properties")
        for child in value.values():
            issues.extend(_strict_schema_issues(child))
    elif isinstance(value, list):
        for child in value:
            issues.extend(_strict_schema_issues(child))
    return issues


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


def test_authoring_schema_is_embedded_for_groq_json_object_mode():
    original = _AuthoringResponse.model_json_schema()
    assert "name" not in original["required"]
    assert _strict_schema_issues(original)

    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        message = json["messages"][0]["content"]
        schema_text = message.rsplit("conforming to this JSON Schema:\n", 1)[-1]
        schema = jsonlib.loads(schema_text)
        if json["response_format"] != {"type": "json_object"} or _strict_schema_issues(schema):
            return SimpleNamespace(
                is_error=True,
                status_code=400,
                headers={},
                json=lambda: {"error": {
                    "code": "json_validate_failed",
                    "param": "response_format.type",
                    "message": "JSON Schema response format unsupported",
                }},
            )
        return SimpleNamespace(is_error=False, json=lambda: {
            "choices": [{"message": {"content": _CAPABILITY_RESPONSE}}],
        })

    provider = GroqProvider(api_key="fake-groq-key", model="test-model")
    with patch("qa_agent.llm.groq.httpx.post", side_effect=fake_post):
        output = provider.create_structured_output(
            "local fake prompt", original, "test_case_authoring_capability"
        )

    message = captured["payload"]["messages"][0]["content"]
    schema = json.loads(
        message.rsplit("conforming to this JSON Schema:\n", 1)[-1]
    )
    assert output == _CAPABILITY_RESPONSE
    assert captured["url"] == GroqProvider._endpoint
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["max_tokens"] == 4096
    assert "max_completion_tokens" not in captured["payload"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert schema == normalize_strict_json_schema(original)
    assert not _strict_schema_issues(schema)
    assert "name" in schema["required"]
    assert schema["properties"]["name"]["anyOf"][-1] == {"type": "null"}


def test_openrouter_authoring_uses_same_strict_schema_normalization():
    schema = _AuthoringResponse.model_json_schema()
    client = MagicMock()
    client.chat.completions.create.return_value = _response(_CAPABILITY_RESPONSE)
    provider = OpenAICompatibleProvider(
        "openrouter", "OPENROUTER_API_KEY", "vendor/test-model",
        "https://router.invalid/v1", api_key="fake-openrouter-key",
    )
    with patch("qa_agent.llm.openai_compatible.OpenAI", return_value=client) as openai:
        output = provider.create_structured_output(
            "local fake prompt", schema, "test_case_authoring_capability"
        )

    assert output == _CAPABILITY_RESPONSE
    assert openai.call_args.kwargs["api_key"] == "fake-openrouter-key"
    assert openai.call_args.kwargs["base_url"] == "https://router.invalid/v1"
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "vendor/test-model"
    assert request["response_format"]["json_schema"]["strict"] is True
    normalized = request["response_format"]["json_schema"]["schema"]
    assert normalized == normalize_strict_json_schema(schema)
    assert not _strict_schema_issues(normalized)


class _FakeStatusError(RuntimeError):
    def __init__(self):
        super().__init__("raw body contains key fake-key, scenario, and response")
        self.status_code = 400
        self.body = {"error": {
            "code": "invalid_json_schema",
            "type": "invalid_request_error",
            "param": "response_format.json_schema.schema.properties.segments",
            "message": "strict json_schema rejected",
        }}
        self.response = SimpleNamespace(status_code=400, headers={})


def test_connection_can_pass_while_authoring_capability_fails_with_same_model_url_and_key():
    secrets = MemorySecrets({"openrouter": "fake-openrouter-key"})
    environment = {
        "LLM_PROVIDER_ORDER": "openrouter,groq,gemini,openai",
        "OPENROUTER_BASE_URL": "https://router.invalid/v1",
    }
    settings = ProviderSettingsService(
        ProviderSettingsRepository(":memory:"), secrets, environment=environment,
    )
    settings.update("openrouter", model="configured-openrouter-model")
    client = MagicMock()
    request_schemas = []

    def create(**kwargs):
        response_format = kwargs["response_format"]["json_schema"]
        request_schemas.append((kwargs["model"], response_format["name"], response_format["schema"]))
        if response_format["name"] == "connection_test":
            return _response('{"ok":true}')
        raise _FakeStatusError()

    client.chat.completions.create.side_effect = create
    with patch("qa_agent.llm.openai_compatible.OpenAI", return_value=client) as openai:
        connection = settings.test_connection("openrouter")
        capability = settings.test_authoring_capability("openrouter")

    assert connection.status == "connected"
    assert capability.status == "failed"
    assert capability.category == "SCHEMA_ERROR"
    assert capability.http_status == 400
    assert capability.provider_error_code == "invalid_json_schema"
    assert capability.provider_error_type == "invalid_request_error"
    assert capability.provider_error_field == "response_format.json_schema.schema.properties.segments"
    assert [item[0] for item in request_schemas] == [
        "configured-openrouter-model", "configured-openrouter-model",
    ]
    assert [item[1] for item in request_schemas] == [
        "connection_test", "test_case_authoring_capability",
    ]
    assert request_schemas[1][2] == normalize_strict_json_schema(
        _AuthoringResponse.model_json_schema()
    )
    assert len(openai.call_args_list) == 2
    for call in openai.call_args_list:
        assert call.kwargs["api_key"] == "fake-openrouter-key"
        assert call.kwargs["base_url"] == "https://router.invalid/v1"

    views = {view.id: view for view in settings.provider_views()}
    assert views["openrouter"].health == "connected"
    assert views["openrouter"].capability_status == "failed"
    assert views["openrouter"].capability_provider_error_code == "invalid_json_schema"

    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository()),
        provider_settings=settings,
        provider_router=settings.create_router(),
    )
    try:
        html = app.handle("GET", "/settings/providers").body.decode()
    finally:
        app.close()
    assert "Connection:</strong> Healthy" in html
    assert "Authoring capability:</strong> Failed · Invalid structured output" in html
    assert "fake-openrouter-key" not in html
    assert "Provider code invalid_json_schema" in html
    assert "Field response_format.json_schema.schema.properties.segments" in html
    assert "strict json_schema rejected" not in html
    assert request_schemas[1][1] == "test_case_authoring_capability"


@pytest.mark.parametrize(("status", "payload", "expected"), [
    (401, {}, "AUTH_ERROR"),
    (404, {"error": {"code": "model_not_found", "message": "model not found"}}, "MODEL_NOT_FOUND"),
    (400, {"error": {"message": "invalid request"}}, "INVALID_REQUEST"),
    (429, {}, "RATE_LIMIT"),
    (400, {"error": {"code": "invalid_json_schema", "message": "schema rejected"}}, "SCHEMA_ERROR"),
    (400, {"error": {"code": "json_validate_failed"}}, "SCHEMA_ERROR"),
    (503, {}, "PROVIDER_UNAVAILABLE"),
])
def test_http_provider_failure_categories_are_safe(status, payload, expected):
    error = provider_http_failure("Groq", status, payload=payload)
    assert error.category == expected
    assert error.http_status == status
    assert expected not in str(payload) or expected == error.category
    assert "raw" not in str(error).casefold()


def test_provider_http_failure_retains_only_sanitized_code_and_field_hints():
    error = provider_http_failure(
        "Groq",
        400,
        payload={
            "error": {
                "code": "unsupported_parameter",
                "type": "invalid_request_error",
                "param": "max_completion_tokens",
                "message": "private prompt and gsk_private-secret-value",
            },
        },
    )
    detail = failure_detail_for("Groq", error)
    public = detail.to_public_dict()

    assert detail.category == "INVALID_REQUEST"
    assert detail.http_status == 400
    assert detail.provider_error_code == "unsupported_parameter"
    assert detail.provider_error_type == "invalid_request_error"
    assert detail.provider_error_field == "max_completion_tokens"
    assert "private prompt" not in json.dumps(public)
    assert "gsk_private-secret-value" not in json.dumps(public)

    unsafe = provider_http_failure(
        "Groq",
        400,
        payload={"error": {
            "code": "gsk_private-secret-value",
            "param": "gsk_private-secret-value",
            "message": "private prompt",
        }},
    )
    unsafe_detail = failure_detail_for("Groq", unsafe).to_public_dict()
    assert unsafe_detail["provider_error_code"] is None
    assert unsafe_detail["provider_error_field"] is None
    assert "private prompt" not in json.dumps(unsafe_detail)


def test_retry_after_and_invalid_response_are_classified_without_raw_body():
    limited = provider_http_failure(
        "Gemini", 429, payload={"error": {"message": "private provider body"}},
        headers={"Retry-After": "17"},
    )
    assert limited.retry_after_seconds == 17
    assert limited.safe_detail == "Rate limited"
    assert "private provider body" not in str(limited)
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=45)
    dated_retry = provider_http_failure(
        "Groq", 429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)}
    )
    assert dated_retry.retry_after_seconds is not None
    assert 1 <= dated_retry.retry_after_seconds <= 46

    with pytest.raises(RetryableLLMError) as failure:
        _parse_authoring_response('{"raw":"provider output"}')
    assert category_for_error(failure.value) == "INVALID_RESPONSE"
    detail = failure_detail_for("OpenRouter", failure.value)
    assert detail.safe_detail == "Response did not match the TestCase schema"
    assert "provider output" not in json.dumps(detail.to_public_dict())


class _FailingProvider(LLMProvider):
    def __init__(self, name, category, status, retry_after=None):
        self.name = name
        self.display_name = name
        self.model = "fake-model"
        self.category = category
        self.status = status
        self.retry_after = retry_after

    @property
    def is_available(self):
        return True

    def create_test_plan(self, task, target_url, page_snapshot):
        raise NotImplementedError

    def create_structured_output(self, prompt, schema, schema_name):
        raise RetryableLLMError(
            "secret-key fake-key; private scenario; raw response body",
            category=self.category,
            http_status=self.status,
            retry_after_seconds=self.retry_after,
            safe_detail={
                "INVALID_REQUEST": "Invalid request",
                "RATE_LIMIT": "Rate limited",
            }[self.category],
        )


def test_fallback_and_all_provider_failure_preserve_only_safe_attempt_details():
    events = []
    router = LLMRouter([
        _FailingProvider("Groq", "INVALID_REQUEST", 400),
        _FailingProvider("Gemini", "RATE_LIMIT", 429, retry_after=20),
    ])
    with llm_usage_scope(operation_type=OP_AUTHOR_TESTCASE):
        with pytest.raises(AllProvidersFailedError) as failure:
            router.create_structured_output(
                "private scenario prompt", {"type": "object"},
                "test_case_authoring",
                progress_callback=lambda event, provider, **kwargs: events.append(
                    (event, provider, kwargs)
                ),
            )

    attempts = failure.value.attempts
    assert [(item.provider_name, item.category, item.http_status) for item in attempts] == [
        ("Groq", "INVALID_REQUEST", 400),
        ("Gemini", "RATE_LIMIT", 429),
    ]
    assert attempts[1].retry_after_seconds == 20
    assert events[2][0:2] == ("fallback", "Groq")
    assert events[2][2]["next_provider"] == "Gemini"
    safe_output = json.dumps([
        {"event": event, "provider": provider, **kwargs}
        for event, provider, kwargs in events
    ]) + str(failure.value) + json.dumps([item.to_public_dict() for item in attempts])
    for secret in ("fake-key", "private scenario", "raw response body"):
        assert secret not in safe_output
    assert "INVALID_REQUEST" in safe_output
    assert "RATE_LIMIT" in safe_output


def test_unconfigured_provider_skips_do_not_create_usage_failures():
    from qa_agent.llm_usage import LLMUsageRepository, LLMUsageService

    usage = LLMUsageService(LLMUsageRepository(":memory:"))
    settings = ProviderSettingsService(
        ProviderSettingsRepository(":memory:"), MemorySecrets(),
        environment={"LLM_PROVIDER_ORDER": "groq,openrouter,gemini,openai"},
        usage_recorder=usage,
    )
    router = settings.create_router()
    with pytest.raises(RuntimeError, match="No configured LLM providers"):
        router.create_structured_output("no providers configured", {}, "test")
    summary = usage.analytics("30d")
    assert summary["requests"] == 0
    assert summary["failed_requests"] == 0


def test_authoring_capability_uses_parser_without_saving_a_testcase():
    class Provider(LLMProvider):
        name = "Groq"
        model = "fake-model"
        is_available = True

        def create_test_plan(self, task, target_url, page_snapshot):
            raise NotImplementedError

        def create_structured_output(self, prompt, schema, schema_name):
            assert schema_name == "test_case_authoring_capability"
            assert schema == _AuthoringResponse.model_json_schema()
            return _CAPABILITY_RESPONSE

    from qa_agent.test_case_authoring import TestCaseAuthoringService

    assert TestCaseAuthoringService(LLMRouter([Provider()])).test_authoring_capability() is None
