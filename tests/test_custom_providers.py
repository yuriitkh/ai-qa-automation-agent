from __future__ import annotations

import sqlite3
from urllib.parse import urlencode, urlsplit

import pytest

from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.usage_metadata import capture_openai_usage
from qa_agent.llm_usage import (
    LLMUsageRepository,
    LLMUsageService,
    OP_AUTHOR_TESTCASE,
    llm_usage_scope,
)
from qa_agent.provider_settings import (
    ProviderSettingsRepository,
    ProviderSettingsService,
    validate_custom_base_url,
)
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.web import LocalWebApplication


class MemorySecrets:
    def __init__(self):
        self.values = {}

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        self.values.pop(name, None)


class FakeProvider:
    def __init__(self, name, key, calls, outcome=None, usage=None, model="fake-model"):
        self.name = name
        self.model = model
        self.key = key
        self.calls = calls
        self.outcome = outcome
        self.usage = usage

    @property
    def is_available(self):
        return bool(self.key) or self.name.startswith("custom_")

    def create_test_plan(self, task, target_url, page_snapshot):
        raise NotImplementedError

    def create_structured_output(self, prompt, schema, schema_name):
        self.calls.append(self.name)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if self.usage is not None:
            capture_openai_usage({"usage": self.usage})
        return '{"ok":true}'


class StatusFailure(RuntimeError):
    def __init__(self, status):
        super().__init__("safe local fake failure")
        self.status_code = status


@pytest.fixture
def provider_env():
    return {"LLM_PROVIDER_ORDER": "openai,gemini,openrouter,groq"}


@pytest.fixture
def setup_provider(tmp_path, provider_env):
    database = tmp_path / "settings.sqlite3"
    secrets = MemorySecrets()
    calls = []
    outcomes = {}

    def factory(name, key, model, timeout):
        return FakeProvider(name, key, calls, outcomes.get(name), model=model)

    service = ProviderSettingsService(
        ProviderSettingsRepository(database), secrets,
        environment=provider_env, provider_factory=factory,
    )
    return database, secrets, calls, outcomes, service, factory, provider_env


def test_compact_builtin_cards_have_edit_controls_collapsed(setup_provider):
    database, _, _, _, service, _, _ = setup_provider
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), provider_settings=service)
    try:
        html = app.handle("GET", "/settings/providers").body.decode()
    finally:
        app.close()
    assert "Priority 1" in html and "Groq" in html
    assert "Configured" in html and "Credential:" in html and "Model:" in html
    assert '<details class="provider-edit">' in html
    assert "Test connection" in html and "Move" in html
    assert "Settings sections" in html


def test_model_and_key_fields_are_inside_collapsed_edit_details(setup_provider):
    _, secrets, _, _, service, _, _ = setup_provider
    secrets.set("groq", "masked-secret-ABC9")
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), provider_settings=service)
    try:
        html = app.handle("GET", "/settings/providers").body.decode()
    finally:
        app.close()
    card = html.split('id="groq"', 1)[1].split("</section>", 1)[0]
    edit = card.split('<details class="provider-edit">', 1)[1]
    assert 'name="model"' in edit
    assert 'id="key-groq"' in edit
    assert "ABC9" in edit and "Key:" in edit


def test_connection_result_is_inline_and_bound_to_its_provider(setup_provider):
    _, _, _, outcomes, service, factory, environment = setup_provider
    environment["GROQ_API_KEY"] = "fake-groq-key"
    outcomes["groq"] = StatusFailure(401)
    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository()),
        provider_settings=service, provider_router=service.create_router(),
    )
    try:
        response = app.handle("POST", "/settings/providers", urlencode({
            "provider_id": "groq", "operation": "test_connection",
        }))
        page = app.handle("GET", response.headers["Location"]).body.decode()
    finally:
        app.close()
    assert response.status == 303
    groq_card = page.split('id="groq"', 1)[1].split("</section>", 1)[0]
    assert "Authentication failed" in groq_card
    assert "Connection test finished" in groq_card
    assert "safe local fake failure" not in page
    assert "Authentication failed" not in page.split('id="gemini"', 1)[1].split("</section>", 1)[0]


def test_settings_tab_links_providers_and_usage_when_usage_service_is_configured(tmp_path, setup_provider):
    database, _, _, _, service, _, _ = setup_provider
    usage = LLMUsageService(LLMUsageRepository(database))
    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository()), provider_settings=service,
        llm_usage=usage,
    )
    try:
        providers = app.handle("GET", "/settings/providers").body.decode()
        analytics = app.handle("GET", "/settings/usage").body.decode()
    finally:
        app.close()
    assert 'href="/settings/usage"' in providers
    assert 'href="/settings/providers"' in analytics


def test_builtin_provider_cannot_be_deleted_as_custom(setup_provider):
    with pytest.raises(ValueError, match="Built-in"):
        setup_provider[4].delete_custom("groq")


def test_custom_provider_creation_uses_generated_stable_identity(setup_provider):
    _, _, _, _, service, _, _ = setup_provider
    provider_id = service.create_custom("My Local LLM", "http://localhost:1234/v1", "llama-local")
    view = next(item for item in service.provider_views() if item.id == provider_id)
    assert provider_id.startswith("custom_")
    assert provider_id != "My Local LLM"
    assert view.display_name == "My Local LLM"
    assert view.is_custom


def test_add_provider_web_form_creates_provider_and_redirects_to_inline_result(setup_provider):
    _, secrets, _, _, service, _, _ = setup_provider
    router = service.create_router()
    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository()),
        provider_settings=service, provider_router=router,
    )
    try:
        response = app.handle("POST", "/settings/providers", urlencode({
            "operation": "create_custom", "display_name": "Web local",
            "base_url": "http://localhost:1234/v1", "model": "local-model",
            "api_key": "web-form-secret", "requires_api_key": "yes", "enabled": "yes",
        }))
        page = app.handle("GET", response.headers["Location"]).body.decode()
    finally:
        app.close()
    provider_id = service.provider_views()[-1].id
    assert response.status == 303
    assert provider_id.startswith("custom_")
    assert secrets.get(provider_id) == "web-form-secret"
    assert "Custom provider added." in page
    assert "web-form-secret" not in page


def test_delete_provider_web_action_requires_confirmation_and_removes_custom(setup_provider):
    _, secrets, _, _, service, _, _ = setup_provider
    provider_id = service.create_custom(
        "Web delete", "http://localhost:1234/v1", "model", api_key="web-delete-secret",
    )
    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository()),
        provider_settings=service,
    )
    try:
        rejected = app.handle("POST", "/settings/providers", urlencode({
            "provider_id": provider_id, "operation": "delete_custom",
        }))
        assert provider_id in [item.id for item in service.provider_views()]
        accepted = app.handle("POST", "/settings/providers", urlencode({
            "provider_id": provider_id, "operation": "delete_custom", "confirmed": "yes",
        }))
        page = app.handle("GET", accepted.headers["Location"]).body.decode()
    finally:
        app.close()
    assert "result=invalid" in rejected.headers["Location"]
    assert provider_id not in secrets.values
    assert provider_id not in [item.id for item in service.provider_views()]
    assert "Historical AI Usage records are retained" in page


def test_custom_provider_configuration_persists_across_repository_reconstruction(setup_provider):
    database, secrets, _, _, service, factory, environment = setup_provider
    provider_id = service.create_custom(
        "Persisted local", "http://127.0.0.1:1234/v1", "model-x", enabled=False,
    )
    restarted = ProviderSettingsService(
        ProviderSettingsRepository(database), secrets,
        environment=environment, provider_factory=factory,
    )
    view = next(item for item in restarted.provider_views() if item.id == provider_id)
    assert view.display_name == "Persisted local"
    assert view.base_url == "http://127.0.0.1:1234/v1"
    assert view.model == "model-x"
    assert not view.enabled


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
        ("http://192.168.1.4:11434/v1", "http://192.168.1.4:11434/v1"),
        ("https://local.example.invalid/v1", "https://local.example.invalid/v1"),
    ],
)
def test_custom_base_url_accepts_http_https_loopback_and_private_hosts(raw, expected):
    assert validate_custom_base_url(raw) == expected


@pytest.mark.parametrize("raw", [
    "ftp://localhost/v1", "http://user:pass@localhost/v1", "http://@localhost/v1",
    "http://localhost/v1?token=x", "http://localhost/v1#fragment", "http://",
    "http://localhost:99999/v1", "http://bad host/v1", "//localhost/v1",
])
def test_custom_base_url_rejects_unsafe_or_malformed_values(raw):
    with pytest.raises(ValueError):
        validate_custom_base_url(raw)


def test_custom_api_key_is_stored_only_in_secret_store(setup_provider):
    database, secrets, _, _, service, _, _ = setup_provider
    key = "private-custom-key-never-sqlite"
    provider_id = service.create_custom(
        "Keyed endpoint", "http://localhost:3333/v1", "model", api_key=key,
        requires_api_key=True,
    )
    assert secrets.get(provider_id) == key
    assert key.encode() not in database.read_bytes()
    assert service.repository.get_custom_all()[0]["provider_type"] == "OPENAI_COMPATIBLE"


def test_custom_key_is_never_rendered_back_to_html(setup_provider):
    _, _, _, _, service, _, _ = setup_provider
    key = "custom-secret-must-not-render-8K"
    provider_id = service.create_custom("Secret endpoint", "http://localhost:4411/v1", "model", api_key=key)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), provider_settings=service)
    try:
        html = app.handle("GET", "/settings/providers").body.decode()
    finally:
        app.close()
    card = html.split(f'id="{provider_id}"', 1)[1].split("</section>", 1)[0]
    assert key not in html
    assert "Not set" not in card
    assert 'type="password"' in card


def test_custom_provider_order_is_used_by_existing_router(setup_provider):
    _, _, _, _, service, _, environment = setup_provider
    environment["GROQ_API_KEY"] = "fake-key"
    custom_id = service.create_custom("Priority endpoint", "http://localhost:5566/v1", "model")
    for provider_id in ("openai", "gemini", "openrouter"):
        service.update(provider_id, enabled=False)
    service.move(custom_id, -1)
    router = service.create_router()
    assert [provider.name for provider in router._providers] == [custom_id, "groq"]


def test_custom_provider_falls_back_through_existing_router(setup_provider):
    _, _, calls, outcomes, service, _, environment = setup_provider
    environment["GROQ_API_KEY"] = "fake-key"
    custom_id = service.create_custom("Backup Local", "http://localhost:5678/v1", "model")
    for provider_id in ("openai", "gemini", "openrouter"):
        service.update(provider_id, enabled=False)
    service.move("groq", -1)
    service.move(custom_id, -1)
    outcomes["groq"] = RetryableLLMError("HTTP 503")
    assert service.create_router().create_structured_output("prompt", {}, "schema") == '{"ok":true}'
    assert calls == ["groq", custom_id]


def test_disabled_custom_provider_is_not_invoked(setup_provider):
    _, _, calls, _, service, _, _ = setup_provider
    provider_id = service.create_custom("Disabled Local", "http://localhost:6789/v1", "model", enabled=False)
    assert provider_id not in [provider.name for provider in service.create_router()._providers]
    assert calls == []


def test_custom_connection_test_uses_fake_adapter_and_stores_safe_status(setup_provider):
    _, _, calls, outcomes, service, _, _ = setup_provider
    provider_id = service.create_custom("Fake local", "http://localhost:7010/v1", "fake-model")
    outcomes[provider_id] = StatusFailure(429)
    result = service.test_connection(provider_id)
    assert result.status == "rate_limited"
    assert service.provider_views()[-1].health == "rate_limited"
    assert calls == [provider_id]


def test_optional_keyless_custom_provider_is_configured_and_router_available(setup_provider):
    _, _, _, _, service, _, _ = setup_provider
    provider_id = service.create_custom("No auth local", "http://127.0.0.1:7777/v1", "local-model")
    view = next(item for item in service.provider_views() if item.id == provider_id)
    provider = next(item for item in service.create_router()._providers if item.name == provider_id)
    assert view.status == "CONFIGURED"
    assert view.credential_source == "No API key required"
    assert provider.is_available


def test_custom_provider_requires_key_when_configured_to_require_it(setup_provider):
    with pytest.raises(ValueError, match="API key is required"):
        setup_provider[4].create_custom(
            "Key required", "http://localhost:8899/v1", "model", requires_api_key=True,
        )


def test_custom_usage_records_stable_id_display_name_and_unknown_cost(setup_provider):
    database, _, _, _, service, _, _ = setup_provider
    usage_repository = LLMUsageRepository(database)
    usage = LLMUsageService(usage_repository)
    provider_id = service.create_custom("Local Analytics", "http://localhost:9010/v1", "local model/β")
    router = service.create_router(usage_recorder=usage)
    with llm_usage_scope(operation_type=OP_AUTHOR_TESTCASE):
        router.create_structured_output("local fake prompt", {}, "author")
    record = usage_repository.list_records()[0]
    assert record.provider_id == provider_id
    assert record.provider_name == "Local Analytics"
    assert record.model == "local model/β"
    assert record.input_tokens is None and record.output_tokens is None
    assert record.estimated_cost_usd is None


def test_custom_usage_stores_only_provider_reported_tokens(setup_provider):
    database, _, _, _, service, _, _ = setup_provider
    usage_repository = LLMUsageRepository(database)
    usage = LLMUsageService(usage_repository)
    provider_id = service.create_custom("Usage local", "http://localhost:9011/v1", "model")
    router = service.create_router(usage_recorder=usage)
    next(item for item in router._providers if item.name == provider_id).usage = {
        "prompt_tokens": 7, "completion_tokens": 3,
    }
    router.create_structured_output("prompt", {}, "schema")
    record = usage_repository.list_records()[0]
    assert record.provider_id == provider_id
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (7, 3, 10)


def test_custom_provider_deletion_removes_configuration_and_saved_credential(setup_provider):
    _, secrets, _, _, service, _, _ = setup_provider
    provider_id = service.create_custom(
        "Delete local", "http://localhost:9012/v1", "model", api_key="delete-me",
    )
    service.delete_custom(provider_id)
    assert provider_id not in secrets.values
    assert provider_id not in [item.id for item in service.provider_views()]


def test_custom_deletion_preserves_historical_usage_and_display_name(setup_provider):
    database, _, _, _, service, _, _ = setup_provider
    usage_repository = LLMUsageRepository(database)
    usage = LLMUsageService(usage_repository)
    provider_id = service.create_custom("History local", "http://localhost:9013/v1", "model")
    router = service.create_router(usage_recorder=usage)
    router.create_structured_output("prompt", {}, "schema")
    service.delete_custom(provider_id)
    record = usage_repository.list_records()[0]
    assert record.provider_id == provider_id
    assert record.provider_name == "History local"


def test_removing_custom_key_updates_configuration_and_runtime_availability(setup_provider):
    database, secrets, _, _, _, _, environment = setup_provider
    service = ProviderSettingsService(
        ProviderSettingsRepository(database), secrets, environment=environment,
    )
    provider_id = service.create_custom(
        "Required local", "http://localhost:9014/v1", "model",
        api_key="remove-from-vault", requires_api_key=True,
    )
    service.remove_key(provider_id)
    view = next(item for item in service.provider_views() if item.id == provider_id)
    provider = next(item for item in service._build_providers(only_provider=provider_id))
    assert provider_id not in secrets.values
    assert view.status == "NOT_CONFIGURED"
    assert view.credential_source == "Not configured"
    assert not provider.is_available


def test_priorities_are_contiguous_after_custom_provider_deletion(setup_provider):
    _, _, _, _, service, _, _ = setup_provider
    first = service.create_custom("First custom", "http://localhost:9100/v1", "model")
    second = service.create_custom("Second custom", "http://localhost:9101/v1", "model")
    service.move(second, -1)
    service.delete_custom(first)
    priorities = [item.priority for item in service.provider_views()]
    assert priorities == list(range(1, len(priorities) + 1))
    assert service.provider_views()[-1].id == second


def test_custom_provider_name_is_escaped_in_html(setup_provider):
    _, _, _, _, service, _, _ = setup_provider
    service.create_custom('<img src=x onerror="alert(1)">', "http://localhost:9200/v1", "model")
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), provider_settings=service)
    try:
        html = app.handle("GET", "/settings/providers").body.decode()
    finally:
        app.close()
    assert '<img src=x onerror="alert(1)">' not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html


def test_custom_display_name_is_used_for_progress_and_fallback(setup_provider):
    _, _, _, outcomes, service, _, environment = setup_provider
    environment["GROQ_API_KEY"] = "fake-key"
    provider_id = service.create_custom("My Local LLM", "http://localhost:9300/v1", "model")
    for other in ("openai", "gemini", "openrouter"):
        service.update(other, enabled=False)
    service.move("groq", -1)
    service.move(provider_id, -1)
    outcomes["groq"] = RetryableLLMError("HTTP 503")
    events = []
    service.create_router().create_structured_output(
        "prompt", {}, "schema",
        progress_callback=lambda event, provider, **kwargs: events.append((event, provider, kwargs)),
    )
    assert events[2][0] == "fallback"
    assert events[2][1] == "Groq"
    assert events[2][2]["next_provider"] == "My Local LLM"
    assert events[3][1] == "My Local LLM"


def test_renaming_custom_provider_keeps_stable_identity(setup_provider):
    _, _, _, _, service, _, _ = setup_provider
    provider_id = service.create_custom("Before rename", "http://localhost:9400/v1", "model")
    service.update_custom(provider_id, display_name="After rename")
    view = next(item for item in service.provider_views() if item.id == provider_id)
    assert view.id == provider_id
    assert view.display_name == "After rename"


def test_environment_only_builtin_configuration_still_works(setup_provider):
    _, _, _, _, service, _, environment = setup_provider
    environment["GROQ_API_KEY"] = "environment-groq"
    view = next(item for item in service.provider_views() if item.id == "groq")
    assert view.status == "CONFIGURED"
    assert view.credential_source == "Environment variable"
    assert next(item for item in service.create_router()._providers if item.name == "groq").is_available


def test_settings_schema_migration_is_idempotent_and_preserves_existing_rows(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE provider_settings (provider_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL, priority INTEGER NOT NULL, model TEXT)"
        )
        connection.execute("INSERT INTO provider_settings VALUES ('groq', 0, 1, 'saved-model')")
    first = ProviderSettingsRepository(database)
    second = ProviderSettingsRepository(database)
    assert first.get_all()["groq"] == {"enabled": False, "priority": 1, "model": "saved-model"}
    assert second.get_custom_all() == []
    with sqlite3.connect(database) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='custom_provider_settings'"
        ).fetchone()
    assert table is not None
