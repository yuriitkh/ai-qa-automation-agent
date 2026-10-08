"""Persist nonsecret provider preferences and resolve provider credentials."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
import json
import os
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4
from urllib.parse import urlsplit

import httpx

from qa_agent.llm.errors import (
    category_for_error,
    failure_detail_for,
    RetryableLLMError,
)
from qa_agent.llm.router import LLMRouter
from qa_agent.redaction import register_secret
from qa_agent.llm_usage import OP_OTHER, llm_usage_scope


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    display_name: str
    env_key: str
    model_env: str
    default_model: str
    base_url_env: str | None = None
    default_base_url: str | None = None
    is_custom: bool = False
    requires_api_key: bool = True


PROVIDER_DEFINITIONS = (
    ProviderDefinition("gemini", "Gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.8-flash"),
    ProviderDefinition("groq", "Groq", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-20b"),
    ProviderDefinition("openai", "OpenAI", "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4.1-mini"),
    ProviderDefinition(
        "openrouter", "OpenRouter", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
        "openai/gpt-4.1-mini", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1",
    ),
)
_DEFINITIONS = {provider.id: provider for provider in PROVIDER_DEFINITIONS}
_SAFE_PROVIDER_ID = re.compile(r"^[a-zA-Z0-9_.-]{1,80}$")


class SecretStore(Protocol):
    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class SecretStoreError(RuntimeError):
    """A safe, detail-free credential storage failure."""


class UnavailableSecretStore:
    """Environment variables remain usable where Windows Credential Manager is absent."""

    def get(self, name: str) -> str | None:
        return None

    def set(self, name: str, value: str) -> None:
        raise SecretStoreError("Secure credential storage is unavailable on this system.")

    def delete(self, name: str) -> None:
        raise SecretStoreError("Secure credential storage is unavailable on this system.")


def create_default_secret_store() -> SecretStore:
    if os.name == "nt":
        try:
            return WindowsCredentialStore()
        except Exception:
            return UnavailableSecretStore()
    return UnavailableSecretStore()


class WindowsCredentialStore:
    """Store API keys in the current Windows user's Credential Manager."""

    _PREFIX = "AI QA Agent/provider/"

    def __init__(self) -> None:
        if os.name != "nt":
            raise SecretStoreError("Windows Credential Manager is unavailable.")
        self._advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        self._cred_free = ctypes.WinDLL("Advapi32.dll", use_last_error=True).CredFree

        class FileTime(ctypes.Structure):
            _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

        class Credential(ctypes.Structure):
            _fields_ = [
                ("Flags", ctypes.c_uint32), ("Type", ctypes.c_uint32),
                ("TargetName", ctypes.c_wchar_p), ("Comment", ctypes.c_wchar_p),
                ("LastWritten", FileTime), ("CredentialBlobSize", ctypes.c_uint32),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                ("Persist", ctypes.c_uint32), ("AttributeCount", ctypes.c_uint32),
                ("Attributes", ctypes.c_void_p), ("TargetAlias", ctypes.c_wchar_p),
                ("UserName", ctypes.c_wchar_p),
            ]

        self._Credential = Credential
        self._advapi32.CredWriteW.argtypes = [ctypes.POINTER(Credential), ctypes.c_uint32]
        self._advapi32.CredWriteW.restype = ctypes.c_int
        self._advapi32.CredReadW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(Credential))]
        self._advapi32.CredReadW.restype = ctypes.c_int
        self._advapi32.CredDeleteW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        self._advapi32.CredDeleteW.restype = ctypes.c_int
        self._cred_free.argtypes = [ctypes.c_void_p]
        self._cred_free.restype = None

    def _target(self, name: str) -> str:
        if not _SAFE_PROVIDER_ID.fullmatch(name):
            raise SecretStoreError("Unknown provider credential.")
        return self._PREFIX + name

    def get(self, name: str) -> str | None:
        credential = ctypes.POINTER(self._Credential)()
        try:
            found = self._advapi32.CredReadW(self._target(name), 1, 0, ctypes.byref(credential))
            if not found:
                error = ctypes.get_last_error()
                if error == 1168:  # ERROR_NOT_FOUND
                    return None
                raise SecretStoreError("Credential storage is unavailable.")
            blob = ctypes.string_at(credential.contents.CredentialBlob, credential.contents.CredentialBlobSize)
            try:
                return blob.decode("utf-8")
            except UnicodeDecodeError as error:
                raise SecretStoreError("Stored credential could not be read.") from error
        except SecretStoreError:
            raise
        except Exception as error:
            raise SecretStoreError("Credential storage is unavailable.") from error
        finally:
            if credential:
                self._cred_free(credential)

    def set(self, name: str, value: str) -> None:
        target = self._target(name)
        blob = value.encode("utf-8")
        if not blob or len(blob) > 2560:
            raise SecretStoreError("The credential has an invalid size.")
        buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        credential = self._Credential(
            Flags=0, Type=1, TargetName=target, Comment=None, LastWritten=(0, 0),
            CredentialBlobSize=len(blob), CredentialBlob=buffer, Persist=2,
            AttributeCount=0, Attributes=None, TargetAlias=None, UserName="AI QA Agent",
        )
        try:
            if not self._advapi32.CredWriteW(ctypes.byref(credential), 0):
                raise SecretStoreError("Credential storage is unavailable.")
        except SecretStoreError:
            raise
        except Exception as error:
            raise SecretStoreError("Credential storage is unavailable.") from error

    def delete(self, name: str) -> None:
        try:
            removed = self._advapi32.CredDeleteW(self._target(name), 1, 0)
            if not removed and ctypes.get_last_error() != 1168:
                raise SecretStoreError("Credential storage is unavailable.")
        except SecretStoreError:
            raise
        except Exception as error:
            raise SecretStoreError("Credential storage is unavailable.") from error


class ProviderSettingsRepository:
    """SQLite storage for nonsecret provider configuration and ordering."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self._database_path != ":memory:":
            Path(self._database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(self._database_path)
            self._memory_connection.row_factory = sqlite3.Row
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS provider_settings ("
                "provider_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), "
                "priority INTEGER NOT NULL CHECK(priority >= 1), model TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS custom_provider_settings ("
                "provider_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, base_url TEXT NOT NULL, "
                "model TEXT NOT NULL, enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), "
                "priority INTEGER NOT NULL CHECK(priority >= 1), provider_type TEXT NOT NULL "
                "CHECK(provider_type = 'OPENAI_COMPATIBLE'), requires_api_key INTEGER NOT NULL "
                "CHECK(requires_api_key IN (0,1)), key_configured INTEGER NOT NULL "
                "CHECK(key_configured IN (0,1)))"
            )

    def _connect(self):
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        if self._memory_connection is not None:
            with self._memory_connection:
                yield self._memory_connection
            return
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get_all(self) -> dict[str, dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT provider_id, enabled, priority, model FROM provider_settings"
            ).fetchall()
        return {
            row["provider_id"]: {
                "enabled": bool(row["enabled"]), "priority": int(row["priority"]), "model": row["model"]
            }
            for row in rows
        }

    def save_all(self, settings: list[dict[str, object]]) -> None:
        with self._connection() as connection:
            connection.executemany(
                "INSERT INTO provider_settings(provider_id, enabled, priority, model) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(provider_id) DO UPDATE SET enabled=excluded.enabled, "
                "priority=excluded.priority, model=excluded.model",
                [
                    (item["provider_id"], int(bool(item["enabled"])), int(item["priority"]), item.get("model"))
                    for item in settings
                ],
            )

    def get_custom_all(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT provider_id, display_name, base_url, model, enabled, priority, "
                "provider_type, requires_api_key, key_configured "
                "FROM custom_provider_settings"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_custom(self, item: dict[str, object]) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO custom_provider_settings(provider_id, display_name, base_url, model, "
                "enabled, priority, provider_type, requires_api_key, key_configured) "
                "VALUES (?, ?, ?, ?, ?, ?, 'OPENAI_COMPATIBLE', ?, ?) "
                "ON CONFLICT(provider_id) DO UPDATE SET display_name=excluded.display_name, "
                "base_url=excluded.base_url, model=excluded.model, enabled=excluded.enabled, "
                "priority=excluded.priority, requires_api_key=excluded.requires_api_key, "
                "key_configured=excluded.key_configured",
                (item["provider_id"], item["display_name"], item["base_url"], item["model"],
                 int(bool(item["enabled"])), int(item["priority"]),
                 int(bool(item["requires_api_key"])), int(bool(item["key_configured"]))),
            )

    def delete_custom(self, provider_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM custom_provider_settings WHERE provider_id = ?", (provider_id,)
            )


@dataclass(frozen=True)
class ProviderView:
    id: str
    display_name: str
    enabled: bool
    priority: int
    status: str
    credential_source: str
    masked_key: str
    model: str
    health: str | None = None
    latency_ms: int | None = None
    is_custom: bool = False
    base_url: str | None = None
    requires_api_key: bool = True
    connection_category: str | None = None
    connection_http_status: int | None = None
    connection_retry_after_seconds: int | None = None
    capability_status: str | None = None
    capability_latency_ms: int | None = None
    capability_category: str | None = None
    capability_http_status: int | None = None
    capability_retry_after_seconds: int | None = None
    connection_provider_error_code: str | None = None
    connection_provider_error_type: str | None = None
    connection_provider_error_field: str | None = None
    capability_provider_error_code: str | None = None
    capability_provider_error_type: str | None = None
    capability_provider_error_field: str | None = None


@dataclass(frozen=True)
class ConnectionTestResult:
    status: str
    latency_ms: int | None = None
    category: str | None = None
    http_status: int | None = None
    retry_after_seconds: int | None = None
    provider_error_code: str | None = None
    provider_error_type: str | None = None
    provider_error_field: str | None = None


@dataclass(frozen=True)
class AuthoringCapabilityTestResult:
    status: str
    latency_ms: int | None = None
    category: str | None = None
    http_status: int | None = None
    retry_after_seconds: int | None = None
    provider_error_code: str | None = None
    provider_error_type: str | None = None
    provider_error_field: str | None = None


class ProviderSettingsService:
    def __init__(
        self,
        repository: ProviderSettingsRepository,
        secret_store: SecretStore,
        *,
        environment: dict[str, str] | None = None,
        provider_factory=None,
        usage_recorder=None,
    ) -> None:
        self.repository = repository
        self.secret_store = secret_store
        self._environment = environment
        self._provider_factory = provider_factory
        self._usage_recorder = usage_recorder
        self._health: dict[str, ConnectionTestResult] = {}
        self._capability: dict[str, AuthoringCapabilityTestResult] = {}

    @property
    def definitions(self) -> dict[str, ProviderDefinition]:
        definitions = dict(_DEFINITIONS)
        for name in (item.strip().lower() for item in self._env("LLM_COMPATIBLE_PROVIDERS").split(",")):
            if not name or name in definitions or not _SAFE_PROVIDER_ID.fullmatch(name):
                continue
            prefix = name.upper()
            definitions[name] = ProviderDefinition(
                name, name.replace("_", " ").title(), f"{prefix}_API_KEY",
                f"{prefix}_MODEL", "", f"{prefix}_BASE_URL", None,
            )
        for item in self.repository.get_custom_all():
            provider_id = str(item["provider_id"])
            definitions[provider_id] = ProviderDefinition(
                provider_id, str(item["display_name"]), "", "", str(item["model"]),
                None, str(item["base_url"]), True, bool(item["requires_api_key"]),
            )
        return definitions

    def _env(self, key: str, default: str = "") -> str:
        source = self._environment if self._environment is not None else os.environ
        return str(source.get(key, default)).strip()

    def _defaults(self) -> list[dict[str, object]]:
        configured_order = [name.strip().lower() for name in self._env("LLM_PROVIDER_ORDER", "openai,gemini,openrouter,groq").split(",") if name.strip()]
        fallback = [provider.id for provider in PROVIDER_DEFINITIONS]
        fallback.extend(
            name for name, definition in self.definitions.items()
            if name not in fallback and not definition.is_custom
        )
        ordered = [
            name for name in configured_order
            if name in self.definitions and not self.definitions[name].is_custom
        ]
        ordered.extend(name for name in fallback if name not in ordered)
        return [
            {"provider_id": name, "enabled": True, "priority": index + 1, "model": None}
            for index, name in enumerate(ordered)
        ]

    def _settings(self) -> list[dict[str, object]]:
        stored = self.repository.get_all()
        defaults = self._defaults()
        merged = []
        for default in defaults:
            merged.append({**default, **stored.get(str(default["provider_id"]), {})})
        custom = sorted(self.repository.get_custom_all(), key=lambda item: (int(item["priority"]), str(item["provider_id"])))
        next_priority = max((int(item["priority"]) for item in merged), default=0) + 1
        for index, item in enumerate(custom):
            merged.append({
                **item,
                "priority": int(item["priority"] or next_priority + index),
                "enabled": bool(item["enabled"]),
                "requires_api_key": bool(item["requires_api_key"]),
                "key_configured": bool(item["key_configured"]),
            })
        return sorted(merged, key=lambda item: (int(item["priority"]), str(item["provider_id"])))

    def _credential(self, provider_id: str) -> tuple[str | None, str]:
        definition = self.definitions[provider_id]
        custom = next((item for item in self.repository.get_custom_all() if item["provider_id"] == provider_id), None)
        if custom is not None and not bool(custom["key_configured"]) and not bool(custom["requires_api_key"]):
            return None, "no_key_required"
        try:
            web_secret = self.secret_store.get(provider_id)
        except SecretStoreError:
            if definition.is_custom:
                return None, "storage_error"
            env_secret = self._env(self.definitions[provider_id].env_key)
            return (env_secret, "environment") if env_secret else (None, "storage_error")
        if web_secret:
            register_secret(web_secret)
            return web_secret, "web"
        if definition.is_custom:
            return (None, "unconfigured") if definition.requires_api_key else (None, "no_key_required")
        env_secret = self._env(self.definitions[provider_id].env_key)
        return (env_secret, "environment") if env_secret else (None, "unconfigured")

    @staticmethod
    def _mask(secret: str | None) -> str:
        if not secret:
            return "Not set"
        return "••••••••••••" + secret[-4:]

    def provider_views(self) -> list[ProviderView]:
        views = []
        for item in self._settings():
            provider_id = str(item["provider_id"])
            definition = self.definitions[provider_id]
            secret, source = self._credential(provider_id)
            enabled = bool(item["enabled"])
            has_configuration = bool(secret) or source == "no_key_required"
            status = "DISABLED" if not enabled else "CONFIGURED" if has_configuration else "NOT_CONFIGURED"
            model = str(item.get("model") or self._env(definition.model_env, definition.default_model))
            health = self._health.get(provider_id)
            capability = self._capability.get(provider_id)
            views.append(ProviderView(
                provider_id, definition.display_name, enabled, int(item["priority"]), status,
                {"web": "Web settings", "environment": "Environment variable", "unconfigured": "Not configured", "storage_error": "Credential storage unavailable", "no_key_required": "No API key required"}.get(source, "Not configured"),
                self._mask(secret), model, health.status if health else None,
                health.latency_ms if health else None, definition.is_custom,
                definition.default_base_url if definition.is_custom else None,
                definition.requires_api_key,
                health.category if health else None,
                health.http_status if health else None,
                health.retry_after_seconds if health else None,
                capability.status if capability else None,
                capability.latency_ms if capability else None,
                capability.category if capability else None,
                capability.http_status if capability else None,
                capability.retry_after_seconds if capability else None,
                health.provider_error_code if health else None,
                health.provider_error_type if health else None,
                health.provider_error_field if health else None,
                capability.provider_error_code if capability else None,
                capability.provider_error_type if capability else None,
                capability.provider_error_field if capability else None,
            ))
        return views

    def _ordered_settings(self) -> list[dict[str, object]]:
        return self._settings()

    def move(self, provider_id: str, direction: int) -> None:
        items = self._ordered_settings()
        index = next((i for i, item in enumerate(items) if item["provider_id"] == provider_id), None)
        if index is None:
            raise ValueError("Unknown provider.")
        target = index + direction
        if target < 0 or target >= len(items):
            return
        items[index], items[target] = items[target], items[index]
        self._save_ordered(items)

    def update(self, provider_id: str, *, enabled: bool | None = None, model: str | None = None) -> None:
        items = self._ordered_settings()
        for item in items:
            if item["provider_id"] == provider_id:
                if enabled is not None:
                    item["enabled"] = enabled
                if model is not None:
                    item["model"] = model or None
                self._save_ordered(items)
                self._clear_diagnostics(provider_id)
                return
        raise ValueError("Unknown provider.")

    def create_custom(
        self,
        display_name: str,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        enabled: bool = True,
        requires_api_key: bool = False,
    ) -> str:
        safe_name = _validate_provider_name(display_name)
        safe_url = validate_custom_base_url(base_url)
        safe_model = _validate_model(model, required=True)
        secret = _validate_api_key(api_key) if api_key else None
        if requires_api_key and not secret:
            raise ValueError("An API key is required for this provider.")
        provider_id = "custom_" + uuid4().hex
        if secret:
            self.secret_store.set(provider_id, secret)
            register_secret(secret)
        items = self._ordered_settings()
        item = {
            "provider_id": provider_id, "display_name": safe_name, "base_url": safe_url,
            "model": safe_model, "enabled": bool(enabled), "priority": len(items) + 1,
            "requires_api_key": bool(requires_api_key), "key_configured": bool(secret),
        }
        try:
            self.repository.save_custom(item)
            items.append(item)
            self._save_ordered(items)
        except Exception:
            try:
                self.repository.delete_custom(provider_id)
            except Exception:
                pass
            if secret:
                try:
                    self.secret_store.delete(provider_id)
                except Exception:
                    pass
            raise
        return provider_id

    def update_custom(
        self,
        provider_id: str,
        *,
        display_name: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        item = self._custom_item(provider_id)
        if display_name is not None:
            item["display_name"] = _validate_provider_name(display_name)
        if base_url is not None:
            item["base_url"] = validate_custom_base_url(base_url)
        if model is not None:
            item["model"] = _validate_model(model, required=True)
        self.repository.save_custom(item)
        self._clear_diagnostics(provider_id)

    def delete_custom(self, provider_id: str) -> None:
        item = self._custom_item(provider_id)
        # Remove the external secret first so a vault failure cannot leave an
        # active configuration behind or falsely report a complete deletion.
        if bool(item["key_configured"]):
            self.secret_store.delete(provider_id)
        self.repository.delete_custom(provider_id)
        self._clear_diagnostics(provider_id)
        self._save_ordered([entry for entry in self._ordered_settings() if entry["provider_id"] != provider_id])

    def _custom_item(self, provider_id: str) -> dict[str, object]:
        item = next((item for item in self.repository.get_custom_all() if item["provider_id"] == provider_id), None)
        if item is None:
            if provider_id in _DEFINITIONS:
                raise ValueError("Built-in providers cannot be deleted or edited as custom providers.")
            raise ValueError("Unknown custom provider.")
        return item

    def save_key(self, provider_id: str, value: str) -> None:
        if provider_id not in self.definitions:
            raise ValueError("Unknown provider.")
        secret = _validate_api_key(value)
        try:
            self.secret_store.set(provider_id, secret)
        except SecretStoreError:
            raise
        register_secret(secret)
        custom = next((item for item in self.repository.get_custom_all() if item["provider_id"] == provider_id), None)
        if custom is not None:
            custom["key_configured"] = True
            self.repository.save_custom(custom)
        self._clear_diagnostics(provider_id)

    def remove_key(self, provider_id: str) -> None:
        if provider_id not in self.definitions:
            raise ValueError("Unknown provider.")
        self.secret_store.delete(provider_id)
        custom = next((item for item in self.repository.get_custom_all() if item["provider_id"] == provider_id), None)
        if custom is not None:
            custom["key_configured"] = False
            self.repository.save_custom(custom)
        self._clear_diagnostics(provider_id)

    def create_router(self, *, usage_recorder=None) -> LLMRouter:
        return LLMRouter(
            self._build_providers(),
            usage_recorder=(usage_recorder or self._usage_recorder),
        )

    def refresh_router(self, router: LLMRouter) -> None:
        router.replace_providers(self._build_providers())

    def _build_providers(
        self,
        *,
        only_provider: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
    ):
        from qa_agent.llm.gemini import GeminiProvider
        from qa_agent.llm.groq import GroqProvider
        from qa_agent.llm.openai_compatible import OpenAICompatibleProvider

        items = self._settings()
        result = []
        output_limit = max_output_tokens or (64 if timeout else None)
        for item in items:
            provider_id = str(item["provider_id"])
            if only_provider is not None and provider_id != only_provider:
                continue
            if only_provider is None and not bool(item["enabled"]):
                continue
            definition = self.definitions[provider_id]
            secret, _source = self._credential(provider_id)
            model = str(item.get("model") or self._env(definition.model_env, definition.default_model))
            if self._provider_factory is not None:
                provider = self._provider_factory(provider_id, secret, model, timeout)
            elif provider_id == "gemini":
                provider = GeminiProvider(
                    api_key=secret, model=model, timeout_seconds=timeout,
                    max_output_tokens=output_limit,
                )
            elif provider_id == "groq":
                provider = GroqProvider(
                    api_key=secret, model=model, timeout_seconds=timeout,
                    max_output_tokens=output_limit,
                )
            elif definition.is_custom:
                provider = OpenAICompatibleProvider(
                    provider_id, "", model, definition.default_base_url,
                    max_output_tokens=output_limit, api_key=secret or "",
                    timeout_seconds=timeout, display_name=definition.display_name,
                    allow_missing_api_key=not definition.requires_api_key,
                )
            else:
                base_url = self._env(definition.base_url_env, definition.default_base_url or "") if definition.base_url_env else None
                provider = OpenAICompatibleProvider(
                    provider_id, definition.env_key, model, base_url,
                    max_output_tokens=output_limit, api_key=secret, timeout_seconds=timeout,
                )
                provider.display_name = definition.display_name
            if definition.is_custom:
                provider.display_name = definition.display_name
                provider.is_custom = True
            result.append(provider)
        return result

    def test_connection(self, provider_id: str, *, timeout_seconds: float = 8.0) -> ConnectionTestResult:
        if provider_id not in self.definitions:
            return ConnectionTestResult("configuration_invalid")
        secret, source = self._credential(provider_id)
        definition = self.definitions[provider_id]
        key_missing = not secret and definition.requires_api_key
        if source == "storage_error" or key_missing:
            result = ConnectionTestResult("configuration_invalid")
            self._health[provider_id] = result
            return result
        try:
            provider = self._build_providers(
                only_provider=provider_id, timeout=timeout_seconds,
                max_output_tokens=64,
            )[0]
        except Exception:
            result = ConnectionTestResult("configuration_invalid")
            self._health[provider_id] = result
            return result
        try:
            if not provider.is_available:
                result = ConnectionTestResult("configuration_invalid")
            else:
                started = time.perf_counter()
                with llm_usage_scope(operation_type=OP_OTHER):
                    LLMRouter(
                        [provider], usage_recorder=self._usage_recorder
                    ).create_structured_output(
                        'Return only {"ok":true}.',
                        {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
                        "connection_test",
                        response_validator=_validate_connection_response,
                    )
                result = ConnectionTestResult(
                    "connected",
                    round((time.perf_counter() - started) * 1000),
                )
        except Exception as error:
            failure = failure_detail_for(definition.display_name, error)
            result = ConnectionTestResult(
                _connection_status(failure.category),
                category=failure.category,
                http_status=failure.http_status,
                retry_after_seconds=failure.retry_after_seconds,
                provider_error_code=failure.provider_error_code,
                provider_error_type=failure.provider_error_type,
                provider_error_field=failure.provider_error_field,
            )
        self._health[provider_id] = result
        return result

    def test_authoring_capability(
        self, provider_id: str, *, timeout_seconds: float = 12.0
    ) -> AuthoringCapabilityTestResult:
        """Exercise the authoring schema, parser, model, and endpoint, without saving a case."""
        if provider_id not in self.definitions:
            return AuthoringCapabilityTestResult("not_configured")
        definition = self.definitions[provider_id]
        secret, source = self._credential(provider_id)
        if source == "storage_error" or (not secret and definition.requires_api_key):
            result = AuthoringCapabilityTestResult("not_configured")
            self._capability[provider_id] = result
            return result
        try:
            provider = self._build_providers(
                only_provider=provider_id,
                timeout=timeout_seconds,
                max_output_tokens=256,
            )[0]
        except Exception:
            result = AuthoringCapabilityTestResult("not_configured")
            self._capability[provider_id] = result
            return result
        if not provider.is_available:
            result = AuthoringCapabilityTestResult("not_configured")
            self._capability[provider_id] = result
            return result

        from qa_agent.test_case_authoring import TestCaseAuthoringService

        started = time.perf_counter()
        try:
            with llm_usage_scope(operation_type=OP_OTHER):
                TestCaseAuthoringService(
                    LLMRouter([provider], usage_recorder=self._usage_recorder)
                ).test_authoring_capability()
            result = AuthoringCapabilityTestResult(
                "passed", round((time.perf_counter() - started) * 1000)
            )
        except Exception as error:
            failure = failure_detail_for(definition.display_name, error)
            result = AuthoringCapabilityTestResult(
                "failed",
                round((time.perf_counter() - started) * 1000),
                category=failure.category,
                http_status=failure.http_status,
                retry_after_seconds=failure.retry_after_seconds,
                provider_error_code=failure.provider_error_code,
                provider_error_type=failure.provider_error_type,
                provider_error_field=failure.provider_error_field,
            )
        self._capability[provider_id] = result
        return result

    def _clear_diagnostics(self, provider_id: str) -> None:
        self._health.pop(provider_id, None)
        self._capability.pop(provider_id, None)

    def _save_ordered(self, items: list[dict[str, object]]) -> None:
        custom_ids = {str(item["provider_id"]) for item in self.repository.get_custom_all()}
        normalized = [
            {**item, "priority": index + 1}
            for index, item in enumerate(items)
        ]
        self.repository.save_all([
            item for item in normalized if str(item["provider_id"]) not in custom_ids
        ])
        for item in normalized:
            if str(item["provider_id"]) in custom_ids:
                custom = next(
                    entry for entry in self.repository.get_custom_all()
                    if entry["provider_id"] == item["provider_id"]
                )
                custom["priority"] = item["priority"]
                custom["enabled"] = item["enabled"]
                self.repository.save_custom(custom)


def _validate_provider_name(value: str) -> str:
    name = value.strip()
    if not name or len(name) > 80 or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in name):
        raise ValueError("Provider name must be between 1 and 80 characters without control characters.")
    return name


def _validate_model(value: str, *, required: bool) -> str:
    model = value.strip()
    if (required and not model) or len(model) > 200 or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in model):
        raise ValueError("A valid model name is required.")
    return model


def _validate_api_key(value: str | None) -> str:
    secret = (value or "").strip()
    if not secret or len(secret) > 2500 or any(ord(char) < 32 or ord(char) == 127 for char in secret):
        raise ValueError("Invalid credential.")
    return secret


def validate_custom_base_url(value: str) -> str:
    """Validate and normalize a configured endpoint without making a request."""
    raw = value.strip()
    if (
        not raw or len(raw) > 2048 or "\\" in raw
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in raw)
        or "?" in raw or "#" in raw
    ):
        raise ValueError("Enter a valid HTTP or HTTPS base URL without credentials, query, or fragment.")
    try:
        parsed = urlsplit(raw)
        # Accessing .port validates its syntax and range.
        port = parsed.port
        host = parsed.hostname
    except ValueError as error:
        raise ValueError("Enter a valid HTTP or HTTPS base URL.") from error
    if parsed.scheme.lower() not in {"http", "https"} or not host or "@" in parsed.netloc:
        raise ValueError("Base URL must use HTTP or HTTPS and must not contain credentials.")
    if parsed.username is not None or parsed.password is not None or (port is not None and not 1 <= port <= 65535):
        raise ValueError("Enter a valid HTTP or HTTPS base URL without credentials.")
    try:
        normalized = httpx.URL(raw)
    except Exception as error:
        raise ValueError("Enter a valid HTTP or HTTPS base URL.") from error
    if normalized.scheme not in {"http", "https"} or not normalized.host or normalized.username or normalized.password:
        raise ValueError("Enter a valid HTTP or HTTPS base URL without credentials.")
    return str(normalized).rstrip("/")


def _classify_connection_error(error: Exception) -> str:
    return _connection_status(category_for_error(error))


def _connection_status(category: str) -> str:
    return {
        "AUTH_ERROR": "authentication_failed",
        "MODEL_NOT_FOUND": "model_not_found",
        "INVALID_REQUEST": "configuration_invalid",
        "RATE_LIMIT": "rate_limited",
        "TIMEOUT": "timeout",
        "SCHEMA_ERROR": "schema_error",
        "INVALID_RESPONSE": "invalid_response",
        "PROVIDER_UNAVAILABLE": "provider_unavailable",
        "OTHER_PROVIDER_ERROR": "other_error",
    }.get(category, "provider_unavailable")


def _validate_connection_response(raw: str) -> None:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise RetryableLLMError(
            "Provider returned an invalid connection response.",
            category="INVALID_RESPONSE",
            safe_detail="Invalid structured response",
        ) from error
    if not isinstance(value, dict) or set(value) != {"ok"} or value.get("ok") is not True:
        raise RetryableLLMError(
            "Provider returned an invalid connection response.",
            category="INVALID_RESPONSE",
            safe_detail="Invalid structured response",
        )
