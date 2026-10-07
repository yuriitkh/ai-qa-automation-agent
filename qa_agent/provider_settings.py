"""Persist nonsecret provider preferences and resolve provider credentials."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from qa_agent.llm.errors import NonRetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.redaction import register_secret


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    display_name: str
    env_key: str
    model_env: str
    default_model: str
    base_url_env: str | None = None
    default_base_url: str | None = None


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
    """SQLite storage for enablement, order, and optional model overrides only."""

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


@dataclass(frozen=True)
class ConnectionTestResult:
    status: str
    latency_ms: int | None = None


class ProviderSettingsService:
    def __init__(
        self,
        repository: ProviderSettingsRepository,
        secret_store: SecretStore,
        *,
        environment: dict[str, str] | None = None,
        provider_factory=None,
    ) -> None:
        self.repository = repository
        self.secret_store = secret_store
        self._environment = environment
        self._provider_factory = provider_factory
        self._health: dict[str, ConnectionTestResult] = {}

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
        return definitions

    def _env(self, key: str, default: str = "") -> str:
        source = self._environment if self._environment is not None else os.environ
        return str(source.get(key, default)).strip()

    def _defaults(self) -> list[dict[str, object]]:
        configured_order = [name.strip().lower() for name in self._env("LLM_PROVIDER_ORDER", "openai,gemini,openrouter,groq").split(",") if name.strip()]
        fallback = [provider.id for provider in PROVIDER_DEFINITIONS]
        fallback.extend(name for name in self.definitions if name not in fallback)
        ordered = [name for name in configured_order if name in self.definitions]
        ordered.extend(name for name in fallback if name not in ordered)
        return [
            {"provider_id": name, "enabled": True, "priority": index + 1, "model": None}
            for index, name in enumerate(ordered)
        ]

    def _settings(self) -> list[dict[str, object]]:
        stored = self.repository.get_all()
        defaults = self._defaults()
        if not stored:
            return defaults
        merged = []
        for default in defaults:
            merged.append({**default, **stored.get(str(default["provider_id"]), {})})
        return sorted(merged, key=lambda item: int(item["priority"]))

    def _credential(self, provider_id: str) -> tuple[str | None, str]:
        try:
            web_secret = self.secret_store.get(provider_id)
        except SecretStoreError:
            env_secret = self._env(self.definitions[provider_id].env_key)
            return (env_secret, "environment") if env_secret else (None, "storage_error")
        if web_secret:
            register_secret(web_secret)
            return web_secret, "web"
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
            status = "DISABLED" if not enabled else "CONFIGURED" if secret else "NOT_CONFIGURED"
            model = str(item.get("model") or self._env(definition.model_env, definition.default_model))
            health = self._health.get(provider_id)
            views.append(ProviderView(
                provider_id, definition.display_name, enabled, int(item["priority"]), status,
                {"web": "Web settings", "environment": "Environment variable", "unconfigured": "Not configured", "storage_error": "Credential storage unavailable"}.get(source, "Not configured"),
                self._mask(secret), model, health.status if health else None,
                health.latency_ms if health else None,
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
                return
        raise ValueError("Unknown provider.")

    def save_key(self, provider_id: str, value: str) -> None:
        if provider_id not in self.definitions:
            raise ValueError("Unknown provider.")
        secret = value.strip()
        if not secret or len(secret) > 2500 or any(ord(char) < 32 for char in secret):
            raise ValueError("Invalid credential.")
        try:
            self.secret_store.set(provider_id, secret)
        except SecretStoreError:
            raise
        register_secret(secret)
        self._health.pop(provider_id, None)

    def remove_key(self, provider_id: str) -> None:
        if provider_id not in self.definitions:
            raise ValueError("Unknown provider.")
        self.secret_store.delete(provider_id)
        self._health.pop(provider_id, None)

    def create_router(self) -> LLMRouter:
        return LLMRouter(self._build_providers())

    def refresh_router(self, router: LLMRouter) -> None:
        router.replace_providers(self._build_providers())

    def _build_providers(self, *, only_provider: str | None = None, timeout: float | None = None):
        from qa_agent.llm.gemini import GeminiProvider
        from qa_agent.llm.groq import GroqProvider
        from qa_agent.llm.openai_compatible import OpenAICompatibleProvider

        items = self._settings()
        result = []
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
                provider = GeminiProvider(api_key=secret, model=model, timeout_seconds=timeout)
            elif provider_id == "groq":
                provider = GroqProvider(api_key=secret, model=model, timeout_seconds=timeout, max_output_tokens=64 if timeout else None)
            else:
                base_url = self._env(definition.base_url_env, definition.default_base_url or "") if definition.base_url_env else None
                provider = OpenAICompatibleProvider(
                    provider_id, definition.env_key, model, base_url,
                    max_output_tokens=64 if timeout else None, api_key=secret, timeout_seconds=timeout,
                )
            result.append(provider)
        return result

    def test_connection(self, provider_id: str, *, timeout_seconds: float = 8.0) -> ConnectionTestResult:
        if provider_id not in self.definitions:
            return ConnectionTestResult("configuration_invalid")
        secret, source = self._credential(provider_id)
        if source == "storage_error" or not secret:
            result = ConnectionTestResult("configuration_invalid")
            self._health[provider_id] = result
            return result
        setting = next(item for item in self._settings() if item["provider_id"] == provider_id)
        definition = self.definitions[provider_id]
        model = str(setting.get("model") or self._env(definition.model_env, definition.default_model))
        if not model:
            result = ConnectionTestResult("configuration_invalid")
            self._health[provider_id] = result
            return result
        try:
            provider = self._build_providers(only_provider=provider_id, timeout=timeout_seconds)[0]
            if not provider.is_available:
                result = ConnectionTestResult("configuration_invalid")
            else:
                started = time.perf_counter()
                provider.create_structured_output(
                    'Return only {"ok":true}.',
                    {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
                    "connection_test",
                )
                result = ConnectionTestResult("connected", round((time.perf_counter() - started) * 1000))
        except Exception as error:
            result = ConnectionTestResult(_classify_connection_error(error))
        self._health[provider_id] = result
        return result

    def _save_ordered(self, items: list[dict[str, object]]) -> None:
        self.repository.save_all([
            {**item, "priority": index + 1}
            for index, item in enumerate(items)
        ])


def _classify_connection_error(error: Exception) -> str:
    chain: list[Exception] = []
    current: Exception | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(item, (TimeoutError, httpx.TimeoutException)) or "timeout" in type(item).__name__.lower() for item in chain):
        return "timeout"
    statuses = []
    for item in chain:
        status = getattr(item, "status_code", None)
        if status is None:
            response = getattr(item, "response", None)
            status = getattr(response, "status_code", None)
        if isinstance(status, int):
            statuses.append(status)
        match = re.search(r"HTTP\s+(\d{3})", str(item), re.IGNORECASE)
        if match:
            statuses.append(int(match.group(1)))
    if any(status in {401, 403} for status in statuses):
        return "authentication_failed"
    if 429 in statuses:
        return "rate_limited"
    if any(status >= 500 for status in statuses):
        return "provider_unavailable"
    if any(status in {400, 404, 422} for status in statuses):
        return "configuration_invalid"
    if any(isinstance(item, (ValueError, NonRetryableLLMError)) for item in chain):
        return "configuration_invalid"
    return "provider_unavailable"
