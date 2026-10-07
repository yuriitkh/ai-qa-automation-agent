"""Shared secret redaction for error reporting and observability.

Centralizes the redaction rules previously inlined in the LLM router so that
error messages, execution traces, and reports never leak configured secrets.
"""

import os
import threading

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")
_REDACTED_PLACEHOLDER = "[REDACTED]"
_MAX_FAILURE_REASON_CHARS = 300
_registered_secrets: set[str] = set()
_registered_secrets_lock = threading.Lock()


def register_secret(value: str) -> None:
    """Register an in-process credential so logs and reports can redact it."""
    if value:
        with _registered_secrets_lock:
            _registered_secrets.add(value)


def redact_secrets(text: str) -> str:
    """Replace configured secret values with a fixed placeholder."""
    with _registered_secrets_lock:
        secrets = set(_registered_secrets)
    for variable, secret in os.environ.items():
        if variable.endswith(_SECRET_ENV_SUFFIXES) and secret:
            secrets.add(secret)
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, _REDACTED_PLACEHOLDER)
    return text


def safe_failure_reason(error: Exception) -> str:
    """Return a redacted, whitespace-normalized, bounded failure reason."""
    reason = " ".join(str(error).split())
    reason = redact_secrets(reason)
    return (reason or type(error).__name__)[:_MAX_FAILURE_REASON_CHARS]
