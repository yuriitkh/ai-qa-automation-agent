"""Shared secret redaction for error reporting and observability.

Centralizes the redaction rules previously inlined in the LLM router so that
error messages, execution traces, and reports never leak configured secrets.
"""

import os

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")
_REDACTED_PLACEHOLDER = "[REDACTED]"
_MAX_FAILURE_REASON_CHARS = 300


def redact_secrets(text: str) -> str:
    """Replace configured secret values with a fixed placeholder."""
    for variable, secret in os.environ.items():
        if variable.endswith(_SECRET_ENV_SUFFIXES) and secret:
            text = text.replace(secret, _REDACTED_PLACEHOLDER)
    return text


def safe_failure_reason(error: Exception) -> str:
    """Return a redacted, whitespace-normalized, bounded failure reason."""
    reason = " ".join(str(error).split())
    reason = redact_secrets(reason)
    return (reason or type(error).__name__)[:_MAX_FAILURE_REASON_CHARS]
