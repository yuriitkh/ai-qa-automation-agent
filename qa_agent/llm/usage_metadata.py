"""Provider response usage extraction without retaining response content."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class ProviderTokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens, self.total_tokens):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("Token usage values must be nonnegative integers or unknown.")
        if self.total_tokens is None and (
            self.input_tokens is not None and self.output_tokens is not None
        ):
            object.__setattr__(
                self, "total_tokens", self.input_tokens + self.output_tokens
            )


@dataclass
class ProviderUsageCapture:
    usage: ProviderTokenUsage | None = None


_CURRENT_CAPTURE: ContextVar[ProviderUsageCapture | None] = ContextVar(
    "llm_provider_usage_capture", default=None
)


@contextmanager
def capture_provider_usage() -> Iterator[ProviderUsageCapture]:
    capture = ProviderUsageCapture()
    token: Token[ProviderUsageCapture | None] = _CURRENT_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CURRENT_CAPTURE.reset(token)


def capture_openai_usage(response_or_usage: Any) -> ProviderTokenUsage | None:
    """Capture OpenAI-compatible ``usage`` fields from an object or mapping."""
    usage = _field(response_or_usage, "usage")
    if usage is None:
        usage = response_or_usage
    normalized = _normalize_usage(
        usage,
        input_names=("prompt_tokens", "input_tokens"),
        output_names=("completion_tokens", "output_tokens"),
        total_names=("total_tokens",),
    )
    _set_current_usage(normalized)
    return normalized


def capture_gemini_usage(response_or_usage: Any) -> ProviderTokenUsage | None:
    """Capture Gemini Interactions/usage metadata using known field variants."""
    usage = _field(response_or_usage, "usage")
    if usage is None:
        usage = _field(response_or_usage, "usage_metadata")
    if usage is None:
        usage = _field(response_or_usage, "usageMetadata")
    if usage is None:
        usage = response_or_usage
    normalized = _normalize_usage(
        usage,
        input_names=(
            "input_tokens", "total_input_tokens", "prompt_token_count",
            "promptTokenCount", "prompt_tokens",
        ),
        output_names=(
            "output_tokens", "total_output_tokens", "candidates_token_count",
            "candidatesTokenCount", "completion_tokens",
        ),
        total_names=("total_tokens", "total_token_count", "totalTokenCount"),
    )
    _set_current_usage(normalized)
    return normalized


def _normalize_usage(
    usage: Any,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    total_names: tuple[str, ...],
) -> ProviderTokenUsage | None:
    if usage is None:
        return None
    input_tokens = _first_count(usage, input_names)
    output_tokens = _first_count(usage, output_names)
    total_tokens = _first_count(usage, total_names)
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return ProviderTokenUsage(input_tokens, output_tokens, total_tokens)


def _first_count(value: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        result = _field(value, name)
        if isinstance(result, bool):
            continue
        if isinstance(result, int) and result >= 0:
            return result
        if isinstance(result, str) and result.isdecimal():
            return int(result)
    return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _set_current_usage(usage: ProviderTokenUsage | None) -> None:
    capture = _CURRENT_CAPTURE.get()
    if capture is not None:
        capture.usage = usage
