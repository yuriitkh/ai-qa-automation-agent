class RetryableLLMError(Exception):
    """A provider failure that allows trying the next provider."""


class NonRetryableLLMError(Exception):
    """A provider failure that should stop routing immediately."""


class AllProvidersFailedError(RuntimeError):
    """A provider chain failed, with a safe machine-readable rate-limit signal."""

    def __init__(self, message: str, *, all_rate_limited: bool) -> None:
        super().__init__(message)
        self.all_rate_limited = all_rate_limited
