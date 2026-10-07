class RetryableLLMError(Exception):
    """A provider failure that allows trying the next provider."""


class NonRetryableLLMError(Exception):
    """A provider failure that should stop routing immediately."""


class AllProvidersFailedError(RuntimeError):
    """A provider chain failed, with a safe machine-readable rate-limit signal."""

    def __init__(
        self,
        message: str,
        *,
        all_rate_limited: bool,
        all_timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.all_rate_limited = all_rate_limited
        self.all_timed_out = all_timed_out
