class RetryableLLMError(Exception):
    """A provider failure that allows trying the next provider."""


class NonRetryableLLMError(Exception):
    """A provider failure that should stop routing immediately."""