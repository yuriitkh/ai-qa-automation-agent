"""Manual, quota-consuming OpenRouter smoke test; excluded from normal suite."""
import os

import pytest

from qa_agent.llm.openai_compatible import OpenAICompatibleProvider


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not configured")
def test_openrouter_generates_a_plan():
    provider = OpenAICompatibleProvider(
        "openrouter", "OPENROUTER_API_KEY",
        os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    result = provider.create_test_plan(
        "Check that the page loads at https://example.com",
        "https://example.com",
        '{"url":"https://example.com","title":"Example Domain","headings":[]}',
    )
    assert result.url == "https://example.com"
