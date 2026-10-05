"""Build the configured provider chain outside the core QA agent."""
import os

from .gemini import GeminiProvider
from .groq import GroqProvider
from .openai_compatible import OpenAICompatibleProvider
from .router import LLMRouter


def configured_provider_order() -> list[str]:
    raw = os.environ.get("LLM_PROVIDER_ORDER", "openai,gemini,openrouter,groq")
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def create_router() -> LLMRouter:
    compatible = {
        "openai": OpenAICompatibleProvider(
            "openai", "OPENAI_API_KEY", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        ),
        "openrouter": OpenAICompatibleProvider(
            "openrouter", "OPENROUTER_API_KEY",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
    }
    providers = {
        **compatible,
        "gemini": GeminiProvider(),
        "groq": GroqProvider(),
    }
    # Optional extra OpenAI-compatible provider: add its name to order and set
    # NAME_API_KEY, NAME_BASE_URL, and NAME_MODEL without editing this module.
    custom = [n.strip().lower() for n in os.getenv("LLM_COMPATIBLE_PROVIDERS", "").split(",") if n.strip()]
    for name in custom:
        prefix = name.upper()
        providers[name] = OpenAICompatibleProvider(
            name, f"{prefix}_API_KEY", os.getenv(f"{prefix}_MODEL", ""),
            os.getenv(f"{prefix}_BASE_URL"),
        )
    order = configured_provider_order()
    unknown = [name for name in order if name not in providers]
    if unknown:
        raise ValueError("Unknown LLM providers in LLM_PROVIDER_ORDER: " + ", ".join(unknown))
    return LLMRouter([providers[name] for name in order])
