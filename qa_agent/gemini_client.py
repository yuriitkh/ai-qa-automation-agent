from .browser_discovery import capture_page_snapshot, extract_target_url
from .models import QATestPlan
from .llm.gemini import GeminiProvider
from .llm.groq import GroqProvider
from .llm.router import LLMRouter


_router = LLMRouter([GeminiProvider(), GroqProvider()])


def create_test_plan(task: str) -> QATestPlan:
    target_url = extract_target_url(task)
    page_snapshot = capture_page_snapshot(target_url)
    print("PAGE SNAPSHOT")
    print(page_snapshot)
    return _router.create_test_plan(task, target_url, page_snapshot)


def get_selected_provider_name() -> str | None:
    return _router.selected_provider_name