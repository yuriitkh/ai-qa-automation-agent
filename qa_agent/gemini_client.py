import json

from .browser_discovery import capture_page_snapshot, extract_target_url
from .models import DiscoveryResult, DiscoveryStatus, QATestPlan
from .llm.gemini import GeminiProvider
from .llm.groq import GroqProvider
from .llm.router import LLMRouter
from .test_plan_generator import LLMTestPlanGenerator


_router = LLMRouter([GeminiProvider(), GroqProvider()])


def create_test_plan(task: str) -> QATestPlan:
    target_url = extract_target_url(task)
    page_snapshot = capture_page_snapshot(target_url)
    print("PAGE SNAPSHOT")
    print(page_snapshot)
    plan = QATestPlan.model_validate(
        _router.create_test_plan(task, target_url, page_snapshot)
    )
    snapshot = json.loads(page_snapshot)
    discovery = DiscoveryResult(
        status=DiscoveryStatus.SUCCESS,
        url=str(snapshot.get("url") or target_url),
        title=str(snapshot.get("title") or ""),
        snapshot=snapshot,
        interactive_elements=snapshot.get("interactive_elements", []),
    )
    LLMTestPlanGenerator._validate_discovery_capabilities(plan, discovery, task)
    return plan


def get_selected_provider_name() -> str | None:
    return _router.selected_provider_name
