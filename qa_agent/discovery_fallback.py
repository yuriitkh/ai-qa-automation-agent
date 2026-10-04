"""Provider-independent AI-assisted Discovery fallback."""

import json
from abc import ABC, abstractmethod

from qa_agent.models import AIDiscoveryResult, DiscoveryResult, TestStep


class DiscoveryFallback(ABC):
    @abstractmethod
    def discover(self, task: str, target_url: str, test_step: TestStep,
                 deterministic_result: DiscoveryResult) -> AIDiscoveryResult:
        raise NotImplementedError


class RouterDiscoveryFallback(DiscoveryFallback):
    def __init__(self, router) -> None:
        self._router = router

    def discover(self, task, target_url, test_step, deterministic_result):
        context = {
            "deterministic_result": deterministic_result.model_dump(mode="json"),
            "task": task,
            "test_step": test_step.model_dump(mode="json"),
            "target_url": target_url,
            "warnings": deterministic_result.warnings,
        }
        return AIDiscoveryResult.model_validate(
            self._router.create_discovery(task, target_url,
                json.dumps(context, ensure_ascii=False))
        )
