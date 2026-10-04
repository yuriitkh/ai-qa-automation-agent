from abc import ABC, abstractmethod

from ..models import AIDiscoveryResult, QATestPlan


class LLMProvider(ABC):
    @property
    def is_available(self) -> bool:
        return True

    @abstractmethod
    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        raise NotImplementedError

    def create_discovery(self, task: str, target_url: str, page_snapshot: str) -> AIDiscoveryResult:
        """Optional structured Discovery capability; providers may implement it."""
        raise NotImplementedError("This LLM provider does not support Discovery output.")
