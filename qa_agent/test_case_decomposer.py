import re
from uuid import NAMESPACE_URL, uuid5

from qa_agent.models import TestCase, TestStep


class TestCaseDecomposer:
    """Create basic atomic steps from common natural-language test tasks.

    This deterministic implementation establishes the domain boundary. It does
    not attempt to infer browser selectors or executable TestPlans.
    """

    _NAVIGATION_PATH = re.compile(
        r"(?:select|navigate to)\s+(.+?)(?=,|\bthen\b|\bverify\b|$)",
        re.IGNORECASE,
    )
    _ARROW = re.compile(r"\s*(?:→|->)\s*")

    def decompose(self, task: str, base_url: str | None = None) -> TestCase:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        normalized_task = task.strip()
        test_case_id = uuid5(
            NAMESPACE_URL,
            f"ai-qa:test-case:{base_url or ''}\0{normalized_task}",
        )
        candidates: list[tuple[str, str, str]] = []
        lowered = normalized_task.casefold()

        if "homepage" in lowered or "home page" in lowered:
            candidates.append((
                "Open homepage",
                "Open the homepage",
                "The homepage is loaded",
            ))

        if re.search(r"\b(open|display|show)\b.{0,30}\b(menu|navigation)\b", lowered):
            candidates.append((
                "Open main navigation",
                "Open the main navigation menu",
                "The main navigation menu is visible",
            ))

        path_match = self._NAVIGATION_PATH.search(normalized_task)
        if path_match:
            path = path_match.group(1).strip().rstrip(" .")
            path = re.sub(r"\s+then\s+", " → ", path, flags=re.IGNORECASE)
            path_parts = [part.strip() for part in self._ARROW.split(path) if part.strip()]
            if path_parts:
                path = " → ".join(path_parts)
                destination = path_parts[-1]
                candidates.append((
                    f"Open {path}",
                    f"Open {path}",
                    f"{destination} page is opened",
                ))

        destination = path_parts[-1] if path_match and path_parts else None
        if re.search(r"\b(verify|check|assert)\b.{0,30}\burl\b|\burl\b.{0,30}\b(verify|check|assert)\b", lowered):
            expected_url = self._expected_value(normalized_task, "url")
            expected = (
                f"URL is {expected_url}"
                if expected_url
                else f"URL is the expected {destination} URL" if destination
                else "The URL matches the expected URL"
            )
            candidates.append((
                "Verify URL",
                "Verify the URL",
                expected,
            ))

        if re.search(r"\b(verify|check|assert)\b.{0,40}\b(heading| h1|page title)\b|\b(heading|page title)\b.{0,40}\b(verify|check|assert)\b", lowered):
            heading = self._expected_value(normalized_task, "heading") or destination
            candidates.append((
                "Verify page heading",
                "Verify the page heading",
                f"Page heading is {heading}" if heading else "The page heading matches the expected heading",
            ))

        if not candidates:
            # A single broad request remains a single step; richer decomposition
            # is intentionally deferred until an LLM-backed implementation exists.
            description = normalized_task.rstrip(" .")
            expected = self._simple_expected(description)
            candidates.append((self._short_name(description), description, expected))

        steps = [
            TestStep(
                id=uuid5(
                    test_case_id,
                    f"step:{index}\0{name}\0{description}\0{expected}",
                ),
                name=name,
                description=description,
                expected=expected,
                order=index,
            )
            for index, (name, description, expected) in enumerate(candidates)
        ]
        return TestCase(
            id=test_case_id,
            name=self._short_name(normalized_task),
            description=normalized_task,
            base_url=base_url,
            steps=steps,
        )

    @staticmethod
    def _expected_value(task: str, kind: str) -> str | None:
        if kind == "url":
            match = re.search(r"(?:expected\s+)?url\s*(?:is|to be|:)?\s*([\"']https?://[^\"']+[\"']|https?://\S+)", task, re.IGNORECASE)
        else:
            match = re.search(r"(?:heading|h1|page title)\s*(?:is|to be|:)?\s*[\"']([^\"']+)[\"']", task, re.IGNORECASE)
        return match.group(1).strip("\"'") if match else None

    @staticmethod
    def _simple_expected(description: str) -> str:
        heading = re.search(r"(?:heading|page title)\s+(?:is\s+)?[\"']?([^\"']+)[\"']?", description, re.IGNORECASE)
        if heading:
            return f"Page heading is {heading.group(1).rstrip('.')}"
        return "The requested condition is satisfied"

    @staticmethod
    def _short_name(value: str) -> str:
        first_sentence = re.split(r"[.!?]", value, maxsplit=1)[0].strip()
        return first_sentence[:80] or "Test case"
