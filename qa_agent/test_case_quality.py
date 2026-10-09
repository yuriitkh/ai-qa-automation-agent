"""Conservative definition checks at human approval, separate from saving."""
import re
from urllib.parse import urlsplit


class TestCaseQualityError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(" ".join(issues))


def quality_issues(test_case):
    issues = []
    placeholders = {"tbd", "todo", "test", "step", "unfinished step", "describe the action for this step", "describe the expected result", "the described behavior works as expected"}
    def meaningful(text):
        clean = text.strip().rstrip(".").casefold()
        return bool(re.search(r"[^\W\d_]{2}", clean, re.UNICODE)) and clean not in placeholders and not re.fullmatch(r"(?:step|test|draft|крок|тест)\s*\d+", clean)
    if not meaningful(test_case.name) or not meaningful(test_case.description):
        issues.append("Definition: provide a meaningful Summary and Scenario before approval.")
    if [segment.order for segment in test_case.segments] != list(range(len(test_case.segments))):
        issues.append("Segments: ordering must be contiguous.")
    if [step.order for step in test_case.steps] != list(range(len(test_case.steps))):
        issues.append("Steps: ordering must be contiguous.")
    if len({step.id for step in test_case.steps}) != len(test_case.steps):
        issues.append("Steps: duplicate IDs are not allowed.")
    for step in test_case.steps:
        if not meaningful(step.description):
            issues.append(f"Step {step.order + 1}: provide a meaningful Description.")
        if not meaningful(step.expected):
            issues.append(f"Step {step.order + 1}: provide an Expected Result; for an action-only requirement, describe the completed action.")
    for index, segment in enumerate(test_case.segments):
        url = segment.base_url or test_case.base_url
        if url:
            try:
                parsed = urlsplit(url)
                _ = parsed.port
                valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username and not parsed.password and not any(character.isspace() or ord(character) < 32 for character in url)
            except ValueError:
                valid = False
            if not valid:
                issues.append(f"Segment {index + 1}: use an HTTP or HTTPS URL without credentials.")
        if segment.is_implicit and segment.base_url and test_case.base_url and segment.base_url != test_case.base_url:
            issues.append(f"Segment {index + 1}: its implicit URL must agree with the TestCase URL.")
    return issues


def validate_test_case_quality(test_case):
    issues = quality_issues(test_case)
    if issues:
        raise TestCaseQualityError(issues)
