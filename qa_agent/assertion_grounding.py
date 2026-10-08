"""Conservative, deterministic grounding for generated plan assertions.

Grounding uses only the TestCase requirement supplied by the caller, the
atomic TestStep when no broader requirement is available, and page evidence
captured by deterministic browser discovery. AI discovery suggestions are not
read from ``DiscoveryResult``'s typed suggestion fields.
"""

import re
from collections.abc import Iterable

from qa_agent.models import (
    AssertionGrounding,
    AssertionGroundingEntry,
    DiscoveryResult,
    DiscoveryStatus,
    QATestPlan,
    TestStep,
)
from qa_agent.test_plan_validation import PlanValidationError, PlanValidationIssue


_ASSERTION_ACTIONS = frozenset({
    "assert_page_loaded",
    "assert_title",
    "assert_visible",
    "assert_hidden",
    "assert_url",
    "assert_text_contains",
    "assert_checked",
    "assert_unchecked",
    "assert_selected",
    "assert_enabled",
    "assert_disabled",
})

_EXACT_ASSERTION_VALUES: dict[str, tuple[str, ...]] = {
    "assert_title": ("expected",),
    "assert_visible": ("expected_text",),
    "assert_url": ("expected",),
    "assert_text_contains": ("expected_text",),
    "assert_selected": ("expected",),
}

_EXAMPLE_TAIL = re.compile(
    r"(?is)(?:\be\s*\.\s*g\s*\.?|\bfor example\b|\bsuch as\b)"
    r"[^.;\n]*(?:[.;\n]|$)"
)

_STRUCTURAL_TERMS: dict[str, tuple[str, ...]] = {
    "assert_page_loaded": ("load", "opened", "open", "available"),
    "assert_visible": ("visible", "display", "displayed", "displays", "shown", "present", "appear", "appears", "state", "status"),
    "assert_hidden": ("hidden", "not visible", "disappear", "not displayed"),
    "assert_checked": ("checked", "ticked"),
    "assert_unchecked": ("unchecked", "not checked", "not be checked", "not ticked", "not be ticked"),
    "assert_selected": ("selected", "chosen"),
    "assert_enabled": ("enabled",),
    "assert_disabled": ("disabled",),
}

_GENERIC_REQUIREMENT_WORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "then", "when",
    "after", "before", "into", "onto", "user", "test", "step", "verify",
    "check", "ensure", "result",
})


def classify_assertions(
    plan: QATestPlan,
    test_step: TestStep,
    discovery: DiscoveryResult,
    *,
    requirement_context: str | None = None,
) -> tuple[AssertionGroundingEntry, ...]:
    """Return value-free grounding categories, one entry per assertion action."""
    requirements = _requirement_texts(test_step, requirement_context)
    observation_texts = _deterministic_observation_texts(discovery)
    entries: list[AssertionGroundingEntry] = []

    for index, action in enumerate(plan.steps):
        if action.action not in _ASSERTION_ACTIONS:
            continue
        values = [
            action.parameters.get(field)
            for field in _EXACT_ASSERTION_VALUES.get(action.action, ())
        ]
        values = [value for value in values if isinstance(value, str) and value.strip()]
        if values:
            category = AssertionGrounding.UNKNOWN
            if any(
                _required_value(action.action, value, requirements)
                for value in values
            ):
                category = AssertionGrounding.REQUIREMENT_GROUNDED
            elif any(
                _observed_value(action.action, value, observation_texts)
                for value in values
            ):
                category = AssertionGrounding.OBSERVATION_GROUNDED
            else:
                category = AssertionGrounding.INFERRED
        elif _structural_assertion_is_required(action.action, requirements):
            category = AssertionGrounding.REQUIREMENT_GROUNDED
        else:
            category = AssertionGrounding.UNKNOWN
        entries.append(AssertionGroundingEntry(step_index=index, category=category))
    return tuple(entries)


def validate_assertion_grounding(
    plan: QATestPlan,
    test_step: TestStep,
    discovery: DiscoveryResult,
    *,
    requirement_context: str | None = None,
) -> tuple[AssertionGroundingEntry, ...]:
    """Reject ungrounded exact assertions without revealing asserted values."""
    entries = classify_assertions(
        plan,
        test_step,
        discovery,
        requirement_context=requirement_context,
    )
    issues = [
        PlanValidationIssue(
            code="UNGROUNDED_ASSERTION",
            path=f"steps[{entry.step_index}].parameters",
            message=(
                "An exact assertion value is not stated in the TestCase requirement "
                "or supported by deterministic page evidence. Use a structural "
                "assertion unless the exact value is required or observed."
            ),
        )
        for entry in entries
        if entry.category == AssertionGrounding.INFERRED
    ]
    if issues:
        raise PlanValidationError(issues)
    return entries


def _requirement_texts(
    test_step: TestStep,
    requirement_context: str | None,
) -> tuple[str, ...]:
    if isinstance(requirement_context, str) and requirement_context.strip():
        step_texts = tuple(
            value for value in (test_step.name, test_step.description, test_step.expected)
            if isinstance(value, str) and value.strip()
        )
        if _requirement_context_covers_step(requirement_context, step_texts):
            # A related original scenario is authoritative about specificity;
            # generated or elaborated step wording cannot upgrade it.
            return (requirement_context,)
        # Some manually authored TestCases use a short workflow summary that
        # does not state each step's requirement. In that case, retain the
        # explicit atomic TestStep requirements as well.
        return (requirement_context, *step_texts)
    return tuple(
        value for value in (test_step.name, test_step.description, test_step.expected)
        if isinstance(value, str) and value.strip()
    )


def _requirement_context_covers_step(context: str, step_texts: Iterable[str]) -> bool:
    context_tokens = _significant_terms(context)
    if not context_tokens:
        return False
    step_tokens = set().union(*(_significant_terms(text) for text in step_texts))
    return bool(context_tokens & step_tokens)


def _significant_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9]+", text.casefold()))
    normalized = set()
    for term in terms:
        if term in _GENERIC_REQUIREMENT_WORDS or len(term) <= 2:
            continue
        if term in {"confirmation", "confirmations", "confirmed", "unconfirmed", "confirming"}:
            term = "confirm"
        elif term in {"status", "state"}:
            term = "state"
        elif term in {
            "display", "displayed", "displaying", "visible", "visibility",
            "show", "shown", "present", "appeared", "appears",
        }:
            term = "visibility"
        if term.endswith("ing") and len(term) > 5:
            term = term[:-3]
        elif term.endswith("ed") and len(term) > 4:
            term = term[:-2]
        normalized.add(term)
    return normalized


def _deterministic_observation_texts(discovery: DiscoveryResult) -> tuple[str, ...]:
    values = [discovery.url, discovery.title]
    snapshot = discovery.snapshot
    # ``snapshot`` is the browser collector's bounded deterministic payload.
    # Typed navigation/element collections may also contain AI suggestions.
    for key in (
        "headings", "links", "buttons", "visible_text_elements",
        "navigation_paths", "direct_navigation_paths",
    ):
        collection = snapshot.get(key)
        if isinstance(collection, list):
            for item in collection:
                if isinstance(item, dict):
                    value = item.get("text")
                    if isinstance(value, str):
                        values.append(value)
                    for nested_key in (
                        "expected_url", "heading_text", "href", "resolved_url",
                    ):
                        nested_value = item.get(nested_key)
                        if isinstance(nested_value, str):
                            values.append(nested_value)
                    steps = item.get("steps")
                    if isinstance(steps, list):
                        for step in steps:
                            if isinstance(step, dict):
                                for nested_key in ("text", "href", "resolved_url"):
                                    nested_value = step.get(nested_key)
                                    if isinstance(nested_value, str):
                                        values.append(nested_value)
    # DiscoveryFallback only merges AI suggestions when the deterministic
    # result is not SUCCESS. Typed navigation records are therefore safe to
    # use as additional evidence only for an unmodified successful result.
    if discovery.status == DiscoveryStatus.SUCCESS:
        for path in (*discovery.navigation_paths, *discovery.direct_navigation_paths):
            for field in ("menu_tab_text", "menu_item_text", "submenu_text", "expected_url", "heading_text"):
                value = getattr(path, field, None)
                if isinstance(value, str):
                    values.append(value)
            for step in getattr(path, "steps", ()):
                for field in ("text", "href", "resolved_url"):
                    value = getattr(step, field, None)
                    if isinstance(value, str):
                        values.append(value)
        for element in discovery.interactive_elements:
            values.extend((element.text, element.accessible_name))
    return tuple(value for value in values if isinstance(value, str) and value.strip())


def _required_literal(value: str, requirements: Iterable[str]) -> bool:
    for requirement in requirements:
        text = _EXAMPLE_TAIL.sub(" ", requirement)
        normalized_text = " ".join(text.casefold().split())
        normalized_value = " ".join(value.casefold().split())
        if not normalized_value:
            continue
        for match in re.finditer(
            rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized_text
        ):
            prefix = normalized_text[max(0, match.start() - 80):match.start()]
            if re.search(
                r"\b(?:not|never|without|avoid|exclude|except|no|rather than|instead of)"
                r"\b(?:\W+\w+){0,3}\W*$",
                prefix,
            ):
                continue
            return True
    return False


def _required_value(action: str, value: str, requirements: Iterable[str]) -> bool:
    if action == "assert_url":
        expected = _normalized_url(value)
        return any(
            _normalized_url(candidate) == expected
            for requirement in requirements
            for candidate in re.findall(r"https?://[^\s\"'<>]+", requirement, re.IGNORECASE)
        )
    return _required_literal(value, requirements)


def _observed_literal(value: str, observations: Iterable[str]) -> bool:
    # The assertion is a contains check, so a concrete phrase included in a
    # larger deterministic observed string is sufficient evidence.
    return any(value in observation for observation in observations)


def _observed_value(
    action: str,
    value: str,
    observations: Iterable[str],
) -> bool:
    if action == "assert_text_contains":
        return _observed_literal(value, observations)
    if action in {"assert_url", "assert_title", "assert_visible", "assert_selected"}:
        if action == "assert_selected":
            # Current discovery snapshots expose labels, not selected state;
            # seeing an option label is not evidence that it is selected.
            return False
        expected = _normalized_observed_value(value)
        return any(_normalized_observed_value(item) == expected for item in observations)
    return _observed_literal(value, observations)


def _normalized_url(value: str) -> str:
    return value.strip().rstrip(".,;:)")


def _normalized_observed_value(value: str) -> str:
    return " ".join(value.strip().split()).rstrip(".,;:")


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(text.casefold().split())
    normalized_phrase = " ".join(phrase.casefold().split())
    if not normalized_phrase:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text
    ) is not None


def _structural_assertion_is_required(
    action: str,
    requirements: Iterable[str],
) -> bool:
    terms = _STRUCTURAL_TERMS.get(action, ())
    for requirement in requirements:
        text = _EXAMPLE_TAIL.sub(" ", requirement).casefold()
        if action == "assert_checked" and re.search(
            r"\b(?:unchecked|not\s+(?:be\s+)?(?:checked|ticked))\b", text
        ):
            continue
        if any(_contains_phrase(text, term) for term in terms):
            return True
    return False
