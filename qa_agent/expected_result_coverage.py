"""Deterministic mapping from TestStep expected results to plan assertions.

Coverage uses the current TestStep, immutable actions and fingerprint-bound
Discovery subject matches. Assertion provenance remains a separate gate: a
relevant assertion can still be ungrounded, and a grounded one irrelevant.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    QATestPlan,
    TestCase,
    TestPlanVersion,
    TestStep,
)
from qa_agent.test_plan_validation import PlanValidationError, PlanValidationIssue


class ExpectedResultCoverageStatus(str, Enum):
    NO_VERIFICATION_REQUIRED = "NO_VERIFICATION_REQUIRED"
    COVERED = "COVERED"
    PARTIALLY_COVERED = "PARTIALLY_COVERED"
    NOT_COVERED = "NOT_COVERED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ExpectedResultCoverage:
    status: ExpectedResultCoverageStatus
    verification_required: bool
    matching_action_indexes: tuple[int, ...] = ()
    total_expectations: int = 0

    @property
    def is_sufficient(self) -> bool:
        return self.status in {
            ExpectedResultCoverageStatus.NO_VERIFICATION_REQUIRED,
            ExpectedResultCoverageStatus.COVERED,
        }

    @property
    def safe_message(self) -> str:
        return "Automation does not verify the TestStep expected result."


class ExpectedResultCoverageError(ValueError):
    """Safe failure raised when a workflow requires complete coverage."""

    def __init__(self) -> None:
        super().__init__("Automation does not verify the TestStep expected result.")


@dataclass(frozen=True)
class _Expectation:
    kind: str
    clause: str
    subject_tokens: frozenset[str]
    exact_values: tuple[str, ...] = ()


_ASSERTION_ACTIONS = frozenset({
    "assert_page_loaded", "assert_title", "assert_visible", "assert_hidden",
    "assert_url", "assert_text_contains", "assert_checked", "assert_unchecked", "assert_selected",
    "assert_enabled", "assert_disabled",
})

_VERIFICATION_INTENT = re.compile(
    r"\b(?:verify|validate|check|confirm|ensure|assert|make sure|test that)\b",
    re.IGNORECASE,
)
_ACTION_STEP = re.compile(
    r"^\s*(?:please\s+)?(?:open|navigate|go|click|press|enter|fill|type|select|choose|submit|save)\b",
    re.IGNORECASE,
)
_OUTPUT_STATE = re.compile(
    r"\b(?:displayed|visible|shown|appears?|present|hidden|disappears?|"
    r"redirected|redirection|title|url|contains?|includes?|checked|selected|"
    r"unchecked|not\s+(?:be\s+)?(?:checked|ticked)|disabled|enabled|loaded|(?:page|site|homepage)\s+loads|exists?|created|saved|submitted|updated|deleted|"
    r"removed|added|accepted|rejected|authenticated|logged\s+in|signed\s+in|"
    r"error|success|confirmation|notification|alert|warning|message|notice|"
    r"available|unavailable|empty|cleared|expanded|collapsed|selected|"
    r"(?:is|are)\s+correct|matches?|equals?)\b",
    re.IGNORECASE,
)
_ACTION_ONLY_EXPECTATION = re.compile(
    r"\b(?:opened|open|navigated|went|clicked|pressed|entered|filled|typed|"
    r"selected|chosen|submitted|saved)\b",
    re.IGNORECASE,
)
_ACTION_COMPLETION = re.compile(
    r"^\s*(?:(?:the\s+)?(?:[\w-]+\s+){0,3}(?:action|step|interaction)"
    r"(?:\s+[\w-]+){0,3}\s+(?:is|was|has\s+been)\s+)?"
    r"(?:done|completed|performed|processed)(?:\s+successfully)?\s*[.!]?\s*$",
    re.I,
)

_HIDDEN = re.compile(r"\b(?:hidden|invisible|not\s+(?:visible|displayed|shown)|disappears?)\b", re.I)
_REDIRECT = re.compile(r"\b(?:redirect(?:ed|s|ing)?|routed|navigated\s+to|lands?\s+on)\b", re.I)
_URL = re.compile(r"\b(?:url|address|path|route)\b|https?://", re.I)
_TITLE = re.compile(r"\b(?:page\s+)?title\b", re.I)
_DISABLED = re.compile(r"\bdisabled\b", re.I)
_ENABLED = re.compile(r"\benabled\b", re.I)
_UNCHECKED = re.compile(r"\b(?:unchecked|not\s+(?:be\s+)?(?:checked|ticked))\b", re.I)
_CHECKED = re.compile(r"\b(?:checked|ticked)\b", re.I)
_SELECTED = re.compile(r"\b(?:selected|chosen)\b", re.I)
_VISIBLE = re.compile(r"\b(?:displayed|visible|shown|appears?|present|exists?)\b", re.I)
_LOADED = re.compile(r"\b(?:loaded|(?:page|site|homepage)\s+loads|open|opened|available)\b", re.I)
_TEXT = re.compile(r"\b(?:text|message|error|success|confirmation|notification|alert|warning|notice|contains?|includes?|says?)\b", re.I)
_NO_ERROR = re.compile(
    r"\b(?:without|no)\s+(?:(?:any|client-side|validation)\s+){0,2}errors?\b",
    re.I,
)
_RESULT = re.compile(
    r"\b(?:created|saved|submitted|updated|deleted|removed|added|accepted|rejected|"
    r"authenticated|logged\s+in|signed\s+in|empty|cleared|expanded|collapsed)\b",
    re.I,
)

_QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,})[\"'“”‘’]")
_WORD = re.compile(r"[a-z0-9]+", re.I)
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "not", "no", "without", "to", "of", "in", "on", "at",
    "by", "for", "from", "with", "after", "before", "when", "then", "that",
    "this", "it", "its", "is", "are", "be", "been", "being", "was", "were",
    "has", "have", "had", "will", "would", "should", "must", "can", "could",
    "may", "might", "page", "step", "expected", "result", "verify",
    "validate", "check", "confirm", "ensure", "assert", "make", "sure", "test",
    "displayed", "visible", "shown", "appears", "appear", "present", "hidden",
    "invisible", "disappears", "redirected", "redirect", "url", "address", "path",
    "route", "title", "contains", "contain", "includes", "include", "checked",
    "ticked", "unchecked", "selected", "chosen", "disabled", "enabled", "loaded", "loads", "open",
    "opened", "created", "saved", "submitted", "updated", "deleted", "removed",
    "added", "accepted", "rejected", "authenticated", "logged", "signed", "empty",
    "cleared", "expanded", "collapsed", "correct", "matches", "match", "equals",
    "equal", "available", "unavailable", "text", "says", "say",
})
_SYNONYMS = {
    "alert": "error", "alerts": "error", "errors": "error", "messages": "message", "notices": "notice",
    "notifications": "notification", "buttons": "button", "fields": "field",
    "inputs": "input", "emails": "email", "accounts": "account", "users": "user",
    "records": "record", "orders": "order", "redirects": "redirect",
    "redirected": "redirect", "redirection": "redirect", "shows": "show",
    "displayed": "display", "displays": "display", "visible": "visibility",
    "visibility": "visibility", "shown": "show", "appears": "appear",
    "appeared": "appear", "saved": "save", "created": "create", "submitted": "submit",
    "updated": "update", "deleted": "delete", "removed": "remove", "added": "add",
    "success": "confirmation", "successful": "confirmation", "successfully": "confirmation",
    "invalid": "error",
}
_GENERIC_SUBJECT_WORDS = frozenset({
    "message", "state", "status", "button", "checkbox", "field", "input",
    "element", "form", "notification", "notice", "radio", "control",
})


def has_error_absence_requirement(text: str) -> bool:
    """Recognize supported negative-error wording without interpreting locators."""
    return bool(_NO_ERROR.search(text))


def expected_result_coverage(
    test_step: TestStep,
    plan: QATestPlan | TestPlanVersion | None,
    *,
    discovery: DiscoveryResult | None = None,
) -> ExpectedResultCoverage:
    """Infer whether plan assertions cover the current expected result."""
    required = _verification_required(test_step)
    if required is False:
        return ExpectedResultCoverage(
            ExpectedResultCoverageStatus.NO_VERIFICATION_REQUIRED,
            verification_required=False,
        )
    if required is None:
        return ExpectedResultCoverage(
            ExpectedResultCoverageStatus.UNKNOWN,
            verification_required=True,
        )

    expectations = _expectations(test_step)
    if expectations is None:
        return ExpectedResultCoverage(
            ExpectedResultCoverageStatus.UNKNOWN,
            verification_required=True,
        )

    version = plan if isinstance(plan, TestPlanVersion) else None
    plan = version.qa_test_plan if version is not None else plan
    assertion_indexes = {
        index for index, action in enumerate(plan.steps if plan is not None else ())
        if action.action in _ASSERTION_ACTIONS
    }
    if not assertion_indexes:
        return ExpectedResultCoverage(
            ExpectedResultCoverageStatus.NOT_COVERED,
            verification_required=True,
            total_expectations=len(expectations),
        )

    matches = assertion_subject_matches(test_step, plan, discovery=discovery)
    if version is not None:
        fingerprint = coverage_fingerprint(test_step, plan)
        for entry in version.assertion_grounding or ():
            if entry.coverage_fingerprint == fingerprint and entry.step_index in assertion_indexes:
                matches[entry.step_index] = tuple(sorted(set(matches.get(entry.step_index, ())) | {
                    index for index in entry.covered_expectation_indexes if 0 <= index < len(expectations)
                }))
    covered_count = len({index for indexes in matches.values() for index in indexes})
    if covered_count == len(expectations):
        status = ExpectedResultCoverageStatus.COVERED
    elif covered_count:
        status = ExpectedResultCoverageStatus.PARTIALLY_COVERED
    else:
        status = ExpectedResultCoverageStatus.NOT_COVERED
    return ExpectedResultCoverage(
        status,
        verification_required=True,
        matching_action_indexes=tuple(sorted(index for index, covered in matches.items() if covered)),
        total_expectations=len(expectations),
    )


def validate_expected_result_coverage(
    test_step: TestStep,
    plan: QATestPlan | TestPlanVersion,
    *,
    discovery: DiscoveryResult | None = None,
) -> ExpectedResultCoverage:
    """Reject required expected results that the executable plan does not cover."""
    coverage = expected_result_coverage(test_step, plan, discovery=discovery)
    if not coverage.is_sufficient:
        guidance = (
            "Clarify the observable expected state; the coverage matcher cannot safely interpret this result."
            if coverage.status == ExpectedResultCoverageStatus.UNKNOWN else
            "Add a relevant supported assertion for every expected state; successful actions alone are not verification."
        )
        if has_error_absence_requirement(test_step.expected):
            guidance += (
                " For 'without error', assert_hidden must check an error element established by Discovery"
                " after the input actions. If none is observed, review Discovery or clarify the check; do not invent a selector."
            )
        raise PlanValidationError([
            PlanValidationIssue(
                code="EXPECTED_RESULT_NOT_COVERED",
                path="steps",
                message=f"{coverage.safe_message} {guidance}",
            )
        ])
    return coverage


def test_case_coverage(test_case: TestCase, plan_store) -> tuple[tuple[TestStep, ExpectedResultCoverage], ...]:
    """Derive coverage for every current TestStep without persisting a copy."""
    results = []
    for step in sorted(test_case.steps, key=lambda item: item.order):
        version = plan_store.find(step.id)
        results.append((
            step,
            expected_result_coverage(
                step,
                version,
            ),
        ))
    return tuple(results)


def has_sufficient_test_case_coverage(test_case: TestCase, plan_store) -> bool:
    """Return whether each step's required expected result has plan coverage."""
    coverage = test_case_coverage(test_case, plan_store)
    return bool(coverage) and all(item.is_sufficient for _, item in coverage)


def _verification_required(test_step: TestStep) -> bool | None:
    combined = " ".join((test_step.name, test_step.description, test_step.expected))
    if _VERIFICATION_INTENT.search(combined):
        return True
    if has_error_absence_requirement(test_step.expected):
        return True
    if _ACTION_COMPLETION.fullmatch(test_step.expected):
        return False
    if _OUTPUT_STATE.search(test_step.expected):
        if _action_only_result(test_step):
            return False
        return True
    if _action_only_result(test_step):
        return False
    # Unrecognized results are not evidence of action-only intent.
    return None


def _action_only_result(test_step: TestStep) -> bool:
    return bool(
        _ACTION_STEP.search(test_step.name)
        and _ACTION_ONLY_EXPECTATION.search(test_step.expected)
        and all(_ACTION_ONLY_EXPECTATION.search(clause) for clause in _result_clauses(test_step.expected))
        and not re.search(
            r"\b(?:error|success|message|confirmation|notification|alert|warning|notice|"
            r"redirect|disabled|enabled|checked|selected|title|url|exists?|created|saved|"
            r"submitted|updated|deleted|removed|added|accepted|rejected|authenticated|"
            r"logged\s+in|signed\s+in|contains?|includes?|visible|displayed|shown|hidden)\b",
            test_step.expected,
            re.I,
        )
    )


def _expectations(test_step: TestStep) -> tuple[_Expectation, ...] | None:
    clauses = _result_clauses(test_step.expected)
    if not clauses:
        return None

    expectations: list[_Expectation] = []
    for clause in clauses:
        kind = _expectation_kind(clause)
        if kind is None:
            return None
        exact_values = tuple(match.group(1).strip() for match in _QUOTED.finditer(clause))
        # Actions can name a different element from the expected result.
        tokens = _subject_tokens(clause)
        expectations.append(_Expectation(kind, clause, frozenset(tokens), exact_values))
        if kind == "no_error":
            absence = _NO_ERROR.search(clause)
            # 'No errors are displayed' describes only absence. In contrast,
            # 'Confirmation is displayed without error' also requires the
            # confirmation assertion; negation must not erase that state.
            if absence.start() or absence.group().casefold().startswith("without"):
                remaining = (clause[:absence.start()] + clause[absence.end():]).strip(" ,.;")
                other_kind = _expectation_kind(remaining)
                if other_kind is not None:
                    expectations.append(_Expectation(
                        other_kind, remaining, frozenset(_subject_tokens(remaining)),
                        tuple(match.group(1).strip() for match in _QUOTED.finditer(remaining)),
                    ))
    return tuple(expectations)


def _result_clauses(expected: str) -> list[str]:
    return [
        clause.strip(" .;\t")
        for clause in re.split(r"\s+\b(?:and|but|also)\b\s+", expected, flags=re.I)
        if clause.strip(" .;\t")
    ]


def _subject_tokens(clause: str) -> set[str]:
    tokens = set(_semantic_tokens(clause))
    # 'Account confirmation' describes the confirmation, not any account node.
    outcome_subject = tokens & {"confirmation", "error"}
    if outcome_subject:
        return outcome_subject
    specific = tokens - _GENERIC_SUBJECT_WORDS
    return specific or tokens


def coverage_fingerprint(test_step: TestStep, plan: QATestPlan) -> str:
    payload = [
        str(test_step.id), test_step.name, test_step.description, test_step.expected,
        plan.model_dump(mode="json"),
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assertion_subject_matches(
    test_step: TestStep,
    plan: QATestPlan,
    *,
    discovery: DiscoveryResult | None = None,
) -> dict[int, tuple[int, ...]]:
    """Match expected subjects to assertions, using only bound DOM evidence."""
    expectations = _expectations(test_step)
    if expectations is None:
        return {}
    matches = {}
    for index, action in enumerate(plan.steps):
        if action.action not in _ASSERTION_ACTIONS:
            continue
        matches[index] = tuple(
            expected_index for expected_index, expectation in enumerate(expectations)
            if _action_covers(expectation, action.action, action.parameters, plan, discovery)
            and (expectation.kind != "no_error" or not any(
                later.action not in _ASSERTION_ACTIONS for later in plan.steps[index + 1:]
            ))
        )
    return matches


def _expectation_kind(clause: str) -> str | None:
    # Negation takes precedence over words such as 'error' or 'displayed'.
    # A positive error-message assertion proves the opposite of this result.
    if has_error_absence_requirement(clause):
        return "no_error"
    if _HIDDEN.search(clause):
        return "hidden"
    if _DISABLED.search(clause):
        return "disabled"
    if _ENABLED.search(clause):
        return "enabled"
    if _UNCHECKED.search(clause):
        return "unchecked"
    if _CHECKED.search(clause):
        return "checked"
    if _SELECTED.search(clause):
        return "selected"
    if _REDIRECT.search(clause):
        return "redirect"
    if _TITLE.search(clause):
        return "title"
    if _URL.search(clause):
        return "url"
    if _VISIBLE.search(clause):
        return "visible"
    if _LOADED.search(clause):
        return "loaded"
    if _TEXT.search(clause):
        return "text"
    if _RESULT.search(clause):
        return "result"
    return None


def _action_covers(
    expectation: _Expectation,
    action: str,
    parameters: dict,
    plan: QATestPlan | None,
    discovery: DiscoveryResult | None = None,
) -> bool:
    allowed = {
        "no_error": {"assert_hidden"},
        "hidden": {"assert_hidden"},
        "disabled": {"assert_disabled"},
        "enabled": {"assert_enabled"},
        "checked": {"assert_checked"},
        "unchecked": {"assert_unchecked"},
        "selected": {"assert_selected"},
        "redirect": {"assert_url", "assert_page_loaded"},
        "url": {"assert_url"},
        "title": {"assert_title"},
        "visible": {"assert_visible", "assert_text_contains"},
        "loaded": {"assert_page_loaded", "assert_url"},
        "text": {"assert_text_contains", "assert_visible"},
        "result": {"assert_text_contains", "assert_visible"},
    }.get(expectation.kind, set())
    if action not in allowed:
        return False

    if action == "assert_page_loaded":
        if expectation.kind == "loaded":
            return True
        return expectation.kind == "redirect" and _plan_url_matches(expectation, plan)
    if action == "assert_url" and expectation.kind == "loaded":
        return _plan_url_matches(expectation, plan, parameters.get("expected"))

    evidence = " ".join(
        str(parameters.get(key, ""))
        for key in ("expected", "expected_text", "selector")
        if isinstance(parameters.get(key), str)
    )
    evidence += " " + _discovered_assertion_subject(action, parameters, discovery)
    evidence_tokens = _semantic_tokens(evidence)
    if expectation.kind == "no_error":
        # Input selectors from the action's context cannot stand in for the
        # error subject. Locator identity is checked separately by Discovery.
        return "error" in evidence_tokens
    if any(
        exact.casefold() in evidence.casefold()
        for exact in expectation.exact_values
        if exact
    ):
        return True

    if (
        action == "assert_url"
        and expectation.kind == "url"
        and not expectation.subject_tokens
        and isinstance(parameters.get("expected"), str)
        and plan is not None
        and _normalize_url(parameters["expected"]) == _normalize_url(plan.url)
    ):
        return True

    expected_tokens = set(expectation.subject_tokens)
    evidence_tokens_set = set(evidence_tokens)
    if expectation.kind in {"redirect", "url"} and action == "assert_url":
        evidence_tokens_set.update(_url_tokens(str(parameters.get("expected", ""))))
    if not expected_tokens or not evidence_tokens_set:
        return False
    overlap = expected_tokens & evidence_tokens_set
    # Require an identifiable semantic subject; generic assertion words alone
    # never make an unrelated assertion count as coverage.
    return bool(overlap)


def _discovered_assertion_subject(
    action: str,
    parameters: dict,
    discovery: DiscoveryResult | None,
) -> str:
    if discovery is None:
        return ""
    selector = parameters.get("selector")
    asserted_text = parameters.get("expected_text")
    records = []
    for key in ("visible_text_elements", "state_elements", "interactive_elements", "inputs", "buttons", "links", "headings"):
        items = discovery.snapshot.get(key)
        if isinstance(items, (list, tuple)):
            records.extend(item for item in items if isinstance(item, dict))
    if discovery.status == DiscoveryStatus.SUCCESS:
        records.extend(item.model_dump() for item in discovery.interactive_elements)
    fields = ("selector", "tag", "kind", "role", "accessible_name", "label", "text", "id", "test_id", "name")
    evidence = []
    for record in records:
        if selector:
            if record.get("selector") != selector:
                continue
        elif (
            action != "assert_text_contains" or not isinstance(asserted_text, str)
            or not asserted_text or asserted_text not in str(record.get("text", ""))
        ):
            continue
        for field in fields:
            # A text assertion must prove its own value, not other page text.
            if field == "text" and (action == "assert_text_contains" or record.get("visible") is False):
                continue
            value = record.get(field)
            if field in {"role", "kind"} and isinstance(value, str) and value.casefold() == "alert":
                # Alert is also used for success notices; it is not an error
                # subject by itself. Require the actual identity or content.
                continue
            if str(record.get("tag", "")).casefold() in {"form", "body", "main", "nav"} and (
                field == "text" or (field == "accessible_name" and value == record.get("text"))
            ):
                # Ancestor text can describe a different child message. The
                # ancestor's visibility does not verify that message's state.
                continue
            if isinstance(value, str):
                evidence.append(value)
    return " ".join(evidence)


def _plan_url_matches(
    expectation: _Expectation,
    plan: QATestPlan | None,
    asserted_url: str | None = None,
) -> bool:
    url = asserted_url or (plan.url if plan is not None else "")
    url_tokens = _url_tokens(url)
    return bool(url_tokens and set(expectation.subject_tokens) & url_tokens)


def _url_tokens(value: str) -> set[str]:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return set(_semantic_tokens(value))
    return set(_semantic_tokens(" ".join((parsed.hostname or "", parsed.path, parsed.query))))


def _normalize_url(value: str) -> str:
    return value.strip().rstrip("/\t\r\n")


def _semantic_tokens(value: str) -> list[str]:
    normalized: list[str] = []
    separated = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    for raw in _WORD.findall(separated.casefold()):
        if raw in _STOP_WORDS:
            continue
        token = _SYNONYMS.get(raw, raw)
        if token in _STOP_WORDS or len(token) < 3:
            continue
        if token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return normalized
