"""Conservative, browser-independent matching for stale planned locators."""

from dataclasses import dataclass
from enum import Enum
import re

from qa_agent.models import (
    DiscoveryResult,
    InteractiveElement,
    LocatorIdentityEntry,
    QATestStep,
)


class RecoveryStatus(str, Enum):
    MATCHED_HIGH_CONFIDENCE = "MATCHED_HIGH_CONFIDENCE"
    MATCHED_ACCEPTABLE = "MATCHED_ACCEPTABLE"
    AMBIGUOUS = "AMBIGUOUS"
    REJECTED_CONFLICT = "REJECTED_CONFLICT"
    NO_MATCH = "NO_MATCH"

    # Compatibility names for callers that used the original three states.
    MATCHED = "MATCHED_HIGH_CONFIDENCE"
    NOT_FOUND = "NO_MATCH"

    @classmethod
    def _missing_(cls, value: object) -> "RecoveryStatus | None":
        # Historical trace JSON remains readable; old MATCHED records do not
        # participate in any new recovery decision.
        return {
            "MATCHED": cls.MATCHED_HIGH_CONFIDENCE,
            "NOT_FOUND": cls.NO_MATCH,
        }.get(value)


@dataclass(frozen=True)
class LocatorRecoveryResult:
    status: RecoveryStatus
    original_selector: str
    candidate: InteractiveElement | None = None
    signals: tuple[str, ...] = ()
    reason: str = ""


def recover_locator(
    planned_interaction: QATestStep,
    discovery_result: DiscoveryResult,
    original_identity: LocatorIdentityEntry | None = None,
) -> LocatorRecoveryResult:
    """Find one compatible candidate using original semantic identity evidence.

    Structure alone (tag/role) never identifies a replacement. Exact labels,
    accessible names, stable identifiers, link destinations, or the original
    visible click text can support a high-confidence match. A matching
    placeholder is accepted with lower confidence. Any known semantic
    disagreement vetoes a candidate, including one still found by the old CSS
    selector. Entered values are never considered.
    """
    if planned_interaction.action not in {"click", "check", "uncheck", "fill"}:
        raise ValueError("Locator recovery supports click, checkbox state, and fill interactions only.")

    params = planned_interaction.parameters
    selector = params.get("selector", "")
    if not isinstance(selector, str):
        selector = ""
    expected = _expected_identity(params, original_identity, planned_interaction.action)
    dialog_scope = _dialog_scope_prefix(selector)
    expected_dialog = expected.get("dialog_identity")
    candidates = [
        element for element in discovery_result.interactive_elements
        if _compatible(planned_interaction.action, element)
        and (
            expected_dialog is None
            or (
                element.selector == selector
                if expected_dialog == "unidentified-dialog"
                else element.dialog_identity == expected_dialog
            )
        )
        and (
            dialog_scope is None
            or dialog_scope.casefold() in element.selector.casefold()
        )
    ]

    exact_selector = [element for element in candidates if selector and element.selector == selector]
    if len(exact_selector) > 1:
        return _ambiguous(selector)

    high: list[tuple[InteractiveElement, tuple[str, ...]]] = []
    acceptable: list[tuple[InteractiveElement, tuple[str, ...]]] = []
    conflicts = False
    for element in candidates:
        conflict = _semantic_conflict(expected, element, planned_interaction.action)
        if conflict:
            conflicts = True
            continue
        stable_id, stable_signals = _stable_identity_match(expected, element)
        structure_compatible, structure_matches = _structure_matches(expected, element)
        semantic_signals = _semantic_matches(expected, element, planned_interaction.action)
        same_selector = bool(selector and element.selector == selector)

        # An exact stable ID/test ID can safely outweigh a weak tag/role change,
        # but never the explicit semantic conflicts checked above.
        if stable_id:
            high.append((element, tuple(dict.fromkeys((*stable_signals, *structure_matches)))))
            continue
        if not structure_compatible:
            continue
        if semantic_signals:
            high.append((element, tuple(dict.fromkeys((*semantic_signals, *structure_matches)))))
            continue
        if _placeholder_match(expected, element):
            acceptable.append((element, tuple(dict.fromkeys(("exact_placeholder", *structure_matches)))))
            continue

        # A legacy plan has no captured evidence to compare. Reusing its exact
        # selector does not redirect the action; a different selector cannot be
        # selected on structural similarity alone.
        if same_selector and not _has_semantic_expectation(expected):
            high.append((element, ("exact_selector", *structure_matches)))

    matches = high if high else acceptable
    if len(matches) == 1:
        element, signals = matches[0]
        status = (
            RecoveryStatus.MATCHED_HIGH_CONFIDENCE
            if high else RecoveryStatus.MATCHED_ACCEPTABLE
        )
        return LocatorRecoveryResult(
            status, selector, element, signals,
            "One compatible element matches the saved identity evidence.",
        )
    if len(matches) > 1:
        return _ambiguous(selector)
    if conflicts:
        return LocatorRecoveryResult(
            RecoveryStatus.REJECTED_CONFLICT, selector,
            reason="Discovered controls conflict with the saved control identity.",
        )
    return LocatorRecoveryResult(
        RecoveryStatus.NO_MATCH, selector,
        reason="No safe locator replacement was found; automation drift needs attention.",
    )


def identity_for_element(step_index: int, element: InteractiveElement, action: str) -> LocatorIdentityEntry:
    """Capture bounded, non-value identity data from deterministic discovery."""
    return LocatorIdentityEntry(
        step_index=step_index,
        accessible_name=_clean(element.accessible_name),
        label=_clean(element.label),
        placeholder=_clean(element.placeholder),
        visible_text=_clean(element.text) if action == "click" else None,
        test_id=_clean(element.test_id),
        element_id=_clean(element.id),
        name=_clean(element.name),
        href=_clean(element.href) if action == "click" else None,
        dialog_identity=_clean(element.dialog_identity),
        tag=_clean(element.tag),
        role=_clean(element.role),
    )


def _expected_identity(
    parameters: dict,
    original: LocatorIdentityEntry | None,
    action: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    if original is not None:
        for key in (
            "accessible_name", "label", "placeholder", "test_id",
            "element_id", "name", "href", "dialog_identity", "tag", "role", "visible_text",
        ):
            value = getattr(original, key)
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()

    # Support older or hand-authored plans that already retain locator hints.
    aliases = {
        "accessible_name": ("accessible_name",),
        "label": ("label",),
        "placeholder": ("placeholder",),
        "test_id": ("test_id", "data-testid"),
        "element_id": ("id",),
        "name": ("name",),
        "href": ("href",),
        "dialog_identity": ("dialog_identity",),
        "tag": ("tag",),
        "role": ("role",),
        "visible_text": ("text", "expected_text"),
    }
    for target, keys in aliases.items():
        if target == "visible_text" and action != "click":
            continue
        for key in keys:
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                result.setdefault(target, value.strip())
                break
    return result


def _candidate_values(element: InteractiveElement) -> dict[str, str]:
    return {
        "accessible_name": element.accessible_name,
        "label": element.label,
        "placeholder": element.placeholder,
        "visible_text": element.text,
        "test_id": element.test_id,
        "element_id": element.id,
        "name": element.name,
        "href": element.href,
        "dialog_identity": element.dialog_identity,
        "tag": element.tag,
        "role": element.role,
    }


def _semantic_conflict(expected: dict[str, str], element: InteractiveElement, action: str) -> bool:
    actual = _candidate_values(element)
    semantic_keys = ["accessible_name", "label", "placeholder"]
    if action == "click":
        semantic_keys.extend(("visible_text", "href"))
    for key in semantic_keys:
        old = expected.get(key)
        new = actual.get(key)
        if old and new and _normalize(old) != _normalize(new):
            return True
    # A plan may have only one label form while discovery's accessible name is
    # populated from that label. Compare those surfaces when the paired field
    # itself is unavailable.
    if expected.get("accessible_name") and not actual.get("accessible_name") and actual.get("label"):
        if _normalize(expected["accessible_name"]) != _normalize(actual["label"]):
            return True
    if expected.get("label") and not actual.get("label") and actual.get("accessible_name"):
        if _normalize(expected["label"]) != _normalize(actual["accessible_name"]):
            return True
    return False


def _stable_identity_match(expected: dict[str, str], element: InteractiveElement) -> tuple[bool, tuple[str, ...]]:
    matches: list[str] = []
    if expected.get("test_id") and element.test_id and expected["test_id"] == element.test_id:
        matches.append("exact_test_id")
    if expected.get("element_id") and element.id and expected["element_id"] == element.id:
        matches.append("exact_id")
    return bool(matches), tuple(matches)


def _structure_matches(
    expected: dict[str, str], element: InteractiveElement
) -> tuple[bool, tuple[str, ...]]:
    signals: list[str] = []
    role = expected.get("role")
    tag = expected.get("tag")
    stable = (
        bool(expected.get("test_id") and element.test_id == expected.get("test_id"))
        or bool(expected.get("element_id") and element.id == expected.get("element_id"))
    )
    if role and element.role and role.casefold() != element.role.casefold() and not stable:
        return False, ()
    if tag and element.tag and tag.casefold() != element.tag.casefold() and not stable:
        return False, ()
    if role and element.role and role.casefold() == element.role.casefold():
        signals.append("exact_role")
    if tag and element.tag and tag.casefold() == element.tag.casefold():
        signals.append("exact_tag")
    return True, tuple(signals)


def _semantic_matches(expected: dict[str, str], element: InteractiveElement, action: str) -> tuple[str, ...]:
    actual = _candidate_values(element)
    signals: list[str] = []
    if expected.get("accessible_name") and actual["accessible_name"] and _same(expected["accessible_name"], actual["accessible_name"]):
        signals.append("exact_accessible_name")
    if expected.get("label") and actual["label"] and _same(expected["label"], actual["label"]):
        signals.append("exact_label")
    if action == "click" and expected.get("visible_text") and actual["visible_text"] and _same(expected["visible_text"], actual["visible_text"]):
        signals.append("exact_text")
    if action == "click" and expected.get("href") and actual["href"] and expected["href"] == actual["href"]:
        signals.append("exact_href")
    if expected.get("name") and actual["name"] and expected["name"] == actual["name"]:
        signals.append("exact_name")
    if expected.get("test_id") and actual["test_id"] and expected["test_id"] == actual["test_id"]:
        signals.append("exact_test_id")
    if expected.get("element_id") and actual["element_id"] and expected["element_id"] == actual["element_id"]:
        signals.append("exact_id")
    return tuple(signals)


def _placeholder_match(expected: dict[str, str], element: InteractiveElement) -> bool:
    return bool(
        expected.get("placeholder") and element.placeholder
        and _same(expected["placeholder"], element.placeholder)
    )


def _has_semantic_expectation(expected: dict[str, str]) -> bool:
    return any(expected.get(key) for key in (
        "accessible_name", "label", "placeholder", "visible_text", "test_id",
        "element_id", "name", "href",
    ))


def _compatible(action: str, element: InteractiveElement) -> bool:
    kind = (element.kind or "").casefold()
    tag = (element.tag or "").casefold()
    role = (element.role or "").casefold()
    if not element.visible or not element.enabled:
        return False
    if action == "fill":
        return tag in {"input", "textarea", "select"} or kind in {
            "input", "textarea", "select", "textbox", "searchbox", "combobox"
        } or role in {"textbox", "searchbox", "combobox"}
    if action in {"check", "uncheck"}:
        return kind == "checkbox" or role == "checkbox"
    return tag in {"a", "button", "input"} or kind in {
        "a", "link", "button", "submit", "checkbox", "radio", "menuitem", "tab"
    } or role in {"link", "button", "menuitem", "tab", "checkbox", "radio"}


def _dialog_scope_prefix(selector: str) -> str | None:
    """Return a dialog ancestor token that a recovered selector must retain."""
    if not isinstance(selector, str):
        return None
    match = re.search(
        r'''(?:\[role\s*=\s*["']?dialog["']?\]|\[aria-modal\s*=\s*["']?true["']?\]|#[\w-]*(?:dialog|modal)[\w-]*|\.[\w-]*(?:dialog|modal)[\w-]*|(?:^|\s)dialog(?=\s|[>+~]))''',
        selector,
        re.IGNORECASE,
    )
    if match is None:
        return None
    tail = selector[match.end():]
    if not re.match(r"(?:\s+|\s*[>+~]\s*)\S", tail):
        return None
    token = match.group(0).strip()
    return token or None


def _clean(value: str | None) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _same(left: str, right: str) -> bool:
    return _normalize(left) == _normalize(right)


def _ambiguous(selector: str) -> LocatorRecoveryResult:
    return LocatorRecoveryResult(
        RecoveryStatus.AMBIGUOUS, selector,
        reason="Multiple compatible elements match the saved identity; no control was selected.",
    )
