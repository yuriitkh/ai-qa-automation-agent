"""Conservative, browser-independent matching for stale planned locators."""

from dataclasses import dataclass
from enum import Enum

from qa_agent.models import DiscoveryResult, InteractiveElement, QATestStep


class RecoveryStatus(str, Enum):
    MATCHED = "MATCHED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"


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
) -> LocatorRecoveryResult:
    """Return a unique, strongly supported equivalent discovered element.

    The matcher does not inspect or control a browser. An exact selector match is
    retained; otherwise a unique compatible candidate needs an exact semantic
    identity (text/name, or link href) plus corroborating type/role evidence.
    """
    if planned_interaction.action not in {"click", "fill"}:
        raise ValueError("Locator recovery supports click and fill interactions only.")
    params = planned_interaction.parameters
    selector = params.get("selector", "")
    if not isinstance(selector, str):
        selector = ""
    candidates = [
        element for element in discovery_result.interactive_elements
        if _compatible(planned_interaction.action, element)
    ]

    exact_selector = [element for element in candidates if selector and element.selector == selector]
    if len(exact_selector) == 1:
        return LocatorRecoveryResult(
            RecoveryStatus.MATCHED, selector, exact_selector[0], ("exact_selector",),
            "The original selector still identifies one compatible discovered element.",
        )
    if len(exact_selector) > 1:
        return _ambiguous(selector, "Multiple compatible elements have the original selector.")

    expected_text = _exact_expected_text(params)
    expected_href = params.get("href")
    expected_role = params.get("role")
    expected_tag = params.get("tag")
    expected_name = params.get("name")

    scored: list[tuple[InteractiveElement, tuple[str, ...]]] = []
    for element in candidates:
        signals: list[str] = []
        text_match = bool(expected_text) and expected_text in {
            element.text, element.accessible_name
        }
        href_match = (
            planned_interaction.action == "click"
            and isinstance(expected_href, str) and bool(expected_href)
            and element.href == expected_href
        )
        attribute_match = (
            isinstance(expected_name, str) and bool(expected_name) and element.name == expected_name
        ) or (
            isinstance(expected_tag, str) and bool(expected_tag) and element.tag == expected_tag
        )
        if text_match:
            signals.append("exact_text")
        if href_match:
            signals.append("exact_href")
        if isinstance(expected_role, str) and expected_role and element.role == expected_role:
            signals.append("exact_role")
        if isinstance(expected_tag, str) and expected_tag and element.tag == expected_tag:
            signals.append("exact_tag")
        if isinstance(expected_name, str) and expected_name and element.name == expected_name:
            signals.append("exact_name")

        # Semantic identity is mandatory; type alone can never redirect an action.
        if (text_match or href_match or (attribute_match and expected_role == element.role)):
            scored.append((element, tuple(signals)))

    if len(scored) == 1:
        element, signals = scored[0]
        return LocatorRecoveryResult(
            RecoveryStatus.MATCHED, selector, element, signals,
            "One compatible element has an exact identity match.",
        )
    if len(scored) > 1:
        return _ambiguous(selector, "Multiple compatible elements have exact identity evidence.")
    return LocatorRecoveryResult(
        RecoveryStatus.NOT_FOUND, selector,
        reason="No compatible element has sufficiently strong exact identity evidence.",
    )


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
    return tag in {"a", "button", "input"} or kind in {
        "a", "link", "button", "submit", "checkbox", "radio", "menuitem", "tab"
    } or role in {"link", "button", "menuitem", "tab", "checkbox", "radio"}


def _exact_expected_text(parameters: dict) -> str:
    for key in ("expected_text", "accessible_name", "text"):
        value = parameters.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _ambiguous(selector: str, reason: str) -> LocatorRecoveryResult:
    return LocatorRecoveryResult(RecoveryStatus.AMBIGUOUS, selector, reason=reason)
