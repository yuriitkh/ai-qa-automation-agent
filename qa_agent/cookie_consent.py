"""Deterministic, opt-out cookie-consent handling for browser runs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import Enum
import re
import unicodedata
from typing import Iterator

from pydantic import BaseModel, ConfigDict


class CookieConsentPolicy(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    LEAVE_UNCHANGED = "LEAVE_UNCHANGED"


class CookieConsentStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    NO_BANNER = "NO_BANNER"
    HANDLED = "HANDLED"
    LEFT_UNCHANGED = "LEFT_UNCHANGED"
    REQUIRES_ATTENTION = "REQUIRES_ATTENTION"


class CookieConsentReason(str, Enum):
    MULTIPLE_DIALOGS = "MULTIPLE_DIALOGS"
    MULTIPLE_CONTAINERS = "MULTIPLE_CONTAINERS"
    NO_SAFE_ACCEPT_ACTION = "NO_SAFE_ACCEPT_ACTION"
    MULTIPLE_ACCEPT_ACTIONS = "MULTIPLE_ACCEPT_ACTIONS"
    ACTION_FAILED = "ACTION_FAILED"
    DETECTION_FAILED = "DETECTION_FAILED"


class CookieConsentRecord(BaseModel):
    """Safe run-level metadata. Dialog contents and storage values are omitted."""

    model_config = ConfigDict(frozen=True)

    policy: CookieConsentPolicy
    status: CookieConsentStatus
    reason: CookieConsentReason | None = None


DEFAULT_COOKIE_CONSENT_POLICY = CookieConsentPolicy.AUTO_HANDLE

CONSENT_CONTAINER_SELECTOR = (
    'dialog, [role="dialog"], [aria-modal="true"], [role="banner"], '
    '[aria-label*="cookie" i], [aria-label*="consent" i], '
    '[aria-label*="privacy" i], [id*="cookie" i], [id*="consent" i], '
    '[id*="privacy" i], [class*="cookie" i], [class*="consent" i], '
    '[class*="privacy" i]'
)
CONSENT_CONTROL_SELECTOR = 'button, a, [role="button"], [role="link"]'

_DETECTION_SCRIPT = r"""selector => {
  const visible = (element) => {
    for (let current = element; current && current !== document.documentElement;
      current = current.parentElement) {
      const style = window.getComputedStyle(current);
      if (style.display === 'none' || style.visibility === 'hidden' ||
          style.visibility === 'collapse' || Number(style.opacity || 1) === 0) return false;
    }
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const consentText = /cookies?|consent|informa[cç][oõ]es?\s+de\s+cookies|informasjonskapsler|samtykke|personvern|privatliv|datenschutz|einwilligung|consentement|consentimiento|privacidad|aceptaci[oó]n\s+de\s+cookies|samtycke|integritet|cookie-instellingen/i;
  const preferenceText = /privacy\s+(settings|preferences|choices)|privacyinstellingen|datenschutzeinstellungen|param[eè]tres\s+de\s+confidentialit[eé]|ajustes\s+de\s+privacidad/i;
  const semanticIdentity = /cookie|consent|privacy/i;
  const all = Array.from(document.querySelectorAll(selector));
  const matched = all.map((element, index) => {
    if (!visible(element) || /^(BUTTON|A|INPUT|SELECT|TEXTAREA)$/.test(element.tagName)) return null;
    const attributes = [
      element.getAttribute('id') || '', element.getAttribute('class') || '',
      element.getAttribute('aria-label') || '', element.getAttribute('role') || ''
    ].join(' ');
    const text = element.innerText || element.textContent || '';
    const identified = semanticIdentity.test(attributes) ||
      consentText.test(text) || preferenceText.test(text);
    return identified ? {element, index} : null;
  }).filter(Boolean);
  const roots = matched.filter(({element}) =>
    !matched.some(other => other.element !== element && other.element.contains(element))
  );
  const visibleDialogs = Array.from(document.querySelectorAll(
    'dialog, [role="dialog"], [aria-modal="true"]'
  )).filter(visible).length;
  return {
    visible_dialog_count: visibleDialogs,
    containers: roots.map(({index}) => ({index}))
  };
}"""

_CONTROLS_SCRIPT = r"""elements => {
  const visible = (element) => {
    for (let current = element; current && current !== document.documentElement;
      current = current.parentElement) {
      const style = window.getComputedStyle(current);
      if (style.display === 'none' || style.visibility === 'hidden' ||
          style.visibility === 'collapse' || Number(style.opacity || 1) === 0) return false;
    }
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const accessibleName = (element) => {
    const direct = element.getAttribute('aria-label');
    if (direct) return direct;
    const labelledBy = element.getAttribute('aria-labelledby');
    if (labelledBy) {
      const labels = labelledBy.split(/\s+/).map(id => document.getElementById(id))
        .filter(Boolean).map(item => item.innerText || item.textContent || '');
      if (labels.length) return labels.join(' ');
    }
    return element.innerText || element.getAttribute('title') || element.textContent || '';
  };
  return elements.map((element, index) => ({
    index,
    name: accessibleName(element).trim(),
    visible: visible(element),
    enabled: !element.disabled && element.getAttribute('aria-disabled') !== 'true'
  }));
}"""

def _normalize_label(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(character for character in ascii_value if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value))


_ACCEPT_LABELS = frozenset(
    _normalize_label(label)
    for label in (
        "accept", "accept all", "accept cookies", "accept all cookies",
        "allow", "allow all", "allow all cookies", "agree", "agree all",
        "i agree", "aceptar", "aceptar todo", "aceptar todas las cookies",
        "accepter", "tout accepter", "accepter les cookies",
        "akzeptieren", "alle akzeptieren", "cookies akzeptieren",
        "godta", "godta alle", "godta informasjonskapsler",
        "acceptera", "acceptera alla", "godkänn", "accepter alle",
        "accepter alle cookies", "permitir todas", "aceitar todos",
    )
)


class _CookieConsentTracker:
    def __init__(self, policy: CookieConsentPolicy) -> None:
        self.policy = policy
        self.status = (
            CookieConsentStatus.LEFT_UNCHANGED
            if policy == CookieConsentPolicy.LEAVE_UNCHANGED
            else CookieConsentStatus.NOT_EVALUATED
        )
        self.reason: CookieConsentReason | None = None

    def observe(self, result: CookieConsentRecord) -> None:
        if self.policy == CookieConsentPolicy.LEAVE_UNCHANGED:
            return
        if self.status == CookieConsentStatus.REQUIRES_ATTENTION:
            return
        if result.status == CookieConsentStatus.REQUIRES_ATTENTION:
            self.status = result.status
            self.reason = result.reason
        elif result.status == CookieConsentStatus.HANDLED:
            self.status = result.status
            self.reason = None
        elif self.status != CookieConsentStatus.HANDLED:
            self.status = result.status
            self.reason = result.reason

    def snapshot(self) -> CookieConsentRecord:
        return CookieConsentRecord(
            policy=self.policy,
            status=self.status,
            reason=self.reason,
        )


_active_policy: ContextVar[CookieConsentPolicy] = ContextVar(
    "qa_agent_cookie_consent_policy", default=DEFAULT_COOKIE_CONSENT_POLICY
)
_active_tracker: ContextVar[_CookieConsentTracker | None] = ContextVar(
    "qa_agent_cookie_consent_tracker", default=None
)


def current_cookie_consent_policy() -> CookieConsentPolicy:
    return _active_policy.get()


def current_cookie_consent_record() -> CookieConsentRecord | None:
    tracker = _active_tracker.get()
    return tracker.snapshot() if tracker is not None else None


def observe_cookie_consent(result: CookieConsentRecord) -> None:
    tracker = _active_tracker.get()
    if tracker is not None:
        tracker.observe(result)


@contextmanager
def cookie_consent_scope(policy: CookieConsentPolicy | None) -> Iterator[None]:
    selected = CookieConsentPolicy(policy or DEFAULT_COOKIE_CONSENT_POLICY)
    policy_token: Token[CookieConsentPolicy] = _active_policy.set(selected)
    tracker_token: Token[_CookieConsentTracker | None] = _active_tracker.set(
        _CookieConsentTracker(selected)
    )
    try:
        yield
    finally:
        _active_tracker.reset(tracker_token)
        _active_policy.reset(policy_token)


def cookie_consent_label(record: CookieConsentRecord) -> str:
    return {
        CookieConsentStatus.NOT_EVALUATED: "Not evaluated",
        CookieConsentStatus.NO_BANNER: "No banner detected",
        CookieConsentStatus.HANDLED: "Handled automatically",
        CookieConsentStatus.LEFT_UNCHANGED: "Left unchanged by run policy",
        CookieConsentStatus.REQUIRES_ATTENTION: "Requires attention",
    }[record.status]


def cookie_consent_policy_label(policy: CookieConsentPolicy) -> str:
    return {
        CookieConsentPolicy.AUTO_HANDLE: "Auto handle cookie consent",
        CookieConsentPolicy.LEAVE_UNCHANGED: "Leave cookie consent unchanged",
    }[policy]


def cookie_consent_reason_label(reason: CookieConsentReason | None) -> str | None:
    if reason is None:
        return None
    return {
        CookieConsentReason.MULTIPLE_DIALOGS: "Multiple visible dialogs",
        CookieConsentReason.MULTIPLE_CONTAINERS: "Multiple consent containers",
        CookieConsentReason.NO_SAFE_ACCEPT_ACTION: "No unique safe accept action",
        CookieConsentReason.MULTIPLE_ACCEPT_ACTIONS: "Multiple safe accept actions",
        CookieConsentReason.ACTION_FAILED: "The consent action did not complete",
        CookieConsentReason.DETECTION_FAILED: "Consent detection could not complete",
    }[reason]


def handle_cookie_consent(page) -> CookieConsentRecord:
    """Handle one uniquely identified consent dialog, or preserve page state."""
    policy = current_cookie_consent_policy()
    if policy == CookieConsentPolicy.LEAVE_UNCHANGED:
        result = CookieConsentRecord(
            policy=policy,
            status=CookieConsentStatus.LEFT_UNCHANGED,
        )
        observe_cookie_consent(result)
        return result

    try:
        detection = page.evaluate(_DETECTION_SCRIPT, CONSENT_CONTAINER_SELECTOR)
    except Exception:
        return _attention(policy, CookieConsentReason.DETECTION_FAILED)

    # Non-dict values are tolerated for simple browser fakes; real Playwright
    # returns the explicit object from _DETECTION_SCRIPT.
    if not isinstance(detection, dict):
        result = CookieConsentRecord(policy=policy, status=CookieConsentStatus.NO_BANNER)
        observe_cookie_consent(result)
        return result

    containers = detection.get("containers")
    if not isinstance(containers, list):
        return _attention(policy, CookieConsentReason.DETECTION_FAILED)
    if not containers:
        result = CookieConsentRecord(policy=policy, status=CookieConsentStatus.NO_BANNER)
        observe_cookie_consent(result)
        return result

    dialog_count = detection.get("visible_dialog_count", 0)
    if not isinstance(dialog_count, int) or dialog_count > 1:
        return _attention(policy, CookieConsentReason.MULTIPLE_DIALOGS)
    if len(containers) > 1:
        return _attention(policy, CookieConsentReason.MULTIPLE_CONTAINERS)

    candidate_index = containers[0].get("index") if isinstance(containers[0], dict) else None
    if not isinstance(candidate_index, int) or candidate_index < 0:
        return _attention(policy, CookieConsentReason.DETECTION_FAILED)

    try:
        container = page.locator(CONSENT_CONTAINER_SELECTOR).nth(candidate_index)
        controls_locator = container.locator(CONSENT_CONTROL_SELECTOR)
        controls = controls_locator.evaluate_all(_CONTROLS_SCRIPT, CONSENT_CONTROL_SELECTOR)
    except Exception:
        return _attention(policy, CookieConsentReason.DETECTION_FAILED)
    if not isinstance(controls, list):
        return _attention(policy, CookieConsentReason.DETECTION_FAILED)

    safe_actions = [
        control
        for control in controls
        if isinstance(control, dict)
        and isinstance(control.get("index"), int)
        and control.get("visible") is True
        and control.get("enabled") is True
        and isinstance(control.get("name"), str)
        and _normalize_label(control["name"]) in _ACCEPT_LABELS
    ]
    if not safe_actions:
        return _attention(policy, CookieConsentReason.NO_SAFE_ACCEPT_ACTION)
    if len(safe_actions) != 1:
        return _attention(policy, CookieConsentReason.MULTIPLE_ACCEPT_ACTIONS)

    try:
        controls_locator.nth(safe_actions[0]["index"]).click(timeout=5_000)
    except Exception:
        return _attention(policy, CookieConsentReason.ACTION_FAILED)

    result = CookieConsentRecord(policy=policy, status=CookieConsentStatus.HANDLED)
    observe_cookie_consent(result)
    return result


def _attention(
    policy: CookieConsentPolicy,
    reason: CookieConsentReason,
) -> CookieConsentRecord:
    result = CookieConsentRecord(
        policy=policy,
        status=CookieConsentStatus.REQUIRES_ATTENTION,
        reason=reason,
    )
    observe_cookie_consent(result)
    return result
