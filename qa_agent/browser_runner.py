from qa_agent.execution_control import current_cancellation, check_cancelled
from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any, Iterator
from uuid import uuid4

from playwright.sync_api import expect, sync_playwright

from .execution_semantics import (
    ACTION_TIMEOUT_MS,
    ASSERTION_TIMEOUT_MS,
    NAVIGATION_LOAD_STATE,
    NAVIGATION_TIMEOUT_MS,
)
from .cookie_consent import (
    CookieConsentRecord,
    CookieConsentPolicy,
    CookieConsentReason,
    CookieConsentStatus,
    current_cookie_consent_policy,
    handle_cookie_consent,
)
from .evidence_policy import (
    EvidenceMode,
    EvidenceScope,
    ScreenshotMode,
    current_evidence_execution,
    current_evidence_policy,
)
from .models import QATestPlan, QATestStep, TestCase
from .execution_diagnostics import capture_action_failure
from .redaction import redact_secrets

_PRIVATE_INPUTS = 'input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]):not([type=reset]):not([type=hidden]), textarea'


class BrowserSession:
    """Own one Playwright runtime, browser, context, and its pages."""

    def __init__(
        self,
        evidence_directory: str | Path | None = None,
        *,
        headless: bool = False,
    ) -> None:
        self.evidence_directory = Path(evidence_directory) if evidence_directory is not None else None
        self.headless = headless
        self._playwright_manager: Any | None = None
        self._playwright: Any | None = None
        self._playwright_entered = False
        self._browser: Any | None = None
        self._context: Any | None = None
        self._pages: list[Any] = []
        self._closed = False

    def start(self) -> None:
        check_cancelled()
        if self._closed or self._playwright_manager is not None:
            raise RuntimeError("BrowserSession has already been started or closed.")
        self._playwright_manager = sync_playwright()
        self._playwright = self._playwright_manager.__enter__()
        self._playwright_entered = True
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context()

    def new_page(self) -> Any:
        check_cancelled()
        if self._closed or self._context is None:
            raise RuntimeError("BrowserSession is not open.")
        page = self._context.new_page()
        self._pages.append(page)
        return page

    def run_plan(self, page: Any, plan: QATestPlan) -> dict[str, Any]:
        if self._closed or not any(page is owned for owned in self._pages):
            raise ValueError("Page must be open and owned by this BrowserSession.")
        return _run_plan_on_page(plan, page, self.evidence_directory)

    def close(self, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_errors: list[BaseException] = []

        resources = [*reversed(self._pages), self._context, self._browser]
        for resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                cleanup_errors.append(error)
        self._pages.clear()

        if self._playwright_entered and self._playwright_manager is not None:
            try:
                if primary_error is None:
                    self._playwright_manager.__exit__(None, None, None)
                else:
                    self._playwright_manager.__exit__(
                        type(primary_error),
                        primary_error,
                        primary_error.__traceback__,
                    )
            except BaseException as error:
                cleanup_errors.append(error)

        if cleanup_errors and primary_error is None:
            raise cleanup_errors[0]


class BrowserRunner:
    """Execute plans in Playwright, optionally saving screenshots on failure."""

    def __init__(
        self,
        evidence_directory: str | Path | None = None,
        *,
        headless: bool = False,
    ) -> None:
        self.evidence_directory = Path(evidence_directory) if evidence_directory is not None else None
        self.headless = headless

    @contextmanager
    def open_session(self) -> Iterator[BrowserSession]:
        session = BrowserSession(self.evidence_directory, headless=self.headless)
        try:
            session.start()
            yield session
        except BaseException as error:
            session.close(primary_error=error)
            raise
        else:
            session.close()

    @contextmanager
    def open_test_case_session(self, test_case: TestCase) -> Iterator["TestCaseBrowserSession"]:
        """Own one browser context across the TestCase's ordered segments.

        Browser startup is lazy so a TestCase that fails while preparing
        automation does not launch a browser it never uses.
        """
        session = TestCaseBrowserSession(self, test_case)
        try:
            yield session
        except BaseException as error:
            session.close(primary_error=error)
            raise
        else:
            session.close()

    def __call__(self, plan: QATestPlan) -> dict[str, Any]:
        with self.open_session() as session:
            page = session.new_page()
            return session.run_plan(page, plan)


class TestCaseBrowserSession:
    """Map each ordered segment to one page inside a shared browser context."""

    def __init__(self, runner: BrowserRunner, test_case: TestCase) -> None:
        self._runner = runner
        self._segment_orders = {segment.order for segment in test_case.segments}
        self._segment_urls = {segment.order: segment.base_url or test_case.base_url or "about:blank" for segment in test_case.segments}
        self._session: BrowserSession | None = None
        self._pages: dict[int, Any] = {}
        self._closed = False

    @property
    def is_started(self) -> bool:
        return self._session is not None

    def run_plan(self, segment_order: int, plan: QATestPlan) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("TestCase browser session is closed.")
        if segment_order not in self._segment_orders:
            raise ValueError("ExecutionSegment does not belong to this TestCase.")
        check_cancelled()
        if self._session is None:
            self._session = BrowserSession(
                self._runner.evidence_directory,
                headless=self._runner.headless,
            )
            self._session.start()
        page = self._pages.get(segment_order)
        if page is None:
            page = self._session.new_page()
            self._pages[segment_order] = page
        return self._session.run_plan(page, plan)

    def capture_discovery(self, segment_order: int):
        from qa_agent.browser_discovery import capture_current_page_discovery
        page = self._pages.get(segment_order)
        if page is None:
            from qa_agent.models import DiscoveryResult, DiscoveryStatus
            return DiscoveryResult(status=DiscoveryStatus.PARTIAL, url=self._segment_urls[segment_order], warnings=["This segment has no observed page. Its configured URL is a navigation target only; an explicit navigation step is required before state-dependent assertions."], strategies_used=["current_page"])
        return capture_current_page_discovery(page)

    def close(self, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            self._session.close(primary_error=primary_error)


def run_test_plan(
    plan: QATestPlan,
    evidence_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper: execute one plan in an isolated session."""
    return BrowserRunner(evidence_directory)(plan)


def _run_plan_on_page(
    plan: QATestPlan,
    page: Any,
    evidence_directory: str | Path | None = None,
) -> dict[str, Any]:
    check_cancelled()
    result: dict[str, Any] = {
        "status": "passed",
        "url": plan.url,
        "steps": [],
        "evidence": [],
        "cookie_consent": {
            "policy": current_cookie_consent_policy().value,
            "status": (
                CookieConsentStatus.LEFT_UNCHANGED.value
                if current_cookie_consent_policy() == CookieConsentPolicy.LEAVE_UNCHANGED
                else CookieConsentStatus.NOT_EVALUATED.value
            ),
            "reason": None,
        },
    }

    verification_actions = frozenset(
        action
        for action in QATestStep.ACTION_PARAMETER_FIELDS
        if action.startswith("assert_")
    )

    if getattr(page, "url", "") and getattr(page, "url", "") != "about:blank":
        consent = handle_cookie_consent(page)
        result["cookie_consent"] = _cookie_consent_public_dict(consent)
        if consent.status == CookieConsentStatus.REQUIRES_ATTENTION:
            _record_cookie_consent_attention(
                result, page, evidence_directory, consent.reason
            )
            return result

    for step_index, step in enumerate(plan.steps):
        if current_cancellation() is not None and current_cancellation().requested:
            result["status"] = "cancelled"
            break
        step_result = {
            "action": step.action,
            "status": "passed",
            "error": "",
        }

        try:
            if step.action == "navigate":
                page.goto(
                    step.parameters["url"],
                    wait_until=NAVIGATION_LOAD_STATE,
                    timeout=NAVIGATION_TIMEOUT_MS,
                )
                consent = handle_cookie_consent(page)
                result["cookie_consent"] = _cookie_consent_public_dict(consent)
                if consent.status == CookieConsentStatus.REQUIRES_ATTENTION:
                    result["steps"].append(step_result)
                    _record_cookie_consent_attention(
                        result, page, evidence_directory, consent.reason
                    )
                    return result
            elif step.action == "click":
                selector = step.parameters["selector"]
                try:
                    element = page.locator(selector)
                    element.click(timeout=ACTION_TIMEOUT_MS)
                except AssertionError:
                    raise
                except Exception as error:
                    raise AssertionError(
                        f"Could not click element matching selector "
                        f"{selector!r}: {error}"
                    ) from error
            elif step.action in {"check", "uncheck"}:
                selector = step.parameters["selector"]
                operation = step.action
                try:
                    element = page.locator(selector)
                    getattr(element, operation)(timeout=ACTION_TIMEOUT_MS)
                except AssertionError:
                    raise
                except Exception as error:
                    raise AssertionError(
                        f"Could not {operation} checkbox matching selector "
                        f"{selector!r}: {error}"
                    ) from error
            elif step.action == "fill":
                selector = step.parameters["selector"]
                value = step.parameters["value"]
                element = page.locator(selector)
                try:
                    element.fill(value, timeout=ACTION_TIMEOUT_MS)
                except AssertionError as error:
                    safe_error = str(error)
                    if value:
                        safe_error = safe_error.replace(value, "[REDACTED]")
                    if safe_error != str(error):
                        raise AssertionError(safe_error) from error
                    raise
                except Exception as error:
                    safe_error = str(error)
                    if value:
                        safe_error = safe_error.replace(value, "[REDACTED]")
                    raise AssertionError(
                        f"Could not fill element matching selector "
                        f"{selector!r}: {safe_error}"
                    ) from error
            elif step.action == "select_option":
                selector = step.parameters["selector"]
                label = step.parameters["option_label"]
                try:
                    element = page.locator(selector)
                    element.select_option(label=label, timeout=ACTION_TIMEOUT_MS)
                except Exception as error:
                    raise AssertionError(
                        f"Could not select option label {label!r} in {selector!r}: {error}"
                    ) from error
            elif step.action == "assert_page_loaded":
                page.wait_for_load_state(
                    NAVIGATION_LOAD_STATE,
                    timeout=NAVIGATION_TIMEOUT_MS,
                )
            elif step.action == "assert_title":
                expected_title = step.parameters["expected"]
                expect(page).to_have_title(
                    expected_title,
                    timeout=ASSERTION_TIMEOUT_MS,
                )
            elif step.action == "assert_url":
                expected_url = step.parameters["expected"]
                expect(page).to_have_url(
                    expected_url,
                    timeout=ASSERTION_TIMEOUT_MS,
                )
            elif step.action == "assert_visible":
                selector = step.parameters["selector"]
                expected_text = step.parameters.get("expected_text")
                element = page.locator(selector)
                expect(element).to_be_visible(timeout=ASSERTION_TIMEOUT_MS)
                if expected_text is not None:
                    expect(element).to_have_js_property(
                        "innerText",
                        expected_text.replace("\\n", "\n"),
                        timeout=ASSERTION_TIMEOUT_MS,
                    )
            elif step.action == "assert_text_contains":
                expected_text = step.parameters["expected_text"].replace("\\n", "\n")
                selector = step.parameters.get("selector")
                target = page.locator(selector) if selector else page.locator("body")
                expect(target).to_be_visible(
                    timeout=ASSERTION_TIMEOUT_MS,
                )
                expected_pattern = re.compile(
                    ".*" + re.escape(expected_text) + ".*",
                    re.DOTALL,
                )
                expect(target).to_have_text(
                    expected_pattern,
                    use_inner_text=True,
                    timeout=ASSERTION_TIMEOUT_MS,
                )
            elif step.action == "assert_checked":
                selector = step.parameters["selector"]
                element = page.locator(selector)
                expect(element).to_be_checked(timeout=ASSERTION_TIMEOUT_MS)
            elif step.action == "assert_unchecked":
                selector = step.parameters["selector"]
                element = page.locator(selector)
                expect(element).not_to_be_checked(timeout=ASSERTION_TIMEOUT_MS)
            elif step.action == "assert_selected":
                selector = step.parameters["selector"]
                element = page.locator(selector)
                expect(element).to_be_visible(timeout=ASSERTION_TIMEOUT_MS)
                element_info = element.evaluate(
                    "element => ({tag: element.tagName.toLowerCase(), type: element.type})"
                )
                expected = step.parameters.get("expected")
                if element_info["tag"] == "input" and element_info["type"] == "radio":
                    if expected is not None:
                        raise AssertionError(
                            f"assert_selected for radio {selector!r} does not take expected."
                        )
                    expect(element).to_be_checked(timeout=ASSERTION_TIMEOUT_MS)
                elif element_info["tag"] == "select":
                    if not isinstance(expected, str):
                        raise AssertionError(
                            f"assert_selected for select {selector!r} requires expected option label or value."
                        )
                    selected = element.locator("option:checked").first
                    options = element.locator("option")
                    option_labels = options.all_inner_texts()
                    option_values = [option.get_attribute("value") for option in options.all()]
                    if expected in option_labels:
                        expect(selected).to_have_js_property(
                            "innerText",
                            expected,
                            timeout=ASSERTION_TIMEOUT_MS,
                        )
                    elif expected in option_values:
                        expect(selected).to_have_attribute(
                            "value",
                            expected,
                            timeout=ASSERTION_TIMEOUT_MS,
                        )
                    else:
                        raise AssertionError(
                            f"Expected option label or value {expected!r} is not available for {selector!r}."
                        )
                else:
                    raise AssertionError(
                        f"assert_selected selector {selector!r} must target a radio or select."
                    )
            elif step.action in {"assert_enabled", "assert_disabled"}:
                selector = step.parameters["selector"]
                element = page.locator(selector)
                if step.action == "assert_enabled":
                    expect(element).to_be_enabled(timeout=ASSERTION_TIMEOUT_MS)
                else:
                    expect(element).to_be_disabled(timeout=ASSERTION_TIMEOUT_MS)

            elif step.action == "assert_hidden":
                selector = step.parameters["selector"]
                try:
                    element = page.locator(selector)
                    expect(element).to_be_hidden(timeout=ASSERTION_TIMEOUT_MS)
                except AssertionError:
                    raise
                except Exception as error:
                    raise AssertionError(
                        f"Could not verify hidden state for selector "
                        f"{selector!r}: {error}"
                    ) from error
            else:
                raise ValueError(f"Unsupported test action: {step.action!r}")
        except Exception as error:
            if current_cancellation() is not None and current_cancellation().requested:
                step_result["status"] = "cancelled"
                result["status"] = "cancelled"
                result["steps"].append(step_result)
                break
            step_result["status"] = "failed"
            safe_error = redact_secrets(str(error))
            for item in plan.steps:
                value = item.parameters.get("value") if item.action == "fill" else None
                if isinstance(value, str) and value:
                    safe_error = safe_error.replace(value, "[REDACTED]")
            step_result["error"] = safe_error
            diagnostic = capture_action_failure(page, step, step_index, plan, error)
            result["status"] = "failed"
            first_evidence = len(result["evidence"])
            _capture_screenshots(
                page,
                evidence_directory,
                result,
                action=step,
                action_index=step_index,
                event_kind="FAILURE",
            )
            step_result["diagnostic"] = diagnostic.model_copy(update={
                "evidence_indexes": tuple(range(first_evidence, len(result["evidence"])))
            }).model_dump(mode="json")
            result["steps"].append(step_result)
            break

        if (
            not (current_cancellation() is not None and current_cancellation().requested)
            and current_evidence_policy().mode == EvidenceMode.EVERY_VERIFICATION
            and step.action in verification_actions
        ):
            _capture_screenshots(
                page,
                evidence_directory,
                result,
                action=step,
                action_index=step_index,
                event_kind="VERIFICATION",
            )
        result["steps"].append(step_result)
        if current_cancellation() is not None and current_cancellation().requested:
            result["status"] = "cancelled"
            break

    if (
        result["status"] == "passed"
        and current_evidence_policy().mode == EvidenceMode.EVERY_STEP
        and plan.steps
    ):
        _capture_screenshots(
            page,
            evidence_directory,
            result,
            action=plan.steps[-1],
            action_index=len(plan.steps),
            event_kind="TEST_STEP",
        )
    return result


def _record_cookie_consent_attention(
    result: dict[str, Any],
    page: Any,
    evidence_directory: str | Path | None,
    reason: CookieConsentReason | None,
) -> None:
    """Stop safely when consent is ambiguous, without treating it as a product assertion."""
    result["status"] = "failed"
    result["cookie_consent_requires_attention"] = True
    result["cookie_consent"] = _cookie_consent_public_dict(CookieConsentRecord(
        policy=current_cookie_consent_policy(),
        status=CookieConsentStatus.REQUIRES_ATTENTION,
        reason=reason,
    ))
    result["steps"].append({
        "action": "cookie_consent_precondition",
        "status": "failed",
        "error": "Cookie consent requires attention.",
    })
    if evidence_directory is None:
        return
    identity = current_evidence_execution()
    execution_id = identity.execution_id if identity is not None else uuid4()
    test_step_id = identity.test_step_id if identity is not None else None
    step_part = test_step_id.hex if test_step_id is not None else "unbound"
    directory = Path(evidence_directory)
    screenshot_path = directory / (
        f"execution-{execution_id.hex}-step-{step_part}-cookie-consent-page.png"
    )
    try:
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot_path), mask=[page.locator(_PRIVATE_INPUTS)])
    except Exception:
        return
    result.setdefault("evidence", []).append({
        "type": "SCREENSHOT",
        "path": str(screenshot_path),
        "description": "Page screenshot captured because cookie consent requires attention.",
        "scope": EvidenceScope.PAGE.value,
        "event": "COOKIE_CONSENT_FAILURE",
    })


def _cookie_consent_public_dict(record: CookieConsentRecord) -> dict[str, str | None]:
    return {
        "policy": record.policy.value,
        "status": record.status.value,
        "reason": record.reason.value if record.reason is not None else None,
    }


def _capture_screenshots(
    page: Any,
    evidence_directory: str | Path | None,
    result: dict[str, Any],
    *,
    action: QATestStep,
    action_index: int,
    event_kind: str,
) -> None:
    """Capture configured scopes without changing the browser action result."""
    if evidence_directory is None:
        return
    if current_cancellation() is not None and current_cancellation().requested:
        return
    policy = current_evidence_policy()
    selector = action.parameters.get("selector")
    has_locator = isinstance(selector, str) and bool(selector.strip())
    scopes = (
        (EvidenceScope.ELEMENT, EvidenceScope.PAGE)
        if policy.screenshot_mode == ScreenshotMode.ELEMENT_AND_PAGE
        else (EvidenceScope.ELEMENT,)
        if policy.screenshot_mode == ScreenshotMode.ELEMENT
        else (EvidenceScope.PAGE,)
    )
    identity = current_evidence_execution()
    run_execution_id = identity.execution_id if identity is not None else uuid4()
    test_step_id = identity.test_step_id if identity is not None else None
    directory = Path(evidence_directory)

    for scope in scopes:
        if current_cancellation() is not None and current_cancellation().requested:
            break
        if scope == EvidenceScope.ELEMENT and not has_locator:
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
            step_part = test_step_id.hex if test_step_id is not None else "unbound"
            screenshot_path = directory / (
                f"execution-{run_execution_id.hex}-step-{step_part}-"
                f"event-{action_index}-{scope.value.casefold()}.png"
            )
            if scope == EvidenceScope.ELEMENT:
                page.locator(selector).screenshot(
                    path=str(screenshot_path),
                    timeout=ACTION_TIMEOUT_MS,
                    mask=[page.locator(_PRIVATE_INPUTS)],
                )
            else:
                page.screenshot(path=str(screenshot_path), mask=[page.locator(_PRIVATE_INPUTS)])
            if event_kind == "FAILURE":
                description = f"Failure screenshot after {action.action} action."
            elif event_kind == "VERIFICATION":
                description = f"Verification screenshot after {action.action}."
            else:
                description = "Screenshot after the TestStep completed."
            result.setdefault("evidence", []).append({
                "type": "SCREENSHOT",
                "path": str(screenshot_path),
                "description": description,
                "scope": scope.value,
                "event": f"{event_kind}:{action.action}:{action_index}",
            })
        except Exception as capture_error:
            warning = f"Screenshot capture failed ({type(capture_error).__name__})."
            result.setdefault("evidence_capture_warnings", []).append(warning)
            # Keep the legacy warning key for callers that inspect it, without
            # exposing exception text, selectors, URLs, or entered values.
            result["evidence_capture_error"] = warning
