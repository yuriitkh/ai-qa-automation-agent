from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from .models import QATestPlan


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
        if self._closed or self._playwright_manager is not None:
            raise RuntimeError("BrowserSession has already been started or closed.")
        self._playwright_manager = sync_playwright()
        self._playwright = self._playwright_manager.__enter__()
        self._playwright_entered = True
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context()

    def new_page(self) -> Any:
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

    def __call__(self, plan: QATestPlan) -> dict[str, Any]:
        with self.open_session() as session:
            page = session.new_page()
            return session.run_plan(page, plan)


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
    result: dict[str, Any] = {
        "status": "passed",
        "url": plan.url,
        "steps": [],
        "evidence": [],
    }

    for step in plan.steps:
        step_result = {
            "action": step.action,
            "status": "passed",
            "error": "",
        }

        try:
            if step.action == "navigate":
                page.goto(step.parameters["url"])
            elif step.action == "click":
                selector = step.parameters["selector"]
                try:
                    element = page.locator(selector)
                    element.wait_for(state="visible", timeout=5000)
                    click_completed = False
                    try:
                        with page.expect_navigation(
                            wait_until="commit", timeout=10000
                        ):
                            element.click()
                            click_completed = True
                    except PlaywrightTimeoutError:
                        if not click_completed:
                            raise
                    page.wait_for_load_state("load", timeout=10000)
                    print(f"CLICK: {selector} -> URL: {page.url}")
                except AssertionError:
                    raise
                except Exception as error:
                    raise AssertionError(
                        f"Could not click element matching selector "
                        f"{selector!r}: {error}"
                    ) from error
            elif step.action == "fill":
                selector = step.parameters["selector"]
                value = step.parameters["value"]
                try:
                    element = page.locator(selector)
                    if not element.count():
                        raise AssertionError(
                            f"Selector {selector!r} was not found on the page."
                        )
                    element.fill(value)
                except AssertionError:
                    raise
                except Exception as error:
                    raise AssertionError(
                        f"Could not fill element matching selector "
                        f"{selector!r}: {error}"
                    ) from error
            elif step.action == "select_option":
                selector = step.parameters["selector"]
                label = step.parameters["option_label"]
                try:
                    page.locator(selector).select_option(label=label)
                except Exception as error:
                    raise AssertionError(
                        f"Could not select option label {label!r} in {selector!r}: {error}"
                    ) from error
            elif step.action == "assert_page_loaded":
                page.wait_for_load_state("load")
            elif step.action == "assert_title":
                expected_title = step.parameters["expected"]
                actual_title = page.title()
                if actual_title != expected_title:
                    raise AssertionError(
                        f"Expected title {expected_title!r}, "
                        f"but got {actual_title!r}."
                    )
            elif step.action == "assert_url":
                expected_url = step.parameters["expected"]
                actual_url = page.url
                if actual_url != expected_url:
                    raise AssertionError(
                        f"Expected URL {expected_url!r}, "
                        f"but got {actual_url!r}."
                    )
            elif step.action == "assert_visible":
                selector = step.parameters["selector"]
                expected_text = step.parameters["expected_text"]
                element = page.locator(selector)
                element_count = element.count()
                if not element_count:
                    raise AssertionError(
                        f"Selector {selector!r} was not found on the page."
                    )
                element = element.first
                if not element.is_visible():
                    raise AssertionError(
                        f"Element matching selector {selector!r} exists "
                        "but is not visible."
                    )
                try:
                    actual_text = element.inner_text(timeout=1000)
                except Exception as error:
                    raise AssertionError(
                        f"Could not read visible text for selector "
                        f"{selector!r} within 1 second: {error}"
                    ) from error
                comparable_expected_text = expected_text.replace("\\n", "\n")
                if actual_text != comparable_expected_text:
                    raise AssertionError(
                        f"Expected text {expected_text!r} was not present "
                        f"as the exact visible text for {selector!r}; "
                        f"got {actual_text!r}."
                    )
            elif step.action == "assert_text_contains":
                expected_text = step.parameters["expected_text"].replace("\\n", "\n")
                selector = step.parameters.get("selector")
                actual_text = (
                    page.locator(selector).inner_text(timeout=1000)
                    if selector else page.locator("body").inner_text(timeout=1000)
                )
                if expected_text not in actual_text:
                    raise AssertionError(
                        f"Expected text {expected_text!r} to be contained in visible text"
                        f"{f' for {selector!r}' if selector else ''}; got {actual_text!r}."
                    )
            elif step.action == "assert_checked":
                selector = step.parameters["selector"]
                if not page.locator(selector).is_checked():
                    raise AssertionError(f"Expected checkbox/radio {selector!r} to be checked.")
            elif step.action == "assert_selected":
                selector = step.parameters["selector"]
                element = page.locator(selector)
                element_info = element.evaluate(
                    "element => ({tag: element.tagName.toLowerCase(), type: element.type})"
                )
                expected = step.parameters.get("expected")
                if element_info["tag"] == "input" and element_info["type"] == "radio":
                    if expected is not None:
                        raise AssertionError(
                            f"assert_selected for radio {selector!r} does not take expected."
                        )
                    if not element.is_checked():
                        raise AssertionError(f"Expected radio {selector!r} to be selected.")
                elif element_info["tag"] == "select":
                    if not isinstance(expected, str):
                        raise AssertionError(
                            f"assert_selected for select {selector!r} requires expected option label or value."
                        )
                    selected = element.locator("option:checked").first
                    actual_label = selected.inner_text()
                    actual_value = selected.get_attribute("value")
                    if expected not in {actual_label, actual_value}:
                        raise AssertionError(
                            f"Expected selected option {expected!r} for {selector!r}, "
                            f"got label={actual_label!r}, value={actual_value!r}."
                        )
                else:
                    raise AssertionError(
                        f"assert_selected selector {selector!r} must target a radio or select."
                    )
            elif step.action in {"assert_enabled", "assert_disabled"}:
                selector = step.parameters["selector"]
                enabled = page.locator(selector).is_enabled()
                expected_enabled = step.action == "assert_enabled"
                if enabled != expected_enabled:
                    state = "enabled" if enabled else "disabled"
                    wanted = "enabled" if expected_enabled else "disabled"
                    raise AssertionError(f"Expected {selector!r} to be {wanted}, but it is {state}.")

            elif step.action == "assert_hidden":
                selector = step.parameters["selector"]
                try:
                    element = page.locator(selector)
                    if not element.is_hidden():
                        raise AssertionError(
                            f"Element matching selector {selector!r} is visible."
                        )
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
            step_result["status"] = "failed"
            step_result["error"] = str(error)
            result["status"] = "failed"
            if evidence_directory is not None:
                try:
                    directory = Path(evidence_directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    screenshot_path = directory / f"execution-{uuid4().hex}.png"
                    page.screenshot(path=str(screenshot_path))
                    result["evidence"].append({
                        "type": "SCREENSHOT",
                        "path": str(screenshot_path),
                        "description": f"Browser state after failed {step.action} step.",
                    })
                except Exception as capture_error:
                    result["evidence_capture_error"] = str(capture_error)
            result["steps"].append(step_result)
            break

        result["steps"].append(step_result)
    return result
