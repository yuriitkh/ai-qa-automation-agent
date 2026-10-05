from pathlib import Path
from typing import Any
from uuid import uuid4

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from .models import QATestPlan


class BrowserRunner:
    """Execute plans in Playwright, optionally saving screenshots on failure."""

    def __init__(self, evidence_directory: str | Path | None = None) -> None:
        self.evidence_directory = Path(evidence_directory) if evidence_directory is not None else None

    def __call__(self, plan: QATestPlan) -> dict[str, Any]:
        return run_test_plan(plan, evidence_directory=self.evidence_directory)


def run_test_plan(
    plan: QATestPlan,
    evidence_directory: str | Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "url": plan.url,
        "steps": [],
        "evidence": [],
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        try:
            page = browser.new_page()

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
                        selector, expected = step.parameters["selector"], step.parameters["expected"]
                        selected = page.locator(selector).locator("option:checked").first
                        actual_label = selected.inner_text()
                        actual_value = selected.get_attribute("value")
                        if expected not in {actual_label, actual_value}:
                            raise AssertionError(
                                f"Expected selected option {expected!r} for {selector!r}, "
                                f"got label={actual_label!r}, value={actual_value!r}."
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
        finally:
            browser.close()

    return result
