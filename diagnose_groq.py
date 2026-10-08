"""Run Groq Structured Outputs diagnostics through GroqProvider.

Run from the project root with GROQ_API_KEY set:
    python diagnose_groq.py
    python diagnose_groq.py --test 4
    python diagnose_groq.py --test 5
    python diagnose_groq.py --test 6

This diagnostic does not print the API key and does not modify production code.
"""

import argparse
import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import httpx

from qa_agent.browser_discovery import capture_page_snapshot
from qa_agent.llm.errors import failure_detail_for, provider_http_failure
from qa_agent.llm.groq import GroqProvider
import qa_agent.llm.groq as groq_module


EXAMPLE_URL = "https://example.com"
DNB_URL = "https://www.dnb.no/"
SIMPLE_TASK = (
    "Create one QA test plan for https://example.com. Return exactly one "
    "navigate action with url https://example.com."
)
DNB_TASK = (
    "Verify that https://www.dnb.no loads successfully. If the cookie consent "
    "banner is visible, accept strictly necessary cookies using the discovered "
    "selector. Using at least three distinct navigation_paths from the browser "
    "snapshot, test each discovered menu and submenu combination: open the menu, "
    "select its discovered tab when present, click the discovered top-level menu "
    "item and submenu, then verify the exact expected URL and destination heading. "
    "Do not use menu positions or invent selectors or URLs."
)
DNB_PATH_TASK_TEMPLATE = (
    "Verify that https://www.dnb.no loads successfully. Using {selection} from "
    "the browser snapshot, open the menu, select its discovered tab when present, "
    "click the discovered top-level menu item and its discovered submenu, then "
    "verify the exact expected URL and destination heading. Do not use menu "
    "positions or invent selectors or URLs."
)
DNB_ONE_PATH_TASK = DNB_PATH_TASK_TEMPLATE.format(
    selection="the first discovered navigation_path"
)
DNB_TWO_PATHS_TASK = DNB_PATH_TASK_TEMPLATE.format(
    selection="the first two discovered navigation_paths"
)
DNB_THREE_PATHS_TASK = DNB_PATH_TASK_TEMPLATE.format(
    selection="the first three discovered navigation_paths"
)


def run_case(name: str, task: str, target_url: str, snapshot: str) -> None:
    original_post = groq_module.httpx.post
    captured: dict[str, Any] = {"response": None, "prompt_length": None}

    def capture_post(*args: Any, **kwargs: Any) -> httpx.Response:
        payload = kwargs.get("json", {})
        messages = payload.get("messages", [])
        if messages and isinstance(messages[0].get("content"), str):
            captured["prompt_length"] = len(messages[0]["content"])
        response = original_post(*args, **kwargs)
        captured["response"] = response
        return response

    groq_module.httpx.post = capture_post
    error: Exception | None = None
    try:
        # Suppress provider diagnostics here so each test has one clear report.
        with redirect_stdout(StringIO()):
            GroqProvider().create_test_plan(task, target_url, snapshot)
    except Exception as caught:
        error = caught
    finally:
        groq_module.httpx.post = original_post

    response = captured["response"]
    print(f"\n=== {name} ===")
    print(f"result: {'success' if error is None else 'failure'}")
    print(f"HTTP status: {response.status_code if response is not None else 'unavailable'}")
    print(f"model: {GroqProvider._model}")
    print(f"prompt length: {captured['prompt_length'] if captured['prompt_length'] is not None else 'unavailable'} characters")
    print(f"snapshot length: {len(snapshot)} characters")
    if response is not None:
        if response.is_error:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            failure = failure_detail_for(
                "Groq",
                provider_http_failure(
                    "Groq", response.status_code,
                    payload=payload, headers=response.headers,
                ),
            )
            print(f"safe error category: {failure.category}")
            if failure.provider_error_code:
                print(f"provider error code: {failure.provider_error_code}")
            if failure.provider_error_type:
                print(f"provider error type: {failure.provider_error_type}")
            if failure.provider_error_field:
                print(f"provider error field: {failure.provider_error_field}")
        else:
            print("Structured output accepted and validated; response content omitted.")
    elif error is not None:
        print(f"Request failed before receiving an HTTP response: {type(error).__name__}")
    if error is not None and response is not None and not response.is_error:
        print(f"Provider rejected the successful HTTP response: {type(error).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        type=int,
        choices=(4, 5, 6),
        help="run only the selected DNB navigation-path test",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is unavailable; no Groq API requests were made.")
        return 2

    # Capture once and reuse the exact same current DNB snapshot for tests 2, 4-6.
    print("Capturing the current DNB page snapshot...")
    try:
        dnb_snapshot = capture_page_snapshot(DNB_URL)
    except Exception as error:
        print(f"DNB snapshot capture failed: {type(error).__name__}: {error}")
        return 1

    if args.test is not None:
        selected_cases = {
            4: ("Test 4 - first one navigation path", DNB_ONE_PATH_TASK),
            5: ("Test 5 - first two navigation paths", DNB_TWO_PATHS_TASK),
            6: ("Test 6 - first three navigation paths", DNB_THREE_PATHS_TASK),
        }
        name, task = selected_cases[args.test]
        run_case(name, task, DNB_URL, dnb_snapshot)
        return 0

    run_case("Test 1 — minimal task, no snapshot", SIMPLE_TASK, EXAMPLE_URL, "")
    run_case("Test 2 — minimal task, current DNB snapshot", SIMPLE_TASK, EXAMPLE_URL, dnb_snapshot)
    run_case("Test 3 — DNB task, no snapshot", DNB_TASK, DNB_URL, "")
    run_case("Test 4 - first one navigation path", DNB_ONE_PATH_TASK, DNB_URL, dnb_snapshot)
    run_case("Test 5 - first two navigation paths", DNB_TWO_PATHS_TASK, DNB_URL, dnb_snapshot)
    run_case("Test 6 - first three navigation paths", DNB_THREE_PATHS_TASK, DNB_URL, dnb_snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
