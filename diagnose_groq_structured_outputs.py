"""Compare Groq strict Structured Outputs with a minimal and DNB-sized prompt.

Run from the project root with GROQ_API_KEY set:
    python diagnose_groq_structured_outputs.py

The API key is used only in the Authorization header and is never printed.
"""

import json
import os
from typing import Any

import httpx

from qa_agent.browser_discovery import capture_page_snapshot
from qa_agent.llm.errors import failure_detail_for, provider_http_failure
from qa_agent.llm.groq import GroqProvider
from qa_agent.models import QATestPlan


DNB_URL = "https://www.dnb.no/"
DNB_TASK = (
    "Verify that https://www.dnb.no loads successfully. If the cookie consent "
    "banner is visible, accept strictly necessary cookies using the discovered "
    "selector. Using at least three distinct navigation_paths from the browser "
    "snapshot, test each discovered menu and submenu combination: open the menu, "
    "select its discovered tab when present, click the discovered top-level menu "
    "item and submenu, then verify the exact expected URL and destination heading. "
    "Do not use menu positions or invent selectors or URLs."
)


def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qa_test_plan",
            "strict": True,
            "schema": GroqProvider._response_schema(),
        },
    }


def _show_request(label: str, prompt: str, snapshot: str) -> None:
    print(f"\n=== {label} ===")
    print(f"model: {GroqProvider._model}")
    print("strict: true")
    print(f"prompt characters: {len(prompt)}")
    print(f"page_snapshot characters: {len(snapshot)}")
    print("schema:")
    print(json.dumps(GroqProvider._response_schema(), ensure_ascii=False, indent=2))


def _post(api_key: str, prompt: str) -> httpx.Response:
    return httpx.post(
        GroqProvider._endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GroqProvider._model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": _response_format(),
        },
        timeout=30.0,
    )


def _report_response(response: httpx.Response) -> bool:
    print(f"HTTP status: {response.status_code}")
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
        print(f"Safe error category: {failure.category}")
        if failure.provider_error_code:
            print(f"Provider error code: {failure.provider_error_code}")
        if failure.provider_error_type:
            print(f"Provider error type: {failure.provider_error_type}")
        if failure.provider_error_field:
            print(f"Provider error field: {failure.provider_error_field}")
        return False
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        print(f"Could not read generated JSON content: {type(error).__name__}")
        return False
    try:
        QATestPlan.model_validate_json(content)
    except Exception as error:
        print(f"Generated content failed QATestPlan validation: {type(error).__name__}")
        return False
    print("Generated content passed QATestPlan.model_validate_json.")
    return True


def main() -> int:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is unavailable; no API request was made.")
        return 2

    minimal_prompt = (
        "Create one QA test step for https://example.com. Return a navigate step. "
        "Set all parameters fields not used by navigate to JSON null."
    )
    _show_request("Minimal prompt", minimal_prompt, "")
    try:
        minimal_response = _post(api_key, minimal_prompt)
    except httpx.RequestError as error:
        print(f"Minimal request transport failure: {type(error).__name__}")
        return 1
    minimal_ok = _report_response(minimal_response)

    print("\nCapturing current DNB page snapshot with existing browser discovery...")
    try:
        snapshot = capture_page_snapshot(DNB_URL)
    except Exception as error:
        print(f"DNB snapshot capture failed: {type(error).__name__}")
        return 1

    # Invoke the production prompt/schema path unchanged. Wrap only the HTTP
    # transport to report the outgoing prompt size before forwarding it.
    import qa_agent.llm.groq as groq_module

    original_post = groq_module.httpx.post

    def diagnostic_post(*args: Any, **kwargs: Any) -> httpx.Response:
        payload = kwargs["json"]
        prompt = payload["messages"][0]["content"]
        _show_request("Production DNB prompt", prompt, snapshot)
        return original_post(*args, **kwargs)

    groq_module.httpx.post = diagnostic_post
    try:
        print("\n=== Production GroqProvider request for DNB ===")
        GroqProvider().create_test_plan(DNB_TASK, DNB_URL, snapshot)
        print("Groq returned a plan accepted by QATestPlan.model_validate_json; content omitted.")
        dnb_ok = True
    except Exception as error:
        print(f"DNB request failed: {type(error).__name__}")
        dnb_ok = False
    finally:
        groq_module.httpx.post = original_post

    if minimal_ok and dnb_ok:
        print("\nResult: both minimal and DNB-sized requests succeeded.")
        return 0
    if minimal_ok:
        print("\nResult: minimal strict schema succeeded; failure is specific to the DNB-sized request/prompt path.")
    elif not dnb_ok:
        print("\nResult: minimal request also failed; strict schema/model generation is implicated before DNB prompt size.")
    else:
        print("\nResult: minimal request failed while the DNB production request succeeded; inspect request differences.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
