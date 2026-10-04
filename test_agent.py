import json
import sys

from qa_agent.gemini_client import create_test_plan, get_selected_provider_name
from qa_agent.browser_runner import run_test_plan

task = (
    "Verify that https://www.nav.no/ loads successfully. If the cookie consent "
    "banner is visible, accept strictly necessary cookies using the discovered "
    "selector. Using at least three distinct navigation_paths from the browser "
    "snapshot, test each discovered menu and submenu combination: open the menu, "
    "select its discovered tab when present, click the discovered top-level menu "
    "item and submenu, then verify the exact expected URL and destination heading. "
    "Do not use menu positions or invent selectors or URLs."
)

plan = create_test_plan(task)
print(f"LLM PROVIDER: {get_selected_provider_name()}")
result = run_test_plan(plan)

print("GENERATED TEST PLAN")
print(plan.model_dump_json(indent=2))
print("\nEXECUTION RESULT")
print(json.dumps(result, indent=2))

if result.get("status") == "failed":
    sys.exit(1)
