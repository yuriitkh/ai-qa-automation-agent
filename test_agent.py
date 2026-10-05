import json
import sys

from qa_agent.gemini_client import create_test_plan, get_selected_provider_name
from qa_agent.browser_runner import run_test_plan

task = (
    "Open https://www.selenium.dev/selenium/web/web-form.html, find the native select labeled "
    "'Dropdown (select)', choose the option 'Two', and verify that 'Two' is selected."
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
