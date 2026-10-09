"""Safe diagnostics survive classification, SQLite history, reports and progress."""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from unittest.mock import Mock, patch

import pytest
from playwright.sync_api import TimeoutError as BrowserTimeout

from qa_agent.browser_runner import _run_plan_on_page
from qa_agent.cookie_consent import CookieConsentPolicy, cookie_consent_scope
from qa_agent.execution_diagnostics import ActionFailureDiagnostic, action_failure_diagnostic, safe_selector_identity
from qa_agent.execution_progress import ExecutionEventType, ExecutionProgressStore, ExecutionProgressReporter, active_execution_progress
from qa_agent.models import AssertionGroundingEntry, AssertionGrounding, Execution, ExecutionStatus, QATestPlan, TestCase as Case, TestPlan as Plan, TestPlanVersion as Version, TestStep as Step, TestRun as Run
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.reporting import RunReportGenerator, TestReportGenerator as DomainReportGenerator
from qa_agent.run_history import WorkflowType
from qa_agent.run_context import RunContext
from qa_agent.storage import create_sqlite_storage
from qa_agent.web import LocalWebApplication


@pytest.mark.parametrize("action,count,grounding,error,classification", [
    ("assert_visible", 1, "REQUIREMENT_GROUNDED", AssertionError("Actual value: unrestricted-private-content\nwaiting for locator timeout"), "PRODUCT_FAILURE"),
    ("assert_hidden", 1, "REQUIREMENT_GROUNDED", AssertionError("Actual value: unrestricted-private-content"), "PRODUCT_FAILURE"),
    ("assert_visible", 1, "OBSERVATION_GROUNDED", AssertionError("Actual value: unrestricted-private-content"), "AUTOMATION_DRIFT"),
    ("assert_visible", 0, "REQUIREMENT_GROUNDED", AssertionError("selector resolved to 0 elements"), "AUTOMATION_EXECUTION_ERROR"),
    ("click", 0, "REQUIREMENT_GROUNDED", BrowserTimeout("Timeout waiting for locator"), "AUTOMATION_DRIFT"),
    ("click", 1, "REQUIREMENT_GROUNDED", RuntimeError("target page, context or browser has been closed"), "INFRASTRUCTURE_ERROR"),
])
def test_safe_action_diagnostic_survives_sqlite_reporting_and_live_progress(tmp_path, action, count, grounding, error, classification):
    private = "unregistered-fixture-password"
    step = Step(name="Verify registration", description="Verify the registration state.", expected="An error is displayed.", order=0)
    case = Case(name="Diagnostic regression", description=step.description, base_url="http://127.0.0.1/demo", steps=[step])
    plan = QATestPlan(url=case.base_url, steps=[
        {"action": "fill", "parameters": {"selector": "#password", "value": private}},
        {"action": action, "parameters": {"selector": "#form-error"}},
    ])
    owner = Plan(test_step_id=step.id, name=step.name)
    version = Version(test_plan_id=owner.id, version=1, qa_test_plan=plan,
        assertion_grounding=(AssertionGroundingEntry(step_index=1, category=AssertionGrounding(grounding)),) if action.startswith("assert_") else None)
    storage = create_sqlite_storage(tmp_path / "diagnostics.sqlite3")
    storage.plan_store.save(step.id, version, test_plan=owner)
    frozen = version.model_dump_json()
    page, target = Mock(), Mock()
    page.url = plan.url
    page.is_closed.return_value = False
    page.evaluate.return_value = "complete"
    page.locator.return_value = target
    target.count.return_value = count
    target.is_visible.return_value = False
    target.is_enabled.return_value = True
    page.screenshot.side_effect = lambda *, path, mask: Path(path).write_bytes(b"png")
    if action == "click":
        target.click.side_effect = error
    assertions = Mock()
    assertions.to_be_visible.side_effect = error
    assertions.to_be_hidden.side_effect = error
    runner = lambda candidate: _run_plan_on_page(candidate, page, tmp_path / "evidence")
    store = ExecutionProgressStore()
    progress_id = store.create(case.id, WorkflowType.AUTOMATION)
    reporter = ExecutionProgressReporter(store, progress_id)
    with cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED), patch("qa_agent.browser_runner.expect", return_value=assertions), active_execution_progress(reporter):
        reporter.test_case_loaded(case)
        outcome = PlanExecutionService(runner, storage.execution_repository).execute(step, version)
    assert outcome.classification.value == classification
    diagnostic = action_failure_diagnostic(outcome.execution.runner_result)
    assert diagnostic.action_index == 1 and diagnostic.action == action
    assert diagnostic.selector_identity == "#form-error" and diagnostic.target_count == count
    assert diagnostic.evidence_indexes == (0,) and outcome.execution.evidence
    assert diagnostic.classification == classification
    public_progress = store.get(progress_id).to_public_dict()
    assert public_progress['steps'][0]['action_failure'] == diagnostic.model_dump(mode='json')
    assert public_progress['events'][-1]['action_failure'] == diagnostic.model_dump(mode='json')
    assert diagnostic.exception_category == ("TIMEOUT" if isinstance(error, BrowserTimeout) else "BROWSER_ERROR" if classification == "INFRASTRUCTURE_ERROR" else "ASSERTION_FAILURE")
    run = Run.from_test_case(case, [outcome.execution])
    storage.run_history.record_completed_run(case, run, workflow_type=WorkflowType.AUTOMATION, outcome=classification)
    app = LocalWebApplication(storage.run_history, progress_store=store, plan_store=storage.plan_store, evidence_root=tmp_path)
    try:
        live = json.loads(app.handle('GET', f'/api/progress/{progress_id}').body)
        assert live['steps'][0]['action_failure'] == diagnostic.model_dump(mode='json')
        assert b'Action failure details' in app.handle('GET', f'/runs/progress/{progress_id}').body
        reporter.finish(run_id=run.id, run_status='FAILED', outcome=classification)
        saved = json.loads(app.handle('GET', f'/api/progress/{progress_id}').body)
        assert saved['steps'][0]['action_failure'] == diagnostic.model_dump(mode='json')
        assert saved['steps'][0]['evidence_links']
    finally:
        app.close()
    record = storage.run_history.get(run.id)
    report = RunReportGenerator().generate_history(record)
    assert report.steps[0].attempts[0].action_failure == diagnostic
    html = RunReportGenerator().to_html(report)
    public = json.dumps(store.get(progress_id).to_public_dict()) + report.to_json() + html + record.model_dump_json() + outcome.execution.model_dump_json()
    assert "Action 2" in public and "#form-error" in public
    assert private not in public and "unrestricted-private-content" not in public
    assert storage.plan_store.get_version(version.id).model_dump_json() == frozen
    assert page.screenshot.call_args.kwargs["mask"] == [target]


def test_complex_or_sensitive_selector_identity_is_hashed_without_input_values():
    selector = 'input[value="private-value"]'
    identity = safe_selector_identity(selector, ["private-value"])
    assert identity.startswith("selector-sha256:") and "private-value" not in identity
    assert safe_selector_identity("#private-value", ["private-value"]).startswith("selector-sha256:")


def test_sensitive_run_context_selector_is_redacted_from_structured_progress_and_history(tmp_path):
    secret = 'session-private-marker'
    context = RunContext()
    context.set_value('session_token', secret, sensitive=True)
    step = Step(name='Verify state', description='Verify state.', expected='The state is displayed.', order=0)
    case = Case(name='Safe diagnostic', description=step.description, steps=[step])
    diagnostic = ActionFailureDiagnostic(action_index=0, action='assert_visible', selector_identity='#' + secret,
        exception_category='ASSERTION_FAILURE', exception_type='AssertionError', classification='PRODUCT_FAILURE')
    store = ExecutionProgressStore()
    progress_id = store.create(case.id, WorkflowType.AUTOMATION)
    reporter = ExecutionProgressReporter(store, progress_id, context)
    reporter.test_case_loaded(case)
    reporter.emit(ExecutionEventType.STEP_STARTED, step=step)
    reporter.emit(ExecutionEventType.STEP_FAILED, step=step, classification='PRODUCT_FAILURE', action_failure=diagnostic)
    assert secret not in json.dumps(store.get(progress_id).to_public_dict())
    execution = Execution(test_step_id=step.id, test_plan_version_id=uuid4(), status=ExecutionStatus.FAILED,
        actual_result='failed', finished_at=datetime.now(timezone.utc),
        runner_result={'status': 'failed', 'classification': 'PRODUCT_FAILURE', 'steps': [
            {'action': 'assert_visible', 'status': 'failed', 'diagnostic': diagnostic.model_dump(mode='json')} ]})
    storage = create_sqlite_storage(tmp_path / 'context-diagnostics.sqlite3')
    run = Run.from_test_case(case, [execution], run_context=context)
    assert secret not in DomainReportGenerator().generate(run).to_json()
    record = storage.run_history.record_completed_run(case, run, workflow_type=WorkflowType.AUTOMATION, outcome='PRODUCT_FAILURE')
    assert secret not in record.model_dump_json()
    report = RunReportGenerator().generate_history(record)
    assert secret not in report.to_json() and secret not in RunReportGenerator().to_html(report)
