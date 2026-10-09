"""Outcome reporting contracts without providers or external browser targets."""

import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from qa_agent.execution_progress import ExecutionEventType as Event, ExecutionProgressReporter, ExecutionProgressStore
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.execution_trace import ProviderAttemptOutcome, ProviderAttemptTrace, RequestKind, StepTrace
from qa_agent.models import (
    AssertionGrounding, AssertionGroundingEntry, Execution, ExecutionStatus,
    PlanVersionOrigin, QATestPlan, QATestStep,
    TestCase as DomainCase, TestPlan as DomainPlan,
    TestPlanVersion as DomainVersion, TestRun as DomainRun, TestStep as DomainStep,
)
from qa_agent.pipeline import _automation_run_outcome
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.reporting import RunReportGenerator, TestReportGenerator as BasicReportGenerator
from qa_agent.result_semantics import history_outcome, result_label, result_outcome
from qa_agent.run_context import RunContext
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryRecord, RunHistoryService, WorkflowType
from qa_agent.suite_runs import (
    SuiteRun, SuiteRunAttempt, SuiteRunConfig, SuiteRunItem, SuiteRunItemStatus,
    SuiteRunStatus, suite_run_report_json,
)
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication, _suite_run_public_dict


NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def case_and_run(outcome="AUTOMATION_EXECUTION_ERROR", *, classification=None, actual="failed", error=None):
    case = DomainCase(name="Account confirmation", description="Verify confirmation.", steps=[
        DomainStep(name="Confirm account", description="Check confirmation.", expected="Account confirmed.", order=0),
    ])
    execution = Execution(
        test_step_id=case.steps[0].id, test_plan_version_id=uuid4(),
        status=ExecutionStatus.PASSED if outcome == "PASSED" else ExecutionStatus.FAILED,
        started_at=NOW, finished_at=NOW, actual_result=actual, error=error,
        runner_result={"qa_classification": classification} if classification else None,
    )
    run = DomainRun.from_test_case(case, [execution])
    record = RunHistoryRecord.from_completed_run(case, run, workflow_type=WorkflowType.REGRESSION, outcome=outcome)
    return case, run, record


@pytest.mark.parametrize("outcome,label", [
    ("PRODUCT_FAILURE", "Failed"), ("AUTOMATION_EXECUTION_ERROR", "Automation Error"),
    ("AUTOMATION_GENERATION_ERROR", "Generation Error"), ("INFRASTRUCTURE_ERROR", "Infrastructure Error"),
    ("SETUP_FAILURE", "Blocked"), ("AUTOMATION_DRIFT", "Automation Error"),
    (None, "Inconclusive"), ("UNKNOWN", "Inconclusive"), ("FAILED", "Inconclusive"),
])
def test_historical_failed_status_never_invents_product_failure(outcome, label):
    case, run, record = case_and_run(outcome)
    record = record.model_copy(update={"outcome": outcome})
    original = record.model_dump_json()
    report = RunReportGenerator().generate_history(record)
    assert report.display_status == label
    assert record.model_dump_json() == original
    payload = json.loads(report.to_json())
    assert payload["status"] == "FAILED"
    assert payload["display_status"] == label
    html = RunReportGenerator().to_html(report)
    assert f'>{label}</span>' in html
    if outcome != "PRODUCT_FAILURE":
        assert '>Failed</span>' not in html
    repository = InMemoryRunHistoryRepository()
    repository.save(record)
    app = LocalWebApplication(RunHistoryService(repository))
    try:
        for url in ("/", "/runs", f"/test-cases/{case.id}", f"/runs/{run.id}", f"/runs/{run.id}/report.html"):
            response = app.handle("GET", url)
            assert response.status == 200
            text = response.body.decode()
            assert f'>{label}</span>' in text
            if label != "Failed":
                assert '>Failed</span>' not in text
    finally:
        app.close()


def test_missing_outcome_loads_and_incomplete_pass_is_inconclusive():
    _, _, record = case_and_run("PASSED")
    payload = record.model_dump(mode="json")
    payload.pop("outcome")
    assert history_outcome(RunHistoryRecord.model_validate(payload)) == "INCONCLUSIVE"
    incomplete = record.model_copy(update={"executions": []})
    assert history_outcome(incomplete) == "INCONCLUSIVE"
    assert result_outcome("FAILED", "PASSED") == "INCONCLUSIVE"


def test_unfinished_execution_and_unlinked_suite_attempt_are_not_reported_passed():
    case, run, _ = case_and_run("PASSED")
    unfinished = run.executions[0].model_copy(update={"finished_at": None})
    report = BasicReportGenerator().generate(run.model_copy(update={"executions": [unfinished], "finished_at": None}))
    assert report.display_outcome == "INCONCLUSIVE"
    assert report.steps[0].display_status == "Inconclusive"
    assert report.steps[0].attempts[0].display_status == "Inconclusive"
    attempt = SuiteRunAttempt(
        attempt_number=3, run_status="PASSED", outcome="PASSED",
        started_at=NOW, finished_at=NOW, duration_ms=0,
    )
    suite_run = SuiteRun(suite_id=uuid4(), suite_name="Smoke", config=SuiteRunConfig(), items=[SuiteRunItem(
        order_index=0, test_case_id=case.id, test_case_name=case.name,
        status=SuiteRunItemStatus.PASSED, attempts=[attempt],
    )])
    assert suite_run.passed_count == 0
    assert json.loads(suite_run_report_json(suite_run))["items"][0]["attempts"][0]["display_status"] == "Inconclusive"
    assert _suite_run_public_dict(suite_run)["items"][0]["attempts"][0]["display_status"] == "Inconclusive"


@pytest.mark.parametrize("grounding,expected", [
    (AssertionGrounding.REQUIREMENT_GROUNDED, "PRODUCT_FAILURE"),
    (AssertionGrounding.INFERRED, "AUTOMATION_EXECUTION_ERROR"),
    (AssertionGrounding.OBSERVATION_GROUNDED, "AUTOMATION_DRIFT"),
    (AssertionGrounding.UNKNOWN, "AUTOMATION_EXECUTION_ERROR"),
])
def test_automation_history_keeps_exact_assertion_grounding(grounding, expected):
    case, _, _ = case_and_run()
    plan = DomainPlan(test_step_id=case.steps[0].id, name="Confirmation")
    version = DomainVersion(
        test_plan_id=plan.id, version=1, origin=PlanVersionOrigin.AI_GENERATED,
        qa_test_plan=QATestPlan(url="http://127.0.0.1/target", steps=[QATestStep(action="assert_title", parameters={"expected": "Confirmed"})]),
        assertion_grounding=(AssertionGroundingEntry(step_index=0, category=grounding),),
    )
    result = PlanExecutionService(lambda _: {
        "status": "failed", "steps": [{"action": "assert_title", "status": "failed", "error": "Actual value: Ready\nCall log:\n - waiting"}],
    }, InMemoryExecutionRepository()).execute(case.steps[0], version)
    run = DomainRun.from_test_case(case, [result.execution])
    assert _automation_run_outcome(run) == expected
    report = RunReportGenerator().generate_current(case, run, workflow_type=WorkflowType.AUTOMATION, outcome=expected)
    assert report.steps[0].attempts[0].classification == expected
    assert report.steps[0].attempts[0].observation == "Ready"
    assert "Call log:" in report.steps[0].attempts[0].diagnostics
    assert BasicReportGenerator().generate(run).display_outcome == expected


def test_mixed_attempts_keep_product_failure_and_technical_classifications():
    case, run, _ = case_and_run("PRODUCT_FAILURE", classification="PRODUCT_FAILURE")
    first = run.executions[0].model_copy(update={
        "id": uuid4(), "runner_result": {"qa_classification": "AUTOMATION_EXECUTION_ERROR"},
    })
    run = run.model_copy(update={"executions": [first, run.executions[0]]})
    report = RunReportGenerator().generate_current(case, run, workflow_type=WorkflowType.REGRESSION, outcome="PRODUCT_FAILURE")
    assert [item.classification for item in report.steps[0].attempts] == ["AUTOMATION_EXECUTION_ERROR", "PRODUCT_FAILURE"]
    assert report.display_status == "Failed"
    legacy = RunHistoryRecord.from_completed_run(case, run, workflow_type=WorkflowType.REGRESSION, outcome="PRODUCT_FAILURE")
    legacy = legacy.model_copy(update={"executions": [item.model_copy(update={"classification": None}) for item in legacy.executions]})
    assert all(item.classification == "INCONCLUSIVE" for item in RunReportGenerator().generate_history(legacy).steps[0].attempts)


def test_progress_generation_stop_summary_grouping_and_unsaved_plan():
    store = ExecutionProgressStore(clock=lambda: NOW)
    case = DomainCase(name="Registration", description="Check flow.", steps=[
        DomainStep(name=f"Action {index + 1}", description="Do action.", expected="Action completes.", order=index)
        for index in range(3)
    ])
    progress_id = store.create(case.id, WorkflowType.AUTOMATION)
    reporter = ExecutionProgressReporter(store, progress_id)
    reporter.emit(Event.RUN_STARTED)
    reporter.test_case_loaded(case)
    for step in case.steps[:2]:
        reporter.emit(Event.PLAN_GENERATION_STARTED, step=step)
        reporter.emit(Event.PLAN_GENERATED, step=step, plan_version=1)
        reporter.emit(Event.STEP_STARTED, step=step)
        reporter.emit(Event.STEP_PASSED, step=step, classification="PASSED")
    reporter.emit(Event.PLAN_GENERATION_STARTED, step=case.steps[2])
    reporter.emit(Event.PLAN_GENERATION_FAILED, step=case.steps[2], classification="AUTOMATION_GENERATION_ERROR", failure_code="PLAN_VALIDATION_FAILED", message="The assertion could not be validated.")
    reporter.finish(outcome="AUTOMATION_GENERATION_ERROR", error_category="AUTOMATION_GENERATION_ERROR")
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), progress_store=store)
    try:
        payload = json.loads(app.handle("GET", f"/api/progress/{progress_id}").body)
        assert payload["phase"] == "Stopped — Generation Error"
        assert payload["summary"]["stopping_step"] == 3
        assert payload["summary"]["completed_steps"] == 2
        assert payload["final_run_id"] is None
        assert all(step["plan_url"] is None for step in payload["steps"])
        stages = [group["stage"] for group in payload["diagnostic_stages"]]
        assert stages[:3] == ["Preparation", "TestCase loading", "Automation generation"]
        assert "Plan validation" in stages
        assert "Browser execution" in stages
        text = app.handle("GET", f"/runs/progress/{progress_id}").body.decode()
        assert "Stopped at Step 3 of 3" in text
        assert "No persisted Run was created" in text
        assert "No saved TestPlan available." in text
        assert "Retry Automation" in text
        assert text.count("data-progress-phase") == 1
        assert '>Failed</span>' not in text
    finally:
        app.close()


def test_diagnostics_collapse_only_redundant_events_and_record_real_attempts():
    store = ExecutionProgressStore(clock=lambda: NOW)
    case, _, _ = case_and_run()
    progress_id = store.create(case.id, WorkflowType.REGRESSION)
    reporter = ExecutionProgressReporter(store, progress_id)
    reporter.test_case_loaded(case)
    reporter.emit(Event.AUTOMATION_PREPARATION_STARTED, message="Preparing automation.")
    reporter.emit(Event.AUTOMATION_PREPARATION_STARTED, message="Preparing automation.")
    for _ in range(2):
        reporter.emit(Event.STEP_STARTED, step=case.steps[0])
        reporter.emit(Event.STEP_FAILED, step=case.steps[0], classification="AUTOMATION_EXECUTION_ERROR")
    reporter.finish(outcome="AUTOMATION_EXECUTION_ERROR")
    payload = store.get(progress_id).to_public_dict()
    events = [event for group in payload["diagnostic_stages"] for event in group["events"]]
    assert sum(event["type"] == "AUTOMATION_PREPARATION_STARTED" for event in events) == 1
    assert [event["attempt_number"] for event in events if event["type"] == "STEP_FAILED"] == [1, 2]
    assert all(event["duration_ms"] == 0 for event in events if event["type"] == "STEP_FAILED")
    assert not any("Repair" in group["stage"] for group in payload["diagnostic_stages"])


def test_progress_reuses_safe_provider_attempts_without_inventing_missing_metadata():
    store = ExecutionProgressStore(clock=lambda: NOW)
    case, _, _ = case_and_run()
    progress_id = store.create(case.id, WorkflowType.AUTOMATION)
    context = RunContext()
    context.set_value("access_token", "private-credential", sensitive=True)
    reporter = ExecutionProgressReporter(store, progress_id, context)
    reporter.test_case_loaded(case)
    trace = SimpleNamespace(steps=[StepTrace(
        test_step_id=case.steps[0].id, order=0, name="Confirm account", description="Check confirmation.", expected="Account confirmed.",
        provider_attempts=[
            ProviderAttemptTrace(
                provider_name="Provider <script>private-credential</script>", model="recorded-model",
                request_kind=RequestKind.TEST_PLAN, outcome=ProviderAttemptOutcome.RETRYABLE_ERROR,
                duration_ms=1200, error_class="RateLimitError", error_message="raw-private-response",
            ),
            ProviderAttemptTrace(
                provider_name="Fallback provider", request_kind=RequestKind.TEST_PLAN,
                outcome=ProviderAttemptOutcome.SUCCESS,
            ),
        ],
    )])
    reporter.capture_provider_diagnostics(trace)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), progress_store=store)
    try:
        payload = json.loads(app.handle("GET", f"/api/progress/{progress_id}").body)
        rows = payload["provider_diagnostics"]
        assert [row["attempt_number"] for row in rows] == [1, 2]
        assert rows[0]["duration_ms"] == 1200
        assert rows[0]["model"] == "recorded-model"
        assert rows[0]["reason"] == "RateLimitError"
        assert "duration_ms" not in rows[1] and "model" not in rows[1]
        assert all("timestamp" not in row for row in rows)
        html = app.handle("GET", f"/runs/progress/{progress_id}").body.decode()
        assert '<h3>Recorded provider attempts</h3>' in html
        assert 'Provider &lt;script&gt;[REDACTED]&lt;/script&gt;' in html
        assert 'recorded-model' in html
        for secret in ("private-credential", "raw-private-response"):
            assert secret not in html + json.dumps(payload)
        reporter.capture_provider_diagnostics(SimpleNamespace(steps=[object()]))
        assert store.get(progress_id).provider_diagnostics == tuple(rows)
        reporter.finish(outcome="AUTOMATION_GENERATION_ERROR")
        assert store.get(progress_id).finished_at is not None
    finally:
        app.close()


def test_reports_reject_incomplete_error_observations_and_redact_escape_diagnostics(monkeypatch):
    monkeypatch.setenv("REPORT_API_KEY", "configured-credential")
    case, run, record = case_and_run(error='Expected result\nCall log: Ready <script>alert(1)</script>\nfill("private-input")\ntoken=unknown-token\nconfigured-credential\nC:\\private\\trace.txt')
    report = RunReportGenerator().generate_history(record)
    assert report.steps[0].attempts[0].observation is None
    html = RunReportGenerator().to_html(report)
    assert "Expected result could not be verified." in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    for secret in ("private-input", "unknown-token", "configured-credential", "C:\\private"):
        assert secret not in html + report.to_json()
    for error in ("Actual value:", "Actual value: None", "Actual value: A\nActual value: B"):
        _, _, record = case_and_run(error=error)
        assert RunReportGenerator().generate_history(record).steps[0].attempts[0].observation is None


def test_suite_breakdown_retry_links_and_incomplete_passes():
    items = []
    outcomes = ["PRODUCT_FAILURE", "AUTOMATION_EXECUTION_ERROR", "INFRASTRUCTURE_ERROR", "SETUP_FAILURE", "FAILED", "PASSED"]
    for index, outcome in enumerate(outcomes):
        attempts = [SuiteRunAttempt(
            attempt_number=1, run_id=uuid4(), run_public_id=f"RUN-{index + 1:06d}",
            run_status="PASSED" if outcome == "PASSED" else "FAILED", outcome=outcome,
            started_at=NOW, finished_at=NOW, duration_ms=0,
        )]
        if outcome == "PASSED":
            attempts.insert(0, attempts[0].model_copy(update={"outcome": "AUTOMATION_EXECUTION_ERROR", "run_status": "FAILED", "run_id": uuid4()}))
            attempts[1] = attempts[1].model_copy(update={"attempt_number": 2})
        items.append(SuiteRunItem(
            order_index=index, test_case_id=uuid4(), test_case_name=f"Case {index}", attempts=attempts,
            status=SuiteRunItemStatus.PASSED_AFTER_RETRY if outcome == "PASSED" else SuiteRunItemStatus.FAILED,
        ))
    items.append(SuiteRunItem(order_index=6, test_case_id=uuid4(), test_case_name="Incomplete", status=SuiteRunItemStatus.PASSED))
    run = SuiteRun(suite_id=uuid4(), suite_name="Smoke", config=SuiteRunConfig(), status=SuiteRunStatus.COMPLETED_WITH_FAILURES, items=items)
    assert run.outcome_counts == {
        "passed": 1, "product_failures": 1, "automation_errors": 1, "generation_errors": 0,
        "infrastructure_errors": 1, "blocked": 1, "inconclusive": 2, "pending": 0,
    }
    assert run.failed_count == 1 and run.flaky_count == 1
    payload = json.loads(suite_run_report_json(run))
    assert payload["outcome_counts"] == run.outcome_counts
    assert [item["order_index"] for item in payload["items"]] == list(range(7))
    assert [attempt["run_id"] for attempt in payload["items"][5]["attempts"]] == [str(item.run_id) for item in items[5].attempts]
    public = _suite_run_public_dict(run)
    assert public["counts"]["failed"] == 1
    assert public["items"][1]["display_status"] == "Automation Error"


def test_dashboard_metrics_and_failed_filter_count_only_product_failures():
    repository = InMemoryRunHistoryRepository()
    for outcome in ("PRODUCT_FAILURE", "AUTOMATION_EXECUTION_ERROR", "AUTOMATION_DRIFT", "INFRASTRUCTURE_ERROR", "SETUP_FAILURE", "FAILED", "PASSED"):
        repository.save(case_and_run(outcome)[2])
    app = LocalWebApplication(RunHistoryService(repository))
    try:
        text = app.handle("GET", "/").body.decode()
        counts = dict(re.findall(r'<div class="card-label">([^<]+)</div><div class="card-value">(\d+)</div>', text))
        assert counts["Passed"] == "1"
        assert counts["Product failures"] == "1"
        assert counts["Automation execution errors"] == "1"
        assert counts["Automation drift"] == "1"
        assert counts["Infrastructure errors"] == "1"
        assert counts["Blocked"] == "1"
        assert counts["Inconclusive"] == "1"
        assert "Failed" not in counts
        filtered = app.handle("GET", "/runs?status=FAILED").body.decode()
        assert "Showing 1 of 7 recent runs" in filtered
    finally:
        app.close()


def test_plan_links_use_persisted_execution_version_after_latest_changes():
    case, run, _ = case_and_run()
    plans = InMemoryPlanStore()
    plan = DomainPlan(test_step_id=case.steps[0].id, name="Account confirmation")
    old = DomainVersion(
        id=run.executions[0].test_plan_version_id, test_plan_id=plan.id,
        version=1, origin=PlanVersionOrigin.AI_GENERATED,
        qa_test_plan=QATestPlan(url="http://127.0.0.1/target", steps=[QATestStep(action="assert_title", parameters={"expected": "Confirmed"})]),
    )
    plans.save(case.steps[0].id, old, test_plan=plan)
    history = RunHistoryService(InMemoryRunHistoryRepository(), plan_store=plans)
    saved = history.record_completed_run(case, run, workflow_type=WorkflowType.REGRESSION, outcome="AUTOMATION_EXECUTION_ERROR")
    cases = InMemoryTestCaseRepository()
    cases.save(case)
    store = ExecutionProgressStore()
    progress_id = store.create(case.id, WorkflowType.REGRESSION)
    reporter = ExecutionProgressReporter(store, progress_id)
    reporter.test_case_loaded(case)
    reporter.emit(Event.PLAN_REUSED, step=case.steps[0], plan_version=1, plan_version_id=old.id, plan_origin="AI_GENERATED")
    reporter.emit(Event.STEP_STARTED, step=case.steps[0])
    reporter.emit(Event.STEP_FAILED, step=case.steps[0], classification="AUTOMATION_EXECUTION_ERROR")
    reporter.finish(run_id=run.id, run_status="FAILED", outcome="AUTOMATION_EXECUTION_ERROR")
    newest = old.model_copy(update={"id": uuid4(), "version": 2, "previous_version_id": old.id})
    plans.save(case.steps[0].id, newest, test_plan=plan)
    original = old.model_dump_json()
    app = LocalWebApplication(history, test_cases=cases, plan_store=plans, progress_store=store)
    try:
        url = f"/runs/{saved.run_id}/plans/{old.id}"
        for page in (f"/runs/{run.id}", f"/runs/{run.id}/report.html", f"/runs/progress/{progress_id}"):
            html = app.handle("GET", page).body.decode()
            assert f'href="{url}"' in html
            assert f'/plans/{newest.id}' not in html
        payload = json.loads(app.handle("GET", f"/api/progress/{progress_id}").body)
        assert payload["steps"][0]["plan_url"] == url
        version_html = app.handle("GET", url).body.decode()
        assert "Saved TestPlan version v1" in version_html
        assert "Read-only" in version_html
        assert '<form' not in version_html
        assert app.handle("GET", f"/runs/{run.id}/plans/{newest.id}").status == 404
        assert app.handle("POST", url, "expected=Changed").status == 405
        assert plans.get_version(old.id).model_dump_json() == original
        assert plans.find(case.steps[0].id).id == newest.id
        cases.save(case.model_copy(update={"steps": [case.steps[0].model_copy(update={"name": "Renamed"})]}))
        assert app.handle("GET", url).status == 200
    finally:
        app.close()


def test_run_details_and_html_report_have_distinct_navigation():
    _, run, record = case_and_run()
    repository = InMemoryRunHistoryRepository()
    history = RunHistoryService(repository)
    repository.save(record)
    app = LocalWebApplication(history)
    try:
        detail = app.handle("GET", f"/runs/{run.id}").body.decode()
        html = app.handle("GET", f"/runs/{run.id}/report.html").body.decode()
        assert '<h1>Run Details</h1>' in detail
        assert 'View HTML Report' in detail
        assert '<h1>HTML Test Report</h1>' in html
        assert 'Back to Run Details' in html
        assert 'View HTML Report' not in html
        assert record.test_case_name in html
    finally:
        app.close()


@pytest.mark.parametrize("url", ["javascript:alert(1)", "//external.invalid/path", "/\n/external.invalid", "/\\external.invalid"])
def test_report_navigation_rejects_nonlocal_urls(url):
    _, _, record = case_and_run()
    report = RunReportGenerator().generate_history(record)
    html = RunReportGenerator().to_html(report, plan_url=lambda *_: url, run_details_url=url)
    assert f'href="{url}"' not in html
    assert "No saved TestPlan available." in html
    assert f'href="/runs/{record.run_id}"' in html
