"""Local SQLite, HTTP and Chromium verification of persisted reporting UX."""

import base64
import json
import threading
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from playwright.sync_api import expect, sync_playwright

from qa_agent.execution_progress import ExecutionEventType as Event, ExecutionProgressReporter, ExecutionProgressStore
from qa_agent.execution_trace import ProviderAttemptOutcome, ProviderAttemptTrace, RequestKind, StepTrace
from qa_agent.models import (
    Evidence, EvidenceScope, EvidenceType, Execution, ExecutionStatus, PlanVersionOrigin,
    QATestPlan, QATestStep, TestCase as DomainCase, TestPlan as DomainPlan,
    TestPlanVersion as DomainVersion, TestRun as DomainRun, TestStep as DomainStep,
)
from qa_agent.storage import create_sqlite_storage
from qa_agent.web import LocalWebApplication, create_http_server
from qa_agent.run_history import WorkflowType
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.suite_runs import SuiteRun, SuiteRunAttempt, SuiteRunConfig, SuiteRunItem, SuiteRunItemStatus, SuiteRunStatus


def test_local_reports_progress_exact_versions_and_evidence_survive_reload(tmp_path):
    storage = create_sqlite_storage(tmp_path / "reporting.sqlite3")
    case = DomainCase(name="Account confirmation", description="Confirm the account.", steps=[
        DomainStep(name="Confirm account", description="Check status.", expected="Account confirmed.", order=0),
    ])
    storage.test_case_repository.save(case)
    plan = DomainPlan(test_step_id=case.steps[0].id, name="Account status")
    versions = []
    for number in range(1, 4):
        version = DomainVersion(
            test_plan_id=plan.id, version=number, origin=PlanVersionOrigin.AI_GENERATED,
            previous_version_id=versions[-1].id if versions else None,
            qa_test_plan=QATestPlan(url="http://127.0.0.1/target", steps=[QATestStep(action="assert_title", parameters={"expected": "Confirmed"})]),
        )
        storage.plan_store.save(case.steps[0].id, version, test_plan=plan)
        versions.append(version)
    now = datetime.now(timezone.utc)
    attempts = []
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=")
    for number, version in enumerate(versions[:2], start=1):
        path = tmp_path / f"attempt-{number}.png"
        path.write_bytes(png)
        execution = Execution(
            test_step_id=case.steps[0].id, test_plan_version_id=version.id,
            started_at=now, finished_at=now, status=ExecutionStatus.FAILED, actual_result="failed",
            error="Expected confirmation could not be verified.\nActual value: Ready\nCall log:\n - waiting",
            runner_result={"qa_classification": "AUTOMATION_EXECUTION_ERROR"},
        )
        execution.evidence = (Evidence(
            execution_id=execution.id, type=EvidenceType.SCREENSHOT, path=str(path),
            description=f"Account status at attempt {number}.", scope=EvidenceScope.PAGE,
            event=f"FAILURE:assert_title:{number}",
        ),)
        execution.runner_result["evidence"] = [{
            "evidence_id": str(execution.evidence[0].id), "type": "SCREENSHOT",
            "path": str(path), "description": execution.evidence[0].description,
            "scope": "PAGE", "event": execution.evidence[0].event,
        }]
        storage.execution_repository.save(execution)
        attempts.append(execution)
    run = DomainRun.from_test_case(case, attempts)
    record = storage.run_history.record_completed_run(case, run, workflow_type=WorkflowType.REGRESSION, outcome="AUTOMATION_EXECUTION_ERROR")
    original_record = record.model_dump_json()
    store = ExecutionProgressStore()
    progress_id = store.create(case.id, WorkflowType.REGRESSION)
    reporter = ExecutionProgressReporter(store, progress_id)
    reporter.test_case_loaded(case)
    for version in versions[:2]:
        reporter.emit(Event.PLAN_REUSED, step=case.steps[0], plan_version_id=version.id, plan_version=version.version, plan_origin="AI_GENERATED")
        reporter.emit(Event.STEP_STARTED, step=case.steps[0])
        reporter.emit(Event.STEP_FAILED, step=case.steps[0], classification="AUTOMATION_EXECUTION_ERROR")
    reporter.capture_provider_diagnostics(SimpleNamespace(steps=[StepTrace(
        test_step_id=case.steps[0].id, order=0, name="Confirm account", description="Check status.", expected="Account confirmed.",
        provider_attempts=[ProviderAttemptTrace(
            provider_name="Recorded provider", model="recorded-model", request_kind=RequestKind.TEST_PLAN,
            outcome=ProviderAttemptOutcome.SUCCESS, duration_ms=1200,
        )],
    )]))
    reporter.finish(run_id=run.id, run_status="FAILED", outcome="AUTOMATION_EXECUTION_ERROR")
    reloaded = create_sqlite_storage(tmp_path / "reporting.sqlite3")
    app = LocalWebApplication(
        reloaded.run_history, test_cases=reloaded.test_case_repository, plan_store=reloaded.plan_store,
        evidence_root=tmp_path, progress_store=store,
    )
    server = create_http_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"{origin}/runs/progress/{progress_id}")
            expect(page.locator('[data-progress-phase]')).to_have_text("Stopped — Automation Error")
            expect(page.locator('[data-progress-result-content]')).to_contain_text("Stopped at Step 1 of 1")
            expect(page.locator('[data-progress-result-content]')).to_contain_text("Observed: Ready")
            expect(page.locator('[data-progress-result-content]')).to_contain_text("View Run Details")
            expect(page.locator('[data-progress-diagnostics]')).to_contain_text("Recorded provider attempts")
            expect(page.locator('[data-progress-diagnostics]')).to_contain_text("recorded-model")
            expect(page.locator('[data-progress-diagnostics]')).to_contain_text("1.2 s")
            page.locator('[data-progress-developer-details] > summary').click()
            provider_section = page.locator('[data-progress-diagnostics] section').filter(has=page.get_by_role('heading', name='Recorded provider attempts', exact=True))
            expect(provider_section.locator('li p')).to_contain_text('recorded-model')
            assert page.locator(f'a[href="/runs/{run.id}/plans/{versions[1].id}"]').count() == 1
            assert not errors
            page.get_by_role('link', name='View Run Details', exact=True).click()
            expect(page.get_by_role('heading', name='Run Details', exact=True)).to_be_visible()
            expect(page.locator('.result-summary')).to_contain_text("Automation Error")
            assert page.locator('.badge', has_text='Failed').count() == 0
            for number, attempt in enumerate(attempts, start=1):
                section = page.locator(f'[data-execution-id="{attempt.id}"]')
                expect(section).to_contain_text(f'Attempt {number}')
                expect(section).to_contain_text('Actual observation: Ready')
                expect(section.locator('img')).to_have_attribute('src', f'/runs/{run.id}/evidence/{attempt.id}/0')
                assert section.locator(f'a[href="/runs/{run.id}/plans/{versions[number - 1].id}"]').count() == 1
            page.get_by_role('link', name='View HTML Report', exact=True).click()
            expect(page.get_by_role('heading', name='HTML Test Report', exact=True)).to_be_visible()
            expect(page.locator('.page-heading')).to_contain_text(record.public_id)
            page.get_by_role('link', name='Back to Run Details', exact=True).click()
            assert page.url == f'{origin}/runs/{run.id}'
            page.locator(f'a[href="/runs/{run.id}/plans/{versions[0].id}"]').click()
            expect(page.get_by_role('heading', name='Saved TestPlan version v1')).to_be_visible()
            assert page.locator('form').count() == 0
            assert reloaded.plan_store.find(case.steps[0].id).id == versions[2].id
            assert reloaded.run_history.get(run.id).model_dump_json() == original_record
            payload = json.loads(app.handle('GET', f'/runs/{run.id}/report.json').body)
            assert [item['plan_version_number'] for item in payload['steps'][0]['attempts']] == [1, 2]
            for attempt in attempts:
                assert app.handle('GET', f'/runs/{run.id}/evidence/{attempt.id}/0').body == png
            assert app.handle('GET', f'/runs/{run.id}/evidence/{attempts[0].id}/1').status == 404
            assert not errors
            browser.close()
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_suite_live_reporting_distinguishes_outcomes_and_preserves_retries():
    now = datetime.now(timezone.utc)
    items = []
    for index, outcome in enumerate(("PRODUCT_FAILURE", "AUTOMATION_EXECUTION_ERROR", "PASSED")):
        attempt = SuiteRunAttempt(
            attempt_number=1, run_id=uuid4(), run_public_id=f"RUN-{index + 1:06d}",
            run_status="PASSED" if outcome == "PASSED" else "FAILED", outcome=outcome,
            started_at=now, finished_at=now, duration_ms=0,
        )
        attempts = [attempt]
        if outcome == "PASSED":
            attempts = [attempt.model_copy(update={"run_id": uuid4(), "run_status": "FAILED", "outcome": "INFRASTRUCTURE_ERROR"}), attempt.model_copy(update={"attempt_number": 2})]
        items.append(SuiteRunItem(
            order_index=index, test_case_id=uuid4(), test_case_name=f"Case {index + 1}", attempts=attempts,
            status=SuiteRunItemStatus.PASSED_AFTER_RETRY if outcome == "PASSED" else SuiteRunItemStatus.FAILED,
        ))
    run = SuiteRun(public_id="SUITE-RUN-000001", suite_id=uuid4(), suite_name="Smoke", config=SuiteRunConfig(), status=SuiteRunStatus.COMPLETED_WITH_FAILURES, items=items)
    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository()),
        suite_run_service=SimpleNamespace(get=lambda _: run, close=lambda: None),
    )
    server = create_http_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/suite-runs/{run.public_id}")
            expect(page.locator('[data-suite-run-items] strong').first).to_have_text(re.compile(r"^1\..*Failed$"))
            expect(page.locator('[data-suite-run-summary]')).to_contain_text('1 product failures')
            expect(page.locator('[data-suite-run-summary]')).to_contain_text('1 automation errors')
            expect(page.locator('[data-suite-run-summary]')).to_contain_text('1 passed after retry')
            expect(page.locator('[data-suite-run-items] strong').nth(1)).to_contain_text('Automation Error')
            expect(page.locator('[data-suite-run-items] strong').nth(2)).to_contain_text('Passed after retry')
            for item in items:
                for attempt in item.attempts:
                    assert page.locator(f'a[href="/runs/{attempt.run_id}"]').count() == 1
            assert page.locator('img').count() == 0
            assert not errors
            browser.close()
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
