"""Local storage, workflow, approval, background progress and browser UI integration."""
import threading
import time
from uuid import uuid4
from unittest.mock import patch

import pytest
from playwright.sync_api import expect, sync_playwright

from qa_agent.automation_lifecycle import AutomationLifecycleService, AutomationStatus
from qa_agent.background_execution import BackgroundRunService
from qa_agent.execution_progress import ExecutionProgressStore
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.models import DiscoveryResult, DiscoveryStatus, QATestPlan, TestCase as Case, TestStep as Step
from qa_agent.pipeline import PipelineStageError, QATestPipeline
from qa_agent.pinned_execution import PlanVersionSet, StepPlanSelection
from qa_agent.reliability import AutomationReliabilitySupervisor, AutomationReviewRequired, ReliabilitySettings
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_execution import RunUnavailableError, TestCaseExecutionService as CaseExecution
from qa_agent.test_case_review import TestCaseReviewService as CaseReview
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.web import LocalWebApplication, create_http_server
from qa_agent.workflows import AutomationWorkflow


class LocalPlanProvider(LLMProvider):
    model = "local-fixture"
    name = "Local fixture"

    def __init__(self):
        self.calls = 0

    def create_test_plan(self, task, target_url, page_snapshot):
        self.calls += 1
        if self.calls == 1:
            return {"url": target_url, "steps": []}
        return QATestPlan(url=target_url, steps=[{"action": "assert_visible", "parameters": {"selector": "#notice"}}])


def wired(tmp_path, *, legacy=False):
    storage = create_sqlite_storage(tmp_path / "reliability.sqlite3")
    step = Step(name="Verify notice", description="Verify the notice is visible.", expected="The notice is visible.", order=0)
    case = Case(name="Notice", description="Verify the notice is visible.", base_url="http://127.0.0.1:9876/target", steps=[step])
    storage.test_case_repository.save(case)
    review = CaseReview(storage.test_case_review_repository, storage.plan_store)
    if not legacy:
        review.approve_test_case(case)
    lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
    supervisor = AutomationReliabilitySupervisor(storage.reliability_repository)
    storage.reliability_repository.save_settings(ReliabilitySettings(automatic_plan_repair=True))
    provider = LocalPlanProvider()
    runner_calls = []
    def runner(plan):
        runner_calls.append(plan)
        return {"status": "passed", "steps": [{"action": "assert_visible", "status": "passed"}], "evidence": []}
    pipeline = QATestPipeline(
        decomposer=None, plan_generator=LLMTestPlanGenerator(LLMRouter([provider]), supervisor),
        discovery=lambda url: DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=url, snapshot={"visible_text_elements": [{"selector": "#notice", "text": "Notice", "tag": "p"}]}),
        runner=runner, plan_store=storage.plan_store, execution_repository=storage.execution_repository, run_history=storage.run_history,
        candidate_review_approved=lambda current: review.record(current.id) is not None and review.validation_approved_for(current),
    )
    service = CaseExecution(
        storage.test_case_repository, storage.plan_store, storage.execution_repository, storage.run_history,
        runner_factory=lambda directory: runner, automation_workflow=AutomationWorkflow(pipeline),
        automation_lifecycle=lifecycle, test_case_review=review,
        candidate_review_required=lambda version_id: supervisor.requires_review(version_id) is not None,
    )
    return storage, case, review, lifecycle, supervisor, provider, runner_calls, service


@pytest.mark.parametrize("legacy", [False, True])
def test_repaired_candidate_stops_execution_and_requires_explicit_approval_after_restart(tmp_path, legacy):
    storage, case, review, lifecycle, supervisor, provider, runner_calls, service = wired(tmp_path, legacy=legacy)
    before_review = review.record(case.id)
    with pytest.raises(PipelineStageError) as stopped:
        service.run(case.id, WorkflowType.AUTOMATION)
    assert isinstance(stopped.value.__cause__, AutomationReviewRequired)
    version = storage.plan_store.find(case.steps[0].id)
    frozen = version.model_dump_json()
    assert version.origin.value == "REPAIRED"
    assert not runner_calls and not storage.execution_repository.list_for_test_step(case.steps[0].id)
    assert review.record(case.id) == before_review
    assert lifecycle.status(case) == AutomationStatus.NEEDS_VALIDATION
    restarted = AutomationReliabilitySupervisor(create_sqlite_storage(storage.database_path).reliability_repository)
    service._candidate_review_required = lambda version_id: restarted.requires_review(version_id) is not None
    pins = PlanVersionSet(selections=(StepPlanSelection(case.steps[0].id, version.id),))
    assert service.pinned_regression_approval_error(case, pins)
    for workflow in (WorkflowType.VALIDATION, WorkflowType.REGRESSION):
        with pytest.raises(RunUnavailableError):
            service.run(case.id, workflow)
    with pytest.raises(PipelineStageError):
        service.run(case.id, WorkflowType.AUTOMATION)
    assert provider.calls == 2 and not runner_calls
    assert supervisor.statistics()["generation_operations"] == 1
    review.approve_for_validation(case)
    assert service.pinned_regression_approval_error(case, pins) is None
    with patch.object(supervisor, "generate", side_effect=AssertionError("Saved-plan execution must not invoke generation")):
        validation = service.run(case.id, WorkflowType.VALIDATION)
        regression = service.run_pinned_regression(case, pins)
    assert validation.test_run.executions[0].test_plan_version_id == version.id
    assert regression.test_run.executions[0].test_plan_version_id == version.id
    assert lifecycle.status(case) == AutomationStatus.AUTOMATION_READY
    assert storage.plan_store.get_version(version.id).model_dump_json() == frozen
    assert provider.calls == 2 and supervisor.statistics()["generation_operations"] == 1
    newer = version.model_copy(update={"id": uuid4(), "version": 2})
    storage.plan_store.save(case.steps[0].id, newer, test_plan=storage.plan_store.find_test_plan(case.steps[0].id))
    historical = storage.run_history.get(validation.test_run.id)
    assert historical.executions[0].test_plan_version_id == version.id
    assert storage.plan_store.get_version(version.id).model_dump_json() == frozen
    assert not review.validation_approved_for(case)


def test_background_review_result_is_blocked_with_saved_candidate_and_truthful_progress(tmp_path):
    storage, case, review, lifecycle, supervisor, provider, runner_calls, service = wired(tmp_path)
    store = ExecutionProgressStore()
    background = BackgroundRunService(service, storage.run_history, store)
    try:
        progress_id = background.start(case.id, WorkflowType.AUTOMATION)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not store.get(progress_id).to_public_dict()["finished"]:
            time.sleep(0.01)
        payload = store.get(progress_id).to_public_dict()
        assert payload["finished"]
        assert payload["summary"]["outcome"] == "BLOCKED"
        assert payload["error_category"] == "AUTOMATION_REVIEW_REQUIRED"
        assert "review" in payload["phase"].lower()
        assert not runner_calls and storage.plan_store.find(case.steps[0].id)
        assert not storage.run_history.list_for_test_case(case.id)
    finally:
        background.close()


def test_settings_browser_ui_uses_local_server_and_persists_controls(tmp_path):
    storage = create_sqlite_storage(tmp_path / "browser.sqlite3")
    supervisor = AutomationReliabilitySupervisor(storage.reliability_repository)
    app = LocalWebApplication(storage.run_history, reliability=supervisor)
    server = create_http_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{server.server_address[1]}/settings/reliability")
                expect(page.get_by_role("heading", name="Automation Reliability", exact=True)).to_be_visible()
                page.locator('[name="additional_retries"]').select_option("on")
                page.locator('[name="provider_fallback"]').select_option("off")
                page.locator('[name="automatic_plan_repair"]').select_option("on")
                page.locator('[name="max_total_attempts"]').select_option("3")
                page.get_by_role("button", name="Save reliability settings").click()
                expect(page.get_by_text("Settings saved. New operations use these values.")).to_be_visible()
                assert storage.reliability_repository.settings() == ReliabilitySettings(additional_retries=True, provider_fallback=False, automatic_plan_repair=True, max_total_attempts=3)
                expect(page.get_by_text("First-attempt success", exact=True)).to_be_visible()
                assert supervisor.statistics()["generation_operations"] == 0
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        app.close()
