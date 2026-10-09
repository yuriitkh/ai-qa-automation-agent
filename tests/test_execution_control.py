"""Deterministic Stop, persistence and concurrency checks without network AI."""
import json
import time
import unittest
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier, Event, Thread
from urllib.parse import urlencode
from uuid import uuid4
from unittest.mock import patch

import pytest

from qa_agent.background_authoring import BackgroundAuthoringService
from qa_agent.background_execution import BackgroundRunService
from qa_agent.browser_runner import _run_plan_on_page
from qa_agent.execution_control import CancellationToken, OperationCancelled, cancellation_scope, completion_boundary, cancellable_call
from qa_agent.execution_preferences import SQLiteExecutionPreferences, PreferencesConflict
from qa_agent.execution_progress import ExecutionProgressStore
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.execution_trace import ExecutionTraceRecorder, TraceStatus
from qa_agent.models import ExecutionStatus, TestCase as Case, QATestPlan, RunContext
from qa_agent.pipeline import QATestPipeline
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.reporting import RunReportGenerator, TestReportGenerator as DomainReportGenerator
from qa_agent.reliability import ReliabilityOperation, ReliabilityStopped
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService, WorkflowType
from qa_agent.setup_orchestration import SetupCleanupCoordinator
from qa_agent.suite_runs import SuiteRunStatus, SuiteRunItemStatus
from qa_agent.suite_run_storage import SQLiteSuiteRunRepository
from qa_agent.test_case_authoring import TestCaseAuthoringService as AuthoringService, TestCaseDraftStore as DraftStore
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.web import LocalWebApplication
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_execution import TestCaseExecutionService as ExecutionService
from qa_agent.workflows import RegressionWorkflow, ValidationWorkflow

from test_background_authoring import AuthoringProvider
from test_automation_reliability import configured, generate, discovery, requirement, valid_plan, LocalProvider
from test_suite_runs import SuiteRunTestHarness, _case, _plan


def eventually(operation, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = operation()
        if predicate(value):
            return value
        time.sleep(0.005)
    raise AssertionError("Local operation did not reach its expected state.")


def authoring_job(service, name="Check title"):
    return service.start(name=name, base_url="http://127.0.0.1:1/local", scenario="Check the page title.")


def test_authoring_stop_discards_late_response_and_never_starts_fallback():
    entered, release = Event(), Event()
    first = AuthoringProvider(entered=entered, release=release, error=RetryableLLMError("timeout", category="TIMEOUT"))
    fallback = AuthoringProvider()
    progress, drafts = ExecutionProgressStore(), DraftStore()
    service = BackgroundAuthoringService(AuthoringService(LLMRouter([first, fallback])), drafts, progress)
    try:
        job = authoring_job(service)
        assert entered.wait(2)
        assert service.cancel(job)
        snapshot = eventually(lambda: progress.get_authoring(job), lambda value: value.finished)
        assert snapshot.state.value == "CANCELLED" and snapshot.success is None
        assert snapshot.review_url is None and not drafts._drafts
        assert not service.cancel(job)
        release.set()
        eventually(lambda: service._controls, lambda value: not value)
        assert progress.get_authoring(job) == snapshot
        assert fallback.calls == 0 and not drafts._drafts
    finally:
        release.set()
        service.close()


def test_queued_stop_targets_only_requested_job_and_repeated_stop_is_safe():
    entered, release = Event(), Event()
    provider = AuthoringProvider(entered=entered, release=release)
    progress, drafts = ExecutionProgressStore(), DraftStore()
    service = BackgroundAuthoringService(AuthoringService(LLMRouter([provider])), drafts, progress, max_workers=1)
    try:
        first = authoring_job(service, "First")
        assert entered.wait(2)
        queued = authoring_job(service, "Queued")
        assert service.cancel(queued) and service.cancel(queued)
        assert progress.get_authoring(queued).state.value == "CANCELLATION_REQUESTED"
        assert progress.get_authoring(first).state.value == "RUNNING"
        release.set()
        done = eventually(lambda: progress.get_authoring(queued), lambda value: value.finished)
        assert done.state.value == "CANCELLED" and provider.calls == 1
        assert progress.get_authoring(first).success is True
        assert len(drafts._drafts) == 1
        assert not service.cancel(first) and not service.cancel("wrong-id")
    finally:
        release.set()
        service.close()


def test_cancelled_sdk_workers_retain_bounded_slots_until_actual_return():
    releases = Event()
    entered = [Event() for _ in range(4)]
    tokens = [CancellationToken() for _ in range(5)]
    failures = []
    def run(index):
        def sdk():
            entered[index].set()
            releases.wait(3)
        try:
            with cancellation_scope(tokens[index]):
                cancellable_call(sdk)
        except OperationCancelled:
            failures.append(index)
    workers = [Thread(target=run,args=(index,)) for index in range(4)]
    try:
        for worker in workers: worker.start()
        assert all(event.wait(2) for event in entered)
        for token in tokens[:4]: token.request()
        for worker in workers: worker.join(2)
        assert sorted(failures) == [0,1,2,3]
        fifth_call = Event()
        def fifth():
            try:
                with cancellation_scope(tokens[4]):
                    cancellable_call(lambda: fifth_call.set())
            except OperationCancelled:
                failures.append(4)
        waiting = Thread(target=fifth); waiting.start()
        tokens[4].request(); waiting.join(2)
        assert not fifth_call.is_set() and 4 in failures
    finally:
        releases.set()
        for worker in workers: worker.join(2)


@pytest.mark.parametrize('cancel_wins', [True, False])
def test_background_run_commit_race_has_one_durable_result(cancel_wins):
    action_entered, release_action, commit_entered, release_commit = Event(), Event(), Event(), Event()
    class Repository(InMemoryRunHistoryRepository):
        def save(self, record):
            commit_entered.set(); release_commit.wait(3)
            return super().save(record)
    case = _case(); plans, executions = InMemoryPlanStore(), InMemoryExecutionRepository()
    plan, version = _plan(case); plans.save(case.steps[0].id, version, test_plan=plan)
    history = RunHistoryService(Repository(), executions, plan_store=plans)
    def runner(_):
        action_entered.set()
        if cancel_wins: release_action.wait(3)
        return {'status':'passed','steps':[]}
    workflow = RegressionWorkflow(PinnedExecutionService(plans, PlanExecutionService(runner, executions)), SetupCleanupCoordinator({}), history)
    class Service:
        def run(self, *_): return workflow.run(case, PlanVersionSet.from_mapping({case.steps[0].id: version.id}))
    background = BackgroundRunService(Service(), history)
    responses = []
    try:
        job = background.start(case.id, WorkflowType.REGRESSION)
        assert action_entered.wait(2)
        if cancel_wins:
            assert background.cancel(job)
            release_action.set()
        assert commit_entered.wait(2)
        stopper = Thread(target=lambda: responses.append(background.cancel(job)))
        stopper.start(); release_commit.set(); stopper.join(2)
        done = eventually(lambda: background.progress_store.get(job), lambda snapshot: snapshot.finished_at is not None)
        record = history.get(done.final_run_id)
        assert record.outcome == done.outcome == ('CANCELLED' if cancel_wins else 'PASSED')
        assert responses == [False]
        assert not background.cancel(job) and record.executions[0].test_plan_version_id == version.id
    finally:
        release_action.set(); release_commit.set(); background.close()


@pytest.mark.parametrize("boundary", ["RELIABILITY_RETRY", "RELIABILITY_FALLBACK", "RELIABILITY_REPAIR", "RELIABILITY_CHECKING"])
def test_stop_at_supervisor_boundary_blocks_new_attempts_and_success_statistics(boundary):
    initial = {"url": valid_plan().url, "steps": []} if boundary == "RELIABILITY_REPAIR" else valid_plan() if boundary == "RELIABILITY_CHECKING" else RetryableLLMError("timeout", category="TIMEOUT")
    first, fallback = LocalProvider([initial, valid_plan()]), LocalProvider([valid_plan()])
    generator, supervisor = configured(first, fallback, additional_retries=boundary == "RELIABILITY_RETRY", automatic_plan_repair=True, max_total_attempts=3)
    token = CancellationToken()
    original = ReliabilityOperation.emit
    def emit(operation, event, message):
        original(operation, event, message)
        if event == boundary:
            token.request()
    with cancellation_scope(token), patch.object(ReliabilityOperation, "emit", emit), pytest.raises(ReliabilityStopped):
        generate(generator)
    record = supervisor.repository.list_records()[0]
    assert record.outcome == "CANCELLED" and record.final_category == "CANCELLED"
    assert len(first.calls) == 1 and not fallback.calls
    assert record.candidate_version_id is None
    stats = supervisor.statistics()
    assert stats["first_attempt_success_rate"] == 0
    assert stats["recovery_success_rate"] in {None, 0}


def test_automation_stop_during_provider_wait_persists_truthful_partial_run():
    entered, release = Event(), Event()
    def delayed():
        entered.set()
        release.wait(3)
        return valid_plan()
    provider = LocalProvider([delayed])
    generator, supervisor = configured(provider)
    case = Case(name="Check notice", description="Verify the notice is visible.", base_url=valid_plan().url, steps=[requirement()])
    plans, executions = InMemoryPlanStore(), InMemoryExecutionRepository()
    history = RunHistoryService(InMemoryRunHistoryRepository(), executions, plan_store=plans)
    pipeline = QATestPipeline(object(), generator, discovery=lambda _: discovery(), plan_store=plans, execution_repository=executions, run_history=history, runner=lambda _: pytest.fail("Cancelled generation must never start execution"))
    class RunService:
        def run(self, *_):
            return pipeline.run_test_case(case)
    service = BackgroundRunService(RunService(), history)
    try:
        job = service.start(case.id, WorkflowType.AUTOMATION)
        assert entered.wait(2)
        assert service.cancel(job)
        snapshot = eventually(lambda: service.progress_store.get(job), lambda value: value.finished_at is not None)
        assert snapshot.outcome == "CANCELLED" and snapshot.final_run_id is not None
        record = history.get(snapshot.final_run_id)
        assert record.status == ExecutionStatus.CANCELLED and record.outcome == "CANCELLED"
        assert not record.executions and record.steps[0].status == ExecutionStatus.NOT_ATTEMPTED
        assert plans.find(case.steps[0].id) is None
        release.set()
        assert supervisor.repository.list_records()[0].outcome == "CANCELLED"
        assert history.get(record.run_id) == record
    finally:
        release.set()
        service.close()


@pytest.mark.parametrize("workflow", [RegressionWorkflow, ValidationWorkflow])
def test_stop_between_steps_preserves_product_failure_evidence_pins_and_reports(tmp_path, workflow):
    # This artifact represents evidence captured by the first runner attempt.
    evidence = tmp_path / "existing-evidence.png"
    evidence.write_bytes(b"retained evidence")
    token = CancellationToken()
    first, pending = _case().steps[0], _case("Pending").steps[0].model_copy(update={"order": 1})
    case = Case(name="Partial title check", description="Verify the page title is Welcome.", base_url=valid_plan().url, steps=[first, pending])
    plans, executions = InMemoryPlanStore(), InMemoryExecutionRepository()
    pins = {}
    for step in case.steps:
        owner = Case(name="Title check", description="Verify the page title is Welcome.", steps=[step])
        plan, version = _plan(owner)
        plans.save(step.id, version, test_plan=plan)
        pins[step.id] = version.id
    history = RunHistoryService(InMemoryRunHistoryRepository(), executions, plan_store=plans)
    def runner(_):
        # An assertion has already completed before Stop. Its finding survives.
        token.request()
        return {"status": "failed", "steps": [{"action": "assert_title", "status": "failed", "error": "Unexpected title"}], "evidence": [{"type": "SCREENSHOT", "path": str(evidence)}]}
    executor = PinnedExecutionService(plans, PlanExecutionService(runner, executions))
    with cancellation_scope(token):
        result = workflow(executor, SetupCleanupCoordinator({}), history).run(case, PlanVersionSet.from_mapping(pins))
    assert result.outcome.value == "CANCELLED" and result.test_run.cancelled
    assert len(result.test_run.executions) == 1
    completed = result.test_run.executions[0]
    assert completed.status == ExecutionStatus.FAILED
    assert completed.runner_result["qa_classification"] == "PRODUCT_FAILURE"
    assert result.test_run.not_attempted_step_ids == [pending.id]
    assert completed.evidence and evidence.read_bytes() == b"retained evidence"
    assert {step.id: plans.find(step.id).id for step in case.steps} == pins
    report = RunReportGenerator().generate_history(history.get_detail(result.test_run.id))
    assert report.display_outcome == "CANCELLED"
    assert report.steps[0].classification == "PRODUCT_FAILURE"
    assert report.steps[1].classification == "NOT_ATTEMPTED"
    assert 'Stopped by user' in RunReportGenerator().to_html(report)
    assert json.loads(report.to_json())["outcome"] == "CANCELLED"
    assert DomainReportGenerator().generate(result.test_run).display_outcome == "CANCELLED"


@pytest.mark.parametrize("runner_status", ["passed", "failed"])
@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
def test_cleanup_completion_keeps_trace_identity_timestamp_and_product_findings(
    tmp_path, runner_status, cancel_during_cleanup,
):
    case = _case()
    plans, executions = InMemoryPlanStore(), InMemoryExecutionRepository()
    plan, version = _plan(case)
    plans.save(case.steps[0].id, version, test_plan=plan)
    original_plan = version.model_dump_json()
    history = RunHistoryService(InMemoryRunHistoryRepository(), executions, plan_store=plans)
    token, cleanup_entered, release_cleanup = CancellationToken(), Event(), Event()
    snapshots = []
    evidence = tmp_path / "completed-assertion.png"
    evidence.write_bytes(b"original evidence")

    class Recorder(ExecutionTraceRecorder):
        def finalize(self, *args, **kwargs):
            snapshot = super().finalize(*args, **kwargs)
            snapshots.append(snapshot)
            return snapshot

    class Pipeline(QATestPipeline):
        def _create_trace_recorder(self, task):
            return Recorder(task)

    class Runner:
        @contextmanager
        def open_test_case_session(self, _):
            try:
                yield self
            finally:
                cleanup_entered.set()
                assert release_cleanup.wait(5)

        def run_plan(self, *_):
            return {
                "status": runner_status,
                "steps": [{"action": "assert_title", "status": runner_status,
                           "error": "Unexpected title" if runner_status == "failed" else ""}],
                "evidence": [{"type": "SCREENSHOT", "path": str(evidence)}],
            }

        def __call__(self, _):
            pytest.fail("The bound session must execute the plan")

    pipeline = Pipeline(object(), object(), plan_store=plans,
                        execution_repository=executions, run_history=history, runner=Runner())

    def execute():
        with cancellation_scope(token):
            return pipeline.run_test_case(case)

    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(execute)
        try:
            assert cleanup_entered.wait(3)
            assert len(snapshots) == 1
            original_trace = snapshots[0]
            assert original_trace.status == (TraceStatus.PASSED if runner_status == "passed" else TraceStatus.FAILED)
            assert history.list_recent() == []
            final_timestamp = original_trace.finished_at + timedelta(seconds=2)
            with patch("qa_agent.pipeline.datetime") as clock:
                clock.now.return_value = final_timestamp
                if cancel_during_cleanup:
                    assert token.request()
                assert not future.done()
                release_cleanup.set()
                result = future.result(timeout=3)
        finally:
            release_cleanup.set()

    record = history.get(result.test_run.id)
    assert record.trace_id == result.trace.trace_id == original_trace.trace_id
    assert result.trace.started_at == original_trace.started_at
    assert result.trace.steps == original_trace.steps
    assert result.trace.totals == original_trace.totals
    assert result.trace.error == original_trace.error
    if cancel_during_cleanup:
        assert result.test_run.status == ExecutionStatus.CANCELLED
        assert record.outcome == "CANCELLED" and result.trace.status == TraceStatus.ERROR
        assert result.trace.finished_at == result.test_run.finished_at == record.finished_at == final_timestamp
        assert result.trace.duration_ms == int((final_timestamp - original_trace.started_at).total_seconds() * 1000)
    else:
        assert result.trace == original_trace
        assert record.outcome == ("PASSED" if runner_status == "passed" else "PRODUCT_FAILURE")
    finding = result.test_run.executions[0]
    assert finding.status.value.lower() == runner_status
    assert finding.id == original_trace.steps[0].execution_attempts[0].execution_id
    assert record.executions[0].classification == ("PASSED" if runner_status == "passed" else "PRODUCT_FAILURE")
    assert list(finding.evidence) == original_trace.steps[0].execution_attempts[0].evidence
    assert evidence.read_bytes() == b"original evidence"
    assert plans.find(case.steps[0].id).model_dump_json() == original_plan
    assert not token.request()


@pytest.mark.parametrize('automation', [True, False])
def test_cancelled_browser_cleanup_error_keeps_partial_history_and_safe_diagnostics(automation):
    from contextlib import contextmanager
    case = _case(); plans, executions = InMemoryPlanStore(), InMemoryExecutionRepository()
    plan, version = _plan(case); plans.save(case.steps[0].id, version, test_plan=plan)
    history = RunHistoryService(InMemoryRunHistoryRepository(), executions, plan_store=plans)
    token = CancellationToken()
    class Runner:
        @contextmanager
        def open_test_case_session(self, _):
            try:
                yield self
            finally:
                raise RuntimeError('private-cleanup-detail')
        def run_plan(self, *_):
            token.request()
            return {'status':'failed','steps':[{'action':'assert_title','status':'failed','error':'Title differs'}]}
        def __call__(self, _):
            pytest.fail('The bound session must be used')
    runner = Runner()
    with cancellation_scope(token):
        if automation:
            result = QATestPipeline(object(), object(), plan_store=plans, execution_repository=executions, run_history=history, runner=runner).run_test_case(case)
        else:
            result = RegressionWorkflow(PinnedExecutionService(plans, PlanExecutionService(runner, executions)), SetupCleanupCoordinator({}), history).run(case, PlanVersionSet.from_mapping({case.steps[0].id:version.id}))
    record = history.get(result.test_run.id)
    assert record.outcome == 'CANCELLED' and len(record.executions) == 1
    assert record.executions[0].classification == 'PRODUCT_FAILURE'
    assert record.cleanup_succeeded is False and record.cleanup_failures
    assert 'private-cleanup-detail' not in record.model_dump_json()
    assert plans.find(case.steps[0].id) == version


def test_browser_action_stop_retains_completed_action_and_does_not_start_next():
    token = CancellationToken()
    class Page:
        url = "about:blank"
        def __init__(self): self.calls = 0
        def wait_for_load_state(self, *args, **kwargs):
            self.calls += 1
            token.request()
        def title(self):
            self.calls += 1
            token.request()
            return "Welcome"
    page = Page()
    plan = QATestPlan(url=valid_plan().url, steps=[{"action": "assert_page_loaded"}, {"action": "assert_page_loaded"}])
    # assert_page_loaded reads page.title and page.url deterministically.
    with cancellation_scope(token):
        result = _run_plan_on_page(plan, page)
    assert result["status"] == "cancelled" and page.calls == 1
    assert len(result["steps"]) == 1 and result["steps"][0]["status"] == "passed"


@pytest.mark.parametrize("cancel_first", [True, False])
def test_completion_boundary_resolves_stop_race_without_late_reclassification(cancel_first):
    token, entered, release = CancellationToken(), Event(), Event()
    committed = []
    def finish():
        with cancellation_scope(token), completion_boundary():
            entered.set()
            release.wait(2)
            committed.append("CANCELLED" if token.requested else "PASSED")
            token.sealed = True
    if cancel_first:
        assert token.request()
    worker = Thread(target=finish)
    worker.start()
    assert entered.wait(1)
    requester = Thread(target=lambda: committed.append(token.request()))
    requester.start()
    release.set()
    worker.join(2); requester.join(2)
    assert committed == (["CANCELLED", False] if cancel_first else ["PASSED", False])


@pytest.mark.parametrize("field,value", [("cookie_policy", "LEAVE_UNCHANGED"), ("evidence_mode", "EVERY_STEP"), ("screenshot_mode", "ELEMENT_AND_PAGE")])
def test_preferences_persist_immediately_across_repositories_and_preserve_other_fields(tmp_path, field, value):
    database, case_id = tmp_path / "prefs.sqlite3", uuid4()
    first = SQLiteExecutionPreferences(database)
    original = first.get(case_id)
    saved = first.update(case_id, field, value, original.revision)
    assert saved.model_dump(mode="json")[field] == value
    assert SQLiteExecutionPreferences(database).get(case_id) == saved
    for other in {"cookie_policy", "evidence_mode", "screenshot_mode"} - {field}:
        assert saved.model_dump()[other] == original.model_dump()[other]


def test_concurrent_preferences_updates_reject_stale_revision(tmp_path):
    database, case_id = tmp_path / "prefs.sqlite3", uuid4()
    repos = [SQLiteExecutionPreferences(database), SQLiteExecutionPreferences(database)]
    barrier = Barrier(2)
    def update(index):
        barrier.wait()
        try:
            return repos[index].update(case_id, "evidence_mode", ["EVERY_STEP", "EVERY_VERIFICATION"][index], 0)
        except PreferencesConflict as conflict:
            return conflict
    with ThreadPoolExecutor(2) as workers:
        values = list(workers.map(update, range(2)))
    assert sum(isinstance(value, PreferencesConflict) for value in values) == 1
    winner = next(value for value in values if not isinstance(value, PreferencesConflict))
    assert repos[0].get(case_id) == winner
    with pytest.raises(PreferencesConflict):
        repos[1].update(case_id, "evidence_mode", "FAILURES_ONLY", 0)
    assert repos[0].get(case_id) == winner


@pytest.mark.parametrize("body,expected", [
    ({"field": "cookie_policy", "value": "LEAVE_UNCHANGED", "revision": "0"}, 200),
    ({"field": "evidence_mode", "value": "invalid", "revision": "0"}, 400),
    ({"field": "cookie_policy", "value": "LEAVE_UNCHANGED", "revision": "-1"}, 400),
    ({"field": "approved", "value": "true", "revision": "0"}, 400),
])
def test_preferences_endpoint_validates_and_never_runs_ai_or_browser(body, expected):
    cases = InMemoryTestCaseRepository(); case = _case(); cases.save(case)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), test_cases=cases)
    before = cases.get(case.id).model_dump_json()
    response = app.handle("POST", f"/test-cases/{case.id}/preferences", urlencode(body), headers={"X-QA-CSRF": app._csrf_token})
    assert response.status == expected
    assert cases.get(case.id).model_dump_json() == before
    assert not app._draft_store._drafts and app._run_history.list_recent() == []
    assert app._background_runs is None and app._background_authoring is None


def test_sqlite_cancelled_history_round_trip_and_validation_keeps_lifecycle(tmp_path):
    from unittest.mock import MagicMock
    storage = create_sqlite_storage(tmp_path / 'application.sqlite3')
    case = _case(); storage.test_case_repository.save(case)
    plan, version = _plan(case); storage.plan_store.save(case.steps[0].id, version, test_plan=plan)
    token = CancellationToken()
    def runner(_):
        token.request()
        return {'status':'cancelled','steps':[]}
    lifecycle = MagicMock()
    service = ExecutionService(storage.test_case_repository, storage.plan_store, storage.execution_repository, storage.run_history, runner_factory=lambda _: runner, automation_lifecycle=lifecycle)
    with cancellation_scope(token):
        result = service.run(case.id, WorkflowType.VALIDATION)
    lifecycle.mark_validation_completed.assert_not_called()
    lifecycle.mark_automation_failed.assert_not_called()
    reopened = create_sqlite_storage(storage.database_path)
    record = reopened.run_history.get(result.test_run.id)
    assert record.status == ExecutionStatus.CANCELLED and record.outcome == 'CANCELLED'
    detail = reopened.run_history.get_detail(record.run_id)
    execution = detail.executions[record.executions[0].execution_id]
    assert execution.status == ExecutionStatus.CANCELLED
    assert execution.runner_result['qa_classification'] == 'CANCELLED'
    assert execution.test_plan_version_id == version.id
    assert reopened.plan_store.find(case.steps[0].id) == version
    report = RunReportGenerator().generate_history(detail)
    assert report.display_status == 'Stopped by user' and record.finished_at is not None
    assert json.loads(report.to_json())['status'] == 'CANCELLED'


@pytest.mark.parametrize("suffix", ["preferences", "stop"])
def test_new_mutations_reject_csrf_and_cross_origin_and_get_is_read_only(suffix):
    cases = InMemoryTestCaseRepository(); case = _case(); cases.save(case)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), test_cases=cases)
    route = f"/test-cases/{case.id}/preferences" if suffix == "preferences" else "/runs/progress/unknown/stop"
    assert app.handle("POST", route, "").status == 403
    assert app.handle("POST", route, "", headers={"X-QA-CSRF": app._csrf_token, "Origin": "http://other.invalid", "Host": "127.0.0.1:8000"}).status == 403
    assert app.handle("POST", route, "", headers={"X-QA-CSRF": app._csrf_token, "Sec-Fetch-Site": "cross-site"}).status == 403
    assert app.handle("GET", route).status != 200
    assert app._execution_preferences.get(case.id).revision == 0


class SuiteCancellationTests(SuiteRunTestHarness, unittest.TestCase):
    def test_failed_suite_submission_cannot_be_changed_to_cancelled(self):
        service = self.make_service(lambda _: pytest.fail('A failed submission cannot execute'))
        try:
            with patch.object(service._executor, 'submit', side_effect=RuntimeError('Local queue unavailable')):
                failed = service.start(self.suite.id).run
            assert failed.status == SuiteRunStatus.FAILED
            assert not service.cancel(failed.public_id)
            assert service.get(failed.public_id) == failed
            assert service._controls == {} and service.readiness_state()['active'] == 0
        finally:
            service.close()

    def test_suite_stop_propagates_preserves_defect_child_pins_and_pending_members(self):
        for name in ["Second", "Third"]:
            case = _case(name); self.cases.save(case)
            plan, version = _plan(case); self.plans.save(case.steps[0].id, version, test_plan=plan)
            self.suites.add_member(self.suite.id, case.id)
        entered, release = Event(), Event()
        calls = []
        def runner(plan):
            calls.append(plan)
            if len(calls) == 1:
                return {"status": "failed", "steps": [{"action": "assert_title", "status": "failed", "error": "Title differs"}]}
            entered.set(); release.wait(3)
            return {"status": "passed", "steps": []}
        service = self.make_service(runner)
        try:
            started = service.start(self.suite.id).run
            assert entered.wait(2)
            assert service.cancel(started.public_id) and service.cancel(started.public_id)
            assert service.get(started.public_id).status == SuiteRunStatus.CANCELLATION_REQUESTED
            release.set()
            done = eventually(lambda: service.get(started.public_id), lambda run: run.status == SuiteRunStatus.CANCELLED)
            assert len(calls) == 2
            assert [item.status for item in done.items] == [SuiteRunItemStatus.FAILED, SuiteRunItemStatus.CANCELLED, SuiteRunItemStatus.NOT_ATTEMPTED]
            assert done.items[0].attempts[0].outcome == "PRODUCT_FAILURE"
            assert done.items[1].attempts[0].outcome == "CANCELLED"
            assert not done.items[2].attempts
            assert [item.pinned_plans for item in done.items] == [item.pinned_plans for item in started.items]
            assert done.outcome_counts["product_failures"] == 1 and done.outcome_counts["cancelled"] == 1
            assert done.passed_count == 0 and not service.cancel(started.public_id)
            assert len(self.history.list_recent()) == 2
        finally:
            release.set(); service.close()

    def test_restart_preserves_cancelled_and_reconciles_requested_without_false_timestamps(self):
        import tempfile
        from pathlib import Path
        from qa_agent.suite_runs import SuiteRun, SuiteRunConfig, SuiteRunItem
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteSuiteRunRepository(Path(directory) / "suites.sqlite3")
            run = repository.save(SuiteRun(suite_id=self.suite.id, suite_name="Local suite", config=SuiteRunConfig(), status=SuiteRunStatus.CANCELLATION_REQUESTED, items=[SuiteRunItem(order_index=0, test_case_id=self.case.id, test_case_name=self.case.name)]))
            assert repository.interrupt_incomplete() == 1
            recovered = repository.get(run.id)
            assert recovered.status == SuiteRunStatus.CANCELLED and recovered.finished_at is None
            assert recovered.items[0].status == SuiteRunItemStatus.NOT_ATTEMPTED
            assert SQLiteSuiteRunRepository(Path(directory) / "suites.sqlite3").interrupt_incomplete() == 0
            assert repository.get(run.id) == recovered
