"""Demo release smoke: real gates, SQLite, Chromium and independent Python export."""

import json
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
from playwright.sync_api import sync_playwright

from qa_agent.automation_lifecycle import AutomationLifecycleService, AutomationStatus
from qa_agent.browser_runner import BrowserRunner
from qa_agent.evidence_policy import EvidenceMode, EvidencePolicy, evidence_policy_scope
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.models import DiscoveryResult, DiscoveryStatus, InteractiveElement, QATestPlan, ExecutionStatus, TestCase as Case, TestStep as Step
from qa_agent.pipeline import QATestPipeline
from qa_agent.provider_settings import UnavailableSecretStore
from qa_agent.reliability import AutomationReliabilitySupervisor, QUALITY_GATES
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_execution import TestCaseExecutionService as ExecutionService, RunUnavailableError
from qa_agent.test_case_review import TestCaseReviewService as Review
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.testplan_export import TestPlanExportService as Exports, TestPlanExportError as ExportError
from qa_agent.web import LocalWebApplication, WebResponse, create_application
from qa_agent.workflows import AutomationWorkflow
from tests.test_m5_security_ci import server_for, request
from tests.test_standalone_export import unpack, run_pytest
from tests.test_test_case_quality_editing import application


class FakeDemoProvider(LLMProvider):
    name = 'Offline Demo Fixture'
    model = 'deterministic-fixture'

    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0

    def create_test_plan(self, task, target_url, page_snapshot):
        self.calls += 1
        assert target_url.startswith('http://127.0.0.1:')
        assert page_snapshot
        return self.plans.pop(0)


def services(storage, evidence, provider):
    lifecycle = AutomationLifecycleService(storage.automation_lifecycle_repository, storage.plan_store)
    review = Review(storage.test_case_review_repository, storage.plan_store)
    supervisor = AutomationReliabilitySupervisor(storage.reliability_repository)
    generator = LLMTestPlanGenerator(LLMRouter([provider]), supervisor)
    pipeline = QATestPipeline(None, generator, runner=BrowserRunner(evidence, headless=True),
        plan_store=storage.plan_store, execution_repository=storage.execution_repository,
        run_history=storage.run_history, candidate_review_approved=review.validation_approved_for)
    execution = ExecutionService(storage.test_case_repository, storage.plan_store,
        storage.execution_repository, storage.run_history, evidence_directory=evidence,
        automation_workflow=AutomationWorkflow(pipeline), automation_lifecycle=lifecycle,
        test_case_review=review, candidate_review_required=lambda version: supervisor.requires_review(version) is not None)
    return lifecycle, review, supervisor, execution


def test_demo_end_to_end_generation_approval_browser_validation_and_independent_export(tmp_path):
    storage = create_sqlite_storage(tmp_path / 'demo.sqlite3')
    evidence = tmp_path / 'evidence'
    app = LocalWebApplication(storage.run_history, test_cases=storage.test_case_repository,
        plan_store=storage.plan_store, evidence_root=evidence)
    with server_for(app) as port:
        url = f'http://127.0.0.1:{port}/demo-target/registration'
        case = Case(name='Demo page status journey', description='Open the registration demo, click Change page state and verify the page status displays "Page state changed.".',
            base_url=url, steps=[
                Step(name='Open registration demo', description='Navigate to the registration demo page.', expected='The page is loaded.', order=0),
                Step(name='Change page state', description='Click the Change page state button and inspect the status message.', expected='The page status displays "Page state changed.".', order=1),
                Step(name='Verify changed page status', description='Verify the page status displays "Page state changed.".', expected='The page status displays "Page state changed.".', order=2),
            ])
        storage.test_case_repository.save(case)
        provider = FakeDemoProvider([
            QATestPlan(url=url, steps=[{'action': 'navigate', 'parameters': {'url': url}}, {'action': 'assert_page_loaded'}]),
            QATestPlan(url=url, steps=[{'action': 'click', 'parameters': {'selector': '#change-status'}},
                {'action': 'assert_visible', 'parameters': {'selector': '#page-status', 'expected_text': 'Page state changed.'}}]),
            QATestPlan(url=url, steps=[{'action': 'assert_visible', 'parameters': {'selector': '#page-status', 'expected_text': 'Page state changed.'}}]),
        ])
        lifecycle, review, supervisor, execution = services(storage, evidence, provider)
        review.mark_ready_for_review(case)
        with pytest.raises(RunUnavailableError, match='Approve the TestCase'):
            execution.run(case.id, WorkflowType.AUTOMATION)
        assert provider.calls == 0
        review.approve_test_case(case)
        with evidence_policy_scope(EvidencePolicy(mode=EvidenceMode.EVERY_STEP)):
            generated = execution.run(case.id, WorkflowType.AUTOMATION)
        assert len(generated.executions) == len(case.steps)
        assert all(item.status == ExecutionStatus.PASSED for item in generated.executions)
        assert lifecycle.status(case) == AutomationStatus.NEEDS_VALIDATION
        records = supervisor.repository.list_records()
        assert len(records) == len(case.steps)
        assert all(record.outcome == 'READY_FOR_REVIEW' for record in records)
        for record in records:
            assert len(record.attempts) == 1
            assert record.attempts[0].quality_gates == dict.fromkeys(QUALITY_GATES, 'PASSED')
        assert provider.calls == 3
        versions = [storage.plan_store.find(step.id).id for step in case.steps]
        with pytest.raises(RunUnavailableError, match='Approve the current saved automation'):
            execution.run(case.id, WorkflowType.VALIDATION)
        exports = Exports(storage.test_case_repository, storage.plan_store, lifecycle=lifecycle, review=review)
        with pytest.raises(ExportError):
            exports.bulk_zip([case.id], 'python')
        approved_fingerprint = review.approve_for_validation(case)
        with evidence_policy_scope(EvidencePolicy(mode=EvidenceMode.EVERY_STEP)):
            validated = execution.run(case.id, WorkflowType.VALIDATION)
        assert validated.outcome.value == 'PASSED'
        assert all(item.status == ExecutionStatus.PASSED for item in validated.test_run.final_executions)
        assert lifecycle.status(case) == AutomationStatus.AUTOMATION_READY
        assert review.plan_fingerprint(case) == approved_fingerprint
        data, _ = exports.bulk_zip([case.id], 'python')
        project = unpack(data, tmp_path / 'independent-project')
        manifest = json.loads((project / 'export-manifest.json').read_text())
        assert manifest['test_cases'][0]['verification'] == 'APPROVED_AND_BROWSER_VALIDATED'
        # Source validation cannot claim standalone execution before it happens.
        assert manifest['standalone_execution'] == 'NOT_TESTED'
        assert all('qa_agent' not in path.read_text() for path in project.rglob('*.py'))
        result = run_pytest(project)
        assert result.returncode == 0, result.stdout + result.stderr
        assert '1 passed' in result.stdout
        assert provider.calls == 3
        assert [storage.plan_store.find(step.id).id for step in case.steps] == versions
        assert evidence.exists() and list(evidence.rglob('*.png'))


@pytest.mark.parametrize('workflow', list(WorkflowType))
def test_unapproved_testcase_cannot_start_any_execution_workflow(tmp_path, workflow):
    storage = create_sqlite_storage(tmp_path / 'unapproved.sqlite3')
    provider = FakeDemoProvider([])
    _, review, _, execution = services(storage, tmp_path / 'evidence', provider)
    case = Case(name='Unapproved local case', description='Open a local demo page.', base_url='http://127.0.0.1:1',
        steps=[Step(name='Open local demo', description='Navigate to the local demo page.', expected='The page is loaded.', order=0)])
    storage.test_case_repository.save(case)
    review.mark_ready_for_review(case)
    execution._runner_factory = Mock(side_effect=AssertionError('Browser execution must not start.'))
    with pytest.raises(RunUnavailableError, match='Approve the TestCase'):
        execution.run(case.id, workflow)
    assert provider.calls == 0
    execution._runner_factory.assert_not_called()
    assert storage.plan_store.find(case.steps[0].id) is None


def test_demo_generation_missing_expected_assertion_is_rejected_by_quality_gates(tmp_path):
    url = 'http://127.0.0.1:1/demo-target/registration'
    step = Step(name='Change page state and inspect status', description='Click Change page state and verify page status.',
        expected='The page status displays "Page state changed.".', order=0)
    provider = FakeDemoProvider([QATestPlan(url=url, steps=[{'action': 'click', 'parameters': {'selector': '#change-status'}}])])
    storage = create_sqlite_storage(tmp_path / 'quality.sqlite3')
    supervisor = AutomationReliabilitySupervisor(storage.reliability_repository)
    discovery = DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=url,
        interactive_elements=[InteractiveElement(kind='button', tag='button', selector='#change-status', accessible_name='Change page state')])
    with pytest.raises(PlanValidationError):
        LLMTestPlanGenerator(LLMRouter([provider]), supervisor).generate_with_plan(step, discovery)
    record = supervisor.repository.list_records()[0]
    assert record.outcome == 'NEEDS_ATTENTION' and record.candidate_version_id is None
    assert record.attempts[0].quality_gates['expected_result_coverage'] == 'FAILED'
    assert storage.plan_store.find(step.id) is None


def test_documented_demo_seed_and_server_startup_use_isolated_storage(tmp_path, monkeypatch):
    database = tmp_path / 'installation.sqlite3'
    evidence = tmp_path / 'installation-evidence'
    monkeypatch.setenv('LLM_PROVIDER_ORDER', '')
    for name in ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'GROQ_API_KEY', 'OPENROUTER_API_KEY']:
        monkeypatch.setenv(name, '')
    result = subprocess.run([sys.executable, '-m', 'qa_agent.demo', '--database', str(database),
        '--demo-base-url', 'http://127.0.0.1:8000/demo-target/registration'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'created' in result.stdout and database.exists()
    with patch('qa_agent.web.create_default_secret_store', return_value=UnavailableSecretStore()):
        app = create_application(database, evidence)
    with server_for(app) as port:
        assert request(port, 'GET', '/health')[0] == 200
        status, _, body = request(port, 'GET', '/demo-target/registration')
        assert status == 200 and b'Registration demo' in body
        assert request(port, 'GET', '/')[0] == 200
    for command in [('qa_agent.web', '--help'), ('qa_agent', 'export', '--help')]:
        help_result = subprocess.run([sys.executable, '-m', *command], capture_output=True, text=True, timeout=20)
        assert help_result.returncode == 0, help_result.stderr


def test_real_browser_static_and_dynamic_post_forms_preserve_same_origin_and_csrf():
    app, _, _ = application()
    app._handle_post = Mock(return_value=WebResponse.json(200, '{"accepted":true}'))
    with server_for(app) as port, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        origin = f'http://127.0.0.1:{port}'
        try:
            for dynamic in (False, True):
                page.goto(origin + '/drafts/new')
                with page.expect_response(lambda response: response.request.method == 'POST') as posted:
                    if dynamic:
                        page.evaluate("""() => {
                            const form = document.createElement('form');
                            form.method = 'post'; form.action = '/drafts';
                            document.body.append(form); form.requestSubmit();
                        }""")
                    else:
                        page.evaluate("""() => {
                            const form = document.querySelector('form');
                            form.querySelectorAll('[required]').forEach(input => {
                                input.value = input.name === 'base_url' ? location.origin : 'Synthetic local draft';
                            });
                            form.requestSubmit();
                        }""")
                response = posted.value
                assert response.status == 200
                assert response.request.headers['origin'] == origin
                assert '_csrf=' in response.request.post_data
                page.wait_for_url(origin + '/drafts')
            assert app._handle_post.call_count == 2
        finally:
            browser.close()
