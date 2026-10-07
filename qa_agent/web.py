"""Local UI for persisted TestCases and TestRun history."""

import argparse
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from uuid import UUID, uuid4

from qa_agent.presentation import (
    UI_CSS,
    badge,
    escape_html,
    format_duration,
    format_timestamp,
    failure_message,
    outcome_label,
    outcome_tone,
)
from qa_agent.reporting import RunAttemptReport, RunEvidenceReport, RunReportGenerator, RunStepReport
from qa_agent.browser_runner import BrowserRunner
from qa_agent.provider_settings import (
    ProviderSettingsRepository,
    ProviderSettingsService,
    SecretStoreError,
    create_default_secret_store,
)
from qa_agent.run_history import RunHistoryDetail, RunHistoryRecord, RunHistoryService, WorkflowType
from qa_agent.automation_lifecycle import (
    AutomationLifecycleRepository,
    AutomationLifecycleService,
    AutomationStatus,
    InMemoryAutomationLifecycleRepository,
)
from qa_agent.drafts import Draft, DraftRepository, InMemoryDraftRepository
from qa_agent.test_case_execution import (
    RunUnavailableError,
    TestCaseExecutionService,
    WorkflowAvailability,
)
from qa_agent.test_case_authoring import (
    TestCaseAuthoringError,
    TestCaseAuthoringService,
    TestCaseDraft,
    TestCaseDraftStore,
    editable_test_case_values,
    merge_test_case_edits,
)
from qa_agent.test_case_repository import TestCaseRepository
from qa_agent.storage import create_sqlite_storage
from qa_agent.llm_usage import LLMUsageService
from qa_agent.pipeline import QATestPipeline
from qa_agent.background_authoring import BackgroundAuthoringService
from qa_agent.background_execution import BackgroundRunService
from qa_agent.redaction import redact_secrets
from qa_agent.execution_progress import (
    AuthoringEventType,
    ExecutionProgressStore,
)
from qa_agent.models import PlanVersionOrigin, TestCase
from qa_agent.plan_store import InMemoryPlanStore, PlanStore
from qa_agent.testplan_export import (
    TestPlanExportError,
    TestPlanExportService,
    portable_zip,
    project_zip,
)
from qa_agent.test_suites import (
    InMemoryTestSuiteRepository,
    SQLiteTestSuiteRepository,
    TestSuiteService,
)
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_case_editing import TestCaseEditError, create_manual_test_case, edit_test_case
from qa_agent.workflows import AutomationWorkflow


_PAGE_LIMIT = 500
logger = logging.getLogger(__name__)
_MAX_FORM_BODY_BYTES = 8192
_MAX_AUTHORING_FORM_BODY_BYTES = 80_000
_MAX_DRAFT_SAVE_BODY_BYTES = 256 * 1024
_MAX_MANUAL_FORM_BODY_BYTES = 1024 * 1024
_WORKFLOWS = {item.value for item in WorkflowType}
_STATUSES = {"PASSED", "FAILED"}
_FAILURE_TYPES = {
    "PRODUCT_FAILURE",
    "AUTOMATION_DRIFT",
    "INFRASTRUCTURE_ERROR",
    "SETUP_FAILURE",
}


@dataclass(frozen=True)
class WebResponse:
    status: int
    content_type: str
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def html(cls, status: int, body: str) -> "WebResponse":
        return cls(status, "text/html; charset=utf-8", body.encode("utf-8"))

    @classmethod
    def json(cls, status: int, body: str) -> "WebResponse":
        return cls(status, "application/json; charset=utf-8", body.encode("utf-8"))

    @classmethod
    def redirect(cls, location: str) -> "WebResponse":
        return cls(303, "text/plain; charset=utf-8", b"Request accepted. Redirecting.\n", {"Location": location})


class LocalWebApplication:
    """Route requests through history/report services, never raw SQL."""

    def __init__(
        self,
        run_history: RunHistoryService,
        reports: RunReportGenerator | None = None,
        evidence_root: str | Path | None = None,
        test_cases: TestCaseRepository | None = None,
        run_service: TestCaseExecutionService | None = None,
        authoring_service: TestCaseAuthoringService | None = None,
        draft_store: TestCaseDraftStore | None = None,
        progress_store: ExecutionProgressStore | None = None,
        plan_store: PlanStore | None = None,
        provider_settings: ProviderSettingsService | None = None,
        provider_router=None,
        llm_usage: LLMUsageService | None = None,
        test_suites: TestSuiteService | None = None,
        drafts: DraftRepository | None = None,
        automation_lifecycle: AutomationLifecycleService | None = None,
        automation_lifecycle_repository: AutomationLifecycleRepository | None = None,
    ) -> None:
        self._run_history = run_history
        self._reports = reports or RunReportGenerator()
        self._evidence_root = Path(evidence_root).expanduser() if evidence_root else None
        self._test_cases = test_cases
        self._run_service = run_service
        self._authoring_service = authoring_service
        self._draft_store = draft_store or TestCaseDraftStore()
        self._progress_store = progress_store or ExecutionProgressStore()
        self._plan_store = plan_store
        self._drafts = drafts or InMemoryDraftRepository()
        lifecycle_repository = automation_lifecycle_repository or InMemoryAutomationLifecycleRepository()
        self._automation_lifecycle = automation_lifecycle or AutomationLifecycleService(
            lifecycle_repository, plan_store or InMemoryPlanStore()
        )
        self._provider_settings = provider_settings
        self._provider_router = provider_router
        self._llm_usage = llm_usage
        self._test_suites = test_suites or TestSuiteService(
            InMemoryTestSuiteRepository(), test_cases
        )
        self._testplan_exports = (
            TestPlanExportService(test_cases, plan_store)
            if test_cases is not None and plan_store is not None else None
        )
        self._background_runs = (
            BackgroundRunService(run_service, run_history, self._progress_store)
            if run_service is not None else None
        )
        self._background_authoring = (
            BackgroundAuthoringService(
                authoring_service,
                self._draft_store,
                self._progress_store,
            )
            if authoring_service is not None else None
        )

    @property
    def progress_store(self) -> ExecutionProgressStore:
        return self._progress_store

    def close(self) -> None:
        if self._background_runs is not None:
            self._background_runs.close()
        if self._background_authoring is not None:
            self._background_authoring.close()

    def handle(self, method: str, target: str, body: bytes | str | None = None) -> WebResponse:
        parsed = urlsplit(target)
        path = parsed.path
        query = parse_qs(parsed.query)
        parts = path.strip("/").split("/")
        if method.upper() == "POST":
            return self._handle_post(parts, body)
        if method.upper() != "GET":
            return WebResponse.html(405, self._page(
                "Method not allowed",
                '<div class="error-state"><h1>Method not allowed</h1>'
                "<p>Use the TestCase run form to start a workflow.</p></div>",
            ))
        if path == "/assets/ui.js":
            return WebResponse(
                200,
                "application/javascript; charset=utf-8",
                _UI_JAVASCRIPT.encode("utf-8"),
            )
        if path == "/health":
            return WebResponse.json(200, '{"status":"ok","service":"ai-qa-agent"}')
        if path == "/drafts":
            return WebResponse.html(200, self._drafts_page())
        if path == "/drafts/new":
            return WebResponse.html(200, self._draft_edit_page())
        if path == "/test-cases/manual":
            return WebResponse.html(200, self._manual_test_case_page(
                name=query.get("name", [""])[0],
                base_url=query.get("base_url", [""])[0],
                scenario=query.get("scenario", [""])[0],
                source_draft_id=query.get("draft_id", [""])[0],
            ))
        if len(parts) == 3 and parts[:2] == ["api", "progress"]:
            return self._progress_json(parts[2])
        if len(parts) == 3 and parts[:2] == ["runs", "progress"]:
            return self._progress_page(parts[2])
        if len(parts) == 4 and parts[:3] == ["api", "test-cases", "authoring-progress"]:
            return self._authoring_progress_json(parts[3])
        if len(parts) == 3 and parts[:2] == ["drafts"]:
            draft_id = _parse_uuid(parts[2])
            if draft_id is None:
                return self._not_found("Draft not found")
            return self._draft_edit_page_response(draft_id)
        if len(parts) == 3 and parts[:2] == ["test-cases", "authoring-progress"]:
            return self._authoring_progress_page(parts[2])
        if path == "/":
            return WebResponse.html(200, self._dashboard())
        if path == "/settings/providers" and self._provider_settings is not None:
            return WebResponse.html(200, self._provider_settings_page(query))
        if path == "/settings/usage" and self._llm_usage is not None:
            return WebResponse.html(200, self._usage_analytics_page(query))
        if path == "/demo-target/registration":
            return WebResponse.html(200, _local_demo_page())
        if path == "/test-cases":
            return WebResponse.html(200, self._test_case_list())
        if path == "/test-suites":
            return WebResponse.html(200, self._test_suites_page())
        if path == "/test-cases/new":
            return WebResponse.html(200, self._new_test_case_page())
        if len(parts) == 3 and parts[0] == "test-cases" and parts[1] == "review":
            return self._draft_review_page(parts[2])
        if path == "/runs":
            return WebResponse.html(200, self._runs_page(query))

        evidence_parts = path.strip("/").split("/")
        if len(evidence_parts) == 5 and evidence_parts[0] == "runs" and evidence_parts[2] == "evidence":
            return self._serve_evidence(evidence_parts[1], evidence_parts[3], evidence_parts[4])

        if len(parts) == 3 and parts[0] == "runs" and parts[2] in {"report.json", "report.html"}:
            run_id = _parse_uuid(parts[1])
            detail = self._run_history.get_detail(run_id) if run_id else None
            if detail is None:
                return self._not_found("Run report not found")
            report = self._reports.generate_history(detail)
            if parts[2] == "report.json":
                return WebResponse.json(200, report.to_json())
            return WebResponse.html(
                200,
                self._reports.to_html(
                    report,
                    evidence_url=lambda step, attempt, evidence, index: self._evidence_url_if_available(
                        detail, run_id, step, attempt, evidence, index
                    ),
                    show_html_report_link=False,
                    run_details_url=f"/runs/{run_id}",
                ),
            )
        if len(parts) == 2 and parts[0] == "runs":
            run_id = _parse_uuid(parts[1])
            detail = self._run_history.get_detail(run_id) if run_id else None
            if detail is None:
                return self._not_found("Run not found")
            report = self._reports.generate_history(detail)
            return WebResponse.html(
                200,
                self._reports.to_html(
                    report,
                    evidence_url=lambda step, attempt, evidence, index: self._evidence_url_if_available(
                        detail, run_id, step, attempt, evidence, index
                    ),
                ),
            )
        if len(parts) == 2 and parts[0] == "test-cases":
            test_case_id = _parse_uuid(parts[1])
            if test_case_id is None:
                return self._not_found("TestCase not found")
            return self._test_case_page(test_case_id)
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "edit":
            test_case_id = _parse_uuid(parts[1])
            return self._test_case_edit_page_response(test_case_id) if test_case_id else self._not_found("TestCase not found")
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "plans":
            test_case_id = _parse_uuid(parts[1])
            return self._test_plan_view(test_case_id) if test_case_id else self._not_found("TestCase not found")
        if len(parts) == 4 and parts[0] == "test-cases" and parts[2] == "export":
            test_case_id = _parse_uuid(parts[1])
            if test_case_id is None:
                return self._not_found("TestCase not found")
            return self._test_case_export(test_case_id, parts[3])
        if len(parts) == 3 and parts[0] == "test-suites" and parts[2] == "export":
            suite_id = _parse_uuid(parts[1])
            if suite_id is None:
                return self._not_found("Test Suite not found")
            return self._test_suite_export(suite_id, query)
        if len(parts) == 2 and parts[0] == "test-suites":
            suite_id = _parse_uuid(parts[1])
            if suite_id is None:
                return self._not_found("Test Suite not found")
            return self._test_suite_page(suite_id)
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "run":
            return WebResponse.html(405, self._page(
                "Method not allowed",
                '<div class="error-state"><h1>POST required</h1>'
                "<p>Starting a run requires submitting the TestCase form.</p></div>",
            ))
        return self._not_found("Page not found")

    def _handle_post(self, parts: list[str], body: bytes | str | None) -> WebResponse:
        if parts == ["drafts"]:
            return self._handle_draft_create(body)
        if len(parts) == 3 and parts[0] == "drafts":
            return self._handle_draft_post(parts[1], parts[2], body)
        if parts == ["test-cases", "manual"]:
            return self._handle_manual_test_case_create(body)
        if parts == ["test-cases", "manual", "prepare"]:
            form, error = _parse_form_body(body, max_bytes=_MAX_AUTHORING_FORM_BODY_BYTES)
            values = {key: items[0] for key, items in form.items()}
            if error:
                return WebResponse.html(400, self._manual_test_case_page(error=error))
            return WebResponse.html(200, self._manual_test_case_page(
                name=values.get("name", ""), base_url=values.get("base_url", ""),
                scenario=values.get("description", ""),
            ))
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "edit":
            return self._handle_test_case_edit(parts[1], body)
        if (
            len(parts) == 4
            and parts[:2] == ["test-cases", "authoring-progress"]
            and parts[3] in {"save-draft", "create-manually"}
        ):
            return self._handle_authoring_failure_action(parts[2], parts[3], body)
        if parts == ["settings", "providers"] and self._provider_settings is not None:
            return self._handle_provider_settings_post(body)
        if parts == ["test-cases", "export"]:
            return self._handle_bulk_export(body)
        if parts == ["test-suites"]:
            return self._handle_test_suite_create(body)
        if len(parts) == 3 and parts[0] == "test-suites" and parts[2] == "update":
            return self._handle_test_suite_update(parts[1], body)
        if len(parts) == 4 and parts[0] == "test-suites" and parts[2] == "members":
            return self._handle_test_suite_members(parts[1], parts[3], body)
        if parts == ["test-cases", "generate"]:
            return self._handle_generate_test_case(body)
        if (
            len(parts) == 4
            and parts[:2] == ["test-cases", "authoring-progress"]
            and parts[3] == "retry"
        ):
            return self._handle_authoring_retry(parts[2], body)
        if len(parts) == 4 and parts[:2] == ["test-cases", "review"]:
            return self._handle_draft_action(parts[2], parts[3], body)
        if len(parts) != 3 or parts[0] != "test-cases" or parts[2] != "run":
            return WebResponse.html(405, self._page(
                "Method not allowed",
                '<div class="error-state"><h1>Method not allowed</h1>'
                "<p>Submit a TestCase run using its run form.</p></div>",
            ))
        test_case_id = _parse_uuid(parts[1])
        if test_case_id is None:
            return self._not_found("TestCase not found")
        if self._test_cases is not None and self._test_cases.get(test_case_id) is None:
            return self._not_found("TestCase not found")
        if self._run_service is None:
            return WebResponse.html(405, self._page(
                "Run unavailable",
                '<div class="error-state"><h1>Run unavailable</h1>'
                "<p>This application has no execution service configured.</p></div>",
            ))
        form, form_error = _parse_form_body(body)
        if form_error is not None:
            return self._run_error(test_case_id, form_error, 400)
        workflow_value = form.get("workflow", [""])[0]
        try:
            workflow = WorkflowType(workflow_value)
        except ValueError:
            return self._run_error(test_case_id, "Choose Automation, Validation, or Regression before starting a run.", 400)
        if workflow not in {WorkflowType.AUTOMATION, WorkflowType.VALIDATION, WorkflowType.REGRESSION}:
            return self._run_error(test_case_id, "Choose Automation, Validation, or Regression before starting a run.", 400)
        if self._background_runs is None:
            return self._run_error(
                test_case_id,
                "Execution is not available for this application.",
                503,
            )
        progress_id = self._background_runs.start(test_case_id, workflow)
        return WebResponse.redirect(f"/runs/progress/{progress_id}")

    def _progress_json(self, progress_id: str) -> WebResponse:
        snapshot = self._progress_store.get(progress_id)
        if snapshot is None:
            return WebResponse.json(404, json.dumps({
                "error": "Execution progress is no longer available."
            }))
        payload = snapshot.to_public_dict()
        payload["redirect_after_ms"] = 1200 if snapshot.final_run_url else None
        return WebResponse.json(
            200,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def _authoring_progress_json(self, progress_id: str) -> WebResponse:
        snapshot = self._progress_store.get_authoring(progress_id)
        if snapshot is None:
            return WebResponse.json(404, json.dumps({
                "error": "Authoring progress is no longer available."
            }))
        return WebResponse.json(
            200,
            json.dumps(snapshot.to_public_dict(), ensure_ascii=False, separators=(",", ":")),
        )

    def _authoring_progress_page(self, progress_id: str) -> WebResponse:
        snapshot = self._progress_store.get_authoring(progress_id)
        if snapshot is None:
            return self._not_found("Authoring progress is no longer available.")
        active_event = {
            "Understanding scenario": AuthoringEventType.AUTHORING_STARTED,
            "Generating TestCase": AuthoringEventType.LLM_REQUEST_STARTED,
            "Validating result": AuthoringEventType.TESTCASE_VALIDATION_STARTED,
        }.get(snapshot.phase)
        events = "".join(
            '<li class="progress-event"><span aria-hidden="true">'
            + ("✕" if event.event_type == AuthoringEventType.AUTHORING_FAILED
               else "◉" if event.event_type == active_event and not snapshot.finished
               else "✓")
            + '</span>'
            + f'<span>{escape_html(event.message)}</span></li>'
            for event in snapshot.events
        )
        result = ""
        failure_actions = ""
        if snapshot.finished:
            if snapshot.success:
                result = (
                    '<p data-authoring-result-message><strong>TestCase draft ready.</strong> '
                    'Redirecting to Review…</p>'
                    f'<a class="button primary" data-authoring-review-link '
                    f'href="{escape_html(snapshot.review_url or "")}">Open Review</a>'
                )
            else:
                provider_details = ""
                if snapshot.provider_failures:
                    rows = "".join(
                        '<li>'
                        + escape_html(
                            f"{failure.provider_name} — {failure.category}"
                            + (f" · HTTP {failure.http_status}" if failure.http_status else "")
                            + f" · {failure.message}"
                            + (f" · Retry after {failure.retry_after_seconds} seconds"
                               if failure.retry_after_seconds else "")
                        )
                        + '</li>'
                        for failure in snapshot.provider_failures
                    )
                    provider_details = (
                        '<details class="technical-details"><summary>Developer details</summary>'
                        f'<ul>{rows}</ul></details>'
                    )
                result = (
                    f'<p data-authoring-result-message><strong>{_authoring_error_label(snapshot.error_category)}</strong></p>'
                    f'<p>{escape_html(snapshot.error_message or "An unexpected authoring error occurred.")}</p>'
                    + provider_details
                )
                failure_actions = (
                    '<div class="actions">'
                    f'<form method="post" action="/test-cases/authoring-progress/{escape_html(progress_id)}/create-manually">'
                    '<button class="button primary" type="submit">Create manually</button></form>'
                    f'<form method="post" action="/test-cases/authoring-progress/{escape_html(progress_id)}/save-draft">'
                    '<button class="button" type="submit">Save as Draft</button></form></div>'
                )
        retry_hidden = " hidden" if not snapshot.finished or snapshot.success else ""
        retry_form = (
            f'<form method="post" action="/test-cases/authoring-progress/{escape_html(progress_id)}/retry" '
            f'data-authoring-retry data-authoring-form{retry_hidden}>'
            '<button class="button" type="submit">Try Again</button></form>'
        )
        body = (
            '<header class="page-heading"><p class="eyebrow">AI TestCase authoring</p>'
            f'<h1>{escape_html(snapshot.test_case_name)}</h1>'
            f'<p class="lead">Base URL: <span>{escape_html(snapshot.base_url)}</span></p></header>'
            '<div class="summary-grid">'
            + _summary_card("Current phase", f'<span data-authoring-phase>{escape_html(snapshot.phase)}</span>', raw=True)
            + _summary_card("Elapsed", f'<span data-authoring-elapsed>{escape_html(format_duration(snapshot.elapsed_ms))}</span>', raw=True)
            + _summary_card("Status", f'<span data-authoring-state>{escape_html(snapshot.state.value.title())}</span>', raw=True)
            + _summary_card("Provider", f'<span data-authoring-provider>{escape_html(snapshot.provider_name or "—")}</span>', raw=True)
            + '</div>'
            + f'<div class="authoring-progress" data-authoring-progress-id="{escape_html(progress_id)}">'
            + '<section class="panel"><h2>Progress</h2>'
            + f'<ul class="compact-list" data-authoring-events>{events}</ul></section>'
            + '<section class="panel progress-result" data-authoring-result'
            + ('' if snapshot.finished else ' hidden')
            + f'><div data-authoring-result-content>{result}{failure_actions}</div>{retry_form}</section>'
            + '<p class="muted" data-authoring-notice aria-live="polite"></p></div>'
        )
        return WebResponse.html(200, self._page(
            "AI TestCase Authoring",
            body,
            breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases")],
        ))

    def _progress_page(self, progress_id: str) -> WebResponse:
        snapshot = self._progress_store.get(progress_id)
        if snapshot is None:
            return self._not_found("Execution progress is no longer available.")

        case_name = snapshot.test_case_name or "TestCase run"
        result = ""
        retry = ""
        if snapshot.state.value == "FINISHED":
            category = snapshot.outcome or snapshot.error_category or "FAILED"
            failure = snapshot.automation_generation_failure
            if category == "AUTOMATION_GENERATION_ERROR" and failure is not None:
                generated_count = sum(
                    step.order < failure.step_order
                    and step.automation_state in {"Generated", "Reused", "Repaired"}
                    for step in snapshot.steps
                )
                remaining_count = sum(
                    step.state.value == "NOT_ATTEMPTED"
                    for step in snapshot.steps
                )
                saved_automation = (
                    "Available from a previous attempt."
                    if failure.prior_plan_exists
                    else "Not available for this step."
                )
                new_plan = "Yes" if failure.new_plan_saved else "No"
                result = (
                    f'<p data-result-summary>{badge(outcome_label(category), outcome_tone(category))} '
                    f'Automation stopped at Step {failure.step_order + 1} of {len(snapshot.steps)}.</p>'
                    f'<p>Generated successfully: {generated_count} steps.</p>'
                    f'<p>Remaining: {remaining_count} steps not attempted.</p>'
                    f'<p>Not attempted because automation generation stopped at Step {failure.step_order + 1}.</p>'
                    f'<p>Reason: {escape_html(failure.safe_reason)}</p>'
                    f'<p>Saved automation: {escape_html(saved_automation)}</p>'
                    f'<p>New plan saved: {new_plan}.</p>'
                    '<details class="technical-details"><summary>Developer details</summary>'
                    f'<p>Classification: <code>{escape_html(failure.technical_classification)}</code></p>'
                    + (
                        '<ul class="validation-issues">' + ''.join(
                            '<li><code>' + escape_html(issue.code) + '</code> at '
                            '<code>' + escape_html(issue.path) + '</code>: '
                            + escape_html(issue.message) + '</li>'
                            for issue in failure.validation_issues
                        ) + '</ul>'
                        if failure.validation_issues else ''
                    )
                    + '</details>'
                )
            else:
                result = (
                    f'<p data-result-summary>{badge(outcome_label(category), outcome_tone(category))} '
                    f'{escape_html(snapshot.error_message or failure_message(category))}</p>'
                )
            result += (
                f'<a class="button primary" data-final-run-link href="{escape_html(snapshot.final_run_url)}">'
                "View Run Details</a>"
                if snapshot.final_run_url else ""
            )
            if snapshot.error_category in {
                "INFRASTRUCTURE_ERROR",
                "AUTOMATION_EXECUTION_ERROR",
                "EXECUTION_ERROR",
                "AUTOMATION_GENERATION_ERROR",
                "SETUP_FAILURE",
            }:
                retry = (
                    f'<form method="post" action="/test-cases/{snapshot.test_case_id}/run" '
                    'data-run-form data-progress-retry><input type="hidden" name="workflow" '
                    f'value="{escape_html(snapshot.workflow_type)}">'
                    f'<button class="button" type="submit">{"Retry Automation" if snapshot.error_category == "AUTOMATION_GENERATION_ERROR" else "Retry Run"}</button></form>'
                )

        state_symbols = {
            "PENDING": ("○", "Pending"),
            "PREPARING_AUTOMATION": ("◌", "Preparing automation"),
            "READY": ("✓", "Automation ready"),
            "RUNNING": ("◉", "Running"),
            "PASSED": ("✓", "Passed"),
            "FAILED": ("✕", "Failed"),
            "BLOCKED": ("⊘", "Blocked"),
            "NOT_ATTEMPTED": ("○", "Not attempted"),
        }
        step_rows = []
        for step in snapshot.steps:
            symbol, label = state_symbols.get(step.state.value, ("○", step.state.value))
            execution_state = (
                f'<span class="progress-step-state">Execution {escape_html(step.execution_state.value.replace("_", " ").title())}</span>'
            )
            automation_state = (
                f'<span class="muted">Automation {escape_html(step.automation_state.casefold())}</span>'
                if step.automation_state else '<span class="muted">Automation not prepared</span>'
            )
            version_label = f" v{step.plan_version}" if step.plan_version is not None else ""
            provenance = (
                f'<span class="muted">Plan {escape_html(_plan_origin_display(step.plan_origin))}'
                f'{version_label}</span>'
                if step.plan_origin else ""
            )
            failure = (
                f'<span class="progress-failure">{escape_html(step.failure_classification.replace("_", " ").title())}: '
                f'{escape_html(step.message or "Step failed.")}</span>'
                if step.failure_classification else ""
            )
            evidence = (
                f'<span class="progress-evidence">Screenshot evidence captured ({step.evidence_count})</span>'
                if step.evidence_count else ""
            )
            step_rows.append(
                f'<li class="progress-step state-{escape_html(step.state.value.casefold())}">'
                f'<span class="progress-symbol" aria-hidden="true">{symbol}</span>'
                f'<span><strong>Step {step.order + 1}: {escape_html(step.name)}</strong>'
                f'<span class="progress-step-state">{escape_html(label)}</span>{execution_state}'
                f'{automation_state}{provenance}{failure}{evidence}</span></li>'
            )

        events = snapshot.events
        developer_events = self._progress_event_list(
            events, {event.event_type.value for event in events}, "all"
        )
        finished = snapshot.state.value == "FINISHED"
        phase = escape_html(snapshot.phase)
        body = (
            '<header class="page-heading"><p class="eyebrow">Live execution</p>'
            f'<h1>{escape_html(case_name)}</h1>'
            f'<p class="lead">{badge(snapshot.workflow_type, "workflow")} '
            f'<span data-progress-phase>{phase}</span></p></header>'
            '<div class="summary-grid">'
            + _summary_card(
                "Current phase",
                f'<span data-progress-phase>{phase}</span>',
                raw=True,
            )
            + _summary_card(
                "Elapsed",
                f'<span data-progress-elapsed>{escape_html(format_duration(snapshot.elapsed_ms))}</span>',
                raw=True,
            )
            + _summary_card(
                "Execution",
                f'<span data-progress-state>{escape_html(snapshot.state.value.title())}</span>',
                raw=True,
            )
            + '</div>'
            + f'<div class="progress-live" data-progress-id="{escape_html(progress_id)}">'
            + '<section class="panel"><h2>TestCase progress</h2>'
            + (
                f'<ol class="progress-steps" id="progress-steps">{"".join(step_rows)}</ol>'
                if step_rows else '<ol class="progress-steps" id="progress-steps"></ol>'
            )
            + '</section><details class="panel technical-details" data-progress-developer-details>'
            + '<summary>Developer details</summary>'
            + (developer_events if developer_events else '<ul class="compact-list" data-progress-events="all"></ul>')
            + '</details></div>'
            + '<section class="panel progress-result" data-progress-result'
            + ('' if finished else ' hidden')
            + '><h2>Result</h2><div data-progress-result-content>' + result + '</div></section>'
            + (f'<div class="actions">{retry}</div>' if retry else "")
            + '<p class="muted" data-progress-notice aria-live="polite"></p>'
        )
        return WebResponse.html(
            200,
            self._page(
                f"{case_name} progress",
                body,
                breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases")],
            ),
        )

    @staticmethod
    def _progress_event_list(events, event_types: set[str], group: str) -> str:
        rows = []
        for event in events:
            if event.event_type.value not in event_types:
                continue
            label = event.message or event.event_type.value.replace("_", " ").title()
            event_type = event.event_type.value
            symbol = (
                "✕" if event_type.endswith("FAILED")
                else "◉" if event_type.endswith("STARTED")
                else "✓"
            )
            rows.append(
                f'<li class="progress-event"><span aria-hidden="true">{symbol}</span>'
                f'<span>{escape_html(label)}</span></li>'
            )
        return (
            f'<ul class="compact-list" data-progress-events="{group}">{"".join(rows)}</ul>'
            if rows else ""
        )

    def _recent_drafts_panel(self) -> str:
        drafts = self._drafts.list(5)
        if not drafts:
            summary = '<p class="muted">Save an unfinished idea and return to it later.</p>'
        else:
            summary = '<ul class="compact-list">' + "".join(
                f'<li><a href="/drafts/{item.id}">{escape_html(item.title)}</a></li>'
                for item in drafts
            ) + '</ul>'
        return (
            '<section class="panel"><div class="section-heading"><h2>Recent Drafts</h2>'
            '<a class="button" href="/drafts">View all drafts</a></div>'
            + summary + '<a class="button" href="/drafts/new">Save draft</a></section>'
        )

    def _drafts_page(self) -> str:
        drafts = self._drafts.list(500)
        rows = "".join(
            '<tr><td><a href="/drafts/' + str(item.id) + '">' + escape_html(item.title) + '</a></td>'
            + f'<td>{escape_html(item.base_url or "—")}</td>'
            + f'<td>{escape_html(format_timestamp(item.updated_at))}</td>'
            + '<td><form method="post" action="/drafts/' + str(item.id) + '/delete">'
            + '<button class="button" type="submit">Delete</button></form></td></tr>'
            for item in drafts
        )
        content = (
            '<header class="page-heading"><h1>Drafts</h1>'
            '<p class="lead">Capture unfinished testing ideas without creating a TestCase.</p></header>'
            '<div class="actions"><a class="button primary" href="/drafts/new">New Draft</a></div>'
            + ('<section class="panel"><div class="table-wrap"><table><thead><tr>'
               '<th>Title</th><th>Base URL</th><th>Updated</th><th>Actions</th></tr></thead>'
               f'<tbody>{rows}</tbody></table></div></section>' if drafts else
               self._empty_state("No Drafts yet.", "Save a testing idea to work on it later."))
        )
        return self._page("Drafts", content, current="Drafts", breadcrumbs=[("Dashboard", "/")])

    def _draft_edit_page_response(self, draft_id: UUID) -> WebResponse:
        draft = self._drafts.get(draft_id)
        if draft is None:
            return self._not_found("Draft not found")
        return WebResponse.html(200, self._draft_edit_page(draft))

    def _draft_edit_page(self, draft: Draft | None = None, error: str | None = None, submitted: dict[str, str] | None = None) -> str:
        editing = draft is not None
        submitted = submitted or {}
        action = f"/drafts/{draft.id}/update" if draft else "/drafts"
        title = submitted.get("title", draft.title if draft else "")
        body = submitted.get("body", draft.body if draft else "")
        base_url = submitted.get("base_url", (draft.base_url or "") if draft else "")
        notes = submitted.get("notes", (draft.notes or "") if draft else "")
        error_html = f'<p class="authoring-error" role="alert">{escape_html(error)}</p>' if error else ""
        content = (
            '<header class="page-heading"><h1>' + ("Edit Draft" if editing else "New Draft") + '</h1>'
            '<p class="lead">Drafts stay separate from TestCases, Runs, and Test Suites.</p></header>'
            + error_html + f'<form method="post" action="{action}" class="panel" data-inline-validation novalidate>'
            + '<div class="field"><label for="draft-title">Title</label>'
            + f'<input id="draft-title" name="title" maxlength="200" required value="{escape_html(title)}"></div>'
            + '<div class="field"><label for="draft-body">Testing idea / scenario</label>'
            + f'<textarea id="draft-body" name="body" maxlength="6000" rows="8" required>{escape_html(body)}</textarea></div>'
            + '<div class="field"><label for="draft-url">Base URL (optional)</label>'
            + f'<input id="draft-url" name="base_url" maxlength="2048" value="{escape_html(base_url)}"></div>'
            + '<div class="field"><label for="draft-notes">Notes (optional)</label>'
            + f'<textarea id="draft-notes" name="notes" maxlength="6000" rows="3">{escape_html(notes)}</textarea></div>'
            + '<div class="actions"><button class="button primary" type="submit">Save Draft</button>'
            + '<a class="button" href="/drafts">Cancel</a></div></form>'
        )
        if draft is not None:
            content += (
                f'<section class="panel"><h2>Use this Draft</h2><div class="actions">'
                f'<form method="post" action="/drafts/{draft.id}/convert">'
                '<button class="button primary" type="submit">Create TestCase manually</button></form>'
                f'<form method="post" action="/drafts/{draft.id}/generate">'
                '<button class="button" type="submit">Generate with AI</button></form></div></section>'
            )
        return self._page(
            "Edit Draft" if editing else "New Draft",
            content,
            current="Drafts",
            breadcrumbs=[("Dashboard", "/"), ("Drafts", "/drafts")],
        )

    def _handle_draft_create(self, body: bytes | str | None) -> WebResponse:
        return self._handle_draft_save(None, body)

    def _handle_draft_post(self, draft_id_text: str, action: str, body: bytes | str | None) -> WebResponse:
        draft_id = _parse_uuid(draft_id_text)
        draft = self._drafts.get(draft_id) if draft_id else None
        if draft is None:
            return self._not_found("Draft not found")
        if action == "delete":
            _form, error = _parse_form_body(body)
            if error:
                return self._not_found("Draft not found")
            self._drafts.delete(draft.id)
            return WebResponse.redirect("/drafts")
        if action == "update":
            return self._handle_draft_save(draft, body)
        form, error = _parse_form_body(body)
        if error:
            return WebResponse.html(400, self._draft_edit_page(draft, error))
        if action == "convert":
            try:
                test_case = create_manual_test_case({
                    "name": draft.title,
                    "description": draft.body,
                    "base_url": draft.base_url or "",
                    "step_name_0": draft.title,
                    "step_action_0": draft.body,
                    "step_expected_0": "The described behavior works as expected.",
                })
                if self._test_cases is None:
                    raise TestCaseEditError("TestCase storage is not configured.")
                self._test_cases.save(test_case)
            except (TestCaseEditError, ValueError) as conversion_error:
                return WebResponse.html(400, self._draft_edit_page(draft, str(conversion_error)))
            return WebResponse.redirect(f"/test-cases/{test_case.id}/edit")
        if action == "generate":
            if self._background_authoring is None or self._authoring_service is None:
                return WebResponse.html(503, self._draft_edit_page(draft, "AI generation is unavailable. You can still create a TestCase manually."))
            try:
                validated = self._authoring_service.validate_input(draft.title, draft.body, draft.base_url or "")
                progress_id = self._background_authoring.start(
                    name=validated.name, base_url=validated.base_url, scenario=validated.scenario
                )
            except TestCaseAuthoringError as generate_error:
                return WebResponse.html(400, self._draft_edit_page(draft, str(generate_error)))
            return WebResponse.redirect(f"/test-cases/authoring-progress/{progress_id}")
        return self._not_found("Draft action not found")

    def _handle_draft_save(self, existing: Draft | None, body: bytes | str | None) -> WebResponse:
        form, error = _parse_form_body(body, max_bytes=_MAX_DRAFT_SAVE_BODY_BYTES)
        values = {key: items[0] for key, items in form.items()}
        if error:
            return WebResponse.html(400, self._draft_edit_page(existing, error, values))
        try:
            scenario = values.get("body", "").strip()
            title = values.get("title", "").strip()
            if not title and scenario:
                title = next((line.strip() for line in scenario.splitlines() if line.strip()), "")[:200]
            if not title or len(title) > 200:
                raise ValueError("Enter a Draft title of 1–200 characters.")
            if not scenario or len(scenario) > 6000:
                raise ValueError("Enter a testing idea of 1–6,000 characters.")
            base_url = values.get("base_url", "").strip()
            notes = values.get("notes", "").strip()
            if len(base_url) > 2048 or len(notes) > 6000:
                raise ValueError("The Base URL or notes are too long.")
            draft_values = {
                "id": existing.id if existing else uuid4(),
                "title": title,
                "body": scenario,
                "base_url": base_url or None,
                "notes": notes or None,
            }
            if existing is not None:
                draft_values["created_at"] = existing.created_at
            draft = Draft(**draft_values)
            self._drafts.save(draft)
        except (ValueError, TypeError) as save_error:
            return WebResponse.html(400, self._draft_edit_page(existing, str(save_error), values))
        return WebResponse.redirect(f"/drafts/{draft.id}")

    def _manual_test_case_page(
        self,
        *,
        name: str = "",
        base_url: str = "",
        scenario: str = "",
        source_draft_id: str = "",
        error: str | None = None,
    ) -> str:
        error_html = f'<p class="authoring-error" role="alert">{escape_html(error)}</p>' if error else ""
        rows = []
        for index in range(8):
            required = " required" if index == 0 else ""
            rows.append(
                f'<fieldset class="subpanel"><legend>Step {index + 1}</legend>'
                f'<div class="field"><label for="step-name-{index}">Title</label>'
                f'<input id="step-name-{index}" name="step_name_{index}" maxlength="200"{required}'
                f' value="{escape_html(name if index == 0 else "")}"></div>'
                f'<div class="field"><label for="step-action-{index}">Action / instruction</label>'
                f'<textarea id="step-action-{index}" name="step_action_{index}" maxlength="6000" rows="3"{required}>'
                f'{escape_html(scenario if index == 0 else "")}</textarea></div>'
                f'<div class="field"><label for="step-expected-{index}">Expected result</label>'
                f'<textarea id="step-expected-{index}" name="step_expected_{index}" maxlength="6000" rows="2"{required}'
                f'>{escape_html("The described behavior works as expected." if index == 0 else "")}</textarea></div>'
                '</fieldset>'
            )
        content = (
            '<header class="page-heading"><h1>Create TestCase manually</h1>'
            '<p class="lead">This path does not call an AI provider. You can add more steps after saving.</p></header>'
            + error_html + '<form method="post" action="/test-cases/manual" class="panel" data-inline-validation novalidate>'
            + (f'<input type="hidden" name="source_draft_id" value="{escape_html(source_draft_id)}">' if source_draft_id else "")
            + '<div class="field"><label for="manual-name">TestCase name</label>'
            + f'<input id="manual-name" name="name" maxlength="200" required value="{escape_html(name)}"></div>'
            + '<div class="field"><label for="manual-description">Scenario / description</label>'
            + f'<textarea id="manual-description" name="description" maxlength="6000" rows="4" required>{escape_html(scenario)}</textarea></div>'
            + '<div class="field"><label for="manual-base-url">Base URL (optional)</label>'
            + f'<input id="manual-base-url" name="base_url" maxlength="2048" value="{escape_html(base_url)}"></div>'
            + '<div class="field"><label for="manual-preconditions">Preconditions (one per line)</label>'
            + '<textarea id="manual-preconditions" name="preconditions" maxlength="180000" rows="3"></textarea></div>'
            + '<h2>Steps</h2>' + "".join(rows)
            + '<div class="actions"><button class="button primary" type="submit">Create TestCase</button>'
            + '<a class="button" href="/test-cases/new">Back to AI authoring</a></div></form>'
        )
        return self._page("Create TestCase manually", content, current="Test Cases", breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases")])

    def _handle_manual_test_case_create(self, body: bytes | str | None) -> WebResponse:
        form, error = _parse_form_body(body, max_bytes=_MAX_MANUAL_FORM_BODY_BYTES)
        values = {key: items[0] for key, items in form.items()}
        if error:
            return WebResponse.html(400, self._manual_test_case_page(error=error))
        try:
            test_case = create_manual_test_case(values)
            if self._test_cases is None:
                raise TestCaseEditError("TestCase storage is not configured.")
            self._test_cases.save(test_case)
        except (TestCaseEditError, ValueError) as create_error:
            return WebResponse.html(400, self._manual_test_case_page(
                name=values.get("name", ""), base_url=values.get("base_url", ""),
                scenario=values.get("description", ""),
                source_draft_id=values.get("source_draft_id", ""), error=str(create_error),
            ))
        return WebResponse.redirect(f"/test-cases/{test_case.id}")

    def _test_case_edit_page_response(self, test_case_id: UUID) -> WebResponse:
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        if test_case is None:
            return self._not_found("TestCase not found")
        return WebResponse.html(200, self._test_case_edit_page(test_case))

    def _test_case_edit_page(self, test_case: TestCase, error: str | None = None, submitted: dict[str, str] | None = None) -> str:
        submitted = submitted or {}
        def value(key: str, default: str) -> str:
            return escape_html(submitted.get(key, default))
        preconditions = "\n".join(item.description for item in test_case.preconditions)
        content = (
            f'<header class="page-heading"><h1>Edit {escape_html(test_case.name)}</h1>'
            '<p class="lead">Step actions stay inside their existing execution segment.</p></header>'
            + (f'<p class="authoring-error" role="alert">{escape_html(error)}</p>' if error else "")
            + f'<form method="post" action="/test-cases/{test_case.id}/edit" class="testcase-editor" data-inline-validation novalidate>'
            + '<section class="panel"><h2>Definition</h2>'
            + f'<div class="field"><label for="edit-name">Name</label><input id="edit-name" name="name" maxlength="200" value="{value("name", test_case.name)}" required></div>'
            + f'<div class="field"><label for="edit-description">Scenario / description</label><textarea id="edit-description" name="description" maxlength="6000" rows="5" required>{value("description", test_case.description)}</textarea></div>'
            + f'<div class="field"><label for="edit-base-url">Base URL</label><input id="edit-base-url" name="base_url" maxlength="2048" value="{value("base_url", test_case.base_url or "")}"></div>'
            + f'<div class="field"><label for="edit-preconditions">Preconditions (one per line)</label><textarea id="edit-preconditions" name="preconditions" maxlength="180000" rows="4">{value("preconditions", preconditions)}</textarea></div>'
            + '</section>'
        )
        for segment_index, segment in enumerate(test_case.segments):
            rows = []
            for step_index, step in enumerate(segment.steps):
                prefix = f"step_{segment_index}_{step_index}_"
                rows.append(
                    f'<li class="subpanel"><h3>Step {step.order + 1}</h3>'
                    + f'<div class="field"><label>Title</label><input name="{prefix}name" maxlength="200" value="{value(prefix + "name", step.name)}" required></div>'
                    + f'<div class="field"><label>Action / instruction</label><textarea name="{prefix}action" maxlength="6000" rows="3" required>{value(prefix + "action", step.description)}</textarea></div>'
                    + f'<div class="field"><label>Expected result</label><textarea name="{prefix}expected" maxlength="6000" rows="2" required>{value(prefix + "expected", step.expected)}</textarea></div>'
                    + '<div class="button-row">'
                    + f'<button class="button" name="operation" value="up:{segment_index}:{step_index}" type="submit">Move up</button>'
                    + f'<button class="button" name="operation" value="down:{segment_index}:{step_index}" type="submit">Move down</button>'
                    + f'<button class="button" name="operation" value="duplicate:{segment_index}:{step_index}" type="submit">Duplicate</button>'
                    + f'<button class="button" name="operation" value="delete:{segment_index}:{step_index}" type="submit"'
                    + (' disabled' if len(segment.steps) == 1 else '') + '>Delete</button></div></li>'
                )
            url_label = segment.base_url or test_case.base_url or "No URL configured"
            content += (
                f'<section class="panel"><h2>Segment {segment_index + 1}</h2>'
                f'<p class="muted">Base URL: {escape_html(url_label)}</p><ol>{"".join(rows)}</ol>'
                f'<button class="button" name="operation" value="add:{segment_index}" type="submit">Add step to this segment</button></section>'
            )
        content += (
            '<section class="panel"><div class="actions"><button class="button primary" name="operation" value="save" type="submit">Save TestCase</button>'
            f'<a class="button" href="/test-cases/{test_case.id}">Cancel</a></div></section></form>'
        )
        return self._page("Edit TestCase", content, current="Test Cases", breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases"), (test_case.name, f"/test-cases/{test_case.id}")])

    def _handle_test_case_edit(self, test_case_id_text: str, body: bytes | str | None) -> WebResponse:
        test_case_id = _parse_uuid(test_case_id_text)
        test_case = self._test_cases.get(test_case_id) if test_case_id and self._test_cases is not None else None
        if test_case is None:
            return self._not_found("TestCase not found")
        form, error = _parse_form_body(body, max_bytes=_MAX_MANUAL_FORM_BODY_BYTES)
        submitted = {key: items[0] for key, items in form.items()}
        if error:
            return WebResponse.html(400, self._test_case_edit_page(test_case, error))
        try:
            edited = edit_test_case(test_case, submitted, submitted.get("operation", "save"))
            self._test_cases.save(edited)
            self._automation_lifecycle.mark_test_case_changed(edited)
        except (TestCaseEditError, ValueError) as edit_error:
            return WebResponse.html(400, self._test_case_edit_page(test_case, str(edit_error), submitted))
        return WebResponse.redirect(f"/test-cases/{edited.id}")

    def _test_plan_view(self, test_case_id: UUID) -> WebResponse:
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        if test_case is None:
            return self._not_found("TestCase not found")
        if self._plan_store is None:
            return WebResponse.html(503, self._page("TestPlan unavailable", self._empty_state("TestPlan unavailable", "No saved plan store is configured.")))
        rows = []
        for step in test_case.steps:
            version = self._plan_store.find(step.id)
            if version is None:
                rows.append(f'<section class="panel"><h2>Step {step.order + 1}: {escape_html(step.name)}</h2><p class="muted">No saved TestPlan for this step.</p></section>')
                continue
            actions = "".join(
                '<li><code>' + escape_html(action.action) + '</code><pre>'
                + escape_html(redact_secrets(json.dumps(action.parameters, ensure_ascii=False, sort_keys=True, indent=2)))
                + '</pre></li>' for action in version.qa_test_plan.steps
            )
            rows.append(
                f'<section class="panel"><h2>Step {step.order + 1}: {escape_html(step.name)}</h2>'
                f'<p>Plan v{version.version} · {_plan_origin_label(version.origin)}</p>'
                f'<p>URL: {escape_html(redact_secrets(version.qa_test_plan.url))}</p><ol>{actions}</ol></section>'
            )
        content = (
            f'<header class="page-heading"><h1>View TestPlan: {escape_html(test_case.name)}</h1>'
            '<p class="lead">Read-only view of the current saved plans. Editing structured automation is planned for a later Automation Editor milestone.</p></header>'
            + "".join(rows)
        )
        return WebResponse.html(200, self._page("View TestPlan", content, current="Test Cases", breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases"), (test_case.name, f"/test-cases/{test_case.id}")]))

    def _handle_generate_test_case(self, body: bytes | str | None) -> WebResponse:
        form, form_error = _parse_form_body(body, max_bytes=_MAX_AUTHORING_FORM_BODY_BYTES)
        values = {
            key: form.get(key, [""])[0]
            for key in ("name", "base_url", "scenario")
        }

        dashboard_entry = form.get("authoring_entry", [""])[0] == "dashboard"
        field_errors: dict[str, str] = {}

        def error_response(status: int, message: str) -> WebResponse:
            if dashboard_entry:
                return WebResponse.html(status, self._dashboard(
                    authoring_error=message,
                    base_url=values["base_url"],
                    scenario=values["scenario"],
                    field_errors=field_errors,
                ))
            return WebResponse.html(status, self._new_test_case_page(
                message, field_errors=field_errors, **values
            ))

        if form_error is not None:
            return error_response(400, form_error)
        field_errors = _authoring_field_errors(
            values["name"], values["base_url"], values["scenario"],
            require_name=not dashboard_entry,
        )
        if field_errors:
            return error_response(400, "")
        try:
            validated = TestCaseAuthoringService.validate_input(
                **values,
                require_name=not dashboard_entry,
            )
        except TestCaseAuthoringError as error:
            return error_response(400, str(error))
        except Exception:
            return error_response(400, "Enter a valid name, Website, and scenario.")
        if self._test_cases is None:
            return error_response(503, "TestCase storage is not configured.")
        if self._authoring_service is None or self._background_authoring is None:
            return error_response(503,
                "AI generation is unavailable. Configure an LLM provider in the environment and try again.",
            )
        try:
            progress_id = self._background_authoring.start(
                name=validated.name,
                base_url=validated.base_url,
                scenario=validated.scenario,
            )
        except Exception:
            logger.error("Could not start authoring request")
            return error_response(503, "Authoring could not be started. Try again later.")
        return WebResponse.redirect(f"/test-cases/authoring-progress/{progress_id}")

    def _handle_authoring_retry(
        self,
        progress_id: str,
        body: bytes | str | None,
    ) -> WebResponse:
        _form, form_error = _parse_form_body(body)
        if form_error is not None:
            return self._not_found("Authoring progress is no longer available.")
        retry_data = self._progress_store.get_authoring_retry_data(progress_id)
        if retry_data is None:
            return self._not_found("Authoring progress is no longer available.")
        if self._authoring_service is None or self._background_authoring is None:
            return WebResponse.html(503, self._page(
                "Authoring unavailable",
                '<div class="error-state"><p>AI generation is unavailable. Try again later.</p></div>',
            ))
        name, base_url, scenario, source_draft_token = retry_data
        try:
            validated = self._authoring_service.validate_input(name, scenario, base_url)
        except TestCaseAuthoringError as error:
            return WebResponse.html(400, self._new_test_case_page(
                str(error), name=name, base_url=base_url, scenario=scenario
            ))
        new_progress_id = self._background_authoring.start(
            name=validated.name,
            base_url=validated.base_url,
            scenario=validated.scenario,
            source_draft_token=source_draft_token,
        )
        return WebResponse.redirect(
            f"/test-cases/authoring-progress/{new_progress_id}"
        )

    def _handle_authoring_failure_action(
        self, progress_id: str, action: str, body: bytes | str | None
    ) -> WebResponse:
        _form, error = _parse_form_body(body)
        if error:
            return self._not_found("Authoring progress is no longer available.")
        retry_data = self._progress_store.get_authoring_retry_data(progress_id)
        if retry_data is None:
            return self._not_found("Authoring progress is no longer available.")
        name, base_url, scenario, _source_draft_token = retry_data
        if action == "save-draft":
            draft = Draft(title=name or "Untitled testing idea", body=scenario, base_url=base_url or None)
            self._drafts.save(draft)
            return WebResponse.redirect(f"/drafts/{draft.id}")
        try:
            test_case = create_manual_test_case({
                "name": name,
                "description": scenario,
                "base_url": base_url,
                "step_name_0": "Review scenario",
                "step_action_0": scenario,
                "step_expected_0": "The described behavior works as expected.",
            })
            if self._test_cases is None:
                return WebResponse.html(503, self._manual_test_case_page(
                    name=name, base_url=base_url, scenario=scenario,
                    error="TestCase storage is not configured.",
                ))
            self._test_cases.save(test_case)
        except (TestCaseEditError, ValueError) as create_error:
            return WebResponse.html(400, self._manual_test_case_page(
                name=name, base_url=base_url, scenario=scenario, error=str(create_error),
            ))
        return WebResponse.redirect(f"/test-cases/{test_case.id}/edit")

    def _handle_draft_action(
        self, token: str, action: str, body: bytes | str | None
    ) -> WebResponse:
        form, form_error = _parse_form_body(
            body,
            max_bytes=(
                _MAX_DRAFT_SAVE_BODY_BYTES if action == "save" else _MAX_FORM_BODY_BYTES
            ),
        )
        if form_error is not None:
            return self._not_found("Draft not found or expired")
        if action == "cancel":
            self._draft_store.take(token)
            return WebResponse.redirect("/test-cases")
        draft = self._draft_store.get(token)
        if draft is None:
            return self._not_found("Draft not found or expired")
        if action == "regenerate":
            if self._authoring_service is None:
                return WebResponse.html(503, self._draft_review_html(
                    draft, token, "AI generation is unavailable. Configure an LLM provider and try again."
                ))
            try:
                validated = self._authoring_service.validate_input(
                    draft.authoring_name or draft.test_case.name,
                    draft.authoring_scenario or draft.test_case.description,
                    draft.authoring_base_url or draft.test_case.base_url or "",
                )
            except TestCaseAuthoringError as error:
                return WebResponse.html(400, self._draft_review_html(draft, token, str(error)))
            except Exception:
                return WebResponse.html(503, self._draft_review_html(
                    draft, token, "Authoring could not be started. Try again later."
                ))
            if self._background_authoring is None:
                return WebResponse.html(503, self._draft_review_html(
                    draft, token, "AI generation is unavailable. Try again later."
                ))
            progress_id = self._background_authoring.start(
                name=validated.name,
                base_url=validated.base_url,
                scenario=validated.scenario,
                source_draft_token=token,
            )
            return WebResponse.redirect(f"/test-cases/authoring-progress/{progress_id}")
        if action != "save":
            return self._not_found("Draft action not found")
        if self._test_cases is None:
            return WebResponse.html(503, self._not_found("TestCase storage is not configured.").body.decode("utf-8"))
        edits = merge_test_case_edits(
            draft.test_case,
            {key: items[0] for key, items in form.items()},
        )
        if edits.errors:
            return WebResponse.html(400, self._draft_review_html(
                draft,
                token,
                values=edits.values,
                field_errors=edits.errors,
            ))
        if edits.test_case is None:
            return self._not_found("Draft not found or expired")
        # Validation runs before consumption so users can correct the form.
        # Taking the token only after validation still makes concurrent/repeated
        # saves single-use.
        consumed = self._draft_store.take(token)
        if consumed is None:
            return self._not_found("Draft not found or expired")
        try:
            self._test_cases.save(edits.test_case)
        except Exception:
            return WebResponse.html(500, self._page(
                "TestCase not saved",
                '<div class="error-state"><h1>TestCase not saved</h1>'
                '<p>The reviewed TestCase could not be saved. Generate a new draft and try again.</p>'
                '<a class="button" href="/test-cases">Return to TestCases</a></div>',
            ))
        if self._llm_usage is not None:
            for workflow_id in consumed.usage_workflow_ids:
                self._llm_usage.associate_workflow(
                    workflow_id,
                    edits.test_case.id,
                    edits.test_case.public_id,
                )
        return WebResponse.redirect(f"/test-cases/{edits.test_case.id}")

    def _draft_review_page(self, token: str, error: str | None = None) -> WebResponse:
        draft = self._draft_store.get(token)
        if draft is None:
            return self._not_found("Draft not found or expired")
        return WebResponse.html(200, self._draft_review_html(draft, token, error))

    def _draft_review_html(
        self,
        draft: TestCaseDraft,
        token: str,
        error: str | None = None,
        *,
        values: dict[str, str] | None = None,
        field_errors: dict[str, str] | None = None,
    ) -> str:
        test_case = draft.test_case
        original_values = editable_test_case_values(test_case)
        values = values or original_values
        field_errors = field_errors or {}
        user_modified = any(
            values.get(key, value) != value for key, value in original_values.items()
        )

        def editable_field(key: str, label: str, *, textarea: bool, maximum: int) -> str:
            field_id = "edit-" + key.replace(".", "-")
            current_value = values.get(key, original_values[key])
            original_value = original_values[key]
            error_message = field_errors.get(key)
            error_html = (
                f'<span class="field-error" role="alert">{escape_html(error_message)}</span>'
                if error_message else ""
            )
            attributes = (
                f'id="{field_id}" name="{escape_html(key)}" '
                f'data-editable-field data-original-value="{escape_html(original_value)}" '
                f'maxlength="{maximum}"'
            )
            if textarea:
                control = f'<textarea {attributes}>{escape_html(current_value)}</textarea>'
            else:
                control = f'<input type="text" {attributes} value="{escape_html(current_value)}">'
            return (
                f'<div class="field"><label for="{field_id}">{escape_html(label)}</label>'
                f'{control}{error_html}</div>'
            )

        preconditions = "".join(
            '<li>' + editable_field(
                f"precondition.{index}.description",
                f"Precondition {index + 1}",
                textarea=True,
                maximum=6000,
            ) + "</li>"
            for index, _item in enumerate(test_case.preconditions)
        )
        segments = []
        step_number = 0
        for segment_index, segment in enumerate(test_case.segments):
            step_rows = []
            for step_index, _step in enumerate(segment.steps):
                step_number += 1
                prefix = f"segment.{segment_index}.step.{step_index}"
                fields = (
                    editable_field(f"{prefix}.name", "Title", textarea=False, maximum=200)
                    + editable_field(f"{prefix}.description", "Action / Description", textarea=True, maximum=6000)
                    + editable_field(f"{prefix}.expected", "Expected Result", textarea=True, maximum=6000)
                )
                step_rows.append(f'<li><h4>Step {step_number}</h4>{fields}</li>')
            steps = "".join(step_rows)
            segments.append(
                f'<section class="subpanel"><h3>Segment {segment_index + 1}</h3><ol>{steps}</ol></section>'
            )
        error_html = (
            f'<div class="error-state"><p>{escape_html(error)}</p></div>' if error else ""
        )
        form_error_html = (
            f'<div class="error-state"><p>{escape_html(field_errors["form"])}</p></div>'
            if "form" in field_errors else ""
        )
        edited_badge = (
            '<span class="badge neutral edited-indicator" data-edited-indicator>Edited</span>'
            if user_modified else
            '<span class="badge neutral edited-indicator" data-edited-indicator hidden>Edited</span>'
        )
        content = (
            '<header class="page-heading"><h1>Review TestCase</h1>'
            '<p class="lead">Review and edit the generated definition before saving it. '
            + edited_badge + '</p></header>'
            + error_html
            + form_error_html
            + f'<form method="post" class="testcase-editor" action="/test-cases/review/{escape_html(token)}/save" data-testcase-editor>'
            + '<section class="panel"><h2>Definition</h2>'
            + editable_field("name", "TestCase name", textarea=False, maximum=200)
            + f'<p><strong>Base URL:</strong> {escape_html(test_case.base_url or "")}</p>'
            + editable_field("description", "Description / scenario", textarea=True, maximum=6000)
            + '</section>'
            + '<section class="panel"><h2>Preconditions</h2>'
            + (f'<ul>{preconditions}</ul>' if preconditions else '<p class="muted">No preconditions proposed.</p>')
            + '</section><section class="panel"><h2>Steps</h2>'
            + "".join(segments)
            + '</section><div class="actions">'
            + '<button class="button primary" type="submit">Save Test Case</button></div></form>'
            + '<div class="actions">'
            + f'<form method="post" action="/test-cases/review/{escape_html(token)}/regenerate" data-authoring-form>'
            + '<button class="button" type="submit">Generate Again</button></form>'
            + f'<form method="post" action="/test-cases/review/{escape_html(token)}/cancel">'
            + '<button class="button" type="submit">Cancel</button></form></div>'
        )
        return self._page(
            "Review TestCase", content, current="Test Cases",
            breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases")],
        )

    def _run_error(self, test_case_id: UUID, message: str, status: int) -> WebResponse:
        content = (
            '<div class="error-state"><h1>Run not started</h1>'
            f"<p>{escape_html(message)}</p>"
            f'<a class="button" href="/test-cases/{test_case_id}">Return to TestCase</a></div>'
        )
        return WebResponse.html(status, self._page("Run not started", content))

    def _dashboard(
        self,
        *,
        authoring_error: str | None = None,
        base_url: str = "",
        scenario: str = "",
        field_errors: dict[str, str] | None = None,
    ) -> str:
        field_errors = field_errors or {}
        records = self._run_history.list_recent(_PAGE_LIMIT)
        counts = {
            "Total runs": len(records),
            "Passed": sum(record.status.value == "PASSED" for record in records),
            "Failed": sum(record.status.value == "FAILED" for record in records),
            "Automation": sum(record.workflow_type == WorkflowType.AUTOMATION for record in records),
            "Validation": sum(record.workflow_type == WorkflowType.VALIDATION for record in records),
            "Regression": sum(record.workflow_type == WorkflowType.REGRESSION for record in records),
            "Product failures": sum(record.outcome == "PRODUCT_FAILURE" for record in records),
            "Automation drift": sum(record.outcome == "AUTOMATION_DRIFT" for record in records),
            "Infrastructure errors": sum(record.outcome == "INFRASTRUCTURE_ERROR" for record in records),
        }
        cards = "".join(
            f'<article class="card"><div class="card-label">{escape_html(label)}</div>'
            f'<div class="card-value">{count}</div></article>'
            for label, count in counts.items()
        )
        recent = records[:12]
        entry_error = (
            f'<p class="authoring-error" id="authoring-entry-error" role="alert">'
            f'{escape_html(authoring_error)}</p>'
            if authoring_error else ""
        )
        error_description = ' aria-describedby="authoring-entry-error"' if authoring_error else ""
        website_error = field_errors.get("base_url")
        scenario_error = field_errors.get("scenario")
        website_invalid = _inline_invalid_attrs("dashboard-website-error", website_error)
        scenario_invalid = _inline_invalid_attrs("dashboard-scenario-error", scenario_error)
        authoring_entry = (
            '<section class="panel authoring-entry" aria-labelledby="authoring-entry-title">'
            '<h2 id="authoring-entry-title">What do you want to test?</h2>'
            '<p class="muted">Describe a web scenario in natural language. AI turns it into a reusable automated test.</p>'
            '<form method="post" action="/test-cases/generate" data-authoring-form data-inline-validation novalidate>'
            '<input type="hidden" name="authoring_entry" value="dashboard">'
            + entry_error
            + '<div class="field"><label for="dashboard-base-url">Website</label>'
            + f'<input id="dashboard-base-url" name="base_url" type="url" maxlength="2048" required value="{escape_html(base_url)}" placeholder="https://example.com"{error_description}{website_invalid}></div>'
            + _inline_error_html("dashboard-website-error", "dashboard-base-url", website_error)
            + '<div class="field"><label for="dashboard-scenario">Scenario</label>'
            + f'<textarea id="dashboard-scenario" name="scenario" rows="8" maxlength="6000" required placeholder="Describe the behavior you want to verify…"{error_description}{scenario_invalid}>{escape_html(scenario)}</textarea></div>'
            + _inline_error_html("dashboard-scenario-error", "dashboard-scenario", scenario_error)
            + self._authoring_voice_controls("dashboard-scenario")
            + '<button class="button primary authoring-submit" type="submit">Generate Test with AI</button>'
            + '</form>'
            + '<form method="post" action="/test-cases/manual/prepare">'
            + f'<input type="hidden" name="base_url" value="{escape_html(base_url)}">'
            + f'<input type="hidden" name="description" value="{escape_html(scenario)}">'
            + '<button class="button" type="submit">Create manually</button></form>'
            + '<form method="post" action="/drafts">'
            + f'<input type="hidden" name="body" value="{escape_html(scenario)}">'
            + f'<input type="hidden" name="base_url" value="{escape_html(base_url)}">'
            + '<button class="button" type="submit">Save as Draft</button></form>'
            + '<ol class="product-flow" aria-label="Test lifecycle">'
            + '<li>Describe</li><li>Generate</li><li>Run</li><li>Reuse</li></ol>'
            + '</section>'
        )
        content = (
            '<header class="page-heading"><h1>AI QA Agent</h1></header>'
            + authoring_entry
            + self._recent_drafts_panel()
            + f'<section aria-label="Run summary"><div class="summary-grid">{cards}</div>'
            '<p class="muted">Summary of the latest recorded history.</p></section>'
            '<section class="panel"><div class="section-heading"><h2>Recent runs</h2>'
            '<a class="button" href="/runs">View all runs</a></div>'
            + self._runs_table(recent)
            + ("" if recent else self._empty_state(
                "No runs recorded yet.",
                "Run a workflow to see results here.",
            ))
            + '</section><section class="panel"><div class="section-heading"><h2>TestCases</h2>'
            '<a class="button" href="/test-cases">Browse TestCases</a></div>'
            '<p class="muted">Browse saved definitions, including TestCases that have not run yet.</p></section>'
        )
        return self._page("Dashboard", content, current="Dashboard")

    def _authoring_voice_controls(self, scenario_id: str) -> str:
        return (
            '<div class="voice-controls" data-voice-control>'
            f'<button class="button voice-button" type="button" data-voice-button '
            f'data-voice-target="{escape_html(scenario_id)}" aria-pressed="false" disabled>'
            '<span aria-hidden="true">&#127908;</span> '
            '<span data-voice-label>Speak scenario</span></button>'
            '<p class="voice-status muted" data-voice-status role="status" aria-live="polite"></p>'
            '</div>'
            '<p class="voice-privacy muted">Your browser handles speech recognition; this app receives recognized text only and never receives audio.</p>'
        )

    def _new_test_case_page(
        self,
        error: str | None = None,
        *,
        name: str = "",
        base_url: str = "",
        scenario: str = "",
        field_errors: dict[str, str] | None = None,
    ) -> str:
        field_errors = field_errors or {}
        error_html = (
            f'<p class="authoring-error" id="new-case-authoring-error" role="alert">{escape_html(error)}</p>'
            if error else ""
        )
        name_invalid = _inline_invalid_attrs("case-name-error", field_errors.get("name"))
        website_invalid = _inline_invalid_attrs("case-website-error", field_errors.get("base_url"))
        scenario_invalid = _inline_invalid_attrs("case-scenario-error", field_errors.get("scenario"))
        content = (
            '<header class="page-heading"><h1>New Test Case</h1>'
            '<p class="lead">Describe the scenario for AI authoring, or create a TestCase yourself.</p></header>'
            + '<section class="panel authoring-entry"><form method="post" action="/test-cases/generate" data-authoring-form data-inline-validation novalidate>'
            + '<input type="hidden" name="authoring_entry" value="new">'
            + error_html
            + '<div class="field"><label for="case-name">Name</label>'
            + f'<input id="case-name" name="name" maxlength="200" required value="{escape_html(name)}"{name_invalid}></div>'
            + _inline_error_html("case-name-error", "case-name", field_errors.get("name"))
            + '<div class="field"><label for="case-base-url">Base URL</label>'
            + f'<input id="case-base-url" name="base_url" type="url" maxlength="2048" required value="{escape_html(base_url)}" placeholder="https://example.com"{website_invalid}></div>'
            + _inline_error_html("case-website-error", "case-base-url", field_errors.get("base_url"))
            + '<div class="field"><label for="case-scenario">Scenario</label>'
            + f'<textarea id="case-scenario" name="scenario" rows="8" maxlength="6000" required placeholder="Describe the behavior you want to verify…"{scenario_invalid}>{escape_html(scenario)}</textarea></div>'
            + _inline_error_html("case-scenario-error", "case-scenario", field_errors.get("scenario"))
            + self._authoring_voice_controls("case-scenario")
            + '<button class="button primary" type="submit">Generate Test with AI</button>'
            + '</form></section>'
            + '<section class="panel"><h2>Create without AI</h2>'
            + '<p class="muted">Manual creation and editing remain available when providers are unconfigured or unavailable.</p>'
            + '<form method="post" action="/test-cases/manual/prepare">'
            + f'<input type="hidden" name="name" value="{escape_html(name)}">'
            + f'<input type="hidden" name="base_url" value="{escape_html(base_url)}">'
            + f'<input type="hidden" name="description" value="{escape_html(scenario)}">'
            + '<button class="button" type="submit">Create manually with these details</button></form> '
            + '<form method="post" action="/drafts">'
            + f'<input type="hidden" name="title" value="{escape_html(name)}">'
            + f'<input type="hidden" name="body" value="{escape_html(scenario)}">'
            + f'<input type="hidden" name="base_url" value="{escape_html(base_url)}">'
            + '<button class="button" type="submit">Save as Draft</button></form></section>'
            + self._recent_drafts_panel()
        )
        return self._page(
            "New Test Case", content, current="Test Cases",
            breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases")],
        )

    def _test_case_list(self) -> str:
        records = self._run_history.list_recent(_PAGE_LIMIT)
        latest_by_case: dict[UUID, RunHistoryRecord] = {}
        grouped: dict[UUID, list[RunHistoryRecord]] = {}
        for record in records:
            grouped.setdefault(record.test_case_id, []).append(record)
            latest_by_case.setdefault(record.test_case_id, record)
        rows = []
        if self._test_cases is not None:
            cases = self._test_cases.list()
        else:
            cases = []
        if self._test_cases is not None:
            for test_case in cases:
                test_case_id = test_case.id
                history = grouped.get(test_case_id, [])
                latest = latest_by_case.get(test_case_id)
                status = badge(latest.status.value) if latest else badge("NOT RUN")
                automation_status = badge(
                    _automation_status_label(self._automation_lifecycle.status(test_case)),
                    "workflow",
                )
                outcome = _outcome_badge(latest) if latest else '<span class="muted">—</span>'
                last_run = escape_html(format_timestamp(latest.started_at)) if latest else "—"
                latest_workflow = (
                    badge(latest.workflow_type.value, "workflow")
                    if latest else '<span class="muted">—</span>'
                )
                rows.append(
                    "<tr>"
                    + (
                        f'<td><input type="checkbox" name="test_case_id" value="{test_case_id}" '
                        f'aria-label="Select {escape_html(test_case.public_id or test_case.name)}"></td>'
                        if self._testplan_exports is not None else ""
                    )
                    + f'<td><div class="status-line"><span class="id-code">{escape_html(test_case.public_id or "")}</span>'
                    f'<a href="/test-cases/{test_case_id}">{escape_html(test_case.name)}</a></div>'
                    f'<details><summary>Technical ID</summary><code>{escape_html(test_case_id)}</code></details></td>'
                    f"<td>{automation_status}</td><td>{status}</td><td>{outcome}</td><td>{last_run}</td>"
                    f"<td>{len(history)}</td><td>{latest_workflow}</td>"
                    '<td><div class="test-case-actions">'
                    + f'<a class="button" href="/test-cases/{test_case_id}">Open</a>'
                    + f'<a class="button" href="/test-cases/{test_case_id}/edit">Edit</a>'
                    + (f'<a class="button" href="/test-cases/{test_case_id}/export/portable">Export</a>' if self._testplan_exports is not None else "")
                    + '</div></td></tr>'
                )
        else:
            for test_case_id, latest in latest_by_case.items():
                history = grouped[test_case_id]
                rows.append(
                    "<tr>"
                    f'<td><div class="status-line"><span class="id-code">{escape_html(latest.test_case_public_id or "")}</span>'
                    f'<a href="/test-cases/{test_case_id}">{escape_html(latest.test_case_name)}</a></div>'
                    f'<details><summary>Technical ID</summary><code>{escape_html(test_case_id)}</code></details></td>'
                    '<td><span class="muted">Unavailable</span></td>'
                    f'<td>{badge(latest.status.value)}</td>'
                    f'<td>{_outcome_badge(latest)}</td>'
                    f'<td>{escape_html(format_timestamp(latest.started_at))}</td>'
                    f'<td>{len(history)}</td>'
                    f'<td>{badge(latest.workflow_type.value, "workflow")}</td>'
                    f'<td><a class="button" href="/test-cases/{test_case_id}">Open</a></td>'
                    "</tr>"
                )
        selection_column = '<th scope="col"><label><input type="checkbox" data-select-all> Select visible</label></th>' if self._testplan_exports is not None else ""
        table = (
            '<div class="table-wrap"><table class="test-case-list"><thead><tr>' + selection_column + '<th>TestCase</th><th>Automation</th><th>Latest status</th>'
            '<th>Latest result</th><th>Last run</th><th>Runs</th><th>Latest workflow</th><th>Actions</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
        bulk_form = (
            '<form method="post" action="/test-cases/export" class="export-selection">'
            '<label for="bulk-export-format">Export selected as</label>'
            '<select id="bulk-export-format" name="target">'
            '<option value="portable">Portable JSON bundle</option>'
            '<option value="python">Python Playwright project</option>'
            '<option value="typescript">TypeScript Playwright project</option>'
            '<option value="csharp">C# Playwright project</option>'
            '</select><button class="button" type="submit">Export selected</button>'
            + table + '</form>'
            if rows and self._testplan_exports is not None else ""
        )
        content = (
            '<header class="page-heading section-heading"><div><h1>Test Cases</h1>'
            '<p class="lead">Saved TestCase definitions and their run history.</p></div>'
            '<a class="button primary" href="/test-cases/new">+ New Test Case</a></header>'
            '<section class="panel"><h2>TestCases</h2>'
            + (bulk_form if self._testplan_exports is not None and rows else table if rows else self._empty_state(
                "No TestCases found.", "Create a TestCase from a natural-language scenario to get started."
            ))
            + "</section>"
        )
        return self._page("Test Cases", content, current="Test Cases", breadcrumbs=[("Dashboard", "/")])

    def _test_case_export(self, test_case_id: UUID, target: str) -> WebResponse:
        if self._testplan_exports is None:
            return self._not_found("TestCase export is unavailable.")
        try:
            if target == "portable":
                content, filename = self._testplan_exports.portable_json(test_case_id)
                return _download_response(content.encode("utf-8"), filename, "application/json; charset=utf-8")
            if target in {"python", "typescript", "csharp"}:
                content, filename = self._testplan_exports.source(test_case_id, target)
                return _download_response(content.encode("utf-8"), filename, "text/plain; charset=utf-8")
            if target.endswith("-zip"):
                language = target.removesuffix("-zip")
                exportable = self._testplan_exports.get(test_case_id)
                if language == "portable":
                    content = portable_zip([exportable])
                    filename = f"{exportable.test_case.public_id or 'testcase'}-portable.zip"
                elif language in {"python", "typescript", "csharp"}:
                    content = project_zip(language, [exportable])
                    filename = f"{exportable.test_case.public_id or 'testcase'}-{language}.zip"
                else:
                    raise TestPlanExportError("Choose a supported export format.")
                return _download_response(content, filename, "application/zip")
            return self._not_found("Export format not found.")
        except TestPlanExportError as error:
            return _export_error_response(str(error), 409)
        except Exception as error:
            logger.warning("TestCase export failed safely (%s)", type(error).__name__)
            return _export_error_response("The saved automation could not be exported.", 500)

    def _handle_bulk_export(self, body: bytes | str | None) -> WebResponse:
        if self._testplan_exports is None:
            return _export_error_response("TestCase export is unavailable.", 503)
        form, error = _parse_form_body(
            body, max_bytes=64 * 1024, allow_repeated_fields={"test_case_id"}
        )
        if error:
            return _export_error_response(error, 400)
        raw_ids = form.get("test_case_id", [])
        test_case_ids = [_parse_uuid(value) for value in raw_ids]
        if not raw_ids or any(value is None for value in test_case_ids):
            return _export_error_response("Select one or more valid TestCases.", 400)
        target = form.get("target", [""])[0]
        try:
            content, filename = self._testplan_exports.bulk_zip(test_case_ids, target)
        except TestPlanExportError as export_error:
            return _export_error_response(str(export_error), 409)
        except Exception as export_error:
            logger.warning("Bulk TestCase export failed safely (%s)", type(export_error).__name__)
            return _export_error_response("The selected automation could not be exported.", 500)
        return _download_response(content, filename, "application/zip")

    def _test_suites_page(self) -> str:
        suites = self._test_suites.list()
        rows = "".join(
            "<tr>"
            f'<td><a href="/test-suites/{suite.id}">{escape_html(suite.name)}</a>'
            f'<div class="muted">{escape_html(suite.description)}</div></td>'
            f'<td>{len(self._test_suites.members(suite.id))}</td>'
            f'<td>{escape_html(format_timestamp(suite.updated_at))}</td>'
            f'<td><a class="button" href="/test-suites/{suite.id}/export?format=portable">Export</a></td>'
            "</tr>"
            for suite in suites
        )
        content = (
            '<header class="page-heading"><h1>Test Suites</h1>'
            '<p class="lead">Group saved TestCases for reusable organization and export.</p></header>'
            '<section class="panel"><h2>Create Test Suite</h2>'
            '<form method="post" action="/test-suites" class="suite-form" data-inline-validation novalidate>'
            '<div class="field"><label for="suite-name">Name</label><input id="suite-name" name="name" maxlength="120" required></div>'
            '<div class="field"><label for="suite-description">Description</label><textarea id="suite-description" name="description" maxlength="1000" rows="3"></textarea></div>'
            '<button class="button primary" type="submit">Create suite</button></form></section>'
            '<section class="panel"><h2>Saved Test Suites</h2>'
            + (
                '<div class="table-wrap"><table><thead><tr><th>Name</th><th>TestCases</th><th>Updated</th><th>Export</th></tr></thead><tbody>'
                + rows + '</tbody></table></div>'
                if suites else self._empty_state("No Test Suites yet.", "Create a suite to group related TestCases.")
            )
            + '</section>'
        )
        return self._page("Test Suites", content, current="Test Suites", breadcrumbs=[("Dashboard", "/")])

    def _test_suite_page(self, suite_id: UUID) -> WebResponse:
        suite = self._test_suites.get(suite_id)
        if suite is None:
            return self._not_found("Test Suite not found.")
        members = self._test_suites.members(suite_id)
        member_ids = {case.id for case in members}
        available = [case for case in (self._test_cases.list() if self._test_cases else []) if case.id not in member_ids]
        member_rows = []
        for index, case in enumerate(members):
            automation_status = self._automation_lifecycle.status(case)
            automation_label = _automation_status_label(automation_status)
            automation_tone = "success" if automation_status == AutomationStatus.AUTOMATION_READY else "workflow"
            member_rows.append(
                '<li class="suite-member"><div class="suite-member-main">'
                f'<span class="id-code">{escape_html(case.public_id or "")}</span> '
                f'<a href="/test-cases/{case.id}">{escape_html(case.name)}</a>'
                f'{badge(automation_label, automation_tone)}</div>'
                '<div class="suite-member-actions">'
                + _suite_member_form(suite_id, case.id, "move-up", "↑", accessible_label=f"Move {case.public_id or case.name} up", disabled=index == 0)
                + _suite_member_form(suite_id, case.id, "move-down", "↓", accessible_label=f"Move {case.public_id or case.name} down", disabled=index == len(members) - 1)
                + _suite_member_form(suite_id, case.id, "remove", "Remove", accessible_label=f"Remove {case.public_id or case.name}")
                + '</div></li>'
            )
        options = "".join(
            f'<option value="{case.id}">{escape_html(case.public_id or "")} · {escape_html(case.name)}</option>'
            for case in available
        )
        add_form = (
            '<form method="post" action="/test-suites/' + str(suite_id) + '/members/add" class="inline-form" data-inline-validation novalidate>'
            '<label for="suite-member">Add TestCase</label><select id="suite-member" name="test_case_id" required>'
            '<option value="">Choose a TestCase</option>' + options
            + '</select><button class="button" type="submit">Add</button></form>'
            if available else '<p class="muted">All available TestCases are already in this suite.</p>'
        )
        content = (
            '<header class="page-heading"><p class="eyebrow">Test Suite</p>'
            f'<h1>{escape_html(suite.name)}</h1><p class="lead">{escape_html(suite.description)}</p></header>'
            '<section class="panel"><div class="section-heading"><h2>Export suite</h2><div class="button-row">'
            f'<a class="button" href="/test-suites/{suite_id}/export?format=portable">Portable JSON ZIP</a>'
            f'<a class="button" href="/test-suites/{suite_id}/export?format=python">Python ZIP</a>'
            f'<a class="button" href="/test-suites/{suite_id}/export?format=typescript">TypeScript ZIP</a>'
            f'<a class="button" href="/test-suites/{suite_id}/export?format=csharp">C# ZIP</a>'
            '</div></div></section>'
            '<section class="panel"><h2>Edit suite</h2>'
            f'<form method="post" action="/test-suites/{suite_id}/update" class="suite-form" data-inline-validation novalidate>'
            f'<div class="field"><label for="suite-name">Name</label><input id="suite-name" name="name" maxlength="120" required value="{escape_html(suite.name)}"></div>'
            f'<div class="field"><label for="suite-description">Description</label><textarea id="suite-description" name="description" maxlength="1000" rows="3">{escape_html(suite.description)}</textarea></div>'
            '<button class="button" type="submit">Save changes</button></form></section>'
            '<section class="panel"><h2>Suite TestCases</h2>' + add_form
            + (f'<ol class="suite-members">{"".join(member_rows)}</ol>' if member_rows else self._empty_state("This suite is empty.", "Add saved TestCases above."))
            + '</section>'
        )
        return WebResponse.html(200, self._page(
            suite.name, content, current="Test Suites",
            breadcrumbs=[("Dashboard", "/"), ("Test Suites", "/test-suites")],
        ))

    def _handle_test_suite_create(self, body: bytes | str | None) -> WebResponse:
        form, error = _parse_form_body(body)
        if error:
            return _export_error_response(error, 400)
        try:
            suite = self._test_suites.create(form.get("name", [""])[0], form.get("description", [""])[0])
        except ValueError as suite_error:
            return _export_error_response(str(suite_error), 400)
        return WebResponse.redirect(f"/test-suites/{suite.id}")

    def _handle_test_suite_update(self, raw_suite_id: str, body: bytes | str | None) -> WebResponse:
        suite_id = _parse_uuid(raw_suite_id)
        if suite_id is None:
            return self._not_found("Test Suite not found.")
        form, error = _parse_form_body(body)
        if error:
            return _export_error_response(error, 400)
        try:
            self._test_suites.update(suite_id, form.get("name", [""])[0], form.get("description", [""])[0])
        except (KeyError, ValueError) as suite_error:
            return _export_error_response(str(suite_error), 400)
        return WebResponse.redirect(f"/test-suites/{suite_id}")

    def _handle_test_suite_members(self, raw_suite_id: str, operation: str, body: bytes | str | None) -> WebResponse:
        suite_id = _parse_uuid(raw_suite_id)
        if suite_id is None:
            return self._not_found("Test Suite not found.")
        form, error = _parse_form_body(body)
        if error:
            return _export_error_response(error, 400)
        case_id = _parse_uuid(form.get("test_case_id", [""])[0])
        try:
            if case_id is None:
                raise ValueError("Choose a valid TestCase.")
            if operation == "add":
                self._test_suites.add_member(suite_id, case_id)
            elif operation == "remove":
                self._test_suites.remove_member(suite_id, case_id)
            elif operation == "move-up":
                self._test_suites.move_member(suite_id, case_id, -1)
            elif operation == "move-down":
                self._test_suites.move_member(suite_id, case_id, 1)
            else:
                return self._not_found("Test Suite operation not found.")
        except (KeyError, ValueError) as suite_error:
            return _export_error_response(str(suite_error), 400)
        return WebResponse.redirect(f"/test-suites/{suite_id}")

    def _test_suite_export(self, suite_id: UUID, query: dict[str, list[str]]) -> WebResponse:
        if self._test_suites.get(suite_id) is None or self._testplan_exports is None:
            return self._not_found("Test Suite export is unavailable.")
        suite = self._test_suites.get(suite_id)
        try:
            ids = self._test_suites.member_ids(suite_id)
            target = query.get("format", ["portable"])[0].casefold()
            if target not in {"portable", "python", "typescript", "csharp"}:
                raise TestPlanExportError("Choose Portable JSON, Python, TypeScript, or C# export.")
            exports = []
            blockers = []
            for case_id in ids:
                try:
                    exports.append(self._testplan_exports.get(case_id))
                except TestPlanExportError as error:
                    if not error.blockers:
                        raise
                    blockers.extend(error.blockers)
            if blockers:
                content = _suite_export_blocker_content(suite_id, tuple(blockers))
                return WebResponse.html(409, self._page(
                    "Suite export needs attention", content, current="Test Suites",
                    breadcrumbs=[("Dashboard", "/"), ("Test Suites", "/test-suites"), (suite.name, f"/test-suites/{suite_id}")],
                ))
            if target == "portable":
                content = portable_zip(exports, suite_name=suite.name)
                filename = f"{_safe_download_slug(suite.name)}-portable.zip"
            else:
                content = project_zip(target, exports, suite_name=suite.name)
                filename = f"{_safe_download_slug(suite.name)}-{target}.zip"
        except TestPlanExportError as export_error:
            return _export_error_response(str(export_error), 409)
        except Exception as export_error:
            logger.warning("Test Suite export failed safely (%s)", type(export_error).__name__)
            return _export_error_response("The saved suite automation could not be exported.", 500)
        return _download_response(content, filename, "application/zip")

    def _runs_page(self, query: dict[str, list[str]]) -> str:
        workflow = _query_choice(query, "workflow", _WORKFLOWS)
        status = _query_choice(query, "status", _STATUSES)
        failure_type = _query_choice(query, "failure_type", _FAILURE_TYPES)
        records = self._run_history.list_recent(_PAGE_LIMIT)
        filtered = [
            record for record in records
            if (workflow == "ALL" or record.workflow_type.value == workflow)
            and (status == "ALL" or record.status.value == status)
            and (failure_type == "ALL" or record.outcome == failure_type)
        ]
        content = (
            '<header class="page-heading"><h1>Runs</h1>'
            f'<p class="lead">Showing {len(filtered)} of {len(records)} recent runs.</p></header>'
            '<section class="panel"><h2>Filter runs</h2>'
            f'<form class="filters" method="get" action="/runs">'
            f'{_select("workflow", "Workflow", workflow, ["ALL", *_sort_values(_WORKFLOWS)])}'
            f'{_select("status", "Status", status, ["ALL", "PASSED", "FAILED"])}'
            f'{_select("failure_type", "Failure type", failure_type, ["ALL", *_sort_values(_FAILURE_TYPES)])}'
            '<button class="button primary" type="submit">Apply filters</button>'
            '<a class="button" href="/runs">Clear</a></form></section>'
            '<section class="panel"><h2>Run history</h2>'
            + (self._runs_table(filtered) if filtered else self._empty_state(
                "No matching runs.", "Change or clear the filters to see more history."
            ))
            + "</section>"
        )
        return self._page("Runs", content, current="Runs", breadcrumbs=[("Dashboard", "/")])

    def _runs_table(self, records: list[RunHistoryRecord]) -> str:
        if not records:
            return ""
        rows = []
        for record in records:
            status_html = badge(record.status.value)
            outcome_html = _outcome_badge(record, stacked=True)
            rows.append(
                "<tr>"
                f"<td>{status_html}{outcome_html}</td>"
                f'<td><span class="muted">{escape_html(record.test_case_public_id or "")}</span> '
                f'<a href="/test-cases/{record.test_case_id}">{escape_html(record.test_case_name)}</a></td>'
                f'<td>{badge(record.workflow_type.value, "workflow")}</td>'
                f'<td><time datetime="{escape_html(record.started_at.isoformat())}">'
                f'{escape_html(format_timestamp(record.started_at))}</time></td>'
                f'<td>{escape_html(format_duration(record.duration_ms))}</td>'
                f'<td><a href="/runs/{record.run_id}" title="{escape_html(record.run_id)}">'
                f'{escape_html(record.public_id or "Run")}</a></td>'
                "</tr>"
            )
        return (
            '<div class="table-wrap"><table><thead><tr><th>Status</th><th>TestCase</th>'
            '<th>Workflow</th><th>Started</th><th>Duration</th><th>Run</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )

    def _test_case_page(self, test_case_id: UUID) -> WebResponse:
        records = self._run_history.list_for_test_case(test_case_id, _PAGE_LIMIT)
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        if self._test_cases is not None and test_case is None:
            return self._not_found("TestCase not found")
        if self._test_cases is None and not records:
            return self._not_found("TestCase not found")
        latest = records[0] if records else None
        passed = sum(record.status.value == "PASSED" for record in records)
        failed = sum(record.status.value == "FAILED" for record in records)
        cards = (
            _summary_card(
                "Latest result",
                badge(latest.status.value) + _outcome_badge(latest, stacked=True)
                if latest else badge("NOT RUN"),
                raw=True,
            )
            + _summary_card("Last run", format_timestamp(latest.started_at) if latest else "Never")
            + _summary_card("Total runs", str(len(records)))
            + _summary_card("Passed / failed", f"{passed} / {failed}")
            + _summary_card(
                "Latest workflow",
                badge(latest.workflow_type.value, "workflow") if latest else "—",
                raw=True,
            )
        )
        if test_case is not None:
            case_public_id = test_case.public_id or (latest.test_case_public_id if latest else None)
            case_name = test_case.name
            case_description = test_case.description
            source_note = "Definition loaded from the saved TestCase."
            steps = "".join(
                f'<li><strong>{escape_html(step.name)}</strong>'
                f'<div class="muted">Expected: {escape_html(step.expected)}</div></li>'
                for step in test_case.steps
            )
            setup_by_id = {
                item.id: item for item in latest.preconditions
            } if latest is not None else {}
            conditions = "".join(
                "<li>"
                + (badge(setup_by_id[item.id].status) + " " if item.id in setup_by_id and setup_by_id[item.id].status else "")
                + escape_html(item.description)
                + (
                    f'<div class="muted">{escape_html(setup_by_id[item.id].error)}</div>'
                    if item.id in setup_by_id and setup_by_id[item.id].error else ""
                )
                + "</li>"
                for item in sorted(test_case.preconditions, key=lambda item: item.order)
            )
            segments = "".join(
                f"<li>Segment {number}: {escape_html(segment.base_url or test_case.base_url or 'No URL configured')} "
                f"({len(segment.steps)} steps)</li>"
                for number, segment in enumerate(sorted(test_case.segments, key=lambda item: item.order), start=1)
            )
        else:
            case_public_id = latest.test_case_public_id
            case_name = latest.test_case_name
            case_description = latest.test_case_description
            source_note = "Definition shown from the most recent recorded run."
            steps = "".join(
                f'<li><strong>{escape_html(step.name)}</strong>'
                f'<div class="muted">Expected: {escape_html(step.expected)}</div></li>'
                for step in sorted(latest.steps, key=lambda item: item.order)
            )
            conditions = "".join(
                "<li>"
                + (badge(condition.status) + " " if condition.status else "")
                + escape_html(condition.description)
                + (f'<div class="muted">{escape_html(condition.error)}</div>' if condition.error else "")
                + "</li>"
                for condition in latest.preconditions
            )
            segments = "".join(
                f"<li>Segment {number}: {escape_html(segment.base_url or 'No URL recorded')} "
                f"({len(segment.test_step_ids)} steps)</li>"
                for number, segment in enumerate(sorted(latest.segments, key=lambda item: item.order), start=1)
            )
        history_rows = "".join(
            "<tr>"
            f'<td>{badge(record.status.value)}</td>'
            f'<td>{_outcome_badge(record)}</td>'
            f'<td>{badge(record.workflow_type.value, "workflow")}</td>'
            f'<td><time datetime="{escape_html(record.started_at.isoformat())}">'
            f'{escape_html(format_timestamp(record.started_at))}</time></td>'
            f'<td>{escape_html(format_duration(record.duration_ms))}</td>'
            f'<td><a href="/runs/{record.run_id}" title="{escape_html(record.run_id)}">'
            f'{escape_html(record.public_id or "Run")}</a></td></tr>'
            for record in records
        )
        history_table = (
            '<div class="table-wrap"><table><thead><tr><th>Status</th><th>Result</th><th>Workflow</th>'
            '<th>Last run</th><th>Duration</th><th>Run</th></tr></thead><tbody>'
            + history_rows
            + '</tbody></table></div>'
            if records else self._empty_state(
                "No runs yet.", "Generate and run automation to create the first plan versions and history."
            )
        )
        run_form = ""
        if self._run_service is not None and test_case is not None:
            availability = self._workflow_availability(test_case_id)
            if availability is not None:
                run_form = self._workflow_panel(test_case_id, test_case, availability)
            else:
                # Compatibility for lightweight adapters that only implement run().
                run_form = self._run_form(test_case_id)
        usage_summary = (
            self._llm_usage.test_case_summary(test_case_id)
            if self._llm_usage is not None else None
        )
        usage_panel = (
            self._test_case_usage_panel(usage_summary)
            if usage_summary is not None else ""
        )
        automation_status = (
            self._automation_lifecycle.status(test_case)
            if test_case is not None else AutomationStatus.NOT_AUTOMATED
        )
        lifecycle_label = _automation_status_label(automation_status)
        lifecycle_note = {
            AutomationStatus.NOT_AUTOMATED: "No complete executable automation is saved yet.",
            AutomationStatus.AUTOMATION_READY: "Current saved automation passed Validation for this TestCase definition.",
            AutomationStatus.NEEDS_UPDATE: "The TestCase changed after its automation was prepared. Generate updated automation before validating it.",
            AutomationStatus.NEEDS_VALIDATION: "Executable automation exists and must pass Validation before it is marked ready.",
            AutomationStatus.AUTOMATION_FAILED: "Automation could not be prepared completely. Review the TestCase and try Automation again.",
        }[automation_status]
        lifecycle_panel = (
            '<section class="panel"><h2>Automation status</h2>'
            f'<p>{badge(lifecycle_label, "workflow")}</p><p class="muted">{escape_html(lifecycle_note)}</p></section>'
            if test_case is not None else ""
        )
        edit_links = (
            f'<div class="actions"><a class="button" href="/test-cases/{test_case_id}/edit">Edit TestCase</a>'
            f'<a class="button" href="/test-cases/{test_case_id}/plans">View TestPlan</a></div>'
            if test_case is not None else ""
        )
        export_panel = ""
        if test_case is not None and self._testplan_exports is not None:
            try:
                exportable = self._testplan_exports.get(test_case_id)
            except TestPlanExportError as error:
                unavailable_message = (
                    "Automation required before code export. Generate and save a complete set of automation plans first."
                    if str(error).startswith("Automation required")
                    else str(error)
                )
                export_panel = (
                    '<section class="panel"><h2>Export</h2>'
                    f'<p class="muted">{escape_html(unavailable_message)}</p></section>'
                )
            else:
                version_summary = ", ".join(
                    f"Step {item.step_order + 1}: v{item.version.version} · "
                    f"{_plan_origin_label(item.version.origin)}"
                    for item in exportable.plans
                )
                export_panel = (
                    '<section class="panel export-panel"><div class="section-heading"><div>'
                    '<h2>Export</h2><p class="muted">Saved automation, ready to use outside AI QA Agent.</p>'
                    '</div><details><summary>Plan versions</summary>'
                    f'<p class="muted">{escape_html(version_summary)}</p></details></div>'
                    '<div class="button-row">'
                    f'<a class="button" href="/test-cases/{test_case_id}/export/portable">Portable JSON</a>'
                    f'<a class="button" href="/test-cases/{test_case_id}/export/python">Python Playwright</a>'
                    f'<a class="button" href="/test-cases/{test_case_id}/export/typescript">TypeScript Playwright</a>'
                    f'<a class="button" href="/test-cases/{test_case_id}/export/csharp">C# Playwright</a>'
                    '</div></section>'
                )
        content = (
            '<header class="page-heading">'
            + (f'<p class="eyebrow">{escape_html(case_public_id)}</p>' if case_public_id else '')
            + '<h1>' + escape_html(case_name) + '</h1>'
            + f'<p class="lead">{escape_html(case_description)}</p>'
            + f'<p class="muted">{escape_html(source_note)}</p></header>'
            + edit_links
            + f'<details class="technical-details"><summary>Technical IDs</summary>'
            + f'<p>TestCase UUID: <code>{escape_html(test_case_id)}</code></p></details>'
            + f'<div class="summary-grid">{cards}</div>'
            + lifecycle_panel
            + run_form
            + export_panel
            + usage_panel
            + '<section class="panel"><h2>Preconditions</h2>'
            + (f"<ul>{conditions}</ul>" if conditions else '<p class="muted">No preconditions recorded.</p>')
            + '</section><section class="panel"><h2>Steps</h2>'
            + (f"<ol>{steps}</ol>" if steps else '<p class="muted">No steps recorded.</p>')
            + '</section><section class="panel"><h2>Execution segments</h2>'
            + (f"<ul>{segments}</ul>" if segments else '<p class="muted">No segment data recorded.</p>')
            + '</section><section class="panel"><h2>Run History</h2>'
            + history_table + '</section>'
        )
        return WebResponse.html(
            200,
            self._page(
                f"{case_public_id} — {case_name}" if case_public_id else case_name,
                content,
                current="Test Cases",
                breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases")],
            ),
        )

    @staticmethod
    def _test_case_usage_panel(summary: dict[str, object]) -> str:
        tokens = summary.get("total_tokens")
        cost = summary.get("estimated_cost_usd")
        cost_label = "Unknown" if cost is None else f"${float(cost):.6f}"
        request_count = int(summary.get("requests", 0))
        usage_known = int(summary.get("usage_known_requests", 0))
        fallback_count = int(summary.get("fallback_operations", 0))
        providers = ", ".join(
            escape_html(name) for name in summary.get("providers", [])
        ) or "Unknown"
        operations = summary.get("by_operation", [])
        rows = "".join(
            "<tr>"
            f"<td>{escape_html(_operation_label(str(item.get('operation_type', ''))))}</td>"
            f"<td>{int(item.get('requests', 0))}</td>"
            f"<td>{int(item.get('successful_requests', 0))} / {int(item.get('failed_requests', 0))}</td>"
            f"<td>{_format_usage_cost(item.get('estimated_cost_usd'))}</td>"
            "</tr>"
            for item in operations if isinstance(item, dict)
        )
        note = (
            f'<p class="muted">Token usage is available for {usage_known} of '
            f'{request_count} provider attempts. Cost is estimated only where a verified '
            f'model price is configured.</p>'
        )
        return (
            '<section class="panel"><h2>AI Usage</h2>'
            f'<p>Providers: {providers}</p>'
            '<div class="summary-grid">'
            + _summary_card("Provider attempts", str(request_count))
            + _summary_card(
                "Total tokens",
                (f"{tokens} (partial)" if summary.get("usage_is_partial") else str(tokens))
                if tokens is not None else "Unknown",
            )
            + _summary_card("Estimated cost", cost_label)
            + _summary_card("Fallback operations", str(fallback_count))
            + '</div>' + note
            + ('<div class="table-wrap"><table><thead><tr><th>Operation</th>'
               '<th>Provider attempts</th><th>Successful / failed</th><th>Estimated cost</th>'
               '</tr></thead><tbody>' + rows + '</tbody></table></div>' if rows else '')
            + '</section>'
        )

    def _usage_analytics_page(self, query: dict[str, list[str]]) -> str:
        window = query.get("range", ["30d"])[0]
        if window not in {"today", "7d", "30d", "all"}:
            window = "30d"
        analytics = self._llm_usage.analytics(window)
        cost = analytics.get("estimated_cost_usd")
        cost_label = _format_usage_cost(cost)
        if cost is not None and analytics.get("cost_is_partial"):
            cost_label += " (partial)"
        tokens = analytics.get("total_tokens")
        tokens_label = (
            "Unknown" if tokens is None else
            f"{tokens} (partial)" if analytics.get("usage_is_partial") else str(tokens)
        )
        attempts = int(analytics.get("requests", 0))
        successes = int(analytics.get("successful_requests", 0))
        failures = int(analytics.get("failed_requests", 0))
        fallback_ops = int(analytics.get("fallbacks", 0))
        operations = int(analytics.get("operations", 0))
        limited_sample = bool(analytics.get("limited_sample"))
        sample_note = (
            '<p class="notice">Limited sample: fewer than five provider attempts. '
            'Reliability rates are descriptive and should not be used to rank providers.</p>'
            if limited_sample else
            '<p class="muted">Reliability is shown with raw counts. The page does not rank providers.</p>'
        )
        links = "".join(
            f'<a class="button{" primary" if selected == window else ""}" '
            f'href="/settings/usage?range={selected}">{label}</a>'
            for selected, label in (
                ("today", "Today"), ("7d", "7 days"),
                ("30d", "30 days"), ("all", "All time"),
            )
        )
        provider_rows = []
        for item in analytics.get("by_provider", []):
            provider_name = str(item.get("provider_name") or item.get("provider_id", "Unknown"))
            rate = _format_usage_rate(item.get("success_rate"))
            if item.get("limited_sample"):
                rate += ' <span class="muted">Limited sample</span>'
            tokens_value = item.get("total_tokens")
            if tokens_value is not None and item.get("usage_is_partial"):
                tokens_value = f"{tokens_value} (partial)"
            provider_cost = _format_usage_cost(item.get("estimated_cost_usd"))
            if item.get("estimated_cost_usd") is not None and item.get("cost_is_partial"):
                provider_cost += " (partial)"
            provider_rows.append(
                "<tr>"
                f"<td>{escape_html(provider_name)}</td>"
                f"<td>{int(item.get('requests', 0))}</td>"
                f"<td>{int(item.get('successful_requests', 0))} / {int(item.get('failed_requests', 0))}</td>"
                f"<td>{rate}</td>"
                f"<td>{int(item.get('fallback_requests', 0))}</td>"
                f"<td>{escape_html(str(tokens_value) if tokens_value is not None else 'Unknown')}</td>"
                f"<td>{_format_usage_latency(item.get('average_latency_ms'))}</td>"
                f"<td>{provider_cost}</td></tr>"
            )
        model_rows = "".join(
            "<tr>"
            f"<td>{escape_html(str(item.get('provider_id', 'unknown')))}</td>"
            f"<td>{escape_html(str(item.get('model', 'unknown')))}</td>"
            f"<td>{int(item.get('requests', 0))}</td>"
            f"<td>{int(item.get('successful_requests', 0))} / {int(item.get('failed_requests', 0))}</td>"
            f"<td>{escape_html((str(item.get('total_tokens')) + ' (partial)') if item.get('total_tokens') is not None and item.get('usage_is_partial') else str(item.get('total_tokens')) if item.get('total_tokens') is not None else 'Unknown')}</td>"
            f"<td>{_format_usage_cost(item.get('estimated_cost_usd'))}</td></tr>"
            for item in analytics.get("by_model", [])
        )
        operation_rows = "".join(
            "<tr>"
            f"<td>{escape_html(_operation_label(str(item.get('operation_type', ''))))}</td>"
            f"<td>{int(item.get('requests', 0))}</td>"
            f"<td>{int(item.get('successful_requests', 0))} / {int(item.get('failed_requests', 0))}</td>"
            f"<td>{_format_usage_latency(item.get('average_latency_ms'))}</td>"
            f"<td>{_format_usage_cost(item.get('estimated_cost_usd'))}</td></tr>"
            for item in analytics.get("by_operation", [])
        )
        provider_table = (
            '<div class="table-wrap"><table><thead><tr><th>Provider</th><th>Attempts</th>'
            '<th>Success / failure</th><th>Success rate</th><th>Fallback attempts</th>'
            '<th>Tokens</th><th>Average latency</th><th>Estimated cost</th></tr></thead><tbody>'
            + "".join(provider_rows) + '</tbody></table></div>'
            if provider_rows else self._empty_state(
                "No provider usage recorded yet.",
                "Usage appears after an LLM provider request completes.",
            )
        )
        model_table = (
            '<div class="table-wrap"><table><thead><tr><th>Provider</th><th>Model</th>'
            '<th>Attempts</th><th>Success / failure</th><th>Tokens</th><th>Estimated cost</th>'
            '</tr></thead><tbody>' + model_rows + '</tbody></table></div>'
            if model_rows else '<p class="muted">No model usage recorded for this range.</p>'
        )
        operation_table = (
            '<div class="table-wrap"><table><thead><tr><th>Operation</th><th>Attempts</th>'
            '<th>Success / failure</th><th>Average latency</th><th>Estimated cost</th>'
            '</tr></thead><tbody>' + operation_rows + '</tbody></table></div>'
            if operation_rows else '<p class="muted">No operation usage recorded for this range.</p>'
        )
        content = (
            '<header class="page-heading"><p class="eyebrow">Analytics</p><h1>AI Usage</h1>'
            '<p class="lead">Provider attempts, actual token counts, latency, fallback, and cost estimates.</p></header>'
            + self._settings_tabs("usage")
            + f'<div class="filters">{links}</div>{sample_note}'
            '<div class="summary-grid">'
            + _summary_card("Provider attempts", str(attempts))
            + _summary_card("Successful / failed", f"{successes} / {failures}")
            + _summary_card("Operations with fallback", f"{fallback_ops} / {operations}")
            + _summary_card("Total tokens", tokens_label)
            + _summary_card("Estimated cost", cost_label)
            + _summary_card("Average latency", _format_usage_latency(analytics.get("average_latency_ms")))
            + '</div><section class="panel"><h2>Provider reliability and usage</h2>'
            + provider_table + '</section><section class="panel"><h2>Model comparison</h2>'
            + model_table + '</section><section class="panel"><h2>Usage by operation</h2>'
            + operation_table + '</section>'
            + '<p class="muted">Token counts come from provider response usage metadata. Cost is an estimate '
            'and is Unknown when no verified rate or complete token breakdown is available.</p>'
        )
        return self._page(
            "AI Usage",
            content,
            current="AI Usage",
            breadcrumbs=[("Dashboard", "/")],
        )

    @staticmethod
    def _run_form(test_case_id: UUID) -> str:
        return (
            '<section class="panel"><h2>Run this TestCase</h2>'
            '<p class="muted">Validation and Regression execute the saved plan versions for each step.</p>'
            f'<form method="post" action="/test-cases/{test_case_id}/run" class="filters" data-run-form>'
            '<div class="field"><label for="run-workflow">Workflow</label>'
            '<select id="run-workflow" name="workflow" required>'
            '<option value="" disabled selected>Choose a workflow</option>'
            '<option value="VALIDATION">Validation</option>'
            '<option value="REGRESSION">Regression</option>'
            '</select></div><button class="button primary" type="submit">Start run</button></form>'
            '</section>'
        )

    def _workflow_availability(self, test_case_id: UUID) -> WorkflowAvailability | None:
        method = getattr(self._run_service, "workflow_availability", None)
        if not callable(method):
            return None
        try:
            value = method(test_case_id)
        except RunUnavailableError:
            return None
        return value if isinstance(value, WorkflowAvailability) else None

    def _workflow_panel(
        self,
        test_case_id: UUID,
        test_case,
        availability: WorkflowAvailability,
    ) -> str:
        forms = []
        if availability.automation_available:
            forms.append(
                f'<form method="post" action="/test-cases/{test_case_id}/run" data-run-form>'
                '<input type="hidden" name="workflow" value="AUTOMATION">'
                '<button class="button primary" type="submit">Generate &amp; Run Automation</button></form>'
            )
        plan_by_step = {
            step_id: (version, version_id)
            for step_id, version, version_id in availability.plan_versions
        }
        version_rows = []
        for step in sorted(test_case.steps, key=lambda item: item.order):
            selected = plan_by_step.get(step.id)
            if selected is not None:
                version, version_id = selected
                saved_version = (
                    self._plan_store.get_version(version_id)
                    if self._plan_store is not None else None
                )
                origin_label = _plan_origin_label(
                    saved_version.origin if saved_version is not None else None
                )
                created = (
                    f'<p>Created: {escape_html(format_timestamp(saved_version.created_at))}</p>'
                    if saved_version is not None else ""
                )
                previous = f'<p>Previous version: v{version - 1}</p>' if version > 1 else ""
                version_rows.append(
                    f'<li><span>Step {step.order + 1}: {escape_html(step.name)}</span> '
                    f'<details class="plan-version-details"><summary>v{version} '
                    f'{badge(origin_label, "workflow")}</summary>'
                    f'<p>Automation version v{version}</p>{created}{previous}'
                    f'<p>Internal ID: <code>{escape_html(version_id)}</code></p></details></li>'
                )
        plan_status = (
            f'{availability.usable_plan_count} / {availability.total_step_count} steps have executable plans.'
        )
        validation = availability.validation_available
        regression = availability.regression_available
        for workflow, label, available in (
            (WorkflowType.VALIDATION, "Validation", validation),
            (WorkflowType.REGRESSION, "Regression", regression),
        ):
            if available:
                forms.append(
                    f'<form method="post" action="/test-cases/{test_case_id}/run" data-run-form>'
                    f'<input type="hidden" name="workflow" value="{workflow.value}">'
                    f'<button class="button" type="submit">Run {label}</button></form>'
                )
        status_rows = (
            '<li>Automation <strong>'
            + ("Available" if availability.automation_available else "Not configured")
            + '</strong> — generates and runs executable plans.</li>'
            + '<li>Validation <strong>' + ("Available" if validation else "Not ready") + '</strong></li>'
            + '<li>Regression <strong>' + ("Available" if regression else "Not ready") + '</strong></li>'
        )
        reason = (
            f'<p class="muted">{escape_html(availability.reason)}</p>'
            if not validation and availability.reason else ""
        )
        versions = (
            f'<ol class="compact-list">{"".join(version_rows)}</ol>'
            if version_rows else '<p class="muted">No executable plan versions have been generated.</p>'
        )
        return (
            '<section class="panel"><h2>Automation</h2>'
            f'<p>{escape_html(plan_status)}</p>{versions}'
            f'<ul class="compact-list">{status_rows}</ul>{reason}'
            + ('<div class="actions">' + ''.join(forms) + '</div>' if forms else '')
            + '</section>'
        )

    def _serve_evidence(self, run_id_text: str, execution_id_text: str, index_text: str) -> WebResponse:
        run_id = _parse_uuid(run_id_text)
        execution_id = _parse_uuid(execution_id_text)
        if run_id is None or execution_id is None or not index_text.isdecimal():
            return self._not_found("Evidence not available")
        detail = self._run_history.get_detail(run_id)
        path = self._resolve_evidence_file(detail, execution_id, int(index_text)) if detail else None
        if path is None:
            return self._not_found("Evidence not available")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        safe_name = path.name.replace("\r", "").replace("\n", "").replace('"', "")
        try:
            return WebResponse(
                200,
                content_type,
                path.read_bytes(),
                {"Content-Disposition": f'inline; filename="{safe_name}"'},
            )
        except OSError:
            return self._not_found("Evidence not available")

    def _resolve_evidence_file(
        self,
        detail: RunHistoryDetail,
        execution_id: UUID,
        index: int,
    ) -> Path | None:
        if self._evidence_root is None or index < 0:
            return None
        reference = next(
            (item for item in detail.record.executions if item.execution_id == execution_id),
            None,
        )
        execution = detail.executions.get(execution_id)
        if (
            reference is None
            or execution is None
            or index >= len(execution.evidence)
            or index >= len(reference.evidence)
            or execution.evidence[index].id != reference.evidence[index].id
            or execution.evidence[index].type.value != "SCREENSHOT"
            or reference.evidence[index].type != "SCREENSHOT"
        ):
            return None
        try:
            root = self._evidence_root.resolve(strict=True)
            evidence_path = Path(execution.evidence[index].path)
            if not evidence_path.is_absolute():
                evidence_path = root / evidence_path
            resolved = evidence_path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                return None
            if resolved.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                return None
            return resolved
        except (OSError, RuntimeError, ValueError):
            return None

    def _evidence_url_if_available(
        self,
        detail: RunHistoryDetail,
        run_id: UUID,
        _step: RunStepReport,
        attempt: RunAttemptReport,
        _evidence: RunEvidenceReport,
        index: int,
    ) -> str | None:
        if self._resolve_evidence_file(detail, attempt.execution_id, index) is None:
            return None
        return f"/runs/{run_id}/evidence/{attempt.execution_id}/{index}"

    def _handle_provider_settings_post(self, body: bytes | str | None) -> WebResponse:
        form, form_error = _parse_form_body(body)
        if form_error is not None:
            return WebResponse.redirect("/settings/providers?notice=invalid")
        provider_id = form.get("provider_id", [""])[0]
        operation = form.get("operation", [""])[0]
        settings = self._provider_settings
        if settings is None:
            return WebResponse.redirect("/settings/providers?notice=unavailable")

        result = "updated"
        if operation == "create_custom":
            try:
                provider_id = settings.create_custom(
                    form.get("display_name", [""])[0],
                    form.get("base_url", [""])[0],
                    form.get("model", [""])[0],
                    api_key=form.get("api_key", [""])[0] or None,
                    enabled=form.get("enabled", [""])[0] == "yes",
                    requires_api_key=form.get("requires_api_key", [""])[0] == "yes",
                )
                self._refresh_provider_router()
                return WebResponse.redirect("/settings/providers?" + urlencode({"provider": provider_id, "result": "created"}) + "#" + quote(provider_id, safe=""))
            except SecretStoreError:
                return WebResponse.redirect("/settings/providers?notice=storage_unavailable#add-provider")
            except ValueError:
                return WebResponse.redirect("/settings/providers?notice=invalid#add-provider")
            except Exception:
                return WebResponse.redirect("/settings/providers?notice=unavailable#add-provider")

        try:
            if operation == "enable":
                settings.update(provider_id, enabled=True)
                self._refresh_provider_router()
            elif operation == "disable":
                settings.update(provider_id, enabled=False)
                self._refresh_provider_router()
            elif operation in {"move_up", "move_down"}:
                settings.move(provider_id, -1 if operation == "move_up" else 1)
                self._refresh_provider_router()
            elif operation == "save_key":
                settings.save_key(provider_id, form.get("api_key", [""])[0])
                self._refresh_provider_router()
                result = "key_saved"
            elif operation == "remove_key":
                settings.remove_key(provider_id)
                self._refresh_provider_router()
                result = "key_removed"
            elif operation == "save_model":
                model = form.get("model", [""])[0].strip()
                if len(model) > 200 or any(ord(char) < 32 for char in model):
                    raise ValueError("Invalid model.")
                settings.update(provider_id, model=model)
                self._refresh_provider_router()
            elif operation == "save_custom":
                settings.update_custom(
                    provider_id,
                    display_name=form.get("display_name", [""])[0],
                    base_url=form.get("base_url", [""])[0],
                    model=form.get("model", [""])[0],
                )
                self._refresh_provider_router()
            elif operation == "test_connection":
                settings.test_connection(provider_id)
                result = "tested"
            elif operation == "test_authoring_capability":
                settings.test_authoring_capability(provider_id)
                result = "capability_tested"
            elif operation == "delete_custom":
                if form.get("confirmed", [""])[0] != "yes":
                    raise ValueError("Confirm provider deletion first.")
                display_name = next((view.display_name for view in settings.provider_views() if view.id == provider_id), "Custom provider")
                settings.delete_custom(provider_id)
                self._refresh_provider_router()
                return WebResponse.redirect("/settings/providers?" + urlencode({"notice": "deleted", "deleted_name": display_name}))
            else:
                result = "invalid"
        except SecretStoreError:
            result = "storage_unavailable"
        except ValueError:
            result = "invalid"
        except Exception:
            # Provider and credential errors are intentionally reduced to a stable notice.
            result = "unavailable"
        return WebResponse.redirect("/settings/providers?" + urlencode({"provider": provider_id, "result": result}) + "#" + quote(provider_id, safe=""))

    def _refresh_provider_router(self) -> None:
        if self._provider_settings is not None and self._provider_router is not None:
            self._provider_settings.refresh_router(self._provider_router)

    def _provider_settings_page(self, query: dict[str, list[str]]) -> str:
        settings = self._provider_settings
        if settings is None:
            return self._page("AI Providers", self._empty_state("Settings unavailable", "Provider settings are not configured."), current="Settings")
        messages = {
            "deleted": "Custom provider removed. Historical AI Usage records are retained.",
            "invalid": "The settings request was invalid. Review the values and try again.",
            "storage_unavailable": "Secure credential storage is unavailable on this system.",
            "unavailable": "The requested provider operation could not be completed.",
        }
        notice_code = (query.get("notice") or [""])[0]
        notice = (
            f'<div class="notice" role="status">{escape_html(messages[notice_code])}'
            + (f' {escape_html((query.get("deleted_name") or ["Custom provider"])[0])}.' if notice_code == "deleted" else "")
            + '</div>'
            if notice_code in messages else ""
        )
        views = settings.provider_views()
        result_messages = {
            "updated": "Provider settings saved.",
            "key_saved": "Credential saved securely.",
            "key_removed": "Web-managed credential removed.",
            "created": "Custom provider added.",
            "tested": "Connection test finished.",
            "capability_tested": "Authoring capability check finished; review its result below.",
            "invalid": "This change could not be saved. Review the provider settings.",
            "storage_unavailable": "Secure credential storage is unavailable.",
            "unavailable": "This provider operation could not be completed.",
        }
        result_provider = (query.get("provider") or [""])[0]
        result_code = (query.get("result") or [""])[0]
        cards = []
        for index, provider in enumerate(views):
            is_configured = provider.credential_source in {"Web settings", "Environment variable", "No API key required"}
            status_label = "Configured" if is_configured else "Needs configuration"
            status_tone = "success" if is_configured else "warning"
            local_result = ""
            connection = _provider_diagnostic_html(
                "Connection", provider.health, provider.connection_category,
                provider.latency_ms, provider.connection_http_status,
                provider.connection_retry_after_seconds,
            )
            capability = _provider_diagnostic_html(
                "Authoring capability", provider.capability_status,
                provider.capability_category, provider.capability_latency_ms,
                provider.capability_http_status,
                provider.capability_retry_after_seconds,
            )
            health = f'<div class="provider-diagnostics">{connection}{capability}</div>'
            if result_provider == provider.id and result_code in result_messages:
                local_result = f'<p class="provider-feedback" role="status">{escape_html(result_messages[result_code])}</p>'
            toggle = "disable" if provider.enabled else "enable"
            toggle_label = "Disable" if provider.enabled else "Enable"
            move_up_disabled = " disabled" if index == 0 else ""
            move_down_disabled = " disabled" if index == len(views) - 1 else ""
            credential = escape_html(provider.credential_source)
            model = escape_html(provider.model)
            provider_id = escape_html(provider.id)
            delete_details = ""
            if provider.is_custom:
                delete_details = (
                    '<details class="provider-delete"><summary>Delete provider</summary>'
                    '<form method="post" action="/settings/providers">'
                    f'<input type="hidden" name="provider_id" value="{provider_id}">'
                    '<input type="hidden" name="operation" value="delete_custom">'
                    f'<label><input type="checkbox" name="confirmed" value="yes" required> Confirm removal of {escape_html(provider.display_name)} and its saved credential.</label>'
                    '<button class="button danger-button" type="submit">Delete custom provider</button>'
                    '</form></details>'
                )
            if provider.is_custom:
                edit_fields = (
                    f'<div class="field"><label for="name-{provider_id}">Provider name</label><input id="name-{provider_id}" name="display_name" maxlength="80" required value="{escape_html(provider.display_name)}"></div>'
                    f'<div class="field"><label for="base-{provider_id}">Base URL</label><input id="base-{provider_id}" name="base_url" maxlength="2048" required value="{escape_html(provider.base_url or "")}"></div>'
                    f'<div class="field"><label for="model-{provider_id}">Model</label><input id="model-{provider_id}" name="model" maxlength="200" required value="{model}"></div>'
                    + f'<input type="hidden" name="operation" value="save_custom"><input type="hidden" name="provider_id" value="{provider_id}">'
                )
            else:
                edit_fields = (
                    f'<div class="field"><label for="model-{provider_id}">Model</label><input id="model-{provider_id}" name="model" maxlength="200" value="{model}"></div>'
                    + f'<input type="hidden" name="operation" value="save_model"><input type="hidden" name="provider_id" value="{provider_id}">'
                )
            key_action = (
                f'<form class="provider-key-form" method="post" action="/settings/providers" autocomplete="off"><input type="hidden" name="provider_id" value="{provider_id}"><input type="hidden" name="operation" value="save_key">'
                f'<div class="field"><label for="key-{provider_id}">{("Replace" if provider.credential_source == "Web settings" else "Add")} API key</label>'
                f'<input id="key-{provider_id}" name="api_key" type="password" maxlength="2500" autocomplete="off" autocapitalize="off" spellcheck="false" data-lpignore="true" data-1p-ignore="true" data-form-type="other" required></div>'
                '<button class="button" type="submit">Save key</button></form>'
            )
            remove_key = ""
            if provider.credential_source == "Web settings":
                remove_key = (
                    f'<form method="post" action="/settings/providers"><input type="hidden" name="provider_id" value="{provider_id}"><input type="hidden" name="operation" value="remove_key">'
                    '<button class="button" type="submit">Remove saved key</button></form>'
                )
            cards.append(
                f'<section class="panel provider-card" id="{provider_id}" aria-labelledby="provider-title-{provider_id}">'
                '<div class="provider-topline"><div><p class="eyebrow">Priority ' + str(provider.priority) + '</p>'
                f'<h2 id="provider-title-{provider_id}">{escape_html(provider.display_name)}</h2></div>'
                + badge(status_label, status_tone)
                + badge("Enabled" if provider.enabled else "Disabled", "success" if provider.enabled else "neutral")
                + '</div><div class="provider-quick-meta">'
                + f'<span><strong>Model:</strong> <code>{model or "Not set"}</code></span>'
                + f'<span><strong>Credential:</strong> {credential}</span>'
                + (health or '<span class="muted">No connection test yet</span>')
                + f'</div>{local_result}<div class="actions provider-actions">'
                + f'<form method="post" action="/settings/providers"><input type="hidden" name="provider_id" value="{provider_id}">'
                + '<input type="hidden" name="operation" value="test_connection"><button class="button" type="submit">Test connection</button></form>'
                + f'<form method="post" action="/settings/providers"><input type="hidden" name="provider_id" value="{provider_id}">'
                + '<input type="hidden" name="operation" value="test_authoring_capability"><button class="button" type="submit">Test authoring capability</button></form>'
                + f'<form method="post" action="/settings/providers"><input type="hidden" name="provider_id" value="{provider_id}">'
                + f'<input type="hidden" name="operation" value="{toggle}"><button class="button" type="submit">{toggle_label}</button></form>'
                + f'<form method="post" action="/settings/providers"><input type="hidden" name="provider_id" value="{provider_id}">'
                + f'<input type="hidden" name="operation" value="move_up"><button class="button" aria-label="Move {escape_html(provider.display_name)} up" type="submit"{move_up_disabled}>↑</button></form>'
                + f'<form method="post" action="/settings/providers"><input type="hidden" name="provider_id" value="{provider_id}">'
                + f'<input type="hidden" name="operation" value="move_down"><button class="button" aria-label="Move {escape_html(provider.display_name)} down" type="submit"{move_down_disabled}>↓</button></form>'
                + f'<details class="provider-edit"><summary>Edit</summary><div class="provider-edit-body"><p class="muted">Credential: {credential} · <span>Key: <code>{escape_html(provider.masked_key)}</code></span></p><form class="provider-model-form" method="post" action="/settings/providers">{edit_fields}<button class="button" type="submit">Save settings</button></form>'
                + key_action + remove_key + delete_details + '</div></details></div></section>'
            )
        configured_count = sum(view.credential_source in {"Web settings", "Environment variable", "No API key required"} for view in views)
        enabled_count = sum(view.enabled for view in views)
        ready_count = sum(
            view.enabled
            and view.credential_source in {"Web settings", "Environment variable", "No API key required"}
            and view.capability_status == "passed"
            for view in views
        )
        healthy_count = sum(view.health == "connected" for view in views)
        attention_count = sum(
            view.credential_source not in {"Web settings", "Environment variable", "No API key required"}
            or view.capability_status == "failed"
            or (view.health is not None and view.health != "connected")
            for view in views
        )
        summary = (
            '<div class="provider-summary" aria-label="Provider summary">'
            + _summary_card("Configured", str(configured_count))
            + _summary_card("Enabled providers", str(enabled_count))
            + _summary_card("Ready for authoring", str(ready_count))
            + _summary_card("Connection healthy", str(healthy_count))
            + _summary_card("Needs attention", str(attention_count))
            + '</div><p class="muted summary-caption">Configured providers have a credential or no-key endpoint. TestCase authoring is verified only after the authoring capability check passes.</p>'
        )
        create_form = (
            '<details class="panel add-provider" id="add-provider"><summary>+ Add provider</summary>'
            '<p class="muted">Connect an endpoint that supports the OpenAI-compatible chat completions API.</p>'
            '<form method="post" action="/settings/providers" class="custom-provider-form">'
            '<input type="hidden" name="operation" value="create_custom">'
            '<div class="field"><label for="custom-display-name">Provider name</label><input id="custom-display-name" name="display_name" maxlength="80" required></div>'
            '<div class="field"><label for="custom-base-url">Base URL</label><input id="custom-base-url" name="base_url" placeholder="http://localhost:1234/v1" maxlength="2048" required></div>'
            '<div class="field"><label for="custom-model">Model</label><input id="custom-model" name="model" maxlength="200" required></div>'
            '<div class="field"><label for="custom-api-key">API key (optional)</label><input id="custom-api-key" name="api_key" type="password" maxlength="2500" autocomplete="off" autocapitalize="off" spellcheck="false" data-lpignore="true" data-1p-ignore="true" data-form-type="other"></div>'
            '<label><input type="checkbox" name="requires_api_key" value="yes"> Endpoint requires an API key</label>'
            '<label><input type="checkbox" name="enabled" value="yes" checked> Enable provider</label>'
            '<button class="button primary" type="submit">Add provider</button></form></details>'
        )
        content = (
            '<header class="page-heading"><p class="eyebrow">Settings</p><h1>AI Providers</h1>'
            '<p class="lead">Enabled providers run in priority order. The router falls back after retryable failures.</p></header>'
            + self._settings_tabs("providers")
            + notice + '<p class="muted">Built-in providers retain environment key fallback. Keys saved here use the operating system credential vault.</p>'
            + summary + create_form + ''.join(cards)
        )
        return self._page("AI Providers", content, current="Settings", breadcrumbs=[("Dashboard", "/")])

    @staticmethod
    def _settings_tabs(active: str) -> str:
        providers_current = ' aria-current="page"' if active == "providers" else ""
        usage_current = ' aria-current="page"' if active == "usage" else ""
        return (
            '<nav class="settings-tabs" aria-label="Settings sections">'
            f'<a href="/settings/providers"{providers_current}>AI Providers</a>'
            f'<a href="/settings/usage"{usage_current}>AI Usage</a></nav>'
        )

    def _page(
        self,
        title: str,
        body: str,
        *,
        current: str | None = None,
        breadcrumbs: list[tuple[str, str]] | None = None,
    ) -> str:
        nav_items = []
        links = [("Dashboard", "/"), ("Test Cases", "/test-cases"), ("Drafts", "/drafts"), ("Runs", "/runs")]
        if self._test_suites is not None:
            links.append(("Test Suites", "/test-suites"))
        if self._llm_usage is not None:
            links.append(("AI Usage", "/settings/usage"))
        if self._provider_settings is not None:
            links.append(("Settings", "/settings/providers"))
        for label, href in links:
            current_attribute = ' aria-current="page"' if label == current else ""
            nav_items.append(
                f'<a href="{href}"{current_attribute}>{escape_html(label)}</a>'
            )
        crumb_html = ""
        if breadcrumbs:
            crumb_html = '<p class="breadcrumbs">' + "<span>›</span>".join(
                f'<a href="{href}">{escape_html(label)}</a>' for label, href in breadcrumbs
            ) + f'<span>›</span>{escape_html(title)}</p>'
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<script src="/assets/ui.js" defer></script>'
            f'<title>{escape_html(title)} · AI QA Agent</title><style>{UI_CSS}</style></head><body>'
            '<header class="topbar"><div class="shell topbar-inner">'
            '<a class="brand" href="/">AI QA Agent</a>'
            '<nav class="nav-links" aria-label="Main navigation">' + "".join(nav_items) + "</nav>"
            '</div></header><main class="shell main">'
            + crumb_html
            + body
            + '</main><footer class="shell">AI QA Agent · Local QA run history</footer></body></html>'
        )

    def _empty_state(self, title: str, detail: str) -> str:
        return f'<div class="empty-state"><strong>{escape_html(title)}</strong><p>{escape_html(detail)}</p></div>'

    def _not_found(self, message: str = "The requested item is not available.") -> WebResponse:
        body = (
            '<div class="error-state"><h1>Not found</h1>'
            f'<p>{escape_html(message)}</p><a class="button" href="/">Back to Dashboard</a></div>'
        )
        return WebResponse.html(404, self._page("Not found", body))


def create_application(
    database_path: str | Path | None = None,
    evidence_directory: str | Path | None = None,
) -> LocalWebApplication:
    storage = create_sqlite_storage(database_path)
    automation_lifecycle = AutomationLifecycleService(
        storage.automation_lifecycle_repository, storage.plan_store
    )
    llm_usage = LLMUsageService(storage.llm_usage_repository)
    evidence_root = (
        Path(evidence_directory).expanduser().resolve()
        if evidence_directory is not None
        else (storage.database_path.parent / ".qa_agent_evidence").resolve()
    )
    provider_settings = ProviderSettingsService(
        ProviderSettingsRepository(storage.database_path),
        create_default_secret_store(),
        usage_recorder=llm_usage,
    )
    automation_router = provider_settings.create_router(usage_recorder=llm_usage)
    automation_workflow = AutomationWorkflow(QATestPipeline(
        decomposer=TestCaseDecomposer(),
        plan_generator=LLMTestPlanGenerator(automation_router),
        runner=BrowserRunner(evidence_root, headless=True),
        plan_store=storage.plan_store,
        execution_repository=storage.execution_repository,
        run_history=storage.run_history,
    ))
    run_service = TestCaseExecutionService(
        storage.test_case_repository,
        storage.plan_store,
        storage.execution_repository,
        storage.run_history,
        evidence_directory=evidence_root,
        automation_workflow=automation_workflow,
        automation_lifecycle=automation_lifecycle,
    )
    return LocalWebApplication(
        storage.run_history,
        evidence_root=evidence_root,
        test_cases=storage.test_case_repository,
        run_service=run_service,
        authoring_service=TestCaseAuthoringService(automation_router),
        plan_store=storage.plan_store,
        provider_settings=provider_settings,
        provider_router=automation_router,
        llm_usage=llm_usage,
        drafts=storage.draft_repository,
        automation_lifecycle=automation_lifecycle,
        automation_lifecycle_repository=storage.automation_lifecycle_repository,
        test_suites=TestSuiteService(
            SQLiteTestSuiteRepository(storage.database_path),
            storage.test_case_repository,
        ),
    )


def create_http_server(
    application: LocalWebApplication,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            response = application.handle("GET", self.path)
            self._send(response)

        def log_message(self, format: str, *args) -> None:
            if should_suppress_successful_progress_log(
                getattr(self, "command", ""), self.path,
                getattr(self, "_response_status", None),
            ):
                return
            super().log_message(format, *args)

        def do_POST(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 8193
            request_path = urlsplit(self.path).path.strip("/").split("/")
            if request_path == ["drafts"] or (
                len(request_path) == 3 and request_path[0] == "drafts"
            ) or request_path == ["test-cases", "manual"] or (
                len(request_path) == 3 and request_path[0] == "test-cases" and request_path[2] == "edit"
            ) or request_path == ["test-cases", "manual", "prepare"]:
                max_body_bytes = _MAX_MANUAL_FORM_BODY_BYTES
            elif (
                len(request_path) == 4
                and request_path[:2] == ["test-cases", "review"]
                and request_path[3] == "save"
            ):
                max_body_bytes = _MAX_DRAFT_SAVE_BODY_BYTES
            elif request_path == ["test-cases", "generate"]:
                max_body_bytes = _MAX_AUTHORING_FORM_BODY_BYTES
            elif request_path == ["test-cases", "export"]:
                max_body_bytes = 64 * 1024
            else:
                max_body_bytes = _MAX_FORM_BODY_BYTES
            if content_length < 0 or content_length > max_body_bytes:
                self.close_connection = True
                body = b"x" * (max_body_bytes + 1)
            else:
                body = self.rfile.read(content_length)
            self._send(application.handle("POST", self.path, body))

        def _send(self, response: WebResponse) -> None:
            self._response_status = response.status
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; img-src 'self'; object-src 'none'",
            )
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.body)

    class ApplicationHTTPServer(ThreadingHTTPServer):
        def server_close(self) -> None:
            try:
                super().server_close()
            finally:
                application.close()

    return ApplicationHTTPServer((host, port), Handler)


def serve(application: LocalWebApplication, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = create_http_server(application, host, port)
    try:
        print(f"AI QA Agent UI listening at http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Browse and run persisted AI QA Agent TestCases.")
    parser.add_argument("--database", default=None, help="SQLite database path")
    parser.add_argument("--evidence-directory", default=None, help="allowed screenshot directory")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (defaults to localhost)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--launcher-instance", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    serve(create_application(args.database, args.evidence_directory), args.host, args.port)
    return 0


def _local_demo_page() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Local registration demo</title></head>
<body><main><h1>Registration demo</h1>
<p>Local-only registration page for exercising saved browser plans.</p>
<form><label>Email <input id="email" type="email" autocomplete="off"></label>
<button type="button">Create account</button></form>
<p>The account confirmation appears after the registration action is connected.</p>
</main></body></html>"""


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None


def should_suppress_successful_progress_log(method: str, target: str, status: int | None) -> bool:
    if method.upper() != "GET" or status is None or not 200 <= status < 300:
        return False
    parts = urlsplit(target).path.strip("/").split("/")
    return (
        len(parts) == 3 and parts[:2] == ["api", "progress"]
    ) or (
        len(parts) == 4 and parts[:3] == ["api", "test-cases", "authoring-progress"]
    )


def _parse_form_body(
    body: bytes | str | None,
    *,
    max_bytes: int = _MAX_FORM_BODY_BYTES,
    allow_repeated_fields: set[str] | None = None,
) -> tuple[dict[str, list[str]], str | None]:
    if isinstance(body, bytes):
        try:
            form_body = body.decode("utf-8")
        except UnicodeDecodeError:
            return {}, "The request was not valid form data."
    else:
        form_body = body or ""
    if len(form_body) > max_bytes:
        return {}, "The request was too large."
    try:
        values = parse_qs(
            form_body,
            keep_blank_values=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=4096,
        )
    except (UnicodeDecodeError, ValueError):
        return {}, "The request was not valid form data."
    allowed_repeated = allow_repeated_fields or set()
    if any(len(items) != 1 and key not in allowed_repeated for key, items in values.items()):
        return {}, "The request contained ambiguous form fields."
    return values, None


def _download_response(content: bytes, filename: str, content_type: str) -> WebResponse:
    safe_name = filename.replace('"', "").replace("\r", "").replace("\n", "")
    return WebResponse(
        200,
        content_type,
        content,
        {"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


def _export_error_response(message: str, status: int) -> WebResponse:
    body = (
        '<div class="error-state"><h1>Export unavailable</h1>'
        f'<p>{escape_html(message)}</p><a class="button" href="/test-cases">Back to TestCases</a></div>'
    )
    return WebResponse.html(status, '<!doctype html><html lang="en"><meta charset="utf-8"><body>' + body + '</body></html>')


def _suite_export_blocker_content(suite_id: UUID, blockers) -> str:
    unique = {}
    for blocker in blockers:
        unique.setdefault(blocker.test_case_id, blocker)
    rows = []
    count = len(unique)
    for blocker in unique.values():
        identity = blocker.public_id or blocker.test_case_name
        step = (
            f'<p>Step {blocker.step_order}</p>'
            if blocker.step_order is not None else ""
        )
        rows.append(
            '<li class="suite-export-blocker"><p><strong>'
            f'{escape_html(identity)}</strong> · {escape_html(blocker.test_case_name)}</p>'
            + step
            + f'<p>{escape_html(blocker.reason)}</p><div class="actions">'
            + f'<a class="button" href="/test-cases/{blocker.test_case_id}">Open TestCase</a>'
            + '</div></li>'
        )
    noun = "TestCase is" if count == 1 else "TestCases are"
    return (
        '<section class="panel error-state">'
        '<h1>Suite export needs attention</h1>'
        f'<p>{count} {noun} not fully automated.</p>'
        f'<ul class="suite-export-blockers">{"".join(rows)}</ul>'
        f'<a class="button" href="/test-suites/{suite_id}">Back to Test Suite</a>'
        '</section>'
    )


def _safe_download_slug(value: str) -> str:
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-._ ")
    return (slug or "test-suite")[:80].rstrip("-._ ")


def _suite_member_form(
    suite_id: UUID,
    case_id: UUID,
    operation: str,
    label: str,
    *,
    accessible_label: str,
    disabled: bool = False,
) -> str:
    disabled_attr = " disabled" if disabled else ""
    return (
        f'<form method="post" action="/test-suites/{suite_id}/members/{operation}">'
        f'<input type="hidden" name="test_case_id" value="{case_id}">'
        f'<button class="button" type="submit" aria-label="{escape_html(accessible_label)}"'
        f' title="{escape_html(accessible_label)}"{disabled_attr}>{escape_html(label)}</button></form>'
    )


def _query_choice(query: dict[str, list[str]], key: str, choices: set[str]) -> str:
    value = query.get(key, ["ALL"])[0].upper()
    return value if value in choices else "ALL"


def _sort_values(values: set[str]) -> list[str]:
    return sorted(values)


def _select(name: str, label: str, selected: str, options: list[str]) -> str:
    option_html = "".join(
        f'<option value="{escape_html(option)}"'
        f'{" selected" if selected == option else ""}>{escape_html("All" if option == "ALL" else outcome_label(option))}</option>'
        for option in options
    )
    return (
        f'<div class="field"><label for="filter-{escape_html(name)}">{escape_html(label)}</label>'
        f'<select id="filter-{escape_html(name)}" name="{escape_html(name)}">{option_html}</select></div>'
    )


def _summary_card(label: str, value: str, *, raw: bool = False) -> str:
    shown = value if raw else escape_html(value)
    return (
        f'<article class="card"><div class="card-label">{escape_html(label)}</div>'
        f'<div class="card-value">{shown}</div></article>'
    )


def _format_usage_cost(value: object) -> str:
    if value is None:
        return "Unknown"
    try:
        return f"${float(value):.6f}"
    except (TypeError, ValueError, OverflowError):
        return "Unknown"


def _format_usage_rate(value: object) -> str:
    if value is None:
        return "Unknown"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError, OverflowError):
        return "Unknown"


def _format_usage_latency(value: object) -> str:
    if value is None:
        return "Unknown"
    try:
        return f"{max(0, int(value))} ms"
    except (TypeError, ValueError, OverflowError):
        return "Unknown"


def _operation_label(operation: str) -> str:
    return {
        "AUTHOR_TESTCASE": "TestCase authoring",
        "GENERATE_AUTOMATION_PLAN": "Automation plan generation",
        "REPAIR_AUTOMATION_PLAN": "Automation plan repair",
        "DISCOVERY": "Discovery",
        "OTHER": "Other",
    }.get(operation, "Other")


def _plan_origin_label(origin: PlanVersionOrigin | None) -> str:
    return {
        PlanVersionOrigin.AI_GENERATED: "Generated by AI",
        PlanVersionOrigin.HUMAN_EDITED: "Edited locally",
        PlanVersionOrigin.REGENERATED: "Updated automatically",
        PlanVersionOrigin.REPAIRED: "Repaired automatically",
    }.get(origin, "Unknown origin")


def _plan_origin_display(value: str) -> str:
    try:
        return _plan_origin_label(PlanVersionOrigin(value))
    except ValueError:
        return "Saved automation" if value == "SAVED" else "Unknown origin"


def _automation_status_label(status: AutomationStatus) -> str:
    return {
        AutomationStatus.NOT_AUTOMATED: "Not automated",
        AutomationStatus.AUTOMATION_READY: "Automation ready",
        AutomationStatus.NEEDS_UPDATE: "Needs update",
        AutomationStatus.NEEDS_VALIDATION: "Needs validation",
        AutomationStatus.AUTOMATION_FAILED: "Automation needs attention",
    }[status]


def _outcome_badge(record: RunHistoryRecord, *, stacked: bool = False) -> str:
    if record.outcome in {record.status.value, "PASSED", "FAILED"}:
        return '<span class="muted">—</span>'
    shown = badge(record.outcome, outcome_tone(record.outcome))
    return f'<div class="row-sub-badge">{shown}</div>' if stacked else shown


def _authoring_error_label(category: str | None) -> str:
    return {
        "AI_RATE_LIMIT": "AI RATE LIMIT",
        "AI_TIMEOUT": "AI TIMEOUT",
        "AI_PROVIDER_ERROR": "AI PROVIDER ERROR",
        "AI_GENERATION_ERROR": "AI GENERATION ERROR",
        "AI_OUTPUT_VALIDATION_ERROR": "AI GENERATION ERROR",
        "INVALID_AUTHORING_INPUT": "INVALID INPUT",
        "AUTHORING_EXECUTION_ERROR": "AUTHORING ERROR",
    }.get(category or "", "AI GENERATION ERROR")


def _provider_diagnostic_html(
    label: str,
    status: str | None,
    category: str | None,
    latency_ms: int | None,
    http_status: int | None,
    retry_after_seconds: int | None,
) -> str:
    category_labels = {
        "AUTH_ERROR": "Authentication failed",
        "MODEL_NOT_FOUND": "Model unavailable",
        "INVALID_REQUEST": "Invalid request",
        "RATE_LIMIT": "Rate limited",
        "TIMEOUT": "Timed out",
        "SCHEMA_ERROR": "Invalid structured output",
        "INVALID_RESPONSE": "Invalid response",
        "PROVIDER_UNAVAILABLE": "Provider unavailable",
        "OTHER_PROVIDER_ERROR": "Provider request failed",
    }
    if status in {None, "not_configured"}:
        value = "Not tested"
    elif status in {"connected", "passed"}:
        value = "Healthy"
    elif status == "configuration_invalid":
        value = "Configuration needs attention"
    elif status == "failed":
        value = "Failed"
    elif status == "authentication_failed":
        value = "Authentication failed"
    elif status == "rate_limited":
        value = "Rate limited"
    elif status == "timeout":
        value = "Timed out"
    elif status == "provider_unavailable":
        value = "Provider unavailable"
    else:
        value = "Failed"
    if category in category_labels and value not in {"Healthy", "Not tested"}:
        category_label = category_labels[category]
        value = (
            f"Failed · {category_label}"
            if status == "failed" else category_label
        )
    if http_status is not None:
        value += f" · HTTP {http_status}"
    if latency_ms is not None:
        value += f" · {max(0, latency_ms)} ms"
    if category == "RATE_LIMIT" and retry_after_seconds is not None:
        value += f" · Try again in {max(1, retry_after_seconds)} seconds"
    elif category == "RATE_LIMIT":
        value += " · Try again later"
    return (
        '<span class="provider-diagnostic" role="status"><strong>'
        f'{escape_html(label)}:</strong> {escape_html(value)}</span>'
    )


def _authoring_field_errors(
    name: str, base_url: str, scenario: str, *, require_name: bool
) -> dict[str, str]:
    errors: dict[str, str] = {}
    if not base_url.strip():
        errors["base_url"] = "Enter the website URL you want to test."
    else:
        try:
            TestCaseAuthoringService.validate_input(
                "Validation sample", "Describe a sample behavior.", base_url
            )
        except TestCaseAuthoringError:
            errors["base_url"] = "Enter a valid URL, for example https://example.com"
    if not scenario.strip() or not any(character.isalnum() for character in scenario):
        errors["scenario"] = "Describe what you want to test."
    elif len(scenario) > 6000:
        errors["scenario"] = "Keep the scenario under 6,000 characters."
    if require_name and not name.strip():
        errors["name"] = "Enter a TestCase name."
    elif len(name.strip()) > 200:
        errors["name"] = "Keep the TestCase name under 200 characters."
    return errors


def _inline_error_html(error_id: str, control_id: str, error: str | None) -> str:
    hidden = " hidden" if not error else ""
    return (
        f'<span class="field-error" id="{escape_html(error_id)}" '
        f'data-inline-error data-error-for="{escape_html(control_id)}" '
        f'role="alert"{hidden}>{escape_html(error or "")}</span>'
    )


def _inline_invalid_attrs(error_id: str, error: str | None) -> str:
    if not error:
        return ""
    return f' class="is-invalid" aria-invalid="true" aria-describedby="{escape_html(error_id)}"'


_UI_JAVASCRIPT = r"""
(() => {
  document.querySelectorAll('[data-select-all]').forEach((master) => {
    master.addEventListener('change', () => {
      const form = master.closest('form');
      form?.querySelectorAll('input[name="test_case_id"]').forEach((checkbox) => {
        checkbox.checked = master.checked;
      });
    });
  });

  const testcaseEditor = document.querySelector('[data-testcase-editor]');
  if (testcaseEditor) {
    const editedIndicator = document.querySelector('[data-edited-indicator]');
    const editableFields = [...testcaseEditor.querySelectorAll('[data-editable-field]')];
    const updateEditedIndicator = () => {
      const edited = editableFields.some((field) =>
        field.value !== field.dataset.originalValue);
      if (editedIndicator) editedIndicator.hidden = !edited;
    };
    testcaseEditor.addEventListener('input', updateEditedIndicator);
    updateEditedIndicator();
  }

  document.querySelectorAll('[data-run-form]').forEach((form) => {
    form.addEventListener('submit', () => {
      form.querySelectorAll('button[type="submit"]').forEach((button) => {
        button.disabled = true;
        button.classList.add('is-disabled');
        button.setAttribute('aria-busy', 'true');
        if (button.textContent.trim() === 'Start run') button.textContent = 'Starting…';
      });
    }, { once: true });
  });

  let generatedInlineValidationControlId = 0;
  document.querySelectorAll('form[data-inline-validation]').forEach((form) => {
    const controls = [...form.querySelectorAll('input[required], textarea[required], select[required]')];
    const ruleFor = (control) => control.dataset.validationRule ||
      (control.name === 'base_url' ? 'website' :
        control.name === 'scenario' || control.name === 'description' || control.name === 'body' ? 'scenario' : 'required');
    const errorFor = (control) => {
      if (!control.id) control.id = `inline-validation-control-${++generatedInlineValidationControlId}`;
      let error = document.querySelector(`[data-error-for="${CSS.escape(control.id)}"]`);
      if (!error) {
        error = document.createElement('span');
        error.className = 'field-error';
        error.dataset.inlineError = '';
        error.dataset.errorFor = control.id;
        error.id = `${control.id}-error`;
        error.setAttribute('role', 'alert');
        (control.closest('.field') || control.parentElement)?.append(error);
      }
      return error;
    };
    const validate = (control) => {
      const value = control.value.trim();
      const rule = ruleFor(control);
      let message = '';
      if (rule === 'website') {
        if (!value) message = 'Enter the website URL you want to test.';
        else {
          try {
            const url = new URL(value);
            if (!['http:', 'https:'].includes(url.protocol) || !url.hostname || url.username || url.password) {
              message = 'Enter a valid URL, for example https://example.com';
            }
          } catch (_error) {
            message = 'Enter a valid URL, for example https://example.com';
          }
        }
      } else if (rule === 'scenario') {
        if (!value || !/[\p{L}\p{N}]/u.test(value)) message = 'Describe what you want to test.';
      } else if (control.required && !value) {
        message = 'Complete this field.';
      }
      const error = errorFor(control);
      if (error) {
        error.textContent = message;
        error.hidden = !message;
      }
      control.classList.toggle('is-invalid', Boolean(message));
      if (message) {
        control.setAttribute('aria-invalid', 'true');
        if (error) control.setAttribute('aria-describedby', error.id);
      } else {
        control.removeAttribute('aria-invalid');
      }
      return message;
    };
    controls.forEach((control) => {
      const existingError = errorFor(control);
      if (existingError && !existingError.hidden && existingError.textContent.trim()) {
        control.classList.add('is-invalid');
        control.setAttribute('aria-invalid', 'true');
        control.setAttribute('aria-describedby', existingError.id);
      }
      control.addEventListener('input', () => {
        const error = errorFor(control);
        if (error && !error.hidden) validate(control);
      });
    });
    form.addEventListener('submit', (event) => {
      const invalid = controls.find((control) => Boolean(validate(control)));
      if (invalid) {
        event.preventDefault();
        event.stopImmediatePropagation();
        invalid.focus();
      }
    });
  });

  document.querySelectorAll('[data-authoring-form]').forEach((form) => {
    form.addEventListener('submit', () => {
      form.querySelectorAll('button[type="submit"]').forEach((button) => {
        button.disabled = true;
        button.classList.add('is-disabled');
        button.setAttribute('aria-busy', 'true');
        button.textContent = 'Starting…';
      });
    }, { once: true });
  });

  document.querySelectorAll('[data-voice-control]').forEach((control) => {
    const button = control.querySelector('[data-voice-button]');
    const label = control.querySelector('[data-voice-label]');
    const status = control.querySelector('[data-voice-status]');
    const scenario = document.getElementById(button?.dataset.voiceTarget || '');
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!button || !label || !status || !scenario) return;

    let voiceState = 'IDLE';
    let recognition = null;
    const setVoiceState = (state, message) => {
      voiceState = state;
      control.dataset.voiceState = state;
      button.disabled = ['UNSUPPORTED', 'LISTENING', 'PROCESSING'].includes(state);
      button.setAttribute('aria-pressed', String(state === 'LISTENING'));
      button.setAttribute('aria-label', state === 'LISTENING' ? 'Listening for scenario' :
        state === 'PROCESSING' ? 'Processing spoken scenario' :
        state === 'UNSUPPORTED' ? 'Voice input unavailable' : 'Speak scenario');
      label.textContent = state === 'LISTENING' ? 'Listening…' :
        state === 'PROCESSING' ? 'Processing…' : 'Speak scenario';
      status.textContent = message;
    };

    setVoiceState('IDLE', '');
    if (!Recognition) {
      setVoiceState('UNSUPPORTED', 'Voice input is not supported in this browser. You can continue typing.');
      return;
    }

    button.addEventListener('click', () => {
      try {
        const activeRecognition = new Recognition();
        recognition = activeRecognition;
        let resultAdded = false;
        activeRecognition.lang = (navigator.languages && navigator.languages[0]) ||
          navigator.language || document.documentElement.lang || '';
        activeRecognition.continuous = false;
        activeRecognition.interimResults = false;
        activeRecognition.maxAlternatives = 1;
        activeRecognition.onstart = () => setVoiceState('LISTENING', 'Listening. Speak your scenario now.');
        activeRecognition.onresult = (event) => {
          if (resultAdded) return;
          setVoiceState('PROCESSING', 'Adding recognized text…');
          const transcripts = [];
          for (let index = event.resultIndex || 0; index < event.results.length; index += 1) {
            const result = event.results[index];
            if (result.isFinal === false) continue;
            const transcript = result[0] && result[0].transcript;
            if (typeof transcript === 'string' && transcript.trim()) transcripts.push(transcript.trim());
          }
          const spokenText = transcripts.join(' ').trim();
          if (!spokenText) {
            setVoiceState('ERROR', 'No speech was recognized. You can continue typing.');
            return;
          }
          resultAdded = true;
          const existingText = scenario.value;
          const separator = existingText && !existingText.endsWith('\n') ? '\n' : '';
          scenario.value = existingText + separator + spokenText;
          scenario.dispatchEvent(new Event('input', { bubbles: true }));
          scenario.focus();
          setVoiceState('IDLE', 'Transcription added. Review and edit the scenario before generating.');
        };
        activeRecognition.onerror = (event) => {
          const messages = {
            'not-allowed': 'Microphone permission was denied. You can continue typing.',
            'service-not-allowed': 'Microphone permission was denied. You can continue typing.',
            'no-speech': 'No speech was recognized. You can continue typing.',
            'audio-capture': 'Microphone is unavailable. You can continue typing.'
          };
          setVoiceState('ERROR', messages[event.error] || 'Voice input could not start. You can continue typing.');
        };
        activeRecognition.onend = () => {
          if (voiceState === 'LISTENING') {
            setVoiceState('IDLE', 'Voice input ended. You can continue typing.');
          }
          if (recognition === activeRecognition) recognition = null;
        };
        setVoiceState('PROCESSING', 'Starting voice input…');
        activeRecognition.start();
      } catch (_error) {
        recognition = null;
        setVoiceState('ERROR', 'Voice input could not start. You can continue typing.');
      }
    });
  });

  const authoringRoot = document.querySelector('.authoring-progress[data-authoring-progress-id]');
  if (authoringRoot) {
    const authoringId = authoringRoot.dataset.authoringProgressId;
    const authoringNotice = authoringRoot.querySelector('[data-authoring-notice]');
    const authoringEvents = authoringRoot.querySelector('[data-authoring-events]');
    const authoringResult = authoringRoot.querySelector('[data-authoring-result]');
    const authoringResultContent = authoringRoot.querySelector('[data-authoring-result-content]');
    const authoringRetry = authoringRoot.querySelector('[data-authoring-retry]');
    let authoringStopped = false;
    let authoringRedirectScheduled = false;
    let elapsedAnchor = null;
    let elapsedTimer = null;

    const renderAuthoringElapsed = (elapsedMs) => {
      const elapsed = authoringRoot.querySelector('[data-authoring-elapsed]');
      if (elapsed) elapsed.textContent = `${(elapsedMs / 1000).toFixed(1)} s`;
    };

    const appendAuthoringEvent = (event, activeType) => {
      const row = document.createElement('li');
      row.className = 'progress-event';
      const symbol = document.createElement('span');
      symbol.setAttribute('aria-hidden', 'true');
      symbol.textContent = event.type.endsWith('FAILED') ? '✕' :
        (event.type === activeType ? '◉' : '✓');
      const label = document.createElement('span');
      label.textContent = event.message;
      row.append(symbol, label);
      authoringEvents.append(row);
    };

    const renderAuthoring = (snapshot) => {
      authoringRoot.querySelectorAll('[data-authoring-phase]').forEach((node) => {
        node.textContent = snapshot.phase;
      });
      if (snapshot.finished) {
        renderAuthoringElapsed(snapshot.elapsed_ms);
        if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
        elapsedTimer = null;
        elapsedAnchor = null;
      } else {
        elapsedAnchor = { milliseconds: snapshot.elapsed_ms, at: performance.now() };
        renderAuthoringElapsed(snapshot.elapsed_ms);
        if (elapsedTimer === null) {
          elapsedTimer = window.setInterval(() => {
            if (!elapsedAnchor || authoringStopped) return;
            renderAuthoringElapsed(
              elapsedAnchor.milliseconds + Math.max(0, performance.now() - elapsedAnchor.at)
            );
          }, 100);
        }
      }
      const provider = authoringRoot.querySelector('[data-authoring-provider]');
      if (provider) provider.textContent = snapshot.provider_name || '—';
      const state = authoringRoot.querySelector('[data-authoring-state]');
      if (state) state.textContent = snapshot.state.toLowerCase().replaceAll('_', ' ')
        .replace(/(^|\s)\S/g, (letter) => letter.toUpperCase());

      const activeType = snapshot.phase === 'Understanding scenario' ? 'AUTHORING_STARTED' :
        snapshot.phase === 'Generating TestCase' ? 'LLM_REQUEST_STARTED' :
        snapshot.phase === 'Validating result' ? 'TESTCASE_VALIDATION_STARTED' : null;
      authoringEvents.replaceChildren();
      snapshot.events.forEach((event) => appendAuthoringEvent(event, activeType));

      if (!snapshot.finished) return;
      authoringStopped = true;
      authoringResult.hidden = false;
      authoringResultContent.replaceChildren();
      if (snapshot.success && snapshot.review_url) {
        const ready = document.createElement('p');
        ready.dataset.authoringResultMessage = '';
        const strong = document.createElement('strong');
        strong.textContent = 'TestCase draft ready.';
        ready.append(strong, document.createTextNode(' Redirecting to Review…'));
        const link = document.createElement('a');
        link.className = 'button primary';
        link.href = snapshot.review_url;
        link.textContent = 'Open Review';
        authoringResultContent.append(ready, link);
        if (authoringRetry) authoringRetry.hidden = true;
        if (!authoringRedirectScheduled) {
          authoringRedirectScheduled = true;
          window.setTimeout(() => window.location.assign(snapshot.review_url), 700);
        }
      } else {
        const heading = document.createElement('p');
        heading.dataset.authoringResultMessage = '';
        const strong = document.createElement('strong');
        const labels = {
          AI_RATE_LIMIT: 'AI RATE LIMIT',
          AI_TIMEOUT: 'AI TIMEOUT',
          AI_PROVIDER_ERROR: 'AI PROVIDER ERROR',
          AI_OUTPUT_VALIDATION_ERROR: 'AI GENERATION ERROR',
          AI_GENERATION_ERROR: 'AI GENERATION ERROR',
          INVALID_AUTHORING_INPUT: 'INVALID INPUT',
          AUTHORING_EXECUTION_ERROR: 'AUTHORING ERROR'
        };
        strong.textContent = labels[snapshot.error_category] || 'AI GENERATION ERROR';
        heading.append(strong);
        const reason = document.createElement('p');
        reason.textContent = `Reason: ${snapshot.error_message || 'An unexpected authoring error occurred.'}`;
        authoringResultContent.append(heading, reason);
        if (Array.isArray(snapshot.provider_failures) && snapshot.provider_failures.length) {
          const details = document.createElement('details');
          details.className = 'technical-details';
          const summary = document.createElement('summary');
          summary.textContent = 'Developer details';
          const list = document.createElement('ul');
          snapshot.provider_failures.forEach((failure) => {
            const item = document.createElement('li');
            const status = failure.http_status ? ` · HTTP ${failure.http_status}` : '';
            const retry = failure.retry_after_seconds ?
              ` · Retry after ${failure.retry_after_seconds} seconds` : '';
            item.textContent = `${failure.provider} — ${failure.category}${status} · ${failure.message}${retry}`;
            list.append(item);
          });
          details.append(summary, list);
          authoringResultContent.append(details);
        }
        const actions = document.createElement('div');
        actions.className = 'actions';
        const addAction = (suffix, label, primary) => {
          const form = document.createElement('form');
          form.method = 'post';
          form.action = `/test-cases/authoring-progress/${encodeURIComponent(authoringId)}/${suffix}`;
          const button = document.createElement('button');
          button.type = 'submit';
          button.className = primary ? 'button primary' : 'button';
          button.textContent = label;
          form.append(button);
          actions.append(form);
        };
        addAction('create-manually', 'Create manually', true);
        addAction('save-draft', 'Save as Draft', false);
        authoringResultContent.append(actions);
        if (authoringRetry) authoringRetry.hidden = false;
      }
    };

    const pollAuthoring = async () => {
      if (authoringStopped) return;
      try {
        const response = await fetch(`/api/test-cases/authoring-progress/${encodeURIComponent(authoringId)}`, {
          headers: { 'Accept': 'application/json' }, cache: 'no-store'
        });
        if (response.status === 404) {
          authoringStopped = true;
          if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
          elapsedTimer = null;
          if (authoringNotice) authoringNotice.textContent = 'Authoring progress is no longer available.';
          return;
        }
        if (!response.ok) throw new Error('Progress unavailable');
        renderAuthoring(await response.json());
      } catch (_error) {
        if (authoringNotice) authoringNotice.textContent = 'Live updates are temporarily unavailable. Retrying…';
      }
      if (!authoringStopped) window.setTimeout(pollAuthoring, 500);
    };
    pollAuthoring();
  }

  const root = document.querySelector('.progress-live[data-progress-id]');
  if (!root) return;
  const progressId = root.dataset.progressId;
  const notice = document.querySelector('[data-progress-notice]');
  const stepsList = document.querySelector('#progress-steps');
  const result = document.querySelector('[data-progress-result]');
  const resultContent = document.querySelector('[data-progress-result-content]');
  let stopped = false;
  let redirectScheduled = false;

  const eventGroups = {
    all: new Set(['RUN_REQUESTED', 'RUN_STARTED', 'TESTCASE_LOADED', 'SETUP_STARTED', 'SETUP_SUCCEEDED', 'SETUP_FAILED', 'AUTOMATION_PREPARATION_STARTED', 'PLAN_REUSED', 'PLAN_GENERATION_STARTED', 'PLAN_REPAIR_STARTED', 'PLAN_REPAIR_SUCCEEDED', 'PLAN_REPAIR_FAILED', 'PLAN_GENERATED', 'PLAN_GENERATION_FAILED', 'STEP_STARTED', 'STEP_PASSED', 'STEP_FAILED', 'STEP_BLOCKED', 'EVIDENCE_CAPTURED', 'CLEANUP_STARTED', 'CLEANUP_SUCCEEDED', 'CLEANUP_FAILED', 'RUN_FINISHED'])
  };
  const stepStates = {
    PENDING: ['○', 'Pending'], PREPARING_AUTOMATION: ['◌', 'Preparing automation'],
    READY: ['✓', 'Automation ready'], RUNNING: ['◉', 'Running'], PASSED: ['✓', 'Passed'],
    FAILED: ['✕', 'Failed'], BLOCKED: ['⊘', 'Blocked'], NOT_ATTEMPTED: ['○', 'Not attempted']
  };

  function appendEvent(list, event) {
    const row = document.createElement('li');
    row.className = 'progress-event';
    const symbol = document.createElement('span');
    symbol.setAttribute('aria-hidden', 'true');
    symbol.textContent = event.type.endsWith('FAILED') ? '✕' :
      (event.type.endsWith('STARTED') ? '◉' : '✓');
    const text = document.createElement('span');
    text.textContent = event.message || event.type.replaceAll('_', ' ').toLowerCase();
    row.append(symbol, text);
    list.append(row);
  }

  function render(snapshot) {
    document.querySelectorAll('[data-progress-phase]').forEach((node) => {
      node.textContent = snapshot.phase;
    });
    document.querySelectorAll('[data-progress-state]').forEach((node) => {
      node.textContent = snapshot.state.toLowerCase().replaceAll('_', ' ')
        .replace(/(^|\s)\S/g, (letter) => letter.toUpperCase());
    });
    const elapsed = document.querySelector('[data-progress-elapsed]');
    if (elapsed) elapsed.textContent = `${(snapshot.elapsed_ms / 1000).toFixed(1)} s`;

    Object.entries(eventGroups).forEach(([name, types]) => {
      const list = document.querySelector(`[data-progress-events="${name}"]`);
      if (!list) return;
      list.replaceChildren();
      snapshot.events.filter((event) => types.has(event.type)).forEach((event) => appendEvent(list, event));
    });

    stepsList.replaceChildren();
    snapshot.steps.forEach((step) => {
      const item = document.createElement('li');
      item.className = `progress-step state-${step.state.toLowerCase()}`;
      const icon = document.createElement('span');
      icon.className = 'progress-symbol';
      icon.setAttribute('aria-hidden', 'true');
      const state = stepStates[step.state] || ['○', step.state];
      icon.textContent = state[0];
      const content = document.createElement('span');
      const name = document.createElement('strong');
      name.textContent = `Step ${step.order + 1}: ${step.name}`;
      const label = document.createElement('span');
      label.className = 'progress-step-state';
      label.textContent = state[1];
      const execution = document.createElement('span');
      execution.className = 'progress-step-state';
      const executionState = step.execution_state || 'PENDING';
      execution.textContent = `Execution ${executionState.toLowerCase().replaceAll('_', ' ')}`;
      content.append(name, label, execution);
      if (step.automation_state) {
        const automation = document.createElement('span');
        automation.className = 'muted';
        automation.textContent = `Automation ${step.automation_state.toLowerCase()}`;
        content.append(automation);
      } else {
        const automation = document.createElement('span');
        automation.className = 'muted';
        automation.textContent = 'Automation not prepared';
        content.append(automation);
      }
      if (step.plan_origin) {
        const provenance = document.createElement('span');
        provenance.className = 'muted';
        provenance.textContent = `Plan ${step.plan_origin.toLowerCase().replaceAll('_', ' ')}${step.plan_version ? ` v${step.plan_version}` : ''}`;
        content.append(provenance);
      }
      if (step.failure_classification) {
        const failure = document.createElement('span');
        failure.className = 'progress-failure';
        failure.textContent = `${step.failure_classification.toLowerCase().replaceAll('_', ' ')}: ${step.message || 'Step failed.'}`;
        content.append(failure);
      }
      const evidenceCount = step.evidence_count || 0;
      if (evidenceCount) {
        const evidence = document.createElement('span');
        evidence.className = 'progress-evidence';
        evidence.textContent = `Screenshot evidence captured (${evidenceCount})`;
        content.append(evidence);
      }
      item.append(icon, content);
      stepsList.append(item);
    });

    if (snapshot.finished) {
      result.hidden = false;
      const category = snapshot.outcome || snapshot.error_category || 'FAILED';
      const failure = snapshot.automation_generation_failure;
      if (category === 'AUTOMATION_GENERATION_ERROR' && failure) {
        const summary = document.createElement('p');
        summary.dataset.resultSummary = '';
        const strong = document.createElement('strong');
        strong.textContent = 'AUTOMATION GENERATION ERROR';
        summary.append(strong, document.createTextNode(
          ` Automation stopped at Step ${failure.step_order + 1} of ${snapshot.steps.length}.`));
        const generatedCount = snapshot.steps.filter((step) =>
          step.order < failure.step_order && ['Generated', 'Reused', 'Repaired'].includes(step.automation_state)).length;
        const remainingCount = snapshot.steps.filter((step) => step.state === 'NOT_ATTEMPTED').length;
        const generated = document.createElement('p');
        generated.textContent = `Generated successfully: ${generatedCount} steps.`;
        const remaining = document.createElement('p');
        remaining.textContent = `Remaining: ${remainingCount} steps not attempted.`;
        const stopped = document.createElement('p');
        stopped.textContent = `Not attempted because automation generation stopped at Step ${failure.step_order + 1}.`;
        const reason = document.createElement('p');
        reason.textContent = `Reason: ${failure.safe_reason}`;
        const saved = document.createElement('p');
        saved.textContent = `Saved automation: ${failure.prior_plan_exists ? 'Available from a previous attempt.' : 'Not available for this step.'}`;
        const newPlan = document.createElement('p');
        newPlan.textContent = `New plan saved: ${failure.new_plan_saved ? 'Yes.' : 'No.'}`;
        const technical = document.createElement('details');
        technical.className = 'technical-details';
        const technicalSummary = document.createElement('summary');
        technicalSummary.textContent = 'Developer details';
        const classification = document.createElement('p');
        classification.textContent = `Classification: ${failure.technical_classification}`;
        technical.append(technicalSummary, classification);
        if (failure.validation_issues && failure.validation_issues.length) {
          const issueList = document.createElement('ul');
          issueList.className = 'validation-issues';
          failure.validation_issues.forEach((issue) => {
            const item = document.createElement('li');
            item.textContent = `${issue.code} at ${issue.path}: ${issue.message}`;
            issueList.append(item);
          });
          technical.append(issueList);
        }
        resultContent.replaceChildren(summary, generated, remaining, stopped, reason, saved, newPlan, technical);
      } else {
        const summary = document.createElement('p');
        summary.dataset.resultSummary = '';
        const strong = document.createElement('strong');
        strong.textContent = category.replaceAll('_', ' ');
        const explanation = document.createTextNode(` ${snapshot.error_message || 'The run has finished.'}`);
        summary.append(strong, explanation);
        resultContent.replaceChildren(summary);
      }
      if (snapshot.final_run_url) {
        const link = document.createElement('a');
        link.className = 'button primary';
        link.href = snapshot.final_run_url;
        link.textContent = 'View Run Details';
        resultContent.append(link);
        if (!redirectScheduled) {
          redirectScheduled = true;
          window.setTimeout(() => window.location.assign(snapshot.final_run_url), snapshot.redirect_after_ms || 1200);
        }
      } else if (notice) {
        notice.textContent = 'No completed Run History record is available for this request.';
      }
      const retryable = ['INFRASTRUCTURE_ERROR', 'AUTOMATION_EXECUTION_ERROR', 'EXECUTION_ERROR', 'AUTOMATION_GENERATION_ERROR', 'SETUP_FAILURE'];
      if (retryable.includes(snapshot.error_category) && !document.querySelector('[data-progress-retry]')) {
        const form = document.createElement('form');
        form.method = 'post';
        form.action = `/test-cases/${snapshot.test_case_id}/run`;
        form.dataset.runForm = '';
        form.dataset.progressRetry = '';
        const workflow = document.createElement('input');
        workflow.type = 'hidden';
        workflow.name = 'workflow';
        workflow.value = snapshot.workflow;
        const button = document.createElement('button');
        button.className = 'button';
        button.type = 'submit';
        button.textContent = snapshot.error_category === 'AUTOMATION_GENERATION_ERROR'
          ? 'Retry Automation' : 'Retry Run';
        form.append(workflow, button);
        const actions = document.createElement('div');
        actions.className = 'actions';
        actions.append(form);
        result.after(actions);
        form.addEventListener('submit', () => {
          button.disabled = true;
          button.classList.add('is-disabled');
        }, { once: true });
      }
      stopped = true;
    }
  }

  async function poll() {
    if (stopped) return;
    try {
      const response = await fetch(`/api/progress/${encodeURIComponent(progressId)}`, {
        headers: { 'Accept': 'application/json' },
        cache: 'no-store'
      });
      if (response.status === 404) {
        stopped = true;
        if (notice) notice.textContent = 'Execution progress is no longer available.';
        return;
      }
      if (!response.ok) throw new Error('Progress unavailable');
      render(await response.json());
    } catch (_error) {
      if (notice) notice.textContent = 'Live updates are temporarily unavailable. Retrying…';
    }
    if (!stopped) window.setTimeout(poll, 750);
  }
  poll();
})();
"""


if __name__ == "__main__":
    raise SystemExit(main())
