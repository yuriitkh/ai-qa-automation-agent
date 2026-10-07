"""Local UI for persisted TestCases and TestRun history."""

import argparse
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

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
from qa_agent.llm.registry import create_router
from qa_agent.run_history import RunHistoryDetail, RunHistoryRecord, RunHistoryService, WorkflowType
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
from qa_agent.pipeline import QATestPipeline
from qa_agent.background_authoring import BackgroundAuthoringService
from qa_agent.background_execution import BackgroundRunService
from qa_agent.execution_progress import (
    AuthoringEventType,
    ExecutionProgressStore,
)
from qa_agent.models import PlanVersionOrigin
from qa_agent.plan_store import PlanStore
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.workflows import AutomationWorkflow


_PAGE_LIMIT = 500
logger = logging.getLogger(__name__)
_MAX_FORM_BODY_BYTES = 8192
_MAX_AUTHORING_FORM_BODY_BYTES = 80_000
_MAX_DRAFT_SAVE_BODY_BYTES = 256 * 1024
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
        if len(parts) == 3 and parts[:2] == ["api", "progress"]:
            return self._progress_json(parts[2])
        if len(parts) == 3 and parts[:2] == ["runs", "progress"]:
            return self._progress_page(parts[2])
        if len(parts) == 4 and parts[:3] == ["api", "test-cases", "authoring-progress"]:
            return self._authoring_progress_json(parts[3])
        if len(parts) == 3 and parts[:2] == ["test-cases", "authoring-progress"]:
            return self._authoring_progress_page(parts[2])
        if path == "/":
            return WebResponse.html(200, self._dashboard())
        if path == "/demo-target/registration":
            return WebResponse.html(200, _local_demo_page())
        if path == "/test-cases":
            return WebResponse.html(200, self._test_case_list())
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
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "run":
            return WebResponse.html(405, self._page(
                "Method not allowed",
                '<div class="error-state"><h1>POST required</h1>'
                "<p>Starting a run requires submitting the TestCase form.</p></div>",
            ))
        return self._not_found("Page not found")

    def _handle_post(self, parts: list[str], body: bytes | str | None) -> WebResponse:
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
        if snapshot.finished:
            if snapshot.success:
                result = (
                    '<p data-authoring-result-message><strong>TestCase draft ready.</strong> '
                    'Redirecting to Review…</p>'
                    f'<a class="button primary" data-authoring-review-link '
                    f'href="{escape_html(snapshot.review_url or "")}">Open Review</a>'
                )
            else:
                result = (
                    f'<p data-authoring-result-message><strong>{_authoring_error_label(snapshot.error_category)}</strong></p>'
                    f'<p>{escape_html(snapshot.error_message or "An unexpected authoring error occurred.")}</p>'
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
            + '</div>'
            + f'<div class="authoring-progress" data-authoring-progress-id="{escape_html(progress_id)}">'
            + '<section class="panel"><h2>Progress</h2>'
            + f'<ul class="compact-list" data-authoring-events>{events}</ul></section>'
            + '<section class="panel progress-result" data-authoring-result'
            + ('' if snapshot.finished else ' hidden')
            + f'><div data-authoring-result-content>{result}</div>{retry_form}</section>'
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
                    f'<p>Classification: <code>{escape_html(failure.technical_classification)}</code></p></details>'
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
            automation_state = (
                f'<span class="muted">Automation {escape_html(step.automation_state.casefold())}</span>'
                if step.automation_state else ""
            )
            evidence_count = sum(
                event.event_type.value == "EVIDENCE_CAPTURED"
                and event.step_id == step.id
                for event in snapshot.events
            )
            evidence = (
                f'<span class="progress-evidence">Screenshot evidence captured ({evidence_count})</span>'
                if evidence_count else ""
            )
            step_rows.append(
                f'<li class="progress-step state-{escape_html(step.state.value.casefold())}">'
                f'<span class="progress-symbol" aria-hidden="true">{symbol}</span>'
                f'<span><strong>Step {step.order + 1}: {escape_html(step.name)}</strong>'
                f'<span class="progress-step-state">{escape_html(label)}</span>{automation_state}{evidence}</span></li>'
            )

        events = snapshot.events
        preparation = self._progress_event_list(events, {
            "TESTCASE_LOADED", "SETUP_STARTED", "SETUP_SUCCEEDED", "SETUP_FAILED",
        }, "preparation")
        automation = self._progress_event_list(events, {
            "AUTOMATION_PREPARATION_STARTED", "PLAN_REUSED", "PLAN_GENERATION_STARTED",
            "PLAN_GENERATED", "PLAN_GENERATION_FAILED",
        }, "automation")
        cleanup = self._progress_event_list(events, {
            "CLEANUP_STARTED", "CLEANUP_SUCCEEDED", "CLEANUP_FAILED",
        }, "cleanup")
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
            + '<section class="panel"><h2>Preparation</h2>'
            + (preparation if preparation else '<ul class="compact-list" data-progress-events="preparation"></ul>')
            + '</section><section class="panel"><h2>Automation</h2>'
            + (automation if automation else '<ul class="compact-list" data-progress-events="automation"></ul>')
            + '</section><section class="panel"><h2>Execution</h2>'
            + (
                f'<ol class="progress-steps" id="progress-steps">{"".join(step_rows)}</ol>'
                if step_rows else '<ol class="progress-steps" id="progress-steps"></ol>'
            )
            + '</section><section class="panel"><h2>Cleanup</h2>'
            + (cleanup if cleanup else '<ul class="compact-list" data-progress-events="cleanup"></ul>')
            + '</section></div>'
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

    def _handle_generate_test_case(self, body: bytes | str | None) -> WebResponse:
        form, form_error = _parse_form_body(body, max_bytes=_MAX_AUTHORING_FORM_BODY_BYTES)
        values = {
            key: form.get(key, [""])[0]
            for key in ("name", "base_url", "scenario")
        }
        if form_error is not None:
            return WebResponse.html(400, self._new_test_case_page(form_error, **values))
        try:
            validated = TestCaseAuthoringService.validate_input(**values)
        except TestCaseAuthoringError as error:
            return WebResponse.html(400, self._new_test_case_page(str(error), **values))
        except Exception:
            return WebResponse.html(400, self._new_test_case_page(
                "Enter a valid name, Base URL, and scenario.", **values
            ))
        if self._test_cases is None:
            return WebResponse.html(503, self._new_test_case_page(
                "TestCase storage is not configured.", **values
            ))
        if self._authoring_service is None or self._background_authoring is None:
            return WebResponse.html(503, self._new_test_case_page(
                "AI generation is unavailable. Configure an LLM provider in the environment and try again.",
                **values,
            ))
        try:
            progress_id = self._background_authoring.start(
                name=validated.name,
                base_url=validated.base_url,
                scenario=validated.scenario,
            )
        except Exception:
            logger.error("Could not start authoring request")
            return WebResponse.html(503, self._new_test_case_page(
                "Authoring could not be started. Try again later.", **values
            ))
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

    def _dashboard(self) -> str:
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
        content = (
            '<header class="page-heading"><h1>AI QA Agent</h1>'
            '<p class="lead">A clear view of recent automation, validation, and regression runs.</p></header>'
            f'<section aria-label="Run summary"><div class="summary-grid">{cards}</div>'
            '<p class="muted">Summary of the latest recorded history.</p></section>'
            '<section class="panel"><div class="section-heading"><h2>Recent runs</h2>'
            '<a class="button" href="/runs">View all runs</a></div>'
            + self._runs_table(recent)
            + ("" if recent else self._empty_state(
                "No runs recorded yet.",
                "Seed demo data or execute a workflow to see results here.",
            ))
            + '</section><section class="panel"><div class="section-heading"><h2>TestCases</h2>'
            '<a class="button" href="/test-cases">Browse TestCases</a></div>'
            '<p class="muted">Browse saved definitions, including TestCases that have not run yet.</p></section>'
        )
        return self._page("Dashboard", content, current="Dashboard")

    def _new_test_case_page(
        self,
        error: str | None = None,
        *,
        name: str = "",
        base_url: str = "",
        scenario: str = "",
    ) -> str:
        error_html = (
            f'<div class="error-state"><p>{escape_html(error)}</p></div>' if error else ""
        )
        content = (
            '<header class="page-heading"><h1>New Test Case</h1>'
            '<p class="lead">Describe the scenario. AI will propose steps for you to review.</p></header>'
            + error_html
            + '<section class="panel"><form method="post" action="/test-cases/generate" data-authoring-form>'
            + '<div class="field"><label for="case-name">Name</label>'
            + f'<input id="case-name" name="name" maxlength="200" required value="{escape_html(name)}"></div>'
            + '<div class="field"><label for="case-base-url">Base URL</label>'
            + f'<input id="case-base-url" name="base_url" type="url" maxlength="2048" required value="{escape_html(base_url)}" placeholder="http://127.0.0.1:8000/demo-target/registration"></div>'
            + '<div class="field"><label for="case-scenario">Scenario</label>'
            + f'<textarea id="case-scenario" name="scenario" rows="6" maxlength="6000" required>{escape_html(scenario)}</textarea></div>'
            + '<button class="button primary" type="submit">Generate Test with AI</button>'
            + '</form></section>'
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
                outcome = _outcome_badge(latest) if latest else '<span class="muted">—</span>'
                last_run = escape_html(format_timestamp(latest.started_at)) if latest else "—"
                latest_workflow = (
                    badge(latest.workflow_type.value, "workflow")
                    if latest else '<span class="muted">—</span>'
                )
                rows.append(
                    "<tr>"
                    f'<td><div class="status-line"><span class="id-code">{escape_html(test_case.public_id or "")}</span>'
                    f'<a href="/test-cases/{test_case_id}">{escape_html(test_case.name)}</a></div>'
                    f'<details><summary>Technical ID</summary><code>{escape_html(test_case_id)}</code></details></td>'
                    f"<td>{status}</td><td>{outcome}</td><td>{last_run}</td>"
                    f"<td>{len(history)}</td><td>{latest_workflow}</td></tr>"
                )
        else:
            for test_case_id, latest in latest_by_case.items():
                history = grouped[test_case_id]
                rows.append(
                    "<tr>"
                    f'<td><div class="status-line"><span class="id-code">{escape_html(latest.test_case_public_id or "")}</span>'
                    f'<a href="/test-cases/{test_case_id}">{escape_html(latest.test_case_name)}</a></div>'
                    f'<details><summary>Technical ID</summary><code>{escape_html(test_case_id)}</code></details></td>'
                    f'<td>{badge(latest.status.value)}</td>'
                    f'<td>{_outcome_badge(latest)}</td>'
                    f'<td>{escape_html(format_timestamp(latest.started_at))}</td>'
                    f'<td>{len(history)}</td>'
                    f'<td>{badge(latest.workflow_type.value, "workflow")}</td>'
                    "</tr>"
                )
        table = (
            '<div class="table-wrap"><table><thead><tr><th>TestCase</th><th>Latest status</th>'
            '<th>Latest result</th><th>Last run</th><th>Runs</th><th>Latest workflow</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
        content = (
            '<header class="page-heading section-heading"><div><h1>Test Cases</h1>'
            '<p class="lead">Saved TestCase definitions and their run history.</p></div>'
            '<a class="button primary" href="/test-cases/new">+ New Test Case</a></header>'
            '<section class="panel"><h2>TestCases</h2>'
            + (table if rows else self._empty_state(
                "No TestCases found.", "Create a TestCase from a natural-language scenario to get started."
            ))
            + "</section>"
        )
        return self._page("Test Cases", content, current="Test Cases", breadcrumbs=[("Dashboard", "/")])

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
        content = (
            '<header class="page-heading">'
            + (f'<p class="eyebrow">{escape_html(case_public_id)}</p>' if case_public_id else '')
            + '<h1>' + escape_html(case_name) + '</h1>'
            + f'<p class="lead">{escape_html(case_description)}</p>'
            + f'<p class="muted">{escape_html(source_note)}</p></header>'
            + f'<details class="technical-details"><summary>Technical IDs</summary>'
            + f'<p>TestCase UUID: <code>{escape_html(test_case_id)}</code></p></details>'
            + f'<div class="summary-grid">{cards}</div>'
            + run_form
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

    def _page(
        self,
        title: str,
        body: str,
        *,
        current: str | None = None,
        breadcrumbs: list[tuple[str, str]] | None = None,
    ) -> str:
        nav_items = []
        for label, href in (("Dashboard", "/"), ("Test Cases", "/test-cases"), ("Runs", "/runs")):
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
    evidence_root = (
        Path(evidence_directory).expanduser().resolve()
        if evidence_directory is not None
        else (storage.database_path.parent / ".qa_agent_evidence").resolve()
    )
    automation_router = create_router()
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
    )
    return LocalWebApplication(
        storage.run_history,
        evidence_root=evidence_root,
        test_cases=storage.test_case_repository,
        run_service=run_service,
        authoring_service=TestCaseAuthoringService(create_router()),
        plan_store=storage.plan_store,
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

        def do_POST(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 8193
            request_path = urlsplit(self.path).path.strip("/").split("/")
            if (
                len(request_path) == 4
                and request_path[:2] == ["test-cases", "review"]
                and request_path[3] == "save"
            ):
                max_body_bytes = _MAX_DRAFT_SAVE_BODY_BYTES
            elif request_path == ["test-cases", "generate"]:
                max_body_bytes = _MAX_AUTHORING_FORM_BODY_BYTES
            else:
                max_body_bytes = _MAX_FORM_BODY_BYTES
            if content_length < 0 or content_length > max_body_bytes:
                self.close_connection = True
                body = b"x" * (max_body_bytes + 1)
            else:
                body = self.rfile.read(content_length)
            self._send(application.handle("POST", self.path, body))

        def _send(self, response: WebResponse) -> None:
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


def _parse_form_body(
    body: bytes | str | None,
    *,
    max_bytes: int = _MAX_FORM_BODY_BYTES,
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
    if any(len(items) != 1 for items in values.values()):
        return {}, "The request contained ambiguous form fields."
    return values, None


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


def _plan_origin_label(origin: PlanVersionOrigin | None) -> str:
    return {
        PlanVersionOrigin.AI_GENERATED: "AI generated",
        PlanVersionOrigin.HUMAN_EDITED: "Human edited",
        PlanVersionOrigin.REGENERATED: "Regenerated",
        PlanVersionOrigin.REPAIRED: "Repaired",
    }.get(origin, "Unknown origin")


def _outcome_badge(record: RunHistoryRecord, *, stacked: bool = False) -> str:
    if record.outcome in {record.status.value, "PASSED", "FAILED"}:
        return '<span class="muted">—</span>'
    shown = badge(record.outcome, outcome_tone(record.outcome))
    return f'<div class="row-sub-badge">{shown}</div>' if stacked else shown


def _authoring_error_label(category: str | None) -> str:
    return {
        "AI_RATE_LIMIT": "AI RATE LIMIT",
        "AI_PROVIDER_ERROR": "AI PROVIDER ERROR",
        "AI_GENERATION_ERROR": "AI GENERATION ERROR",
        "AI_OUTPUT_VALIDATION_ERROR": "AI GENERATION ERROR",
        "INVALID_AUTHORING_INPUT": "INVALID INPUT",
        "AUTHORING_EXECUTION_ERROR": "AUTHORING ERROR",
    }.get(category or "", "AI GENERATION ERROR")


_UI_JAVASCRIPT = r"""
(() => {
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
      const elapsed = authoringRoot.querySelector('[data-authoring-elapsed]');
      if (elapsed) elapsed.textContent = `${(snapshot.elapsed_ms / 1000).toFixed(1)} s`;
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
    preparation: new Set(['TESTCASE_LOADED', 'SETUP_STARTED', 'SETUP_SUCCEEDED', 'SETUP_FAILED']),
    automation: new Set(['AUTOMATION_PREPARATION_STARTED', 'PLAN_REUSED', 'PLAN_GENERATION_STARTED', 'PLAN_GENERATED', 'PLAN_GENERATION_FAILED']),
    cleanup: new Set(['CLEANUP_STARTED', 'CLEANUP_SUCCEEDED', 'CLEANUP_FAILED'])
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
      content.append(name, label);
      if (step.automation_state) {
        const automation = document.createElement('span');
        automation.className = 'muted';
        automation.textContent = `Automation ${step.automation_state.toLowerCase()}`;
        content.append(automation);
      }
      const evidenceCount = snapshot.events.filter((event) =>
        event.type === 'EVIDENCE_CAPTURED' && event.step_id === step.id).length;
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
