"""Local UI for persisted TestCases and TestRun history."""

import argparse
import json
import re
import logging
from secrets import token_urlsafe, compare_digest
import mimetypes
from dataclasses import dataclass, field
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit
from uuid import UUID, uuid4

from qa_agent.execution_preferences import SQLiteExecutionPreferences, InMemoryExecutionPreferences, PreferencesConflict
from qa_agent.presentation import (
    UI_CSS,
    badge,
    escape_html,
    format_duration,
    format_timestamp,
    failure_message,
    outcome_label,
    outcome_tone,
    result_badge,
)
from qa_agent.result_semantics import history_outcome, result_label, result_outcome, terminal_phase
from qa_agent.reliability import AutomationReliabilitySupervisor, ReliabilitySettings
from qa_agent.cookie_consent import (
    DEFAULT_COOKIE_CONSENT_POLICY,
    CookieConsentPolicy,
    cookie_consent_policy_label,
)
from qa_agent.reporting import RunAttemptReport, RunEvidenceReport, RunReportGenerator, RunStepReport
from qa_agent.browser_runner import BrowserRunner
from qa_agent.evidence_policy import (
    EvidenceMode,
    EvidencePolicy,
    ScreenshotMode,
    evidence_mode_label,
    screenshot_mode_label,
)
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
    has_complete_plans,
)
from qa_agent.expected_result_coverage import expected_result_coverage
from qa_agent.drafts import Draft, DraftRepository, DraftStatus, InMemoryDraftRepository
from qa_agent.test_case_review import (
    InMemoryTestCaseReviewRepository,
    TestCaseReviewService,
    TestCaseReviewStatus,
)
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
    scenario_input_error,
    _derive_test_case_name,
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
from qa_agent.models import PlanVersionOrigin, QATestStep, TestCase, TestPlan, TestPlanVersion
from qa_agent.automation_editor import (
    ACTION_EDITOR_FIELDS,
    ACTION_LABELS,
    FIELD_LABELS,
    AutomationEditorError,
    plan_from_form,
)
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
from qa_agent.suite_runs import (
    AIPolicy,
    SuiteEligibility,
    SuiteRun,
    SuiteRunConfig,
    SuiteRunItemStatus,
    SuiteRunService,
    SuiteRunStatus,
    suite_run_report_json,
    suite_item_outcome,
    suite_status_label,
    suite_item_label,
    suite_attempt_outcome,
)
from qa_agent.suite_run_storage import SQLiteSuiteRunRepository
from qa_agent.readiness import (
    ReadinessCheck,
    ReadinessReport,
    ReadinessStatus,
    SystemReadinessService,
)
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_case_editing import TestCaseEditError, create_manual_test_case, edit_test_case
from qa_agent.test_case_naming import step_display_name
from qa_agent.workflows import AutomationWorkflow


_PAGE_LIMIT = 500
_EXPORT_PAGE_SIZE = 100
logger = logging.getLogger(__name__)
_MAX_FORM_BODY_BYTES = 8192
_MAX_AUTHORING_FORM_BODY_BYTES = 80_000
_MAX_DRAFT_SAVE_BODY_BYTES = 256 * 1024
_MAX_MANUAL_FORM_BODY_BYTES = 1024 * 1024
_WORKFLOWS = {item.value for item in WorkflowType}
_STATUSES = {"CANCELLED", "PASSED", "FAILED", "PRODUCT_FAILURE", "AUTOMATION_EXECUTION_ERROR", "AUTOMATION_DRIFT", "AUTOMATION_GENERATION_ERROR", "INFRASTRUCTURE_ERROR", "BLOCKED", "INCONCLUSIVE"}
_FAILURE_TYPES = {
    "PRODUCT_FAILURE",
    "AUTOMATION_DRIFT",
    "AUTOMATION_EXECUTION_ERROR",
    "AUTOMATION_GENERATION_ERROR",
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
        suite_run_service: SuiteRunService | None = None,
        test_case_review: TestCaseReviewService | None = None,
        database_path: str | Path | None = None,
        readiness_service: SystemReadinessService | None = None,
        reliability: AutomationReliabilitySupervisor | None = None,
        execution_preferences=None,
    ) -> None:
        self._csrf_token = token_urlsafe(32)
        self._execution_preferences = execution_preferences or (SQLiteExecutionPreferences(database_path) if database_path is not None else InMemoryExecutionPreferences())
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
        self._reliability = reliability
        self._test_suites = test_suites or TestSuiteService(
            InMemoryTestSuiteRepository(), test_cases
        )
        self._suite_run_service = suite_run_service
        self._test_case_review = test_case_review or TestCaseReviewService(
            InMemoryTestCaseReviewRepository(), plan_store or InMemoryPlanStore()
        )
        self._testplan_exports = (
            TestPlanExportService(test_cases, plan_store)
            if test_cases is not None and plan_store is not None else None
        )
        self._background_runs = (
            BackgroundRunService(run_service, run_history, self._progress_store)
            if run_service is not None else None
        )
        self._readiness_service = readiness_service
        if self._readiness_service is None and database_path is not None:
            self._readiness_service = SystemReadinessService(
                database_path=database_path,
                evidence_directory=self._evidence_root,
                application_ready=all((
                    self._run_history is not None,
                    self._test_cases is not None,
                    self._run_service is not None,
                    self._plan_store is not None,
                    self._background_runs is not None,
                    self._suite_run_service is not None,
                )),
                run_service=self._run_service,
                background_runs=self._background_runs,
                suite_run_service=self._suite_run_service,
                provider_settings=self._provider_settings,
                test_cases=self._test_cases,
                automation_lifecycle=self._automation_lifecycle,
                test_case_review=self._test_case_review,
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
        if self._reliability is not None:
            self._reliability.cancel_all()
        if self._background_runs is not None:
            self._background_runs.close()
        if self._suite_run_service is not None:
            self._suite_run_service.close()
        if self._background_authoring is not None:
            self._background_authoring.close()

    def _valid_control_request(self, body, headers, *, max_bytes=_MAX_FORM_BODY_BYTES) -> bool:
        request_headers = {key.lower(): value for key, value in (headers or {}).items()}
        form, error = _parse_form_body(body, max_bytes=max_bytes)
        if error:
            return False
        token = request_headers.get("x-qa-csrf", form.get("_csrf", [""])[0])
        if not isinstance(token, str) or not token.isascii() or not compare_digest(token, self._csrf_token):
            return False
        if request_headers.get("sec-fetch-site") == "cross-site":
            return False
        host = request_headers.get("host")
        if host:
            try:
                if urlsplit("http://" + host).hostname not in {"localhost", "127.0.0.1", "::1"}:
                    return False
            except ValueError:
                return False
        origin = request_headers.get("origin")
        if origin and (not host or origin != "http://" + host):
            return False
        return True

    def _handle_preferences(self, raw_id, body) -> WebResponse:
        case_id = _parse_uuid(raw_id)
        if case_id is None or self._test_cases is None or self._test_cases.get(case_id) is None:
            return WebResponse.json(404, json.dumps({"error": "TestCase not found."}))
        form, error = _parse_form_body(body)
        try:
            if error or set(form) - {"field", "value", "revision", "_csrf"} or any(len(values) != 1 for values in form.values()):
                raise ValueError("Invalid preferences request.")
            revision = form.get("revision", [""])[0]
            if not revision.isascii() or not revision.isdecimal():
                raise ValueError("A valid preferences revision is required.")
            saved = self._execution_preferences.update(case_id, form.get("field", [""])[0], form.get("value", [""])[0], int(revision))
            return WebResponse.json(200, saved.model_dump_json())
        except PreferencesConflict as conflict:
            return WebResponse.json(409, json.dumps({"error": str(conflict), "preferences": conflict.current.model_dump(mode="json")}))
        except (ValueError, TypeError):
            return WebResponse.json(400, json.dumps({"error": "Choose a valid execution preference and revision."}))

    def _handle_stop(self, parts) -> WebResponse:
        accepted = False
        if len(parts) == 4 and parts[:2] == ["runs", "progress"]:
            snapshot = self._progress_store.get(parts[2])
            if snapshot is None:
                return WebResponse.json(404, '{"error":"Run progress not found."}')
            accepted = self._background_runs is not None and self._background_runs.cancel(parts[2])
        elif len(parts) == 4 and parts[:2] == ["test-cases", "authoring-progress"]:
            snapshot = self._progress_store.get_authoring(parts[2])
            if snapshot is None:
                return WebResponse.json(404, '{"error":"Authoring progress not found."}')
            accepted = self._background_authoring is not None and self._background_authoring.cancel(parts[2])
        elif len(parts) == 3 and parts[0] == "suite-runs":
            if self._suite_run_service is None or self._suite_run_service.get(parts[1]) is None:
                return WebResponse.json(404, '{"error":"Suite Run not found."}')
            accepted = self._suite_run_service.cancel(parts[1])
        else:
            return WebResponse.json(404, '{"error":"Operation not found."}')
        return WebResponse.json(202 if accepted else 409, json.dumps({
            "cancellation_requested": bool(accepted),
            "message": "Cancellation requested. Waiting for safe cleanup." if accepted else "The operation has already finished or cannot be stopped.",
        }))

    def _stop_control(self, url, finished, state) -> str:
        if finished:
            return ""
        stopping = state == "CANCELLATION_REQUESTED"
        return (
            f'<form method="post" action="{escape_html(url)}" data-stop-form class="actions">'
            f'<input type="hidden" name="_csrf" value="{escape_html(self._csrf_token)}">'
            + '<button class="button" type="submit" data-stop-button'
            + (' disabled' if stopping else '') + '>' + ('Stopping…' if stopping else 'Stop')
            + '</button><span data-stop-notice role="status">'
            + ('Cancellation requested. Waiting for safe cleanup.' if stopping else '') + '</span></form>'
        )

    def _preferences_panel(self, case_id) -> str:
        preferences = self._execution_preferences.get(case_id)
        controls = self._evidence_controls(f"preferences-{case_id}") + self._cookie_consent_controls(f"preferences-{case_id}")
        # Existing controls share enum values; only selected attributes change.
        controls = controls.replace(' selected', '')
        for value in (preferences.evidence_mode.value, preferences.screenshot_mode.value, preferences.cookie_policy.value):
            controls = controls.replace(f'value="{value}"', f'value="{value}" selected')
        return (
            f'<section class="panel" data-execution-preferences="/test-cases/{case_id}/preferences" data-revision="{preferences.revision}">'
            '<h2>Execution preferences</h2><p class="muted">Changes are saved immediately for future operations. Active Runs keep their recorded policy.</p>'
            + controls + '<p role="status" data-preferences-status>Saved</p>'
            '<button class="button" type="button" data-preferences-retry hidden>Retry</button>'
            f'<form method="post" action="/test-cases/{case_id}/readiness" data-run-form class="actions">'
            '<div class="field"><label for="readiness-workflow">Check workflow</label>'
            '<select id="readiness-workflow" name="workflow"><option value="AUTOMATION">Automation</option>'
            '<option value="VALIDATION">Validation</option><option value="REGRESSION">Regression</option></select></div>'
            '<button class="button" type="submit">Check readiness</button></form></section>'
        )

    def handle(self, method: str, target: str, body: bytes | str | None = None, *, headers=None) -> WebResponse:
        parsed = urlsplit(target)
        path = parsed.path
        query = parse_qs(parsed.query)
        parts = path.strip("/").split("/")
        if method.upper() == "POST":
            protected = parts[-1:] in (["stop"], ["preferences"]) or (parts[:2] == ["settings", "reliability"] and parts[-1:] == ["cancel"])
            definition_edit = (len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "edit") or (len(parts) == 4 and parts[:2] == ["test-cases", "review"] and parts[3] in {"save", "cancel", "regenerate"})
            if (protected or definition_edit) and not self._valid_control_request(body, headers, max_bytes=_MAX_MANUAL_FORM_BODY_BYTES if definition_edit else _MAX_FORM_BODY_BYTES):
                return WebResponse.json(403, json.dumps({"error": "Request validation failed. Refresh this page and retry."}))
            if parts[-1:] == ["stop"]:
                return self._handle_stop(parts)
            if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "preferences":
                return self._handle_preferences(parts[1], body)
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
        if path == "/system/health":
            report = (
                self._readiness_service.system_report()
                if self._readiness_service is not None
                else ReadinessReport((ReadinessCheck(
                    "application", "Application", ReadinessStatus.NOT_APPLICABLE,
                    "Detailed readiness checks are unavailable for this application instance.",
                ),))
            )
            return self._system_health_page(report)
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
        if len(parts) == 2 and parts[0] == "drafts":
            draft_id = _parse_uuid(parts[1])
            if draft_id is None:
                return self._not_found("Draft not found")
            return self._draft_edit_page_response(
                draft_id, saved=query.get("saved") == ["1"]
            )
        if len(parts) == 3 and parts[:2] == ["test-cases", "authoring-progress"]:
            return self._authoring_progress_page(parts[2])
        if len(parts) == 3 and parts[:2] == ["api", "suite-runs"]:
            return self._suite_run_progress_json(parts[2])
        if len(parts) == 3 and parts[0] == "suite-runs" and parts[2] == "report.json":
            run = self._suite_run_service.get(parts[1]) if self._suite_run_service else None
            if run is None:
                return self._not_found("Suite Run report not found.")
            return WebResponse.json(200, suite_run_report_json(run))
        if len(parts) == 2 and parts[0] == "suite-runs":
            return self._suite_run_page(parts[1])
        if path == "/":
            draft_id = _parse_uuid(query.get("draft_id", [""])[0])
            selected = self._drafts.get(draft_id) if draft_id else None
            if selected is not None and selected.status != DraftStatus.ACTIVE:
                selected = None
            return WebResponse.html(200, self._dashboard(
                name=selected.title if selected else "",
                source_draft_id=str(selected.id) if selected else "",
                base_url=selected.base_url or "" if selected else "",
                scenario=selected.body if selected else "",
            ))
        if path == "/settings/providers" and self._provider_settings is not None:
            return WebResponse.html(200, self._provider_settings_page(query))
        if path == "/settings/usage" and self._llm_usage is not None:
            return WebResponse.html(200, self._usage_analytics_page(query))
        if path == "/settings/reliability" and self._reliability is not None:
            return WebResponse.html(200, self._reliability_page(query))
        if len(parts) == 3 and parts[:2] == ["settings", "reliability"] and self._reliability is not None:
            operation_id = _parse_uuid(parts[2])
            return self._reliability_operation_page(operation_id) if operation_id else self._not_found("Generation operation not found")
        if path == "/demo-target/registration":
            return WebResponse.html(200, _local_demo_page())
        if path == "/assets/demo-registration.js":
            return WebResponse(200, "application/javascript; charset=utf-8", _local_demo_javascript().encode("utf-8"))
        if path == "/demo-target/registration/help":
            return WebResponse.html(200, _local_demo_help_page())
        if path == "/test-cases":
            return WebResponse.html(200, self._test_case_list())
        if path == "/export":
            return WebResponse.html(200, self._export_workspace(query))
        if path == "/test-suites":
            return WebResponse.html(200, self._test_suites_page())
        if path == "/test-cases/new":
            draft_id = _parse_uuid(query.get("draft_id", [""])[0])
            selected = self._drafts.get(draft_id) if draft_id else None
            if selected is not None and selected.status != DraftStatus.ACTIVE:
                selected = None
            return WebResponse.html(200, self._new_test_case_page(
                name=selected.title if selected else "",
                source_draft_id=str(selected.id) if selected else "",
                base_url=selected.base_url or "" if selected else "",
                scenario=selected.body if selected else "",
            ))
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
            report = self._run_report(detail)
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
            report = self._run_report(detail)
            return WebResponse.html(
                200,
                self._reports.to_html(
                    report,
                    report_view=False,
                    evidence_url=lambda step, attempt, evidence, index: self._evidence_url_if_available(
                        detail, run_id, step, attempt, evidence, index
                    ),
                ),
            )
        if len(parts) == 4 and parts[0] == "runs" and parts[2] == "plans":
            run_id, version_id = _parse_uuid(parts[1]), _parse_uuid(parts[3])
            return self._run_plan_view(run_id, version_id) if run_id and version_id else self._not_found("TestPlan version not found")
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
        if len(parts) == 4 and parts[0] == "test-cases" and parts[2:] == ["automation", "edit"]:
            test_case_id = _parse_uuid(parts[1])
            return self._automation_editor_page(test_case_id, query) if test_case_id else self._not_found("TestCase not found")
        if (
            len(parts) == 7 and parts[0] == "test-cases"
            and parts[2:4] == ["automation", "steps"]
            and parts[5] == "versions"
        ):
            test_case_id = _parse_uuid(parts[1])
            step_id = _parse_uuid(parts[4])
            version_id = _parse_uuid(parts[6])
            if test_case_id is None or step_id is None or version_id is None:
                return self._not_found("Automation version not found")
            return self._automation_version_view(test_case_id, step_id, version_id)
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
        if len(parts) == 3 and parts[0] == "test-suites" and parts[2] == "run":
            suite_id = _parse_uuid(parts[1])
            if suite_id is None:
                return self._not_found("Test Suite not found.")
            return self._suite_run_config_page(suite_id)
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
        if parts == ["settings", "reliability"] and self._reliability is not None:
            return self._handle_reliability_settings(body)
        if len(parts) == 4 and parts[:2] == ["settings", "reliability"] and parts[3] == "cancel" and self._reliability is not None:
            operation_id = _parse_uuid(parts[2])
            if operation_id is None or not self._reliability.cancel(operation_id):
                return WebResponse.html(409, self._page("Automation Reliability", '<p>No active generation operation can be cancelled.</p>'))
            return WebResponse.redirect(f"/settings/reliability/{operation_id}")
        if parts == ["system", "health", "refresh"]:
            return self._handle_system_health_refresh()
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "readiness":
            return self._handle_test_case_readiness(parts[1], body)
        if len(parts) == 3 and parts[0] == "test-suites" and parts[2] == "readiness":
            return self._handle_suite_readiness(parts[1], body)
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
            scenario = values.get("scenario", values.get("description", ""))
            return WebResponse.html(200, self._manual_test_case_page(
                name=values.get("name", "").strip() or _derive_test_case_name(scenario),
                base_url=values.get("base_url", ""),
                scenario=scenario,
                source_draft_id=values.get("source_draft_id", ""),
            ))
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] == "edit":
            return self._handle_test_case_edit(parts[1], body)
        if (
            len(parts) == 6 and parts[0] == "test-cases"
            and parts[2:4] == ["automation", "steps"] and parts[5] == "save"
        ):
            return self._handle_automation_step_save(parts[1], parts[4], body)
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
        if parts == ["export"]:
            return self._handle_export_workspace(body)
        if parts == ["test-suites"]:
            return self._handle_test_suite_create(body)
        if len(parts) == 3 and parts[0] == "test-suites" and parts[2] == "update":
            return self._handle_test_suite_update(parts[1], body)
        if len(parts) == 4 and parts[0] == "test-suites" and parts[2] == "members":
            return self._handle_test_suite_members(parts[1], parts[3], body)
        if len(parts) == 3 and parts[0] == "test-suites" and parts[2] == "run":
            return self._handle_suite_run_start(parts[1], body)
        if parts == ["test-cases", "generate"]:
            return self._handle_generate_test_case(body)
        if len(parts) == 3 and parts[0] == "test-cases" and parts[2] in {
            "approve", "approve-validation"
        }:
            return self._handle_test_case_approval(parts[1], parts[2], body)
        if parts == ["test-cases", "drafts", "save"]:
            return self._handle_new_test_case_draft_save(body)
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
        preferences = self._execution_preferences.get(test_case_id)
        try:
            evidence_policy = EvidencePolicy(
                mode=EvidenceMode(form.get("evidence_mode", [preferences.evidence_mode.value])[0]),
                screenshot_mode=ScreenshotMode(form.get("screenshot_mode", [preferences.screenshot_mode.value])[0]),
            )
        except ValueError:
            return self._run_error(test_case_id, "Choose a valid evidence mode and screenshot scope.", 400)
        try:
            cookie_policy = CookieConsentPolicy(
                form.get("cookie_policy", [preferences.cookie_policy.value])[0]
            )
        except ValueError:
            return self._run_error(test_case_id, "Choose a valid cookie consent policy.", 400)
        if self._background_runs is None:
            return self._run_error(
                test_case_id,
                "Execution is not available for this application.",
                503,
            )
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        review_status = self._test_case_review.status(test_case_id)
        if review_status != TestCaseReviewStatus.APPROVED:
            return self._run_error(
                test_case_id,
                "Review and approve this TestCase before generating automation or starting a run.",
                409,
            )
        review_record = self._test_case_review.record(test_case_id)
        if (
            workflow in {WorkflowType.VALIDATION, WorkflowType.REGRESSION}
            and review_record is not None
            and test_case is not None
            and not self._test_case_review.validation_approved_for(test_case)
        ):
            return self._run_error(
                test_case_id,
                "Review the current saved automation and approve it for Validation first.",
                409,
            )
        if self._readiness_service is not None:
            report = self._readiness_service.testcase_report(
                test_case,
                workflow,
                evidence_policy=evidence_policy,
                cookie_policy=cookie_policy,
            )
            if not report.can_continue:
                return self._readiness_result_page(
                    report,
                    title=f"Readiness blocked — {test_case.name}",
                    return_url=f"/test-cases/{test_case.id}",
                )
        progress_id = self._background_runs.start(
            test_case_id,
            workflow,
            evidence_policy=evidence_policy,
            cookie_policy=cookie_policy,
        )
        return WebResponse.redirect(f"/runs/progress/{progress_id}")

    def _handle_test_case_approval(
        self, test_case_id_text: str, action: str, body: bytes | str | None
    ) -> WebResponse:
        _form, error = _parse_form_body(body)
        if error:
            return self._not_found("TestCase approval request is invalid.")
        test_case_id = _parse_uuid(test_case_id_text)
        test_case = (
            self._test_cases.get(test_case_id)
            if test_case_id is not None and self._test_cases is not None else None
        )
        if test_case is None:
            return self._not_found("TestCase not found")
        if action == "approve":
            record = self._test_case_review.record(test_case.id)
            if record is not None and record.status == TestCaseReviewStatus.READY_FOR_REVIEW:
                try:
                    self._test_case_review.approve_test_case(test_case)
                except ValueError as quality_error:
                    return WebResponse.html(409, self._page("TestCase needs review", '<div class="error-state"><h1>TestCase needs review</h1>' + self._quality_feedback(test_case) + '</div>' + f'<a class="button" href="/test-cases/{test_case.id}/edit">Edit TestCase</a>'))
            return WebResponse.redirect(f"/test-cases/{test_case.id}")
        try:
            if self._automation_lifecycle.status(test_case) in {AutomationStatus.NEEDS_UPDATE, AutomationStatus.AUTOMATION_FAILED}:
                raise ValueError("Generate updated automation for this definition before approving it for Validation.")
            self._test_case_review.approve_for_validation(test_case)
        except ValueError as approval_error:
            return WebResponse.html(409, self._page(
                "Automation review required",
                '<div class="error-state"><h1>Automation not approved</h1>'
                f'<p>{escape_html(approval_error)}</p>'
                f'<a class="button" href="/test-cases/{test_case.id}">Return to TestCase</a></div>',
            ))
        return WebResponse.redirect(f"/test-cases/{test_case.id}")

    def _handle_new_test_case_draft_save(
        self, body: bytes | str | None
    ) -> WebResponse:
        form, error = _parse_form_body(body, max_bytes=_MAX_DRAFT_SAVE_BODY_BYTES)
        values = {key: items[0] for key, items in form.items()}
        dashboard_entry = values.get("authoring_entry") == "dashboard"

        def error_response(message: str, status: int = 400) -> WebResponse:
            details = {key: values.get(key, "") for key in ("name", "base_url", "scenario", "source_draft_id")}
            page = (
                self._dashboard(authoring_error=message, **details) if dashboard_entry
                else self._new_test_case_page(message, **details)
            )
            return WebResponse.html(status, page)

        if error:
            return error_response(error)
        scenario = values.get("scenario", "").strip()
        base_url = values.get("base_url", "").strip()
        source_text = values.get("source_draft_id", "").strip()
        source = None
        if source_text:
            source_id = _parse_uuid(source_text)
            source = self._drafts.get(source_id) if source_id is not None else None
            if source is None or source.status != DraftStatus.ACTIVE:
                return error_response("The selected Draft is no longer active. Choose another Draft.", 409)
        name = values.get("name", "").strip()
        if len(name) > 200 or len(scenario) > 6000 or len(base_url) > 2048:
            return error_response("Keep Summary under 200 characters, Scenario under 6,000, and Website under 2,048.")
        title = name or (source.title if source is not None else next(
            (line.strip() for line in scenario.splitlines() if line.strip()), ""
        )[:200]) or "Untitled Draft"
        draft_values = {
            "id": source.id if source is not None else uuid4(),
            "title": title,
            "body": scenario,
            "base_url": base_url or None,
            "notes": source.notes if source is not None else None,
            "status": source.status if source is not None else DraftStatus.ACTIVE,
            "converted_test_case_id": (source.converted_test_case_id if source is not None else None),
        }
        if source is not None:
            draft_values["created_at"] = source.created_at
        draft = Draft(**draft_values)
        self._drafts.save(draft)
        return WebResponse.redirect(f"/drafts/{draft.id}?saved=1")

    def _saved_version(self, step_id: UUID, version_id: UUID | None):
        if self._plan_store is None or version_id is None:
            return None
        version = self._plan_store.get_version(version_id)
        plan = self._plan_store.find_test_plan(step_id)
        return version if version is not None and plan is not None and version.test_plan_id == plan.id else None

    def _run_report(self, detail: RunHistoryDetail):
        report = self._reports.generate_history(detail)
        steps = []
        for step in report.steps:
            attempts = []
            for attempt in step.attempts:
                version = self._saved_version(step.step_id, attempt.test_plan_version_id)
                attempts.append(attempt.model_copy(update={
                    "plan_url": f"/runs/{report.run_id}/plans/{version.id}" if version else None,
                    "plan_version_number": version.version if version else attempt.plan_version_number,
                    "plan_version_origin": version.origin if version else attempt.plan_version_origin,
                }))
            steps.append(step.model_copy(update={"attempts": attempts}))
        return report.model_copy(update={"steps": steps})

    def _run_plan_view(self, run_id: UUID, version_id: UUID) -> WebResponse:
        detail = self._run_history.get_detail(run_id)
        reference = next((item for item in detail.record.executions if item.test_plan_version_id == version_id), None) if detail else None
        version = self._saved_version(reference.test_step_id, version_id) if reference else None
        if version is None:
            return self._not_found("Saved TestPlan version not found for this Run")
        actions = ''.join(
            '<li><strong>' + escape_html(ACTION_LABELS.get(action.action, action.action)) + '</strong><dl>'
            + ''.join(
                f'<dt>{escape_html(FIELD_LABELS.get(key, key))}</dt>'
                f'<dd><code>{escape_html("[REDACTED]" if action.action == "fill" and key == "value" else _safe_automation_url_display(str(value)) if key == "url" else redact_secrets(str(value)))}</code></dd>'
                for key, value in sorted(action.parameters.items())
            ) + '</dl></li>' for action in version.qa_test_plan.steps
        )
        content = (
            f'<header class="page-heading"><h1>Saved TestPlan version v{version.version}</h1>'
            f'<p>{escape_html(detail.record.public_id or "Run")} · {_plan_origin_label(version.origin)}</p></header>'
            '<p class="notice">Read-only version pinned to this execution. This page cannot edit the saved plan.</p>'
            f'<section class="panel"><ol>{actions}</ol></section>'
            f'<a class="button" href="/runs/{run_id}">Back to Run Details</a>'
        )
        return WebResponse.html(200, self._page("Saved TestPlan", content))

    def _progress_payload(self, snapshot) -> dict:
        payload = snapshot.to_public_dict()
        payload["stop_url"] = f"/runs/progress/{snapshot.progress_id}/stop" if not payload["finished"] and self._background_runs is not None else None
        persisted = self._run_history.get(snapshot.final_run_id) if snapshot.final_run_id else None
        payload["final_run_id"] = str(persisted.run_id) if persisted else None
        payload["final_run_url"] = f"/runs/{persisted.run_id}" if persisted else None
        detail = self._run_history.get_detail(persisted.run_id) if persisted else None
        report = self._run_report(detail) if detail else None
        report_steps = {step.step_id: step for step in report.steps} if report else {}
        if report:
            payload["summary"] = {**report.result_summary, "history_note": None}
            payload["display_status"] = report.display_status
            payload["phase"] = terminal_phase(report.display_outcome)
        elif payload["finished"]:
            payload["summary"]["history_note"] = "No persisted Run was created. Partial progress diagnostics are available for this request."
        for step in payload["steps"]:
            saved_step = report_steps.get(UUID(step["id"]))
            step["evidence_links"] = []
            step["observation"] = None
            step.setdefault("action_failure", None)
            if saved_step:
                step["display_status"] = saved_step.display_status
                if saved_step.attempts:
                    last = saved_step.attempts[-1]
                    step["plan_version_id"] = str(last.test_plan_version_id)
                    step["plan_version"] = last.plan_version_number
                    step["plan_origin"] = last.plan_version_origin.value if last.plan_version_origin else None
                    step["observation"] = last.observation
                    step["action_failure"] = last.action_failure.model_dump(mode="json") if last.action_failure else None
                for attempt_number, attempt in enumerate(saved_step.attempts, start=1):
                    for index, evidence in enumerate(attempt.evidence):
                        url = self._evidence_url_if_available(detail, persisted.run_id, saved_step, attempt, evidence, index)
                        if url:
                            step["evidence_links"].append({
                                "url": url, "label": f"Attempt {attempt_number}: {evidence.name}",
                                "execution_id": str(attempt.execution_id), "event": evidence.event,
                            })
            version_id = _parse_uuid(step["plan_version_id"] or "")
            version = self._saved_version(UUID(step["id"]), version_id)
            if version and persisted and any(
                item.test_step_id == UUID(step["id"]) and item.test_plan_version_id == version.id
                for item in persisted.executions
            ):
                step["plan_url"] = f"/runs/{persisted.run_id}/plans/{version.id}"
            elif version and self._test_cases is not None:
                case = self._test_cases.get(snapshot.test_case_id)
                step["plan_url"] = (
                    f"/test-cases/{snapshot.test_case_id}/automation/steps/{step['id']}/versions/{version.id}"
                    if case and any(item.id == UUID(step["id"]) for item in case.steps) else None
                )
            else:
                step["plan_url"] = None
        payload["actions"] = [{"label": "Open TestCase", "url": f"/test-cases/{snapshot.test_case_id}"}]
        if self._reliability is not None:
            operation_ids = dict.fromkeys(event.reliability_operation_id for event in snapshot.events if event.reliability_operation_id)
            for operation_id in operation_ids:
                payload["actions"].append({"label": "Generation decisions", "url": f"/settings/reliability/{operation_id}"})
        if persisted:
            payload["actions"].insert(0, {"label": "View Run Details", "url": f"/runs/{persisted.run_id}"})
        if payload["summary"]["outcome"] == "INFRASTRUCTURE_ERROR":
            payload["actions"].append({"label": "Open System Health", "url": "/system/health"})
        if payload["summary"]["outcome"] == "AUTOMATION_GENERATION_ERROR":
            payload["actions"].append({"label": "Edit TestCase", "url": f"/test-cases/{snapshot.test_case_id}/edit"})
        if payload["summary"]["outcome"] in {"AUTOMATION_EXECUTION_ERROR", "AUTOMATION_DRIFT"}:
            payload["actions"].append({"label": "Edit Automation", "url": f"/test-cases/{snapshot.test_case_id}/automation/edit"})
        payload["redirect_after_ms"] = None
        return payload

    def _progress_json(self, progress_id: str) -> WebResponse:
        snapshot = self._progress_store.get(progress_id)
        if snapshot is None:
            return WebResponse.json(404, json.dumps({
                "error": "Execution progress is no longer available."
            }))
        payload = self._progress_payload(snapshot)
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
            json.dumps({**snapshot.to_public_dict(), "stop_url": f"/test-cases/authoring-progress/{progress_id}/stop" if not snapshot.finished else None}, ensure_ascii=False, separators=(",", ":")),
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
            + self._stop_control(f"/test-cases/authoring-progress/{progress_id}/stop", snapshot.finished, snapshot.state.value)
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
        payload = self._progress_payload(snapshot)
        summary = payload["summary"]
        esc = escape_html
        step_rows = []
        for number, step in enumerate(payload["steps"], start=1):
            version_label = (
                f'{_plan_origin_display(step["plan_origin"])} v{step["plan_version"]}'
                if step["plan_version"] is not None else 'Saved TestPlan'
            )
            plan = (
                f'<a href="{esc(step["plan_url"])}">{esc(version_label)}</a>' if step["plan_url"]
                else '<span class="muted">No saved TestPlan available.</span>'
            )
            evidence = ''.join(
                f'<a href="{esc(item["url"])}" target="_blank" rel="noopener">Open full-size evidence · {esc(item["label"])}</a><br>'
                for item in step["evidence_links"]
            )
            step_rows.append(
                '<li class="progress-step"><span class="progress-symbol" aria-hidden="true">'
                + ('✓' if step["display_status"] == 'Passed' else '○') + '</span><span>'
                + f'<strong>Step {number}: {esc(step["name"])}</strong>'
                + f'<span class="progress-step-state">{esc(step["display_status"])}</span>'
                + (f'<span class="muted">Automation {esc(step["automation_state"].lower())}</span>' if step["automation_state"] and step["automation_state"] != 'Failed' else '')
                + f'<span class="muted">{plan}</span>'
                + (f'<span>Observed: {esc(step["observation"])}</span>' if step["observation"] else '')
                + (f'<details><summary>Action failure details</summary><pre>{esc(json.dumps(step["action_failure"], indent=2))}</pre></details>' if step.get("action_failure") else '')
                + (f'<span class="progress-evidence">Screenshot evidence captured ({step["evidence_count"]})</span>' if step["evidence_count"] else '')
                + evidence + '</span></li>'
            )
        result = self._progress_summary_html(payload) if payload["finished"] else ''
        retry = ''
        if payload["finished"] and summary["outcome"] in {
            'AUTOMATION_EXECUTION_ERROR', 'AUTOMATION_DRIFT', 'AUTOMATION_GENERATION_ERROR', 'INFRASTRUCTURE_ERROR',
        }:
            retry = (
                f'<form method="post" action="/test-cases/{snapshot.test_case_id}/run" data-run-form data-progress-retry>'
                f'<input type="hidden" name="workflow" value="{esc(snapshot.workflow_type)}">'
                f'<button class="button" type="submit">{"Retry Automation" if summary["outcome"] == "AUTOMATION_GENERATION_ERROR" else "Retry Run"}</button></form>'
            )
        content = (
            '<header class="page-heading"><p class="eyebrow">Run Progress</p>'
            f'<h1>{esc(snapshot.test_case_name or "TestCase run")}</h1>'
            f'<p class="lead">Current operation: {badge(snapshot.workflow_type, "workflow")}</p></header>'
            '<div class="summary-grid">'
            + _summary_card('Current stage', f'<span data-progress-phase>{esc(payload["phase"])}</span>', raw=True)
            + _summary_card('Elapsed', f'<span data-progress-elapsed>{esc(format_duration(snapshot.elapsed_ms))}</span>', raw=True)
            + '</div>'
            + self._stop_control(f"/runs/progress/{progress_id}/stop", payload["finished"], snapshot.state.value)
            + '<section class="panel progress-result" data-progress-result' + ('' if payload['finished'] else ' hidden')
            + '><h2>Result summary</h2><div data-progress-result-content>' + result + '</div>' + retry + '</section>'
            + f'<div class="progress-live" data-progress-id="{esc(progress_id)}">'
            + '<section class="panel"><h2>Completed and pending steps</h2><ol class="progress-steps" id="progress-steps">'
            + ''.join(step_rows) + '</ol></section>'
            + '<details class="panel technical-details" data-progress-developer-details><summary>Developer details</summary>'
            + '<div data-progress-diagnostics>' + self._progress_diagnostics_html(payload['diagnostic_stages'], payload['provider_diagnostics']) + '</div></details></div>'
            + '<p class="muted" data-progress-notice aria-live="polite"></p>'
        )
        return WebResponse.html(200, self._page(
            'Run Progress', content, breadcrumbs=[('Dashboard', '/'), ('Test Cases', '/test-cases')],
        ))

    @staticmethod
    def _progress_summary_html(payload: dict) -> str:
        summary = payload['summary']
        text = result_badge(summary['outcome'])
        text += f'<p>{summary["completed_steps"]} of {summary["total_steps"]} steps verified.</p>'
        if summary['stopping_detail']:
            text += f'<p>{escape_html(summary["stopping_detail"])}</p>'
        text += f'<p>{escape_html(summary["explanation"])}</p>'
        position = summary['stopping_step']
        if position and payload['steps'][position - 1].get('observation'):
            text += f'<p><strong>Observed:</strong> {escape_html(payload["steps"][position - 1]["observation"])}</p>'
        text += f'<p><strong>Recommended action:</strong> {escape_html(summary["recommended_action"])}</p>'
        if summary.get('history_note'):
            text += f'<p>{escape_html(summary["history_note"])}</p>'
        text += '<div class="actions">' + ''.join(
            f'<a class="button" href="{escape_html(item["url"])}">{escape_html(item["label"])}</a>' for item in payload['actions']
        ) + '</div>'
        return text

    @staticmethod
    def _progress_diagnostics_html(stages: list[dict], provider_attempts: list[dict] | None = None) -> str:
        sections = []
        for stage in stages:
            rows = []
            for event in stage['events']:
                fields = [event['display_status']]
                if event['step_order'] is not None:
                    fields.append(f'Step {event["step_order"] + 1}')
                if event['attempt_number'] is not None:
                    fields.append(f'Attempt {event["attempt_number"]}')
                if event['duration_ms'] is not None:
                    fields.append(format_duration(event['duration_ms']))
                rows.append(
                    '<li class="progress-event"><div>'
                    + f'<time>{escape_html(event["timestamp"])}</time> · {escape_html(" · ".join(fields))}'
                    + f'<p>{escape_html(event["reason"])}</p>'
                    + '<details><summary>Raw details</summary>'
                    + f'<pre>{escape_html(json.dumps(event, ensure_ascii=False, indent=2))}</pre></details></div></li>'
                )
            sections.append(f'<section><h3>{escape_html(stage["stage"])}</h3><ol>{"".join(rows)}</ol></section>')
        if provider_attempts:
            rows = []
            for attempt in provider_attempts:
                fields = [
                    f'Step {attempt["step_number"]}', f'Attempt {attempt["attempt_number"]}',
                    attempt['request_kind'], attempt['provider'], attempt['status'],
                ]
                if attempt.get('model'):
                    fields.append(attempt['model'])
                if attempt.get('duration_ms') is not None:
                    fields.append(format_duration(attempt['duration_ms']))
                rows.append(
                    '<li class="progress-event"><div>' + escape_html(' · '.join(fields))
                    + (f'<p>{escape_html(attempt["reason"])}</p>' if attempt.get('reason') else '')
                    + '<details><summary>Raw details</summary>'
                    + f'<pre>{escape_html(json.dumps(attempt, ensure_ascii=False, indent=2))}</pre></details></div></li>'
                )
            sections.append('<section><h3>Recorded provider attempts</h3><ol>' + ''.join(rows) + '</ol></section>')
        return ''.join(sections)

    def _drafts_page(self) -> str:
        drafts = self._drafts.list(500)
        rows = "".join(
            '<tr><td><a href="/drafts/' + str(item.id) + '">' + escape_html(item.title) + '</a></td>'
            + f'<td>{escape_html(item.base_url or "—")}</td>'
            + f'<td>{"Used" if item.status == DraftStatus.USED else "Active"}</td>'
            + (f'<td><a href="/test-cases/{item.converted_test_case_id}">View TestCase</a></td>'
               if item.converted_test_case_id else '<td>вЂ”</td>')
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
               '<th>Title</th><th>Base URL</th><th>Status</th><th>TestCase</th><th>Updated</th><th>Actions</th></tr></thead>'
               f'<tbody>{rows}</tbody></table></div></section>' if drafts else
               self._empty_state("No Drafts yet.", "Save a testing idea to work on it later."))
        )
        return self._page("Drafts", content, current="Drafts", breadcrumbs=[("Dashboard", "/")])

    def _draft_edit_page_response(self, draft_id: UUID, *, saved: bool = False) -> WebResponse:
        draft = self._drafts.get(draft_id)
        if draft is None:
            return self._not_found("Draft not found")
        return WebResponse.html(200, self._draft_edit_page(draft, saved=saved))

    def _draft_edit_page(
        self,
        draft: Draft | None = None,
        error: str | None = None,
        submitted: dict[str, str] | None = None,
        *,
        saved: bool = False,
    ) -> str:
        editing = draft is not None
        submitted = submitted or {}
        action = f"/drafts/{draft.id}/update" if draft else "/drafts"
        title = submitted.get("title", draft.title if draft else "")
        body = submitted.get("body", draft.body if draft else "")
        base_url = submitted.get("base_url", (draft.base_url or "") if draft else "")
        notes = submitted.get("notes", (draft.notes or "") if draft else "")
        error_html = f'<p class="authoring-error" role="alert">{escape_html(error)}</p>' if error else ""
        success_html = '<p class="authoring-success" role="status">Draft saved successfully.</p>' if saved else ""
        state_html = (
            '<section class="panel"><h2>Used Draft</h2>'
            + (f'<p>Linked TestCase: <a href="/test-cases/{draft.converted_test_case_id}">View TestCase</a></p>'
               if draft.converted_test_case_id else '<p>No linked TestCase is recorded.</p>')
            + '<p class="muted">This Draft remains saved and can be reused.</p></section>'
            if draft is not None and draft.status == DraftStatus.USED else ""
        )
        content = (
            '<header class="page-heading"><h1>' + ("Edit Draft" if editing else "New Draft") + '</h1>'
            '<p class="lead">Drafts stay separate from TestCases, Runs, and Test Suites.</p></header>'
            + error_html + success_html + state_html
            + f'<form method="post" action="{action}" class="panel" data-inline-validation novalidate>'
            + '<div class="field"><label for="draft-title">Title</label>'
            + f'<input id="draft-title" name="title" maxlength="200" required value="{escape_html(title)}"></div>'
            + '<div class="field"><label for="draft-body">Testing idea / scenario</label>'
            + f'<textarea id="draft-body" name="body" maxlength="6000" rows="8">{escape_html(body)}</textarea></div>'
            + '<div class="field"><label for="draft-url">Base URL (optional)</label>'
            + f'<input id="draft-url" name="base_url" maxlength="2048" value="{escape_html(base_url)}"></div>'
            + '<div class="field"><label for="draft-notes">Notes (optional)</label>'
            + f'<textarea id="draft-notes" name="notes" maxlength="6000" rows="3">{escape_html(notes)}</textarea></div>'
            + '<div class="actions"><button class="button primary" type="submit">Save Draft</button>'
            + '<a class="button" href="/drafts">Back to Drafts</a></div></form>'
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
                self._test_case_review.mark_ready_for_review(test_case)
                self._drafts.mark_used(draft.id, test_case.id)
            except (TestCaseEditError, ValueError) as conversion_error:
                return WebResponse.html(400, self._draft_edit_page(draft, str(conversion_error)))
            return WebResponse.redirect(f"/test-cases/{test_case.id}")
        if action == "generate":
            if self._background_authoring is None or self._authoring_service is None:
                return WebResponse.html(503, self._draft_edit_page(draft, "AI generation is unavailable. You can still create a TestCase manually."))
            try:
                validated = self._authoring_service.validate_input(draft.title, draft.body, draft.base_url or "")
                progress_id = self._background_authoring.start(
                    name=validated.name, base_url=validated.base_url, scenario=validated.scenario,
                    source_draft_id=draft.id,
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
            if len(scenario) > 6000:
                raise ValueError("Keep the testing idea under 6,000 characters.")
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
                "status": existing.status if existing else DraftStatus.ACTIVE,
                "converted_test_case_id": existing.converted_test_case_id if existing else None,
            }
            if existing is not None:
                draft_values["created_at"] = existing.created_at
            draft = Draft(**draft_values)
            self._drafts.save(draft)
        except (ValueError, TypeError) as save_error:
            return WebResponse.html(400, self._draft_edit_page(existing, str(save_error), values))
        return WebResponse.redirect(f"/drafts/{draft.id}?saved=1")

    def _manual_test_case_page(
        self,
        *,
        name: str = "",
        base_url: str = "",
        scenario: str = "",
        source_draft_id: str = "",
        error: str | None = None,
        submitted: dict[str, str] | None = None,
    ) -> str:
        submitted = submitted or {}
        error_html = f'<p class="authoring-error" role="alert">{escape_html(error)}</p>' if error else ""
        rows = []
        for index in range(8):
            rows.append(
                f'<fieldset class="subpanel"><legend>Step {index + 1}</legend>'
                '<div class="field">'
                f'<input type="hidden" id="step-name-{index}" name="step_name_{index}" maxlength="200"'
                f' value="{escape_html(submitted.get(f"step_name_{index}", ""))}"></div>'
                f'<div class="field"><label for="step-action-{index}">Action / instruction</label>'
                f'<textarea id="step-action-{index}" name="step_action_{index}" maxlength="6000" rows="3">'
                f'{escape_html(submitted.get(f"step_action_{index}", scenario if index == 0 else ""))}</textarea></div>'
                f'<div class="field"><label for="step-expected-{index}">Expected result</label>'
                f'<textarea id="step-expected-{index}" name="step_expected_{index}" maxlength="6000" rows="2"'
                f'>{escape_html(submitted.get(f"step_expected_{index}", ""))}</textarea></div>'
                '</fieldset>'
            )
        content = (
            '<header class="page-heading"><h1>Create TestCase manually</h1>'
            '<p class="lead">This path does not call an AI provider. You can add more steps after saving.</p></header>'
            + error_html + '<form method="post" action="/test-cases/manual" class="panel" data-inline-validation novalidate>'
            + (f'<input type="hidden" name="source_draft_id" value="{escape_html(source_draft_id)}">' if source_draft_id else "")
            + '<div class="field"><label for="manual-name">Summary (optional)</label>'
            + f'<input id="manual-name" name="name" maxlength="200" value="{escape_html(name)}"></div>'
            + '<div class="field"><label for="manual-description">Scenario / description</label>'
            + f'<textarea id="manual-description" name="description" maxlength="6000" rows="4" required>{escape_html(scenario)}</textarea></div>'
            + '<div class="field"><label for="manual-base-url">Base URL (optional)</label>'
            + f'<input id="manual-base-url" name="base_url" maxlength="2048" value="{escape_html(base_url)}"></div>'
            + '<div class="field"><label for="manual-preconditions">Preconditions (one per line)</label>'
            + f'<textarea id="manual-preconditions" name="preconditions" maxlength="180000" rows="3">{escape_html(submitted.get("preconditions", ""))}</textarea></div>'
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
        source_text = values.get("source_draft_id", "").strip()
        source_id = _parse_uuid(source_text) if source_text else None
        source_draft = self._drafts.get(source_id) if source_id is not None else None
        if source_text and (source_draft is None or source_draft.status != DraftStatus.ACTIVE):
            return WebResponse.html(409, self._manual_test_case_page(
                name=values.get("name", ""), base_url=values.get("base_url", ""),
                scenario=values.get("description", ""),
                error="The selected Draft is no longer active. Choose another Draft.",
            ))
        try:
            test_case = create_manual_test_case(values)
            if self._test_cases is None:
                raise TestCaseEditError("TestCase storage is not configured.")
            self._test_cases.save(test_case)
        except (TestCaseEditError, ValueError) as create_error:
            return WebResponse.html(400, self._manual_test_case_page(
                name=values.get("name", ""), base_url=values.get("base_url", ""),
                scenario=values.get("description", ""),
                source_draft_id=values.get("source_draft_id", ""), error=str(create_error), submitted=values,
            ))
        self._test_case_review.mark_ready_for_review(test_case)
        if source_draft is not None:
            self._drafts.mark_used(source_draft.id, test_case.id)
        return WebResponse.redirect(f"/test-cases/{test_case.id}")

    def _test_case_edit_page_response(self, test_case_id: UUID) -> WebResponse:
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        if test_case is None:
            return self._not_found("TestCase not found")
        return WebResponse.html(200, self._test_case_edit_page(test_case))

    def _definition_editor(self, test_case: TestCase, action: str, cancel_url: str, error: str | None = None, submitted: dict[str, str] | None = None, *, review: bool = False) -> str:
        from qa_agent.automation_lifecycle import definition_fingerprint
        submitted = submitted or {}
        payload = submitted.get("steps_json", json.dumps([
            {"id": str(segment.id), "steps": [
                {"id": str(step.id), "description": step.description.strip(), "expected": step.expected.strip()}
                for step in segment.steps]}
            for segment in test_case.segments], ensure_ascii=False))
        def value(key, default):
            return escape_html(submitted.get(key, default))
        fingerprint = submitted.get("definition_fingerprint", definition_fingerprint(test_case))
        title = "Review TestCase" if review else "Edit TestCase"
        conditions = "\n".join(item.description for item in test_case.preconditions)
        segment_urls = json.dumps([segment.base_url or test_case.base_url or "No URL configured" for segment in test_case.segments])
        return (
            f'<header class="page-heading"><h1>{title}</h1><p class="lead">Edit Description and Expected Result. Step changes stay unsaved until Save TestCase.</p></header>'
            + (f'<p class="authoring-error" role="alert">{escape_html(error)}</p><p><a href="{escape_html(cancel_url + ("/edit" if not review else ""))}">Reload the saved definition</a></p>' if error else '')
            + f'<form method="post" action="{escape_html(action)}" class="testcase-editor" data-testcase-editor data-structured-editor data-max-body="{_MAX_DRAFT_SAVE_BODY_BYTES if review else _MAX_MANUAL_FORM_BODY_BYTES}">'
            + f'<input type="hidden" name="_csrf" value="{self._csrf_token}">'
            + f'<input type="hidden" name="definition_fingerprint" value="{escape_html(fingerprint)}">'
            + f'<input type="hidden" name="steps_json" data-step-payload value="{escape_html(payload)}">'
            + '<section class="panel"><h2>Definition</h2>'
            + f'<div class="field"><label for="edit-name">Summary</label><input id="edit-name" name="name" maxlength="200" value="{value("name", test_case.name)}"></div>'
            + f'<div class="field"><label for="edit-description">Scenario</label><textarea id="edit-description" name="description" maxlength="6000" required>{value("description", test_case.description)}</textarea></div>'
            + (f'<input type="hidden" name="base_url" value="{escape_html(test_case.base_url or "")}"><p>Website: {escape_html(test_case.base_url or "No URL configured")}</p>' if review else
               f'<div class="field"><label for="edit-url">Website</label><input id="edit-url" name="base_url" maxlength="2048" value="{value("base_url", test_case.base_url or "")}"></div>')
            + f'<div class="field"><label for="edit-preconditions">Preconditions (one per line)</label><textarea id="edit-preconditions" name="preconditions" maxlength="180000" rows="3">{value("preconditions", conditions)}</textarea></div></section>'
            + '<section class="panel"><h2>Steps</h2><p class="muted">Empty fields can be saved as unfinished work. Approval requires meaningful content. Moves stay within their segment.</p>'
            + f'<div data-structured-steps data-segment-urls="{escape_html(segment_urls)}"></div><noscript>Enable JavaScript to insert or reorder steps.</noscript></section>'
            + '<div class="actions"><button class="button primary" type="submit">Save TestCase</button>'
            + (f'<button class="button" type="submit" data-regenerate-testcase formnovalidate formaction="{escape_html(action.rsplit("/", 1)[0] + "/regenerate")}">Regenerate TestCase with AI</button>' if review else '')
            + f'<a class="button" data-cancel-testcase href="{escape_html(cancel_url)}">Cancel unsaved changes</a></div></form>'
            + ('<p class="muted">Regenerate TestCase with AI keeps your current Summary and uses the original Scenario.</p>' if review else '')
        )

    def _test_case_edit_page(self, test_case: TestCase, error: str | None = None, submitted: dict[str, str] | None = None) -> str:
        content = self._definition_editor(test_case, f"/test-cases/{test_case.id}/edit", f"/test-cases/{test_case.id}", error, submitted)
        return self._page("Edit TestCase", content, current="Test Cases", breadcrumbs=[("Test Cases", "/test-cases"), (test_case.name, f"/test-cases/{test_case.id}")])

    def _handle_test_case_edit(self, test_case_id_text: str, body: bytes | str | None) -> WebResponse:
        from qa_agent.automation_lifecycle import definition_fingerprint
        from qa_agent.test_case_repository import TestCaseConflict
        test_case_id = _parse_uuid(test_case_id_text)
        test_case = self._test_cases.get(test_case_id) if test_case_id and self._test_cases is not None else None
        if test_case is None:
            return self._not_found("TestCase not found")
        form, error = _parse_form_body(body, max_bytes=_MAX_MANUAL_FORM_BODY_BYTES)
        submitted = {key: items[0] for key, items in form.items()}
        try:
            if error or any(len(items) != 1 for items in form.values()):
                raise TestCaseEditError(error or "Submit each field once.")
            if submitted.get("definition_fingerprint") != definition_fingerprint(test_case):
                raise TestCaseConflict("This TestCase changed elsewhere. Reload it before saving; your submitted changes are shown below.")
            known = {"name", "description", "base_url", "preconditions", "steps_json", "definition_fingerprint", "_csrf"}
            if "steps_json" not in submitted:
                known.add("operation")
                for segment_index, segment in enumerate(test_case.segments):
                    for index, _ in enumerate(segment.steps):
                        known.update(f"step_{segment_index}_{index}_{suffix}" for suffix in ("name", "action", "expected"))
            if set(submitted) - known:
                raise TestCaseEditError("Unexpected TestCase edit fields.")
            edited = edit_test_case(test_case, submitted, submitted.get("operation", "save"))
            self._test_cases.save(edited, expected_fingerprint=submitted["definition_fingerprint"])
            if definition_fingerprint(edited) != definition_fingerprint(test_case):
                self._automation_lifecycle.mark_test_case_changed(edited)
                self._test_case_review.mark_ready_for_review(edited)
        except (TestCaseEditError, ValueError) as edit_error:
            return WebResponse.html(409 if isinstance(edit_error, TestCaseConflict) else 400, self._test_case_edit_page(test_case, str(edit_error), submitted))
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
            coverage_html = _expected_result_coverage_html(
                expected_result_coverage(
                    step, version
                )
            )
            if version is None:
                rows.append(f'<section class="panel"><h2>Step {step.order + 1}: {escape_html(step.name)}</h2>{coverage_html}<p class="muted">No saved TestPlan for this step.</p></section>')
                continue
            actions = "".join(
                '<li><code>' + escape_html(action.action) + '</code><pre>'
                + escape_html(_safe_parameter_json(action.parameters))
                + '</pre></li>' for action in version.qa_test_plan.steps
            )
            rows.append(
                f'<section class="panel"><h2>Step {step.order + 1}: {escape_html(step.name)}</h2>'
                f'<p>Plan v{version.version} · {_plan_origin_label(version.origin)}</p>'
                f'{coverage_html}'
                f'<p>URL: {escape_html(_safe_automation_url_display(version.qa_test_plan.url))}</p><ol>{actions}</ol></section>'
            )
        content = (
            f'<header class="page-heading"><h1>View TestPlan: {escape_html(test_case.name)}</h1>'
            '<p class="lead">Read-only view of current saved plans. Edit supported actions in the Automation Editor.</p></header>'
            + "".join(rows)
        )
        return WebResponse.html(200, self._page("View TestPlan", content, current="Test Cases", breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases"), (test_case.name, f"/test-cases/{test_case.id}")]))

    def _automation_editor_page(
        self,
        test_case_id: UUID,
        query: dict[str, list[str]] | None = None,
        *,
        submitted_step_id: UUID | None = None,
        submitted_values: dict[str, str] | None = None,
        field_errors: dict[str, str] | None = None,
        error_message: str | None = None,
        status: int = 200,
    ) -> WebResponse:
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        if test_case is None:
            return self._not_found("TestCase not found")
        if self._test_case_review.status(test_case.id) != TestCaseReviewStatus.APPROVED:
            return WebResponse.html(409, self._page(
                "TestCase review required",
                '<div class="error-state"><h1>TestCase review required</h1>'
                '<p>Approve the TestCase before editing automation.</p>'
                f'<a class="button" href="/test-cases/{test_case.id}">Review TestCase</a></div>',
            ))
        if self._plan_store is None:
            return WebResponse.html(503, self._page(
                "Automation editor unavailable",
                self._empty_state("Automation editor unavailable", "No saved plan store is configured."),
            ))
        if self._automation_lifecycle.status(test_case) == AutomationStatus.NEEDS_UPDATE:
            return WebResponse.html(409, self._page("Automation needs update", '<p>Generate updated automation before editing plans for this definition.</p>' + f'<a href="/test-cases/{test_case.id}/plans">View saved plans</a>'))

        query = query or {}
        submitted_values = submitted_values or {}
        field_errors = field_errors or {}
        complete = has_complete_plans(test_case, self._plan_store)
        notice = ""
        if query.get("saved") == ["1"]:
            notice = (
                '<div class="success-state" role="status"><strong>Automation saved.</strong> '
                'Review the current TestPlan and approve these saved versions for Validation. '
                + f' <a href="/test-cases/{test_case.id}">Back to TestCase</a></div>'
            )
        if error_message:
            notice = f'<div class="error-state" role="alert">{escape_html(error_message)}</div>'

        sections: list[str] = []
        for step in test_case.steps:
            version = self._plan_store.find(step.id)
            test_plan = self._plan_store.find_test_plan(step.id)
            if version is not None and (test_plan is None or test_plan.id != version.test_plan_id):
                sections.append(
                    f'<section class="panel"><h2>Step {step.order + 1}: {escape_html(step.name)}</h2>'
                    '<p class="error-state">The saved plan relationship is invalid. This step is read-only.</p></section>'
                )
                continue

            active_form = submitted_step_id == step.id and bool(submitted_values)
            if active_form:
                state_url = submitted_values.get("plan_url", "")
                expected_version = submitted_values.get("expected_version", "0")
                actions = _automation_submitted_actions(submitted_values)
            elif version is not None:
                state_url = version.qa_test_plan.url
                expected_version = str(version.version)
                actions = [
                    {"type": action.action, "parameters": action.parameters}
                    for action in version.qa_test_plan.steps
                ]
            else:
                segment = next((item for item in test_case.segments if step in item.steps), None)
                state_url = (segment.base_url if segment else None) or test_case.base_url or ""
                expected_version = "0"
                actions = []

            title = f"Step {step.order + 1}: {escape_html(step.name)}"
            state = (
                '<p class="muted">Not automated. Create automation manually with supported actions.</p>'
                if version is None else
                f'<p>Current plan: <strong>v{version.version}</strong> · '
                f'{escape_html(_plan_origin_label(version.origin))} · '
                f'{escape_html(format_timestamp(version.created_at))}</p>'
            )
            coverage_html = _expected_result_coverage_html(
                expected_result_coverage(
                    step, version
                )
            )
            form_errors = "".join(
                f'<p class="field-error" role="alert">{escape_html(message)}</p>'
                for key, message in field_errors.items()
                if key in {"actions", "expected_version"}
            ) if active_form else ""
            action_html = "".join(
                _automation_action_card(
                    index, str(action.get("type", "")), action.get("parameters", {}),
                    field_errors if active_form else {},
                )
                for index, action in enumerate(actions)
            )
            url_error = field_errors.get("plan_url", "") if active_form else ""
            if _url_contains_credentials(state_url) or redact_secrets(state_url) != state_url:
                state_url = ""
                url_error = url_error or "This saved URL contains credentials. Replace it before saving."
            url_error_html = (
                f'<span class="field-error" role="alert">{escape_html(url_error)}</span>'
                if url_error else ""
            )
            form = (
                f'<form method="post" class="panel automation-editor-form" '
                f'action="/test-cases/{test_case.id}/automation/steps/{step.id}/save" data-automation-form>'
                f'<input type="hidden" name="expected_version" value="{escape_html(expected_version)}">'
                + form_errors
                + '<label class="field"><span>Plan URL</span>'
                f'<input name="plan_url" type="url" required value="{escape_html(state_url)}" data-automation-url></label>'
                + url_error_html
                + '<div class="automation-actions" data-automation-actions>' + action_html + '</div>'
                + '<div class="actions"><button class="button" type="button" data-action-add>Add action</button>'
                + '<button class="button primary" type="submit">Save automation</button>'
                + f'<a class="button" href="/test-cases/{test_case.id}">Cancel</a></div></form>'
            )
            history = self._plan_store.list_versions(step.id)
            history_html = _automation_history_html(test_case, step.id, version, history)
            sections.append(
                f'<section class="panel automation-step"><h2>{title}</h2>{state}{coverage_html}{form}{history_html}</section>'
            )

        content = (
            f'<header class="page-heading"><h1>Edit Automation: {escape_html(test_case.name)}</h1>'
            '<p class="lead">Edit saved browser actions by TestStep. Each save creates a new version, and current automation must pass Validation before it is ready.</p></header>'
            + notice
            + f'<div class="actions"><a class="button" href="/test-cases/{test_case.id}/plans">View TestPlan</a>'
            + f'<a class="button" href="/test-cases/{test_case.id}">Back to TestCase</a></div>'
            + ''.join(sections)
        )
        return WebResponse.html(status, self._page(
            "Edit Automation", content, current="Test Cases",
            breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases"),
                         (test_case.name, f"/test-cases/{test_case.id}")],
        ))

    def _automation_version_view(self, test_case_id: UUID, step_id: UUID, version_id: UUID) -> WebResponse:
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        step = next((item for item in test_case.steps if item.id == step_id), None) if test_case else None
        version = self._plan_store.get_version(version_id) if self._plan_store is not None else None
        test_plan = self._plan_store.find_test_plan(step_id) if self._plan_store is not None else None
        if (
            test_case is None or step is None or version is None or test_plan is None
            or version.test_plan_id != test_plan.id
        ):
            return self._not_found("Automation version not found")
        current = self._plan_store.find(step_id)
        actions = ''.join(
            '<li><strong>' + escape_html(ACTION_LABELS.get(action.action, action.action)) + '</strong><dl>'
            + ''.join(
                f'<dt>{escape_html(FIELD_LABELS.get(key, key))}</dt>'
                f'<dd><code>{escape_html(_safe_automation_url_display(str(value)) if key == "url" else redact_secrets(str(value)))}</code></dd>'
                for key, value in sorted(action.parameters.items())
            )
            + '</dl></li>' for action in version.qa_test_plan.steps
        )
        content = (
            f'<header class="page-heading"><h1>Automation version v{version.version}</h1>'
            f'<p class="lead">{escape_html(step.name)} · {_plan_origin_label(version.origin)} · '
            f'{escape_html(format_timestamp(version.created_at))}</p></header>'
            + _expected_result_coverage_html(
                expected_result_coverage(step, version)
            )
            + f'<section class="panel"><p>Plan URL: <code>{escape_html(_safe_automation_url_display(version.qa_test_plan.url))}</code></p>'
            + f'<ol>{actions}</ol></section>'
            + (f'<p class="muted">This is the current version.</p>' if current and current.id == version.id else '<p class="muted">Read-only historical version.</p>')
            + f'<div class="actions"><a class="button" href="/test-cases/{test_case.id}/automation/edit">Back to Automation Editor</a>'
            + f'<a class="button" href="/test-cases/{test_case.id}/plans">View TestPlan</a></div>'
        )
        return WebResponse.html(200, self._page(
            f"Automation v{version.version}", content, current="Test Cases",
            breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases"),
                         (test_case.name, f"/test-cases/{test_case.id}")],
        ))

    def _handle_automation_step_save(
        self, test_case_id_text: str, step_id_text: str, body: bytes | str | None
    ) -> WebResponse:
        test_case_id = _parse_uuid(test_case_id_text)
        step_id = _parse_uuid(step_id_text)
        test_case = self._test_cases.get(test_case_id) if test_case_id and self._test_cases is not None else None
        if test_case is None or step_id is None:
            return self._not_found("TestCase or TestStep not found")
        if self._test_case_review.status(test_case.id) != TestCaseReviewStatus.APPROVED:
            return WebResponse.html(409, self._page(
                "TestCase review required",
                '<div class="error-state"><h1>TestCase review required</h1>'
                '<p>Approve the TestCase before editing automation.</p>'
                f'<a class="button" href="/test-cases/{test_case.id}">Review TestCase</a></div>',
            ))
        step = next((item for item in test_case.steps if item.id == step_id), None)
        if self._automation_lifecycle.status(test_case) == AutomationStatus.NEEDS_UPDATE:
            return WebResponse.html(409, self._page("Automation needs update", '<p>Generate updated automation before editing plans for this definition.</p>'))
        if step is None:
            return self._not_found("TestStep not found")
        if self._plan_store is None:
            return WebResponse.html(503, self._page(
                "Automation editor unavailable",
                self._empty_state("Automation editor unavailable", "No saved plan store is configured."),
            ))
        form, parse_error = _parse_form_body(body, max_bytes=_MAX_MANUAL_FORM_BODY_BYTES)
        values = {key: items[0] for key, items in form.items()}
        if parse_error:
            return self._automation_editor_page(
                test_case.id, submitted_step_id=step.id, submitted_values=values,
                error_message=parse_error, status=400,
            )
        try:
            expected_version = int(values.get("expected_version", ""))
            if expected_version < 0:
                raise ValueError
        except ValueError:
            return self._automation_editor_page(
                test_case.id, submitted_step_id=step.id, submitted_values=values,
                field_errors={"expected_version": "Reload the editor before saving this plan."},
                status=400,
            )

        current = self._plan_store.find(step.id)
        current_number = current.version if current is not None else 0
        if current_number != expected_version:
            return self._automation_editor_page(
                test_case.id, submitted_step_id=step.id, submitted_values=values,
                error_message="This automation changed since you opened the editor. Reload and review the latest version.",
                status=409,
            )
        try:
            edited_plan = plan_from_form(values)
        except AutomationEditorError as error:
            return self._automation_editor_page(
                test_case.id, submitted_step_id=step.id, submitted_values=values,
                field_errors=error.field_errors, status=400,
            )

        test_plan = self._plan_store.find_test_plan(step.id)
        if current is not None and (test_plan is None or test_plan.id != current.test_plan_id):
            return self._automation_editor_page(
                test_case.id, submitted_step_id=step.id, submitted_values=values,
                error_message="The saved plan relationship changed. Reload and review the latest version.",
                status=409,
            )
        if test_plan is None:
            test_plan = TestPlan(test_step_id=step.id, name=step.name)
        version = TestPlanVersion(
            test_plan_id=test_plan.id,
            version=current_number + 1,
            origin=PlanVersionOrigin.HUMAN_EDITED,
            qa_test_plan=edited_plan,
        )
        try:
            self._plan_store.save(step.id, version, test_plan=test_plan)
        except ValueError:
            latest = self._plan_store.find(step.id)
            if (latest.version if latest is not None else 0) != expected_version:
                return self._automation_editor_page(
                    test_case.id, submitted_step_id=step.id, submitted_values=values,
                    error_message="This automation changed since you opened the editor. Reload and review the latest version.",
                    status=409,
                )
            return self._automation_editor_page(
                test_case.id, submitted_step_id=step.id, submitted_values=values,
                error_message="The automation could not be saved. Review the form and try again.",
                status=400,
            )

        if has_complete_plans(test_case, self._plan_store):
            self._automation_lifecycle.mark_automation_completed(test_case)
        return WebResponse.redirect(
            f"/test-cases/{test_case.id}/automation/edit?saved=1&step={step.id}"
        )

    def _handle_generate_test_case(self, body: bytes | str | None) -> WebResponse:
        form, form_error = _parse_form_body(body, max_bytes=_MAX_AUTHORING_FORM_BODY_BYTES)
        values = {
            key: form.get(key, [""])[0]
            for key in ("name", "base_url", "scenario")
        }
        source_draft_text = form.get("source_draft_id", [""])[0].strip()
        source_draft_id = _parse_uuid(source_draft_text) if source_draft_text else None
        source_draft = self._drafts.get(source_draft_id) if source_draft_id else None

        dashboard_entry = form.get("authoring_entry", [""])[0] == "dashboard"
        require_name = False
        field_errors: dict[str, str] = {}

        def error_response(status: int, message: str) -> WebResponse:
            if dashboard_entry:
                return WebResponse.html(status, self._dashboard(
                    authoring_error=message,
                    name=values["name"],
                    source_draft_id=source_draft_text,
                    base_url=values["base_url"],
                    scenario=values["scenario"],
                    field_errors=field_errors,
                ))
            return WebResponse.html(status, self._new_test_case_page(
                message or field_errors.get("name") or field_errors.get("base_url")
                or field_errors.get("scenario"),
                field_errors=field_errors,
                source_draft_id=source_draft_text, **values
            ))

        if form_error is not None:
            return error_response(400, form_error)
        if source_draft_text and (
            source_draft is None or source_draft.status != DraftStatus.ACTIVE
        ):
            return error_response(409, "The selected Draft is no longer active. Choose another Draft.")
        field_errors = _authoring_field_errors(
            values["name"], values["base_url"], values["scenario"],
            require_name=require_name,
        )
        if field_errors:
            return error_response(400, "")
        try:
            validated = TestCaseAuthoringService.validate_input(
                **values,
                require_name=require_name,
            )
        except TestCaseAuthoringError as error:
            return error_response(400, str(error))
        except Exception:
            return error_response(400, "Enter a valid Website and scenario.")
        if self._test_cases is None:
            return error_response(503, "TestCase storage is not configured.")
        if self._authoring_service is None or self._background_authoring is None:
            return error_response(503,
                "AI generation is unavailable. Configure an LLM provider in the environment and try again.",
            )
        try:
            progress_id = self._background_authoring.start(
                name=values["name"].strip(),
                base_url=validated.base_url,
                scenario=validated.scenario,
                source_draft_id=source_draft.id if source_draft is not None else None,
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
        source_draft_id = self._progress_store.get_authoring_source_draft_id(progress_id)
        try:
            validated = self._authoring_service.validate_input(name, scenario, base_url)
        except TestCaseAuthoringError as error:
            return WebResponse.html(400, self._new_test_case_page(
                str(error), name=name, base_url=base_url, scenario=scenario,
                source_draft_id=str(source_draft_id or ""),
            ))
        new_progress_id = self._background_authoring.start(
            name=name.strip(),
            base_url=validated.base_url,
            scenario=validated.scenario,
            source_draft_token=source_draft_token,
            source_draft_id=source_draft_id,
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
        source_draft_id = self._progress_store.get_authoring_source_draft_id(progress_id)
        if action == "save-draft":
            draft = Draft(title=name or "Untitled testing idea", body=scenario, base_url=base_url or None)
            self._drafts.save(draft)
            return WebResponse.redirect(f"/drafts/{draft.id}?saved=1")
        try:
            test_case = create_manual_test_case({
                "name": name,
                "description": scenario,
                "base_url": base_url,
                "step_name_0": "",
                "step_action_0": scenario,
                "step_expected_0": "",
            })
            if self._test_cases is None:
                return WebResponse.html(503, self._manual_test_case_page(
                    name=name, base_url=base_url, scenario=scenario,
                    error="TestCase storage is not configured.",
                ))
            self._test_cases.save(test_case)
            self._test_case_review.mark_ready_for_review(test_case)
        except (TestCaseEditError, ValueError) as create_error:
            return WebResponse.html(400, self._manual_test_case_page(
                name=name, base_url=base_url, scenario=scenario, error=str(create_error),
            ))
        if source_draft_id is not None:
            self._drafts.mark_used(source_draft_id, test_case.id)
        return WebResponse.redirect(f"/test-cases/{test_case.id}")

    def _handle_draft_action(
        self, token: str, action: str, body: bytes | str | None
    ) -> WebResponse:
        form, form_error = _parse_form_body(
            body,
            max_bytes=(
                _MAX_DRAFT_SAVE_BODY_BYTES if action in {"save", "regenerate"} else _MAX_FORM_BODY_BYTES
            ),
        )
        if form_error is not None:
            return self._not_found("Draft not found or expired")
        if any(len(items) != 1 for items in form.values()):
            return WebResponse.html(400, self._page("Invalid review", '<p>Submit each field once.</p>'))
        if action == "cancel":
            self._draft_store.take(token)
            return WebResponse.redirect("/test-cases")
        draft = self._draft_store.get(token)
        if draft is None:
            return self._not_found("Draft not found or expired")
        if action == "regenerate":
            summary = form.get("name", [draft.authoring_name or draft.test_case.name])[0]
            submitted_values = {key: items[0] for key, items in form.items()}
            if self._authoring_service is None:
                return WebResponse.html(503, self._draft_review_html(
                    draft, token, "AI generation is unavailable. Configure an LLM provider and try again.",
                    values=submitted_values,
                ))
            try:
                validated = self._authoring_service.validate_input(
                    summary,
                    draft.authoring_scenario or draft.test_case.description,
                    draft.authoring_base_url or draft.test_case.base_url or "",
                )
            except TestCaseAuthoringError as error:
                return WebResponse.html(400, self._draft_review_html(
                    draft, token, str(error), values=submitted_values,
                ))
            except Exception:
                return WebResponse.html(503, self._draft_review_html(
                    draft, token, "Authoring could not be started. Try again later.",
                    values=submitted_values,
                ))
            if self._background_authoring is None:
                return WebResponse.html(503, self._draft_review_html(
                    draft, token, "AI generation is unavailable. Try again later.",
                    values=submitted_values,
                ))
            progress_id = self._background_authoring.start(
                name=summary.strip(),
                base_url=validated.base_url,
                scenario=validated.scenario,
                source_draft_token=token,
                source_draft_id=draft.source_draft_id,
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
            retry_token = self._draft_store.put(consumed)
            return WebResponse.html(500, self._draft_review_html(
                consumed, retry_token, "The TestCase could not be saved. Your changes are retained; retry Save TestCase.",
                values={key: items[0] for key, items in form.items()},
            ))
        self._test_case_review.mark_ready_for_review(edits.test_case)
        if consumed.source_draft_id is not None:
            self._drafts.mark_used(consumed.source_draft_id, edits.test_case.id)
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

    def _draft_review_html(self, draft: TestCaseDraft, token: str, error: str | None = None, *, values: dict[str, str] | None = None, field_errors: dict[str, str] | None = None) -> str:
        if values and "steps_json" not in values:
            values = dict(values)
            values["steps_json"] = json.dumps([
                {"id": str(segment.id), "steps": [
                    {"id": str(step.id), "description": values.get(f"segment.{si}.step.{index}.description", step.description),
                     "expected": values.get(f"segment.{si}.step.{index}.expected", step.expected)}
                    for index, step in enumerate(segment.steps)]}
                for si, segment in enumerate(draft.test_case.segments)], ensure_ascii=False)
            values.setdefault("preconditions", "\n".join(values.get(f"precondition.{index}.description", item.description) for index, item in enumerate(draft.test_case.preconditions)))
        message = error or " ".join((field_errors or {}).values()) or None
        content = self._definition_editor(draft.test_case, f"/test-cases/review/{token}/save", f"/test-cases/review/{token}", message, values, review=True)
        content += (f'<form method="post" action="/test-cases/review/{escape_html(token)}/cancel" class="actions" data-discard-draft>'
                    f'<input type="hidden" name="_csrf" value="{self._csrf_token}">'
                    '<button class="button" type="submit">Discard Draft</button></form>')
        return self._page("Review TestCase", content, current="Test Cases")

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
        name: str = "",
        source_draft_id: str = "",
        base_url: str = "",
        scenario: str = "",
        field_errors: dict[str, str] | None = None,
    ) -> str:
        field_errors = field_errors or {}
        records = self._run_history.list_recent(_PAGE_LIMIT)
        counts = {
            "Total runs": len(records),
            "Passed": sum(history_outcome(record) == "PASSED" for record in records),
            "Automation runs": sum(record.workflow_type == WorkflowType.AUTOMATION for record in records),
            "Validation runs": sum(record.workflow_type == WorkflowType.VALIDATION for record in records),
            "Regression runs": sum(record.workflow_type == WorkflowType.REGRESSION for record in records),
            "Product failures": sum(history_outcome(record) == "PRODUCT_FAILURE" for record in records),
            "Automation execution errors": sum(history_outcome(record) == "AUTOMATION_EXECUTION_ERROR" for record in records),
            "Generation errors": sum(history_outcome(record) == "AUTOMATION_GENERATION_ERROR" for record in records),
            "Blocked": sum(history_outcome(record) == "BLOCKED" for record in records),
            "Inconclusive": sum(history_outcome(record) == "INCONCLUSIVE" for record in records),
            "Automation drift": sum(record.outcome == "AUTOMATION_DRIFT" for record in records),
            "Infrastructure errors": sum(history_outcome(record) == "INFRASTRUCTURE_ERROR" for record in records),
        }
        cards = "".join(
            f'<article class="card"><div class="card-label">{escape_html(label)}</div>'
            f'<div class="card-value">{count}</div></article>'
            for label, count in counts.items()
        )
        recent = records[:12]
        authoring_entry = self._test_case_creation_form(
            entry="dashboard", error=authoring_error, name=name, base_url=base_url,
            scenario=scenario, source_draft_id=source_draft_id, field_errors=field_errors,
        )
        content = (
            '<header class="page-heading"><h1>AI QA Agent</h1></header>'
            + authoring_entry
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

    def _test_case_creation_form(
        self, *, entry: str = "new", error: str | None = None, name: str = "",
        base_url: str = "", scenario: str = "", source_draft_id: str = "",
        field_errors: dict[str, str] | None = None,
    ) -> str:
        """Render the same editable creation form and Drafts panel on both pages."""
        field_errors = field_errors or {}
        selected_id = _parse_uuid(source_draft_id) if source_draft_id else None
        selected = self._drafts.get(selected_id) if selected_id else None
        if selected is not None and selected.status != DraftStatus.ACTIVE:
            selected = None
        error_html = (
            f'<p class="authoring-error" id="case-authoring-error" role="alert">{escape_html(error)}</p>'
            if error else ""
        )
        selected_html = (
            '<p class="draft-selection-status" data-draft-selection-status role="status" aria-live="polite">'
            f'Loaded Draft: <strong>{escape_html(selected.title)}</strong>. Edits are saved only when you choose Save Draft.</p>'
            if selected is not None else
            '<p class="draft-selection-status muted" data-draft-selection-status role="status" aria-live="polite">Choose a Draft to load its Summary, Website and Scenario.</p>'
        )
        return (
            '<div class="new-testcase-layout"><section class="panel authoring-entry" aria-labelledby="authoring-entry-title">'
            '<h2 id="authoring-entry-title">What do you want to test?</h2>'
            '<form method="post" action="/test-cases/generate" data-authoring-form data-inline-validation novalidate>'
            f'<input type="hidden" name="authoring_entry" value="{escape_html(entry)}">'
            f'<input type="hidden" name="source_draft_id" value="{escape_html(source_draft_id)}" data-source-draft-id>'
            + error_html + selected_html
            + '<div class="field"><label for="case-name">Summary (optional)</label>'
            + f'<input id="case-name" name="name" maxlength="200" value="{escape_html(name)}"'
            + (_inline_invalid_attrs("case-name-error", field_errors.get("name")) if field_errors.get("name") else ' aria-describedby="case-summary-help"') + '>'
            + '<small id="case-summary-help">Leave this empty for a suggestion from your Scenario. You can edit it during review.</small></div>'
            + _inline_error_html("case-name-error", "case-name", field_errors.get("name"))
            + '<div class="field"><label for="case-base-url">Website</label>'
            + f'<input id="case-base-url" name="base_url" type="url" maxlength="2048" required value="{escape_html(base_url)}" placeholder="https://example.com"{_inline_invalid_attrs("case-website-error", field_errors.get("base_url"))}></div>'
            + _inline_error_html("case-website-error", "case-base-url", field_errors.get("base_url"))
            + '<div class="field"><label for="case-scenario">Scenario</label>'
            + f'<textarea id="case-scenario" name="scenario" rows="8" maxlength="6000" required placeholder="Describe an action and the result you expect..."{_inline_invalid_attrs("case-scenario-error", field_errors.get("scenario"))}>{escape_html(scenario)}</textarea></div>'
            + _inline_error_html("case-scenario-error", "case-scenario", field_errors.get("scenario"))
            + self._authoring_voice_controls("case-scenario")
            + '<div class="actions creation-actions"><button class="button primary" type="submit">Generate TestCase</button>'
            + '<button class="button" type="submit" formaction="/test-cases/drafts/save" formnovalidate>Save Draft</button>'
            + '<button class="button" type="submit" formaction="/test-cases/manual/prepare" formnovalidate>Create Manually</button></div>'
            + '</form><p class="muted">Review the TestCase before approving it. Saving does not start automation.</p>'
            + '</section>' + self._draft_selection_panel(selected.id if selected else None) + '</div>'
        )

    def _new_test_case_page(
        self, error: str | None = None, *, name: str = "", base_url: str = "",
        scenario: str = "", source_draft_id: str = "",
        field_errors: dict[str, str] | None = None,
    ) -> str:
        content = (
            '<header class="page-heading"><h1>New Test Case</h1>'
            '<p class="lead">Describe what you want to verify, then review the generated TestCase.</p></header>'
            + self._test_case_creation_form(
                error=error, name=name, base_url=base_url, scenario=scenario,
                source_draft_id=source_draft_id, field_errors=field_errors,
            )
        )
        return self._page(
            "New Test Case", content, current="Test Cases",
            breadcrumbs=[("Dashboard", "/"), ("Test Cases", "/test-cases")],
        )

    def _draft_selection_panel(self, selected_draft_id: UUID | None = None) -> str:
        active = [item for item in self._drafts.list(500) if item.status == DraftStatus.ACTIVE][:20]
        rows = []
        for item in active:
            preview = " ".join(item.body.split())
            preview = preview[:240].rstrip() + ("…" if len(preview) > 240 else "")
            rows.append(
                f'<li><button class="draft-select-button" type="button" data-draft-select '
                f'data-draft-id="{item.id}" data-title="{escape_html(item.title)}" '
                f'data-website="{escape_html(item.base_url or "")}" '
                f'data-scenario="{escape_html(item.body)}" '
                f'aria-pressed="{"true" if item.id == selected_draft_id else "false"}">'
                f'<strong>{escape_html(item.title)}</strong>'
                f'<span class="draft-scenario-preview">{escape_html(preview or "No scenario yet.")}</span>'
                f'</button><a class="draft-details" href="/drafts/{item.id}">View details</a></li>'
            )
        list_html = (
            '<ul class="draft-sidebar-list">' + "".join(rows) + '</ul>' if rows else
            '<p class="muted drafts-empty">No active Drafts yet. Save an unfinished idea to reuse it here.</p>'
        )
        return (
            '<aside class="panel drafts-sidebar" aria-labelledby="case-drafts-title">'
            '<div class="section-heading"><h2 id="case-drafts-title">Drafts</h2>'
            '<a class="button" href="/drafts">View all</a></div>'
            + list_html + '<a class="button" href="/drafts/new">New Draft</a></aside>'
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
                status = _outcome_badge(latest) if latest else badge("NOT RUN")
                automation_status = badge(
                    _automation_status_label(self._automation_lifecycle.status(test_case)),
                    "workflow",
                )
                last_run = (
                    f'<time datetime="{escape_html(latest.started_at.isoformat())}">'
                    f'{escape_html(format_timestamp(latest.started_at))}</time>'
                    if latest else '<span class="muted">No runs yet</span>'
                )
                latest_workflow = (
                    badge(latest.workflow_type.value, "workflow")
                    if latest else '<span class="muted">—</span>'
                )
                run_count = f'{len(history)} {"run" if len(history) == 1 else "runs"}'
                rows.append(
                    "<tr>"
                    + f'<td class="test-case-main-cell"><div class="status-line"><span class="id-code">{escape_html(test_case.public_id or "")}</span>'
                    f'<a href="/test-cases/{test_case_id}">{escape_html(test_case.name)}</a></div>'
                    f'<details><summary>Technical ID</summary><code>{escape_html(test_case_id)}</code></details></td>'
                    f'<td class="test-case-state-cell"><div class="test-case-state">{automation_status}'
                    f'<span class="test-case-latest-status"><span class="muted">Latest result:</span> {status}</span></div>'
                    f'<div class="test-case-meta"><span>Last run: {last_run}</span>'
                    f'<span>{run_count}</span>'
                    f'<span>Latest workflow: {latest_workflow}</span></div></td>'
                    '<td class="test-case-actions-cell"><div class="test-case-actions">'
                    + f'<a class="button" href="/test-cases/{test_case_id}">Open</a>'
                    + f'<a class="button" href="/test-cases/{test_case_id}/edit">Edit</a>'
                    + (
                        f'<a class="button" href="/export?mode=testcases&amp;case={quote(test_case.public_id or "")}">Export</a>'
                        if self._testplan_exports is not None else ""
                    )
                    + '</div></td></tr>'
                )
        else:
            for test_case_id, latest in latest_by_case.items():
                history = grouped[test_case_id]
                rows.append(
                    "<tr>"
                    f'<td class="test-case-main-cell"><div class="status-line"><span class="id-code">{escape_html(latest.test_case_public_id or "")}</span>'
                    f'<a href="/test-cases/{test_case_id}">{escape_html(latest.test_case_name)}</a></div>'
                    f'<details><summary>Technical ID</summary><code>{escape_html(test_case_id)}</code></details></td>'
                    f'<td class="test-case-state-cell"><div class="test-case-state"><span class="badge neutral">Automation unavailable</span>'
                    f'<span class="test-case-latest-status"><span class="muted">Latest result:</span> {_outcome_badge(latest)}</span></div>'
                    f'<div class="test-case-meta"><span>Last run: <time datetime="{escape_html(latest.started_at.isoformat())}">{escape_html(format_timestamp(latest.started_at))}</time></span>'
                    f'<span>{len(history)} {"run" if len(history) == 1 else "runs"}</span>'
                    f'<span>Latest workflow: {badge(latest.workflow_type.value, "workflow")}</span></div></td>'
                    f'<td class="test-case-actions-cell"><div class="test-case-actions"><a class="button" href="/test-cases/{test_case_id}">Open</a></div></td>'
                    "</tr>"
                )
        table = (
            '<div class="table-wrap"><table class="test-case-list"><thead><tr><th scope="col">TestCase</th>'
            '<th scope="col">Automation and run history</th><th scope="col">Actions</th></tr></thead><tbody>'
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

    def _export_workspace(self, query: dict[str, list[str]], *, notice: str | None = None) -> str:
        mode = query.get("mode", ["testcases"])[0].casefold()
        mode = mode if mode in {"testcases", "suites"} else "testcases"
        search = query.get("q", [""])[0].strip()[:160]
        page_raw = query.get("page", ["1"])[0]
        page = int(page_raw) if page_raw.isdigit() else 1
        page = min(max(page, 1), 100_000)

        selected_case_keys = []
        if self._test_cases is not None:
            for key in query.get("case", []):
                key = key.strip().upper()
                if key.startswith("TC-") and len(key) >= 7 and key[3:].isdigit():
                    if self._test_cases.get_by_public_id(key) is not None and key not in selected_case_keys:
                        selected_case_keys.append(key)
        selected_suites = []
        for raw_suite_id in query.get("suite", []):
            suite_id = _parse_uuid(raw_suite_id)
            if suite_id is not None and self._test_suites.get(suite_id) is not None and suite_id not in selected_suites:
                selected_suites.append(suite_id)

        extra_messages = [notice] if notice else []
        if self._testplan_exports is None:
            extra_messages.append("Export is unavailable because saved TestCase and plan storage is not configured.")

        date_values: dict[str, str] = {}
        date_bounds: dict[str, str] = {}
        for key in ("created_from", "created_to", "updated_from", "updated_to"):
            raw = query.get(key, [""])[0]
            date_values[key] = raw
            if raw:
                try:
                    parsed_date = date.fromisoformat(raw)
                    date_bounds[key] = (
                        (parsed_date + timedelta(days=1)).isoformat()
                        if key.endswith("_to") else parsed_date.isoformat()
                    )
                except ValueError:
                    extra_messages.append(f"{key.replace('_', ' ').capitalize()} must be a valid date.")
                    date_values[key] = ""
                    date_bounds[key] = ""
        for lower_key, upper_key in (("created_from", "created_to"), ("updated_from", "updated_to")):
            if date_values[lower_key] and date_values[upper_key] and date_values[lower_key] > date_values[upper_key]:
                extra_messages.append(f"{lower_key.split('_')[0].capitalize()} date range is reversed.")
                date_bounds[lower_key] = date_bounds[upper_key] = ""
                date_values[lower_key] = date_values[upper_key] = ""

        selected_test_suite = query.get("test_suite", [""])[0]
        suite_filter_id = _parse_uuid(selected_test_suite) if selected_test_suite else None
        if suite_filter_id is not None and self._test_suites.get(suite_filter_id) is None:
            suite_filter_id = None
            selected_test_suite = ""
        lifecycle_filter = query.get("lifecycle", ["ALL"])[0].upper()
        if lifecycle_filter not in {"ALL", *(status.value for status in AutomationStatus)}:
            lifecycle_filter = "ALL"
        suite_readiness_filter = query.get("readiness", ["ALL"])[0].upper()
        if suite_readiness_filter not in {"ALL", "READY", "BLOCKED"}:
            suite_readiness_filter = "ALL"

        case_mode_current = ' aria-current="page"' if mode == "testcases" else ""
        suite_mode_current = ' aria-current="page"' if mode == "suites" else ""
        tabs = (
            '<nav class="export-mode-tabs" aria-label="Export source">'
            f'<a href="{escape_html(self._export_mode_url("testcases", selected_case_keys, selected_suites))}"'
            f'{case_mode_current}>TestCases</a>'
            f'<a href="{escape_html(self._export_mode_url("suites", selected_case_keys, selected_suites))}"'
            f'{suite_mode_current}>Test Suites</a>'
            '</nav>'
        )

        selected_cases_by_id: dict[UUID, TestCase] = {}
        for public_id in selected_case_keys:
            case = self._test_cases.get_by_public_id(public_id) if self._test_cases is not None else None
            if case is not None:
                selected_cases_by_id[case.id] = case
        selected_suite_names = []
        for suite_id in selected_suites:
            suite = self._test_suites.get(suite_id)
            if suite is None:
                continue
            selected_suite_names.append(suite.name)
            for case in self._test_suites.members(suite_id):
                selected_cases_by_id.setdefault(case.id, case)

        selected_panel = self._export_selection_panel(
            selected_case_keys, selected_suites, selected_suite_names,
            list(selected_cases_by_id.values()),
        )
        main_list = ""
        page_links = ""

        if mode == "testcases":
            suite_options = ['<option value="">All Test Suites</option>']
            for suite in self._test_suites.list():
                selected = " selected" if suite_filter_id == suite.id else ""
                suite_options.append(
                    f'<option value="{suite.id}"{selected}>{escape_html(suite.name)}</option>'
                )
            status_options = ['<option value="ALL">All lifecycle statuses</option>']
            for status in AutomationStatus:
                selected = " selected" if lifecycle_filter == status.value else ""
                status_options.append(
                    f'<option value="{status.value}"{selected}>{escape_html(_automation_status_label(status))}</option>'
                )
            form_hidden = [f'<input type="hidden" name="mode" value="testcases">']
            visible_case_keys: set[str] = set()
            entries: list = []
            total = 0
            has_next = False
            if self._test_cases is not None:
                member_ids = (
                    self._test_suites.member_ids(suite_filter_id)
                    if suite_filter_id is not None else None
                )
                catalog_options = dict(
                    search=search,
                    created_from=date_bounds.get("created_from", ""),
                    created_before=date_bounds.get("created_to", ""),
                    updated_from=date_bounds.get("updated_from", ""),
                    updated_before=date_bounds.get("updated_to", ""),
                    test_case_ids=member_ids,
                )
                if lifecycle_filter == "ALL":
                    entries, total = self._test_cases.list_export_catalog(
                        **catalog_options,
                        offset=(page - 1) * _EXPORT_PAGE_SIZE,
                        limit=_EXPORT_PAGE_SIZE,
                    )
                    has_next = page * _EXPORT_PAGE_SIZE < total
                else:
                    wanted_start = (page - 1) * _EXPORT_PAGE_SIZE
                    matched = 0
                    source_offset = 0
                    source_total = None
                    while source_total is None or source_offset < source_total:
                        batch, source_total = self._test_cases.list_export_catalog(
                            **catalog_options, offset=source_offset, limit=500,
                        )
                        if not batch:
                            break
                        for entry in batch:
                            if self._automation_lifecycle.status(entry.test_case).value != lifecycle_filter:
                                continue
                            if wanted_start <= matched < wanted_start + _EXPORT_PAGE_SIZE:
                                entries.append(entry)
                            elif matched >= wanted_start + _EXPORT_PAGE_SIZE:
                                has_next = True
                                break
                            matched += 1
                        if has_next:
                            break
                        source_offset += len(batch)
                    total = matched
            visible_case_keys = {entry.test_case.public_id or "" for entry in entries}
            for key in selected_case_keys:
                if key not in visible_case_keys:
                    form_hidden.append(f'<input type="hidden" name="case" value="{escape_html(key)}">')
            form_hidden.extend(
                f'<input type="hidden" name="suite" value="{suite_id}">'
                for suite_id in selected_suites
            )
            rows = []
            for entry in entries:
                case = entry.test_case
                public_id = case.public_id or ""
                visible_case_keys.add(public_id)
                lifecycle_status, exportable, reason = self._export_case_readiness(case)
                readiness = (
                    '<span class="badge success">Exportable</span>'
                    if exportable else '<span class="badge warning">Blocked</span>'
                )
                if reason:
                    readiness += f'<p class="muted export-reason">{escape_html(reason)}</p>'
                selection = " checked" if public_id in selected_case_keys else ""
                rows.append(
                    '<tr><td><label class="export-select-label">'
                    f'<input type="checkbox" name="case" value="{escape_html(public_id)}" aria-label="Select {escape_html(public_id)} {escape_html(case.name)}"{selection}>'
                    f'<span><span class="id-code">{escape_html(public_id)}</span> '
                    f'<a href="/test-cases/{case.id}">{escape_html(case.name)}</a></span></label></td>'
                    f'<td>{badge(_automation_status_label(lifecycle_status), "workflow")}</td>'
                    f'<td>{readiness}</td>'
                    f'<td><time datetime="{escape_html(entry.created_at.isoformat())}">{escape_html(format_timestamp(entry.created_at))}</time></td>'
                    f'<td><time datetime="{escape_html(entry.updated_at.isoformat())}">{escape_html(format_timestamp(entry.updated_at))}</time></td></tr>'
                )
            filter_form = (
                '<form class="export-filter-form" method="get" action="/export">'
                + "".join(form_hidden)
                + '<input type="hidden" name="page" value="1">'
                + '<div class="filters export-filters">'
                + f'<div class="field"><label for="export-search">Search TestCases</label><input id="export-search" name="q" type="search" value="{escape_html(search)}" placeholder="Public ID or name"></div>'
                + '<div class="field"><label for="export-suite-filter">Test Suite</label><select id="export-suite-filter" name="test_suite">'
                + "".join(suite_options) + '</select></div>'
                + '<div class="field"><label for="export-lifecycle-filter">Automation lifecycle</label><select id="export-lifecycle-filter" name="lifecycle">'
                + "".join(status_options) + '</select></div>'
                + "".join(
                    f'<div class="field"><label for="filter-{key}">{label}</label><input id="filter-{key}" type="date" name="{key}" value="{escape_html(date_values[key])}"></div>'
                    for key, label in (("created_from", "Created from"), ("created_to", "Created to"), ("updated_from", "Updated from"), ("updated_to", "Updated to"))
                )
                + '<button class="button" type="submit">Apply filters and selection</button></div></form>'
            )
            table = (
                '<div class="table-wrap"><table class="export-case-table"><thead><tr>'
                '<th scope="col">TestCase</th><th scope="col">Automation lifecycle</th>'
                '<th scope="col">Export readiness</th><th scope="col">Created</th><th scope="col">Updated</th>'
                '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
                if rows else self._empty_state("No TestCases match these filters.", "Change the search or filters and try again.")
            )
            if page > 1 or has_next:
                nav_params = [(key, value) for key, values in query.items() for value in values if key != "page"]
                links = []
                if page > 1:
                    links.append(f'<a class="button" href="/export?{escape_html(urlencode(nav_params + [("page", str(page - 1))]))}">Previous</a>')
                if has_next:
                    links.append(f'<a class="button" href="/export?{escape_html(urlencode(nav_params + [("page", str(page + 1))]))}">Next</a>')
                page_links = '<nav class="export-pagination" aria-label="TestCase pages">' + "".join(links) + '</nav>'
            main_list = (
                '<section class="panel"><h2>Choose TestCases</h2>'
                f'<p class="muted">Showing {len(entries)} on this page' + (f' of {total} matching TestCases.' if lifecycle_filter == "ALL" else ' matching the selected filters.') + '</p>'
                + filter_form + table + page_links + '</section>'
            )
        else:
            suite_search = search
            form_hidden = ['<input type="hidden" name="mode" value="suites">']
            form_hidden.extend(f'<input type="hidden" name="case" value="{escape_html(key)}">' for key in selected_case_keys)
            suite_rows = []
            visible_suite_ids: set[UUID] = set()
            page_suites: list = []
            total_suites = 0
            has_next = False
            if suite_readiness_filter == "ALL":
                page_suites, total_suites = self._test_suites.list_page(
                    search=suite_search, offset=(page - 1) * _EXPORT_PAGE_SIZE, limit=_EXPORT_PAGE_SIZE,
                )
                has_next = page * _EXPORT_PAGE_SIZE < total_suites
            else:
                matched = 0
                source_offset = 0
                source_total = 0
                while source_offset < source_total or source_offset == 0:
                    batch, source_total = self._test_suites.list_page(
                        search=suite_search, offset=source_offset, limit=500,
                    )
                    if not batch:
                        break
                    for suite in batch:
                        summary = self._suite_export_counts(suite.id)
                        is_match = summary["ready"] > 0 if suite_readiness_filter == "READY" else summary["blocked"] > 0
                        if not is_match:
                            continue
                        if (page - 1) * _EXPORT_PAGE_SIZE <= matched < page * _EXPORT_PAGE_SIZE:
                            page_suites.append(suite)
                        elif matched >= page * _EXPORT_PAGE_SIZE:
                            has_next = True
                            break
                        matched += 1
                    if has_next:
                        break
                    source_offset += len(batch)
                total_suites = matched
            for suite in page_suites:
                visible_suite_ids.add(suite.id)
                summary = self._suite_export_counts(suite.id)
                selected = " checked" if suite.id in selected_suites else ""
                suite_rows.append(
                    '<tr><td><label class="export-select-label">'
                    f'<input type="checkbox" name="suite" value="{suite.id}" aria-label="Select Test Suite {escape_html(suite.name)}"{selected}>'
                    f'<span><a href="/test-suites/{suite.id}">{escape_html(suite.name)}</a>'
                    + (f'<span class="muted export-suite-description">{escape_html(suite.description)}</span>' if suite.description else '')
                    + '</span></label></td>'
                    + f'<td>{summary["members"]} TestCases</td>'
                    + f'<td>{summary["ready"]} Automation ready</td>'
                    + f'<td>{summary["blocked"]} blocked</td></tr>'
                )
            form_hidden.extend(
                f'<input type="hidden" name="suite" value="{suite_id}">'
                for suite_id in selected_suites if suite_id not in visible_suite_ids
            )
            readiness_options = (
                '<option value="ALL">Any readiness</option>'
                f'<option value="READY"{" selected" if suite_readiness_filter == "READY" else ""}>Has Automation Ready tests</option>'
                f'<option value="BLOCKED"{" selected" if suite_readiness_filter == "BLOCKED" else ""}>Has blocked tests</option>'
            )
            filter_form = (
                '<form class="export-filter-form" method="get" action="/export">'
                + "".join(form_hidden)
                + '<input type="hidden" name="page" value="1">'
                + '<div class="filters export-filters">'
                + f'<div class="field"><label for="export-search">Search Test Suites</label><input id="export-search" name="q" type="search" value="{escape_html(suite_search)}" placeholder="Suite name"></div>'
                + '<div class="field"><label for="export-readiness-filter">Readiness</label><select id="export-readiness-filter" name="readiness">'
                + readiness_options + '</select></div><button class="button" type="submit">Apply filters and selection</button></div></form>'
            )
            table = (
                '<div class="table-wrap"><table class="export-suite-table"><thead><tr>'
                '<th scope="col">Test Suite</th><th scope="col">Members</th><th scope="col">Automation Ready</th><th scope="col">Blocked</th>'
                '</tr></thead><tbody>' + "".join(suite_rows) + '</tbody></table></div>'
                if suite_rows else self._empty_state("No Test Suites match these filters.", "Change the search or readiness filter and try again.")
            )
            if page > 1 or has_next:
                nav_params = [(key, value) for key, values in query.items() for value in values if key != "page"]
                links = []
                if page > 1:
                    links.append(f'<a class="button" href="/export?{escape_html(urlencode(nav_params + [("page", str(page - 1))]))}">Previous</a>')
                if has_next:
                    links.append(f'<a class="button" href="/export?{escape_html(urlencode(nav_params + [("page", str(page + 1))]))}">Next</a>')
                page_links = '<nav class="export-pagination" aria-label="Test Suite pages">' + "".join(links) + '</nav>'
            main_list = (
                '<section class="panel"><h2>Choose Test Suites</h2>'
                f'<p class="muted">Showing {len(page_suites)} on this page' + (f' of {total_suites} matching Test Suites.' if suite_readiness_filter == "ALL" else ' matching the selected filters.') + '</p>'
                + filter_form + table + page_links + '</section>'
            )

        notices = "".join(
            f'<div class="notice warning">{escape_html(message)}</div>'
            for message in extra_messages
        )
        body = (
            '<header class="page-heading"><h1>Export</h1>'
            '<p class="lead">Select TestCases or suites, review readiness, then export saved automation.</p></header>'
            + tabs + notices + selected_panel + main_list
        )
        return self._page("Export", body, current="Export", breadcrumbs=[("Dashboard", "/")])

    def _export_mode_url(self, mode: str, case_keys: list[str], suite_ids: list[UUID]) -> str:
        params = [("mode", mode)]
        params.extend(("case", key) for key in case_keys)
        params.extend(("suite", str(suite_id)) for suite_id in suite_ids)
        return "/export?" + urlencode(params)

    def _export_case_readiness(self, test_case: TestCase) -> tuple[AutomationStatus, bool, str]:
        try:
            status = self._automation_lifecycle.status(test_case)
        except Exception as error:
            logger.warning("Export readiness check failed safely (%s)", type(error).__name__)
            return AutomationStatus.AUTOMATION_FAILED, False, "Automation readiness could not be verified safely."
        blocked_reasons = {
            AutomationStatus.NOT_AUTOMATED: "Automation is not complete.",
            AutomationStatus.NEEDS_VALIDATION: "Saved automation needs Validation before export.",
            AutomationStatus.NEEDS_UPDATE: "Saved automation is stale and needs an update.",
            AutomationStatus.AUTOMATION_FAILED: "Automation needs attention before export.",
        }
        if status != AutomationStatus.AUTOMATION_READY:
            return status, False, blocked_reasons.get(status, "Automation is not ready for export.")
        if self._testplan_exports is None:
            return status, False, "The saved automation exporter is unavailable."
        try:
            self._testplan_exports.get(test_case.id)
        except TestPlanExportError as error:
            reason = "; ".join(dict.fromkeys(blocker.reason for blocker in error.blockers)) or str(error)
            return status, False, reason
        except Exception as error:
            logger.warning("Export plan check failed safely (%s)", type(error).__name__)
            return status, False, "Saved automation could not be verified safely."
        return status, True, ""

    def _suite_export_counts(self, suite_id: UUID) -> dict[str, int]:
        members = self._test_suites.members(suite_id)
        ready = 0
        blocked = 0
        for case in members:
            status, exportable, _reason = self._export_case_readiness(case)
            if status == AutomationStatus.AUTOMATION_READY:
                ready += 1
            if not exportable:
                blocked += 1
        return {"members": len(members), "ready": ready, "blocked": blocked}

    def _export_selection_panel(
        self,
        case_keys: list[str],
        suite_ids: list[UUID],
        suite_names: list[str],
        test_cases: list[TestCase],
    ) -> str:
        rows = []
        for case in sorted(test_cases, key=lambda item: (item.public_id or "", item.name.casefold())):
            status, exportable, reason = self._export_case_readiness(case)
            readiness = '<span class="badge success">Exportable</span>' if exportable else '<span class="badge warning">Blocked</span>'
            if reason:
                readiness += f'<span class="muted export-reason">{escape_html(reason)}</span>'
            rows.append(
                '<tr><td><span class="id-code">' + escape_html(case.public_id or "") + '</span> '
                + f'<a href="/test-cases/{case.id}">{escape_html(case.name)}</a></td>'
                + f'<td>{badge(_automation_status_label(status), "workflow")}</td><td>{readiness}</td></tr>'
            )
        source_labels = []
        source_labels.extend(escape_html(key) for key in case_keys)
        source_labels.extend(escape_html(name) for name in suite_names)
        selected_label = (
            f'{len(case_keys)} direct TestCases and {len(suite_ids)} Test Suites; '
            f'{len(test_cases)} unique TestCases after suite membership is resolved.'
        )
        if not case_keys and not suite_ids:
            selected_label = "No items selected yet. Select TestCases or Test Suites below."
        source_html = (
            '<p class="muted">Selected sources: ' + (", ".join(source_labels) if source_labels else "none") + '</p>'
        )
        clear_selection = (
            '<a class="button" href="/export">Clear selection</a>'
            if case_keys or suite_ids else ""
        )
        export_form = ""
        if self._testplan_exports is not None and test_cases:
            hidden = ''.join(
                f'<input type="hidden" name="case" value="{escape_html(key)}">' for key in case_keys
            ) + ''.join(
                f'<input type="hidden" name="suite" value="{suite_id}">' for suite_id in suite_ids
            )
            export_form = (
                '<form method="post" action="/export" class="export-download-form">'
                + hidden
                + '<div class="filters"><div class="field"><label for="export-target">Format</label><select id="export-target" name="target">'
                + '<option value="portable">Portable JSON</option><option value="python">Python Playwright</option>'
                + '<option value="typescript">TypeScript Playwright</option><option value="csharp">C# Playwright</option>'
                + '</select></div>'
                + '<div class="export-policies"><button class="button primary" type="submit" name="policy" value="require_all">Export all selected (require all exportable)</button>'
                + '<button class="button" type="submit" name="policy" value="ready_only">Export ready items only</button></div></div>'
                + '<p class="muted">Ready-only export excludes blocked cases listed above. Review the reasons before choosing that option.</p></form>'
            )
        return (
            '<section class="panel export-selected"><div class="section-heading"><h2>Selected items and readiness</h2>'
            + clear_selection + '</div>'
            + f'<p>{escape_html(selected_label)}</p>' + source_html
            + ('<div class="table-wrap"><table class="export-selected-table"><thead><tr><th scope="col">TestCase</th>'
               '<th scope="col">Automation lifecycle</th><th scope="col">Export readiness</th></tr></thead><tbody>'
               + "".join(rows) + '</tbody></table></div>' if rows else '')
            + export_form + '</section>'
        )

    def _handle_export_workspace(self, body: bytes | str | None) -> WebResponse:
        if self._testplan_exports is None or self._test_cases is None:
            return WebResponse.html(503, self._page(
                "Export unavailable", '<section class="panel error-state"><h1>Export unavailable</h1>'
                '<p>Saved TestCase and plan storage is not configured.</p></section>', current="Export",
            ))
        form, error = _parse_form_body(
            body, max_bytes=64 * 1024, allow_repeated_fields={"case", "suite"},
        )
        if error:
            return self._export_post_error(error, 400)
        raw_case_keys = list(dict.fromkeys(value.strip().upper() for value in form.get("case", []) if value.strip()))
        raw_suite_ids = list(dict.fromkeys(value.strip() for value in form.get("suite", []) if value.strip()))
        if not raw_case_keys and not raw_suite_ids:
            return self._export_post_error("Select one or more TestCases or Test Suites.", 400)
        test_cases: dict[UUID, TestCase] = {}
        for public_id in raw_case_keys:
            if not public_id.startswith("TC-") or len(public_id) < 7 or not public_id[3:].isdigit():
                return self._export_post_error("A selected TestCase ID is invalid.", 400)
            case = self._test_cases.get_by_public_id(public_id)
            if case is None:
                return self._export_post_error("A selected TestCase is no longer available.", 404)
            test_cases[case.id] = case
        selected_suite_uuids = []
        for raw_suite_id in raw_suite_ids:
            suite_id = _parse_uuid(raw_suite_id)
            if suite_id is None:
                return self._export_post_error("A selected Test Suite is invalid.", 400)
            suite = self._test_suites.get(suite_id)
            if suite is None:
                return self._export_post_error("A selected Test Suite is no longer available.", 404)
            selected_suite_uuids.append(suite_id)
            for case in self._test_suites.members(suite_id):
                test_cases.setdefault(case.id, case)
        if not test_cases:
            return self._export_post_error("The selected sources do not contain any TestCases.", 400)

        target = form.get("target", [""])[0].casefold()
        if target not in {"portable", "python", "typescript", "csharp"}:
            return self._export_post_error("Choose Portable JSON, Python, TypeScript, or C# export.", 400)
        policy = form.get("policy", [""])[0]
        if policy not in {"require_all", "ready_only"}:
            return self._export_post_error("Choose whether all selected items must be exportable or only ready items should be exported.", 400)

        exportable_ids = []
        blocked = []
        for case in test_cases.values():
            _status, ready, reason = self._export_case_readiness(case)
            if ready:
                exportable_ids.append(case.id)
            else:
                blocked.append((case, reason))
        if blocked and policy == "require_all":
            return self._export_blocked_response(blocked, raw_case_keys, selected_suite_uuids)
        if not exportable_ids:
            return self._export_post_error("No selected TestCases are exportable yet.", 409)
        try:
            if target == "portable" and len(exportable_ids) == 1:
                content, filename = self._testplan_exports.portable_json(exportable_ids[0])
                return _download_response(content.encode("utf-8"), filename, "application/json; charset=utf-8")
            content, filename = self._testplan_exports.bulk_zip(exportable_ids, target)
            return _download_response(content, filename, "application/zip")
        except TestPlanExportError as export_error:
            return self._export_post_error(str(export_error), 409)
        except Exception as export_error:
            logger.warning("Central export failed safely (%s)", type(export_error).__name__)
            return self._export_post_error("The selected saved automation could not be exported.", 500)

    def _export_post_error(self, message: str, status: int) -> WebResponse:
        content = (
            '<section class="panel error-state"><h1>Export unavailable</h1>'
            f'<p>{escape_html(message)}</p><a class="button" href="/export">Back to Export</a></section>'
        )
        return WebResponse.html(status, self._page("Export unavailable", content, current="Export"))

    def _export_blocked_response(
        self,
        blocked: list[tuple[TestCase, str]],
        case_keys: list[str],
        suite_ids: list[UUID],
    ) -> WebResponse:
        rows = []
        for case, reason in blocked:
            rows.append(
                '<li><strong>' + escape_html(case.public_id or case.name) + '</strong> '
                + f'<a href="/test-cases/{case.id}">{escape_html(case.name)}</a>: {escape_html(reason)}</li>'
            )
        query = [("mode", "testcases")]
        query.extend(("case", key) for key in case_keys)
        query.extend(("suite", str(suite_id)) for suite_id in suite_ids)
        review_href = escape_html("/export?" + urlencode(query))
        content = (
            '<section class="panel error-state"><h1>Some selected items are blocked</h1>'
            '<p>Require-all export stopped before creating a file. Make every selected TestCase exportable or choose ready-only in the workspace.</p>'
            '<ul class="export-blocked-list">' + "".join(rows) + '</ul>'
            f'<a class="button" href="{review_href}">Review selection</a></section>'
        )
        return WebResponse.html(409, self._page("Export needs attention", content, current="Export"))

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
        suite_rows = []
        for suite in suites:
            members = self._test_suites.members(suite.id)
            automation_ready_count = sum(
                self._automation_lifecycle.status(case) == AutomationStatus.AUTOMATION_READY
                for case in members
            )
            description = (
                f'<div class="muted suite-description">{escape_html(suite.description)}</div>'
                if suite.description.strip() else ""
            )
            case_count_label = f'{len(members)} TestCase{"s" if len(members) != 1 else ""}'
            suite_rows.append(
                "<tr>"
                f'<td class="suite-list-name"><a href="/test-suites/{suite.id}">{escape_html(suite.name)}</a>'
                + description
                + '</td>'
                + f'<td><span class="suite-case-count">{case_count_label}</span></td>'
                + f'<td><span class="suite-automation-count">{automation_ready_count} of {len(members)} Automation ready</span></td>'
                + f'<td><time datetime="{escape_html(suite.updated_at.isoformat())}">{escape_html(format_timestamp(suite.updated_at))}</time></td>'
                + '<td><div class="suite-list-actions">'
                + f'<a class="button" href="/test-suites/{suite.id}">Open</a>'
                + f'<a class="button" href="/export?mode=suites&amp;suite={suite.id}">Export</a>'
                + '</div></td></tr>'
            )
        rows = "".join(suite_rows)
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
                '<div class="table-wrap"><table class="suite-list"><thead><tr><th scope="col">Test Suite</th>'
                '<th scope="col">TestCases</th><th scope="col">Automation ready</th><th scope="col">Updated</th>'
                '<th scope="col">Actions</th></tr></thead><tbody>'
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
                f'<span class="suite-member-position" aria-hidden="true">{index + 1}.</span>'
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
            '<div class="field"><label for="suite-member">Add TestCase</label><select id="suite-member" name="test_case_id" required>'
            '<option value="">Choose a TestCase</option>' + options
            + '</select></div><button class="button" type="submit">Add</button></form>'
            if available else '<p class="muted">All available TestCases are already in this suite.</p>'
        )
        content = (
            '<header class="page-heading"><p class="eyebrow">Test Suite</p>'
            f'<h1>{escape_html(suite.name)}</h1><p class="lead">{escape_html(suite.description)}</p></header>'
            '<section class="panel"><div class="section-heading"><div><h2>Export suite</h2>'
            '<p class="muted">Review member readiness and choose a format in the Export workspace.</p></div>'
            f'<a class="button primary" href="/export?mode=suites&amp;suite={suite_id}">Export suite &rarr;</a>'
            '</div></section>'
            '<section class="panel"><div class="section-heading"><div><h2>Execute suite</h2>'
            '<p class="muted">Run saved automation in member order, with pinned versions and optional retries.</p></div>'
            f'<a class="button primary" href="/test-suites/{suite_id}/run">Configure Suite Run</a>'
            '</div></section>'
            '<section class="panel"><h2>Edit suite</h2>'
            f'<form method="post" action="/test-suites/{suite_id}/update" class="suite-form" data-inline-validation novalidate>'
            f'<div class="field"><label for="suite-name">Name</label><input id="suite-name" name="name" maxlength="120" required value="{escape_html(suite.name)}"></div>'
            f'<div class="field"><label for="suite-description">Description</label><textarea id="suite-description" name="description" maxlength="1000" rows="3">{escape_html(suite.description)}</textarea></div>'
            '<button class="button primary" type="submit">Save changes</button></form></section>'
            '<section class="panel"><h2>Suite TestCases</h2>' + add_form
            + (f'<ol class="suite-members">{"".join(member_rows)}</ol>' if member_rows else self._empty_state("This suite is empty.", "Add saved TestCases above."))
            + '</section>'
        )
        return WebResponse.html(200, self._page(
            suite.name, content, current="Test Suites",
            breadcrumbs=[("Dashboard", "/"), ("Test Suites", "/test-suites")],
        ))

    def _suite_run_config_page(self, suite_id: UUID, error: str | None = None, status: int = 200) -> WebResponse:
        if self._suite_run_service is None:
            return self._not_found("Suite execution is unavailable.")
        preview = self._suite_run_service.preview(suite_id)
        if preview.suite is None:
            return self._not_found("Test Suite not found.")
        rows = []
        for item in preview.eligibility:
            pin_summary = ", ".join(
                f"{escape_html(pin.test_step_id)} v{pin.version_number}"
                for pin in item.pinned_plans
            ) or "No saved plan versions"
            reasons = "".join(f"<li>{escape_html(reason)}</li>" for reason in item.reasons)
            state = "Ready" if item.eligible else "Blocked"
            rows.append(
                f'<tr><td>{item.order_index + 1}</td>'
                f'<td><span class="id-code">{escape_html(item.test_case_public_id or str(item.test_case_id))}</span> '
                f'{escape_html(item.test_case_name)}</td>'
                f'<td>{badge(state, "success" if item.eligible else "error")}'
                + (f'<ul class="suite-run-blockers">{reasons}</ul>' if reasons else "")
                + f'</td><td><small>{pin_summary}</small></td></tr>'
            )
        eligibility = (
            '<div class="table-wrap"><table><thead><tr><th>Order</th><th>TestCase</th>'
            '<th>Eligibility</th><th>Pinned plan versions at start</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>'
            if rows else self._empty_state("No suite members.", "Add TestCases before starting a Suite Run.")
        )
        issue = error or preview.suite_error
        if issue is None and not preview.can_start:
            issue = "Resolve every blocked member before starting. Suite Runs never omit ineligible TestCases."
        error_html = f'<div class="error-state" role="alert">{escape_html(issue)}</div>' if issue else ""
        can_start = preview.can_start
        form = (
            '<form method="post" class="suite-run-config" data-inline-validation novalidate '
            f'action="/test-suites/{suite_id}/run">'
            '<div class="field"><label for="suite-workflow">Workflow</label>'
            '<select id="suite-workflow" name="workflow"><option value="REGRESSION" selected>Regression — saved automation</option></select></div>'
            '<div class="field"><label for="suite-execution-type">Execution type</label>'
            '<select id="suite-execution-type" name="execution_type"><option value="SEQUENTIAL" selected>Sequential</option></select></div>'
            '<div class="field"><label for="suite-ai-policy">AI policy</label><select id="suite-ai-policy" name="ai_policy">'
            '<option value="DISABLED" selected>Disabled</option><option value="ALLOWED">Allowed</option></select>'
            '<small>This Regression workflow uses saved plans and makes no LLM calls under either policy.</small></div>'
            '<div class="field"><label for="suite-retries">Retry count</label><select id="suite-retries" name="retry_count">'
            '<option value="0" selected>0 retries</option><option value="1">1 retry</option><option value="2">2 retries</option></select>'
            '<small>Each retry is a fresh TestCase Run against the same pinned plan versions.</small></div>'
            + self._evidence_controls(f"suite-{suite_id}")
            + self._cookie_consent_controls(f"suite-{suite_id}")
            + f'<button class="button" type="submit" formaction="/test-suites/{suite_id}/readiness">Check readiness</button>'
            + f'<button class="button primary" type="submit"{("" if can_start else " disabled aria-disabled=\"true\"")}>Start Suite Run</button>'
            + '</form>'
        )
        content = (
            '<header class="page-heading"><p class="eyebrow">Suite Run setup</p>'
            f'<h1>{escape_html(preview.suite.name)}</h1>'
            '<p class="lead">Review every ordered member and the exact saved plan versions that will be pinned.</p></header>'
            + error_html
            + '<section class="panel"><h2>Eligibility</h2>' + eligibility + '</section>'
            + '<section class="panel"><h2>Run configuration</h2>' + form + '</section>'
        )
        return WebResponse.html(status, self._page(
            "Configure Suite Run", content, current="Test Suites",
            breadcrumbs=[("Dashboard", "/"), ("Test Suites", "/test-suites"), (preview.suite.name, f"/test-suites/{suite_id}")],
        ))

    def _handle_system_health_refresh(self) -> WebResponse:
        if self._readiness_service is None:
            return WebResponse.html(503, self._page(
                "System Health unavailable",
                '<section class="panel error-state"><h1>System Health unavailable</h1>'
                '<p>Detailed readiness checks are not configured for this application instance.</p></section>',
            ))
        report = self._readiness_service.system_report(deep_browser=True, probe_evidence=True)
        return self._system_health_page(report, refreshed=True)

    def _handle_test_case_readiness(
        self, test_case_id_text: str, body: bytes | str | None
    ) -> WebResponse:
        test_case_id = _parse_uuid(test_case_id_text)
        test_case = (
            self._test_cases.get(test_case_id)
            if test_case_id is not None and self._test_cases is not None else None
        )
        if test_case is None:
            return self._not_found("TestCase not found.")
        if self._readiness_service is None:
            return self._run_error(test_case.id, "Detailed pre-run readiness is unavailable.", 503)
        form, error = _parse_form_body(body)
        if error:
            return self._run_error(test_case.id, error, 400)
        preferences = self._execution_preferences.get(test_case.id)
        try:
            workflow = WorkflowType(form.get("workflow", [""])[0])
            if workflow not in {WorkflowType.AUTOMATION, WorkflowType.VALIDATION, WorkflowType.REGRESSION}:
                raise ValueError
            evidence_policy = EvidencePolicy(
                mode=EvidenceMode(form.get("evidence_mode", [preferences.evidence_mode.value])[0]),
                screenshot_mode=ScreenshotMode(form.get("screenshot_mode", [preferences.screenshot_mode.value])[0]),
            )
            cookie_policy = CookieConsentPolicy(
                form.get("cookie_policy", [preferences.cookie_policy.value])[0]
            )
        except (ValueError, TypeError):
            return self._run_error(test_case.id, "Choose a valid workflow, evidence mode, and cookie policy.", 400)
        report = self._readiness_service.testcase_report(
            test_case, workflow,
            evidence_policy=evidence_policy,
            cookie_policy=cookie_policy,
        )
        fields = {
            "workflow": workflow.value,
            "evidence_mode": evidence_policy.mode.value,
            "screenshot_mode": evidence_policy.screenshot_mode.value,
            "cookie_policy": cookie_policy.value,
        }
        return self._readiness_result_page(
            report,
            title=f"Readiness — {test_case.name}",
            return_url=f"/test-cases/{test_case.id}",
            start_action=f"/test-cases/{test_case.id}/run",
            fields=fields,
        )

    def _handle_suite_readiness(
        self, suite_id_text: str, body: bytes | str | None
    ) -> WebResponse:
        suite_id = _parse_uuid(suite_id_text)
        if suite_id is None or self._suite_run_service is None:
            return self._not_found("Test Suite not found.")
        if self._readiness_service is None:
            return self._suite_run_config_page(suite_id, error="Detailed pre-run readiness is unavailable.", status=503)
        form, error = _parse_form_body(body)
        if error:
            return self._suite_run_config_page(suite_id, error=error, status=400)
        if form.get("workflow", [""])[0] != WorkflowType.REGRESSION.value:
            return self._suite_run_config_page(suite_id, error="Choose the saved-plan Regression workflow.", status=400)
        if form.get("execution_type", [""])[0] != "SEQUENTIAL":
            return self._suite_run_config_page(suite_id, error="Choose sequential execution.", status=400)
        try:
            evidence_policy = EvidencePolicy(
                mode=EvidenceMode(form.get("evidence_mode", [EvidenceMode.FAILURES_ONLY.value])[0]),
                screenshot_mode=ScreenshotMode(form.get("screenshot_mode", [ScreenshotMode.PAGE.value])[0]),
            )
            cookie_policy = CookieConsentPolicy(
                form.get("cookie_policy", [DEFAULT_COOKIE_CONSENT_POLICY.value])[0]
            )
            config = SuiteRunConfig(
                ai_policy=AIPolicy(form.get("ai_policy", [AIPolicy.DISABLED.value])[0]),
                retry_count=int(form.get("retry_count", ["0"])[0]),
                evidence_policy=evidence_policy,
                cookie_policy=cookie_policy,
            )
        except (ValueError, TypeError):
            return self._suite_run_config_page(suite_id, error="Choose a valid AI policy, evidence mode, cookie policy, and retry count from 0 to 2.", status=400)
        report = self._readiness_service.suite_report(
            lambda: self._suite_run_service.preview(suite_id),
            evidence_policy=evidence_policy,
            cookie_policy=cookie_policy,
        )
        return self._readiness_result_page(
            report,
            title="Suite readiness",
            return_url=f"/test-suites/{suite_id}/run",
            start_action=f"/test-suites/{suite_id}/run",
            fields={
                "workflow": WorkflowType.REGRESSION.value,
                "execution_type": "SEQUENTIAL",
                "ai_policy": config.ai_policy.value,
                "retry_count": str(config.retry_count),
                "evidence_mode": evidence_policy.mode.value,
                "screenshot_mode": evidence_policy.screenshot_mode.value,
                "cookie_policy": cookie_policy.value,
            },
        )

    def _readiness_result_page(
        self,
        report: ReadinessReport,
        *,
        title: str,
        return_url: str,
        start_action: str | None = None,
        fields: dict[str, str] | None = None,
    ) -> WebResponse:
        continuation = ""
        if report.can_continue and start_action and fields:
            hidden = "".join(
                f'<input type="hidden" name="{escape_html(key)}" value="{escape_html(value)}">'
                for key, value in fields.items()
            )
            continuation = (
                f'<form method="post" action="{escape_html(start_action)}">{hidden}'
                '<button class="button primary" type="submit">Continue to start</button></form>'
            )
        else:
            continuation = (
                f'<a class="button" href="{escape_html(return_url)}">Return to setup</a>'
            )
        content = (
            '<header class="page-heading"><p class="eyebrow">Pre-run check</p>'
            f'<h1>{escape_html(title)}</h1>'
            f'<p class="lead">Overall: {badge(report.status.value, _readiness_tone(report.status))}</p></header>'
            + _readiness_checks_html(report.checks)
            + '<div class="actions readiness-actions">' + continuation
            + f'<a class="button" href="{escape_html(return_url)}">Back</a></div>'
        )
        status = 409 if not report.can_continue else 200
        return WebResponse.html(status, self._page(
            title, content, current="Test Cases" if "/test-cases/" in return_url else "Test Suites",
        ))

    def _system_health_page(
        self, report: ReadinessReport, *, refreshed: bool = False
    ) -> WebResponse:
        notice = '<p class="muted">Checks refreshed locally; no provider connection test was made.</p>' if refreshed else ""
        content = (
            '<header class="page-heading"><p class="eyebrow">System</p><h1>System Health</h1>'
            f'<p class="lead">Overall readiness: {badge(report.status.value, _readiness_tone(report.status))}</p></header>'
            + notice
            + _readiness_checks_html(report.checks)
            + '<form method="post" action="/system/health/refresh">'
            '<button class="button primary" type="submit">Refresh checks</button></form>'
        )
        return WebResponse.html(200, self._page(
            "System Health", content, current="System Health",
            breadcrumbs=[("Dashboard", "/")],
        ))

    def _handle_suite_run_start(self, raw_suite_id: str, body: bytes | str | None) -> WebResponse:
        suite_id = _parse_uuid(raw_suite_id)
        if suite_id is None:
            return self._not_found("Test Suite not found.")
        if self._suite_run_service is None:
            return self._not_found("Suite execution is unavailable.")
        form, form_error = _parse_form_body(body)
        if form_error:
            return self._suite_run_config_page(suite_id, error=form_error, status=400)
        if form.get("workflow", [""])[0] != WorkflowType.REGRESSION.value:
            return self._suite_run_config_page(suite_id, error="Choose the saved-plan Regression workflow.", status=400)
        if form.get("execution_type", [""])[0] != "SEQUENTIAL":
            return self._suite_run_config_page(suite_id, error="Choose sequential execution.", status=400)
        try:
            ai_policy = AIPolicy(form.get("ai_policy", [AIPolicy.DISABLED.value])[0])
            retry_count = int(form.get("retry_count", ["0"])[0])
            evidence_policy = EvidencePolicy(
                mode=EvidenceMode(form.get("evidence_mode", [EvidenceMode.FAILURES_ONLY.value])[0]),
                screenshot_mode=ScreenshotMode(form.get("screenshot_mode", [ScreenshotMode.PAGE.value])[0]),
            )
            cookie_policy = CookieConsentPolicy(
                form.get("cookie_policy", [DEFAULT_COOKIE_CONSENT_POLICY.value])[0]
            )
            config = SuiteRunConfig(
                ai_policy=ai_policy,
                retry_count=retry_count,
                evidence_policy=evidence_policy,
                cookie_policy=cookie_policy,
            )
        except (ValueError, TypeError):
            return self._suite_run_config_page(suite_id, error="Choose a valid AI policy, cookie consent policy, evidence mode, screenshot scope, and retry count from 0 to 2.", status=400)
        if self._readiness_service is not None:
            report = self._readiness_service.suite_report(
                lambda: self._suite_run_service.preview(suite_id),
                evidence_policy=config.evidence_policy,
                cookie_policy=config.cookie_policy,
            )
            if not report.can_continue:
                return self._readiness_result_page(
                    report,
                    title="Suite readiness blocked",
                    return_url=f"/test-suites/{suite_id}/run",
                )
        result = self._suite_run_service.start(
            suite_id,
            ai_policy=config.ai_policy,
            retry_count=config.retry_count,
            evidence_policy=config.evidence_policy,
            cookie_policy=config.cookie_policy,
        )
        if result.run is None:
            message = result.suite_error or "Resolve every blocked member before starting this Suite Run."
            return self._suite_run_config_page(suite_id, error=message, status=409)
        return WebResponse.redirect(f"/suite-runs/{result.run.public_id}")

    def _suite_run_progress_json(self, public_id: str) -> WebResponse:
        run = self._suite_run_service.get(public_id) if self._suite_run_service else None
        if run is None:
            return WebResponse.json(404, json.dumps({"error": "Suite Run is not available."}))
        return WebResponse.json(200, json.dumps(_suite_run_public_dict(run), ensure_ascii=False, sort_keys=True))

    def _suite_run_page(self, public_id: str) -> WebResponse:
        run = self._suite_run_service.get(public_id) if self._suite_run_service else None
        if run is None:
            return self._not_found("Suite Run not found.")
        item_rows = []
        for item in run.items:
            attempt_rows = []
            for attempt in item.attempts:
                duration = escape_html(format_duration(attempt.duration_ms))
                run_link = (
                    f'<a href="/runs/{attempt.run_id}">'
                    f'{escape_html(attempt.run_public_id or "Open TestCase Run")}</a>'
                    if attempt.run_id else "No Run History record"
                )
                attempt_rows.append(
                    f'<li>Attempt {attempt.attempt_number} — {escape_html(result_label(suite_attempt_outcome(attempt)))}'
                    f' · {duration} → {run_link}'
                    '<details class="technical-details"><summary>Developer details</summary>'
                    f'<p>Stored outcome: {escape_html(attempt.outcome)} · Classifications: {escape_html(", ".join(attempt.failure_classifications) or "Unknown")}</p></details></li>'
                )
            attempts = "".join(attempt_rows)
            item_rows.append(
                f'<li class="suite-run-item" data-suite-item="{item.order_index}">'
                f'<div><strong>{escape_html(item.test_case_public_id or "")} — '
                f'{escape_html(item.test_case_name)}</strong> {badge(suite_item_label(item), "success" if suite_item_outcome(item) == "PASSED" else outcome_tone(suite_item_outcome(item)))}'
                f'<span class="muted">{escape_html(format_duration(item.duration_ms))}'
                + '</span></div>'
                + (f'<ul>{attempts}</ul>' if attempts else f'<p class="muted">{escape_html(result_label(suite_item_outcome(item)))}</p>')
                + '</li>'
            )
        report_url = f"/suite-runs/{escape_html(run.public_id or public_id)}/report.json"
        live = run.status in {SuiteRunStatus.QUEUED, SuiteRunStatus.RUNNING, SuiteRunStatus.CANCELLATION_REQUESTED}
        content = (
            '<header class="page-heading"><p class="eyebrow">Suite Run</p>'
            f'<h1>{escape_html(run.public_id or public_id)} — {escape_html(run.suite_name)}</h1>'
            f'<p class="lead">{badge(suite_status_label(run))} Workflow: Regression · Execution: Sequential · '
            f'AI policy: {escape_html(run.config.ai_policy.value)} · Retries: {run.config.retry_count}</p></header>'
            + f'<p class="muted">Cookie consent: {escape_html(cookie_consent_policy_label(run.config.cookie_policy))}</p>'
            + f'<p class="muted">Evidence: {escape_html(evidence_mode_label(run.config.evidence_policy.mode))} · '
            f'{escape_html(screenshot_mode_label(run.config.evidence_policy.screenshot_mode))}</p>'
            + self._stop_control(f"/suite-runs/{run.public_id or public_id}/stop", not live, run.status.value)
            + f'<section class="panel" data-suite-run-progress="/api/suite-runs/{escape_html(run.public_id or public_id)}">'
            '<h2>Progress</h2>'
            f'<p data-suite-run-summary>{_suite_counts_text(run)}</p>'
            + ('<p data-suite-run-live>Refreshing saved progress…</p>' if live else f'<p data-suite-run-live>{escape_html(suite_status_label(run))}.</p>')
            + '<ol data-suite-run-items>' + "".join(item_rows) + '</ol></section>'
            + f'<p><a class="button" href="{report_url}" download>Download JSON report</a> '
            + f'<a class="button" href="/test-suites/{run.suite_id}">Back to Test Suite</a></p>'
        )
        return WebResponse.html(200, self._page(
            run.public_id or "Suite Run", content, current="Test Suites",
            breadcrumbs=[("Dashboard", "/"), ("Test Suites", "/test-suites"), (run.suite_name, f"/test-suites/{run.suite_id}")],
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
            and (status == "ALL" or history_outcome(record) == ("PRODUCT_FAILURE" if status == "FAILED" else status))
            and (failure_type == "ALL" or record.outcome == failure_type)
        ]
        content = (
            '<header class="page-heading"><h1>Runs</h1>'
            f'<p class="lead">Showing {len(filtered)} of {len(records)} recent runs.</p></header>'
            '<section class="panel"><h2>Filter runs</h2>'
            f'<form class="filters" method="get" action="/runs">'
            f'{_select("workflow", "Workflow", workflow, ["ALL", *_sort_values(_WORKFLOWS)])}'
            f'{_select("status", "Result", status, ["ALL", *_sort_values(_STATUSES - {"FAILED"})])}'
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
            status_html = _outcome_badge(record)
            outcome_html = ""
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

    @staticmethod
    def _quality_feedback(test_case: TestCase) -> str:
        from qa_agent.test_case_quality import quality_issues
        items = []
        for issue in quality_issues(test_case):
            match = re.match(r"Step (\d+):", issue)
            step = next((step for step in test_case.steps if step.order + 1 == int(match[1])), None) if match else None
            url = f'/test-cases/{test_case.id}/edit' + (f'#structured-{step.id}-description' if step else '')
            items.append(f'<li><a href="{url}">{escape_html(issue)}</a></li>')
        return '<ul class="quality-feedback">' + ''.join(items) + '</ul>' if items else ''

    def _test_case_page(self, test_case_id: UUID) -> WebResponse:
        records = self._run_history.list_for_test_case(test_case_id, _PAGE_LIMIT)
        test_case = self._test_cases.get(test_case_id) if self._test_cases is not None else None
        if self._test_cases is not None and test_case is None:
            return self._not_found("TestCase not found")
        if self._test_cases is None and not records:
            return self._not_found("TestCase not found")
        latest = records[0] if records else None
        passed = sum(history_outcome(record) == "PASSED" for record in records)
        failed = sum(history_outcome(record) == "PRODUCT_FAILURE" for record in records)
        cards = (
            _summary_card(
                "Latest result",
                _outcome_badge(latest)
                if latest else badge("NOT RUN"),
                raw=True,
            )
            + _summary_card("Last run", format_timestamp(latest.started_at) if latest else "Never")
            + _summary_card("Total runs", str(len(records)))
            + _summary_card("Passed / product failures", f"{passed} / {failed}")
            + _summary_card("Technical / inconclusive", str(len(records) - passed - failed))
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
                f'<li id="test-step-{step.id}"><strong>Step {step.order + 1}'
                + (f': {escape_html(step.name)}' if step.name.strip().rstrip('.').casefold() not in {step.description.strip().rstrip('.').casefold(), step_display_name(step.description).casefold()} else '') + '</strong>'
                +
                f'<div>{escape_html(step.description)}</div>'
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
            '<div class="table-wrap"><table><thead><tr><th>Result</th><th>Workflow</th>'
            '<th>Last run</th><th>Duration</th><th>Run</th></tr></thead><tbody>'
            + history_rows
            + '</tbody></table></div>'
            if records else self._empty_state(
                "No runs yet.", "Generate and run automation to create the first plan versions and history."
            )
        )
        run_form = ""
        has_workflow_panel = False
        if (
            self._run_service is not None and test_case is not None
            and self._test_case_review.status(test_case.id) == TestCaseReviewStatus.APPROVED
        ):
            availability = self._workflow_availability(test_case_id)
            if availability is not None:
                run_form = self._workflow_panel(test_case_id, test_case, availability)
                has_workflow_panel = True
            else:
                # Compatibility for lightweight adapters that only implement run().
                run_form = self._run_form(test_case_id)
        if test_case is not None:
            run_form = self._preferences_panel(test_case_id) + run_form
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
        if (
            automation_status == AutomationStatus.NEEDS_VALIDATION
            and test_case is not None
            and self._plan_store is not None
            and any(
                not expected_result_coverage(
                    step,
                    self._plan_store.find(step.id),
                ).is_sufficient
                for step in test_case.steps
            )
        ):
            lifecycle_note = "Automation does not verify the TestStep expected result."
        lifecycle_panel = (
            '<section class="panel"><h2>Automation status</h2>'
            f'<p>{badge(lifecycle_label, "workflow")}</p><p class="muted">{escape_html(lifecycle_note)}</p></section>'
            if test_case is not None else ""
        )
        run_status_panel = (
            '<section class="panel"><h2>Run status</h2>'
            + (f'<p>{_outcome_badge(latest)}</p>'
               f'<p class="muted">{escape_html(latest.workflow_type.value.title())} · '
               f'{escape_html(format_timestamp(latest.started_at))}</p>'
               if latest is not None else '<p class="muted">Not run yet.</p>')
            + '</section>'
            if test_case is not None else ""
        )
        review_status = (
            self._test_case_review.status(test_case.id)
            if test_case is not None else TestCaseReviewStatus.APPROVED
        )
        review_panel = (
            '<section class="panel review-status-panel"><h2>Review status</h2>'
            + f'<p>{badge("Ready for review" if review_status == TestCaseReviewStatus.READY_FOR_REVIEW else "Approved", "workflow")}</p>'
            + ('<p>Please review the TestCase content below before approving it.</p>'
               if review_status == TestCaseReviewStatus.READY_FOR_REVIEW else
               '<p class="muted">TestCase content is approved.</p>')
            + '</section>'
            if test_case is not None else ""
        )
        review_action_panel = (
            f'<section class="panel review-action-panel"><h2>Review TestCase</h2>'
            '<p>Review the scenario, preconditions, steps, and expected results above.</p>'
            f'<form method="post" action="/test-cases/{test_case_id}/approve">'
            '<button class="button primary" type="submit">Approve TestCase</button></form></section>'
            if test_case is not None and review_status == TestCaseReviewStatus.READY_FOR_REVIEW else
            ('<section class="panel review-action-panel"><h2>Next step</h2>'
             '<p>TestCase approved. Generate automation when you are ready.</p></section>'
             if test_case is not None and not has_workflow_panel else "")
        )
        edit_links = (
            f'<div class="actions"><a class="button" href="/test-cases/{test_case_id}/edit">Edit TestCase</a>'
            + (f'<a class="button" href="/test-cases/{test_case_id}/plans">View TestPlan</a>' if not has_workflow_panel else '') + '</div>'
            if test_case is not None else ""
        )
        if (
            test_case is not None
            and not has_workflow_panel
            and review_status == TestCaseReviewStatus.APPROVED
            and self._plan_store is not None and any(
            self._plan_store.find(step.id) is not None for step in test_case.steps
            )
        ):
            edit_links = edit_links.replace(
                '</div>',
                f'<a class="button" href="/test-cases/{test_case_id}/automation/edit">Edit Automation</a></div>',
            )
        export_panel = ""
        if test_case is not None and self._testplan_exports is not None:
            export_panel = (
                '<section class="panel export-panel"><div class="section-heading"><div>'
                '<h2>Export</h2><p class="muted">Review automation readiness and choose a format in the Export workspace.</p>'
                '</div>'
                f'<a class="button primary" href="/export?mode=testcases&amp;case={quote(test_case.public_id or "")}">Export &rarr;</a>'
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
            + review_panel
            + (self._quality_feedback(test_case) if test_case is not None and review_status == TestCaseReviewStatus.READY_FOR_REVIEW else '')
            + lifecycle_panel
            + run_status_panel
            + '<section class="panel"><h2>Preconditions</h2>'
            + (f"<ul>{conditions}</ul>" if conditions else '<p class="muted">No preconditions recorded.</p>')
            + '</section><section class="panel"><h2>Steps</h2>'
            + (f'<ol class="testcase-steps">{steps}</ol>' if steps else '<p class="muted">No steps recorded.</p>')
            + '</section><section class="panel"><h2>Execution segments</h2>'
            + (f"<ul>{segments}</ul>" if segments else '<p class="muted">No segment data recorded.</p>')
            + '</section>'
            + review_action_panel
            + run_form
            + '<section class="panel"><h2>Run History</h2>' + history_table + '</section>'
            + export_panel + usage_panel
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
            '</select></div>'
            + '<button class="button primary" type="submit">Start run</button></form>'
            '</section>'
        )

    @staticmethod
    def _evidence_controls(prefix: str) -> str:
        mode_id = f"{prefix}-evidence-mode"
        scope_id = f"{prefix}-screenshot-mode"
        return (
            '<details class="run-evidence-options"><summary>Evidence settings</summary>'
            '<div class="run-evidence-fields">'
            f'<div class="field"><label for="{mode_id}">Evidence</label>'
            f'<select id="{mode_id}" name="evidence_mode">'
            f'<option value="{EvidenceMode.FAILURES_ONLY.value}" selected>Failures only</option>'
            f'<option value="{EvidenceMode.EVERY_VERIFICATION.value}">Every verification</option>'
            f'<option value="{EvidenceMode.EVERY_STEP.value}">Every step</option></select></div>'
            f'<div class="field"><label for="{scope_id}">Screenshot</label>'
            f'<select id="{scope_id}" name="screenshot_mode">'
            f'<option value="{ScreenshotMode.ELEMENT.value}">Element</option>'
            f'<option value="{ScreenshotMode.PAGE.value}" selected>Page</option>'
            f'<option value="{ScreenshotMode.ELEMENT_AND_PAGE.value}">Element + Page</option></select></div>'
            '</div><p class="muted">Screenshots may include sensitive values visible on the page. Review them before sharing.</p>'
            '</details>'
        )

    @staticmethod
    def _cookie_consent_controls(prefix: str) -> str:
        policy_id = f"{prefix}-cookie-policy"
        return (
            '<details class="run-cookie-options"><summary>Cookie consent</summary>'
            '<div class="field"><label for="' + escape_html(policy_id) + '">Policy</label>'
            '<select id="' + escape_html(policy_id) + '" name="cookie_policy">'
            f'<option value="{CookieConsentPolicy.AUTO_HANDLE.value}" selected>'
            f'{escape_html(cookie_consent_policy_label(CookieConsentPolicy.AUTO_HANDLE))}</option>'
            f'<option value="{CookieConsentPolicy.LEAVE_UNCHANGED.value}">'
            f'{escape_html(cookie_consent_policy_label(CookieConsentPolicy.LEAVE_UNCHANGED))}</option>'
            '</select></div><p class="muted">Choose Leave unchanged when this TestCase checks cookie behavior.</p>'
            '</details>'
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
        review_status = self._test_case_review.status(test_case.id)
        if review_status != TestCaseReviewStatus.APPROVED:
            return (
                '<section class="panel"><h2>Automation</h2>'
                '<p>Approve the TestCase before generating automation.</p></section>'
            )
        forms: list[str] = []
        review_record = self._test_case_review.record(test_case.id)
        legacy_case = review_record is None
        complete = (
            availability.total_step_count > 0
            and availability.usable_plan_count == availability.total_step_count
        )
        current_plan_approval = self._test_case_review.validation_approved_for(test_case)
        plan_by_step = {
            step_id: (version, version_id)
            for step_id, version, version_id in availability.plan_versions
        }
        version_rows = []
        for step in sorted(test_case.steps, key=lambda item: item.order):
            selected = plan_by_step.get(step.id)
            if selected is None:
                continue
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
        if complete:
            if legacy_case and availability.automation_available:
                forms.append(self._workflow_form(
                    test_case_id, WorkflowType.AUTOMATION, "Generate Automation", primary=True
                ))
            lifecycle_status = self._automation_lifecycle.status(test_case)
            if current_plan_approval:
                status_message = (
                    "The saved plans passed Validation."
                    if lifecycle_status == AutomationStatus.AUTOMATION_READY else
                    "Automation is approved for Validation. Run Validation to confirm it is ready."
                )
            else:
                status_message = "Automation is ready for review."
            links = (
                f'<div class="actions"><a class="button" href="/test-cases/{test_case_id}/plans">View TestPlan</a>'
                f'<a class="button" href="/test-cases/{test_case_id}/automation/edit">Edit Automation</a></div>'
            )
            if not current_plan_approval and all(expected_result_coverage(step, self._plan_store.find(step.id) if self._plan_store is not None else None).is_sufficient for step in test_case.steps):
                forms.append(
                    f'<form method="post" action="/test-cases/{test_case_id}/approve-validation">'
                    '<button class="button primary" type="submit">Approve for Validation</button></form>'
                )
            if current_plan_approval:
                if availability.validation_available:
                    forms.append(self._workflow_form(test_case_id, WorkflowType.VALIDATION, "Run Validation", primary=True))
                if availability.regression_available:
                    forms.append(self._workflow_form(test_case_id, WorkflowType.REGRESSION, "Run Regression"))
            review_copy = (
                '<p>Review saved plan versions, then approve this exact set for Validation. '
                'Automation stays in Needs validation until Validation passes.</p>'
                if not current_plan_approval else
                '<p>Validation is the gate to Automation Ready. The approved plan versions are checked again when the run starts.</p>'
            )
            reason = (
                f'<p class="muted">{escape_html(availability.reason)}</p>'
                if not availability.validation_available and availability.reason else ""
            )
            plan_review = (
                '<div class="automation-review-state"><h3>Automation review</h3>'
                f'<p>{escape_html(status_message)}</p>{review_copy}{links}'
                f'<ol class="compact-list">{"".join(version_rows)}</ol>{reason}'
                + ('<div class="actions">' + ''.join(forms) + '</div>' if forms else '')
                + '</div>'
            )
        else:
            status_message = (
                "Saved versions are from an earlier or incomplete generation. Generate updated automation."
                if self._automation_lifecycle.status(test_case) in {AutomationStatus.NEEDS_UPDATE, AutomationStatus.AUTOMATION_FAILED}
                else
                "Automation is incomplete. Review saved versions and retry generation."
                if availability.usable_plan_count > 0
                else "Automation has not been generated yet."
            )
            versions = (
                f'<ol class="compact-list">{"".join(version_rows)}</ol>'
                if version_rows else '<p class="muted">No executable plan versions have been generated.</p>'
            )
            links = (
                f'<div class="actions"><a class="button" href="/test-cases/{test_case_id}/plans">View TestPlan</a>'
                + (f'<a class="button" href="/test-cases/{test_case_id}/automation/edit">Edit Automation</a>'
                   if version_rows else '')
                + '</div>'
            )
            if availability.automation_available:
                forms.append(
                    f'<form method="post" action="/test-cases/{test_case_id}/run" data-run-form>'
                    '<input type="hidden" name="workflow" value="AUTOMATION">'
                    + '<button class="button primary" type="submit">Generate Automation</button></form>'
                )
            if legacy_case:
                if availability.validation_available:
                    forms.append(self._workflow_form(
                        test_case_id, WorkflowType.VALIDATION, "Run Validation"
                    ))
                if availability.regression_available:
                    forms.append(self._workflow_form(
                        test_case_id, WorkflowType.REGRESSION, "Run Regression"
                    ))
            run_note = (
                '<p class="muted">This creates and runs an initial automation attempt. '
                'Validation is still required before Automation Ready.</p>'
            )
            plan_review = (
                f'<p>{escape_html(status_message)}</p>'
                f'{versions}{links}{run_note}'
                + ('<div class="actions">' + ''.join(forms) + '</div>' if forms else '')
            )
        return (
            '<section class="panel"><h2>Automation</h2>'
            + f'<p>{escape_html(plan_status)}</p>'
            + plan_review
            + '</section>'
        )

    def _workflow_form(
        self, test_case_id: UUID, workflow: WorkflowType, label: str, *, primary: bool = False
    ) -> str:
        return (
            f'<form method="post" action="/test-cases/{test_case_id}/run" data-run-form>'
            f'<input type="hidden" name="workflow" value="{workflow.value}">'
            + f'<button class="button{" primary" if primary else ""}" type="submit">{label}</button></form>'
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
                provider.connection_provider_error_code,
                provider.connection_provider_error_field,
            )
            capability = _provider_diagnostic_html(
                "Authoring capability", provider.capability_status,
                provider.capability_category, provider.capability_latency_ms,
                provider.capability_http_status,
                provider.capability_retry_after_seconds,
                provider.capability_provider_error_code,
                provider.capability_provider_error_field,
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

    def _settings_tabs(self, active: str) -> str:
        providers_current = ' aria-current="page"' if active == "providers" else ""
        usage_current = ' aria-current="page"' if active == "usage" else ""
        reliability_current = ' aria-current="page"' if active == "reliability" else ""
        return (
            '<nav class="settings-tabs" aria-label="Settings sections">'
            f'<a href="/settings/providers"{providers_current}>AI Providers</a>'
            f'<a href="/settings/usage"{usage_current}>AI Usage</a>'
            + (f'<a href="/settings/reliability"{reliability_current}>Automation Reliability</a>' if self._reliability is not None else '')
            + '</nav>'
        )

    def _handle_reliability_settings(self, body) -> WebResponse:
        form, error = _parse_form_body(body, max_bytes=8192)
        fields = {"additional_retries", "provider_fallback", "automatic_plan_repair", "max_total_attempts"}
        if error or set(form) != fields or any(len(values) != 1 for values in form.values()):
            return WebResponse.html(400, self._reliability_page({}, error="Submit one allowed value for each reliability setting."))
        values = {key: items[0] for key, items in form.items()}
        if any(values[key] not in {"on", "off"} for key in fields - {"max_total_attempts"}) or values["max_total_attempts"] not in {"1", "2", "3"}:
            return WebResponse.html(400, self._reliability_page({}, error="Toggles must be ON or OFF; maximum total attempts must be 1, 2 or 3."))
        settings = ReliabilitySettings(
            **{key: values[key] == "on" for key in fields - {"max_total_attempts"}},
            max_total_attempts=int(values["max_total_attempts"]),
        )
        try:
            self._reliability.repository.save_settings(settings)
        except Exception:
            return WebResponse.html(503, self._page("Automation Reliability", '<p>Reliability settings could not be saved. Review local storage in System Health.</p>'))
        return WebResponse.redirect("/settings/reliability?saved=1")

    def _reliability_page(self, query: dict, *, error: str | None = None) -> str:
        settings = self._reliability.repository.settings()
        stats = self._reliability.statistics()
        esc = escape_html
        controls = []
        for key, title, description in (
            ("additional_retries", "Additional retries", "Retry transient provider timeouts, rate limits and temporary connection failures within the shared budget."),
            ("provider_fallback", "Provider fallback", "Use the next eligible configured provider in your saved order. Each fallback consumes an attempt."),
            ("automatic_plan_repair", "Automatic plan repair", "Allow one targeted correction of a structurally invalid unapproved candidate or explicit missing verification coverage. Uncertain assertions and locators require human review."),
        ):
            enabled = getattr(settings, key)
            controls.append(f'<label>{title}<select name="{key}"><option value="off"' + ('' if enabled else ' selected') + '>OFF</option><option value="on"' + (' selected' if enabled else '') + f'>ON</option></select><span class="muted">{description}</span></label>')
        controls.append('<label>Maximum total attempts<select name="max_total_attempts">' + ''.join(f'<option value="{number}"' + (' selected' if settings.max_total_attempts == number else '') + f'>{number}</option>' for number in (1, 2, 3)) + '</select><span class="muted">Per atomic TestStep generation, including the initial request, retries, fallback and repair together.</span></label>')
        rate = lambda value: f'{value * 100:.1f}%' if value is not None else 'Unknown'
        cards = ''.join(_summary_card(label, value) for label, value in (
            ("Generation operations", str(stats["generation_operations"])),
            ("First-attempt success", rate(stats["first_attempt_success_rate"])),
            ("Recovery success", rate(stats["recovery_success_rate"])),
            ("Recovery attempts", str(stats["recovery_attempts"])),
            ("Provider fallback events", str(stats["fallback_count"])),
            ("Average attempts", f'{stats["average_attempts"]:.2f}' if stats["average_attempts"] is not None else 'Unknown'),
            ("Average generation duration", format_duration(stats["average_generation_duration_ms"]) if stats["average_generation_duration_ms"] is not None else 'Unknown'),
        ))
        provider_rows = ''.join(f'<tr><td>{esc(row["provider"] or "Unknown")}</td><td>{esc(row["model"] or "Unknown")}</td><td>{row["count"]}</td></tr>' for row in stats["provider_model_attempts"])
        gate_rows = ''.join(f'<li>{esc(category)}: {count}</li>' for category, count in sorted(stats["quality_gate_rejections"].items()))
        operations = self._reliability.repository.list_records()
        rows = ''.join(
            f'<tr><td><a href="/settings/reliability/{record.id}">{esc(str(record.id))}</a></td>'
            + f'<td>{esc(format_timestamp(record.started_at))}</td><td>{esc(record.outcome.replace("_", " ").title())}</td><td>{len(record.attempts)}</td></tr>'
            for record in operations[:100]
        )
        usage = f'<p>Input tokens: {stats["input_tokens"] if stats["input_tokens"] is not None else "Unknown"} · Output tokens: {stats["output_tokens"] if stats["output_tokens"] is not None else "Unknown"}. Complete token metadata: {stats["usage_known_attempts"]} of {stats["total_attempts"]} attempts.</p>'
        usage += f'<p>Estimated cost: {"$" + format(stats["estimated_cost_usd"], ".6f") if stats["estimated_cost_usd"] is not None else "Unknown"}. Verified pricing coverage: {stats["cost_known_attempts"]} of {stats["total_attempts"]} attempts. Partial sums cover recorded values only and are not provider billing.</p>'
        content = (
            '<header class="page-heading"><h1>Automation Reliability</h1><p class="lead">Safe recovery for automation generation.</p></header>'
            + self._settings_tabs("reliability")
            + (f'<p class="notice danger">{esc(error)}</p>' if error else '<p class="notice">Settings saved. New operations use these values.</p>' if query.get("saved") == ["1"] else '')
            + '<section class="panel"><h2>Current effective settings</h2><form method="post" action="/settings/reliability">' + ''.join(controls) + '<button class="button primary" type="submit">Save reliability settings</button></form>'
            + '<p class="muted">Quality gates always run. Recovery never changes expected results, repairs confirmed product failures, or approves automation. Repaired candidates require human review and Approve for Validation before execution. Each generation has a 60-second overall limit.</p></section>'
            + '<section class="panel"><h2>Generation statistics</h2><div class="summary-grid">' + cards + '</div>'
            + '<p>Success here means a candidate passed quality gates. It does not mean human approval, Browser Validation PASS, or product correctness. Rates use completed recorded operations; recovery success uses completed operations with more than one attempt.</p>'
            + usage + '<h3>Quality Gate rejections</h3><ul>' + (gate_rows or '<li>No recorded Quality Gate rejections.</li>') + '</ul>'
            + '<h3>Provider and model attempts</h3><div class="table-wrap"><table><thead><tr><th>Provider</th><th>Model</th><th>Attempts</th></tr></thead><tbody>' + provider_rows + '</tbody></table></div></section>'
            + '<section class="panel"><h2>Generation operations</h2><p class="muted">Showing up to 100 most recent operations. Aggregate statistics cover all recorded operations.</p><div class="table-wrap"><table><thead><tr><th>Operation</th><th>Started</th><th>Quality result</th><th>Attempts</th></tr></thead><tbody>' + rows + '</tbody></table></div></section>'
        )
        return self._page("Automation Reliability", content, current="Settings", breadcrumbs=[("Dashboard", "/")])

    def _reliability_operation_page(self, operation_id) -> WebResponse:
        record = self._reliability.repository.get(operation_id)
        if record is None:
            return self._not_found("Generation operation not found")
        esc = escape_html
        rows = []
        for attempt in record.attempts:
            rows.append(
                f'<li><h3>Attempt {attempt.index} — {esc(attempt.action.replace("_", " ").title())}</h3>'
                + f'<p>{esc(attempt.status.replace("_", " ").title())} · {esc(attempt.reason)}</p>'
                + (f'<p>Provider: {esc(attempt.provider)}</p>' if attempt.provider else '')
                + (f'<p>Model: {esc(attempt.model)}</p>' if attempt.model else '')
                + (f'<p>Provider request duration: {esc(format_duration(attempt.duration_ms))}</p>' if attempt.duration_ms is not None else '')
                + '<details class="technical-details"><summary>Developer details</summary><pre>' + esc(attempt.model_dump_json(indent=2)) + '</pre></details></li>'
            )
        decisions = ''.join(f'<li>{esc(format_timestamp(item.timestamp))} · {esc(item.action)} · {esc(item.reason)}</li>' for item in record.decisions)
        actions = '<a class="button" href="/settings/reliability">Back to Automation Reliability</a>'
        case = self._test_cases.get(record.test_case_id) if self._test_cases is not None and record.test_case_id else None
        if case is not None and any(step.id == record.test_step_id for step in case.steps):
            actions += f'<a class="button" href="/test-cases/{record.test_case_id}">Review TestCase and automation</a>'
            version = self._saved_version(record.test_step_id, record.candidate_version_id)
            if version:
                actions += f'<a class="button" href="/test-cases/{record.test_case_id}/automation/steps/{record.test_step_id}/versions/{version.id}">View saved candidate v{version.version}</a>'
        if record.outcome == "RUNNING":
            actions += f'<form method="post" action="/settings/reliability/{record.id}/cancel"><input type="hidden" name="_csrf" value="{self._csrf_token}"><button class="button" type="submit">Cancel generation</button></form>'
        content = (
            '<header class="page-heading"><h1>Automation generation operation</h1>' + f'<p>{esc(record.id)}</p><p>{esc(record.outcome.replace("_", " ").title())}</p></header>'
            + '<p>Generation quality is separate from human approval and Browser Validation. No product PASS is inferred.</p>'
            + '<div class="actions">' + actions + '</div><section class="panel"><h2>Effective settings for this operation</h2><pre>' + esc(record.settings.model_dump_json(indent=2)) + '</pre></section>'
            + '<section class="panel"><h2>Attempts</h2><ol>' + ''.join(rows) + '</ol></section>'
            + '<section class="panel"><h2>Recovery decisions</h2><ol>' + decisions + '</ol></section>'
        )
        return WebResponse.html(200, self._page("Automation generation operation", content, current="Settings"))

    def _page(
        self,
        title: str,
        body: str,
        *,
        current: str | None = None,
        breadcrumbs: list[tuple[str, str]] | None = None,
    ) -> str:
        nav_items = []
        links = [("Dashboard", "/"), ("Test Cases", "/test-cases"), ("Export", "/export"), ("Drafts", "/drafts"), ("Runs", "/runs"), ("System Health", "/system/health")]
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
            + f'<meta name="qa-csrf-token" content="{escape_html(self._csrf_token)}">'
            + '<script src="/assets/ui.js" defer></script>'
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


def _readiness_tone(status: ReadinessStatus) -> str:
    return {
        ReadinessStatus.READY: "success",
        ReadinessStatus.WARNING: "warning",
        ReadinessStatus.BLOCKED: "error",
        ReadinessStatus.NOT_APPLICABLE: "workflow",
    }[status]


def _readiness_checks_html(checks: tuple[ReadinessCheck, ...]) -> str:
    rows = []
    for check in checks:
        elapsed = (
            f'<span class="muted">{check.elapsed_ms} ms</span>'
            if check.elapsed_ms is not None else ""
        )
        rows.append(
            f'<li class="readiness-check" data-readiness-check="{escape_html(check.id)}">'
            '<div class="readiness-check-heading">'
            f'<strong>{escape_html(check.name)}</strong> '
            f'{badge(check.status.value, _readiness_tone(check.status))}{elapsed}</div>'
            f'<p>{escape_html(check.safe_message)}</p>'
            + (f'<p class="muted">Next: {escape_html(check.remediation)}</p>' if check.remediation else "")
            + '</li>'
        )
    return '<ul class="readiness-list">' + "".join(rows) + '</ul>'


def create_application(
    database_path: str | Path | None = None,
    evidence_directory: str | Path | None = None,
) -> LocalWebApplication:
    storage = create_sqlite_storage(database_path)
    automation_lifecycle = AutomationLifecycleService(
        storage.automation_lifecycle_repository, storage.plan_store
    )
    test_case_review = TestCaseReviewService(
        storage.test_case_review_repository, storage.plan_store
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
    reliability = AutomationReliabilitySupervisor(storage.reliability_repository)
    reliability.reconcile_interrupted()
    automation_workflow = AutomationWorkflow(QATestPipeline(
        decomposer=TestCaseDecomposer(),
        plan_generator=LLMTestPlanGenerator(automation_router, reliability),
        runner=BrowserRunner(evidence_root, headless=True),
        plan_store=storage.plan_store,
        execution_repository=storage.execution_repository,
        run_history=storage.run_history,
        candidate_review_approved=lambda case: test_case_review.record(case.id) is not None and test_case_review.validation_approved_for(case),
    ))
    run_service = TestCaseExecutionService(
        storage.test_case_repository,
        storage.plan_store,
        storage.execution_repository,
        storage.run_history,
        evidence_directory=evidence_root,
        automation_workflow=automation_workflow,
        automation_lifecycle=automation_lifecycle,
        test_case_review=test_case_review,
        candidate_review_required=lambda version_id: reliability.requires_review(version_id) is not None,
    )
    test_suites = TestSuiteService(
        SQLiteTestSuiteRepository(storage.database_path),
        storage.test_case_repository,
    )
    suite_run_service = SuiteRunService(
        test_suites,
        storage.test_case_repository,
        run_service,
        storage.run_history,
        SQLiteSuiteRunRepository(storage.database_path),
    )
    return LocalWebApplication(
        storage.run_history,
        evidence_root=evidence_root,
        database_path=storage.database_path,
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
        test_suites=test_suites,
        suite_run_service=suite_run_service,
        test_case_review=test_case_review,
        reliability=reliability,
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
                and request_path[3] in {"save", "regenerate"}
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
            self._send(application.handle("POST", self.path, body, headers=dict(self.headers)))

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
<body><main>
<h1>Registration demo</h1>
<p>Local-only page for exercising common browser interactions.</p>
<nav aria-label="Demo navigation">
  <a id="help-link" href="/demo-target/registration/help">Help</a>
  <a id="details-link" href="#interaction-controls">Interaction controls</a>
</nav>
<p id="page-status" role="status">Ready</p>
<button id="change-status" type="button">Change page state</button>
<button id="reset-demo" type="button">Reset demo</button>
<form id="registration-form" novalidate>
  <h2 id="registration-fields">Registration</h2>
  <label>Email <input id="email" name="email" type="email" autocomplete="off"></label>
  <label>Password <input id="password" name="password" type="password" autocomplete="new-password"></label>
  <label><input id="terms" name="terms" type="checkbox"> Accept terms</label>
  <fieldset><legend>Plan</legend>
    <label><input id="plan-basic" name="plan" type="radio" value="basic"> Basic</label>
    <label><input id="plan-pro" name="plan" type="radio" value="pro"> Pro</label>
  </fieldset>
  <label>Region <select id="region" name="region">
    <option value="">Choose a region</option>
    <option value="north">North</option>
    <option value="south">South</option>
  </select></label>
  <p id="form-error" role="alert" hidden></p>
  <p id="registration-success" role="status" hidden>Registration submitted.</p>
  <button id="create-account" type="submit">Create account</button>
  <button id="reset-form" type="reset">Reset form</button>
</form>
<section id="interaction-controls">
  <h2>Dialog controls</h2>
  <button id="open-details" type="button">Open details</button>
  <dialog id="details-dialog" aria-label="Account details">
    <p>Review account details.</p>
    <button id="confirm-details" type="button">Confirm details</button>
    <button id="close-details" type="button">Close dialog</button>
  </dialog>
  <p id="dialog-result" role="status" hidden>Details confirmed.</p>
</section>
<section id="cookie-consent" role="dialog" aria-label="Cookie preferences">
  <p>This local demo uses cookies for preferences.</p>
  <button id="accept-cookies" type="button">Accept all cookies</button>
</section>
</main>
<script src="/assets/demo-registration.js" defer></script></body></html>"""


def _local_demo_javascript() -> str:
    """Same-origin script served under the application's unchanged CSP."""
    return """const form = document.querySelector('#registration-form');
const error = document.querySelector('#form-error');
const success = document.querySelector('#registration-success');
form.addEventListener('submit', event => {
  event.preventDefault(); error.hidden = true; success.hidden = true;
  const email = document.querySelector('#email');
  if (!email.value.includes('@')) {
    error.textContent = 'Enter a valid email address.';
    error.hidden = false; email.setAttribute('aria-invalid', 'true'); return;
  }
  email.removeAttribute('aria-invalid'); success.hidden = false;
});
form.addEventListener('reset', () => {
  error.hidden = true; success.hidden = true;
  document.querySelector('#email').removeAttribute('aria-invalid');
});
document.querySelector('#change-status').addEventListener('click', () => {
  document.querySelector('#page-status').textContent = 'Page state changed.';
});
document.querySelector('#reset-demo').addEventListener('click', () => window.location.reload());
document.querySelector('#open-details').addEventListener('click', () => {
  document.querySelector('#details-dialog').showModal();
});
document.querySelector('#confirm-details').addEventListener('click', () => {
  document.querySelector('#details-dialog').close();
  document.querySelector('#dialog-result').hidden = false;
});
document.querySelector('#close-details').addEventListener('click', () => {
  document.querySelector('#details-dialog').close();
});
document.querySelector('#accept-cookies').addEventListener('click', () => {
  document.querySelector('#cookie-consent').hidden = true;
});
"""


def _local_demo_help_page() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Registration help</title></head>
<body><main><h1>Registration help</h1><p>Use a valid email address to continue.</p>
<a href="/demo-target/registration">Back to registration</a></main></body></html>"""


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None


def _suite_counts_text(run: SuiteRun) -> str:
    counts = run.outcome_counts
    return (
        f"{counts['passed']} passed · {counts['product_failures']} product failures · "
        f"{counts['automation_errors']} automation errors · {counts['generation_errors']} generation errors · "
        f"{counts['infrastructure_errors']} infrastructure errors · {counts['blocked']} blocked · "
        f"{counts['inconclusive']} inconclusive · {counts['pending']} pending · "
        f"{run.flaky_count} passed after retry · {len(run.items)} total"
    )


def _suite_run_public_dict(run: SuiteRun) -> dict:
    return {
        "public_id": run.public_id,
        "status": run.status.value,
        "display_status": suite_status_label(run),
        "summary_text": _suite_counts_text(run),
        "workflow_type": run.config.workflow_type.value,
        "execution_type": run.config.execution_type,
        "ai_policy": run.config.ai_policy.value,
        "retry_count": run.config.retry_count,
        "cookie_policy": run.config.cookie_policy.value,
        "evidence_policy": {
            "mode": run.config.evidence_policy.mode.value,
            "screenshot_mode": run.config.evidence_policy.screenshot_mode.value,
        },
        "counts": {
            **run.outcome_counts,
            "failed": run.failed_count,
            "flaky": run.flaky_count,
            "total": len(run.items),
        },
        "items": [
            {
                "order_index": item.order_index,
                "test_case_id": str(item.test_case_id),
                "test_case_public_id": item.test_case_public_id,
                "test_case_name": item.test_case_name,
                "status": item.status.value,
                "display_status": suite_item_label(item),
                "classification": item.classification,
                "duration_ms": item.duration_ms,
                "error_category": item.error_category,
                "attempts": [
                    {
                        "attempt_number": attempt.attempt_number,
                        "run_id": str(attempt.run_id) if attempt.run_id else None,
                        "run_public_id": attempt.run_public_id,
                        "run_status": attempt.run_status,
                        "display_status": result_label(suite_attempt_outcome(attempt)),
                        "outcome": attempt.outcome,
                        "duration_ms": attempt.duration_ms,
                        "failure_classifications": attempt.failure_classifications,
                        "error_category": attempt.error_category,
                    }
                    for attempt in item.attempts
                ],
            }
            for item in run.items
        ],
    }


def _url_contains_credentials(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        if parsed.username or parsed.password:
            return True
        secret_query_keys = {
            "api_key", "api-key", "apikey", "access_token", "access-token",
            "token", "password", "secret", "authorization", "credential",
        }
        return any(
            key.casefold() in secret_query_keys
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        )
    except ValueError:
        return False


def _safe_automation_url_display(value: str) -> str:
    if _url_contains_credentials(value):
        return "[credential-bearing URL redacted]"
    return redact_secrets(value)


def _safe_parameter_json(parameters: dict[str, object]) -> str:
    safe_values = dict(parameters)
    if isinstance(safe_values.get("url"), str):
        safe_values["url"] = _safe_automation_url_display(safe_values["url"])
    return redact_secrets(json.dumps(
        safe_values, ensure_ascii=False, sort_keys=True, indent=2
    ))


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


def _automation_submitted_actions(values: dict[str, str]) -> list[dict[str, object]]:
    actions: dict[int, dict[str, object]] = {}
    for name, value in values.items():
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "action" or not parts[1].isdigit():
            continue
        index = int(parts[1])
        action = actions.setdefault(index, {"type": "", "parameters": {}})
        if parts[2] == "type":
            action["type"] = value
        elif len(parts) == 4 and parts[2] == "param":
            parameters = action["parameters"]
            if isinstance(parameters, dict):
                parameters[parts[3]] = value
    return [actions[index] for index in sorted(actions)]


def _automation_action_card(
    index: int,
    action_type: str,
    parameters: dict[str, object],
    field_errors: dict[str, str],
) -> str:
    action_options = ['<option value="">Choose action type</option>']
    for value, label in ACTION_LABELS.items():
        selected = ' selected' if value == action_type else ''
        action_options.append(
            f'<option value="{value}"{selected}>{escape_html(label)}</option>'
        )
    type_error = field_errors.get(f"action.{index}.type", "")
    type_error_html = (
        f'<span class="field-error" role="alert">{escape_html(type_error)}</span>'
        if type_error else ""
    )
    rendered_fields: list[str] = []
    supported = ACTION_EDITOR_FIELDS.get(action_type, ())
    required = QATestStep.ACTION_PARAMETER_FIELDS.get(action_type, ())
    for field in supported:
        field_name = f"action.{index}.param.{field}"
        value = parameters.get(field, "")
        error = field_errors.get(field_name, "")
        if field == "url" and isinstance(value, str) and (
            _url_contains_credentials(value) or redact_secrets(value) != value
        ):
            value = ""
            error = error or "This saved URL contains credentials. Replace it before saving."
        rendered_fields.append(
            '<label class="field"><span>' + escape_html(FIELD_LABELS.get(field, field)) + '</span>'
            + f'<input name="{field_name}" value="{escape_html(str(value))}"'
            + (' required' if field in required else '')
            + f' data-action-param="{field}" autocomplete="off"></label>'
            + (f'<span class="field-error" role="alert">{escape_html(error)}</span>' if error else '')
        )
    unsupported_errors = ''.join(
        f'<span class="field-error" role="alert">{escape_html(message)}</span>'
        for name, message in field_errors.items()
        if name.startswith(f"action.{index}.param.")
        and name.rsplit(".", 1)[-1] not in supported
    )
    return (
        f'<fieldset class="automation-action" data-action-card><legend>Action {index + 1}</legend>'
        + '<div class="automation-action-controls">'
        + '<button class="button" type="button" data-action-move="up" aria-label="Move action up">↑</button>'
        + '<button class="button" type="button" data-action-move="down" aria-label="Move action down">↓</button>'
        + '<button class="button" type="button" data-action-duplicate>Duplicate</button>'
        + '<button class="button" type="button" data-action-remove>Delete</button></div>'
        + f'<label class="field"><span>Action type</span><select name="action.{index}.type" data-action-type>'
        + ''.join(action_options) + '</select></label>' + type_error_html
        + '<div class="automation-action-fields" data-action-fields>'
        + ''.join(rendered_fields) + unsupported_errors + '</div></fieldset>'
    )


def _automation_history_html(test_case, step_id, current_version, versions) -> str:
    if not versions:
        return '<section class="automation-history"><h4>Version history</h4><p class="muted">No saved versions yet.</p></section>'
    current_id = current_version.id if current_version is not None else None
    rows = []
    for version in versions:
        current = version.id == current_id
        label = (
            f'v{version.version} · {_plan_origin_label(version.origin)}'
            + (' · current' if current else '')
            + f' · {format_timestamp(version.created_at)}'
        )
        href = f'/test-cases/{test_case.id}/automation/steps/{step_id}/versions/{version.id}'
        rows.append(f'<li><a href="{href}">{escape_html(label)}</a></li>')
    return '<section class="automation-history"><h4>Version history</h4><ol>' + ''.join(rows) + '</ol></section>'


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
            '<p class="suite-export-step"><strong>'
            f'Step {blocker.step_order}'
            + (f' — {escape_html(blocker.step_name)}' if blocker.step_name else "")
            + '</strong></p>'
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
    button_class = "button danger-button" if operation == "remove" else "button"
    return (
        f'<form method="post" action="/test-suites/{suite_id}/members/{operation}">'
        f'<input type="hidden" name="test_case_id" value="{case_id}">'
        f'<button class="{button_class}" type="submit" aria-label="{escape_html(accessible_label)}"'
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


def _expected_result_coverage_html(coverage) -> str:
    labels = {
        "NO_VERIFICATION_REQUIRED": "Not required",
        "COVERED": "Covered",
        "PARTIALLY_COVERED": "Partially covered",
        "NOT_COVERED": "Missing verification",
        "UNKNOWN": "Unknown",
    }
    value = coverage.status.value
    label = labels.get(value, "Unknown")
    tone = "success" if coverage.is_sufficient else "warning"
    message = (
        ""
        if coverage.is_sufficient
        else f'<p class="muted coverage-warning">{escape_html(coverage.safe_message)}</p>'
    )
    return (
        f'<p class="coverage-status">Expected result coverage: '
        f'<strong class="{tone}">{escape_html(label)}</strong></p>{message}'
    )


def _plan_origin_label(origin: PlanVersionOrigin | None) -> str:
    return {
        PlanVersionOrigin.AI_GENERATED: "Generated by AI",
        PlanVersionOrigin.HUMAN_EDITED: "Edited locally",
        PlanVersionOrigin.REGENERATED: "Regenerated by AI",
        PlanVersionOrigin.REPAIRED: "Updated automatically",
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
    shown = result_badge(history_outcome(record))
    return f'<div class="row-sub-badge">{shown}</div>' if stacked else shown


def _authoring_error_label(category: str | None) -> str:
    return {
        "AI_RATE_LIMIT": "AI RATE LIMIT",
        "AI_TIMEOUT": "AI TIMEOUT",
        "AI_PROVIDER_ERROR": "AI PROVIDER ERROR",
        "AI_GENERATION_ERROR": "AI GENERATION ERROR",
        "AI_OUTPUT_VALIDATION_ERROR": "AI GENERATION ERROR",
        "INVALID_AUTHORING_INPUT": "INVALID INPUT",
        "CANCELLED": "Stopped by user",
        "AUTHORING_EXECUTION_ERROR": "AUTHORING ERROR",
    }.get(category or "", "AI GENERATION ERROR")


def _provider_diagnostic_html(
    label: str,
    status: str | None,
    category: str | None,
    latency_ms: int | None,
    http_status: int | None,
    retry_after_seconds: int | None,
    provider_error_code: str | None = None,
    provider_error_field: str | None = None,
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
    if provider_error_code:
        value += f" · Provider code {provider_error_code}"
    if provider_error_field:
        value += f" · Field {provider_error_field}"
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
    scenario_error = scenario_input_error(scenario)
    if scenario_error:
        errors["scenario"] = scenario_error
    elif len(scenario) > 6000:
        errors["scenario"] = "Keep the scenario under 6,000 characters."
    if require_name and not name.strip():
        errors["name"] = "Enter a TestCase name."
    elif len(name.strip()) > 200:
        errors["name"] = "Keep Summary under 200 characters."
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
  document.querySelectorAll('[data-structured-editor]').forEach((form) => {
    const payload = form.querySelector('[data-step-payload]');
    const root = form.querySelector('[data-structured-steps]');
    const segmentUrls = JSON.parse(root.dataset.segmentUrls || '[]');
    let segments, changed = false, saving = false;
    try {
      segments = JSON.parse(payload.value);
      if (!Array.isArray(segments) || !segments.every(segment => segment && typeof segment.id === 'string' && Array.isArray(segment.steps) && segment.steps.every(step => step && ['id', 'description', 'expected'].every(key => typeof step[key] === 'string')))) throw new Error('Invalid list');
    } catch (_) {
      root.textContent = 'The submitted step list cannot be displayed. Your submitted data is retained below. Reload the saved definition before editing.';
      const raw = document.createElement('pre'); raw.textContent = payload.value; root.append(raw);
      form.querySelector('button[type="submit"]').disabled = true;
      return;
    }
    const sync = () => { payload.value = JSON.stringify(segments); changed = true; };
    const node = (tag, text, className) => {
      const element = document.createElement(tag);
      if (text) element.textContent = text;
      if (className) element.className = className;
      return element;
    };
    const render = (focusId) => {
      root.replaceChildren();
      let number = 0;
      segments.forEach((segment, segmentIndex) => {
        const group = node('section', '', 'subpanel');
        group.append(node('h3', `Segment ${segmentIndex + 1}`));
        group.append(node('p', segmentUrls[segmentIndex] || 'No URL configured', 'muted'));
        const list = node('div');
        segment.steps.forEach((step, index) => {
          number += 1;
          const card = node('fieldset', '', 'structured-step');
          card.dataset.stepId = step.id;
          card.append(node('legend', `Step ${number}`));
          ['description', 'expected'].forEach((key) => {
            const field = node('div', '', 'field');
            const id = `structured-${step.id}-${key}`;
            const label = node('label', key === 'description' ? 'Description' : 'Expected Result');
            label.htmlFor = id;
            const input = node('textarea');
            input.id = id; input.rows = key === 'description' ? 4 : 3;
            input.maxLength = 6000; input.value = step[key];
            input.addEventListener('input', () => { step[key] = input.value; sync(); });
            field.append(label, input); card.append(field);
          });
          const actions = node('div', '', 'actions step-actions');
          const button = (label, act, disabled = false, destructive = false) => {
            const control = node('button', label, destructive ? 'button danger-button' : 'button');
            control.type = 'button'; control.disabled = disabled;
            control.setAttribute('aria-label', `${label} ${label.startsWith('Insert Step') ? '' : 'Step '}${number}`);
            control.addEventListener('click', () => { if (act() === false) return; sync(); render(step.id); });
            actions.append(control);
          };
          const insert = (offset) => {
            const added = {id: `new:${crypto.randomUUID()}`, description: '', expected: ''};
            segment.steps.splice(index + offset, 0, added);
            step = added;
          };
          button('Insert Step before', () => insert(0));
          button('Insert Step after', () => insert(1));
          button('Move up', () => { [segment.steps[index - 1], segment.steps[index]] = [step, segment.steps[index - 1]]; }, index === 0);
          button('Move down', () => { [segment.steps[index + 1], segment.steps[index]] = [step, segment.steps[index + 1]]; }, index === segment.steps.length - 1);
          button('Delete', () => {
            if (!window.confirm('Delete this step? Changes are applied only when you save the TestCase.')) return false;
            segment.steps.splice(index, 1);
            step = segment.steps[Math.min(index, segment.steps.length - 1)];
          }, segment.steps.length === 1, true);
          card.append(actions); list.append(card);
        });
        group.append(list); root.append(group);
      });
      if (focusId) document.getElementById(`structured-${focusId}-description`)?.focus();
    };
    render();
    if (location.hash) document.getElementById(location.hash.slice(1))?.focus();
    form.addEventListener('input', () => { changed = true; });
    form.addEventListener('submit', (event) => {
      payload.value = JSON.stringify(segments);
      const bytes = new TextEncoder().encode(new URLSearchParams(new FormData(form)).toString()).length;
      if (bytes > Number(form.dataset.maxBody)) {
        event.preventDefault();
        let notice = form.querySelector('[data-editor-size-error]');
        if (!notice) { notice = node('p', '', 'authoring-error'); notice.dataset.editorSizeError = ''; notice.setAttribute('role', 'alert'); form.append(notice); }
        notice.textContent = 'This edit exceeds the request size limit. Your changes remain here; reduce the total text before saving.';
        notice.scrollIntoView(); return;
      }
      saving = true;
    });
    form.querySelector('[data-cancel-testcase]')?.addEventListener('click', (event) => {
      if (changed && !window.confirm('Discard your unsaved changes?')) event.preventDefault();
      else saving = true;
    });
    document.querySelector('[data-discard-draft]')?.addEventListener('submit', (event) => {
      if (!window.confirm('Discard this Draft and its unsaved changes?')) event.preventDefault();
      else saving = true;
    });
    window.addEventListener('beforeunload', (event) => {
      if (changed && !saving) { event.preventDefault(); event.returnValue = ''; }
    });
  });
})();
(() => {
  const csrf = document.querySelector('meta[name="qa-csrf-token"]')?.content || '';
  window.qaRenderStop = (snapshot) => {
    document.querySelectorAll('[data-stop-form]').forEach((form) => {
      if (snapshot.finished) { form.hidden = true; return; }
      const stopping = snapshot.state === 'CANCELLATION_REQUESTED' || form.dataset.requested === 'true';
      const button = form.querySelector('[data-stop-button]');
      button.disabled = stopping;
      button.textContent = stopping ? 'Stopping…' : 'Stop';
      if (stopping) form.querySelector('[data-stop-notice]').textContent = 'Cancellation requested. Waiting for safe cleanup.';
    });
  };
  document.querySelectorAll('[data-stop-form]').forEach((form) => {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (form.dataset.requested === 'true') return;
      form.dataset.requested = 'true';
      const button = form.querySelector('[data-stop-button]');
      const notice = form.querySelector('[data-stop-notice]');
      button.disabled = true; button.textContent = 'Stopping…';
      notice.textContent = 'Sending cancellation request…';
      try {
        const response = await fetch(form.action, {
          method: 'POST', headers: {'X-QA-CSRF': csrf, 'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({_csrf: csrf})
        });
        const result = await response.json();
        if (!response.ok && response.status !== 409) throw new Error('Stop unavailable');
        notice.textContent = result.message;
      } catch (_error) {
        form.dataset.requested = 'false'; button.disabled = false; button.textContent = 'Stop';
        notice.textContent = 'Stop request could not be confirmed. Retry.';
      }
    });
  });

  const panel = document.querySelector('[data-execution-preferences]');
  if (!panel) return;
  const status = panel.querySelector('[data-preferences-status]');
  const retry = panel.querySelector('[data-preferences-retry]');
  const pending = new Map();
  let saving = false, failed = false, revision = Number(panel.dataset.revision);
  const preferenceSelects = panel.querySelectorAll('select[name="cookie_policy"], select[name="evidence_mode"], select[name="screenshot_mode"]');
  const reconcile = (preferences) => {
    revision = preferences.revision; panel.dataset.revision = String(revision);
    preferenceSelects.forEach((select) => {
      if (!pending.has(select.name)) select.value = preferences[select.name];
    });
  };
  const dirty = () => saving || failed || pending.size > 0;
  const setRunButtons = () => {
    document.querySelectorAll('[data-run-form] button[type="submit"]').forEach((button) => { button.disabled = dirty(); });
  };
  document.addEventListener('submit', (event) => {
    if (event.target.matches('[data-run-form]') && dirty()) {
      event.preventDefault(); event.stopImmediatePropagation();
      status.textContent = failed ? 'Save failed — Retry' : 'Saving…';
    }
  }, true);
  const save = async () => {
    if (saving || failed || !pending.size) return;
    saving = true; status.textContent = 'Saving…'; retry.hidden = true; setRunButtons();
    try {
      while (pending.size) {
        const [field, value] = pending.entries().next().value;
        const response = await fetch(panel.dataset.executionPreferences, {
          method: 'POST', headers: {'X-QA-CSRF': csrf, 'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({field, value, revision: String(revision)})
        });
        const result = await response.json();
        if (response.status === 409 && result.preferences) reconcile(result.preferences);
        if (!response.ok) throw new Error('Save failed');
        if (pending.get(field) === value) pending.delete(field);
        reconcile(result);
      }
      status.textContent = 'Saved';
    } catch (_error) {
      failed = true; status.textContent = 'Save failed — Retry'; retry.hidden = false;
    } finally { saving = false; setRunButtons(); }
  };
  preferenceSelects.forEach((select) => {
    select.addEventListener('change', () => { pending.set(select.name, select.value); setRunButtons(); save(); });
  });
  retry.addEventListener('click', () => { failed = false; save(); });
})();

(() => {
  const actionFields = {
    navigate: ['url'],
    assert_page_loaded: [],
    assert_title: ['expected'],
    assert_visible: ['selector', 'expected_text'],
    click: ['selector'],
    check: ['selector'],
    uncheck: ['selector'],
    fill: ['selector', 'value'],
    assert_hidden: ['selector'],
    assert_url: ['expected'],
    select_option: ['selector', 'option_label'],
    assert_text_contains: ['expected_text', 'selector'],
    assert_checked: ['selector'], assert_unchecked: ['selector'],
    assert_selected: ['selector', 'expected'],
    assert_enabled: ['selector'],
    assert_disabled: ['selector']
  };
  const requiredActionFields = {
    navigate: ['url'], assert_page_loaded: [], assert_title: ['expected'],
    assert_visible: ['selector'], click: ['selector'], check: ['selector'],
    uncheck: ['selector'], fill: ['selector', 'value'],
    assert_hidden: ['selector'], assert_url: ['expected'],
    select_option: ['selector', 'option_label'],
    assert_text_contains: ['expected_text'], assert_checked: ['selector'],
    assert_unchecked: ['selector'],
    assert_selected: ['selector'], assert_enabled: ['selector'], assert_disabled: ['selector']
  };
  const actionLabels = {
    navigate: 'Navigate to URL', assert_page_loaded: 'Assert page loaded',
    assert_title: 'Assert page title', assert_visible: 'Assert visible', click: 'Click',
    check: 'Check checkbox', uncheck: 'Uncheck checkbox',
    fill: 'Fill field', assert_hidden: 'Assert hidden', assert_url: 'Assert URL',
    select_option: 'Select option', assert_text_contains: 'Assert text contains',
    assert_checked: 'Assert checked', assert_unchecked: 'Assert unchecked',
    assert_selected: 'Assert selected',
    assert_enabled: 'Assert enabled', assert_disabled: 'Assert disabled'
  };
  const fieldLabels = {
    url: 'URL', selector: 'Selector', value: 'Value', expected: 'Expected value',
    expected_text: 'Expected text', option_label: 'Option label'
  };
  const draftSource = document.querySelector('[data-source-draft-id]');
  const draftStatus = document.querySelector('[data-draft-selection-status]');
  const draftSummary = document.getElementById('case-name');
  const selectedDraft = [...document.querySelectorAll('[data-draft-select]')].find(button => button.dataset.draftId === draftSource?.value);
  let loadingDraft = false, summaryEdited = !!draftSummary?.value && draftSummary.value !== selectedDraft?.dataset.title;
  draftSummary?.addEventListener('input', () => { if (!loadingDraft) summaryEdited = true; });
  document.querySelectorAll('[data-draft-select]').forEach((button) => {
    button.addEventListener('click', () => {
      const summary = document.getElementById('case-name');
      const website = document.getElementById('case-base-url');
      const scenario = document.getElementById('case-scenario');
      if (!summary || !website || !scenario || !draftSource || !draftStatus) return;
      loadingDraft = true;
      if (!summaryEdited || !summary.value.trim()) summary.value = button.dataset.title || '';
      website.value = button.dataset.website || '';
      scenario.value = button.dataset.scenario || '';
      draftSource.value = button.dataset.draftId || '';
      summary.dispatchEvent(new Event('input', { bubbles: true }));
      website.dispatchEvent(new Event('input', { bubbles: true }));
      scenario.dispatchEvent(new Event('input', { bubbles: true }));
      loadingDraft = false;
      document.querySelectorAll('[data-draft-select]').forEach((item) => {
        item.setAttribute('aria-pressed', String(item === button));
      });
      draftStatus.textContent = `Loaded Draft: ${button.dataset.title || 'Draft'}. Changes here are saved only when you choose Save Draft.`;
      draftStatus.classList.remove('muted');
    });
  });
  document.querySelectorAll('[data-automation-form]').forEach((form) => {
    const actions = form.querySelector('[data-automation-actions]');
    const makeButton = (label, attribute, value) => {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'button'; button.textContent = label;
      button.dataset[attribute] = value || '';
      return button;
    };
    const readParameters = (card) => Object.fromEntries(
      [...card.querySelectorAll('[data-action-param]')].map((field) => [field.dataset.actionParam, field.value])
    );
    const renderFields = (card, values = {}) => {
      const fields = card.querySelector('[data-action-fields]');
      const type = card.querySelector('[data-action-type]').value;
      fields.replaceChildren();
      (actionFields[type] || []).forEach((name) => {
        const label = document.createElement('label'); label.className = 'field';
        const caption = document.createElement('span'); caption.textContent = fieldLabels[name] || name;
        const input = document.createElement('input');
        input.name = ''; input.value = values[name] || ''; input.autocomplete = 'off';
        input.dataset.actionParam = name;
        if ((requiredActionFields[type] || []).includes(name)) input.required = true;
        label.append(caption, input); fields.append(label);
      });
    };
    const makeCard = (type = 'assert_page_loaded', values = {}) => {
      const card = document.createElement('fieldset'); card.className = 'automation-action'; card.dataset.actionCard = '';
      const legend = document.createElement('legend'); legend.textContent = 'Action'; card.append(legend);
      const controls = document.createElement('div'); controls.className = 'automation-action-controls';
      controls.append(
        makeButton('↑', 'actionMove', 'up'), makeButton('↓', 'actionMove', 'down'),
        makeButton('Duplicate', 'actionDuplicate'), makeButton('Delete', 'actionRemove')
      ); card.append(controls);
      const typeLabel = document.createElement('label'); typeLabel.className = 'field';
      const typeCaption = document.createElement('span'); typeCaption.textContent = 'Action type';
      const select = document.createElement('select'); select.dataset.actionType = '';
      const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose action type';
      select.append(placeholder);
      Object.keys(actionFields).forEach((key) => {
        const option = document.createElement('option'); option.value = key;
        option.textContent = actionLabels[key]; select.append(option);
      });
      select.value = type; select.addEventListener('change', () => renderFields(card, readParameters(card)));
      typeLabel.append(typeCaption, select); card.append(typeLabel);
      const container = document.createElement('div'); container.className = 'automation-action-fields';
      container.dataset.actionFields = ''; card.append(container); renderFields(card, values);
      return card;
    };
    const renumber = () => {
      [...actions.querySelectorAll('[data-action-card]')].forEach((card, index) => {
        card.querySelector('legend').textContent = `Action ${index + 1}`;
        card.querySelector('[data-action-type]').name = `action.${index}.type`;
        card.querySelectorAll('[data-action-param]').forEach((field) => {
          field.name = `action.${index}.param.${field.dataset.actionParam}`;
        });
        const controls = card.querySelector('.automation-action-controls');
        controls.querySelector('[data-action-move="up"]').disabled = index === 0;
        controls.querySelector('[data-action-move="down"]').disabled = index === actions.children.length - 1;
      });
    };
    form.addEventListener('click', (event) => {
      const target = event.target.closest('button');
      if (!target) return;
      if (target.matches('[data-action-add]')) {
        actions.append(makeCard()); renumber(); return;
      }
      const card = target.closest('[data-action-card]');
      if (!card) return;
      if (target.matches('[data-action-remove]')) card.remove();
      else if (target.matches('[data-action-duplicate]')) {
        const clone = makeCard(card.querySelector('[data-action-type]').value, readParameters(card));
        card.after(clone);
      } else if (target.matches('[data-action-move="up"]') && card.previousElementSibling) {
        actions.insertBefore(card, card.previousElementSibling);
      } else if (target.matches('[data-action-move="down"]') && card.nextElementSibling) {
        actions.insertBefore(card.nextElementSibling, card);
      }
      renumber();
    });
    renumber();
  });

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

  const scenarioInputError = (value) => {
    const words = value.match(/\p{L}+/gu) || [];
    const insufficient = !words.length || words.every((word) => [...word].length === 1) ||
      words.every((word) => new Set([...word.toLowerCase()]).size === 1) ||
      (words.length === 1 && /^[A-Za-z]+$/.test(words[0]) && words[0].length <= 2 && words[0].toLowerCase() !== 'go');
    return insufficient ? 'Describe an action and what you expect to happen, for example "Check login". You can also choose Create Manually without AI.' : '';
  };
  let generatedInlineValidationControlId = 0;
  document.querySelectorAll('form[data-inline-validation]').forEach((form) => {
    const controls = [...form.querySelectorAll('input[required], textarea[required], select[required], #case-name')];
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
        if (form.hasAttribute('data-authoring-form')) message = scenarioInputError(value);
      } else if (control.required && !value) {
        message = 'Complete this field.';
      }
      if (form.hasAttribute('data-authoring-form') && control.maxLength > 0 && control.value.length > control.maxLength) {
        message = control.name === 'name' ? 'Keep Summary under 200 characters.' : 'Shorten this field to the displayed limit.';
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
      if (event.submitter?.formNoValidate) return;
      const invalid = controls.find((control) => Boolean(validate(control)));
      if (invalid) {
        event.preventDefault();
        event.stopImmediatePropagation();
        invalid.focus();
      }
    });
  });

  document.querySelectorAll('[data-authoring-form], [data-testcase-editor]').forEach((form) => {
    form.addEventListener('submit', (event) => {
      const button = event.submitter;
      if (!button || event.defaultPrevented) return;
      if (form.hasAttribute('data-testcase-editor') && !button.hasAttribute('data-regenerate-testcase')) return;
      button.disabled = true;
      button.classList.add('is-disabled');
      button.setAttribute('aria-busy', 'true');
      button.textContent = button.formAction.includes('/drafts/save') ? 'Saving...' :
        button.formAction.includes('/manual/prepare') ? 'Opening manual form...' : 'Starting...';
    });
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
      const elapsed = document.querySelector('[data-authoring-elapsed]');
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
      window.qaRenderStop(snapshot);
      document.querySelectorAll('[data-authoring-phase]').forEach((node) => {
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
      const provider = document.querySelector('[data-authoring-provider]');
      if (provider) provider.textContent = snapshot.provider_name || '—';
      const state = document.querySelector('[data-authoring-state]');
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
          CANCELLED: 'Stopped by user',
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

  function textNode(tag, text, className = '') {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function localLink(url, label) {
    const node = textNode('a', label);
    if (url && url.startsWith('/') && !url.startsWith('//') && !url.includes('\\')) node.href = url;
    return node;
  }

  function render(snapshot) {
    window.qaRenderStop(snapshot);
    document.querySelectorAll('[data-progress-phase]').forEach((node) => {
      node.textContent = snapshot.phase;
    });
    const elapsed = document.querySelector('[data-progress-elapsed]');
    if (elapsed) elapsed.textContent = `${(snapshot.elapsed_ms / 1000).toFixed(1)} s`;
    const diagnostics = document.querySelector('[data-progress-diagnostics]');
    diagnostics.replaceChildren();
    snapshot.diagnostic_stages.forEach((stage) => {
      const section = document.createElement('section');
      section.append(textNode('h3', stage.stage));
      const list = document.createElement('ol');
      stage.events.forEach((event) => {
        const row = document.createElement('li');
        row.className = 'progress-event';
        const detail = document.createElement('div');
        const fields = [event.display_status];
        if (event.step_order != null) fields.push(`Step ${event.step_order + 1}`);
        if (event.attempt_number != null) fields.push(`Attempt ${event.attempt_number}`);
        if (event.duration_ms != null) fields.push(`${(event.duration_ms / 1000).toFixed(1)} s`);
        detail.append(textNode('time', event.timestamp), document.createTextNode(` \u00b7 ${fields.join(' \u00b7 ')}`), textNode('p', event.reason));
        const raw = document.createElement('details');
        raw.append(textNode('summary', 'Raw details'), textNode('pre', JSON.stringify(event, null, 2)));
        detail.append(raw); row.append(detail); list.append(row);
      });
      section.append(list); diagnostics.append(section);
    });
    if ((snapshot.provider_diagnostics || []).length) {
      const section = document.createElement('section');
      section.append(textNode('h3', 'Recorded provider attempts'));
      const list = document.createElement('ol');
      snapshot.provider_diagnostics.forEach((attempt) => {
        const row = document.createElement('li'); row.className = 'progress-event';
        const detail = document.createElement('div');
        const fields = [`Step ${attempt.step_number}`, `Attempt ${attempt.attempt_number}`, attempt.request_kind, attempt.provider, attempt.status];
        if (attempt.model) fields.push(attempt.model);
        if (attempt.duration_ms != null) fields.push(`${(attempt.duration_ms / 1000).toFixed(1)} s`);
        detail.append(textNode('p', fields.join(' \u00b7 ')));
        if (attempt.reason) detail.append(textNode('p', attempt.reason));
        const raw = document.createElement('details');
        raw.append(textNode('summary', 'Raw details'), textNode('pre', JSON.stringify(attempt, null, 2)));
        detail.append(raw); row.append(detail); list.append(row);
      });
      section.append(list); diagnostics.append(section);
    }

    stepsList.replaceChildren();
    snapshot.steps.forEach((step, index) => {
      const row = document.createElement('li'); row.className = 'progress-step';
      const icon = textNode('span', step.display_status === 'Passed' ? '\u2713' : '\u25cb', 'progress-symbol');
      icon.setAttribute('aria-hidden', 'true');
      const content = document.createElement('span');
      content.append(textNode('strong', `Step ${index + 1}: ${step.name}`), textNode('span', step.display_status, 'progress-step-state'));
      if (step.automation_state && step.automation_state !== 'Failed') content.append(textNode('span', `Automation ${step.automation_state.toLowerCase()}`, 'muted'));
      if (step.plan_url) {
        const plan = localLink(step.plan_url, `Plan version v${step.plan_version}`);
        const line = document.createElement('span'); line.className = 'muted'; line.append(plan); content.append(line);
      } else content.append(textNode('span', 'No saved TestPlan available.', 'muted'));
      if (step.observation) content.append(textNode('p', `Observed: ${step.observation}`));
      if (step.action_failure) {
        const details = document.createElement('details');
        details.append(textNode('summary', 'Action failure details'), textNode('pre', JSON.stringify(step.action_failure, null, 2)));
        content.append(details);
      }
      if (step.evidence_count) content.append(textNode('span', `Screenshot evidence captured (${step.evidence_count})`, 'progress-evidence'));
      (step.evidence_links || []).forEach((evidence) => {
        const link = localLink(evidence.url, `Open full-size evidence \u00b7 ${evidence.label}`);
        link.target = '_blank'; link.rel = 'noopener'; content.append(link, document.createElement('br'));
      });
      row.append(icon, content); stepsList.append(row);
    });

    if (snapshot.finished) {
      result.hidden = false;
      resultContent.replaceChildren();
      const summary = snapshot.summary;
      resultContent.append(textNode('strong', summary.status), textNode('p', `${summary.completed_steps} of ${summary.total_steps} steps verified.`));
      if (summary.stopping_detail) resultContent.append(textNode('p', summary.stopping_detail));
      resultContent.append(textNode('p', summary.explanation));
      const issue = summary.stopping_step ? snapshot.steps[summary.stopping_step - 1] : null;
      if (issue && issue.observation) resultContent.append(textNode('p', `Observed: ${issue.observation}`));
      resultContent.append(textNode('p', `Recommended action: ${summary.recommended_action}`));
      if (summary.history_note) resultContent.append(textNode('p', summary.history_note));
      const actions = document.createElement('div'); actions.className = 'actions';
      snapshot.actions.forEach((action) => {
        const link = localLink(action.url, action.label); link.className = 'button'; actions.append(link);
      });
      resultContent.append(actions);
      if (['AUTOMATION_EXECUTION_ERROR', 'AUTOMATION_DRIFT', 'AUTOMATION_GENERATION_ERROR', 'INFRASTRUCTURE_ERROR'].includes(summary.outcome) && !document.querySelector('[data-progress-retry]')) {
        const form = document.createElement('form'); form.method = 'post';
        form.action = `/test-cases/${encodeURIComponent(snapshot.test_case_id)}/run`;
        form.dataset.runForm = ''; form.dataset.progressRetry = '';
        const workflow = document.createElement('input'); workflow.type = 'hidden'; workflow.name = 'workflow'; workflow.value = snapshot.workflow;
        const button = textNode('button', summary.outcome === 'AUTOMATION_GENERATION_ERROR' ? 'Retry Automation' : 'Retry Run', 'button'); button.type = 'submit';
        form.append(workflow, button); result.append(form);
        form.addEventListener('submit', () => { button.disabled = true; }, {once: true});
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

(() => {
  const root = document.querySelector('[data-suite-run-progress]');
  if (!root) return;
  const endpoint = root.dataset.suiteRunProgress;
  const summary = root.querySelector('[data-suite-run-summary]');
  const live = root.querySelector('[data-suite-run-live]');
  const list = root.querySelector('[data-suite-run-items]');
  const terminal = new Set(['COMPLETED', 'COMPLETED_WITH_FAILURES', 'INTERRUPTED', 'FAILED', 'CANCELLED']);
  const render = (run) => {
    window.qaRenderStop({finished: terminal.has(run.status), state: run.status});
    summary.textContent = run.summary_text;
    live.textContent = terminal.has(run.status) ? `${run.display_status}.` : `Run ${run.status.toLowerCase()}…`;
    list.replaceChildren();
    run.items.forEach((item) => {
      const row = document.createElement('li'); row.className = 'suite-run-item';
      const heading = document.createElement('strong');
      const itemDuration = item.duration_ms == null ? '' : ` · ${(item.duration_ms / 1000).toFixed(1)} s`;
      heading.textContent = `${item.order_index + 1}. ${item.test_case_public_id || ''} — ${item.test_case_name} — ${item.display_status}${itemDuration}`;
      row.append(heading);
      if (item.attempts.length) {
        const attempts = document.createElement('ul');
        item.attempts.forEach((attempt) => {
          const line = document.createElement('li');
          const attemptDuration = attempt.duration_ms == null ? '' : ` · ${(attempt.duration_ms / 1000).toFixed(1)} s`;
          line.append(document.createTextNode(`Attempt ${attempt.attempt_number} — ${attempt.display_status}${attemptDuration} → `));
          if (attempt.run_id) {
            const link = document.createElement('a'); link.href = `/runs/${encodeURIComponent(attempt.run_id)}`;
            link.textContent = attempt.run_public_id || 'Open TestCase Run'; line.append(link);
          } else {
            line.append(document.createTextNode('No persisted Run record'));
          }
          attempts.append(line);
        });
        row.append(attempts);
      } else {
        const waiting = document.createElement('p'); waiting.className = 'muted';
        waiting.textContent = item.display_status; row.append(waiting);
      }
      list.append(row);
    });
  };
  const pollSuite = async () => {
    try {
      const response = await fetch(endpoint, {headers: {'Accept': 'application/json'}, cache: 'no-store'});
      if (!response.ok) throw new Error('Progress unavailable');
      const run = await response.json(); render(run);
      if (terminal.has(run.status)) return;
    } catch (_error) {
      live.textContent = 'Live updates are temporarily unavailable. Retrying…';
    }
    window.setTimeout(pollSuite, 1000);
  };
  pollSuite();
})();
"""


if __name__ == "__main__":
    raise SystemExit(main())
