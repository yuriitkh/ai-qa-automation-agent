"""Local UI for persisted TestCases and TestRun history."""

import argparse
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
    outcome_label,
    outcome_tone,
    short_id,
)
from qa_agent.reporting import RunAttemptReport, RunEvidenceReport, RunReportGenerator, RunStepReport
from qa_agent.run_history import RunHistoryDetail, RunHistoryRecord, RunHistoryService, WorkflowType
from qa_agent.test_case_execution import RunUnavailableError, TestCaseExecutionService
from qa_agent.test_case_repository import TestCaseRepository
from qa_agent.storage import create_sqlite_storage


_PAGE_LIMIT = 500
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
        return cls(303, "text/plain; charset=utf-8", b"Run recorded. Redirecting.\n", {"Location": location})


class LocalWebApplication:
    """Route requests through history/report services, never raw SQL."""

    def __init__(
        self,
        run_history: RunHistoryService,
        reports: RunReportGenerator | None = None,
        evidence_root: str | Path | None = None,
        test_cases: TestCaseRepository | None = None,
        run_service: TestCaseExecutionService | None = None,
    ) -> None:
        self._run_history = run_history
        self._reports = reports or RunReportGenerator()
        self._evidence_root = Path(evidence_root).expanduser() if evidence_root else None
        self._test_cases = test_cases
        self._run_service = run_service

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
        if path == "/":
            return WebResponse.html(200, self._dashboard())
        if path == "/demo-target/registration":
            return WebResponse.html(200, _local_demo_page())
        if path == "/test-cases":
            return WebResponse.html(200, self._test_case_list())
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
        if isinstance(body, bytes):
            try:
                form_body = body.decode("utf-8")
            except UnicodeDecodeError:
                return self._run_error(test_case_id, "The run request was not valid form data.", 400)
        else:
            form_body = body or ""
        if len(form_body) > 8192:
            return self._run_error(test_case_id, "The run request was too large.", 400)
        workflow_value = parse_qs(form_body, keep_blank_values=True).get("workflow", [""])[0]
        try:
            workflow = WorkflowType(workflow_value)
        except ValueError:
            return self._run_error(test_case_id, "Choose Validation or Regression before starting a run.", 400)
        if workflow not in {WorkflowType.VALIDATION, WorkflowType.REGRESSION}:
            return self._run_error(test_case_id, "Choose Validation or Regression before starting a run.", 400)
        try:
            result = self._run_service.run(test_case_id, workflow)
        except RunUnavailableError as error:
            return self._run_error(test_case_id, str(error), 409)
        except Exception:
            return self._run_error(
                test_case_id,
                "The run could not be started. Review saved plans and setup configuration.",
                500,
            )
        return WebResponse.redirect(f"/runs/{result.test_run.id}")

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
            '<p class="lead">A clear view of recent validation and regression runs.</p></header>'
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
                    f'<td><a href="/test-cases/{test_case_id}">{escape_html(test_case.name)}</a>'
                    f'<div class="muted">{escape_html(short_id(test_case_id))}</div></td>'
                    f"<td>{status}</td><td>{outcome}</td><td>{last_run}</td>"
                    f"<td>{len(history)}</td><td>{latest_workflow}</td></tr>"
                )
        else:
            for test_case_id, latest in latest_by_case.items():
                history = grouped[test_case_id]
                rows.append(
                    "<tr>"
                    f'<td><a href="/test-cases/{test_case_id}">{escape_html(latest.test_case_name)}</a>'
                    f'<div class="muted">{escape_html(short_id(test_case_id))}</div></td>'
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
            '<header class="page-heading"><h1>Test Cases</h1>'
            '<p class="lead">Saved TestCase definitions and their run history.</p></header>'
            '<section class="panel"><h2>TestCases</h2>'
            + (table if rows else self._empty_state(
                "No TestCases found.", "Persist a TestCase definition to see it here."
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
                f'<td><a href="/test-cases/{record.test_case_id}">{escape_html(record.test_case_name)}</a></td>'
                f'<td>{badge(record.workflow_type.value, "workflow")}</td>'
                f'<td><time datetime="{escape_html(record.started_at.isoformat())}">'
                f'{escape_html(format_timestamp(record.started_at))}</time></td>'
                f'<td>{escape_html(format_duration(record.duration_ms))}</td>'
                f'<td><a href="/runs/{record.run_id}" title="{escape_html(record.run_id)}">'
                f'{escape_html(short_id(record.run_id))}</a></td>'
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
            f'{escape_html(short_id(record.run_id))}</a></td></tr>'
            for record in records
        )
        history_table = (
            '<div class="table-wrap"><table><thead><tr><th>Status</th><th>Result</th><th>Workflow</th>'
            '<th>Last run</th><th>Duration</th><th>Run</th></tr></thead><tbody>'
            + history_rows
            + '</tbody></table></div>'
            if records else self._empty_state("No runs yet.", "Start Validation or Regression to add run history.")
        )
        run_form = self._run_form(test_case_id) if self._run_service is not None and test_case is not None else ""
        content = (
            '<header class="page-heading"><h1>' + escape_html(case_name) + '</h1>'
            + f'<p class="lead">{escape_html(case_description)}</p>'
            + f'<p class="muted">{escape_html(source_note)}</p></header>'
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
                case_name,
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
            f'<form method="post" action="/test-cases/{test_case_id}/run" class="filters">'
            '<div class="field"><label for="run-workflow">Workflow</label>'
            '<select id="run-workflow" name="workflow" required>'
            '<option value="" disabled selected>Choose a workflow</option>'
            '<option value="VALIDATION">Validation</option>'
            '<option value="REGRESSION">Regression</option>'
            '</select></div><button class="button primary" type="submit">Start run</button></form>'
            '</section>'
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
    run_service = TestCaseExecutionService(
        storage.test_case_repository,
        storage.plan_store,
        storage.execution_repository,
        storage.run_history,
        evidence_directory=evidence_root,
    )
    return LocalWebApplication(
        storage.run_history,
        evidence_root=evidence_root,
        test_cases=storage.test_case_repository,
        run_service=run_service,
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
            if content_length < 0 or content_length > 8192:
                self.close_connection = True
                body = b"x" * 8193
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

    return ThreadingHTTPServer((host, port), Handler)


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


def _outcome_badge(record: RunHistoryRecord, *, stacked: bool = False) -> str:
    if record.outcome in {record.status.value, "PASSED", "FAILED"}:
        return '<span class="muted">—</span>'
    shown = badge(record.outcome, outcome_tone(record.outcome))
    return f'<div class="row-sub-badge">{shown}</div>' if stacked else shown


if __name__ == "__main__":
    raise SystemExit(main())
