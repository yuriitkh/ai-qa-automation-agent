"""Deterministic JSON reports derived from domain TestRun data."""

from pathlib import Path
from datetime import datetime
from typing import Any
from uuid import UUID
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.models import (
    Evidence,
    EvidenceType,
    Execution,
    ExecutionStatus,
    PlanVersionOrigin,
    TestRun,
)
from qa_agent.presentation import (
    UI_CSS,
    badge as presentation_badge,
    escape_html,
    format_duration,
    format_timestamp,
    outcome_tone,
)
from qa_agent.redaction import redact_secrets
from qa_agent.run_history import (
    HistoryExecutionReference,
    HistoryPrecondition,
    RunHistoryDetail,
    RunHistoryRecord,
    WorkflowType,
)
from qa_agent.setup_orchestration import CleanupOutcome, SetupRunOutcome


class EvidenceReport(BaseModel):
    id: UUID
    execution_id: UUID
    type: EvidenceType
    path: str
    description: str | None = None
    timestamp: datetime | None = None

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "EvidenceReport":
        return cls(
            id=evidence.id,
            execution_id=evidence.execution_id,
            type=evidence.type,
            path=evidence.path,
            description=evidence.description,
            timestamp=evidence.timestamp,
        )


class TestAttemptReport(BaseModel):
    execution_id: UUID
    test_plan_version_id: UUID
    status: ExecutionStatus
    started_at: datetime
    finished_at: datetime | None
    actual_result: Any
    error: str | None
    evidence: list[EvidenceReport] = Field(default_factory=list)

    @classmethod
    def from_execution(cls, execution: Execution) -> "TestAttemptReport":
        return cls(
            execution_id=execution.id,
            test_plan_version_id=execution.test_plan_version_id,
            status=execution.status,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            actual_result=execution.actual_result,
            error=execution.error,
            evidence=[EvidenceReport.from_evidence(item) for item in execution.evidence],
        )


class TestStepReport(BaseModel):
    step_id: UUID
    name: str
    status: ExecutionStatus | None
    attempts: list[TestAttemptReport]


class TestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID
    test_case_id: UUID
    status: ExecutionStatus
    started_at: datetime
    finished_at: datetime | None
    steps: list[TestStepReport]

    def to_json(self) -> str:
        """Serialize with stable field order, enum values, UUIDs, and ISO datetimes."""
        return self.model_dump_json(indent=2)


class TestReportGenerator:
    """Build reports from a TestRun without depending on execution infrastructure."""

    def generate(self, test_run: TestRun) -> TestReport:
        executions_by_step: dict[UUID, list[Execution]] = {
            step_id: [] for step_id in test_run.test_step_ids
        }
        for execution in test_run.executions:
            executions_by_step[execution.test_step_id].append(execution)

        blocked_ids = set(test_run.blocked_step_ids)
        names = test_run.test_step_names or ["" for _ in test_run.test_step_ids]
        steps = []
        for step_id, name in zip(test_run.test_step_ids, names):
            attempts = executions_by_step[step_id]
            steps.append(
                TestStepReport(
                    step_id=step_id,
                    name=name,
                    status=(
                        ExecutionStatus.BLOCKED
                        if step_id in blocked_ids
                        else (attempts[-1].status if attempts else None)
                    ),
                    attempts=[
                        self._safe_attempt(item, test_run)
                        for item in attempts
                    ],
                )
            )

        return TestReport(
            run_id=test_run.id,
            test_case_id=test_run.test_case_id,
            status=test_run.overall_status,
            started_at=test_run.started_at,
            finished_at=test_run.finished_at,
            steps=steps,
        )

    @staticmethod
    def _safe_attempt(execution: Execution, test_run: TestRun) -> TestAttemptReport:
        attempt = TestAttemptReport.from_execution(execution)
        return attempt.model_copy(update={
            "actual_result": _safe_report_value(attempt.actual_result, test_run),
            "error": _safe_report_value(attempt.error, test_run),
            "evidence": [
                item.model_copy(update={
                    "path": _safe_report_value(item.path, test_run),
                    "description": _safe_report_value(item.description, test_run),
                })
                for item in attempt.evidence
            ],
        })

    def write_json(self, report: TestReport, path: str | Path) -> None:
        Path(path).write_text(report.to_json(), encoding="utf-8")


class RunEvidenceReport(BaseModel):
    id: UUID
    execution_id: UUID
    type: str
    name: str
    description: str | None = None


class RunAttemptReport(BaseModel):
    execution_id: UUID
    test_plan_version_id: UUID
    plan_version_number: int | None = None
    plan_version_origin: PlanVersionOrigin | None = None
    status: ExecutionStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    actual_result: Any = None
    error: str | None = None
    evidence: list[RunEvidenceReport] = Field(default_factory=list)


class RunStepReport(BaseModel):
    step_id: UUID
    order: int
    name: str
    description: str
    expected: str
    status: str
    attempts: list[RunAttemptReport] = Field(default_factory=list)


class RunReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1"
    run_id: UUID
    run_public_id: str | None = None
    test_case_id: UUID
    test_case_public_id: str | None = None
    test_case_name: str
    test_case_description: str
    workflow_type: WorkflowType
    outcome: str
    status: ExecutionStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    trace_id: UUID | None = None
    base_url: str | None = None
    steps: list[RunStepReport] = Field(default_factory=list)
    segments: list[dict[str, Any]] = Field(default_factory=list)
    preconditions: list[HistoryPrecondition] = Field(default_factory=list)
    setup_status: str | None = None
    cleanup_succeeded: bool | None = None
    cleanup_failures: list[dict[str, str]] = Field(default_factory=list)
    failed_step_ids: list[UUID] = Field(default_factory=list)
    blocked_step_ids: list[UUID] = Field(default_factory=list)
    run_context_safe: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class RunReportGenerator:
    """Build safe JSON and standalone HTML from live or historical runs."""

    def generate_current(
        self,
        test_case,
        test_run: TestRun,
        *,
        workflow_type: WorkflowType,
        outcome: str | None = None,
        setup: SetupRunOutcome | None = None,
        cleanup: CleanupOutcome | None = None,
        trace_id: UUID | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> RunReport:
        record = RunHistoryRecord.from_completed_run(
            test_case,
            test_run,
            workflow_type=workflow_type,
            outcome=outcome,
            setup=setup,
            cleanup=cleanup,
            trace_id=trace_id,
            started_at=started_at,
            finished_at=finished_at,
        )
        return self.generate_history(record)

    def generate_history(
        self,
        source: RunHistoryRecord | RunHistoryDetail,
        executions: Mapping[UUID, Execution] | None = None,
    ) -> RunReport:
        if isinstance(source, RunHistoryDetail):
            record = source.record
            execution_map = source.executions
        else:
            record = source
            execution_map = executions or {}
        references_by_step: dict[UUID, list[HistoryExecutionReference]] = {
            step.id: [] for step in record.steps
        }
        for reference in record.executions:
            references_by_step.setdefault(reference.test_step_id, []).append(reference)

        step_reports: list[RunStepReport] = []
        for step in sorted(record.steps, key=lambda item: item.order):
            attempts = [
                self._attempt_report(reference, execution_map.get(reference.execution_id))
                for reference in references_by_step.get(step.id, [])
            ]
            step_reports.append(RunStepReport(
                step_id=step.id,
                order=step.order,
                name=step.name,
                description=step.description,
                expected=step.expected,
                status=(step.status.value if step.status is not None else "NOT_RUN"),
                attempts=attempts,
            ))

        return RunReport(
            run_id=record.run_id,
            run_public_id=record.public_id,
            test_case_id=record.test_case_id,
            test_case_public_id=record.test_case_public_id,
            test_case_name=record.test_case_name,
            test_case_description=record.test_case_description,
            workflow_type=record.workflow_type,
            outcome=record.outcome,
            status=record.status,
            started_at=record.started_at,
            finished_at=record.finished_at,
            duration_ms=record.duration_ms,
            trace_id=record.trace_id,
            base_url=record.base_url,
            steps=step_reports,
            segments=[item.model_dump(mode="json") for item in record.segments],
            preconditions=record.preconditions,
            setup_status=record.setup_status,
            cleanup_succeeded=record.cleanup_succeeded,
            cleanup_failures=[item.model_dump(mode="json") for item in record.cleanup_failures],
            failed_step_ids=record.failed_step_ids,
            blocked_step_ids=record.blocked_step_ids,
            run_context_safe=record.run_context_safe,
        )

    @staticmethod
    def _attempt_report(
        reference: HistoryExecutionReference,
        execution: Execution | None,
    ) -> RunAttemptReport:
        started_at = execution.started_at if execution is not None else reference.started_at
        finished_at = execution.finished_at if execution is not None else reference.finished_at
        duration_ms = (
            max(0, int((finished_at - started_at).total_seconds() * 1000))
            if finished_at is not None else None
        )
        evidence = [
            RunEvidenceReport(
                id=item.id,
                execution_id=item.execution_id,
                type=item.type,
                name=item.name,
                description=item.description,
            )
            for item in reference.evidence
        ]
        return RunAttemptReport(
            execution_id=reference.execution_id,
            test_plan_version_id=reference.test_plan_version_id,
            plan_version_number=reference.plan_version_number,
            plan_version_origin=reference.plan_version_origin,
            status=reference.status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            actual_result=reference.safe_actual_result,
            error=reference.safe_error,
            evidence=evidence,
        )

    def to_html(
        self,
        report: RunReport,
        *,
        evidence_url: Callable[
            [RunStepReport, RunAttemptReport, RunEvidenceReport, int], str | None
        ] | None = None,
        show_html_report_link: bool = True,
        run_details_url: str | None = None,
    ) -> str:
        """Render a standalone, safe HTML report; evidence URLs come from the host."""
        esc = escape_html
        special_outcomes = {
            "PRODUCT_FAILURE": (
                "The automation executed the check and detected unexpected product behavior."
            ),
            "AUTOMATION_DRIFT": "The automation no longer matches the UI.",
            "INFRASTRUCTURE_ERROR": "The test could not be reliably executed.",
            "SETUP_FAILURE": "Setup failed, so product execution did not begin.",
        }
        steps_html: list[str] = []
        for number, step in enumerate(report.steps, start=1):
            attempts_html: list[str] = []
            evidence_available = False
            for attempt in step.attempts:
                details: list[str] = []
                if attempt.status == ExecutionStatus.FAILED:
                    details.append(
                        '<div class="detail-grid">'
                        f'<div class="detail-box"><h4>Expected</h4><p>{esc(step.expected)}</p></div>'
                        f'<div class="detail-box"><h4>Actual</h4><p>{esc(_display_value(attempt.actual_result))}</p></div>'
                        "</div>"
                    )
                    if attempt.error:
                        details.append(
                            f'<div class="notice danger"><strong>Error:</strong> {esc(attempt.error)}</div>'
                        )
                elif attempt.actual_result is not None:
                    details.append(
                        f'<p class="muted"><strong>Observed:</strong> '
                        f'{esc(_display_value(attempt.actual_result))}</p>'
                    )

                if attempt.status == ExecutionStatus.FAILED:
                    version_value = (
                        f'<code title="{esc(attempt.test_plan_version_id)}">'
                        f'{esc(_plan_version_label(attempt))}</code>'
                    )
                else:
                    version_value = (
                        f'<code title="{esc(attempt.test_plan_version_id)}">'
                        f'{esc(_plan_version_label(attempt))}</code>'
                    )
                details.append(
                    f'<p class="muted"><strong>Automation · Plan version:</strong> {version_value}</p>'
                )
                details.append(
                    '<details class="plan-version-details"><summary>Version details</summary>'
                    f'<p>Plan version ID: <code>{esc(attempt.test_plan_version_id)}</code></p>'
                    '</details>'
                )
                if attempt.duration_ms is not None:
                    details.append(
                        f'<p class="muted">Attempt duration: {esc(format_duration(attempt.duration_ms))}</p>'
                    )
                if attempt.evidence:
                    evidence_items = []
                    for evidence_index, evidence in enumerate(attempt.evidence):
                        name = esc(evidence.name)
                        url = evidence_url(step, attempt, evidence, evidence_index) if evidence_url else None
                        if url:
                            safe_url = esc(url)
                            evidence_items.append(
                                f'<a href="{safe_url}" target="_blank" rel="noopener">'
                                f'<img class="evidence-preview" src="{safe_url}" alt="Screenshot: {name}">'
                                f"Open {name}</a>"
                            )
                            evidence_available = True
                        elif evidence_url is None:
                            evidence_items.append(
                                f'<span>{name} (local image link unavailable in exported file)</span>'
                            )
                        else:
                            evidence_items.append(f'<span class="muted">{name} — unavailable</span>')
                    details.append(
                        '<div class="detail-box"><h4>Evidence</h4>'
                        + "<br>".join(evidence_items)
                        + "</div>"
                    )
                elif not evidence_available:
                    details.append('<p class="muted">No evidence available.</p>')

                attempts_html.append(
                    '<div class="detail-box"><div class="status-line">'
                    + presentation_badge(attempt.status.value)
                    + f'<span class="muted">Attempt · {esc(format_timestamp(attempt.started_at))}</span>'
                    + "</div>"
                    + "".join(details)
                    + "</div>"
                )

            status_class = "passed" if step.status == "PASSED" else "failed" if step.status == "FAILED" else "blocked" if step.status == "BLOCKED" else ""
            if step.status == "BLOCKED":
                outcome_detail = (
                    '<div class="notice warning">Not executed because a previous step '
                    "blocked continuation.</div>"
                )
            elif not step.attempts:
                outcome_detail = '<p class="muted">No execution attempt was recorded.</p>'
            else:
                outcome_detail = "".join(attempts_html)
            steps_html.append(
                f'<article class="step-card {status_class}"><div class="step-heading">'
                + presentation_badge(step.status, title=f"Step status: {step.status}")
                + f"<h3>Step {number} — {esc(step.name)}</h3></div>"
                + (f'<p class="muted">{esc(step.description)}</p>' if step.description else "")
                + (f'<p><strong>Expected:</strong> {esc(step.expected)}</p>' if step.status != "FAILED" else "")
                + outcome_detail
                + "</article>"
            )

        setup_html = ""
        if report.preconditions or report.setup_status is not None:
            rows = []
            for condition in report.preconditions:
                rows.append(
                    "<li>"
                    + (presentation_badge(condition.status) + " " if condition.status else "")
                    + esc(condition.description)
                    + (f'<p class="muted">{esc(condition.error)}</p>' if condition.error else "")
                    + "</li>"
                )
            status = report.setup_status
            summary = presentation_badge(status) if status else '<span class="muted">Not recorded</span>'
            no_execution = (
                '<div class="notice danger">Setup did not succeed; product execution did not begin.</div>'
                if status and status != "SUCCEEDED"
                else ""
            )
            setup_html = (
                '<section class="panel"><h2>Preconditions and setup</h2>'
                f'<p>Setup: {summary}</p>{no_execution}'
                + ("<ul>" + "".join(rows) + "</ul>" if rows else '<p class="muted">No preconditions recorded.</p>')
                + "</section>"
            )

        cleanup_html = ""
        if report.cleanup_succeeded is not None:
            cleanup_key = "SUCCEEDED" if report.cleanup_succeeded else "CLEANUP_FAILURE"
            failures = "".join(
                f'<div class="notice warning"><strong>{esc(item.get("label", "Cleanup"))}:</strong> '
                f'{esc(item.get("message", "Cleanup failed."))}</div>'
                for item in report.cleanup_failures
            )
            cleanup_html = (
                '<section class="panel"><h2>Cleanup</h2>'
                + presentation_badge(cleanup_key)
                + ("<p>Cleanup completed successfully.</p>" if report.cleanup_succeeded else "<p>Cleanup had failures; the primary run result remains shown above.</p>")
                + failures
                + "</section>"
            )

        result_heading = presentation_badge(report.status.value)
        outcome_badge = presentation_badge(report.outcome, outcome_tone(report.outcome))
        outcome_text = special_outcomes.get(report.outcome)
        if not outcome_text and report.cleanup_succeeded is False:
            outcome_text = "Cleanup failed after the primary test result was recorded."
        actions = [
            f'<a class="button" href="/test-cases/{esc(report.test_case_id)}">Back to TestCase</a>',
            f'<a class="button" href="/runs/{esc(report.run_id)}/report.json">View JSON</a>',
        ]
        if show_html_report_link:
            actions.append(
                f'<a class="button" href="/runs/{esc(report.run_id)}/report.html" '
                'target="_blank" rel="noopener">View HTML Report</a>'
            )
        if run_details_url is not None:
            actions.insert(
                0,
                f'<a class="button" href="{esc(run_details_url)}">Back to Run</a>',
            )
        summary = (
            '<section class="panel"><div class="status-line">'
            + result_heading
            + presentation_badge(report.workflow_type.value, "workflow")
            + outcome_badge
            + "</div>"
            + (f'<div class="notice {"danger" if report.outcome == "PRODUCT_FAILURE" else "warning" if report.outcome in {"SETUP_FAILURE", "CLEANUP_FAILURE"} else ""}">{esc(outcome_text)}</div>' if outcome_text else "")
            + '<div class="meta-grid">'
            + _report_meta("Started", format_timestamp(report.started_at))
            + _report_meta("Finished", format_timestamp(report.finished_at))
            + _report_meta("Duration", format_duration(report.duration_ms))
            + _report_meta("Run", report.run_public_id or "Unknown")
            + _report_meta("Base URL", report.base_url or "Unavailable")
            + _report_meta("TestCase", report.test_case_public_id or "Unknown")
            + "</div></section>"
        )
        technical_ids = (
            '<details class="technical-details"><summary>Technical IDs</summary>'
            f'<p>Run UUID: <code>{esc(report.run_id)}</code></p>'
            f'<p>TestCase UUID: <code>{esc(report.test_case_id)}</code></p>'
            '</details>'
        )
        navigation = (
            '<header class="topbar"><div class="shell topbar-inner">'
            '<a class="brand" href="/">AI QA Agent</a><nav class="nav-links" aria-label="Main navigation">'
            '<a href="/">Dashboard</a><a href="/test-cases">Test Cases</a><a href="/runs">Runs</a>'
            "</nav></div></header>"
        )
        title = f"{esc(report.test_case_name)} — Run report"
        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{title}</title><style>{UI_CSS}</style></head><body>"
            + navigation
            + '<main class="shell main">'
            + f'<p class="breadcrumbs"><a href="/">Dashboard</a><span>›</span>'
            f'<a href="/test-cases/{esc(report.test_case_id)}">'
            f'{esc(report.test_case_public_id or report.test_case_name)} — {esc(report.test_case_name)}</a>'
            f'<span>›</span>{esc(report.run_public_id or "Run")}</p>'
            + '<header class="page-heading"><h1>' + esc(report.test_case_name) + '</h1>'
            + f'<p class="lead">{esc(report.test_case_description)}</p></header>'
            + '<div class="actions"><a class="button" href="/">Back to Dashboard</a>'
            + "".join(actions)
            + "</div>"
            + summary
            + technical_ids
            + setup_html
            + '<section class="panel"><h2>Test steps</h2>'
            + ("".join(steps_html) if steps_html else '<div class="empty-state">No steps recorded.</div>')
            + "</section>"
            + cleanup_html
            + '</main><footer class="shell">AI QA Agent · Local run report</footer></body></html>'
        )

    def write_json(self, report: RunReport, path: str | Path) -> None:
        Path(path).write_text(report.to_json(), encoding="utf-8")

    def write_html(self, report: RunReport, path: str | Path) -> None:
        Path(path).write_text(self.to_html(report), encoding="utf-8")


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _plan_origin_label(origin: PlanVersionOrigin | None) -> str:
    return {
        PlanVersionOrigin.AI_GENERATED: "Generated by AI",
        PlanVersionOrigin.HUMAN_EDITED: "Edited locally",
        PlanVersionOrigin.REGENERATED: "Regenerated by AI",
        PlanVersionOrigin.REPAIRED: "Updated automatically",
    }.get(origin, "Unknown origin")


def _plan_version_label(attempt: RunAttemptReport) -> str:
    version = (
        f"v{attempt.plan_version_number}"
        if attempt.plan_version_number is not None
        else "Version unknown"
    )
    return f"{version} · {_plan_origin_label(attempt.plan_version_origin)}"


def _report_meta(label: str, value: object, *, raw: bool = False) -> str:
    shown = str(value) if raw else escape_html(value)
    return (
        f'<div class="meta-item"><span class="meta-label">{escape_html(label)}</span>'
        f'<span class="meta-value">{shown}</span></div>'
    )


def _safe_report_value(value: Any, test_run: TestRun) -> Any:
    if isinstance(value, str):
        safe_value = value
        for secret in _run_context_secrets(test_run):
            safe_value = safe_value.replace(secret, "[REDACTED]")
        return redact_secrets(safe_value)
    if isinstance(value, dict):
        return {
            str(_safe_report_value(key, test_run)): _safe_report_value(child, test_run)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_report_value(child, test_run) for child in value]
    return value


def _run_context_secrets(test_run: TestRun) -> list[str]:
    found: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value:
                found.add(value)
        elif isinstance(value, bytes):
            if value:
                found.add(value.decode("utf-8", errors="replace"))
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(key)
                collect(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                collect(item)
        elif value is not None:
            found.add(str(value))

    for item in test_run.run_context.values.values():
        if item.sensitive:
            collect(item.value)
    return sorted(found, key=len, reverse=True)
