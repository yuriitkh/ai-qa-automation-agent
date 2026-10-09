"""Deterministic JSON reports derived from domain TestRun data."""

import json
from pathlib import Path
from datetime import datetime
from typing import Any
from uuid import UUID
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.cookie_consent import (
    CookieConsentRecord,
    cookie_consent_label,
    cookie_consent_policy_label,
    cookie_consent_reason_label,
)
from qa_agent.evidence_policy import (
    DEFAULT_EVIDENCE_POLICY,
    EvidencePolicy,
    EvidenceScope,
    evidence_mode_label,
    screenshot_mode_label,
)
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
    result_badge,
    safe_local_url,
)
from qa_agent.redaction import redact_secrets, redact_diagnostic
from qa_agent.result_semantics import (
    execution_classification, history_outcome, result_outcome, result_label,
    result_summary,
)
from qa_agent.run_history import (
    HistoryExecutionReference,
    HistoryPrecondition,
    RunHistoryDetail,
    RunHistoryRecord,
    WorkflowType,
)
from qa_agent.execution_diagnostics import ActionFailureDiagnostic, action_failure_diagnostic, redact_action_failure
from qa_agent.setup_orchestration import CleanupOutcome, SetupRunOutcome


class EvidenceReport(BaseModel):
    id: UUID
    execution_id: UUID
    type: EvidenceType
    path: str
    description: str | None = None
    timestamp: datetime | None = None
    scope: EvidenceScope | None = None
    event: str | None = None

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "EvidenceReport":
        return cls(
            id=evidence.id,
            execution_id=evidence.execution_id,
            type=evidence.type,
            path=evidence.path,
            description=evidence.description,
            timestamp=evidence.timestamp,
            scope=evidence.scope,
            event=evidence.event,
        )


class TestAttemptReport(BaseModel):
    execution_id: UUID
    test_plan_version_id: UUID
    status: ExecutionStatus
    classification: str = "INCONCLUSIVE"
    display_status: str = "Inconclusive"
    started_at: datetime
    finished_at: datetime | None
    actual_result: Any
    error: str | None
    action_failure: ActionFailureDiagnostic | None = None
    evidence: list[EvidenceReport] = Field(default_factory=list)

    @classmethod
    def from_execution(cls, execution: Execution) -> "TestAttemptReport":
        return cls(
            execution_id=execution.id,
            test_plan_version_id=execution.test_plan_version_id,
            status=execution.status,
            classification=result_outcome(execution.status, execution_classification(execution), complete=execution.finished_at is not None),
            display_status=result_label(result_outcome(execution.status, execution_classification(execution), complete=execution.finished_at is not None)),
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            actual_result=execution.actual_result,
            error=execution.error,
            action_failure=action_failure_diagnostic(execution.runner_result),
            evidence=[EvidenceReport.from_evidence(item) for item in execution.evidence],
        )


class TestStepReport(BaseModel):
    step_id: UUID
    name: str
    status: ExecutionStatus | None
    classification: str = "INCONCLUSIVE"
    display_status: str = "Inconclusive"
    attempts: list[TestAttemptReport]


class TestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID
    test_case_id: UUID
    status: ExecutionStatus
    display_outcome: str = "INCONCLUSIVE"
    display_status: str = "Inconclusive"
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
            classification = (
                "BLOCKED" if step_id in blocked_ids
                else result_outcome(attempts[-1].status, execution_classification(attempts[-1]), complete=attempts[-1].finished_at is not None) if attempts
                else "NOT_ATTEMPTED" if step_id in test_run.not_attempted_step_ids else "NOT_RUN"
            )
            steps.append(
                TestStepReport(
                    step_id=step_id,
                    name=name,
                    classification=classification,
                    display_status=result_label(classification),
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

        outcomes = {step.classification for step in steps}
        outcome = next((item for item in (
            "INFRASTRUCTURE_ERROR", "AUTOMATION_DRIFT", "AUTOMATION_EXECUTION_ERROR",
            "PRODUCT_FAILURE", "BLOCKED", "INCONCLUSIVE", "NOT_RUN",
        ) if item in outcomes), "PASSED" if outcomes == {"PASSED"} else "INCONCLUSIVE")
        if test_run.cancelled:
            outcome = "CANCELLED"
        return TestReport(
            run_id=test_run.id,
            test_case_id=test_run.test_case_id,
            status=test_run.overall_status,
            display_outcome=outcome,
            display_status=result_label(outcome),
            started_at=test_run.started_at,
            finished_at=test_run.finished_at,
            steps=steps,
        )

    @staticmethod
    def _safe_attempt(execution: Execution, test_run: TestRun) -> TestAttemptReport:
        attempt = TestAttemptReport.from_execution(execution)
        return attempt.model_copy(update={
            "actual_result": _safe_report_value(attempt.actual_result, test_run),
            "error": _redact_report_value(_safe_report_value(attempt.error, test_run)),
            "action_failure": redact_action_failure(attempt.action_failure, lambda value: _safe_report_value(value, test_run)),
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
    scope: EvidenceScope | None = None
    event: str | None = None


class RunAttemptReport(BaseModel):
    execution_id: UUID
    test_plan_version_id: UUID
    plan_version_number: int | None = None
    plan_version_origin: PlanVersionOrigin | None = None
    plan_url: str | None = None
    status: ExecutionStatus
    classification: str = "INCONCLUSIVE"
    display_status: str = "Inconclusive"
    observation: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    actual_result: Any = None
    error: str | None = None
    diagnostics: str | None = None
    action_failure: ActionFailureDiagnostic | None = None
    evidence: list[RunEvidenceReport] = Field(default_factory=list)


class RunStepReport(BaseModel):
    step_id: UUID
    order: int
    name: str
    description: str
    expected: str
    status: str
    classification: str = "INCONCLUSIVE"
    display_status: str = "Inconclusive"
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
    evidence_policy: EvidencePolicy | None = None
    cookie_consent: CookieConsentRecord | None = None
    outcome: str | None = None
    status: ExecutionStatus
    display_outcome: str = "INCONCLUSIVE"
    display_status: str = "Inconclusive"
    result_summary: dict[str, Any] = Field(default_factory=dict)
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
        public_outcome = history_outcome(record)
        failed_references = [item for item in record.executions if item.status == ExecutionStatus.FAILED]
        legacy_classification = (
            public_outcome if len(failed_references) == 1 and public_outcome in {
                "PRODUCT_FAILURE", "AUTOMATION_EXECUTION_ERROR", "AUTOMATION_DRIFT", "INFRASTRUCTURE_ERROR",
            } else None
        )
        for step in sorted(record.steps, key=lambda item: item.order):
            attempts = [
                self._attempt_report(reference, execution_map.get(reference.execution_id), legacy_classification)
                for reference in references_by_step.get(step.id, [])
            ]
            classification = (
                "NOT_ATTEMPTED" if step.status == ExecutionStatus.NOT_ATTEMPTED
                else "BLOCKED" if step.status == ExecutionStatus.BLOCKED
                else attempts[-1].classification if attempts
                else "NOT_RUN"
            )
            step_reports.append(RunStepReport(
                step_id=step.id,
                order=step.order,
                name=step.name,
                description=step.description,
                expected=step.expected,
                status=(step.status.value if step.status is not None else "NOT_RUN"),
                attempts=attempts,
                classification=classification,
                display_status=result_label(classification),
            ))

        return RunReport(
            run_id=record.run_id,
            run_public_id=record.public_id,
            test_case_id=record.test_case_id,
            test_case_public_id=record.test_case_public_id,
            test_case_name=record.test_case_name,
            test_case_description=record.test_case_description,
            workflow_type=record.workflow_type,
            evidence_policy=record.evidence_policy,
            cookie_consent=record.cookie_consent,
            outcome=record.outcome,
            status=record.status,
            display_outcome=public_outcome,
            display_status=result_label(public_outcome),
            result_summary=result_summary(public_outcome, [
                {"name": step.name, "outcome": step.classification} for step in step_reports
            ]),
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
        legacy_classification: str | None = None,
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
                scope=item.scope,
                event=item.event,
            )
            for item in reference.evidence
        ]
        classification = result_outcome(reference.status, (
            reference.classification
            or (execution_classification(execution) if execution is not None else None)
            or ("PASSED" if reference.status == ExecutionStatus.PASSED else legacy_classification)
        ), complete=reference.finished_at is not None)
        observation = _actual_observation(reference.safe_actual_result, reference.safe_diagnostics or reference.safe_error)
        return RunAttemptReport(
            execution_id=reference.execution_id,
            test_plan_version_id=reference.test_plan_version_id,
            plan_version_number=reference.plan_version_number,
            plan_version_origin=reference.plan_version_origin,
            status=reference.status,
            classification=classification,
            display_status=result_label(classification),
            observation=observation,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            actual_result=_redact_report_value(reference.safe_actual_result),
            error=redact_diagnostic(reference.safe_error) if reference.safe_error else None,
            diagnostics=redact_diagnostic(reference.safe_diagnostics or reference.safe_error) if reference.safe_diagnostics or reference.safe_error else None,
            action_failure=reference.action_failure,
            evidence=evidence,
        )

    def to_html(
        self,
        report: RunReport,
        *,
        evidence_url: Callable[
            [RunStepReport, RunAttemptReport, RunEvidenceReport, int], str | None
        ] | None = None,
        plan_url: Callable[[RunStepReport, RunAttemptReport], str | None] | None = None,
        show_html_report_link: bool = True,
        run_details_url: str | None = None,
        report_view: bool = True,
    ) -> str:
        """Render shared result content with explicit report or Run Details navigation."""
        esc = escape_html
        steps_html = []
        for number, step in enumerate(report.steps, start=1):
            attempts_html = []
            for attempt_number, attempt in enumerate(step.attempts, start=1):
                observed = attempt.observation or (
                    "Expected result could not be verified."
                    if attempt.classification != "PASSED" else "The recorded verification passed."
                )
                url = safe_local_url(plan_url(step, attempt) if plan_url else attempt.plan_url)
                version = esc(_plan_version_label(attempt))
                version_html = (
                    f'<a href="{esc(url)}">Plan version {version}</a>' if url
                    else '<span class="muted">No saved TestPlan available.</span>'
                )
                evidence_items = []
                for index, evidence in enumerate(attempt.evidence):
                    url = safe_local_url(evidence_url(step, attempt, evidence, index) if evidence_url else None)
                    label = esc(_evidence_label(evidence))
                    name = esc(evidence.name)
                    description = esc(redact_diagnostic(evidence.description or ""))
                    if url:
                        evidence_items.append(
                            f'<figure><a href="{esc(url)}" target="_blank" rel="noopener">'
                            f'<img class="evidence-preview" src="{esc(url)}" alt="{label}: {name}">'
                            f'Open full-size evidence: {name}</a>'
                            f'<figcaption>{label}' + (f' · {description}' if description else '') + '</figcaption></figure>'
                        )
                    else:
                        evidence_items.append(f'<p class="muted">{label}: {name} · image link unavailable.</p>')
                raw = (
                    (f'<p>{esc(attempt.error)}</p>' if attempt.error else '')
                    + f'<pre>{esc(attempt.diagnostics)}</pre>' if attempt.diagnostics else '<p>No error details recorded.</p>'
                )
                attempts_html.append(
                    f'<section class="detail-box" data-execution-id="{attempt.execution_id}">'
                    '<div class="status-line">' + result_badge(attempt.classification)
                    + f'<strong>Attempt {attempt_number}</strong></div>'
                    + f'<p><strong>Actual observation:</strong> {esc(observed)}</p>'
                    + (f'<p class="action-diagnostic">{esc(attempt.action_failure.summary())}</p>' if attempt.action_failure else '')
                    + (f'<pre>{esc(json.dumps(attempt.action_failure.model_dump(mode="json", exclude_none=True), indent=2))}</pre>' if attempt.action_failure else '')
                    + f'<p>{version_html}</p>'
                    + (f'<p class="muted">Attempt duration: {esc(format_duration(attempt.duration_ms))}</p>' if attempt.duration_ms is not None else '')
                    + '<div class="attempt-evidence"><h4>Evidence</h4>'
                    + (''.join(evidence_items) or '<p class="muted">No evidence available.</p>') + '</div>'
                    + '<details class="technical-details"><summary>Developer details</summary>'
                    + f'<p>{esc(format_timestamp(attempt.started_at))} · {esc(attempt.classification)}</p>'
                    + f'<p>Plan version: <code>{esc(attempt.test_plan_version_id)}</code> · {version}</p>'
                    + raw + '</details></section>'
                )
            if step.classification == "BLOCKED":
                detail = '<div class="notice warning">Not executed because a prerequisite or preceding step blocked continuation.</div>'
            elif not step.attempts:
                detail = '<p class="muted">No execution attempt was recorded. No saved TestPlan available.</p>'
            else:
                detail = ''.join(attempts_html)
            tone = {"PASSED": "passed", "PRODUCT_FAILURE": "failed", "BLOCKED": "blocked"}.get(step.classification, "")
            steps_html.append(
                f'<article class="step-card {tone}"><div class="step-heading">'
                + result_badge(step.classification) + f'<h3>Step {number} · {esc(step.name)}</h3></div>'
                + (f'<p class="muted">{esc(step.description)}</p>' if step.description else '')
                + f'<p><strong>Expected:</strong> {esc(step.expected)}</p>' + detail + '</article>'
            )
        summary = report.result_summary
        issue = report.steps[summary["stopping_step"] - 1] if summary.get("stopping_step") else None
        observation = issue.attempts[-1].observation if issue and issue.attempts else None
        summary_html = (
            '<section class="panel result-summary"><h2>Result summary</h2>'
            + result_badge(report.display_outcome)
            + f'<p>{summary.get("completed_steps", 0)} of {len(report.steps)} steps verified.</p>'
            + (f'<p>{esc(summary["stopping_detail"])}</p>' if summary.get("stopping_detail") else '')
            + f'<p>{esc(summary.get("explanation", ""))}</p>'
            + (f'<p><strong>Observed:</strong> {esc(observation)}</p>' if observation else '')
            + f'<p><strong>Recommended action:</strong> {esc(summary.get("recommended_action", ""))}</p>'
            + (
                '<a class="button" href="/system/health">Open System Health</a>'
                if report.display_outcome == "INFRASTRUCTURE_ERROR" else ''
            )
            + '</section>'
        )
        metadata = (
            '<section class="panel"><h2>TestCase and workflow</h2><div class="meta-grid">'
            + _report_meta("Run", report.run_public_id or "Unknown")
            + _report_meta("TestCase", report.test_case_public_id or "Unknown")
            + _report_meta("Workflow", report.workflow_type.value)
            + _report_meta("Started", format_timestamp(report.started_at))
            + _report_meta("Finished", format_timestamp(report.finished_at))
            + _report_meta("Duration", format_duration(report.duration_ms))
            + _report_meta("Base URL", report.base_url or "Unavailable")
            + _report_meta("Evidence mode", evidence_mode_label((report.evidence_policy or DEFAULT_EVIDENCE_POLICY).mode))
            + _report_meta("Screenshot scope", screenshot_mode_label((report.evidence_policy or DEFAULT_EVIDENCE_POLICY).screenshot_mode))
            + '</div></section>'
        )
        setup_rows = ''.join(
            '<li>' + (presentation_badge(condition.status) + ' ' if condition.status else '')
            + esc(condition.description)
            + (f'<p>{esc(redact_diagnostic(condition.error))}</p>' if condition.error else '') + '</li>'
            for condition in report.preconditions
        )
        if report.cookie_consent:
            consent = report.cookie_consent
            setup_rows += '<li>Cookie consent: ' + esc(cookie_consent_policy_label(consent.policy)) + ' · ' + esc(cookie_consent_label(consent)) + ' · ' + esc(cookie_consent_reason_label(consent.reason)) + '</li>'
        setup_html = (
            '<section class="panel"><h2>Preconditions and setup</h2>'
            + (f'<p>Setup: {presentation_badge(report.setup_status)}</p>' if report.setup_status else '')
            + ('<p>Setup did not succeed; product execution did not begin.</p>' if report.setup_status and report.setup_status != 'SUCCEEDED' else '')
            + f'<ul>{setup_rows}</ul></section>'
            if setup_rows or report.setup_status else ''
        )
        cleanup_html = (
            '<section class="panel"><h2>Cleanup</h2>'
            + ('<p>Cleanup completed successfully.</p>' if report.cleanup_succeeded else '<p>Cleanup had failures; the primary run result remains shown above.</p>')
            + ''.join(f'<p>{esc(item.get("label", "Cleanup"))}: {esc(redact_diagnostic(item.get("message", "")))}</p>' for item in report.cleanup_failures)
            + '</section>' if report.cleanup_succeeded is not None else ''
        )
        actions = [
            f'<a class="button" href="/test-cases/{report.test_case_id}">Back to TestCase</a>',
            f'<a class="button" href="/runs/{report.run_id}/report.json">View JSON</a>',
        ]
        if not report_view and show_html_report_link:
            actions.append(f'<a class="button" href="/runs/{report.run_id}/report.html">View HTML Report</a>')
        if report_view:
            actions.insert(0, f'<a class="button" href="{esc(safe_local_url(run_details_url) or f"/runs/{report.run_id}")}">Back to Run Details</a>')
        heading = 'HTML Test Report' if report_view else 'Run Details'
        technical = (
            '<details class="panel technical-details"><summary>Developer details · Technical IDs</summary>'
            f'<p>Run UUID: <code>{report.run_id}</code></p><p>TestCase UUID: <code>{report.test_case_id}</code></p>'
            f'<p>Internal execution status: {esc(report.status.value)} · Stored outcome: {esc(report.outcome or "Unknown")}</p>'
            '</details>'
        )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{heading} · {esc(report.run_public_id or "Run")} · {esc(report.test_case_name)}</title>'
            f'<style>{UI_CSS}</style></head><body>'
            '<header class="topbar"><div class="shell topbar-inner"><a class="brand" href="/">AI QA Agent</a>'
            '<nav class="nav-links" aria-label="Main navigation"><a href="/">Dashboard</a><a href="/test-cases">Test Cases</a><a href="/runs">Runs</a></nav></div></header>'
            '<main class="shell main"><header class="page-heading">'
            f'<p class="eyebrow">{heading} · {esc(report.run_public_id or "Run")}</p>'
            f'<h1>{heading}</h1><p class="lead">{esc(report.test_case_name)}</p>'
            f'<p>{esc(report.test_case_description)}</p></header>'
            + '<div class="actions">' + ''.join(actions) + '</div>'
            + summary_html + metadata + setup_html
            + '<section class="panel"><h2>Steps and attempts</h2>'
            + (''.join(steps_html) or '<p>No steps recorded.</p>') + '</section>'
            + cleanup_html + technical
            + f'</main><footer class="shell">AI QA Agent · {heading}</footer></body></html>'
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


def _redact_report_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_diagnostic(value)
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).casefold() in {
                "password", "token", "access_token", "api_key", "authorization", "secret",
            } else _redact_report_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_report_value(child) for child in value]
    return value


def _actual_observation(value: Any, error: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or value.casefold() not in {
        "passed", "failed", "running", "pending", "blocked", "",
    }):
        return _display_value(_redact_report_value(value))
    # Only accept one complete, explicit Playwright actual-value header. Call
    # logs and incomplete error fragments are not observations of application state.
    import re

    matches = re.findall(r"(?m)^Actual value: ([^\r\n]+)$", (error or "").replace("\r\n", "\n"))
    if len(matches) == 1 and matches[0].strip() not in {"None", "null", "undefined"}:
        return redact_diagnostic(matches[0].strip())
    return None


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


def _evidence_label(evidence: RunEvidenceReport) -> str:
    scope = {
        EvidenceScope.ELEMENT: "Element",
        EvidenceScope.PAGE: "Page",
    }.get(evidence.scope, "Recorded")
    event = (evidence.event or "").split(":", 1)[0]
    if event == "FAILURE":
        return f"Failure · {scope} screenshot"
    if event == "VERIFICATION":
        return f"Verification · {scope} screenshot"
    if event == "TEST_STEP":
        return f"TestStep · {scope} screenshot"
    return f"{scope} screenshot"


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
