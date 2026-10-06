"""Deterministic JSON reports derived from domain TestRun data."""

from pathlib import Path
from datetime import datetime
from html import escape
from typing import Any
from uuid import UUID
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.models import Evidence, EvidenceType, Execution, ExecutionStatus, TestRun
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
    test_case_id: UUID
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
            test_case_id=record.test_case_id,
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
        evidence_url: Callable[[RunStepReport, RunAttemptReport, RunEvidenceReport, int], str] | None = None,
    ) -> str:
        """Render a standalone document; evidence links are supplied by a safe host."""
        esc = lambda value: escape(str(value), quote=True)
        badge_status = esc(report.status.value)
        steps_html = []
        for step in report.steps:
            attempts_html = []
            for attempt in step.attempts:
                evidence_html = []
                for evidence_index, evidence in enumerate(attempt.evidence):
                    name = esc(evidence.name)
                    if evidence_url is None:
                        evidence_html.append(f"<li>{name}</li>")
                    else:
                        url = esc(evidence_url(step, attempt, evidence, evidence_index))
                        evidence_html.append(
                            f'<li><a href="{url}"><img class="evidence" src="{url}" '
                            f'alt="{name}"><br>{name}</a></li>'
                        )
                attempt_detail = []
                if attempt.error:
                    attempt_detail.append(f"<p><strong>Error:</strong> {esc(attempt.error)}</p>")
                if attempt.actual_result is not None:
                    attempt_detail.append(
                        f"<p><strong>Actual:</strong> {esc(_display_value(attempt.actual_result))}</p>"
                    )
                if attempt.evidence:
                    attempt_detail.append(
                        "<p><strong>Evidence</strong></p><ul>" + "".join(evidence_html) + "</ul>"
                    )
                attempts_html.append(
                    f'<div class="attempt"><strong>Attempt {esc(attempt.status.value)}</strong> '
                    f'<span class="muted">Plan version {esc(attempt.test_plan_version_id)}</span>'
                    + "".join(attempt_detail)
                    + "</div>"
                )
            steps_html.append(
                f'<article class="step"><h3><span class="badge {esc(step.status.lower())}">'
                f'{esc(step.status)}</span> Step {step.order} — {esc(step.name)}</h3>'
                f'<p>{esc(step.description)}</p><p><strong>Expected:</strong> {esc(step.expected)}</p>'
                + ("".join(attempts_html) if attempts_html else '<p class="muted">No execution attempt.</p>')
                + "</article>"
            )

        preconditions_html = ""
        if report.preconditions or report.setup_status is not None:
            preconditions_html = "<section><h2>Initial conditions / Setup</h2><ul>" + "".join(
                f"<li><strong>{esc(item.status or 'NOT_RUN')}</strong> — "
                f"{esc(item.description)}"
                + (f"<p>{esc(item.error)}</p>" if item.error else "")
                + "</li>"
                for item in report.preconditions
            ) + "</ul>"
            if report.setup_status:
                preconditions_html += f"<p>Setup: {esc(report.setup_status)}</p>"
            preconditions_html += "</section>"

        cleanup_html = ""
        if report.cleanup_succeeded is not None:
            cleanup_status = "SUCCEEDED" if report.cleanup_succeeded else "FAILED"
            cleanup_html = (
                f"<section><h2>Cleanup</h2><p>{cleanup_status}</p>"
                + "".join(
                    f"<p>{esc(item.get('label', 'cleanup'))}: "
                    f"{esc(item.get('message', 'Cleanup failed.'))}</p>"
                    for item in report.cleanup_failures
                )
                + "</section>"
            )
        duration = (
            f"{report.duration_ms} ms" if report.duration_ms is not None else "Unavailable"
        )
        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>AI QA Agent Report — {esc(report.test_case_name)}</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#1f2937}"
            "h1,h2{color:#111827}.summary,.step{border:1px solid #d1d5db;border-radius:8px;padding:1rem;margin:1rem 0}"
            ".badge{display:inline-block;border-radius:999px;padding:.2rem .65rem;background:#e5e7eb;font-size:.85rem}"
            ".passed{background:#dcfce7;color:#166534}.failed,.product_failure{background:#fee2e2;color:#991b1b}"
            ".blocked{background:#fef3c7;color:#92400e}.automation_drift,.infrastructure_error{background:#ffedd5;color:#9a3412}"
            ".attempt{border-left:3px solid #cbd5e1;margin:1rem 0;padding:.5rem 1rem}.muted{color:#6b7280;font-size:.9rem}"
            ".evidence{max-width:360px;max-height:240px;object-fit:contain;border:1px solid #d1d5db}"
            "code{overflow-wrap:anywhere}</style></head><body>"
            f"<h1>AI QA Agent Test Report</h1><section class=\"summary\"><h2>"
            f"{esc(report.test_case_name)}</h2><p>{esc(report.test_case_description)}</p>"
            f"<p>Workflow: <strong>{esc(report.workflow_type.value)}</strong></p>"
            f"<p>Run: <code>{esc(report.run_id)}</code></p>"
            f"<p>TestCase ID: <code>{esc(report.test_case_id)}</code></p>"
            + (f"<p>Base URL: {esc(report.base_url)}</p>" if report.base_url else "")
            + (f"<p>Setup: {esc(report.setup_status)}</p>" if report.setup_status else "")
            + f"<p>Overall: <span class=\"badge {badge_status.lower()}\">{badge_status}</span>"
            f" — {esc(report.outcome)}</p><p>Started: {esc(report.started_at.isoformat())}</p>"
            f"<p>Finished: {esc(report.finished_at.isoformat() if report.finished_at else 'Unavailable')}</p>"
            f"<p>Duration: {esc(duration)}</p></section>"
            + preconditions_html
            + "<section><h2>Test Steps</h2>" + "".join(steps_html) + "</section>"
            + cleanup_html
            + f"<section><h2>Final Result</h2><p>{esc(report.outcome)} ({badge_status})</p></section>"
            + "</body></html>"
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
