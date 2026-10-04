"""Deterministic JSON reports derived from domain TestRun data."""

from pathlib import Path
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from qa_agent.models import Evidence, EvidenceType, Execution, ExecutionStatus, TestRun


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

        names = test_run.test_step_names or ["" for _ in test_run.test_step_ids]
        steps = []
        for step_id, name in zip(test_run.test_step_ids, names):
            attempts = executions_by_step[step_id]
            steps.append(
                TestStepReport(
                    step_id=step_id,
                    name=name,
                    status=attempts[-1].status if attempts else None,
                    attempts=[TestAttemptReport.from_execution(item) for item in attempts],
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

    def write_json(self, report: TestReport, path: str | Path) -> None:
        Path(path).write_text(report.to_json(), encoding="utf-8")
