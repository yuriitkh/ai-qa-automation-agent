import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from qa_agent.models import (
    Evidence,
    EvidenceType,
    Execution,
    ExecutionStatus,
    TestCase as DomainTestCase,
    TestRun as DomainTestRun,
    TestStep as DomainTestStep,
)
from qa_agent.reporting import TestReport as DomainTestReport, TestReportGenerator


class TestReportGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.steps = [self.make_step("Third", 2), self.make_step("First", 0), self.make_step("Second", 1)]
        self.case = DomainTestCase(
            name="Ordered checks",
            description="Check stable reporting order.",
            steps=self.steps,
        )
        self.generator = TestReportGenerator()

    @staticmethod
    def make_step(name: str, order: int) -> DomainTestStep:
        return DomainTestStep(
            name=name,
            description=f"Check {name}.",
            expected="Pass",
            order=order,
        )

    def execution(
        self,
        step: DomainTestStep,
        status: ExecutionStatus,
        offset: int,
        *,
        actual_result: object | None = None,
        error: str | None = None,
    ) -> Execution:
        when = self.started_at + timedelta(seconds=offset)
        return Execution(
            test_step_id=step.id,
            test_plan_version_id=uuid4(),
            status=status,
            started_at=when,
            finished_at=when + timedelta(seconds=1),
            actual_result=(actual_result if actual_result is not None else status.value),
            error=error,
        )

    def evidence(
        self,
        execution_id,
        path: str,
        *,
        description: str | None = None,
        timestamp: datetime | None = None,
        evidence_type: EvidenceType = EvidenceType.SCREENSHOT,
    ) -> Evidence:
        return Evidence(
            execution_id=execution_id,
            type=evidence_type,
            path=path,
            description=description,
            timestamp=timestamp,
        )

    def test_pass_report_has_valid_json_and_round_trips(self) -> None:
        first, second = sorted(self.steps, key=lambda step: step.order)[:2]
        run = DomainTestRun.from_test_case(
            self.case,
            [
                self.execution(first, ExecutionStatus.PASSED, 0),
                self.execution(second, ExecutionStatus.PASSED, 2),
                self.execution(sorted(self.steps, key=lambda step: step.order)[2], ExecutionStatus.PASSED, 4),
            ],
        )

        report = self.generator.generate(run)
        serialized = report.to_json()
        parsed = json.loads(serialized)
        restored = DomainTestReport.model_validate_json(serialized)

        self.assertEqual(parsed["status"], "PASSED")
        self.assertEqual([step["name"] for step in parsed["steps"]], ["First", "Second", "Third"])
        self.assertEqual([step["status"] for step in parsed["steps"]], ["PASSED"] * 3)
        self.assertEqual([len(step["attempts"]) for step in parsed["steps"]], [1, 1, 1])
        self.assertEqual(restored, report)
        self.assertEqual(parsed["steps"][0]["attempts"][0]["started_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(parsed["steps"][0]["attempts"][0]["evidence"], [])

    def test_attempt_serializes_evidence_fields_in_execution_order(self) -> None:
        step = min(self.steps, key=lambda item: item.order)
        execution_id = uuid4()
        evidence_items = (
            self.evidence(
                execution_id,
                "artifacts/second.png",
                description="Second capture",
                timestamp=self.started_at + timedelta(seconds=5),
            ),
            self.evidence(
                execution_id,
                "artifacts/dom.json",
                description=None,
                evidence_type=EvidenceType.DOM_SNAPSHOT,
            ),
        )
        execution = Execution(
            id=execution_id,
            test_step_id=step.id,
            test_plan_version_id=uuid4(),
            status=ExecutionStatus.FAILED,
            started_at=self.started_at,
            finished_at=self.started_at + timedelta(seconds=1),
            actual_result="failed",
            error="assertion failed",
            evidence=evidence_items,
        )
        run = DomainTestRun.from_test_case(self.case, [execution])

        report = self.generator.generate(run)
        serialized = report.to_json()
        parsed = json.loads(serialized)
        attempt = parsed["steps"][0]["attempts"][0]
        evidence = attempt["evidence"]

        self.assertEqual([item["path"] for item in evidence], [
            "artifacts/second.png", "artifacts/dom.json"
        ])
        self.assertEqual(evidence[0]["id"], str(evidence_items[0].id))
        self.assertEqual(evidence[0]["execution_id"], str(execution_id))
        self.assertEqual(evidence[0]["type"], "SCREENSHOT")
        self.assertEqual(evidence[0]["description"], "Second capture")
        self.assertEqual(evidence[0]["timestamp"], "2026-01-01T00:00:05Z")
        self.assertIsNone(evidence[1]["description"])
        self.assertIsNone(evidence[1]["timestamp"])
        self.assertEqual(DomainTestReport.model_validate_json(serialized), report)

    def test_failed_report_preserves_raw_error_and_actual_result(self) -> None:
        first, second, third = sorted(self.steps, key=lambda step: step.order)
        failure = self.execution(
            second,
            ExecutionStatus.FAILED,
            2,
            actual_result={"assertion": "heading missing"},
            error="raw runner error: heading missing",
        )
        run = DomainTestRun.from_test_case(
            self.case,
            [
                self.execution(first, ExecutionStatus.PASSED, 0),
                failure,
                self.execution(third, ExecutionStatus.PASSED, 4),
            ],
        )

        report = self.generator.generate(run)
        failed_step = report.steps[1]

        self.assertEqual(report.status, ExecutionStatus.FAILED)
        self.assertEqual(failed_step.status, ExecutionStatus.FAILED)
        self.assertEqual(failed_step.attempts[0].error, "raw runner error: heading missing")
        self.assertEqual(failed_step.attempts[0].actual_result, {"assertion": "heading missing"})

    def test_stale_retry_retains_both_attempts_in_execution_order(self) -> None:
        first, second = sorted(self.steps, key=lambda step: step.order)[:2]
        stale = self.execution(first, ExecutionStatus.FAILED, 0, error="stale selector")
        stale.evidence = (self.evidence(
            stale.id, "artifacts/stale.png", description="Before plan retry"
        ),)
        retry = self.execution(first, ExecutionStatus.PASSED, 2)
        run = DomainTestRun.from_test_case(
            self.case,
            [
                stale,
                retry,
                self.execution(second, ExecutionStatus.PASSED, 4),
                self.execution(sorted(self.steps, key=lambda step: step.order)[2], ExecutionStatus.PASSED, 6),
            ],
        )

        report = self.generator.generate(run)
        first_report = report.steps[0]

        self.assertEqual(report.status, ExecutionStatus.PASSED)
        self.assertEqual(first_report.status, ExecutionStatus.PASSED)
        self.assertEqual(len(first_report.attempts), 2)
        self.assertEqual(
            [attempt.test_plan_version_id for attempt in first_report.attempts],
            [stale.test_plan_version_id, retry.test_plan_version_id],
        )
        self.assertEqual(
            [attempt.status for attempt in first_report.attempts],
            [ExecutionStatus.FAILED, ExecutionStatus.PASSED],
        )
        self.assertEqual(first_report.attempts[0].evidence[0].path, "artifacts/stale.png")
        self.assertEqual(first_report.attempts[1].evidence, [])

    def test_blocked_steps_are_reported_distinctly_from_failed_steps(self) -> None:
        first, second, third = sorted(self.steps, key=lambda step: step.order)
        run = DomainTestRun.from_test_case(
            self.case,
            [self.execution(first, ExecutionStatus.FAILED, 0, error="wrong title")],
            blocked_step_ids=[second.id, third.id],
        )

        report = self.generator.generate(run)
        serialized = report.to_json()
        parsed = json.loads(serialized)

        # The failed step shows FAILED with its attempt; the blocked steps
        # show BLOCKED with no attempts and are never reported as FAILED.
        self.assertEqual(report.status, ExecutionStatus.FAILED)
        self.assertEqual(
            [step.status for step in report.steps],
            [ExecutionStatus.FAILED, ExecutionStatus.BLOCKED, ExecutionStatus.BLOCKED],
        )
        self.assertEqual([len(step.attempts) for step in report.steps], [1, 0, 0])
        self.assertEqual(
            [step["status"] for step in parsed["steps"]],
            ["FAILED", "BLOCKED", "BLOCKED"],
        )
        self.assertEqual(DomainTestReport.model_validate_json(serialized), report)

    def test_file_output_writes_json_to_explicit_path(self) -> None:
        step = min(self.steps, key=lambda item: item.order)
        run = DomainTestRun.from_test_case(
            self.case,
            [self.execution(step, ExecutionStatus.PASSED, 0)],
        )
        report = self.generator.generate(run)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            self.generator.write_json(report, output)
            contents = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(contents["run_id"], str(report.run_id))
        self.assertEqual(contents["status"], "FAILED")  # Other steps have no final attempt.


if __name__ == "__main__":
    unittest.main()
