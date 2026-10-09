import json
import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from qa_agent.evidence_policy import (
    EvidenceMode,
    EvidencePolicy,
    EvidenceScope,
    ScreenshotMode,
    evidence_policy_scope,
)
from qa_agent.models import (
    Evidence,
    EvidenceType,
    Execution,
    ExecutionStatus,
    PlanVersionOrigin,
    RunContext,
    TestCase as DomainTestCase,
    TestRun as DomainTestRun,
    TestStep as DomainTestStep,
)
from qa_agent.reporting import RunReportGenerator
from qa_agent.run_history import RunHistoryRecord, WorkflowType


class RunReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started = datetime(2026, 4, 1, tzinfo=timezone.utc)
        self.steps = [
            self.make_step("Open page", 0),
            self.make_step("Verify result", 1),
            self.make_step("Next step", 2),
        ]
        self.case = DomainTestCase(
            public_id="TC-0007",
            name="Registration <script>alert(1)</script>",
            description="Report a registration outcome.",
            steps=self.steps,
        )
        self.secret = "FAKE_REPORT_SECRET_6e32"
        context = RunContext()
        context.set_value("password", self.secret, sensitive=True)
        passed = self.make_execution(self.steps[0], ExecutionStatus.PASSED, 0)
        failed = self.make_execution(
            self.steps[1],
            ExecutionStatus.FAILED,
            2,
            error=f"Expected welcome; token={self.secret}",
            actual=f"Actual value {self.secret}",
        )
        failed.evidence = (Evidence(
            execution_id=failed.id,
            type=EvidenceType.SCREENSHOT,
            path="artifacts/failure-view.png",
            description="Page after failed assertion.",
            scope=EvidenceScope.PAGE,
            event="FAILURE:assert_title:1",
        ),)
        self.run = DomainTestRun.from_test_case(
            self.case,
            [passed, failed],
            blocked_step_ids=[self.steps[2].id],
            run_context=context,
        )
        self.generator = RunReportGenerator()

    @staticmethod
    def make_step(name, order):
        return DomainTestStep(
            name=name,
            description=f"Perform {name}.",
            expected=f"{name} is correct.",
            order=order,
        )

    def make_execution(self, step, status, offset, *, error=None, actual=None):
        start = self.started + timedelta(seconds=offset)
        return Execution(
            test_step_id=step.id,
            test_plan_version_id=uuid4(),
            status=status,
            started_at=start,
            finished_at=start + timedelta(seconds=1),
            actual_result=actual if actual is not None else status.value,
            error=error,
        )

    def test_json_report_has_workflow_steps_versions_and_redacted_values(self) -> None:
        report = self.generator.generate_current(
            self.case,
            self.run,
            workflow_type=WorkflowType.REGRESSION,
            outcome="PRODUCT_FAILURE",
            started_at=self.started,
            finished_at=self.started + timedelta(seconds=4),
        )
        payload = json.loads(report.to_json())

        self.assertEqual(payload["workflow_type"], "REGRESSION")
        self.assertEqual(payload["outcome"], "PRODUCT_FAILURE")
        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(
            [step["status"] for step in payload["steps"]],
            ["PASSED", "FAILED", "BLOCKED"],
        )
        failed_attempt = payload["steps"][1]["attempts"][0]
        self.assertEqual(
            failed_attempt["test_plan_version_id"],
            str(self.run.executions[1].test_plan_version_id),
        )
        self.assertEqual(failed_attempt["evidence"][0]["name"], "failure-view.png")
        self.assertEqual(failed_attempt["evidence"][0]["scope"], "PAGE")
        self.assertEqual(failed_attempt["evidence"][0]["event"], "FAILURE:assert_title:1")
        self.assertEqual(failed_attempt["error"], "Expected welcome; token=[REDACTED]")
        self.assertEqual(failed_attempt["actual_result"], "Actual value [REDACTED]")
        self.assertNotIn(self.secret, report.to_json())
        self.assertEqual(report.duration_ms, 4000)

    def test_html_is_standalone_escaped_and_distinguishes_blocked(self) -> None:
        report = self.generator.generate_current(
            self.case,
            self.run,
            workflow_type=WorkflowType.VALIDATION,
            outcome="PRODUCT_FAILURE",
        )
        html = self.generator.to_html(report)

        self.assertIn("<!doctype html>", html.lower())
        self.assertIn("<style>", html)
        self.assertIn("Blocked", html)
        self.assertIn("Plan version", html)
        self.assertIn("failure-view.png", html)
        self.assertIn("Registration &lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("<script src=", html)
        self.assertNotIn("<link rel=", html)
        self.assertNotIn("cdnjs", html)
        self.assertNotIn(self.secret, html)

    def test_historical_record_generates_same_safe_report_without_live_test_run(self) -> None:
        record = RunHistoryRecord.from_completed_run(
            self.case,
            self.run,
            workflow_type=WorkflowType.AUTOMATION,
            outcome="FAILED",
        )

        report = self.generator.generate_history(record)

        self.assertEqual(report.run_id, self.run.id)
        self.assertEqual(report.workflow_type, WorkflowType.AUTOMATION)
        self.assertEqual(
            [step.status for step in report.steps], ["PASSED", "FAILED", "BLOCKED"]
        )
        self.assertEqual(
            report.steps[1].attempts[0].test_plan_version_id,
            self.run.executions[1].test_plan_version_id,
        )
        self.assertNotIn(self.secret, report.to_json())

    def test_ungrounded_assertion_reason_survives_json_and_html_safely(self) -> None:
        reason = (
            "This assertion was not grounded in the requirement or deterministic "
            "page evidence, so its failure cannot be attributed to product behavior."
        )
        executions = list(self.run.executions)
        executions[1] = executions[1].model_copy(update={"error": reason})
        run = self.run.model_copy(update={"executions": executions})

        report = self.generator.generate_current(
            self.case, run, workflow_type=WorkflowType.REGRESSION,
            outcome="AUTOMATION_EXECUTION_ERROR",
        )
        payload = json.loads(report.to_json())
        html = self.generator.to_html(report)

        self.assertEqual(payload["steps"][1]["attempts"][0]["error"], reason)
        self.assertIn(reason, html)
        self.assertNotIn("prompt", report.to_json().casefold())
        self.assertNotIn(self.secret, report.to_json())

    def test_public_ids_and_version_provenance_are_additive_and_readable(self) -> None:
        record = RunHistoryRecord.from_completed_run(
            self.case,
            self.run,
            workflow_type=WorkflowType.REGRESSION,
            outcome="PRODUCT_FAILURE",
        )
        references = [
            item.model_copy(update={
                "plan_version_number": 4,
                "plan_version_origin": PlanVersionOrigin.AI_GENERATED,
            })
            for item in record.executions
        ]
        record = record.model_copy(update={
            "public_id": "RUN-000253",
            "executions": references,
        })
        report = self.generator.generate_history(record)
        payload = json.loads(report.to_json())

        self.assertEqual(payload["run_id"], str(self.run.id))
        self.assertEqual(payload["test_case_id"], str(self.case.id))
        self.assertEqual(payload["run_public_id"], "RUN-000253")
        self.assertEqual(payload["test_case_public_id"], "TC-0007")
        self.assertEqual(
            payload["steps"][0]["attempts"][0]["test_plan_version_id"],
            str(self.run.executions[0].test_plan_version_id),
        )
        self.assertEqual(payload["steps"][0]["attempts"][0]["plan_version_number"], 4)
        self.assertEqual(
            payload["steps"][0]["attempts"][0]["plan_version_origin"],
            "AI_GENERATED",
        )

        html = self.generator.to_html(report)
        self.assertIn("RUN-000253", html)
        self.assertIn("TC-0007", html)
        self.assertIn("v4 · Generated by AI", html)
        self.assertIn("Technical IDs", html)
        self.assertIn(str(self.run.executions[0].test_plan_version_id), html)

    def test_html_uses_only_explicitly_supplied_local_evidence_url(self) -> None:
        report = self.generator.generate_current(
            self.case, self.run, workflow_type=WorkflowType.REGRESSION
        )
        html = self.generator.to_html(
            report,
            evidence_url=lambda _step, attempt, _evidence, index: (
                f"/runs/{report.run_id}/evidence/{attempt.execution_id}/{index}"
            ),
        )

        self.assertIn(
            f"/runs/{report.run_id}/evidence/{self.run.executions[1].id}/0",
            html,
        )

    def test_reports_include_effective_evidence_policy_and_capture_label(self) -> None:
        policy = EvidencePolicy(
            mode=EvidenceMode.EVERY_VERIFICATION,
            screenshot_mode=ScreenshotMode.ELEMENT_AND_PAGE,
        )
        with evidence_policy_scope(policy):
            report = self.generator.generate_current(
                self.case, self.run, workflow_type=WorkflowType.AUTOMATION
            )

        payload = json.loads(report.to_json())
        capture = payload["steps"][1]["attempts"][0]["evidence"][0]
        html = self.generator.to_html(
            report,
            evidence_url=lambda _step, _attempt, _evidence, _index: "/local-evidence.png",
        )
        self.assertEqual(payload["evidence_policy"], {
            "mode": "EVERY_VERIFICATION",
            "screenshot_mode": "ELEMENT_AND_PAGE",
        })
        self.assertEqual(capture["scope"], "PAGE")
        self.assertIn("Failure · Page screenshot", html)
        self.assertIn("Evidence mode", html)
        self.assertIn("Every verification", html)
        self.assertIn("Element + Page", html)

    def test_pre_policy_history_defaults_to_legacy_evidence_display(self) -> None:
        record = RunHistoryRecord.from_completed_run(
            self.case, self.run, workflow_type=WorkflowType.REGRESSION
        ).model_copy(update={"evidence_policy": None})

        report = self.generator.generate_history(record)
        html = self.generator.to_html(report)

        self.assertIsNone(report.evidence_policy)
        self.assertIn("Failures only", html)
        self.assertIn("Page</span>", html)

    def test_run_json_includes_success_evidence_and_html_escapes_its_name(self) -> None:
        passed_execution = self.run.executions[0].model_copy(update={
            "evidence": (Evidence(
                execution_id=self.run.executions[0].id,
                type=EvidenceType.SCREENSHOT,
                path="artifacts/success.png",
                description="After a passing assertion.",
                scope=EvidenceScope.ELEMENT,
                event="VERIFICATION:assert_visible:0",
            ),),
        })
        run = self.run.model_copy(update={
            "executions": (passed_execution, *self.run.executions[1:]),
        })
        record = RunHistoryRecord.from_completed_run(
            self.case, run, workflow_type=WorkflowType.REGRESSION
        )
        evidence_refs = [
            item.model_copy(update={"name": "<script>alert(1)</script>.png"})
            if item.name == "success.png" else item
            for execution in record.executions
            for item in execution.evidence
        ]
        first_reference, *other_references = record.executions
        record = record.model_copy(update={
            "executions": [
                first_reference.model_copy(update={"evidence": evidence_refs[:1]}),
                *other_references,
            ],
        })

        report = self.generator.generate_history(record)
        payload = json.loads(report.to_json())
        html = self.generator.to_html(report)

        evidence = payload["steps"][0]["attempts"][0]["evidence"][0]
        self.assertEqual(evidence["scope"], "ELEMENT")
        self.assertEqual(evidence["event"], "VERIFICATION:assert_visible:0")
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;.png", html)
        self.assertNotIn("<script>alert(1)</script>.png", html)


if __name__ == "__main__":
    unittest.main()
