import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from qa_agent.models import (
    Execution,
    ExecutionStatus,
    RunContext,
    TestCase as DomainTestCase,
    TestRun as _DomainTestRun,
    TestStep as DomainTestStep,
)


class TestRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.step_one = self.make_step("Step 1", 0)
        self.step_two = self.make_step("Step 2", 1)
        self.test_case = DomainTestCase(
            name="Example run",
            description="Run example checks.",
            steps=[self.step_one, self.step_two],
        )

    @staticmethod
    def make_step(name: str, order: int) -> DomainTestStep:
        return DomainTestStep(
            name=name,
            description=f"Run {name}.",
            expected=f"{name} passes.",
            order=order,
        )

    def make_execution(
        self,
        step: DomainTestStep,
        status: ExecutionStatus,
        offset: int,
    ) -> Execution:
        started_at = self.started_at + timedelta(seconds=offset)
        return Execution(
            test_step_id=step.id,
            test_plan_version_id=uuid4(),
            status=status,
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            actual_result=status.value,
            error="assertion failed" if status == ExecutionStatus.FAILED else None,
        )

    def test_one_step_pass_produces_passed_test_run(self) -> None:
        case = DomainTestCase(
            name="Single step",
            description="One check.",
            steps=[self.step_one],
        )
        execution = self.make_execution(self.step_one, ExecutionStatus.PASSED, 0)

        run = _DomainTestRun.from_test_case(case, [execution])

        self.assertEqual(run.test_case_id, case.id)
        self.assertEqual(run.test_step_ids, [self.step_one.id])
        self.assertEqual(run.status, ExecutionStatus.PASSED)
        self.assertEqual(run.started_at, execution.started_at)
        self.assertEqual(run.finished_at, execution.finished_at)
        self.assertEqual(run.passed_steps, [self.step_one.id])
        self.assertEqual(run.failed_steps, [])

    def test_runs_have_independent_contexts_and_can_use_fresh_values(self) -> None:
        first_run = _DomainTestRun.from_test_case(self.test_case, [])
        second_run = _DomainTestRun.from_test_case(self.test_case, [])

        first_run.run_context.set_value(
            "registration_email", "unique-1@example.test", source="test-data"
        )
        second_run.run_context.set_value(
            "registration_email", "unique-2@example.test", source="test-data"
        )

        self.assertIsNot(first_run.run_context, second_run.run_context)
        self.assertEqual(
            first_run.run_context.get_value("registration_email"),
            "unique-1@example.test",
        )
        self.assertEqual(
            second_run.run_context.get_value("registration_email"),
            "unique-2@example.test",
        )
        first_run.run_context.replace_value(
            "registration_email", "updated-1@example.test"
        )
        self.assertEqual(
            first_run.run_context.get_value("registration_email"),
            "updated-1@example.test",
        )
        self.assertEqual(
            second_run.run_context.get_value("registration_email"),
            "unique-2@example.test",
        )

    def test_run_context_api_has_deterministic_key_and_update_behavior(self) -> None:
        context = RunContext()
        self.assertFalse(context.has_value("created_user_id"))
        with self.assertRaises(KeyError):
            context.get_value("created_user_id")
        with self.assertRaises(ValueError):
            context.set_value("  ", "value")

        context.set_value("created_user_id", "user-123", source="registration-step")
        with self.assertRaises(ValueError):
            context.set_value("created_user_id", "user-456")
        context.replace_value("created_user_id", "user-456")

        self.assertEqual(context.get_value("created_user_id"), "user-456")
        self.assertTrue(context.has_value("created_user_id"))
        self.assertEqual(
            context.values["created_user_id"].source, "registration-step"
        )

    def test_sensitive_context_values_are_accessible_but_safe_outputs_redact(self) -> None:
        secret = "FAKE_TOKEN_do_not_leak_9f4a"
        run = _DomainTestRun.from_test_case(self.test_case, [])
        run.run_context.set_value(
            "api_token", secret, sensitive=True, source="fixture-provider"
        )
        run.run_context.replace_value("api_token", secret, sensitive=False)
        run.run_context.set_value("product_id", "sku-42", source="catalog-fixture")

        self.assertEqual(run.run_context.get_value("api_token"), secret)
        safe_text = str(run.run_context)
        self.assertNotIn(secret, safe_text)
        self.assertNotIn(secret, repr(run.run_context))
        self.assertNotIn(secret, repr(run))
        self.assertNotIn(secret, str(run))
        safe_data = run.run_context.safe_dump()
        self.assertNotIn(secret, str(safe_data))
        self.assertEqual(safe_data["api_token"]["value"], "[REDACTED]")
        self.assertEqual(safe_data["product_id"]["value"], "sku-42")

    def test_run_context_persistence_round_trip_preserves_values_and_ids(self) -> None:
        secret = "FAKE_PASSWORD_keep_for_runtime"
        run = _DomainTestRun.from_test_case(self.test_case, [])
        run.run_context.set_value("password", secret, sensitive=True)

        dumped = run.model_dump()
        self.assertEqual(dumped["run_context"]["values"]["password"]["value"], secret)
        restored = _DomainTestRun.model_validate(dumped)

        self.assertEqual(restored.id, run.id)
        self.assertEqual(restored.test_case_id, run.test_case_id)
        self.assertEqual(restored.run_context.get_value("password"), secret)
        self.assertNotIn(secret, repr(restored.run_context))

    def test_legacy_test_run_without_context_gets_a_new_empty_context(self) -> None:
        first = _DomainTestRun.from_test_case(self.test_case, [])
        second = _DomainTestRun.from_test_case(self.test_case, [])

        self.assertEqual(first.run_context.values, {})
        self.assertEqual(second.run_context.values, {})
        self.assertIsNot(first.run_context, second.run_context)

    def test_multiple_steps_all_pass(self) -> None:
        executions = [
            self.make_execution(self.step_one, ExecutionStatus.PASSED, 0),
            self.make_execution(self.step_two, ExecutionStatus.PASSED, 2),
        ]

        run = _DomainTestRun.from_test_case(self.test_case, executions)

        self.assertEqual(run.status, ExecutionStatus.PASSED)
        self.assertEqual(run.passed_steps, [self.step_one.id, self.step_two.id])
        self.assertEqual(run.failed_steps, [])

    def test_one_failed_final_step_fails_test_run(self) -> None:
        passed = self.make_execution(self.step_one, ExecutionStatus.PASSED, 0)
        failed = self.make_execution(self.step_two, ExecutionStatus.FAILED, 2)

        run = _DomainTestRun.from_test_case(self.test_case, [passed, failed])

        self.assertEqual(run.status, ExecutionStatus.FAILED)
        self.assertEqual(run.failed_steps, [self.step_two.id])
        self.assertEqual(run.passed_steps, [self.step_one.id])

    def test_stale_failure_then_pass_preserves_history_and_uses_final_attempt(self) -> None:
        stale_failure = self.make_execution(self.step_one, ExecutionStatus.FAILED, 0)
        retry_pass = self.make_execution(self.step_one, ExecutionStatus.PASSED, 2)
        second_step_pass = self.make_execution(self.step_two, ExecutionStatus.PASSED, 4)
        executions = [stale_failure, retry_pass, second_step_pass]

        run = _DomainTestRun.from_test_case(self.test_case, executions)

        self.assertEqual(run.status, ExecutionStatus.PASSED)
        self.assertEqual(run.executions, executions)
        self.assertEqual(run.final_executions, [retry_pass, second_step_pass])
        self.assertIs(run.final_execution_for_step(self.step_one.id), retry_pass)
        self.assertEqual(run.final_execution_for_step(self.step_one.id).status, ExecutionStatus.PASSED)
        self.assertEqual(run.passed_steps, [self.step_one.id, self.step_two.id])
        self.assertEqual(run.failed_steps, [])

    def test_missing_step_execution_is_a_failed_final_outcome(self) -> None:
        only_first_step = self.make_execution(self.step_one, ExecutionStatus.PASSED, 0)

        run = _DomainTestRun.from_test_case(self.test_case, [only_first_step])

        self.assertEqual(run.status, ExecutionStatus.FAILED)
        self.assertIsNone(run.final_execution_for_step(self.step_two.id))
        self.assertEqual(run.failed_steps, [self.step_two.id])

    def test_rejects_execution_for_step_outside_test_case(self) -> None:
        unrelated = self.make_execution(self.make_step("Other", 2), ExecutionStatus.PASSED, 0)

        with self.assertRaises(ValueError):
            _DomainTestRun.from_test_case(self.test_case, [unrelated])

    def test_failed_and_blocked_steps_stay_distinguishable(self) -> None:
        failed = self.make_execution(self.step_one, ExecutionStatus.FAILED, 0)

        run = _DomainTestRun.from_test_case(
            self.test_case,
            [failed],
            blocked_step_ids=[self.step_two.id],
        )

        self.assertEqual(run.failed_steps, [self.step_one.id])
        self.assertEqual(run.blocked_steps, [self.step_two.id])
        self.assertTrue(set(run.failed_steps).isdisjoint(run.blocked_steps))
        self.assertIsNone(run.final_execution_for_step(self.step_two.id))
        self.assertEqual(run.status, ExecutionStatus.FAILED)

    def test_blocked_steps_fail_the_run_without_becoming_failed_steps(self) -> None:
        run = _DomainTestRun.from_test_case(
            self.test_case,
            [],
            blocked_step_ids=[self.step_one.id, self.step_two.id],
        )

        self.assertEqual(run.blocked_steps, [self.step_one.id, self.step_two.id])
        self.assertEqual(run.failed_steps, [])
        self.assertEqual(run.status, ExecutionStatus.FAILED)

    def test_rejects_blocked_step_with_execution_or_unknown_step(self) -> None:
        executed = self.make_execution(self.step_one, ExecutionStatus.PASSED, 0)

        with self.assertRaises(ValueError):
            _DomainTestRun.from_test_case(
                self.test_case,
                [executed],
                blocked_step_ids=[self.step_one.id],
            )

        with self.assertRaises(ValueError):
            _DomainTestRun.from_test_case(
                self.test_case,
                [],
                blocked_step_ids=[uuid4()],
            )


if __name__ == "__main__":
    unittest.main()
