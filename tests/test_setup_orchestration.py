import unittest
from datetime import datetime, timezone
from uuid import uuid4

from qa_agent.models import (
    Execution,
    ExecutionStatus,
    Precondition as DomainPrecondition,
    TestCase as DomainTestCase,
    TestRun as DomainTestRun,
    TestStep as DomainTestStep,
)
from qa_agent.run_context import RunContext
from qa_agent.setup_orchestration import (
    CleanupManager,
    SetupCleanupCoordinator,
    SetupOperationResult,
    SetupStatus,
)


class _Setup:
    def __init__(self, callback):
        self.callback = callback

    def execute(self, precondition, run_context, register_cleanup):
        return self.callback(precondition, run_context, register_cleanup)


class SetupOrchestrationTests(unittest.TestCase):
    def make_case(self, conditions=()):
        return DomainTestCase(
            name="Test case",
            description="Verify product behavior.",
            steps=[DomainTestStep(
                name="Verify",
                description="Verify behavior.",
                expected="Behavior is correct.",
                order=0,
            )],
            preconditions=list(conditions),
        )

    @staticmethod
    def success(*keys):
        return SetupOperationResult(
            status=SetupStatus.SUCCEEDED,
            produced_data_keys=tuple(keys),
        )

    def test_zero_preconditions_runs_product_and_has_no_cleanup(self) -> None:
        case = self.make_case()
        context = RunContext()
        product_calls = []
        result = SetupCleanupCoordinator({}).run(
            case,
            context,
            lambda context: product_calls.append(context) or "product result",
        )

        self.assertTrue(result.setup.succeeded)
        self.assertEqual(result.setup.preconditions, ())
        self.assertIs(result.run_context, context)
        self.assertEqual(product_calls, [context])
        self.assertTrue(result.product_started)
        self.assertEqual(result.product_result, "product result")
        self.assertTrue(result.cleanup.succeeded)

    def test_multiple_preconditions_execute_in_order_and_share_run_context(self) -> None:
        first = DomainPrecondition(
            description="An unused registration email is available",
            order=0,
            provided_data_keys=["registration_email"],
        )
        second = DomainPrecondition(
            description="A catalog product is selected",
            order=1,
            provided_data_keys=["product_id"],
        )
        case = self.make_case([first, second])
        events = []

        def operation(label, key, value):
            def execute(precondition, context, register_cleanup):
                events.append(label)
                context.set_value(key, value, source=label)
                return self.success(key)
            return _Setup(execute)

        context = RunContext()
        product_seen = []
        result = SetupCleanupCoordinator({
            first.id: operation("email", "registration_email", "fresh@example.test"),
            second.id: operation("product", "product_id", "sku-42"),
        }).run(
            case,
            context,
            lambda run_context: product_seen.append((
                run_context.get_value("registration_email"),
                run_context.get_value("product_id"),
            )),
        )

        self.assertEqual(events, ["email", "product"])
        self.assertTrue(result.setup.succeeded)
        self.assertEqual(
            [item.produced_data_keys for item in result.setup.preconditions],
            [("registration_email",), ("product_id",)],
        )
        self.assertEqual(product_seen, [("fresh@example.test", "sku-42")])

    def test_missing_declared_data_key_prevents_product_execution(self) -> None:
        condition = DomainPrecondition(
            description="An unused email is available",
            order=0,
            provided_data_keys=["registration_email"],
        )
        case = self.make_case([condition])
        product_calls = []
        result = SetupCleanupCoordinator({
            condition.id: _Setup(lambda *_: self.success()),
        }).run(case, RunContext(), lambda _: product_calls.append("started"))

        self.assertEqual(result.setup.status, SetupStatus.PRECONDITION_NOT_ESTABLISHED)
        self.assertIn("registration_email", result.setup.preconditions[0].error or "")
        self.assertFalse(result.product_started)
        self.assertEqual(product_calls, [])

    def test_setup_can_explicitly_report_precondition_not_established(self) -> None:
        condition = DomainPrecondition(description="Email is unused", order=0)
        result = SetupCleanupCoordinator({
            condition.id: _Setup(lambda *_: SetupOperationResult(
                status=SetupStatus.PRECONDITION_NOT_ESTABLISHED,
                error="Could not obtain an unused email.",
            )),
        }).run(self.make_case([condition]), RunContext(), lambda _: self.fail("ran"))

        self.assertEqual(result.setup.status, SetupStatus.PRECONDITION_NOT_ESTABLISHED)
        self.assertFalse(result.product_started)
        self.assertEqual(result.setup.preconditions[0].error,
                         "Could not obtain an unused email.")

    def test_setup_exception_is_infrastructure_error_and_blocks_product_execution(self) -> None:
        condition = DomainPrecondition(description="User is prepared", order=0)

        def raise_setup(*_):
            raise ConnectionError("setup API unavailable")

        result = SetupCleanupCoordinator({
            condition.id: _Setup(raise_setup),
        }).run(self.make_case([condition]), RunContext(), lambda _: self.fail("ran"))

        self.assertEqual(result.setup.status, SetupStatus.SETUP_INFRASTRUCTURE_ERROR)
        self.assertEqual(result.setup.preconditions[0].error_type, "ConnectionError")
        self.assertEqual(result.setup.preconditions[0].error, "setup API unavailable")
        self.assertFalse(result.product_started)

    def test_run_context_insertion_failure_is_reported_as_setup_error(self) -> None:
        condition = DomainPrecondition(description="User exists", order=0)

        def insert_duplicate(_, context, __):
            context.set_value("user_id", "first")
            context.set_value("user_id", "second")
            return self.success("user_id")

        result = SetupCleanupCoordinator({
            condition.id: _Setup(insert_duplicate),
        }).run(self.make_case([condition]), RunContext(), lambda _: self.fail("ran"))

        self.assertEqual(result.setup.status, SetupStatus.SETUP_INFRASTRUCTURE_ERROR)
        self.assertEqual(result.setup.preconditions[0].error_type, "ValueError")
        self.assertIn("already exists", result.setup.preconditions[0].error or "")

    def test_setup_exception_redacts_sensitive_run_context_values(self) -> None:
        condition = DomainPrecondition(description="Token is available", order=0)
        secret = "FAKE_SETUP_EXCEPTION_71c2"

        def fail_after_secret(_, context, __):
            context.set_value("api_token", secret, sensitive=True)
            raise RuntimeError(f"provider failed with token {secret}")

        result = SetupCleanupCoordinator({
            condition.id: _Setup(fail_after_secret),
        }).run(self.make_case([condition]), RunContext(), lambda _: self.fail("ran"))

        self.assertEqual(result.setup.status, SetupStatus.SETUP_INFRASTRUCTURE_ERROR)
        self.assertNotIn(secret, result.setup.preconditions[0].error or "")
        self.assertNotIn(secret, repr(result))

    def test_setup_contract_error_and_sensitive_values_are_redacted(self) -> None:
        condition = DomainPrecondition(
            description="Credentials are available",
            order=0,
            provided_data_keys=["registration_email"],
        )
        secret = "FAKE_SETUP_SECRET_5d81"
        events = []

        def execute(_, context, register_cleanup):
            context.set_value("api_token", secret, sensitive=True)
            register_cleanup(
                lambda cleanup_context: events.append(
                    cleanup_context.get_value("api_token")
                ),
                "remove temporary credential",
            )
            # The strategy claims success but omits the produced key.
            return self.success()

        context = RunContext()
        result = SetupCleanupCoordinator({
            condition.id: _Setup(execute),
        }).run(self.make_case([condition]), context, lambda _: self.fail("ran"))

        self.assertEqual(result.setup.status, SetupStatus.PRECONDITION_NOT_ESTABLISHED)
        self.assertEqual(result.setup.preconditions[0].error_type, "SetupContractError")
        self.assertTrue(result.setup.preconditions[0].cleanup_registered)
        self.assertEqual(events, [secret])
        self.assertEqual(context.get_value("api_token"), secret)
        self.assertNotIn(secret, repr(context))
        self.assertNotIn(secret, repr(result))
        self.assertNotIn(secret, str(context.safe_dump()))

    def test_partial_setup_failure_cleans_prior_resources_in_reverse_order(self) -> None:
        conditions = [
            DomainPrecondition(description=f"Resource {label}", order=index)
            for index, label in enumerate(("A", "B", "C"))
        ]
        case = self.make_case(conditions)
        events = []

        def creates(label):
            def execute(_, context, register_cleanup):
                context.set_value(f"resource_{label}", label)
                register_cleanup(
                    lambda run_context: events.append(
                        f"cleanup-{run_context.get_value(f'resource_{label}')}"
                    ),
                    f"cleanup-{label}",
                )
                events.append(f"setup-{label}")
                return self.success(f"resource_{label}")
            return _Setup(execute)

        def cannot_create(*_):
            events.append("setup-C")
            return SetupOperationResult(
                status=SetupStatus.PRECONDITION_NOT_ESTABLISHED,
                error="Resource C could not be created.",
            )

        product_calls = []
        result = SetupCleanupCoordinator({
            conditions[0].id: creates("A"),
            conditions[1].id: creates("B"),
            conditions[2].id: _Setup(cannot_create),
        }).run(case, RunContext(), lambda _: product_calls.append("product"))

        self.assertEqual(events, ["setup-A", "setup-B", "setup-C", "cleanup-B", "cleanup-A"])
        self.assertEqual(result.setup.status, SetupStatus.PRECONDITION_NOT_ESTABLISHED)
        self.assertFalse(result.product_started)
        self.assertEqual(product_calls, [])
        self.assertTrue(result.cleanup.succeeded)

    def test_cleanup_runs_after_product_results_and_exceptions(self) -> None:
        scenarios = (
            ("passed", lambda _: "PASS", lambda value: value == "PASS", None),
            ("product failure", lambda _: "PRODUCT_FAILURE", lambda value: value == "PRODUCT_FAILURE", None),
            ("blocked", self._blocked_test_run, lambda value: value.status == ExecutionStatus.FAILED and len(value.blocked_step_ids) == 1, None),
            ("infrastructure", self._raise_product, lambda value: value is None, "RuntimeError"),
        )
        for label, product, expected_result, expected_error in scenarios:
            with self.subTest(label=label):
                condition = DomainPrecondition(description="Fixture exists", order=0)
                cleanup_calls = []

                def setup(_, _context, register_cleanup):
                    register_cleanup(lambda __: cleanup_calls.append("cleaned"), label)
                    return self.success()

                result = SetupCleanupCoordinator({
                    condition.id: _Setup(setup),
                }).run(self.make_case([condition]), RunContext(), product)

                self.assertEqual(cleanup_calls, ["cleaned"])
                self.assertTrue(result.cleanup.succeeded)
                self.assertTrue(expected_result(result.product_result))
                self.assertEqual(result.product_error_type, expected_error)

    def _blocked_test_run(self, _context):
        first_step = DomainTestStep(
            name="Failing step", description="Fails first.", expected="Passes", order=0
        )
        blocked_step = DomainTestStep(
            name="Blocked step", description="Does not run.", expected="Passes", order=1
        )
        run = DomainTestRun(
            test_case_id=self.make_case().id,
            test_step_ids=[first_step.id, blocked_step.id],
            started_at=datetime.now(timezone.utc),
            executions=[Execution(
                test_step_id=first_step.id,
                test_plan_version_id=uuid4(),
                status=ExecutionStatus.FAILED,
                actual_result="failed",
            )],
            blocked_step_ids=[blocked_step.id],
        )
        return run

    @staticmethod
    def _raise_product(_context):
        raise RuntimeError("product executor unavailable")

    def test_cleanup_runs_lifo_once_and_collects_all_failures(self) -> None:
        context = RunContext()
        manager = CleanupManager(context)
        events = []

        def action(name):
            def run(_):
                events.append(name)
                raise RuntimeError(f"{name} cleanup failed")
            return run

        manager.register(action("A"), "cleanup-A")
        manager.register(action("B"), "cleanup-B")
        manager.register(action("C"), "cleanup-C")

        first = manager.run()
        second = manager.run()

        self.assertIs(first, second)
        self.assertEqual(events, ["C", "B", "A"])
        self.assertEqual([item.label for item in first.failures], [
            "cleanup-C", "cleanup-B", "cleanup-A",
        ])
        self.assertTrue(all(item.error_type == "RuntimeError" for item in first.failures))

    def test_duplicate_cleanup_registration_is_rejected(self) -> None:
        manager = CleanupManager(RunContext())
        cleanup_action = lambda _: None
        manager.register(cleanup_action)
        with self.assertRaises(ValueError):
            manager.register(cleanup_action)

    def test_cleanup_failure_preserves_product_result_and_redacts_secret(self) -> None:
        condition = DomainPrecondition(description="Credential exists", order=0)
        secret = "FAKE_CLEANUP_SECRET_a330"
        context = RunContext()

        def setup(_, run_context, register_cleanup):
            run_context.set_value("password", secret, sensitive=True)

            def fail_cleanup(cleanup_context):
                self.assertEqual(cleanup_context.get_value("password"), secret)
                raise RuntimeError(f"could not delete credential {secret}")

            register_cleanup(fail_cleanup, f"delete {secret}")
            return self.success("password")

        result = SetupCleanupCoordinator({
            condition.id: _Setup(setup),
        }).run(
            self.make_case([condition]),
            context,
            lambda run_context: run_context.get_value("password"),
        )

        self.assertEqual(result.product_result, secret)
        self.assertFalse(result.product_error)
        self.assertFalse(result.cleanup.succeeded)
        self.assertNotIn(secret, result.cleanup.failures[0].label)
        self.assertNotIn(secret, result.cleanup.failures[0].message)
        self.assertNotIn(secret, repr(result))

    def test_product_exception_is_preserved_separately_from_cleanup_failure(self) -> None:
        condition = DomainPrecondition(description="Fixture exists", order=0)
        events = []

        def setup(_, _context, register_cleanup):
            register_cleanup(
                lambda __: (_ for _ in ()).throw(OSError("cleanup failed")),
                "remove fixture",
            )
            return self.success()

        result = SetupCleanupCoordinator({
            condition.id: _Setup(setup),
        }).run(
            self.make_case([condition]),
            RunContext(),
            lambda _: (_ for _ in ()).throw(RuntimeError("product failed")),
        )

        self.assertEqual(result.product_error_type, "RuntimeError")
        self.assertEqual(result.product_error, "product failed")
        self.assertEqual(len(result.cleanup.failures), 1)
        self.assertEqual(result.cleanup.failures[0].error_type, "OSError")
        self.assertEqual(result.cleanup.failures[0].message, "cleanup failed")


if __name__ == "__main__":
    unittest.main()
