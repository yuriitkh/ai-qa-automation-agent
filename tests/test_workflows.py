import unittest
from uuid import uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    ExecutionStatus,
    FailurePolicy,
    Precondition,
    QATestPlan,
    QATestStep,
    RunContext,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.pinned_execution import (
    PinnedExecutionService,
    PlanSelectionError,
    PlanVersionSet,
    StepPlanSelection,
)
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.run_history import (
    InMemoryRunHistoryRepository,
    RunHistoryService,
    WorkflowType,
)
from qa_agent.setup_orchestration import (
    SetupCleanupCoordinator,
    SetupOperationResult,
    SetupStatus,
)
from qa_agent.workflows import (
    AutomationWorkflow,
    RegressionWorkflow,
    ValidationWorkflow,
    WorkflowOutcome,
)


class PinnedWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = self.make_step("First", 0)
        self.second = self.make_step("Second", 1)
        self.test_case = DomainTestCase(
            name="Catalog flow",
            description="Verify the catalog flow.",
            # Input order is deliberately reversed; pinned execution follows
            # TestStep.order deterministically.
            steps=[self.second, self.first],
        )
        self.plan_store = InMemoryPlanStore()
        self.versions = {}
        self.plans = {}
        for step in (self.first, self.second):
            test_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
            self.plans[step.id] = test_plan
            version_one = self.make_version(test_plan, 1, f"step-{step.order}-v1")
            version_two = self.make_version(test_plan, 2, f"step-{step.order}-v2")
            self.plan_store.save(step.id, version_one, test_plan=test_plan)
            self.plan_store.save(step.id, version_two, test_plan=test_plan)
            self.versions[step.id] = (version_one, version_two)
        self.repository = InMemoryExecutionRepository()
        self.received_plans = []
        self.runner = lambda plan: self.record_pass(plan)
        self.executor = self.make_executor(self.runner)

    @staticmethod
    def make_step(name, order, failure_policy=FailurePolicy.CONTINUE):
        return DomainTestStep(
            name=name,
            description=f"Check {name}.",
            expected=f"{name} is correct.",
            order=order,
            failure_policy=failure_policy,
        )

    @staticmethod
    def make_version(test_plan, number, marker):
        return DomainTestPlanVersion(
            test_plan_id=test_plan.id,
            version=number,
            qa_test_plan=QATestPlan(
                url="https://example.test/",
                steps=[QATestStep(
                    action="assert_title", parameters={"expected": marker}
                )],
            ),
        )

    def record_pass(self, plan):
        self.received_plans.append(plan)
        return {"status": "passed", "steps": []}

    def make_executor(self, runner):
        return PinnedExecutionService(
            self.plan_store,
            PlanExecutionService(runner, self.repository),
        )

    def full_selection(self, version_index=0):
        return PlanVersionSet(tuple(
            StepPlanSelection(step.id, self.versions[step.id][version_index].id)
            for step in (self.first, self.second)
        ))

    def test_validation_runs_exact_pinned_versions_in_test_step_order(self) -> None:
        workflow = ValidationWorkflow(self.executor)

        result = workflow.run(self.test_case, self.full_selection(version_index=0))

        self.assertEqual(result.outcome, WorkflowOutcome.PASSED)
        self.assertEqual(
            [item.test_plan_version_id for item in result.execution.step_executions],
            [self.versions[self.first.id][0].id, self.versions[self.second.id][0].id],
        )
        self.assertEqual(
            [plan.steps[0].parameters["expected"] for plan in self.received_plans],
            ["step-0-v1", "step-1-v1"],
        )
        self.assertEqual(
            [execution.test_plan_version_id for execution in result.test_run.executions],
            [self.versions[self.first.id][0].id, self.versions[self.second.id][0].id],
        )
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)

    def test_regression_does_not_substitute_newer_cached_versions(self) -> None:
        workflow = RegressionWorkflow(self.executor)

        result = workflow.run(self.test_case, self.full_selection(version_index=0))

        self.assertEqual(result.outcome, WorkflowOutcome.PASSED)
        self.assertEqual(
            [execution.test_plan_version_id for execution in result.test_run.executions],
            [self.versions[self.first.id][0].id, self.versions[self.second.id][0].id],
        )
        self.assertEqual(
            [self.plan_store.find(step.id).id for step in (self.first, self.second)],
            [self.versions[self.first.id][1].id, self.versions[self.second.id][1].id],
        )

    def test_exact_execution_does_not_call_latest_lookup(self) -> None:
        class NoLatestLookupStore:
            def __init__(inner_self, store):
                inner_self.store = store

            def find(inner_self, _):
                raise AssertionError("Pinned workflows must not look up latest versions.")

            def get_version(inner_self, version_id):
                return inner_self.store.get_version(version_id)

            def find_test_plan(inner_self, step_id):
                return inner_self.store.find_test_plan(step_id)

        executor = PinnedExecutionService(
            NoLatestLookupStore(self.plan_store),
            PlanExecutionService(self.runner, self.repository),
        )

        result = RegressionWorkflow(executor).run(
            self.test_case, self.full_selection()
        )

        self.assertEqual(result.outcome, WorkflowOutcome.PASSED)

    def test_assertion_failure_is_product_failure_without_plan_mutation(self) -> None:
        def assertion_failure(plan):
            self.received_plans.append(plan)
            return {
                "status": "failed",
                "steps": [{
                    "action": "assert_title",
                    "status": "failed",
                    "error": "Expected title, but actual title differs.",
                }],
            }

        executor = self.make_executor(assertion_failure)
        workflow = ValidationWorkflow(executor)

        result = workflow.run(self.test_case, self.full_selection())

        self.assertEqual(result.outcome, WorkflowOutcome.PRODUCT_FAILURE)
        self.assertEqual(result.test_run.executions[0].status, ExecutionStatus.FAILED)
        self.assertEqual(len(self.plan_store.get_version(self.versions[self.first.id][1].id).qa_test_plan.steps), 1)
        self.assertEqual(self.plan_store.find(self.first.id).id, self.versions[self.first.id][1].id)
        self.assertEqual(len(self.repository.list_for_test_step(self.first.id)), 1)

        regression = RegressionWorkflow(self.make_executor(assertion_failure)).run(
            self.test_case, self.full_selection()
        )
        self.assertEqual(regression.outcome, WorkflowOutcome.PRODUCT_FAILURE)
        self.assertEqual(
            self.plan_store.find(self.first.id).id, self.versions[self.first.id][1].id
        )

    def test_drift_is_reported_without_discovery_repair_or_regeneration(self) -> None:
        click_version = DomainTestPlanVersion(
            test_plan_id=self.plans[self.first.id].id,
            version=3,
            qa_test_plan=QATestPlan(
                url="https://example.test/",
                steps=[QATestStep(action="click", parameters={"selector": "#old"})],
            ),
        )
        self.plan_store.save(
            self.first.id,
            click_version,
            test_plan=self.plans[self.first.id],
        )

        def stale_locator(_):
            return {
                "status": "failed",
                "steps": [{
                    "action": "click",
                    "status": "failed",
                    "error": "Selector '#old' was not found on the page.",
                }],
            }

        selected = PlanVersionSet((
            StepPlanSelection(self.first.id, click_version.id),
            StepPlanSelection(self.second.id, self.versions[self.second.id][0].id),
        ))
        regression = RegressionWorkflow(self.make_executor(stale_locator)).run(
            self.test_case, selected
        )
        validation = ValidationWorkflow(self.make_executor(stale_locator)).run(
            self.test_case, selected
        )

        self.assertEqual(regression.outcome, WorkflowOutcome.AUTOMATION_DRIFT)
        self.assertEqual(validation.outcome, WorkflowOutcome.AUTOMATION_DRIFT)
        self.assertEqual(len(regression.execution.step_executions), 2)
        self.assertEqual(
            [item.test_plan_version_id for item in regression.execution.step_executions],
            [click_version.id, self.versions[self.second.id][0].id],
        )
        self.assertEqual(
            self.plan_store.find(self.first.id).id, click_version.id
        )
        self.assertEqual(len(self.plan_store.get_version(self.versions[self.first.id][0].id).qa_test_plan.steps), 1)

    def test_infrastructure_error_is_distinct_and_does_not_continue(self) -> None:
        def broken_runner(_):
            raise RuntimeError("browser launch failed")

        workflow = ValidationWorkflow(self.make_executor(broken_runner))

        result = workflow.run(self.test_case, self.full_selection())

        self.assertEqual(result.outcome, WorkflowOutcome.INFRASTRUCTURE_ERROR)
        self.assertEqual(len(result.test_run.executions), 1)
        self.assertEqual(len(self.repository.list_for_test_step(self.first.id)), 1)
        self.assertEqual(self.repository.list_for_test_step(self.second.id), [])

    def test_run_context_identity_flows_through_setup_test_run_and_cleanup(self) -> None:
        precondition = Precondition(
            description="A session is available.",
            order=0,
            provided_data_keys=["session_token"],
        )
        case = self.case_with_preconditions([precondition])
        context = RunContext()
        observed = []

        def setup(_condition, run_context, register_cleanup):
            self.assertIs(run_context, context)
            run_context.set_value("session_token", "runtime-token", sensitive=True)
            register_cleanup(lambda cleanup_context: observed.append(cleanup_context))
            return SetupOperationResult(
                status=SetupStatus.SUCCEEDED,
                produced_data_keys=("session_token",),
            )

        workflow = ValidationWorkflow(
            self.executor,
            SetupCleanupCoordinator({precondition.id: _Setup(setup)}),
        )

        result = workflow.run(case, self.full_selection(), run_context=context)

        self.assertIs(result.run_context, context)
        self.assertIs(result.test_run.run_context, context)
        self.assertEqual(result.run_context.get_value("session_token"), "runtime-token")
        self.assertEqual(observed, [context])
        self.assertTrue(result.cleanup.succeeded)

    def test_setup_failure_skips_execution_and_still_runs_prior_cleanup(self) -> None:
        first = Precondition(description="Resource allocated.", order=0)
        second = Precondition(description="Second resource ready.", order=1)
        case = self.case_with_preconditions([first, second])
        events = []

        def allocate(_condition, _context, register_cleanup):
            register_cleanup(lambda _: events.append("cleanup"), "allocated")
            return SetupOperationResult(status=SetupStatus.SUCCEEDED)

        def fail_setup(*_):
            events.append("setup-failed")
            return SetupOperationResult(
                status=SetupStatus.PRECONDITION_NOT_ESTABLISHED,
                error="Second resource unavailable.",
            )

        workflow = ValidationWorkflow(
            self.executor,
            SetupCleanupCoordinator({
                first.id: _Setup(allocate), second.id: _Setup(fail_setup)
            }),
        )

        result = workflow.run(case, self.full_selection())

        self.assertEqual(result.outcome, WorkflowOutcome.SETUP_FAILURE)
        self.assertFalse(result.product_started)
        self.assertEqual(result.test_run.executions, [])
        self.assertEqual(result.test_run.status, ExecutionStatus.FAILED)
        self.assertEqual(events, ["setup-failed", "cleanup"])
        self.assertEqual(self.received_plans, [])
        self.assertTrue(result.cleanup.succeeded)

    def test_cleanup_failure_is_separate_and_does_not_erase_product_result(self) -> None:
        precondition = Precondition(description="Prepared.", order=0)
        case = self.case_with_preconditions([precondition])

        def setup(_condition, _context, register_cleanup):
            def broken_cleanup(_):
                raise RuntimeError("cleanup failed")

            register_cleanup(broken_cleanup, "release")
            return SetupOperationResult(status=SetupStatus.SUCCEEDED)

        workflow = ValidationWorkflow(
            self.executor,
            SetupCleanupCoordinator({precondition.id: _Setup(setup)}),
        )

        result = workflow.run(case, self.full_selection())

        self.assertEqual(result.outcome, WorkflowOutcome.PASSED)
        self.assertFalse(result.cleanup.succeeded)
        self.assertEqual(result.cleanup.failures[0].label, "release")

    def test_cleanup_runs_after_product_failure_drift_infrastructure_and_block_rest(self) -> None:
        cases = (
            ("product failure", "product", FailurePolicy.CONTINUE,
             WorkflowOutcome.PRODUCT_FAILURE),
            ("automation drift", "drift", FailurePolicy.CONTINUE,
             WorkflowOutcome.AUTOMATION_DRIFT),
            ("infrastructure error", "infrastructure", FailurePolicy.CONTINUE,
             WorkflowOutcome.INFRASTRUCTURE_ERROR),
            ("block rest", "product", FailurePolicy.BLOCK_REST,
             WorkflowOutcome.PRODUCT_FAILURE),
        )
        for label, failure_kind, policy, expected_outcome in cases:
            with self.subTest(case=label):
                first = self.first.model_copy(update={"failure_policy": policy})
                case = DomainTestCase(
                    name="Lifecycle case",
                    description="Always clean up after execution.",
                    steps=[first, self.second],
                )
                precondition = Precondition(description="Prepared.", order=0)
                case = case.model_copy(update={"preconditions": [precondition]})
                cleanup_contexts = []

                def setup(_condition, _context, register_cleanup):
                    register_cleanup(
                        lambda cleanup_context: cleanup_contexts.append(cleanup_context)
                    )
                    return SetupOperationResult(status=SetupStatus.SUCCEEDED)

                def runner(_plan):
                    if failure_kind == "infrastructure":
                        raise RuntimeError("browser unavailable")
                    action = "click" if failure_kind == "drift" else "assert_title"
                    error = (
                        "Selector '#old' was not found on the page."
                        if failure_kind == "drift"
                        else "Expected title is missing."
                    )
                    return {
                        "status": "failed",
                        "steps": [{"action": action, "status": "failed", "error": error}],
                    }

                context = RunContext()
                workflow = ValidationWorkflow(
                    self.make_executor(runner),
                    SetupCleanupCoordinator({precondition.id: _Setup(setup)}),
                )
                result = workflow.run(
                    case,
                    self.full_selection_for(case, [
                        self.versions[self.first.id][0],
                        self.versions[self.second.id][0],
                    ]),
                    run_context=context,
                )

                self.assertEqual(result.outcome, expected_outcome)
                self.assertEqual(cleanup_contexts, [context])
                self.assertIs(result.test_run.run_context, context)

    def test_block_rest_marks_later_steps_blocked(self) -> None:
        blocking_step = self.first.model_copy(update={
            "failure_policy": FailurePolicy.BLOCK_REST,
        })
        following_step = self.second
        case = DomainTestCase(
            name="Blocked flow",
            description="Stop after the first failed step.",
            steps=[blocking_step, following_step],
        )
        selected = self.full_selection_for(case, [
            self.versions[self.first.id][0], self.versions[self.second.id][0]
        ])
        workflow = ValidationWorkflow(self.executor)

        result = workflow.run(case, selected)

        self.assertEqual(result.outcome, WorkflowOutcome.PASSED)

        def failed_runner(_):
            return {
                "status": "failed",
                "steps": [{"action": "assert_title", "status": "failed", "error": "wrong"}],
            }

        failed_result = ValidationWorkflow(self.make_executor(failed_runner)).run(
            case, selected
        )
        self.assertEqual(failed_result.outcome, WorkflowOutcome.PRODUCT_FAILURE)
        self.assertEqual(failed_result.test_run.blocked_steps, [following_step.id])
        self.assertEqual(len(failed_result.test_run.executions), 1)

    def test_continue_policy_executes_steps_after_product_failure(self) -> None:
        calls = []

        def fail_first(plan):
            calls.append(plan.steps[0].parameters["expected"])
            if len(calls) == 1:
                return {
                    "status": "failed",
                    "steps": [{"action": "assert_title", "status": "failed", "error": "wrong"}],
                }
            return {"status": "passed", "steps": []}

        result = ValidationWorkflow(self.make_executor(fail_first)).run(
            self.test_case, self.full_selection()
        )

        self.assertEqual(result.outcome, WorkflowOutcome.PRODUCT_FAILURE)
        self.assertEqual(len(result.test_run.executions), 2)
        self.assertEqual(len(calls), 2)

    def test_regression_runs_use_independent_default_contexts(self) -> None:
        workflow = RegressionWorkflow(self.executor)

        first = workflow.run(self.test_case, self.full_selection())
        second = workflow.run(self.test_case, self.full_selection())

        self.assertIsNot(first.run_context, second.run_context)
        self.assertIs(first.test_run.run_context, first.run_context)
        self.assertIs(second.test_run.run_context, second.run_context)

    def test_workflow_persists_completed_run_and_safe_context(self) -> None:
        history_repository = InMemoryRunHistoryRepository()
        history = RunHistoryService(history_repository, self.repository)
        context = RunContext()
        secret = "FAKE_WORKFLOW_HISTORY_SECRET"
        context.set_value("token", secret, sensitive=True)
        workflow = RegressionWorkflow(self.executor, run_history=history)

        result = workflow.run(
            self.test_case, self.full_selection(), run_context=context
        )
        saved = history.get(result.test_run.id)

        self.assertIsNotNone(saved)
        self.assertEqual(saved.workflow_type, WorkflowType.REGRESSION)
        self.assertEqual(
            [reference.execution_id for reference in saved.executions],
            [execution.id for execution in result.test_run.executions],
        )
        self.assertEqual(
            [reference.test_plan_version_id for reference in saved.executions],
            [execution.test_plan_version_id for execution in result.test_run.executions],
        )
        self.assertNotIn(secret, saved.model_dump_json())

    def test_validation_rejects_incomplete_foreign_duplicate_and_unknown_pins(self) -> None:
        workflow = ValidationWorkflow(self.executor)
        valid = self.full_selection()
        unknown_step = uuid4()
        unknown_version = uuid4()

        invalid_sets = [
            PlanVersionSet(valid.selections[:1]),
            PlanVersionSet(valid.selections + (
                StepPlanSelection(unknown_step, self.versions[self.first.id][0].id),
            )),
            PlanVersionSet((
                valid.selections[0], valid.selections[0], valid.selections[1],
            )),
            PlanVersionSet((
                StepPlanSelection(self.first.id, unknown_version), valid.selections[1],
            )),
        ]
        for selection in invalid_sets:
            with self.subTest(selection=selection):
                with self.assertRaises(PlanSelectionError):
                    workflow.run(self.test_case, selection)
        self.assertEqual(self.received_plans, [])

    def test_version_from_another_test_step_is_rejected(self) -> None:
        selection = PlanVersionSet((
            StepPlanSelection(self.first.id, self.versions[self.second.id][0].id),
            StepPlanSelection(self.second.id, self.versions[self.second.id][0].id),
        ))

        with self.assertRaises(PlanSelectionError):
            RegressionWorkflow(self.executor).run(self.test_case, selection)

    def test_automation_workflow_delegates_to_compatibility_pipeline(self) -> None:
        context = RunContext()
        expected = object()

        class Pipeline:
            def run(inner_self, task, base_url, run_context):
                self.assertEqual(task, "Check catalog")
                self.assertEqual(base_url, "https://example.test/")
                self.assertIs(run_context, context)
                return expected

        result = AutomationWorkflow(Pipeline()).run(
            "Check catalog", "https://example.test/", context
        )

        self.assertIs(result, expected)

    def case_with_preconditions(self, preconditions):
        return self.test_case.model_copy(update={
            "preconditions": preconditions,
        })

    def full_selection_for(self, case, versions):
        return PlanVersionSet(tuple(
            StepPlanSelection(step.id, version.id)
            for step, version in zip(sorted(case.steps, key=lambda item: item.order), versions)
        ))


class _Setup:
    def __init__(self, callback):
        self.callback = callback

    def execute(self, precondition, run_context, register_cleanup):
        return self.callback(precondition, run_context, register_cleanup)


if __name__ == "__main__":
    unittest.main()
