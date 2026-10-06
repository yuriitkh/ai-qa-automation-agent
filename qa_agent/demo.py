"""Seed deterministic, safe run history for demonstrating the local UI."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid5

from qa_agent.models import (
    Execution,
    ExecutionSegment,
    ExecutionStatus,
    FailurePolicy,
    Precondition,
    QATestPlan,
    QATestStep,
    TestCase,
    TestRun,
    TestPlan,
    TestPlanVersion,
    TestStep,
)
from qa_agent.run_context import RunContext
from qa_agent.run_history import RunHistoryService, WorkflowType
from qa_agent.pinned_execution import WorkflowOutcome
from qa_agent.setup_orchestration import (
    CleanupOutcome,
    PreconditionSetupOutcome,
    SetupRunOutcome,
    SetupStatus,
)
from qa_agent.sqlite_storage import (
    SQLitePlanStore,
    SQLiteRunHistoryRepository,
    SQLiteTestCaseRepository,
)


_NAMESPACE = UUID("43aaea91-7c8e-5f4b-b704-c9869eae4eed")
_DEMO_SECRET = "demo-only-sensitive-value-not-a-credential"


@dataclass(frozen=True)
class DemoSeedResult:
    created: int
    skipped: int
    run_ids: tuple[UUID, ...]


def seed_demo_data(
    database_path: str | Path,
    *,
    demo_base_url: str = "http://127.0.0.1:8000/demo-target/registration",
) -> DemoSeedResult:
    """Add deterministic definitions, plans, and missing demo runs."""
    repository = SQLiteRunHistoryRepository(database_path)
    history = RunHistoryService(repository)
    test_cases, run_specs = _demo_data(demo_base_url)
    test_case_repository = SQLiteTestCaseRepository(database_path)
    plan_store = SQLitePlanStore(database_path)
    for test_case in test_cases:
        existing = test_case_repository.get(test_case.id)
        if existing is not None and existing != test_case:
            raise ValueError(
                f"Demo TestCase {test_case.name!r} already exists with different content; preserving it."
            )
    executable_case = test_cases[-1]
    _save_local_demo_plans(plan_store, executable_case, demo_base_url)
    for test_case in test_cases:
        if test_case_repository.get(test_case.id) is None:
            test_case_repository.save(test_case)
    created = 0
    skipped = 0
    run_ids: list[UUID] = []
    for test_case, test_run, workflow, outcome, setup, cleanup, started_at, finished_at in run_specs:
        run_ids.append(test_run.id)
        if history.get(test_run.id) is not None:
            skipped += 1
            continue
        history.record_completed_run(
            test_case,
            test_run,
            workflow_type=workflow,
            outcome=outcome,
            setup=setup,
            cleanup=cleanup,
            started_at=started_at,
            finished_at=finished_at,
        )
        created += 1
    return DemoSeedResult(created=created, skipped=skipped, run_ids=tuple(run_ids))


def _demo_data(demo_base_url: str = "http://127.0.0.1:8000/demo-target/registration"):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    registration = _case(
        "User registration",
        "Create an account and confirm the welcome state.",
        "https://shop.example.test/register",
        [
            ("Open registration page", "Registration form is visible.", FailurePolicy.CONTINUE),
            ("Submit valid account details", "Account is created.", FailurePolicy.BLOCK_REST),
            ("Verify welcome message", "Welcome message is displayed.", FailurePolicy.CONTINUE),
        ],
        precondition=True,
    )
    checkout = _case(
        "Checkout flow",
        "Place an order using a saved delivery address.",
        "https://shop.example.test/checkout",
        [
            ("Review order summary", "Items and total are correct.", FailurePolicy.CONTINUE),
            ("Place order", "Order confirmation is displayed.", FailurePolicy.CONTINUE),
        ],
    )
    local_demo = _case(
        "Local registration demo",
        "Run browser checks against the built-in local registration page. "
        "The confirmation assertion is intentionally unmet so the UI can show failure evidence.",
        demo_base_url,
        [
            ("Open local registration page", "The registration form is visible.", FailurePolicy.CONTINUE),
            ("Verify account confirmation", "The account confirmation is displayed.", FailurePolicy.BLOCK_REST),
            ("Verify the completed registration", "The completion state is displayed.", FailurePolicy.CONTINUE),
        ],
    )
    runs = [
        _run_spec(
            registration, "registration-validation", WorkflowType.VALIDATION, WorkflowOutcome.PASSED,
            now - timedelta(hours=2), [ExecutionStatus.PASSED] * 3, [],
            setup=True, cleanup=True,
        ),
        _run_spec(
            registration, "registration-regression", WorkflowType.REGRESSION,
            WorkflowOutcome.PRODUCT_FAILURE, now - timedelta(minutes=55),
            [ExecutionStatus.PASSED, ExecutionStatus.FAILED],
            [registration.steps[2].id], setup=True, cleanup=True,
            failure_error="Expected account confirmation, but the page showed an error.",
        ),
        _run_spec(
            checkout, "checkout-regression", WorkflowType.REGRESSION, WorkflowOutcome.PASSED,
            now - timedelta(minutes=12), [ExecutionStatus.PASSED] * 2, [],
            setup=False, cleanup=True,
        ),
    ]
    return (registration, checkout, local_demo), runs


def _save_local_demo_plans(
    plan_store: SQLitePlanStore,
    test_case: TestCase,
    base_url: str,
) -> None:
    plans = (
        QATestPlan(
            url=base_url,
            steps=[
                QATestStep(action="navigate", parameters={"url": base_url}),
                QATestStep(action="assert_text_contains", parameters={
                    "expected_text": "Registration demo",
                }),
            ],
        ),
        QATestPlan(
            url=base_url,
            steps=[
                QATestStep(action="navigate", parameters={"url": base_url}),
                QATestStep(action="assert_text_contains", parameters={
                    "expected_text": "Account confirmation displayed",
                }),
            ],
        ),
        QATestPlan(
            url=base_url,
            steps=[
                QATestStep(action="navigate", parameters={"url": base_url}),
                QATestStep(action="assert_text_contains", parameters={
                    "expected_text": "Registration demo",
                }),
            ],
        ),
    )
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    plan_specs = []
    for step, qa_test_plan in zip(test_case.steps, plans, strict=True):
        test_plan = TestPlan(
            id=_id(f"local-test-plan:{step.id}"),
            test_step_id=step.id,
            name=f"Local demo: {step.name}",
        )
        version = TestPlanVersion(
            id=_id(f"local-plan-version:{step.id}"),
            test_plan_id=test_plan.id,
            version=1,
            created_at=created_at,
            qa_test_plan=qa_test_plan,
        )
        plan_specs.append((step, test_plan, version))

    for step, test_plan, version in plan_specs:
        current_version = plan_store.find(step.id)
        current_plan = plan_store.find_test_plan(step.id)
        if current_version is not None and (
            current_version != version or current_plan != test_plan
        ):
            raise ValueError(
                f"Demo plan for {step.name!r} already exists with different content; preserving it."
            )
    for step, test_plan, version in plan_specs:
        if plan_store.find(step.id) is None:
            plan_store.save(step.id, version, test_plan=test_plan)


def _case(name, description, base_url, step_specs, *, precondition=False) -> TestCase:
    case_id = _id(f"case:{name}")
    steps = [
        TestStep(
            id=_id(f"step:{name}:{order}"),
            name=step_name,
            description=f"Perform the {step_name.lower()} action.",
            expected=expected,
            order=order,
            failure_policy=policy,
        )
        for order, (step_name, expected, policy) in enumerate(step_specs)
    ]
    preconditions = (
        [Precondition(
            id=_id("precondition:registration:account-available"),
            description="A registration email is available for this run.",
            order=0,
            provided_data_keys=["registration_email"],
        )]
        if precondition else []
    )
    return TestCase(
        id=case_id,
        name=name,
        description=description,
        base_url=base_url,
        preconditions=preconditions,
        segments=[ExecutionSegment(
            id=_id(f"segment:{name}"),
            order=0,
            base_url=base_url,
            steps=steps,
        )],
    )


def _run_spec(
    test_case: TestCase,
    key: str,
    workflow: WorkflowType,
    outcome: WorkflowOutcome,
    started_at: datetime,
    statuses: list[ExecutionStatus],
    blocked_step_ids: list[UUID],
    *,
    setup: bool,
    cleanup: bool,
    failure_error: str | None = None,
):
    run_id = _id(f"run:{key}")
    context = RunContext()
    if key == "registration-regression":
        context.set_value("demo_token", _DEMO_SECRET, sensitive=True, source="demo seed")
    elif key == "registration-validation":
        context.set_value(
            "registration_email", "qa-demo-user@example.test", source="demo seed"
        )
    executions = []
    for index, status in enumerate(statuses):
        step = test_case.steps[index]
        execution_id = _id(f"execution:{key}:{step.id}")
        attempt_started = started_at + timedelta(seconds=index * 8)
        executions.append(Execution(
            id=execution_id,
            test_step_id=step.id,
            test_plan_version_id=_id(f"plan-version:{step.id}"),
            status=status,
            started_at=attempt_started,
            finished_at=attempt_started + timedelta(seconds=4),
            actual_result=(
                "Assertion did not match the expected account confirmation."
                if status == ExecutionStatus.FAILED
                else "Expected state observed."
            ),
            error=failure_error if status == ExecutionStatus.FAILED else None,
        ))
    test_run = TestRun.from_test_case(
        test_case,
        executions,
        blocked_step_ids=blocked_step_ids,
        run_context=context,
    ).model_copy(update={"id": run_id})
    setup_outcome = None
    if setup and test_case.preconditions:
        precondition = test_case.preconditions[0]
        setup_outcome = SetupRunOutcome(
            SetupStatus.SUCCEEDED,
            (PreconditionSetupOutcome(
                precondition_id=precondition.id,
                status=SetupStatus.SUCCEEDED,
                produced_data_keys=tuple(precondition.provided_data_keys),
            ),),
        )
    finished_at = started_at + timedelta(seconds=45 if len(statuses) == 3 else 34)
    return (
        test_case,
        test_run,
        workflow,
        outcome,
        setup_outcome,
        CleanupOutcome() if cleanup else None,
        started_at,
        finished_at,
    )


def _id(name: str) -> UUID:
    return uuid5(_NAMESPACE, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed safe demo runs for the local AI QA Agent UI.")
    parser.add_argument("--database", required=True, help="SQLite database path to seed")
    parser.add_argument(
        "--demo-base-url",
        default="http://127.0.0.1:8000/demo-target/registration",
        help="local URL used by the executable registration demo plans",
    )
    args = parser.parse_args(argv)
    result = seed_demo_data(args.database, demo_base_url=args.demo_base_url)
    print(
        f"Demo history: created {result.created}, skipped {result.skipped}; "
        f"database: {Path(args.database).expanduser()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
