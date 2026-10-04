import unittest
from uuid import uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    Execution,
    ExecutionStatus,
    TestCase as DomainTestCase,
    TestStep as DomainTestStep,
)


class InMemoryExecutionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryExecutionRepository()
        self.step_a = self.make_step("Step A", 0)
        self.step_b = self.make_step("Step B", 1)

    @staticmethod
    def make_step(name: str, order: int) -> DomainTestStep:
        return DomainTestStep(
            name=name,
            description=f"Run {name}.",
            expected=f"{name} passes.",
            order=order,
        )

    @staticmethod
    def make_execution(
        test_step_id,
        *,
        status: ExecutionStatus = ExecutionStatus.PASSED,
        execution_id=None,
    ) -> Execution:
        return Execution(
            id=execution_id or uuid4(),
            test_step_id=test_step_id,
            test_plan_version_id=uuid4(),
            status=status,
            actual_result=status.value,
        )

    def test_save_and_get_returns_saved_execution(self) -> None:
        execution = self.make_execution(self.step_a.id)

        self.repository.save(execution)

        self.assertIs(self.repository.get(execution.id), execution)

    def test_unknown_execution_and_step_have_empty_results(self) -> None:
        self.assertIsNone(self.repository.get(uuid4()))
        self.assertEqual(self.repository.list_for_test_step(uuid4()), [])

    def test_duplicate_execution_id_is_rejected_without_replacing_record(self) -> None:
        execution_id = uuid4()
        original = self.make_execution(self.step_a.id, execution_id=execution_id)
        duplicate = self.make_execution(
            self.step_a.id,
            status=ExecutionStatus.FAILED,
            execution_id=execution_id,
        )
        self.repository.save(original)

        with self.assertRaises(ValueError):
            self.repository.save(duplicate)

        self.assertIs(self.repository.get(execution_id), original)

    def test_step_history_is_isolated_and_keeps_insertion_order(self) -> None:
        first_a = self.make_execution(self.step_a.id)
        only_b = self.make_execution(self.step_b.id)
        second_a = self.make_execution(self.step_a.id, status=ExecutionStatus.FAILED)
        for execution in (first_a, only_b, second_a):
            self.repository.save(execution)

        self.assertEqual(
            self.repository.list_for_test_step(self.step_a.id),
            [first_a, second_a],
        )
        self.assertEqual(self.repository.list_for_test_step(self.step_b.id), [only_b])

    def test_test_case_history_uses_step_ids_and_preserves_global_order(self) -> None:
        first_a = self.make_execution(self.step_a.id)
        only_b = self.make_execution(self.step_b.id)
        second_a = self.make_execution(self.step_a.id, status=ExecutionStatus.FAILED)
        unrelated = self.make_execution(uuid4())
        for execution in (first_a, only_b, second_a, unrelated):
            self.repository.save(execution)
        test_case = DomainTestCase(
            name="Two step case",
            description="Run steps A and B.",
            steps=[self.step_a, self.step_b],
        )

        self.assertEqual(
            self.repository.list_for_test_case(test_case),
            [first_a, only_b, second_a],
        )


if __name__ == "__main__":
    unittest.main()
