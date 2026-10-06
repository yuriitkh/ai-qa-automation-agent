import tempfile
import unittest
from pathlib import Path

from qa_agent.models import (
    ExecutionSegment,
    Precondition,
    TestCase as DomainTestCase,
    TestStep as DomainTestStep,
)
from qa_agent.sqlite_storage import SQLiteTestCaseRepository
from qa_agent.test_case_repository import InMemoryTestCaseRepository


def _case() -> DomainTestCase:
    first = DomainTestStep(
        name="Open registration",
        description="Open the local registration page.",
        expected="The page is visible.",
        order=0,
    )
    second = DomainTestStep(
        name="Check confirmation",
        description="Check the account confirmation.",
        expected="The account is confirmed.",
        order=1,
    )
    return DomainTestCase(
        name="Registration",
        description="A persisted definition used by local tests.",
        base_url="http://127.0.0.1:8000/register",
        preconditions=[Precondition(
            description="A local test email is available.",
            order=0,
            provided_data_keys=["email"],
        )],
        segments=[ExecutionSegment(
            order=0,
            base_url="http://127.0.0.1:8000/register",
            steps=[first, second],
        )],
    )


class TestCaseRepositoryTests(unittest.TestCase):
    def test_sqlite_repository_round_trips_canonical_definition_without_run_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "definitions.sqlite3"
            original = _case()
            repository = SQLiteTestCaseRepository(database)
            repository.save(original)

            restored = SQLiteTestCaseRepository(database).get(original.id)
            self.assertEqual(restored, original)
            self.assertEqual([step.id for step in restored.steps], [step.id for step in original.steps])
            self.assertEqual(restored.segments[0].id, original.segments[0].id)
            self.assertEqual(restored.preconditions, original.preconditions)
            self.assertEqual(restored.base_url, original.base_url)
            self.assertEqual(SQLiteTestCaseRepository(database).list(), [original])

    def test_saving_an_existing_id_updates_the_definition_and_keeps_one_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteTestCaseRepository(Path(directory) / "definitions.sqlite3")
            original = _case()
            repository.save(original)
            updated = original.model_copy(update={"description": "Updated definition."})
            repository.save(updated)

            self.assertEqual(repository.get(original.id).description, "Updated definition.")
            self.assertEqual([item.id for item in repository.list()], [original.id])

    def test_in_memory_repository_returns_isolated_snapshots(self) -> None:
        repository = InMemoryTestCaseRepository()
        original = _case()
        repository.save(original)
        loaded = repository.get(original.id)
        loaded.segments[0].steps[0].name = "Changed in caller"

        self.assertEqual(repository.get(original.id).steps[0].name, "Open registration")


if __name__ == "__main__":
    unittest.main()
