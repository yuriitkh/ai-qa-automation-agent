import unittest

from qa_agent.models import TestCase as DomainTestCase, TestStep as DomainTestStep
from qa_agent.test_case_decomposer import TestCaseDecomposer


class TestCaseDecomposerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decomposer = TestCaseDecomposer()

    def test_empty_or_invalid_task_is_rejected(self) -> None:
        for task in ("", "   ", None, 42):
            with self.subTest(task=task):
                with self.assertRaises(ValueError):
                    self.decomposer.decompose(task)  # type: ignore[arg-type]

    def test_simple_single_step_task(self) -> None:
        case = self.decomposer.decompose('Verify the page title is "Example".')

        self.assertEqual(len(case.steps), 1)
        self.assertEqual(case.steps[0].expected, "Page heading is Example")

    def test_multi_step_navigation_task_is_decomposed(self) -> None:
        task = (
            "Verify that DNB navigation works from the homepage. Open the menu, "
            "select Lån → Boliglån, then verify the URL and page heading."
        )

        case = self.decomposer.decompose(task, base_url="https://www.dnb.no")

        self.assertEqual(
            [step.description for step in case.steps],
            [
                "Open the homepage",
                "Open the main navigation menu",
                "Open Lån → Boliglån",
                "Verify the URL",
                "Verify the page heading",
            ],
        )
        self.assertEqual(case.steps[2].expected, "Boliglån page is opened")
        self.assertEqual(case.steps[3].expected, "URL is the expected Boliglån URL")
        self.assertEqual(case.steps[4].expected, "Page heading is Boliglån")

    def test_step_order_starts_at_zero_and_is_sequential(self) -> None:
        case = self.decomposer.decompose("Open the homepage and verify the URL.")

        self.assertEqual([step.order for step in case.steps], list(range(len(case.steps))))

    def test_every_step_has_description_and_expected(self) -> None:
        case = self.decomposer.decompose("Open the homepage, then verify the URL.")

        self.assertTrue(all(step.description.strip() for step in case.steps))
        self.assertTrue(all(step.expected.strip() for step in case.steps))

    def test_independent_checks_are_separate_atomic_steps(self) -> None:
        case = self.decomposer.decompose(
            "Open the homepage, then verify the URL and page heading."
        )

        descriptions = [step.description for step in case.steps]
        self.assertIn("Verify the URL", descriptions)
        self.assertIn("Verify the page heading", descriptions)
        self.assertNotIn("Verify the URL and page heading", descriptions)

    def test_result_is_a_compatible_test_case_with_test_steps(self) -> None:
        case = self.decomposer.decompose("Open the homepage.", "https://example.com")

        self.assertIsInstance(case, DomainTestCase)
        self.assertTrue(all(isinstance(step, DomainTestStep) for step in case.steps))
        self.assertEqual(case.description, "Open the homepage.")
        self.assertEqual(case.base_url, "https://example.com")

    def test_repeated_decomposition_keeps_case_and_step_ids_stable(self) -> None:
        task = "Open the homepage and verify the URL."
        first = self.decomposer.decompose(task, "https://example.com/")
        second = self.decomposer.decompose(task, "https://example.com/")

        self.assertEqual(first.id, second.id)
        self.assertEqual(
            [step.id for step in first.steps],
            [step.id for step in second.steps],
        )

    def test_different_task_contexts_do_not_share_step_ids(self) -> None:
        first = self.decomposer.decompose("Open the homepage.", "https://example.com/")
        second = self.decomposer.decompose("Open the homepage.", "https://example.org/")

        self.assertNotEqual(first.id, second.id)
        self.assertNotEqual(first.steps[0].id, second.steps[0].id)


if __name__ == "__main__":
    unittest.main()
