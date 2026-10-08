import unittest

from qa_agent.locator_recovery import RecoveryStatus, recover_locator
from qa_agent.models import (
    DiscoveryResult, DiscoveryStatus, InteractiveElement, QATestStep,
)


class LocatorRecoveryTests(unittest.TestCase):
    def element(self, selector: str, **kwargs) -> InteractiveElement:
        defaults = {"kind": "link", "tag": "a", "role": "link"}
        defaults.update(kwargs)
        return InteractiveElement(selector=selector, **defaults)

    def discover(self, *elements: InteractiveElement) -> DiscoveryResult:
        return DiscoveryResult(
            status=DiscoveryStatus.SUCCESS, url="https://example.test/",
            interactive_elements=list(elements),
        )

    def test_exact_selector_wins_for_compatible_element(self) -> None:
        plan = QATestStep(action="click", parameters={"selector": "#loans"})
        candidate = self.element("#loans")
        result = recover_locator(plan, self.discover(candidate))
        self.assertEqual(result.status, RecoveryStatus.MATCHED)
        self.assertIs(result.candidate, candidate)
        self.assertEqual(result.signals, ("exact_selector",))

    def test_exact_text_matches_only_same_intended_text(self) -> None:
        plan = QATestStep(action="click", parameters={"selector": "#old", "expected_text": "Boliglån"})
        candidate = self.element("#new", text="Boliglån")
        result = recover_locator(plan, self.discover(candidate))
        self.assertEqual(result.status, RecoveryStatus.MATCHED)
        self.assertEqual(result.signals, ("exact_text",))

    def test_compatible_type_and_exact_role_are_reported_with_identity(self) -> None:
        plan = QATestStep(action="click", parameters={
            "selector": "#old", "expected_text": "Open", "role": "button", "tag": "button"
        })
        candidate = self.element("#new", kind="button", tag="button", role="button", text="Open")
        result = recover_locator(plan, self.discover(candidate))
        self.assertEqual(result.status, RecoveryStatus.MATCHED)
        self.assertIn("exact_role", result.signals)
        self.assertIn("exact_tag", result.signals)

    def test_href_can_match_navigation_link(self) -> None:
        plan = QATestStep(action="click", parameters={"selector": "#old", "href": "/bolig"})
        candidate = self.element("#new", href="/bolig")
        result = recover_locator(plan, self.discover(candidate))
        self.assertEqual(result.status, RecoveryStatus.MATCHED)
        self.assertEqual(result.signals, ("exact_href",))

    def test_wrong_semantic_candidate_is_rejected_as_conflict(self) -> None:
        result = recover_locator(
            QATestStep(action="click", parameters={"selector": "#gone", "expected_text": "Apply"}),
            self.discover(self.element("#other", text="Cancel")),
        )
        self.assertEqual(result.status, RecoveryStatus.REJECTED_CONFLICT)
        self.assertEqual(result.original_selector, "#gone")

    def test_multiple_exact_text_candidates_are_ambiguous(self) -> None:
        plan = QATestStep(action="click", parameters={"selector": "#old", "expected_text": "Open"})
        result = recover_locator(plan, self.discover(
            self.element("#one", text="Open"), self.element("#two", text="Open")
        ))
        self.assertEqual(result.status, RecoveryStatus.AMBIGUOUS)
        self.assertIsNone(result.candidate)

    def test_incompatible_element_type_is_rejected(self) -> None:
        plan = QATestStep(action="fill", parameters={"selector": "#old", "expected_text": "Email"})
        result = recover_locator(plan, self.discover(
            InteractiveElement(kind="button", selector="#old", tag="button", text="Email")
        ))
        self.assertEqual(result.status, RecoveryStatus.NOT_FOUND)

    def test_boliglan_never_matches_billan(self) -> None:
        plan = QATestStep(action="click", parameters={"selector": "#old", "expected_text": "Boliglån"})
        result = recover_locator(plan, self.discover(self.element("#car-loan", text="Billån")))
        self.assertEqual(result.status, RecoveryStatus.REJECTED_CONFLICT)

    def test_email_input_does_not_match_phone_input(self) -> None:
        plan = QATestStep(action="fill", parameters={"selector": "#old", "expected_text": "Email"})
        candidate = InteractiveElement(
            kind="input", selector="#phone", tag="input", role="textbox",
            name="phone", accessible_name="Phone",
        )
        result = recover_locator(plan, self.discover(candidate))
        self.assertEqual(result.status, RecoveryStatus.NOT_FOUND)

    def test_ambiguous_result_never_selects_a_candidate(self) -> None:
        plan = QATestStep(action="click", parameters={"selector": "#old", "expected_text": "Menu"})
        candidates = [self.element("#one", text="Menu"), self.element("#two", text="Menu")]
        for _ in range(5):
            result = recover_locator(plan, self.discover(*candidates))
            self.assertEqual(result.status, RecoveryStatus.AMBIGUOUS)
            self.assertIsNone(result.candidate)

    def test_non_interaction_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recover_locator(QATestStep(action="navigate"), self.discover())


if __name__ == "__main__":
    unittest.main()
