import unittest

from qa_agent.locator_recovery import RecoveryStatus, identity_for_element, recover_locator
from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    InteractiveElement,
    LocatorIdentityEntry,
    QATestStep,
)


class LocatorRecoveryIdentityTests(unittest.TestCase):
    def original(self, **values):
        return LocatorIdentityEntry(step_index=0, **values)

    def discover(self, *elements):
        return DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://example.test/",
            interactive_elements=list(elements),
        )

    def fill_control(self, selector, label, **values):
        return InteractiveElement(
            kind="input", selector=selector, tag="input", role="textbox",
            label=label, accessible_name=label, **values,
        )

    def checkbox_control(self, selector, label):
        return InteractiveElement(
            kind="checkbox", selector=selector, tag="input", role="checkbox",
            label=label, accessible_name=label,
        )

    def test_label_and_accessible_name_are_sufficient_with_compatible_type(self):
        plan = QATestStep(action="fill", parameters={"selector": "#old"})
        original = self.original(
            label="Email", accessible_name="Email", tag="input", role="textbox"
        )
        candidate = self.fill_control("#new", "Email")

        result = recover_locator(plan, self.discover(candidate), original)

        self.assertEqual(result.status, RecoveryStatus.MATCHED_HIGH_CONFIDENCE)
        self.assertIs(result.candidate, candidate)
        self.assertIn("exact_label", result.signals)
        self.assertIn("exact_accessible_name", result.signals)

    def test_stable_test_id_survives_weak_role_and_tag_change(self):
        plan = QATestStep(action="fill", parameters={"selector": "#old"})
        original = self.original(
            test_id="profile-name", accessible_name="Profile name",
            tag="input", role="textbox",
        )
        candidate = InteractiveElement(
            kind="combobox", selector="#new", tag="textarea", role="combobox",
            test_id="profile-name", accessible_name="Profile name",
        )

        result = recover_locator(plan, self.discover(candidate), original)

        self.assertEqual(result.status, RecoveryStatus.MATCHED_HIGH_CONFIDENCE)
        self.assertIn("exact_test_id", result.signals)

    def test_placeholder_match_is_acceptable_but_not_high_confidence(self):
        plan = QATestStep(action="fill", parameters={"selector": "#old"})
        original = self.original(
            placeholder="Search products", tag="input", role="textbox"
        )
        candidate = self.fill_control(
            "#new", "", placeholder="Search products"
        )

        result = recover_locator(plan, self.discover(candidate), original)

        self.assertEqual(result.status, RecoveryStatus.MATCHED_ACCEPTABLE)

    def test_structure_only_is_never_enough(self):
        plan = QATestStep(action="fill", parameters={"selector": "#old"})
        original = self.original(tag="input", role="textbox")
        candidate = InteractiveElement(
            kind="input", selector="#new", tag="input", role="textbox"
        )

        result = recover_locator(plan, self.discover(candidate), original)

        self.assertEqual(result.status, RecoveryStatus.NO_MATCH)
        self.assertIsNone(result.candidate)

    def test_explicit_identity_conflicts_are_rejected_without_label_special_cases(self):
        pairs = (
            ("Email", "Phone", "fill"),
            ("Username", "Password", "fill"),
            ("Submit", "Cancel", "click"),
            ("First name", "Last name", "fill"),
        )
        for expected, discovered, action in pairs:
            with self.subTest(expected=expected, discovered=discovered):
                is_click = action == "click"
                plan = QATestStep(action=action, parameters={"selector": "#old"})
                original = self.original(
                    accessible_name=expected,
                    label=None if is_click else expected,
                    visible_text=expected if is_click else None,
                    tag="button" if is_click else "input",
                    role="button" if is_click else "textbox",
                )
                candidate = InteractiveElement(
                    kind="button" if is_click else "input",
                    selector="#new",
                    tag="button" if is_click else "input",
                    role="button" if is_click else "textbox",
                    accessible_name=discovered,
                    label="" if is_click else discovered,
                    text=discovered if is_click else "",
                )

                result = recover_locator(plan, self.discover(candidate), original)

                self.assertEqual(result.status, RecoveryStatus.REJECTED_CONFLICT)
                self.assertIsNone(result.candidate)
                self.assertNotIn(expected, result.reason)
                self.assertNotIn(discovered, result.reason)

    def test_old_selector_does_not_override_identity_conflict(self):
        plan = QATestStep(action="fill", parameters={"selector": "#email"})
        original = self.original(
            accessible_name="Email", label="Email", tag="input", role="textbox"
        )
        candidate = self.fill_control("#email", "Phone")

        result = recover_locator(plan, self.discover(candidate), original)

        self.assertEqual(result.status, RecoveryStatus.REJECTED_CONFLICT)

    def test_duplicate_labels_are_ambiguous_and_never_dom_order_selected(self):
        plan = QATestStep(action="fill", parameters={"selector": "#old"})
        original = self.original(label="Email", tag="input", role="textbox")
        candidates = (
            self.fill_control("#one", "Email"),
            self.fill_control("#two", "Email"),
        )

        result = recover_locator(plan, self.discover(*candidates), original)

        self.assertEqual(result.status, RecoveryStatus.AMBIGUOUS)
        self.assertIsNone(result.candidate)

    def test_stale_legacy_plan_with_only_tag_and_role_is_not_repaired(self):
        plan = QATestStep(action="fill", parameters={"selector": "#old"})
        candidate = self.fill_control("#new", "Email")

        result = recover_locator(plan, self.discover(candidate))

        self.assertEqual(result.status, RecoveryStatus.NO_MATCH)

    def test_checkbox_recovery_preserves_checkbox_label_identity(self):
        plan = QATestStep(action="check", parameters={"selector": "#old"})
        original = self.original(
            accessible_name="Marketing emails", label="Marketing emails",
            tag="input", role="checkbox",
        )

        result = recover_locator(
            plan,
            self.discover(self.checkbox_control("#newsletter", "Product updates")),
            original,
        )

        self.assertEqual(result.status, RecoveryStatus.REJECTED_CONFLICT)
        self.assertIsNone(result.candidate)
        self.assertNotIn("Marketing emails", result.reason)
        self.assertNotIn("Product updates", result.reason)

    def test_checkbox_identity_metadata_contains_label_but_no_entered_value(self):
        checkbox = self.checkbox_control("#marketing", "Marketing emails")

        identity = identity_for_element(0, checkbox, "check")

        self.assertEqual(identity.accessible_name, "Marketing emails")
        self.assertEqual(identity.label, "Marketing emails")
        self.assertIsNone(identity.visible_text)

    def test_modal_control_does_not_recover_to_a_similar_page_control(self):
        plan = QATestStep(action="click", parameters={"selector": "#old-confirm"})
        original = self.original(
            accessible_name="Confirm", visible_text="Confirm",
            tag="button", role="button", dialog_identity="id:details-dialog",
        )
        page_control = InteractiveElement(
            kind="button", selector="button.confirm", tag="button", role="button",
            accessible_name="Confirm", text="Confirm",
        )

        result = recover_locator(plan, self.discover(page_control), original)
        same_dialog_control = InteractiveElement(
            kind="button", selector="#new-confirm", tag="button", role="button",
            accessible_name="Confirm", text="Confirm",
            dialog_identity="id:details-dialog",
        )
        same_dialog_result = recover_locator(
            plan, self.discover(same_dialog_control), original
        )

        self.assertEqual(result.status, RecoveryStatus.NO_MATCH)
        self.assertIsNone(result.candidate)
        self.assertEqual(same_dialog_result.status, RecoveryStatus.MATCHED_HIGH_CONFIDENCE)
        self.assertIs(same_dialog_result.candidate, same_dialog_control)


if __name__ == "__main__":
    unittest.main()
