from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from qa_agent.execution_progress import (
    AuthoringEventType,
    ExecutionProgressStore,
    ProgressState,
)


class AuthoringProgressStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
        self.store = ExecutionProgressStore(
            clock=lambda: self.now,
            finished_ttl=timedelta(minutes=5),
        )

    def create_requested(self) -> str:
        progress_id = self.store.create_authoring(
            name="Registration scenario",
            base_url="https://example.test/register?access_token=private#fragment",
            scenario="Secret scenario text must remain internal.",
        )
        self.store.append_authoring(progress_id, AuthoringEventType.AUTHORING_REQUESTED)
        self.store.append_authoring(progress_id, AuthoringEventType.INPUT_VALIDATED)
        return progress_id

    def test_creation_events_elapsed_time_and_allowlisted_serialization(self) -> None:
        progress_id = self.create_requested()
        with self.assertRaises(ValueError):
            UUID(progress_id)
        self.now += timedelta(seconds=2, milliseconds=500)
        self.store.append_authoring(progress_id, AuthoringEventType.AUTHORING_STARTED)
        snapshot = self.store.get_authoring(progress_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.kind, "AUTHORING")
        self.assertEqual(snapshot.state, ProgressState.RUNNING)
        self.assertEqual(snapshot.phase, "Understanding scenario")
        self.assertEqual(snapshot.elapsed_ms, 0)
        self.assertIsNotNone(snapshot.started_at)

        self.now += timedelta(seconds=3)
        snapshot = self.store.get_authoring(progress_id)
        self.assertEqual(snapshot.elapsed_ms, 3000)
        public = snapshot.to_public_dict()
        self.assertEqual(public["workflow"], "AUTHORING")
        self.assertEqual(public["test_case_name"], "Registration scenario")
        self.assertEqual(public["base_url"], "https://example.test/register")
        self.assertNotIn("scenario", public)
        self.assertNotIn("access_token", str(public))
        self.assertNotIn("fragment", str(public))
        self.assertEqual(
            [event["type"] for event in public["events"]],
            ["AUTHORING_REQUESTED", "INPUT_VALIDATED", "AUTHORING_STARTED"],
        )

    def test_success_associates_review_url_and_cannot_return_to_running(self) -> None:
        progress_id = self.create_requested()
        events = (
            AuthoringEventType.AUTHORING_STARTED,
            AuthoringEventType.LLM_REQUEST_STARTED,
            AuthoringEventType.LLM_RESPONSE_RECEIVED,
            AuthoringEventType.TESTCASE_VALIDATION_STARTED,
            AuthoringEventType.TESTCASE_VALIDATED,
            AuthoringEventType.DRAFT_CREATED,
        )
        for event in events:
            self.store.append_authoring(progress_id, event)
        self.store.finish_authoring(progress_id, review_url="/test-cases/review/opaque-draft")
        before = self.store.get_authoring(progress_id)
        self.assertEqual(before.state, ProgressState.FINISHED)
        self.assertTrue(before.finished)
        self.assertTrue(before.success)
        self.assertEqual(before.phase, "Ready for review")
        self.assertEqual(before.review_url, "/test-cases/review/opaque-draft")
        self.assertEqual(before.events[-1].event_type, AuthoringEventType.AUTHORING_FINISHED)
        with self.assertRaises(RuntimeError):
            self.store.append_authoring(progress_id, AuthoringEventType.AUTHORING_STARTED)
        self.store.finish_authoring(
            progress_id,
            error_category="AUTHORING_EXECUTION_ERROR",
            error_message="must not replace first terminal result",
        )
        self.assertEqual(self.store.get_authoring(progress_id), before)

    def test_failure_is_safe_and_expired_progress_returns_none(self) -> None:
        progress_id = self.create_requested()
        with patch.dict("os.environ", {"AUTHORING_TEST_API_KEY": "private-provider-key"}):
            self.store.finish_authoring(
                progress_id,
                error_category="AI_PROVIDER_ERROR",
                error_message="Provider said private-provider-key",
            )
            snapshot = self.store.get_authoring(progress_id)
        self.assertFalse(snapshot.success)
        self.assertEqual(snapshot.error_category, "AI_PROVIDER_ERROR")
        self.assertNotIn("private-provider-key", str(snapshot.to_public_dict()))
        self.assertFalse(snapshot.review_url)
        self.now += timedelta(minutes=6)
        self.assertIsNone(self.store.get_authoring(progress_id))

    def test_event_order_is_enforced(self) -> None:
        progress_id = self.store.create_authoring(
            name="Case", base_url="http://127.0.0.1/", scenario="Check the page."
        )
        with self.assertRaises(ValueError):
            self.store.append_authoring(progress_id, AuthoringEventType.AUTHORING_STARTED)

    def test_shared_store_keeps_execution_and_authoring_kinds_separate(self) -> None:
        execution_id = self.store.create(uuid4(), "AUTOMATION")
        authoring_id = self.store.create_authoring(
            name="Authoring",
            base_url="https://example.test/",
            scenario="Private scenario.",
        )
        self.assertNotEqual(execution_id, authoring_id)
        self.assertEqual(self.store.get(execution_id).to_public_dict()["kind"], "EXECUTION")
        self.assertEqual(self.store.get_authoring(authoring_id).to_public_dict()["kind"], "AUTHORING")


if __name__ == "__main__":
    unittest.main()
