import io
import json
import time
import unittest
from contextlib import redirect_stdout
from threading import Event

from qa_agent.background_authoring import BackgroundAuthoringService
from qa_agent.execution_progress import AuthoringEventType, ExecutionProgressStore
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan
from qa_agent.test_case_authoring import TestCaseAuthoringService, TestCaseDraftStore


_VALID_RESPONSE = (
    '{"preconditions":[],"segments":[{"steps":[{"name":"Open page",'
    '"description":"Open the requested page.","expected":"The page is shown."}]}]}'
)


class AuthoringProvider(LLMProvider):
    def __init__(self, *, output=_VALID_RESPONSE, error=None, entered=None, release=None):
        self.output = output
        self.error = error
        self.entered = entered
        self.release = release
        self.calls = 0

    def create_test_plan(self, task, target_url, page_snapshot):
        return QATestPlan(url=target_url, steps=[{"action": "assert_page_loaded"}])

    def create_structured_output(self, prompt, schema, schema_name):
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(5)
        if self.error is not None:
            raise self.error
        return self.output


class BackgroundAuthoringTests(unittest.TestCase):
    def make_service(self, providers, *, workers=2, pending=4):
        progress = ExecutionProgressStore()
        drafts = TestCaseDraftStore()
        service = BackgroundAuthoringService(
            TestCaseAuthoringService(LLMRouter(providers)),
            drafts,
            progress,
            max_workers=workers,
            max_pending=pending,
        )
        return service, progress, drafts

    def wait_for(self, store, progress_id, *, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = store.get_authoring(progress_id)
            if snapshot is not None and snapshot.finished:
                return snapshot
            time.sleep(0.005)
        self.fail("Authoring job did not finish in time.")

    def test_success_fallback_creates_one_draft_and_emits_real_ordered_events(self):
        primary = AuthoringProvider(error=RetryableLLMError(
            "API key private-token; raw provider response; HTTP 429 rate limit"
        ))
        primary.name = "groq"
        fallback = AuthoringProvider()
        fallback.name = "gemini"
        service, progress, drafts = self.make_service([primary, fallback])
        try:
            with redirect_stdout(io.StringIO()):
                progress_id = service.start(
                    name="Register user",
                    base_url="http://127.0.0.1:8000/register",
                    scenario="Register a user and verify the page.",
                )
                snapshot = self.wait_for(progress, progress_id)
            self.assertTrue(snapshot.success)
            self.assertEqual(snapshot.error_category, None)
            self.assertEqual(primary.calls, 1)
            self.assertEqual(fallback.calls, 1)
            self.assertEqual(len(drafts._drafts), 1)
            token = snapshot.review_url.rsplit("/", 1)[1]
            self.assertIsNotNone(drafts.get(token))
            self.assertEqual(
                [event.event_type for event in snapshot.events],
                [
                    AuthoringEventType.AUTHORING_REQUESTED,
                    AuthoringEventType.INPUT_VALIDATED,
                    AuthoringEventType.AUTHORING_STARTED,
                    AuthoringEventType.LLM_REQUEST_STARTED,
                    AuthoringEventType.LLM_PROVIDER_SELECTED,
                    AuthoringEventType.LLM_PROVIDER_FAILED,
                    AuthoringEventType.LLM_PROVIDER_FALLBACK,
                    AuthoringEventType.LLM_PROVIDER_SELECTED,
                    AuthoringEventType.LLM_PROVIDER_COMPLETED,
                    AuthoringEventType.LLM_RESPONSE_RECEIVED,
                    AuthoringEventType.TESTCASE_VALIDATION_STARTED,
                    AuthoringEventType.TESTCASE_VALIDATED,
                    AuthoringEventType.DRAFT_CREATED,
                    AuthoringEventType.AUTHORING_FINISHED,
                ],
            )
            self.assertEqual(snapshot.provider_name, "Gemini")
            public = json.dumps(snapshot.to_public_dict())
            self.assertIn("Provider: Groq", public)
            self.assertIn("Groq failed: Rate limited [RATE_LIMIT]", public)
            self.assertIn("Groq failed [RATE_LIMIT]. Trying Gemini...", public)
            self.assertIn("Completed with Gemini", public)
            self.assertNotIn("private-token", public)
            self.assertNotIn("raw provider response", public)
        finally:
            service.close()

    def test_all_rate_limits_finish_safely_without_a_draft(self):
        providers = [
            AuthoringProvider(error=RetryableLLMError("request failed with HTTP 429")),
            AuthoringProvider(error=RetryableLLMError("rate limit reached")),
        ]
        service, progress, drafts = self.make_service(providers)
        try:
            with redirect_stdout(io.StringIO()):
                progress_id = service.start(
                    name="Rate limited",
                    base_url="https://example.test/",
                    scenario="Check the page.",
                )
                snapshot = self.wait_for(progress, progress_id)
            self.assertFalse(snapshot.success)
            self.assertEqual(snapshot.error_category, "AI_RATE_LIMIT")
            self.assertEqual(snapshot.state.value, "FAILED")
            self.assertIn("Try again later", snapshot.error_message)
            self.assertNotIn("429", snapshot.error_message)
            self.assertEqual(len(drafts._drafts), 0)
            self.assertEqual(snapshot.events[-1].event_type, AuthoringEventType.AUTHORING_FAILED)
        finally:
            service.close()

    def test_all_provider_timeouts_finish_as_failed_with_safe_terminal_message(self):
        providers = [
            AuthoringProvider(error=RetryableLLMError("provider request timed out")),
            AuthoringProvider(error=RetryableLLMError("another request timed out")),
        ]
        providers[0].name = "groq"
        providers[1].name = "gemini"
        service, progress, _drafts = self.make_service(providers)
        try:
            with redirect_stdout(io.StringIO()):
                progress_id = service.start(
                    name="Timeout case",
                    base_url="https://example.test/",
                    scenario="Check the page.",
                )
                snapshot = self.wait_for(progress, progress_id)
            self.assertEqual(snapshot.state.value, "FAILED")
            self.assertTrue(snapshot.finished)
            self.assertEqual(snapshot.error_category, "AI_TIMEOUT")
            self.assertIn("timed out", snapshot.error_message)
            self.assertEqual(snapshot.events[-1].event_type, AuthoringEventType.AUTHORING_FAILED)
        finally:
            service.close()

    def test_authoring_provider_logs_are_safe_and_name_the_attempt_and_fallback(self):
        primary = AuthoringProvider(
            error=RetryableLLMError(
                "private-key-never-log; raw-provider-body-never-log; HTTP 429"
            )
        )
        primary.name = "groq"
        fallback = AuthoringProvider()
        fallback.name = "gemini"
        router = LLMRouter([primary, fallback])
        events = []
        with redirect_stdout(io.StringIO()), self.assertLogs(
            "qa_agent.llm.router", level="INFO"
        ) as captured:
            router.create_structured_output(
                "private prompt never log",
                {},
                "test_case_authoring",
                progress_callback=lambda event, provider, **details: events.append(
                    (event, provider, details)
                ),
            )
        log_text = "\n".join(captured.output)
        self.assertIn("LLM authoring provider selected: Groq", log_text)
        self.assertIn("LLM authoring provider failed: Groq [RATE_LIMIT]", log_text)
        self.assertIn("LLM authoring fallback: Groq -> Gemini", log_text)
        self.assertIn("LLM authoring completed: Gemini", log_text)
        for unsafe in (
            "private-key-never-log",
            "raw-provider-body-never-log",
            "private prompt never log",
        ):
            self.assertNotIn(unsafe, log_text)
        self.assertEqual(
            [event[0] for event in events],
            ["selected", "failed", "fallback", "selected", "completed"],
        )

    def test_provider_failure_and_invalid_output_do_not_create_drafts(self):
        cases = (
            (AuthoringProvider(error=RetryableLLMError("provider secret 503")), "AI_PROVIDER_ERROR"),
            (AuthoringProvider(output="not json"), "AI_OUTPUT_VALIDATION_ERROR"),
            (AuthoringProvider(output='{"preconditions":[],"segments":[]}'), "AI_GENERATION_ERROR"),
        )
        for provider, category in cases:
            with self.subTest(category=category):
                service, progress, drafts = self.make_service([provider])
                try:
                    with redirect_stdout(io.StringIO()):
                        progress_id = service.start(
                            name="Failure case",
                            base_url="https://example.test/",
                            scenario="Check the page.",
                        )
                        snapshot = self.wait_for(progress, progress_id)
                    public = json.dumps(snapshot.to_public_dict())
                    self.assertEqual(snapshot.error_category, category)
                    self.assertNotIn("provider secret", public)
                    self.assertNotIn("not json", public)
                    self.assertEqual(len(drafts._drafts), 0)
                finally:
                    service.close()

    def test_blocked_provider_leaves_progress_running_without_creating_a_draft(self):
        entered = Event()
        release = Event()
        provider = AuthoringProvider(entered=entered, release=release)
        provider.name = "groq"
        service, progress, drafts = self.make_service([provider])
        try:
            with redirect_stdout(io.StringIO()):
                progress_id = service.start(
                    name="Slow case",
                    base_url="https://example.test/",
                    scenario="Check a slow provider.",
                )
                self.assertTrue(entered.wait(1))
                snapshot = progress.get_authoring(progress_id)
                self.assertFalse(snapshot.finished)
                self.assertEqual(snapshot.phase, "Generating TestCase")
                # Provider selection is reported before the blocking provider call.
                # The active progress read therefore shows which provider is in use.
                snapshot = progress.get_authoring(progress_id)
                self.assertEqual(snapshot.provider_name, "Groq")
                elapsed_at_first_poll = snapshot.elapsed_ms
                time.sleep(0.04)
                self.assertGreater(
                    progress.get_authoring(progress_id).elapsed_ms,
                    elapsed_at_first_poll,
                )
                self.assertEqual(provider.calls, 1)
                self.assertEqual(len(drafts._drafts), 0)
                release.set()
                finished = self.wait_for(progress, progress_id)
                self.assertTrue(finished.success)
                frozen_elapsed = finished.elapsed_ms
                time.sleep(0.02)
                self.assertEqual(
                    progress.get_authoring(progress_id).elapsed_ms,
                    frozen_elapsed,
                )
        finally:
            release.set()
            service.close()

    def test_authoring_queue_is_bounded_and_finishes_rejected_requests_safely(self):
        entered = Event()
        release = Event()
        provider = AuthoringProvider(entered=entered, release=release)
        service, progress, drafts = self.make_service([provider], workers=1, pending=0)
        try:
            first_id = service.start(
                name="First request",
                base_url="https://example.test/",
                scenario="Check the page.",
            )
            self.assertTrue(entered.wait(1))
            second_id = service.start(
                name="Queued request",
                base_url="https://example.test/",
                scenario="Check a second page.",
            )
            rejected = self.wait_for(progress, second_id)
            self.assertEqual(rejected.error_category, "AUTHORING_EXECUTION_ERROR")
            self.assertIn("queue is busy", rejected.error_message)
            self.assertEqual(provider.calls, 1)
            release.set()
            self.assertTrue(self.wait_for(progress, first_id).success)
            self.assertEqual(len(drafts._drafts), 1)
        finally:
            release.set()
            service.close()

    def test_unexpected_background_error_is_logged_and_hidden(self):
        class BrokenAuthoring:
            @staticmethod
            def validate_input(name, scenario, base_url):
                return type("Input", (), {"name": name, "scenario": scenario, "base_url": base_url})()

            @staticmethod
            def generate(*_args, **_kwargs):
                raise RuntimeError("secret/path/stack must not be public")

        progress = ExecutionProgressStore()
        drafts = TestCaseDraftStore()
        service = BackgroundAuthoringService(BrokenAuthoring(), drafts, progress)
        try:
            progress_id = service.start(
                name="Broken case",
                base_url="http://127.0.0.1/",
                scenario="Check the page.",
            )
            snapshot = self.wait_for(progress, progress_id)
            public = json.dumps(snapshot.to_public_dict())
            self.assertEqual(snapshot.error_category, "AUTHORING_EXECUTION_ERROR")
            self.assertEqual(snapshot.error_message, "An unexpected authoring error occurred.")
            self.assertNotIn("secret/path/stack", public)
            self.assertEqual(len(drafts._drafts), 0)
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
