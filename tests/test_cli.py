"""Focused tests for the QA agent CLI.

These tests never touch the network, launch a browser, or call a
real LLM provider: the pipeline is built from small fakes, and
``qa_agent.cli.build_pipeline`` is patched so ``main()`` never
constructs the real components.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from qa_agent.cli import (
    EXIT_FAILED,
    EXIT_PASSED,
    EXIT_USAGE,
    _build_parser,
    build_pipeline,
    main,
)
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.models import (
    DiscoveryResult,
    DiscoveryStatus,
    ExecutionStatus,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.pipeline import QATestPipeline
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.sqlite_storage import (
    SQLiteExecutionRepository,
    SQLitePlanStore,
    SQLiteRunHistoryRepository,
)
from qa_agent.test_plan_generator import (
    GeneratedTestPlan,
    LLMTestPlanGenerator,
)


class _FakeDecomposer:
    def __init__(self, test_case: DomainTestCase) -> None:
        self._test_case = test_case

    def decompose(
        self, task: str, base_url: str | None = None
    ) -> DomainTestCase:
        return self._test_case


class _FakeGenerator:
    def generate_with_plan(
        self,
        test_step: DomainTestStep,
        discovery_result: DiscoveryResult,
        *,
        existing_test_plan: DomainTestPlan | None = None,
        version_number: int = 1,
    ) -> GeneratedTestPlan:
        plan = (
            existing_test_plan
            or DomainTestPlan(test_step_id=test_step.id, name=test_step.name)
        )
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=version_number,
            qa_test_plan=_plan(),
        )
        return GeneratedTestPlan(test_plan=plan, test_plan_version=version)


class _StubProvider(LLMProvider):
    """In-process provider stub used to exercise provider fallback."""

    def __init__(
        self, name: str, *, error: Exception | None = None
    ) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.error = error

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        if self.error is not None:
            raise self.error
        return _plan()


def _plan() -> QATestPlan:
    return QATestPlan(
        url="https://example.com/",
        steps=[QATestStep(action="assert_page_loaded")],
    )


def _test_case() -> DomainTestCase:
    step = DomainTestStep(
        name="Check page",
        description="Check the example page.",
        expected="The example page is loaded.",
        order=0,
    )
    return DomainTestCase(
        name="Example flow",
        description="Open https://example.com/ and check the example page.",
        base_url="https://example.com/",
        steps=[step],
    )


def _discovery(url: str) -> DiscoveryResult:
    return DiscoveryResult(
        status=DiscoveryStatus.SUCCESS,
        url=url,
        title="Example Domain",
    )


def _broken_discovery(url: str) -> DiscoveryResult:
    raise RuntimeError("browser unavailable")


def _passing_runner(plan: QATestPlan) -> dict[str, Any]:
    return {"status": "passed", "url": plan.url, "steps": []}


def _failing_runner(plan: QATestPlan) -> dict[str, Any]:
    return {
        "status": "failed",
        "url": plan.url,
        "steps": [
            {
                "action": "assert_visible",
                "status": "failed",
                "error": "wrong title",
            }
        ],
    }


def _pipeline(
    *,
    discovery=_discovery,
    runner=_passing_runner,
    plan_store=None,
    test_case=None,
) -> QATestPipeline:
    # The same TestCase instance must be reused when a test needs the
    # pipeline to observe a cached plan: fresh instances carry fresh
    # step UUIDs, so the pipeline would (correctly) report a cache MISS.
    return QATestPipeline(
        decomposer=_FakeDecomposer(
            test_case if test_case is not None else _test_case()
        ),
        plan_generator=_FakeGenerator(),
        discovery=discovery,
        runner=runner,
        plan_store=plan_store,
    )


class CliArgumentParsingTests(unittest.TestCase):
    def test_parses_task_and_all_options(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "Open https://example.com/ and check it.",
                "--evidence-directory",
                "ev",
                "--json",
            ]
        )
        self.assertEqual(args.task, "Open https://example.com/ and check it.")
        self.assertEqual(args.evidence_directory, "ev")
        self.assertIsNone(args.database)
        self.assertTrue(args.json)

    def test_options_default_to_off(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["Open https://example.com/."])
        self.assertIsNone(args.evidence_directory)
        self.assertIsNone(args.database)
        self.assertFalse(args.json)

    def test_database_option_is_parsed(self) -> None:
        args = _build_parser().parse_args([
            "Open https://example.com/.", "--database", "history.sqlite3"
        ])
        self.assertEqual(args.database, "history.sqlite3")


class CliRunTests(unittest.TestCase):
    def test_build_pipeline_wires_all_sqlite_repositories_to_one_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "cli.sqlite3"
            with patch("qa_agent.provider_settings.ProviderSettingsService.create_router", return_value=Mock()):
                pipeline = build_pipeline(database_path=database)

            self.assertIsInstance(pipeline._plan_store, SQLitePlanStore)
            self.assertIsInstance(pipeline._execution_repository, SQLiteExecutionRepository)
            self.assertIsInstance(pipeline._run_history._repository, SQLiteRunHistoryRepository)

    def test_successful_run_returns_zero(self) -> None:
        with (
            patch("qa_agent.cli.build_pipeline", return_value=_pipeline()),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = main(["Open https://example.com/ and check the page."])

        self.assertEqual(code, EXIT_PASSED)
        text = output.getvalue()
        for section in (
            "TASK",
            "RESULT",
            "TEST CASE",
            "LLM PROVIDERS",
            "EXECUTION",
            "EVIDENCE",
            "TRACE",
        ):
            self.assertIn(section, text)
        self.assertIn(
            "task: Open https://example.com/ and check the page.", text
        )
        self.assertIn("target URL: https://example.com/", text)
        self.assertIn("status: PASSED", text)
        self.assertIn("Check page: PASSED", text)
        self.assertIn("attempts: 1", text)
        self.assertIn("plan v1", text)
        self.assertIn("no failure evidence", text)
        self.assertIn("trace_id:", text)
        self.assertIn("schema_version: 2", text)

    def test_failed_run_returns_one(self) -> None:
        pipeline = _pipeline(runner=_failing_runner)
        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = main(["Open https://example.com/ and check the page."])

        self.assertEqual(code, EXIT_FAILED)
        text = output.getvalue()
        self.assertIn("status: FAILED", text)
        self.assertIn("Check page: FAILED", text)
        self.assertIn("error: wrong title", text)

    def test_pipeline_stage_error_returns_one_and_shows_trace_id(
        self,
    ) -> None:
        pipeline = _pipeline(discovery=_broken_discovery)
        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = main(["Open https://example.com/ and check the page."])

        self.assertEqual(code, EXIT_FAILED)
        text = output.getvalue()
        self.assertIn("status: ERROR", text)
        self.assertIn(
            "failed stage: discovery (step 0: Check page)", text
        )
        self.assertIn("error: browser unavailable", text)
        self.assertIn("trace_id:", text)

    def test_unexpected_exception_returns_one(self) -> None:
        pipeline = Mock()
        pipeline.run.side_effect = RuntimeError("unexpected boom")
        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as error_output,
        ):
            code = main(["Open https://example.com/."])

        self.assertEqual(code, EXIT_FAILED)
        self.assertIn("unexpected boom", error_output.getvalue())

    def test_usage_error_returns_two(self) -> None:
        with (
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main([]), EXIT_USAGE)
            self.assertEqual(main(["--nonexistent-option"]), EXIT_USAGE)

    def test_json_outputs_valid_trace_json(self) -> None:
        pipeline = _pipeline()
        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = main(["Open https://example.com/.", "--json"])

        self.assertEqual(code, EXIT_PASSED)
        trace = json.loads(output.getvalue())
        self.assertEqual(trace["status"], "PASSED")
        self.assertEqual(trace["schema_version"], "2")
        self.assertIn("trace_id", trace)
        self.assertEqual(trace["totals"]["steps"], 1)
        self.assertEqual(trace["totals"]["execution_attempts"], 1)

    def test_json_failed_run_prints_pure_machine_readable_trace(self) -> None:
        # P2-5a: a failed QA result with --json must be exit 1 and stdout
        # must be exactly one JSON document — no human section headers,
        # no traceback, and the failure detail lives inside the trace.
        pipeline = _pipeline(runner=_failing_runner)
        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            redirect_stdout(io.StringIO()) as output,
            redirect_stderr(io.StringIO()) as error_output,
        ):
            code = main(["Open https://example.com/ and check the page.", "--json"])

        self.assertEqual(code, EXIT_FAILED)
        text = output.getvalue()
        # Whole stdout parses as JSON: nothing else was printed.
        trace = json.loads(text)
        self.assertEqual(trace["status"], "FAILED")
        self.assertEqual(trace["schema_version"], "2")
        self.assertIn("trace_id", trace)
        # Failure/error information is present inside the serialized trace.
        attempt = trace["steps"][0]["execution_attempts"][0]
        self.assertEqual(attempt["status"], "FAILED")
        self.assertEqual(attempt["error"], "wrong title")
        # No human-readable headers and no traceback leaked into stdout.
        self.assertNotIn("status:", text)
        self.assertNotIn("Traceback", text)
        self.assertEqual(error_output.getvalue(), "")

    def test_json_pipeline_stage_error_prints_redacted_trace_json(
        self,
    ) -> None:
        # P2-5b: a PipelineStageError with --json must be exit 1 with a
        # valid, redacted JSON trace on stdout — status ERROR, the failed
        # stage, the failure reason (secrets replaced), exactly one
        # "Pipeline stage" prefix, and no traceback or human headers.
        secret = "unit-test-secret-value"

        def secret_leaking_discovery(url: str) -> DiscoveryResult:
            raise RuntimeError(f"browser unavailable token={secret}")

        pipeline = _pipeline(discovery=secret_leaking_discovery)
        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            patch.dict("os.environ", {"GEMINI_API_KEY": secret}),
            redirect_stdout(io.StringIO()) as output,
            redirect_stderr(io.StringIO()) as error_output,
        ):
            code = main(["Open https://example.com/ and check the page.", "--json"])

        self.assertEqual(code, EXIT_FAILED)
        text = output.getvalue()
        trace = json.loads(text)
        self.assertEqual(trace["status"], "ERROR")
        self.assertIn("discovery", trace["error_stage"] or "")
        self.assertIn("browser unavailable", trace["error"] or "")
        # The composed stage message appears exactly once (no duplicated
        # "Pipeline stage ... failed" prefix in the JSON output).
        self.assertEqual((trace["error"] or "").count("Pipeline stage"), 1)
        # Secret is redacted and absent from the whole output.
        self.assertIn("[REDACTED]", trace["error"] or "")
        self.assertNotIn(secret, text)
        # No traceback and no human-readable summary in stdout/stderr.
        self.assertNotIn("Traceback", text)
        self.assertNotIn("status:", text)
        self.assertEqual(error_output.getvalue(), "")

    def test_evidence_directory_is_passed_through(self) -> None:
        pipeline = _pipeline()
        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline) as factory,
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                ["Open https://example.com/.", "--evidence-directory", "ev-dir"]
            )

        self.assertEqual(code, EXIT_PASSED)
        self.assertEqual(
            factory.call_args.kwargs.get("evidence_directory"),
            "ev-dir",
        )

    def test_provider_summary_shows_attempts_and_selection(self) -> None:
        failing = _StubProvider(
            "failing", error=RetryableLLMError("request failed with HTTP 429")
        )
        succeeding = _StubProvider("succeeding")
        pipeline = QATestPipeline(
            decomposer=_FakeDecomposer(_test_case()),
            plan_generator=LLMTestPlanGenerator(
                LLMRouter([failing, succeeding])
            ),
            discovery=_discovery,
            runner=_passing_runner,
        )

        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = main(["Open https://example.com/."])

        self.assertEqual(code, EXIT_PASSED)
        text = output.getvalue()
        self.assertIn("TEST_PLAN requests:", text)
        self.assertIn("failing (failing-model): RETRYABLE_ERROR", text)
        self.assertIn("succeeding (succeeding-model): SUCCESS", text)
        self.assertIn("[selected]", text)

    def test_cache_hit_reports_no_provider_calls(self) -> None:
        test_case = _test_case()
        step = test_case.steps[0]
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=1,
            qa_test_plan=_plan(),
        )
        store = InMemoryPlanStore()
        store.save(step.id, version, test_plan=plan)
        pipeline = _pipeline(plan_store=store, test_case=test_case)

        with (
            patch("qa_agent.cli.build_pipeline", return_value=pipeline),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = main(["Open https://example.com/."])

        self.assertEqual(code, EXIT_PASSED)
        text = output.getvalue()
        self.assertIn(
            "all plans loaded from cache; no LLM provider calls were made",
            text,
        )
        self.assertIn("attempts: 1", text)
        self.assertIn("plan v1", text)


if __name__ == "__main__":
    unittest.main()
