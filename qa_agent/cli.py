"""Clean command-line interface for the AI QA Automation Agent.

The CLI is a thin presentation layer. It parses arguments, builds the
existing pipeline from existing components, runs it, and formats
``PipelineResult`` and ``ExecutionTrace`` for humans. It owns no QA
business logic: decomposition, discovery, plan generation, execution,
locator recovery, regeneration, provider fallback, and secret redaction
all remain inside the pipeline, the LLM router, and the trace.
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence
from uuid import UUID

from qa_agent.execution_trace import ExecutionTrace, RequestKind
from qa_agent.llm.registry import create_router
from qa_agent.models import ExecutionStatus
from qa_agent.pipeline import PipelineResult, PipelineStageError, QATestPipeline
from qa_agent.redaction import safe_failure_reason
from qa_agent.storage import create_sqlite_storage
from qa_agent.llm_usage import LLMUsageService
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.reliability import AutomationReliabilitySupervisor

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def build_pipeline(
    evidence_directory: str | Path | None = None,
    database_path: str | Path | None = None,
) -> QATestPipeline:
    """Build the real pipeline from existing components and defaults.

    Deterministic browser discovery, the Playwright ``BrowserRunner``,
    the in-memory plan store, and the in-memory execution repository
    are backed by the shared SQLite database so completed runs remain
    available to reports and the local Web UI.
    """
    storage = create_sqlite_storage(database_path)
    usage_service = LLMUsageService(storage.llm_usage_repository)
    router = create_router(usage_recorder=usage_service)
    return QATestPipeline(
        decomposer=TestCaseDecomposer(),
        plan_generator=LLMTestPlanGenerator(router, AutomationReliabilitySupervisor(storage.reliability_repository)),
        evidence_directory=evidence_directory,
        plan_store=storage.plan_store,
        execution_repository=storage.execution_repository,
        run_history=storage.run_history,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI invocation and return the process exit code."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        # argparse already printed usage or help; preserve its exit
        # status (2 for usage errors, 0 for --help).
        code = error.code
        if code is None:
            return EXIT_PASSED
        return code if isinstance(code, int) else EXIT_USAGE

    pipeline = build_pipeline(
        evidence_directory=args.evidence_directory,
        database_path=args.database,
    )
    try:
        result = pipeline.run(args.task)
    except PipelineStageError as error:
        return _handle_pipeline_error(args, error)
    except Exception as error:
        print(
            f"ERROR: unexpected failure: {safe_failure_reason(error)}",
            file=sys.stderr,
        )
        return EXIT_FAILED

    if args.json:
        _print_trace_json(result.trace)
    else:
        _print_summary(result)
    return (
        EXIT_PASSED
        if result.test_run.status == ExecutionStatus.PASSED
        else EXIT_FAILED
    )


def _handle_pipeline_error(
    args: argparse.Namespace,
    error: PipelineStageError,
) -> int:
    if args.json:
        _print_trace_json(error.trace)
    else:
        _print_error_summary(error, error.trace)
    return EXIT_FAILED


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa_agent",
        description=(
            "Run a natural-language QA task with the AI QA Automation Agent."
        ),
    )
    parser.add_argument(
        "task",
        help="natural-language QA task; include the target http(s) URL in the task text",
    )
    parser.add_argument(
        "--evidence-directory",
        metavar="PATH",
        default=None,
        help="directory where failure evidence (for example screenshots) is saved",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the structured ExecutionTrace as JSON instead of the summary",
    )
    parser.add_argument(
        "--database",
        metavar="PATH",
        default=None,
        help="SQLite database path (defaults to ~/.qa_agent/qa_agent.sqlite3)",
    )
    return parser


def _print_trace_json(trace: ExecutionTrace | None) -> None:
    if trace is None:
        print("execution trace is unavailable", file=sys.stderr)
        return
    print(trace.model_dump_json(indent=2))


def _print_error_summary(
    error: PipelineStageError,
    trace: ExecutionTrace | None,
) -> None:
    error_message = (
        trace.error
        if trace is not None and trace.error
        else safe_failure_reason(error)
    )
    # The stage already has its own line above; drop the composed
    # "Pipeline stage '<stage>' failed: " prefix when present. If
    # redaction altered the prefix, keep the full redacted message.
    prefix = f"Pipeline stage '{error.stage}' failed: "
    if error_message.startswith(prefix):
        error_message = error_message[len(prefix):]
    print("RESULT")
    print("  status: ERROR")
    if trace is not None and trace.duration_ms is not None:
        print(f"  duration: {_format_duration(trace.duration_ms)}")
    print(f"  failed stage: {error.stage}")
    print(f"  error: {error_message or '(no details provided)'}")
    if trace is not None:
        print(f"  trace_id: {trace.trace_id}")


def _print_summary(result: PipelineResult) -> None:
    trace = result.trace
    _print_task_section(result, trace)
    _print_result_section(result, trace)
    _print_test_case_section(result, trace)
    _print_provider_section(trace)
    _print_execution_section(trace)
    _print_evidence_section(result)
    _print_trace_section(trace)


def _print_task_section(
    result: PipelineResult,
    trace: ExecutionTrace | None,
) -> None:
    print("TASK")
    task = trace.task if trace is not None else result.test_case.description
    print(f"  task: {task}")
    target_url = (
        trace.target_url if trace is not None else result.test_case.base_url
    )
    print(f"  target URL: {target_url or '(resolved from the task text)'}")


def _print_result_section(
    result: PipelineResult,
    trace: ExecutionTrace | None,
) -> None:
    print("RESULT")
    print(f"  status: {result.test_run.status.value}")
    if trace is not None and trace.duration_ms is not None:
        print(f"  duration: {_format_duration(trace.duration_ms)}")


def _print_test_case_section(
    result: PipelineResult,
    trace: ExecutionTrace | None,
) -> None:
    print("TEST CASE")
    print(f"  name: {result.test_case.name}")
    trace_steps = (
        {step.test_step_id: step for step in trace.steps}
        if trace is not None
        else {}
    )
    plan_versions: dict[UUID, int] = {}
    for generated in result.test_plans:
        plan_versions[generated.test_plan.test_step_id] = (
            generated.test_plan_version.version
        )
    blocked_ids = set(result.test_run.blocked_step_ids)
    for step in sorted(result.test_case.steps, key=lambda item: item.order):
        final = result.test_run.final_execution_for_step(step.id)
        if final is not None:
            status = final.status.value
        elif step.id in blocked_ids:
            status = ExecutionStatus.BLOCKED.value
        else:
            status = "NOT_RUN"
        step_trace = trace_steps.get(step.id)
        attempts = (
            len(step_trace.execution_attempts)
            if step_trace is not None
            else (1 if final is not None else 0)
        )
        version = plan_versions.get(step.id)
        version_text = f"v{version}" if version is not None else "unknown"
        print(
            f"  - [{step.order}] {step.name}: {status} "
            f"(attempts: {attempts}, plan {version_text})"
        )


def _print_provider_section(trace: ExecutionTrace | None) -> None:
    print("LLM PROVIDERS")
    if trace is None:
        print("  trace unavailable; provider summary unavailable")
        return
    attempts = [
        attempt
        for step in trace.steps
        for attempt in step.provider_attempts
    ]
    if not attempts:
        if trace.steps and all(
            step.plan_cache is not None and step.plan_cache.hit
            for step in trace.steps
        ):
            print("  all plans loaded from cache; no LLM provider calls were made")
        else:
            print("  no LLM provider attempts were recorded")
        return
    for kind in RequestKind:
        kind_attempts = [
            attempt for attempt in attempts if attempt.request_kind == kind
        ]
        if not kind_attempts:
            continue
        print(f"  {kind.value} requests:")
        for attempt in kind_attempts:
            model = attempt.model or "(unknown model)"
            duration = (
                _format_duration(attempt.duration_ms)
                if attempt.duration_ms is not None
                else "n/a"
            )
            selected = " [selected]" if attempt.is_selected else ""
            print(
                f"    - {attempt.provider_name} ({model}): "
                f"{attempt.outcome.value}, {duration}{selected}"
            )


def _print_execution_section(trace: ExecutionTrace | None) -> None:
    print("EXECUTION")
    if trace is None:
        print("  trace unavailable; execution summary unavailable")
        return
    for step in trace.steps:
        attempts = step.execution_attempts
        if not attempts:
            print(f"  - {step.name}: not executed")
            continue
        last = attempts[-1]
        duration = (
            _format_duration(last.duration_ms)
            if last.duration_ms is not None
            else "n/a"
        )
        print(f"  - {step.name}: {last.status.value}, {duration}")
        if last.error:
            print(f"      error: {last.error}")
        recovery = step.locator_recovery
        if recovery is not None:
            candidate = recovery.candidate_selector or "(no candidate)"
            print(
                f"      locator recovery: {recovery.status.value} "
                f"({recovery.original_selector} -> {candidate})"
            )
            if recovery.reason:
                print(f"      recovery detail: {recovery.reason}")
        regeneration = step.regeneration
        if regeneration is not None:
            print(
                f"      regeneration: v{regeneration.from_version} -> "
                f"v{regeneration.to_version} ({regeneration.reason})"
            )


def _print_evidence_section(result: PipelineResult) -> None:
    print("EVIDENCE")
    evidence_paths = [
        item.path
        for execution in result.executions
        if execution.status == ExecutionStatus.FAILED
        for item in execution.evidence
    ]
    if not evidence_paths:
        print("  no failure evidence")
        return
    for path in evidence_paths:
        print(f"  - {path}")


def _print_trace_section(trace: ExecutionTrace | None) -> None:
    print("TRACE")
    if trace is None:
        print("  unavailable")
        return
    print(f"  trace_id: {trace.trace_id}")
    print(f"  schema_version: {trace.schema_version}")
    totals = trace.totals
    print(
        f"  totals: steps={totals.steps}, "
        f"provider_attempts={totals.provider_attempts}, "
        f"execution_attempts={totals.execution_attempts}, "
        f"locator_recoveries={totals.locator_recoveries}, "
        f"regenerations={totals.regenerations}"
    )


def _format_duration(duration_ms: int) -> str:
    if duration_ms < 1000:
        return f"{duration_ms} ms"
    return f"{duration_ms / 1000:.2f} s"
