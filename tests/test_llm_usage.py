from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import AllProvidersFailedError, RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.llm.usage_metadata import (
    ProviderTokenUsage,
    capture_gemini_usage,
    capture_openai_usage,
    capture_provider_usage,
)
from qa_agent.llm_usage import (
    LLMUsageRepository,
    LLMUsageService,
    LLMPricing,
    LLMPricingCatalog,
    OP_AUTHOR_TESTCASE,
    OP_GENERATE_AUTOMATION_PLAN,
    OP_REPAIR_AUTOMATION_PLAN,
    REQUEST_FAILED,
    REQUEST_SUCCESS,
    llm_usage_scope,
)
from qa_agent.models import (
    ExecutionSegment,
    TestCase as DomainTestCase,
    TestStep as DomainTestStep,
)
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_authoring import TestCaseDraft as Draft, TestCaseDraftStore
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication


class FakeProvider(LLMProvider):
    def __init__(self, name: str, *, fail: bool = False, usage=None) -> None:
        self.name = name
        self.model = "mock-model"
        self.fail = fail
        self.usage = usage

    def create_test_plan(self, task, target_url, page_snapshot):
        raise NotImplementedError

    def create_structured_output(self, prompt, schema, schema_name):
        if self.usage is not None:
            capture_openai_usage({"usage": self.usage})
        if self.fail:
            raise RetryableLLMError("HTTP 429; raw response-secret; api_key=sk-sensitive-value")
        return '{"safe":"mock output"}'


def test_provider_usage_metadata_normalizes_actual_response_fields():
    with capture_provider_usage() as captured:
        assert capture_openai_usage(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=17, completion_tokens=8, total_tokens=25
        ))) == ProviderTokenUsage(17, 8, 25)
    assert captured.usage == ProviderTokenUsage(17, 8, 25)

    with capture_provider_usage() as captured:
        assert capture_gemini_usage({"usage": {
            "input_tokens": 13, "output_tokens": 6,
        }}) == ProviderTokenUsage(13, 6, 19)
    assert captured.usage == ProviderTokenUsage(13, 6, 19)

    with capture_provider_usage() as captured:
        assert capture_openai_usage({"usage": {"cached_tokens": 99}}) is None
    assert captured.usage is None


def test_router_records_provider_selection_fallback_and_safe_metadata(tmp_path: Path):
    db_path = tmp_path / "usage.sqlite3"
    service = LLMUsageService(LLMUsageRepository(db_path))
    first = FakeProvider("Groq", fail=True)
    second = FakeProvider("Gemini", usage={
        "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
    })
    events = []

    with llm_usage_scope(
        operation_type=OP_AUTHOR_TESTCASE,
        related_workflow_id="progress-123_safe",
    ):
        result = LLMRouter([first, second], usage_recorder=service).create_structured_output(
            "private authoring prompt",
            {"type": "object"},
            "test_case_authoring",
            progress_callback=lambda event, provider, **kwargs: events.append(
                (event, provider, kwargs)
            ),
        )

    assert result == '{"safe":"mock output"}'
    assert [event[0] for event in events] == ["selected", "failed", "fallback", "selected", "completed"]
    assert events[0][1] == "Groq"
    assert events[2][2]["next_provider"] == "Gemini"
    records = LLMUsageRepository(db_path).list_records()
    assert len(records) == 2
    assert [(item.provider_name, item.request_status) for item in records] == [
        ("Groq", REQUEST_FAILED), ("Gemini", REQUEST_SUCCESS),
    ]
    assert records[0].error_category == "RATE_LIMIT"
    assert records[1].fallback_used is True
    assert records[1].input_tokens == 11
    assert records[1].output_tokens == 7
    assert records[1].total_tokens == 18
    assert all(item.operation_type == OP_AUTHOR_TESTCASE for item in records)
    assert all(item.related_workflow_id == "progress-123_safe" for item in records)

    persisted = db_path.read_bytes()
    for forbidden in (b"private authoring prompt", b"raw response-secret", b"sk-sensitive-value", b"mock output"):
        assert forbidden not in persisted


def test_all_provider_failures_are_persisted_as_terminal_failed_attempts(tmp_path: Path):
    service = LLMUsageService(LLMUsageRepository(tmp_path / "failed.sqlite3"))
    router = LLMRouter([
        FakeProvider("Groq", fail=True),
        FakeProvider("Gemini", fail=True),
    ], usage_recorder=service)
    with llm_usage_scope(operation_type=OP_AUTHOR_TESTCASE):
        with pytest.raises(AllProvidersFailedError):
            router.create_structured_output("prompt", {}, "authoring")
    records = service.repository.list_records()
    assert len(records) == 2
    assert all(item.request_status == REQUEST_FAILED for item in records)
    assert service.analytics("all")["failed_requests"] == 2


def test_usage_sqlite_migration_is_idempotent_and_survives_reconstruction(tmp_path: Path):
    path = tmp_path / "persistent.sqlite3"
    first = LLMUsageRepository(path)
    _record(first, provider="groq", status=REQUEST_SUCCESS)
    LLMUsageRepository(path)  # Reopening runs the idempotent schema migration.
    reopened = LLMUsageRepository(path)
    assert len(reopened.list_records()) == 1
    assert reopened.list_records()[0].provider_id == "groq"


def test_analytics_ranges_operation_groups_and_limited_sample_guard(tmp_path: Path):
    service = LLMUsageService(LLMUsageRepository(tmp_path / "analytics.sqlite3"))
    now = datetime(2026, 5, 12, 14, tzinfo=timezone.utc)
    _record(service.repository, provider="groq", status=REQUEST_SUCCESS, started=now - timedelta(days=2), operation_type=OP_AUTHOR_TESTCASE)
    _record(service.repository, provider="gemini", status=REQUEST_FAILED, started=now - timedelta(days=8), operation_type=OP_GENERATE_AUTOMATION_PLAN)
    _record(service.repository, provider="gemini", status=REQUEST_SUCCESS, started=now - timedelta(days=40), operation_type=OP_REPAIR_AUTOMATION_PLAN)

    assert service.analytics("today", now=now)["requests"] == 0
    week = service.analytics("7d", now=now)
    assert week["requests"] == 1
    assert week["by_provider"][0]["provider_name"] == "Groq"
    assert week["limited_sample"] is True
    assert service.analytics("30d", now=now)["requests"] == 2
    all_time = service.analytics("all", now=now)
    assert all_time["requests"] == 3
    assert {item["operation_type"] for item in all_time["by_operation"]} == {
        OP_AUTHOR_TESTCASE, OP_GENERATE_AUTOMATION_PLAN, OP_REPAIR_AUTOMATION_PLAN,
    }


def test_cost_requires_verified_rate_and_uses_token_breakdown():
    usage = ProviderTokenUsage(input_tokens=1000, output_tokens=500)
    assert LLMPricingCatalog().estimate("groq", "mock-model", usage) is None
    unverified = LLMPricingCatalog({("groq", "mock-model"): LLMPricing(
        provider_id="groq", model="mock-model",
        input_cost_per_1m_tokens=Decimal("0.20"),
        output_cost_per_1m_tokens=Decimal("0.60"),
        source="",
    )})
    assert unverified.estimate("groq", "mock-model", usage) is None
    pricing = LLMPricingCatalog({("groq", "mock-model"): LLMPricing(
        provider_id="groq", model="mock-model",
        input_cost_per_1m_tokens=Decimal("0.20"),
        output_cost_per_1m_tokens=Decimal("0.60"),
        source="verified-local-fixture",
        verified=True,
    )})
    assert pricing.estimate("GROQ", "MOCK-MODEL", usage) == pytest.approx(0.0005)
    assert pricing.estimate("groq", "mock-model", ProviderTokenUsage(total_tokens=5)) is None


def test_testcase_usage_association_and_details_are_conditional(tmp_path: Path):
    path = tmp_path / "case.sqlite3"
    repository = LLMUsageRepository(path)
    service = LLMUsageService(repository)
    workflow_id = "authoring-flow-42"
    _record(repository, provider="groq", status=REQUEST_SUCCESS, workflow=workflow_id)
    case_id = uuid4()
    assert service.test_case_summary(case_id) is None
    service.associate_workflow(workflow_id, case_id, "TC-0001")
    summary = service.test_case_summary(case_id)
    assert summary is not None
    assert summary["providers"] == ["Groq"]
    assert summary["fallback_operations"] == 0
    assert "AI Usage" in LocalWebApplication._test_case_usage_panel(summary)
    assert LocalWebApplication._test_case_usage_panel({"requests": 0})


def test_reviewed_draft_save_associates_usage_and_renders_testcase_details(tmp_path: Path):
    repository = LLMUsageRepository(tmp_path / "saved-case.sqlite3")
    service = LLMUsageService(repository)
    flow = "saved-authoring-flow"
    _record(repository, provider="groq", status=REQUEST_SUCCESS, workflow=flow)
    case = DomainTestCase(
        name="Saved Case",
        description="Check the example page.",
        base_url="https://example.test/",
        segments=[ExecutionSegment(order=0, steps=[DomainTestStep(
            name="Open page", description="Open it.", expected="It is visible.", order=0,
        )])],
    )
    drafts = TestCaseDraftStore()
    token = drafts.put(Draft(case, usage_workflow_ids=(flow,)))
    cases = InMemoryTestCaseRepository()
    app = LocalWebApplication(
        RunHistoryService(InMemoryRunHistoryRepository()),
        test_cases=cases,
        draft_store=drafts,
        llm_usage=service,
    )
    try:
        response = app.handle("POST", f"/test-cases/review/{token}/save", b"", headers={"X-QA-CSRF": app._csrf_token})
        assert response.status == 303
        saved = cases.list()[0]
        summary = service.test_case_summary(saved.id)
        assert summary is not None
        assert summary["providers"] == ["Groq"]
        detail = app.handle("GET", f"/test-cases/{saved.id}")
        assert b"AI Usage" in detail.body
        assert b"Provider attempts" in detail.body
    finally:
        app.close()


def test_usage_analytics_page_and_range_selector_are_safe(tmp_path: Path):
    service = LLMUsageService(LLMUsageRepository(tmp_path / "ui.sqlite3"))
    app = LocalWebApplication(run_history=SimpleNamespace(), llm_usage=service)
    response = app.handle("GET", "/settings/usage?range=all")
    assert response.status == 200
    page = response.body.decode()
    assert "AI Usage" in page
    assert "All time" in page
    assert "Unknown" in page
    assert "does not rank providers" in page or "fewer than five" in page
    assert app.handle("GET", "/settings/usage?range=script").status == 200


def _record(
    repository: LLMUsageRepository,
    *,
    provider: str = "groq",
    status: str = REQUEST_SUCCESS,
    started: datetime | None = None,
    operation_type: str = OP_AUTHOR_TESTCASE,
    workflow: str | None = None,
) -> None:
    service = LLMUsageService(repository)
    now = started or datetime.now(timezone.utc)
    service.record_attempt(
        operation_id=str(uuid4()),
        operation_type=operation_type,
        provider_id=provider,
        provider_name=provider.title(),
        model="mock-model",
        started_at=now,
        finished_at=now + timedelta(milliseconds=18),
        latency_ms=18,
        request_status=status,
        usage=ProviderTokenUsage(10, 5, 15),
        fallback_from_provider="groq" if provider == "gemini" else None,
        error_category="RATE_LIMIT" if status == REQUEST_FAILED else None,
        related_test_case_id=None,
        related_test_case_public_id=None,
        related_workflow_id=workflow,
    )
