"""Deterministic recovery, integrity and persistence checks without provider access."""
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from threading import Event
from urllib.parse import urlencode
from uuid import uuid4
from unittest.mock import patch
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from qa_agent.execution_progress import (
    ExecutionEventType, ExecutionProgressReporter, ExecutionProgressStore,
    active_execution_progress,
)
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import NonRetryableLLMError, RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.llm.gemini import GeminiProvider
from qa_agent.llm.groq import GroqProvider
from qa_agent.llm.openai_compatible import OpenAICompatibleProvider
from qa_agent.llm.usage_metadata import capture_openai_usage
from qa_agent.llm_usage import LLMPricing, LLMPricingCatalog, llm_usage_scope
from qa_agent.models import DiscoveryResult, DiscoveryStatus, PlanVersionOrigin, QATestPlan, RunContext, TestStep as Step
from qa_agent.reliability import (
    AutomationReliabilitySupervisor, InMemoryReliabilityRepository,
    ReliabilityRecord, ReliabilitySettings, ReliabilityStopped, SQLiteReliabilityRepository,
    classify_failure, current_reliability_operation,
)
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.web import LocalWebApplication, create_application


URL = "http://127.0.0.1:9876/target"


def requirement():
    return Step(name="Verify notice", description="Verify the notice is visible.", expected="The notice is visible.", order=0)


def discovery():
    return DiscoveryResult(status=DiscoveryStatus.SUCCESS, url=URL,
        snapshot={"visible_text_elements": [{"selector": "#notice", "tag": "p", "text": "Welcome"}]})


def valid_plan():
    return QATestPlan(url=URL, steps=[{"action": "assert_visible", "parameters": {"selector": "#notice"}}])


class LocalProvider(LLMProvider):
    def __init__(self, results, *, name="Local provider", available=True, usage=None):
        self.results, self.name, self.available, self.usage = list(results), name, available, usage
        self.model = "local-model"
        self.calls = []

    @property
    def is_available(self):
        return self.available

    def create_test_plan(self, task, target_url, page_snapshot):
        self.calls.append((task, target_url, page_snapshot))
        if self.usage is not None:
            capture_openai_usage(self.usage)
        value = self.results.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value() if callable(value) else value


def configured(*providers, repository=None, **settings):
    repo = repository or InMemoryReliabilityRepository()
    repo.save_settings(ReliabilitySettings(**settings))
    supervisor = AutomationReliabilitySupervisor(repo)
    return LLMTestPlanGenerator(LLMRouter(list(providers)), supervisor), supervisor


def generate(generator, step=None):
    return generator.generate_with_plan(step or requirement(), discovery())


@pytest.mark.parametrize("values", [
    {"max_total_attempts": value} for value in [0, 4, True, "2", 2.0]
] + [{"additional_retries": "on"}, {"provider_fallback": 1}, {"automatic_plan_repair": None}, {"extra": True}])
def test_settings_reject_invalid_types_and_values(values):
    with pytest.raises(ValidationError):
        ReliabilitySettings(**values)


def test_additive_migration_preserves_existing_rows_and_settings(tmp_path):
    database = tmp_path / "existing.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE legacy (value TEXT)")
        connection.execute("INSERT INTO legacy VALUES ('historical plan and run')")
    repo = SQLiteReliabilityRepository(database)
    assert repo.settings() == ReliabilitySettings()
    assert repo.settings().provider_fallback is True
    settings = ReliabilitySettings(additional_retries=True, provider_fallback=False, automatic_plan_repair=True, max_total_attempts=3)
    repo.save_settings(settings)
    assert SQLiteReliabilityRepository(database).settings() == settings
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM legacy").fetchone()[0] == "historical plan and run"


def test_legacy_three_provider_fallback_migrates_without_overriding_later_choices(tmp_path):
    database = tmp_path / "legacy-providers.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE provider_settings (provider_id TEXT, enabled INTEGER)")
        connection.executemany("INSERT INTO provider_settings VALUES (?, 1)", [("first",), ("second",), ("third",)])
    repo = SQLiteReliabilityRepository(database)
    assert repo.settings().max_total_attempts == 3 and repo.settings().provider_fallback
    repo.save_settings(ReliabilitySettings(provider_fallback=False, max_total_attempts=1))
    assert SQLiteReliabilityRepository(database).settings().max_total_attempts == 1
    assert not SQLiteReliabilityRepository(database).settings().provider_fallback


@pytest.mark.parametrize("maximum", [1, 2, 3])
def test_fallback_chain_has_one_shared_attempt_budget(maximum):
    providers = [LocalProvider([RetryableLLMError("outage", category="PROVIDER_UNAVAILABLE")], name=f"P{i}") for i in range(4)]
    generator, supervisor = configured(*providers, max_total_attempts=maximum)
    with pytest.raises(ReliabilityStopped, match="attempt limit"):
        generate(generator)
    record = supervisor.repository.list_records()[0]
    assert len(record.attempts) == maximum == sum(len(provider.calls) for provider in providers)
    assert record.final_category == "ATTEMPT_LIMIT"
    assert record.outcome == "NEEDS_ATTENTION"
    assert record.decisions[-1].action == "STOP"


def test_off_controls_disable_recovery_but_all_gates_remain_mandatory():
    first = LocalProvider([valid_plan()])
    generator, supervisor = configured(first, provider_fallback=False)
    generate(generator)
    assert set(supervisor.repository.list_records()[0].attempts[0].quality_gates.values()) == {"PASSED"}
    first.results = [{"url": URL, "steps": []}, valid_plan()]
    with pytest.raises(PlanValidationError):
        generate(generator)
    assert len(first.calls) == 2
    assert len(supervisor.repository.list_records()[0].attempts) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_fallback_separately_configurable_and_ordered(enabled):
    unavailable = LocalProvider([], name="Unavailable", available=False)
    first = LocalProvider([RetryableLLMError("bad key", category="AUTH_ERROR")], name="First")
    second = LocalProvider([valid_plan()], name="Second")
    third = LocalProvider([], name="Third")
    generator, supervisor = configured(unavailable, first, second, third, provider_fallback=enabled, additional_retries=True)
    if enabled:
        generate(generator)
    else:
        with pytest.raises(RetryableLLMError):
            generate(generator)
    assert len(first.calls) == 1
    assert len(second.calls) == int(enabled)
    assert not unavailable.calls and not third.calls
    record = supervisor.repository.list_records()[0]
    assert [a.provider for a in record.attempts] == (["First", "Second"] if enabled else ["First"])


@pytest.mark.parametrize("error", [RuntimeError("unknown"), NonRetryableLLMError("invalid config")])
def test_unknown_and_nonretryable_errors_stop(error):
    first, second = LocalProvider([error]), LocalProvider([valid_plan()])
    generator, supervisor = configured(first, second, additional_retries=True, automatic_plan_repair=True, max_total_attempts=3)
    with pytest.raises(type(error)):
        generate(generator)
    assert len(first.calls) == 1 and not second.calls
    assert supervisor.repository.list_records()[0].outcome == "NEEDS_ATTENTION"


def test_no_provider_and_no_fallback_have_safe_diagnostics():
    generator, supervisor = configured()
    with pytest.raises(ReliabilityStopped, match="No eligible configured provider"):
        generate(generator)
    assert supervisor.repository.list_records()[0].final_category == "NO_PROVIDER"
    first = LocalProvider([RetryableLLMError("outage", category="PROVIDER_UNAVAILABLE")])
    generator, supervisor = configured(first)
    with pytest.raises(RetryableLLMError):
        generate(generator)
    assert supervisor.repository.list_records()[0].decisions[-1].action == "STOP"


def test_transient_retry_and_rate_limit_delay():
    provider = LocalProvider([RetryableLLMError("429", category="RATE_LIMIT", retry_after_seconds=1), valid_plan()])
    generator, supervisor = configured(provider, additional_retries=True, provider_fallback=False)
    started = time.monotonic()
    generate(generator)
    assert time.monotonic() - started >= 0.9
    record = supervisor.repository.list_records()[0]
    assert [a.action for a in record.attempts] == ["INITIAL_GENERATION", "PROVIDER_RETRY"]
    assert record.outcome == "READY_FOR_REVIEW"


def test_long_retry_after_does_not_sleep_or_repeat_same_provider():
    first = LocalProvider([RetryableLLMError("429", category="RATE_LIMIT", retry_after_seconds=600)])
    second = LocalProvider([valid_plan()])
    generator, supervisor = configured(first, second, additional_retries=True)
    generate(generator)
    assert len(first.calls) == 1
    assert supervisor.repository.list_records()[0].attempts[-1].action == "PROVIDER_FALLBACK"


@pytest.mark.parametrize("enabled", [False, True])
def test_structural_repair_requires_explicit_setting_and_preserves_requirement(enabled):
    provider = LocalProvider([{"url": URL, "steps": []}, valid_plan()])
    generator, supervisor = configured(provider, automatic_plan_repair=enabled)
    step = requirement()
    before = step.model_dump_json()
    if enabled:
        candidate = generate(generator, step)
        assert candidate.test_plan_version.origin == PlanVersionOrigin.REPAIRED
        assert supervisor.requires_review(candidate.test_plan_version.id)
    else:
        with pytest.raises(PlanValidationError):
            generate(generator, step)
    assert step.model_dump_json() == before
    assert len(provider.calls) == 1 + int(enabled)
    record = supervisor.repository.list_records()[0]
    if enabled:
        assert record.attempts[0].quality_gates["schema_and_actions"] == "FAILED"
        assert set(record.attempts[1].quality_gates.values()) == {"PASSED"}
        assert step.expected in provider.calls[1][0]


def test_repair_and_fallback_do_not_multiply_budget():
    first = LocalProvider([{"url": URL, "steps": []}, RetryableLLMError("repair timeout", category="TIMEOUT")])
    second = LocalProvider([valid_plan()])
    generator, supervisor = configured(first, second, automatic_plan_repair=True, max_total_attempts=2)
    with pytest.raises(ReliabilityStopped):
        generate(generator)
    assert len(first.calls) == 2 and not second.calls
    assert len(supervisor.repository.list_records()[0].attempts) == 2
    first.results = [{"url": URL, "steps": []}, RetryableLLMError("outage", category="PROVIDER_UNAVAILABLE")]
    supervisor.repository.save_settings(ReliabilitySettings(automatic_plan_repair=True, max_total_attempts=3))
    generate(generator)
    assert len(supervisor.repository.list_records()[0].attempts) == 3
    assert supervisor.repository.list_records()[0].repaired


def test_invalid_provider_response_can_receive_one_targeted_repair():
    provider = LocalProvider([RetryableLLMError("private malformed content", category="INVALID_RESPONSE"), valid_plan()])
    generator, supervisor = configured(provider, provider_fallback=False, automatic_plan_repair=True)
    generate(generator)
    assert supervisor.repository.list_records()[0].attempts[-1].action == "TARGETED_REPAIR"


@pytest.mark.parametrize("plan,category", [
    (QATestPlan(url=URL, steps=[{"action": "assert_visible", "parameters": {"selector": "#guessed"}}]), "LOCATOR_IDENTITY_UNCERTAIN"),
    (QATestPlan(url=URL, steps=[{"action": "assert_text_contains", "parameters": {"expected_text": "Invented success"}}]), "ASSERTION_NOT_GROUNDED"),
    (QATestPlan(url=URL, steps=[{"action": "fill", "parameters": {"selector": "#notice"}}]), "PLAN_SCHEMA_INVALID"),
])
def test_unsafe_candidates_stop_without_repair(plan, category):
    provider = LocalProvider([plan, valid_plan()])
    generator, supervisor = configured(provider, automatic_plan_repair=True, max_total_attempts=3)
    with pytest.raises(PlanValidationError):
        generate(generator)
    assert len(provider.calls) == 1
    assert supervisor.repository.list_records()[0].final_category == category


def test_repaired_candidate_still_rejects_ungrounded_assertion():
    invented = QATestPlan(url=URL, steps=[{"action": "assert_text_contains", "parameters": {"expected_text": "Invented success"}}])
    provider = LocalProvider([{"url": URL, "steps": []}, invented])
    generator, supervisor = configured(provider, automatic_plan_repair=True)
    with pytest.raises(PlanValidationError):
        generate(generator)
    record = supervisor.repository.list_records()[0]
    assert record.attempts[-1].quality_gates["assertion_grounding"] == "FAILED"
    assert record.candidate_version_id is None


def test_action_only_plan_cannot_replace_verification_and_repair_cannot_drop_actions():
    action_only = QATestPlan(url=URL, steps=[{"action": "navigate", "parameters": {"url": URL}}])
    provider = LocalProvider([action_only, valid_plan()])
    generator, supervisor = configured(provider, automatic_plan_repair=True)
    with pytest.raises(PlanValidationError) as raised:
        generate(generator)
    assert raised.value.issues[0].code == "REPAIR_CHANGED_SEMANTICS"
    assert supervisor.repository.list_records()[0].final_category == "UNSAFE_REPAIR"


def test_product_failure_never_allows_repair():
    provider = LocalProvider([ReliabilityStopped("PRODUCT_FAILURE"), valid_plan()])
    generator, supervisor = configured(provider, automatic_plan_repair=True, additional_retries=True)
    with pytest.raises(ReliabilityStopped):
        generate(generator)
    assert len(provider.calls) == 1
    assert not supervisor.repository.list_records()[0].repaired


def test_missing_requirements_stop_before_provider():
    step = requirement().model_copy(update={"expected": ""})
    provider = LocalProvider([valid_plan()])
    generator, supervisor = configured(provider)
    with pytest.raises(ReliabilityStopped):
        generate(generator, step)
    assert not provider.calls
    assert supervisor.repository.list_records()[0].final_category == "INSUFFICIENT_TESTCASE_REQUIREMENTS"


def test_timeout_and_late_return_do_not_start_recovery_or_change_history():
    release, entered = Event(), Event()
    def blocked():
        entered.set()
        release.wait(2)
        return valid_plan()
    provider = LocalProvider([blocked, valid_plan()])
    generator, supervisor = configured(provider, additional_retries=True, automatic_plan_repair=True)
    supervisor.timeout_seconds = 0.06
    try:
        with pytest.raises(ReliabilityStopped):
            generate(generator)
        record = supervisor.repository.list_records()[0]
        assert entered.is_set() and record.final_category == "TIME_LIMIT"
        frozen = record.model_dump_json()
    finally:
        release.set()
    time.sleep(0.04)
    assert supervisor.repository.get(record.id).model_dump_json() == frozen
    assert len(provider.calls) == 1
    assert current_reliability_operation() is None


def test_cancellation_interrupts_backoff_and_stops_recovery():
    provider = LocalProvider([RetryableLLMError("timeout", category="TIMEOUT"), valid_plan()])
    generator, supervisor = configured(provider, additional_retries=True)
    errors = []
    def run():
        try:
            generate(generator)
        except Exception as error:
            errors.append(error)
    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        records = supervisor.repository.list_records()
        if records and any(d.action == "PROVIDER_RETRY" for d in records[0].decisions):
            assert supervisor.cancel(records[0].id)
            break
        time.sleep(0.005)
    worker.join(2)
    assert not worker.is_alive() and errors
    assert supervisor.repository.list_records()[0].outcome == "CANCELLED"
    assert len(provider.calls) == 1


def test_persistence_safe_metadata_and_effective_setting_snapshot(tmp_path):
    repo = SQLiteReliabilityRepository(tmp_path / "safe.sqlite3")
    secret = "PRIVATE_INPUT_49021"
    context = RunContext()
    context.set_value("password", secret, sensitive=True)
    store = ExecutionProgressStore()
    progress_id = store.create(uuid4(), "AUTOMATION")
    reporter = ExecutionProgressReporter(store, progress_id, context)
    provider = LocalProvider([RetryableLLMError(f"secret {secret} private prompt", category="AUTH_ERROR")], name=f"Provider {secret}")
    generator, supervisor = configured(provider, repository=repo, provider_fallback=False)
    case_id = uuid4()
    with active_execution_progress(reporter), llm_usage_scope(related_test_case_id=case_id):
        with pytest.raises(RetryableLLMError):
            generate(generator)
    record = repo.list_records()[0]
    repo.save_settings(ReliabilitySettings(max_total_attempts=3))
    restarted = SQLiteReliabilityRepository(repo.path)
    assert restarted.get(record.id).settings.max_total_attempts == 2
    assert restarted.get(record.id).test_case_id == case_id
    with sqlite3.connect(repo.path) as connection:
        persisted = connection.execute("SELECT record_json FROM automation_reliability_operations").fetchone()[0]
    assert secret not in persisted and "private prompt" not in persisted
    assert record.attempts[0].input_tokens is None and record.attempts[0].estimated_cost_usd is None
    assert record.attempts[0].model == "local-model"


def test_usage_and_verified_pricing_reuse_existing_metadata():
    provider = LocalProvider([valid_plan()], usage={"prompt_tokens": 100, "completion_tokens": 20})
    generator, supervisor = configured(provider)
    pricing = LLMPricing("local provider", "local-model", Decimal("1"), Decimal("2"), "local verified fixture", verified=True)
    generator._router._usage_recorder = type("Recorder", (), {"pricing": LLMPricingCatalog({("local provider", "local-model"): pricing}), "record_attempt": lambda *a, **k: None})()
    generate(generator)
    attempt = supervisor.repository.list_records()[0].attempts[0]
    assert (attempt.input_tokens, attempt.output_tokens) == (100, 20)
    assert attempt.estimated_cost_usd == pytest.approx(0.00014)


def test_statistics_distinguish_first_success_recovery_and_failures(tmp_path):
    repo = SQLiteReliabilityRepository(tmp_path / "stats.sqlite3")
    provider = LocalProvider([valid_plan(), {"url": URL, "steps": []}, valid_plan(), {"url": URL, "steps": []}, {"url": URL, "steps": []}])
    generator, supervisor = configured(provider, repository=repo, automatic_plan_repair=True)
    generate(generator)
    generate(generator)
    with pytest.raises(PlanValidationError):
        generate(generator)
    stats = AutomationReliabilitySupervisor(SQLiteReliabilityRepository(repo.path)).statistics()
    assert stats["generation_operations"] == 3
    assert stats["first_attempt_success_rate"] == pytest.approx(1/3)
    assert stats["recovery_success_rate"] == 0.5
    assert stats["recovery_attempts"] == 2 and stats["fallback_count"] == 0
    assert stats["average_attempts"] == pytest.approx(5/3)
    assert stats["quality_gate_rejections"] == {"PLAN_SCHEMA_INVALID": 3}
    assert stats["input_tokens"] is None and stats["estimated_cost_usd"] is None
    assert stats["provider_model_attempts"][0]["count"] == 5


def test_restart_does_not_fabricate_completion_duration():
    repo = InMemoryReliabilityRepository()
    interrupted = ReliabilityRecord(test_step_id=uuid4(), settings=repo.settings(), started_at=datetime.now(timezone.utc))
    repo.save(interrupted)
    supervisor = AutomationReliabilitySupervisor(repo)
    supervisor.reconcile_interrupted()
    record = repo.get(interrupted.id)
    assert record.outcome == "INTERRUPTED"
    assert record.finished_at is None and record.elapsed_ms is None
    assert supervisor.statistics()["first_attempt_success_rate"] is None


def test_progress_shows_real_fallback_repair_and_quality_result():
    first = LocalProvider([RetryableLLMError("outage", category="PROVIDER_UNAVAILABLE")])
    second = LocalProvider([{"url": URL, "steps": []}, valid_plan()])
    generator, supervisor = configured(first, second, automatic_plan_repair=True, max_total_attempts=3)
    store = ExecutionProgressStore()
    progress_id = store.create(uuid4(), "AUTOMATION")
    with active_execution_progress(ExecutionProgressReporter(store, progress_id)):
        generate(generator)
    events = store.get(progress_id).events
    kinds = [event.event_type for event in events]
    assert ExecutionEventType.RELIABILITY_FALLBACK in kinds
    assert ExecutionEventType.RELIABILITY_REPAIR in kinds
    assert kinds[-1] == ExecutionEventType.RELIABILITY_READY_FOR_REVIEW
    assert all(event.reliability_operation_id == supervisor.repository.list_records()[0].id for event in events)
    assert any("unavailable" in (event.message or "") and "attempt 2" in (event.message or "") for event in events)


def test_settings_and_operation_ui_validate_escape_and_persist(tmp_path):
    repo = SQLiteReliabilityRepository(tmp_path / "ui.sqlite3")
    provider = LocalProvider([valid_plan()], name="Provider <script>")
    generator, supervisor = configured(provider, repository=repo)
    generate(generator)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), reliability=supervisor)
    try:
        response = app.handle("GET", "/settings/reliability")
        assert response.status == 200
        assert b"Automation Reliability" in response.body and b"Browser Validation" in response.body
        assert b"Provider &lt;script&gt;" in response.body
        values = {"additional_retries": "on", "provider_fallback": "off", "automatic_plan_repair": "on", "max_total_attempts": "3"}
        assert app.handle("POST", "/settings/reliability", urlencode(values)).status == 303
        assert SQLiteReliabilityRepository(repo.path).settings().max_total_attempts == 3
        for invalid in ["4", "<script>", "2&max_total_attempts=3"]:
            assert app.handle("POST", "/settings/reliability", urlencode(values | {"max_total_attempts": invalid})).status == 400
        assert app.handle("POST", "/settings/reliability", "max_total_attempts=2").status == 400
        record = repo.list_records()[0]
        detail = app.handle("GET", f"/settings/reliability/{record.id}")
        assert detail.status == 200 and b"Attempt 1" in detail.body and b"Ready For Review" in detail.body
        assert b"Provider &lt;script&gt;" in detail.body and b"<script>" not in detail.body
        assert app.handle("POST", f"/settings/reliability/{record.id}/cancel", "").status == 409
        assert app.handle("GET", f"/settings/reliability/{uuid4()}").status == 404
    finally:
        app.close()


def test_factory_wires_same_persistent_supervisor_without_provider_requests(tmp_path):
    application = create_application(tmp_path / "application.sqlite3", tmp_path / "evidence")
    try:
        assert application.handle("GET", "/settings/reliability").status == 200
        assert application._reliability.statistics()["generation_operations"] == 0
    finally:
        application.close()


def test_supervised_openai_compatible_disables_sdk_retry_and_bounds_timeout():
    provider = OpenAICompatibleProvider("openai", "LOCAL_KEY", "local-model", api_key="local-placeholder", timeout_seconds=120)
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = valid_plan().model_dump_json()
    with patch("qa_agent.llm.openai_compatible.OpenAI", return_value=client) as constructor:
        generator, supervisor = configured(provider)
        generate(generator)
    assert constructor.call_args.kwargs["max_retries"] == 0
    assert 0 < constructor.call_args.kwargs["timeout"] <= 30
    assert client.chat.completions.create.call_count == 1


def test_supervised_gemini_bounds_per_call_timeout_and_classifies_malformed_response():
    client = MagicMock()
    client.interactions.create.return_value.output_text = "invalid json with private content"
    with patch("qa_agent.llm.gemini.genai.Client", return_value=client):
        provider = GeminiProvider(api_key="local-placeholder")
    generator, supervisor = configured(provider, provider_fallback=False)
    with pytest.raises(RetryableLLMError):
        generate(generator)
    assert 0 < client.interactions.create.call_args.kwargs["timeout"] <= 30
    assert supervisor.repository.list_records()[0].final_category == "INVALID_RESPONSE"
    assert "private content" not in supervisor.repository.list_records()[0].model_dump_json()


def test_supervised_groq_has_one_bounded_transport_request():
    response = MagicMock()
    response.is_error = False
    response.json.return_value = {"choices": [{"message": {"content": valid_plan().model_dump_json()}}]}
    provider = GroqProvider(api_key="local-placeholder", timeout_seconds=120)
    with patch("qa_agent.llm.groq.httpx.post", return_value=response) as post:
        generator, supervisor = configured(provider)
        generate(generator)
    assert post.call_count == 1 and 0 < post.call_args.kwargs["timeout"] <= 30
    assert supervisor.repository.list_records()[0].attempts[0].provider == "GroqProvider"
