"""Definition edits, approval boundaries and generation diagnostics without providers."""
import json
from copy import deepcopy
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from qa_agent.automation_lifecycle import AutomationStatus, definition_fingerprint
from qa_agent.models import (
    ExecutionSegment, Precondition, TestCase as Case, TestStep as Step,
    TestPlan as Plan, TestPlanVersion as Version, QATestPlan, DiscoveryResult,
    DiscoveryStatus, InteractiveElement,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.sqlite_storage import SQLiteTestCaseRepository
from qa_agent.test_case_repository import InMemoryTestCaseRepository, TestCaseConflict as Conflict
from qa_agent.test_case_editing import edit_test_case, create_manual_test_case, TestCaseEditError as EditError
from qa_agent.test_case_naming import fallback_summary, content_title
from qa_agent.test_case_quality import quality_issues, TestCaseQualityError as QualityError
from qa_agent.test_case_review import SQLiteTestCaseReviewRepository, TestCaseReviewStatus as ReviewStatus
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.pipeline import QATestPipeline
from qa_agent.pipeline import PipelineStageError
from qa_agent.execution_progress import ExecutionProgressStore, ExecutionProgressReporter, active_execution_progress
from qa_agent.redaction import register_secret
from qa_agent.run_history import RunHistoryService, InMemoryRunHistoryRepository
from qa_agent.web import LocalWebApplication


def case_fixture():
    return Case(name="Account checks", description="Open the account and check its state.",
        base_url="http://127.0.0.1/account", preconditions=[Precondition(description="A local account is available.", order=0)],
        segments=[ExecutionSegment(order=i, base_url=f"http://127.0.0.1/page-{i}", steps=[
            Step(name=f"Historical title {i}-{j}", description=f"Open account section {i}-{j}.",
                 expected="The section is visible.", order=i * 3 + j) for j in range(3)]) for i in range(2)])


def payload(case):
    return [{"id": str(segment.id), "steps": [
        {"id": str(step.id), "description": step.description, "expected": step.expected}
        for step in segment.steps]} for segment in case.segments]


def fields(case, steps=None):
    return {"name": case.name, "description": case.description, "base_url": case.base_url or "",
            "preconditions": "\n".join(item.description for item in case.preconditions),
            "definition_fingerprint": definition_fingerprint(case),
            "steps_json": json.dumps(steps if steps is not None else payload(case), ensure_ascii=False)}


def application(case=None, repository=None, plans=None):
    repository = repository or InMemoryTestCaseRepository()
    case = case or case_fixture()
    repository.save(case)
    app = LocalWebApplication(RunHistoryService(InMemoryRunHistoryRepository()), test_cases=repository,
                              plan_store=plans or InMemoryPlanStore())
    return app, case, repository


def post(app, path, data):
    return app.handle("POST", path, urlencode(data), headers={"X-QA-CSRF": app._csrf_token})


@pytest.mark.parametrize("scenario,expected", [
    ("1. Load Load homepage and inspect cookie banner", "Load homepage and inspect cookie banner"),
    ("Крок 1. Відкрити Відкрити сторінку та перевірити банер згоди", "Відкрити сторінку та перевірити банер згоди"),
    ("New Draft 1\nOpen account settings and inspect preferences", "Open account settings and inspect preferences"),
    ("2FA authentication for account sign in", "authentication for account sign in 2FA"),
])
def test_names_extract_only_supplied_unicode_words(scenario, expected):
    assert fallback_summary(scenario) == expected
    assert not content_title("1 Behavior Behavior").startswith("1")


@pytest.mark.parametrize("text", ["Test Test", "New Draft 1", "1", "Registration Registration", ""])
def test_insufficient_summary_needs_input(text):
    with pytest.raises(ValueError, match="Summary"):
        fallback_summary(text)


def test_explicit_duplicate_summary_and_historical_names_are_preserved():
    case = case_fixture()
    original = deepcopy(case)
    edited = edit_test_case(case, {**fields(case), "name": " My explicit Summary! "})
    assert edited.name == "My explicit Summary!"
    assert [step.name for step in edited.steps] == [step.name for step in original.steps]
    repo = InMemoryTestCaseRepository()
    repo.save(edited)
    other = edited.model_copy(update={"id": uuid4(), "public_id": None})
    repo.save(other)
    assert other.name == edited.name and other.public_id != edited.public_id
    assert case == original


def test_sqlite_duplicate_summaries_keep_distinct_public_identities(tmp_path):
    repo = SQLiteTestCaseRepository(tmp_path / "summaries.sqlite3")
    first, second = case_fixture(), case_fixture()
    repo.save(first); repo.save(second)
    assert first.name == second.name and first.public_id != second.public_id
    assert len(repo.list()) == 2


def test_sqlite_legacy_review_is_invalidated_in_the_definition_transaction(tmp_path):
    path = tmp_path / "legacy-review.sqlite3"
    repo = SQLiteTestCaseRepository(path)
    review_repo = SQLiteTestCaseReviewRepository(path)
    case = case_fixture(); repo.save(case)
    assert review_repo.get(case.id) is None
    repo.save(case.model_copy(update={"description": "Review the local account settings."}), expected_fingerprint=definition_fingerprint(case))
    assert review_repo.get(case.id).status == ReviewStatus.READY_FOR_REVIEW
    assert review_repo.get(case.id).approved_plan_fingerprint is None


@pytest.mark.parametrize("operation,index", [("insert", 0), ("insert", 1), ("insert", 3),
    ("delete", 0), ("delete", 1), ("delete", 2), ("reorder", 0)])
def test_structural_edits_preserve_segments_metadata_and_stable_ids(operation, index):
    case = case_fixture()
    rows = payload(case)
    if operation == "insert":
        rows[0]["steps"].insert(index, {"id": f"new:{uuid4()}", "description": "Відкрити налаштування профілю.", "expected": "Налаштування видимі."})
    elif operation == "delete":
        rows[0]["steps"].pop(index)
    else:
        rows[0]["steps"].reverse()
    edited = edit_test_case(case, fields(case, rows))
    assert [step.order for step in edited.steps] == list(range(len(edited.steps)))
    assert [segment.id for segment in edited.segments] == [segment.id for segment in case.segments]
    assert [segment.base_url for segment in edited.segments] == [segment.base_url for segment in case.segments]
    assert edited.preconditions == case.preconditions
    assert edited.segments[1] == case.segments[1].model_copy(update={"steps": [
        step.model_copy(update={"order": len(edited.segments[0].steps) + j}) for j, step in enumerate(case.segments[1].steps)]})
    expected_ids = {row["id"] for row in rows[0]["steps"] if not row["id"].startswith("new:")}
    assert expected_ids <= {str(step.id) for step in edited.segments[0].steps}
    assert len({step.id for step in edited.steps}) == len(edited.steps)


def test_description_changes_derive_name_expected_only_preserves_manual_name():
    case = case_fixture()
    rows = payload(case)
    rows[0]["steps"][0]["expected"] = "An error is not displayed."
    rows[0]["steps"][1]["description"] = "Перевірити відсутність помилки в акаунті."
    edited = edit_test_case(case, fields(case, rows))
    assert edited.steps[0].name == case.steps[0].name
    assert edited.steps[1].name == "Перевірити відсутність помилки в акаунті"
    assert edited.steps[1].id == case.steps[1].id
    rows = payload(case)
    rows[0]["steps"][0]["description"] = "  Open   account section 0-0.\r\n"
    assert edit_test_case(case, fields(case, rows)).steps[0].name == case.steps[0].name


def test_empty_summary_uses_edited_steps_without_reusing_deleted_requirements():
    case = case_fixture()
    rows = payload(case)
    rows[0]["steps"] = [{"id": f"new:{uuid4()}", "description": "Inspect shipping address preferences.", "expected": "Shipping address is visible."}]
    edited = edit_test_case(case, {**fields(case, rows), "name": "", "description": "TBD"})
    assert edited.name == "Inspect shipping address preferences"
    manual = create_manual_test_case({"description": "TBD", "step_action_0": "Inspect shipping address preferences."})
    assert manual.name == edited.name


def test_precondition_ids_survive_reordering_and_setup_contract_edits_are_blocked():
    case = case_fixture()
    case.preconditions.append(Precondition(description="Local mail is available.", order=1))
    edited = edit_test_case(case, {**fields(case), "preconditions": "Local mail is available.\nA local account is available."})
    assert [item.id for item in edited.preconditions] == [item.id for item in reversed(case.preconditions)]
    contracted = case.model_copy(deep=True)
    contracted.preconditions[0].provided_data_keys = ["account_email"]
    with pytest.raises(EditError, match="setup data contracts"):
        edit_test_case(contracted, {**fields(contracted), "preconditions": "Different setup."})


@pytest.mark.parametrize("bad", ["duplicate", "foreign", "flatten", "segment", "empty", "fields", "nontext", "oversize", "invalid-new"])
def test_invalid_structures_are_rejected_without_mutating_original(bad):
    case = case_fixture()
    before = deepcopy(case)
    rows = payload(case)
    if bad == "duplicate": rows[0]["steps"][1] = deepcopy(rows[0]["steps"][0])
    if bad == "foreign": rows[0]["steps"][0] = rows[1]["steps"].pop(0)
    if bad == "flatten": rows.pop()
    if bad == "segment": rows[0]["id"] = str(uuid4())
    if bad == "empty": rows[0]["steps"] = []
    if bad == "fields": rows[0]["steps"][0]["name"] = "Injected"
    if bad == "nontext": rows[0]["steps"][0]["expected"] = []
    if bad == "oversize": rows[0]["steps"][0]["description"] = "x" * 6001
    if bad == "invalid-new": rows[0]["steps"][0]["id"] = "new:bad"
    with pytest.raises(EditError): edit_test_case(case, fields(case, rows))
    assert case == before


def test_unfinished_save_quality_feedback_and_legitimate_negative_action_only():
    app, case, repo = application()
    try:
        rows = payload(case)
        rows[0]["steps"][0].update(description="", expected="")
        response = post(app, f"/test-cases/{case.id}/edit", fields(case, rows))
        assert response.status == 303
        saved = repo.get(case.id)
        assert saved.steps[0].description.strip() == saved.steps[0].expected.strip() == ""
        assert app._test_case_review.status(case.id) == ReviewStatus.READY_FOR_REVIEW
        with pytest.raises(QualityError): app._test_case_review.approve_test_case(saved)
        assert post(app, f"/test-cases/{case.id}/approve", {}).status == 409
        page = app.handle("GET", f"/test-cases/{case.id}").body.decode()
        assert f"#structured-{saved.steps[0].id}-description" in page
        meaningful = create_manual_test_case({"description": "Open profile settings without verification.",
            "step_action_0": "Open profile settings.", "step_expected_0": "Settings opened; no verification required."})
        assert not quality_issues(meaningful)
        negative = meaningful.model_copy(deep=True)
        negative.steps[0].expected = "An error is not displayed."
        assert not quality_issues(negative)
    finally: app.close()


@pytest.mark.parametrize("sqlite", [False, True])
def test_atomic_stale_save_and_approval_invalidation_preserve_plan_history(tmp_path, sqlite):
    path = tmp_path / "cases.sqlite3"
    repo = SQLiteTestCaseRepository(path) if sqlite else InMemoryTestCaseRepository()
    case = case_fixture()
    plans = InMemoryPlanStore()
    plan = Plan(test_step_id=case.steps[0].id, name="Historical plan")
    version = Version(test_plan_id=plan.id, version=1, qa_test_plan=QATestPlan(url=case.base_url, steps=[{"action": "assert_page_loaded"}]))
    plans.save(case.steps[0].id, version, test_plan=plan)
    app, case, repo = application(case, repo, plans)
    try:
        app._test_case_review.mark_ready_for_review(case)
        app._test_case_review.approve_test_case(case)
        stale = fields(case)
        changed = fields(case)
        changed["name"] = "Current edited account checks"
        assert post(app, f"/test-cases/{case.id}/edit", changed).status == 303
        assert app._test_case_review.status(case.id) == ReviewStatus.READY_FOR_REVIEW
        current = repo.get(case.id)
        assert app._automation_lifecycle.status(current) == AutomationStatus.NEEDS_UPDATE
        conflict = post(app, f"/test-cases/{case.id}/edit", {**stale, "name": "Unsaved old tab"})
        assert conflict.status == 409 and b"Unsaved old tab" in conflict.body
        with pytest.raises(Conflict): repo.save(case, expected_fingerprint=stale["definition_fingerprint"])
        assert repo.get(case.id).name == changed["name"]
        assert plans.get_version(version.id) == version
        app._test_case_review.approve_test_case(current)
        assert post(app, f"/test-cases/{case.id}/approve-validation", {}).status == 409
        assert app._automation_lifecycle.mark_automation_failed(current) == AutomationStatus.NEEDS_UPDATE
        assert app.handle("GET", f"/test-cases/{case.id}/automation/edit").status == 409
        assert post(app, f"/test-cases/{case.id}/automation/steps/{case.steps[0].id}/save", {}).status == 409
        if sqlite:
            review_repo = SQLiteTestCaseReviewRepository(path)
            from qa_agent.test_case_review import TestCaseReviewService as Review
            review = Review(review_repo, plans)
            review.approve_test_case(current)
            repo.save(current.model_copy(update={"name": "Another material edit"}))
            assert review.status(case.id) == ReviewStatus.READY_FOR_REVIEW
            assert review.record(case.id).approved_plan_fingerprint is None
    finally: app.close()


def test_editor_security_and_errors_preserve_all_submitted_values():
    app, case, repo = application()
    try:
        path = f"/test-cases/{case.id}/edit"
        data = fields(case)
        assert app.handle("POST", path, urlencode(data)).status == 403
        assert app.handle("POST", path, urlencode(data), headers={"X-QA-CSRF": app._csrf_token, "Origin": "https://evil.invalid"}).status == 403
        assert post(app, path, {**data, "unexpected": "field"}).status == 400
        assert post(app, path, {**data, "step_0_0_name": "Injected title"}).status == 400
        assert post(app, path, {**data, "operation": "delete:0:0"}).status == 400
        malicious = '<img src=x onerror="alert(1)">'
        rows = payload(case)
        rows[0]["steps"][0]["description"] = malicious
        response = post(app, path, {**fields(case, rows), "name": "Unsaved name", "description": ""})
        assert response.status == 400
        assert b"Unsaved name" in response.body and b"&lt;img" in response.body
        assert malicious.encode() not in response.body
        assert repo.get(case.id) == case
        assert app.handle("GET", path + "?operation=delete:0:0").status == 200
        assert repo.get(case.id) == case
        duplicate = urlencode(list(data.items()) + [("name", "Other")])
        assert app.handle("POST", path, duplicate, headers={"X-QA-CSRF": app._csrf_token}).status == 403
    finally: app.close()


def test_quality_feedback_links_legacy_noncontiguous_order_to_actual_step():
    app, case, _ = application()
    try:
        case.steps[0].order = 99
        case.steps[0].expected = " "
        feedback = app._quality_feedback(case)
        assert "Step 100" in feedback and f"#structured-{case.steps[0].id}-description" in feedback
    finally: app.close()


class LocalPlanProvider(LLMProvider):
    def __init__(self, actions): self.actions, self.calls = actions, 0
    def create_test_plan(self, task, target_url, page_snapshot):
        self.calls += 1
        return QATestPlan(url=target_url, steps=self.actions)


@pytest.mark.parametrize("selector,observed", [(None, False), ("#cookie-banner", True), ("#invented", True), ("#cookie-banner", False)])
def test_navigation_has_no_locator_requirement_dom_assertions_do(selector, observed):
    step = Step(name="Open homepage", description="Load homepage for the first time." if not selector else "Inspect the cookie banner.",
                expected="Cookie banner is visible on the page." if selector else "The page is loaded.", order=0)
    discovery = DiscoveryResult(status=DiscoveryStatus.PARTIAL, url="http://127.0.0.1/",
        interactive_elements=[InteractiveElement(tag="div", role="dialog", kind="dialog", selector="#cookie-banner", text="Cookie banner")] if observed else [])
    actions = [{"action": "navigate", "parameters": {"url": discovery.url}}]
    actions += [{"action": "assert_visible", "parameters": {"selector": selector}}] if selector else [{"action": "assert_page_loaded"}]
    provider = LocalPlanProvider(actions)
    generator = LLMTestPlanGenerator(LLMRouter([provider]))
    if selector and (not observed or selector == "#invented"):
        with pytest.raises(PlanValidationError) as caught: generator.generate_with_plan(step, discovery)
        assert caught.value.provider_response_succeeded is True
        assert caught.value.rejected_gate == "locator_identity"
        record = generator.supervisor.repository.get(caught.value.reliability_operation_id)
        assert record.outcome == "NEEDS_ATTENTION" and provider.calls == 1
        assert record.attempts[-1].quality_gates["locator_identity"] == "FAILED"
    else:
        assert generator.generate_with_plan(step, discovery).test_plan_version.version == 1


def test_changed_definition_regenerates_new_immutable_version():
    step = Step(name="Open homepage", description="Open homepage.", expected="The page is loaded.", order=0)
    case = Case(name="Homepage navigation", description=step.description, base_url="http://127.0.0.1/", steps=[step])
    plans = InMemoryPlanStore()
    owner = Plan(test_step_id=step.id, name=step.name)
    first = Version(test_plan_id=owner.id, version=1, qa_test_plan=QATestPlan(url=case.base_url, steps=[{"action": "assert_page_loaded"}]))
    plans.save(step.id, first, test_plan=owner)
    provider = LocalPlanProvider([{"action": "assert_page_loaded"}])
    pipeline = QATestPipeline(None, LLMTestPlanGenerator(LLMRouter([provider])),
        discovery=lambda url: DiscoveryResult(status=DiscoveryStatus.PARTIAL, url=url), plan_store=plans,
        runner=lambda plan: {"status": "passed", "steps": [], "assertions_passed": 1, "assertions_failed": 0})
    pipeline.run_test_case(case, regenerate=True)
    assert provider.calls == 1
    assert plans.find(step.id).version == 2 and plans.find(step.id).test_plan_id == owner.id
    assert plans.get_version(first.id) == first


def test_successful_provider_rejected_gate_progress_is_safe_and_links_operation():
    secret = "LOCAL_TEST_SECRET_QUALITY_77"
    register_secret(secret)
    step = Step(name="Inspect consent", description=f"Inspect cookie banner for {secret}.",
        expected="Cookie banner is visible on the page.", order=0)
    case = Case(name="Cookie consent check", description="Inspect the cookie banner.", base_url="http://127.0.0.1/", steps=[step])
    class Unavailable(LocalPlanProvider):
        @property
        def is_available(self): return False
    unavailable = Unavailable([])
    provider = LocalPlanProvider([{"action": "assert_visible", "parameters": {"selector": "#invented-banner"}}])
    generator = LLMTestPlanGenerator(LLMRouter([unavailable, provider]))
    pipeline = QATestPipeline(None, generator, discovery=lambda url: DiscoveryResult(status=DiscoveryStatus.PARTIAL, url=url), runner=lambda _: pytest.fail("Rejected automation must never execute"))
    app, case, _ = application(case)
    app._reliability = generator.supervisor
    progress_id = app.progress_store.create(case.id, "AUTOMATION")
    reporter = ExecutionProgressReporter(app.progress_store, progress_id)
    try:
        with active_execution_progress(reporter), pytest.raises(PipelineStageError):
            pipeline.run_test_case(case)
        snapshot = app.progress_store.get(progress_id)
        failure = snapshot.automation_generation_failure
        assert failure is not None and failure.step_id == step.id
        assert "Provider responded successfully, but Locator Identity" in failure.safe_reason
        assert "unobserved selector" in failure.safe_reason and "Leave unchanged" in failure.safe_reason
        assert failure.failure_category == "AUTOMATION_GENERATION_ERROR"
        assert unavailable.calls == 0 and provider.calls == 1
        assert secret not in json.dumps(snapshot.to_public_dict())
        payload_data = app._progress_payload(snapshot)
        assert any(action["url"].startswith('/settings/reliability/') for action in payload_data["actions"])
        assert b"Developer details" in app.handle("GET", f"/runs/progress/{progress_id}").body
    finally: app.close()
