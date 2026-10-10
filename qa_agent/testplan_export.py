"""Deterministic, local-only export of saved canonical TestPlanVersions."""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass, replace
from importlib.metadata import version as installed_version
from pathlib import PurePosixPath
from typing import Callable
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from qa_agent.models import QATestPlan, QATestStep, TestCase, TestPlanVersion
from qa_agent.execution_semantics import (
    ACTION_TIMEOUT_MS,
    ASSERTION_TIMEOUT_MS,
    NAVIGATION_LOAD_STATE,
    NAVIGATION_TIMEOUT_MS,
)
from qa_agent.plan_store import PlanStore
from qa_agent.test_case_repository import TestCaseRepository
from qa_agent.test_plan_validation import PlanValidationError, validate_executable_plan
from qa_agent.automation_lifecycle import (
    AutomationLifecycleService, AutomationStatus, definition_fingerprint,
    plan_fingerprint, plan_fingerprint_for_versions,
)
from qa_agent.test_case_review import TestCaseReviewService, TestCaseReviewStatus
from qa_agent.redaction import redact_secrets
from qa_agent.export_templates import PYTHON_RUNTIME, PYTHON_CONFTEST, TYPESCRIPT_RUNTIME, CSHARP_RUNTIME, PYTHON_CI


PORTABLE_SCHEMA_VERSION = 1
EXPORT_FORMAT_VERSION = "1.1"


@dataclass(frozen=True)
class ExportBlocker:
    """Safe, actionable reason a TestCase cannot be exported yet."""

    test_case_id: UUID
    public_id: str
    test_case_name: str
    step_order: int | None
    reason: str
    step_name: str | None = None


class TestPlanExportError(ValueError):
    """Safe error that can be shown to a local user without internal details."""

    def __init__(self, message: str, *, blockers: tuple[ExportBlocker, ...] = ()) -> None:
        self.blockers = blockers
        super().__init__(message)


@dataclass(frozen=True)
class SavedStepPlan:
    test_step_id: UUID
    segment_order: int
    segment_is_implicit: bool
    segment_base_url: str | None
    step_order: int
    step_name: str
    step_description: str
    step_expected: str
    failure_policy: str
    version: TestPlanVersion


@dataclass(frozen=True)
class ExportableTestCase:
    test_case: TestCase
    plans: tuple[SavedStepPlan, ...]
    verification: str = "REVIEW_ONLY"


def _plans_for_case(test_case: TestCase, plan_store: PlanStore) -> ExportableTestCase:
    plans: list[SavedStepPlan] = []
    for segment in sorted(test_case.segments, key=lambda item: item.order):
        for step in sorted(segment.steps, key=lambda item: item.order):
            version = plan_store.find(step.id)
            if version is None:
                blocker = ExportBlocker(
                    test_case_id=test_case.id,
                    public_id=test_case.public_id or "",
                    test_case_name=test_case.name,
                    step_order=step.order + 1,
                    reason="No executable plan is saved.",
                    step_name=step.name,
                )
                raise TestPlanExportError(
                    f"Automation required before export: {test_case.public_id or test_case.name}, "
                    f"step {step.order + 1} ({step.name}) has no saved executable plan.",
                    blockers=(blocker,),
                )
            test_plan = plan_store.find_test_plan(step.id)
            if (
                test_plan is None
                or test_plan.test_step_id != step.id
                or test_plan.id != version.test_plan_id
            ):
                raise TestPlanExportError(
                    f"Cannot export {test_case.public_id or test_case.name}: "
                    f"the saved plan relationship for step {step.order + 1} is invalid."
                )
            try:
                validate_executable_plan(version.qa_test_plan)
            except PlanValidationError as error:
                if any(issue.code == "UNSUPPORTED_ACTION" for issue in error.issues):
                    raise TestPlanExportError(
                        f"Cannot export {test_case.public_id or test_case.name}: "
                        f"step {step.order + 1} uses an unsupported action."
                    ) from error
                raise TestPlanExportError(
                    f"Cannot export {test_case.public_id or test_case.name}: "
                    f"step {step.order + 1} has an invalid saved automation plan."
                ) from error
            metadata_strings = [test_case.name, test_case.base_url or ""]
            metadata_strings.extend(
                value
                for item in test_case.preconditions
                for value in [item.description, *item.provided_data_keys]
            )
            metadata_strings.extend(
                value
                for candidate_segment in test_case.segments
                for value in [candidate_segment.base_url or ""]
            )
            metadata_strings.extend(
                value
                for candidate_step in test_case.steps
                for value in [candidate_step.name, candidate_step.description, candidate_step.expected]
            )
            if _contains_local_path_or_url_credentials(version.qa_test_plan, metadata_strings):
                raise TestPlanExportError(
                    f"Cannot export {test_case.public_id or test_case.name}: "
                    "the saved plan contains a local path or credential-bearing URL."
                )
            plans.append(SavedStepPlan(
                test_step_id=step.id,
                segment_order=segment.order,
                segment_is_implicit=segment.is_implicit,
                segment_base_url=segment.base_url,
                step_order=step.order,
                step_name=step.name,
                step_description=step.description,
                step_expected=step.expected,
                failure_policy=step.failure_policy.value,
                version=version,
            ))
    if not plans:
        blocker = ExportBlocker(
            test_case_id=test_case.id,
            public_id=test_case.public_id or "",
            test_case_name=test_case.name,
            step_order=None,
            reason="No executable plan is saved.",
        )
        raise TestPlanExportError(
            f"Automation required before export: {test_case.public_id or test_case.name} has no steps.",
            blockers=(blocker,),
        )
    return ExportableTestCase(test_case=test_case, plans=tuple(plans))


def portable_testplan(exportable: ExportableTestCase) -> dict:
    case = exportable.test_case
    return {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "source_application": "AI QA Agent",
        "test_case": {
            "public_id": case.public_id,
            "name": case.name,
            "base_url": case.base_url,
            "preconditions": [
                {
                    "order": item.order,
                    "description": item.description,
                    "provided_data_keys": list(item.provided_data_keys),
                }
                for item in sorted(case.preconditions, key=lambda item: item.order)
            ],
            "segments": _portable_segments(exportable),
        },
    }


def _portable_segments(exportable: ExportableTestCase) -> list[dict]:
    case = exportable.test_case
    by_step = {item.test_step_id: item for item in exportable.plans}
    segments: list[dict] = []
    for segment in sorted(case.segments, key=lambda item: item.order):
        segment_steps = []
        for step in sorted(segment.steps, key=lambda item: item.order):
            saved = by_step[step.id]
            segment_steps.append({
                "order": step.order,
                "name": step.name,
                "description": step.description,
                "expected": step.expected,
                "failure_policy": step.failure_policy.value,
                "testplan_version": {
                    "version": saved.version.version,
                    "created_at": saved.version.created_at.isoformat(),
                    "provenance": saved.version.origin.value if saved.version.origin else None,
                    "plan": saved.version.qa_test_plan.model_dump(mode="json"),
                },
            })
        segments.append({
            "order": segment.order,
            "base_url": segment.base_url,
            "is_implicit": segment.is_implicit,
            "steps": segment_steps,
        })
    return segments


_WINDOWS_PATH = re.compile(r"(?i)(?<![a-z])(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/])")
_COMMON_POSIX_PATH = re.compile(r"(?:^|[\s=:'\"(])/(?:home|users|private|tmp|var|etc|mnt|workspace|root)/")


def _is_unsafe_export_value(value: str) -> bool:
    if _WINDOWS_PATH.search(value) or _COMMON_POSIX_PATH.search(value) or value.casefold().startswith("file://"):
        return True
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    if parsed.scheme.casefold() not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return True
    secret_query_keys = {
        "api_key", "api-key", "apikey", "access_token", "access-token",
        "token", "password", "secret", "authorization", "credential",
    }
    return any(key.casefold() in secret_query_keys for key, _ in parse_qsl(parsed.query, keep_blank_values=True))


_SENSITIVE_FIELD = re.compile(r"(?i)password|passwd|pwd|api[_-]?key|token|secret|authorization|credential|cookie|session")
_SECRET_LITERAL = re.compile(
    r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|secret|authorization|cookie|session[_-]?token)\s*[=:]\s*[^\s,;<>]+"
    r"|\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{16,}|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
)


def _validate_export_safety(exportable: ExportableTestCase, suite_name: str | None = None) -> None:
    # Scan exactly what is serialized, including direct renderer callers. Fail
    # instead of redacting executable data and thereby changing assertions.
    strings = [suite_name or ""]
    def collect(value):
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(portable_testplan(exportable))
    for value in strings:
        embedded_urls = re.findall(r"https?://[^\s<>\"']+", value)
        if _is_unsafe_export_value(value) or any(_is_unsafe_export_value(url) for url in embedded_urls):
            raise TestPlanExportError("Export contains a local path or credential-bearing URL.")
        if redact_secrets(value) != value or _SECRET_LITERAL.search(value):
            raise TestPlanExportError("Export contains sensitive data; use a secret-free test plan.")
    for saved in exportable.plans:
        for action in saved.version.qa_test_plan.steps:
            p = action.parameters
            if any(p.get(key) for key in ("value", "expected", "expected_text")) and _SENSITIVE_FIELD.search(p.get("selector", "")):
                raise TestPlanExportError("Export contains a credential field value; use a secret-free test plan.")


def _contains_local_path_or_url_credentials(
    plan: QATestPlan,
    metadata_strings: list[str] | None = None,
) -> bool:
    strings = [plan.url, *(metadata_strings or [])]
    for action in plan.steps:
        strings.extend(value for value in action.parameters.values() if isinstance(value, str))
    return any(_is_unsafe_export_value(value) for value in strings)


def portable_json(exportable: ExportableTestCase) -> str:
    _validate_action_parameters(exportable)
    return json.dumps(
        portable_testplan(exportable),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _python_string(value: str) -> str:
    return _json_string(value)


def _js_string(value: str) -> str:
    return _json_string(value)


def _csharp_string(value: str) -> str:
    escaped = []
    for char in value:
        code = ord(char)
        if char == "\\":
            escaped.append("\\\\")
        elif char == '"':
            escaped.append('\\"')
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif code < 32 or code in {0x2028, 0x2029}:
            escaped.append(f"\\u{code:04x}")
        else:
            escaped.append(char)
    return '"' + "".join(escaped) + '"'


def _normalized_expected(value: str) -> str:
    return value.replace("\\n", "\n")


def _regex_escape(value: str) -> str:
    return re.sub(r"([\\^$.*+?()\[\]{}|])", r"\\\1", value)


def _emit_python(action: str, p: dict, i: int) -> list[str]:
    if action == "navigate":
        return [f"page.goto({_python_string(p['url'])}, wait_until={_python_string(NAVIGATION_LOAD_STATE)}, timeout={NAVIGATION_TIMEOUT_MS})"]
    if action == "click":
        return [f"page.locator({_python_string(p['selector'])}).click(timeout={ACTION_TIMEOUT_MS})"]
    if action in {"check", "uncheck"}:
        method = action
        return [f"page.locator({_python_string(p['selector'])}).{method}(timeout={ACTION_TIMEOUT_MS})"]
    if action == "fill":
        return [f"page.locator({_python_string(p['selector'])}).fill({_python_string(p['value'])}, timeout={ACTION_TIMEOUT_MS})"]
    if action == "select_option":
        return [f"page.locator({_python_string(p['selector'])}).select_option(label={_python_string(p['option_label'])}, timeout={ACTION_TIMEOUT_MS})"]
    if action == "assert_page_loaded":
        return [f"page.wait_for_load_state({_python_string(NAVIGATION_LOAD_STATE)}, timeout={NAVIGATION_TIMEOUT_MS})"]
    if action == "assert_title":
        return [f"expect(page).to_have_title({_python_string(p['expected'])}, timeout={ASSERTION_TIMEOUT_MS})"]
    if action == "assert_url":
        return [f"expect(page).to_have_url({_python_string(p['expected'])}, timeout={ASSERTION_TIMEOUT_MS})"]
    if action == "assert_visible":
        lines = [f"element_{i} = page.locator({_python_string(p['selector'])})", f"expect(element_{i}).to_be_visible(timeout={ASSERTION_TIMEOUT_MS})"]
        if p.get("expected_text") is not None:
            lines.append(f"expect(element_{i}).to_have_js_property('innerText', {_python_string(_normalized_expected(p['expected_text']))}, timeout={ASSERTION_TIMEOUT_MS})")
        return lines
    if action == "assert_hidden":
        return [f"expect(page.locator({_python_string(p['selector'])})).to_be_hidden(timeout={ASSERTION_TIMEOUT_MS})"]
    if action == "assert_value":
        return [f"expect(page.locator({_python_string(p['selector'])})).to_have_value({_python_string(p['expected'])}, timeout={ASSERTION_TIMEOUT_MS})"]
    if action == "assert_text_contains":
        target = f"page.locator({_python_string(p['selector'])})" if p.get("selector") else "page.locator('body')"
        pattern = _python_string(".*" + _regex_escape(_normalized_expected(p["expected_text"])) + ".*")
        return [f"target_{i} = {target}", f"expect(target_{i}).to_be_visible(timeout={ASSERTION_TIMEOUT_MS})", f"expect(target_{i}).to_have_text(re.compile({pattern}, re.DOTALL), use_inner_text=True, timeout={ASSERTION_TIMEOUT_MS})"]
    if action == "assert_checked":
        return [f"expect(page.locator({_python_string(p['selector'])})).to_be_checked(timeout={ASSERTION_TIMEOUT_MS})"]
    if action == "assert_unchecked":
        return [f"expect(page.locator({_python_string(p['selector'])})).not_to_be_checked(timeout={ASSERTION_TIMEOUT_MS})"]
    if action == "assert_selected":
        lines = [
            f"element_{i} = page.locator({_python_string(p['selector'])})",
            f"expect(element_{i}).to_be_visible(timeout={ASSERTION_TIMEOUT_MS})",
            f"kind_{i} = element_{i}.evaluate(\"element => ({{tag: element.tagName.toLowerCase(), type: element.type}})\")",
        ]
        lines.extend(_python_selected_assertion(p, i))
        return lines
    if action in {"assert_enabled", "assert_disabled"}:
        method = "to_be_enabled" if action == "assert_enabled" else "to_be_disabled"
        return [f"expect(page.locator({_python_string(p['selector'])})).{method}(timeout={ASSERTION_TIMEOUT_MS})"]
    raise TestPlanExportError(f"The Python exporter does not support action {action}.")


def _python_selected_assertion(p: dict, i: int) -> list[str]:
    expected = p.get("expected")
    lines = [f"if kind_{i}['tag'] == 'input' and kind_{i}['type'] == 'radio':"]
    if expected is not None:
        lines.append("    raise AssertionError('Radio assert_selected does not accept an expected value.')")
    else:
        lines.append(f"    expect(element_{i}).to_be_checked(timeout={ASSERTION_TIMEOUT_MS})")
    lines.extend([f"elif kind_{i}['tag'] == 'select':"])
    if expected is None:
        lines.append("    raise AssertionError('Select assert_selected requires an expected label or value.')")
    else:
        lines.extend([
            f"    selected_{i} = element_{i}.locator('option:checked').first",
            f"    options_{i} = element_{i}.locator('option')",
            f"    option_labels_{i} = options_{i}.all_inner_texts()",
            f"    option_values_{i} = [option.get_attribute('value') for option in options_{i}.all()]",
            f"    if {_python_string(expected)} in option_labels_{i}:",
            f"        expect(selected_{i}).to_have_js_property('innerText', {_python_string(expected)}, timeout={ASSERTION_TIMEOUT_MS})",
            f"    elif {_python_string(expected)} in option_values_{i}:",
            f"        expect(selected_{i}).to_have_attribute('value', {_python_string(expected)}, timeout={ASSERTION_TIMEOUT_MS})",
            "    else:",
            "        raise AssertionError('Expected option label or value is not available.')",
        ])
    lines.append("else:")
    lines.append("    raise AssertionError('assert_selected supports radio inputs and select elements.')")
    return lines


def _emit_typescript(action: str, p: dict, i: int) -> list[str]:
    if action == "navigate":
        return [f"await page.goto({_js_string(p['url'])}, {{ waitUntil: {_js_string(NAVIGATION_LOAD_STATE)}, timeout: {NAVIGATION_TIMEOUT_MS} }});"]
    if action == "click":
        return [f"await page.locator({_js_string(p['selector'])}).click({{ timeout: {ACTION_TIMEOUT_MS} }});"]
    if action in {"check", "uncheck"}:
        method = "check" if action == "check" else "uncheck"
        return [f"await page.locator({_js_string(p['selector'])}).{method}({{ timeout: {ACTION_TIMEOUT_MS} }});"]
    if action == "fill":
        return [f"await page.locator({_js_string(p['selector'])}).fill({_js_string(p['value'])}, {{ timeout: {ACTION_TIMEOUT_MS} }});"]
    if action == "select_option":
        return [f"await page.locator({_js_string(p['selector'])}).selectOption({{ label: {_js_string(p['option_label'])} }}, {{ timeout: {ACTION_TIMEOUT_MS} }});"]
    if action == "assert_page_loaded":
        return [f"await page.waitForLoadState({_js_string(NAVIGATION_LOAD_STATE)}, {{ timeout: {NAVIGATION_TIMEOUT_MS} }});"]
    if action == "assert_title":
        return [f"await expect(page).toHaveTitle({_js_string(p['expected'])}, {{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_url":
        return [f"await expect(page).toHaveURL({_js_string(p['expected'])}, {{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_visible":
        lines = [f"const element_{i} = page.locator({_js_string(p['selector'])});", f"await expect(element_{i}).toBeVisible({{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
        if p.get("expected_text") is not None:
            lines.append(f"await expect(element_{i}).toHaveJSProperty('innerText', {_js_string(_normalized_expected(p['expected_text']))}, {{ timeout: {ASSERTION_TIMEOUT_MS} }});")
        return lines
    if action == "assert_hidden":
        return [f"await expect(page.locator({_js_string(p['selector'])})).toBeHidden({{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_value":
        return [f"await expect(page.locator({_js_string(p['selector'])})).toHaveValue({_js_string(p['expected'])}, {{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_text_contains":
        target = f"page.locator({_js_string(p['selector'])})" if p.get("selector") else "page.locator('body')"
        pattern = _js_string(".*" + _regex_escape(_normalized_expected(p["expected_text"])) + ".*")
        return [f"const target_{i} = {target};", f"await expect(target_{i}).toBeVisible({{ timeout: {ASSERTION_TIMEOUT_MS} }});", f"await expect(target_{i}).toHaveText(new RegExp({pattern}, 's'), {{ useInnerText: true, timeout: {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_checked":
        return [f"await expect(page.locator({_js_string(p['selector'])})).toBeChecked({{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_unchecked":
        return [f"await expect(page.locator({_js_string(p['selector'])})).not.toBeChecked({{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_selected":
        lines = [f"const element_{i} = page.locator({_js_string(p['selector'])});", f"await expect(element_{i}).toBeVisible({{ timeout: {ASSERTION_TIMEOUT_MS} }});", f"const kind_{i} = await element_{i}.evaluate(node => ({{ tag: node.tagName.toLowerCase(), type: (node as HTMLInputElement | HTMLSelectElement).type }}));"]
        lines.extend(_typescript_selected_assertion(p, i))
        return lines
    if action in {"assert_enabled", "assert_disabled"}:
        method = "toBeEnabled" if action == "assert_enabled" else "toBeDisabled"
        return [f"await expect(page.locator({_js_string(p['selector'])})).{method}({{ timeout: {ASSERTION_TIMEOUT_MS} }});"]
    raise TestPlanExportError(f"The TypeScript exporter does not support action {action}.")


def _typescript_selected_assertion(p: dict, i: int) -> list[str]:
    expected = p.get("expected")
    lines = [f"if (kind_{i}.tag === 'input' && kind_{i}.type === 'radio') {{"]
    if expected is not None:
        lines.append("  throw new Error('Radio assert_selected does not accept an expected value.');")
    else:
        lines.append(f"  await expect(element_{i}).toBeChecked({{ timeout: {ASSERTION_TIMEOUT_MS} }});")
    lines.append(f"}} else if (kind_{i}.tag === 'select') {{")
    if expected is None:
        lines.append("  throw new Error('Select assert_selected requires an expected label or value.');")
    else:
        lines.extend([
            f"  const selected_{i} = element_{i}.locator('option:checked').first();",
            f"  const options_{i} = await element_{i}.locator('option').all();",
            f"  const option_labels_{i} = await Promise.all(options_{i}.map(option => option.innerText()));",
            f"  const option_values_{i} = await Promise.all(options_{i}.map(option => option.getAttribute('value')));",
            f"  if (option_labels_{i}.includes({_js_string(expected)})) {{",
            f"    await expect(selected_{i}).toHaveJSProperty('innerText', {_js_string(expected)}, {{ timeout: {ASSERTION_TIMEOUT_MS} }});",
            f"  }} else if (option_values_{i}.includes({_js_string(expected)})) {{",
            f"    await expect(selected_{i}).toHaveAttribute('value', {_js_string(expected)}, {{ timeout: {ASSERTION_TIMEOUT_MS} }});",
            "  } else {",
            "    throw new Error('Expected option label or value is not available.');",
            "  }",
        ])
    lines.extend(["} else {", "  throw new Error('assert_selected supports radio inputs and select elements.');", "}"])
    return lines


def _emit_csharp(action: str, p: dict, i: int) -> list[str]:
    if action == "navigate":
        return [f"await page.GotoAsync({_csharp_string(p['url'])}, new() {{ WaitUntil = WaitUntilState.{NAVIGATION_LOAD_STATE.title()}, Timeout = {NAVIGATION_TIMEOUT_MS} }});"]
    if action == "click":
        return [f"await page.Locator({_csharp_string(p['selector'])}).ClickAsync(new() {{ Timeout = {ACTION_TIMEOUT_MS} }});"]
    if action in {"check", "uncheck"}:
        method = "CheckAsync" if action == "check" else "UncheckAsync"
        return [f"await page.Locator({_csharp_string(p['selector'])}).{method}(new() {{ Timeout = {ACTION_TIMEOUT_MS} }});"]
    if action == "fill":
        return [f"await page.Locator({_csharp_string(p['selector'])}).FillAsync({_csharp_string(p['value'])}, new() {{ Timeout = {ACTION_TIMEOUT_MS} }});"]
    if action == "select_option":
        return [f"await page.Locator({_csharp_string(p['selector'])}).SelectOptionAsync(new SelectOptionValue {{ Label = {_csharp_string(p['option_label'])} }}, new() {{ Timeout = {ACTION_TIMEOUT_MS} }});"]
    if action == "assert_page_loaded":
        return [f"await page.WaitForLoadStateAsync(LoadState.{NAVIGATION_LOAD_STATE.title()}, new() {{ Timeout = {NAVIGATION_TIMEOUT_MS} }});"]
    if action == "assert_title":
        return [f"await Expect(page).ToHaveTitleAsync({_csharp_string(p['expected'])}, new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_url":
        return [f"await Expect(page).ToHaveURLAsync({_csharp_string(p['expected'])}, new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_visible":
        lines = [f"var element_{i} = page.Locator({_csharp_string(p['selector'])});", f"await Expect(element_{i}).ToBeVisibleAsync(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
        if p.get("expected_text") is not None:
            lines.append(f"await Expect(element_{i}).ToHaveJSPropertyAsync(\"innerText\", {_csharp_string(_normalized_expected(p['expected_text']))}, new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});")
        return lines
    if action == "assert_hidden":
        return [f"await Expect(page.Locator({_csharp_string(p['selector'])})).ToBeHiddenAsync(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_value":
        return [f"await Expect(page.Locator({_csharp_string(p['selector'])})).ToHaveValueAsync({_csharp_string(p['expected'])}, new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_text_contains":
        target = f"page.Locator({_csharp_string(p['selector'])})" if p.get("selector") else 'page.Locator("body")'
        pattern = _csharp_string(".*" + _regex_escape(_normalized_expected(p["expected_text"])) + ".*")
        return [f"var target_{i} = {target};", f"await Expect(target_{i}).ToBeVisibleAsync(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});", f"await Expect(target_{i}).ToHaveTextAsync(new Regex({pattern}, RegexOptions.Singleline), new() {{ UseInnerText = true, Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_checked":
        return [f"await Expect(page.Locator({_csharp_string(p['selector'])})).ToBeCheckedAsync(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_unchecked":
        return [f"await Expect(page.Locator({_csharp_string(p['selector'])})).Not.ToBeCheckedAsync(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    if action == "assert_selected":
        lines = [f"var element_{i} = page.Locator({_csharp_string(p['selector'])});", f"await Expect(element_{i}).ToBeVisibleAsync(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});", f"var tag_{i} = await element_{i}.EvaluateAsync<string>(\"element => element.tagName.toLowerCase()\");", f"var type_{i} = await element_{i}.EvaluateAsync<string>(\"element => element.type\");", f"if (tag_{i} == \"input\" && type_{i} == \"radio\") {{"]
        expected = p.get("expected")
        if expected is not None:
            lines.append('    Assert.Fail("Radio assert_selected does not accept an expected value.");')
        else:
            lines.append(f"    await Expect(element_{i}).ToBeCheckedAsync(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});")
        lines.append(f"}} else if (tag_{i} == \"select\") {{")
        if expected is None:
            lines.append('    Assert.Fail("Select assert_selected requires an expected label or value.");')
        else:
            lines.extend([
                f"    var selected_{i} = element_{i}.Locator(\"option:checked\");",
                f"    var has_label_{i} = false;",
                f"    var has_value_{i} = false;",
                f"    foreach (var option_{i} in await element_{i}.Locator(\"option\").AllAsync()) {{",
                f"        has_label_{i} |= await option_{i}.InnerTextAsync() == {_csharp_string(expected)};",
                f"        has_value_{i} |= await option_{i}.GetAttributeAsync(\"value\") == {_csharp_string(expected)};",
                "    }",
                f"    if (has_label_{i}) {{",
                f"        await Expect(selected_{i}).ToHaveJSPropertyAsync(\"innerText\", {_csharp_string(expected)}, new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});",
                f"    }} else if (has_value_{i}) {{",
                f"        await Expect(selected_{i}).ToHaveAttributeAsync(\"value\", {_csharp_string(expected)}, new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});",
                "    } else {",
                '        Assert.Fail("Expected option label or value is not available.");',
                "    }",
            ])
        lines.extend(["} else {", '    Assert.Fail("assert_selected supports radio inputs and select elements.");', "}"])
        return lines
    if action in {"assert_enabled", "assert_disabled"}:
        method = "ToBeEnabledAsync" if action == "assert_enabled" else "ToBeDisabledAsync"
        return [f"await Expect(page.Locator({_csharp_string(p['selector'])})).{method}(new() {{ Timeout = {ASSERTION_TIMEOUT_MS} }});"]
    raise TestPlanExportError(f"The C# exporter does not support action {action}.")


# This explicit table is the support contract for all three target languages.
ACTION_EXPORT_HANDLERS: dict[str, tuple[Callable, Callable, Callable]] = {
    "navigate": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_page_loaded": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_title": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_visible": (_emit_python, _emit_typescript, _emit_csharp),
    "click": (_emit_python, _emit_typescript, _emit_csharp),
    "check": (_emit_python, _emit_typescript, _emit_csharp),
    "uncheck": (_emit_python, _emit_typescript, _emit_csharp),
    "fill": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_hidden": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_url": (_emit_python, _emit_typescript, _emit_csharp),
    "select_option": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_text_contains": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_value": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_checked": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_unchecked": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_selected": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_enabled": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_disabled": (_emit_python, _emit_typescript, _emit_csharp),
}

_ALLOWED_EXPORT_PARAMETERS = {
    "navigate": {"url"}, "assert_page_loaded": set(), "assert_title": {"expected"},
    "assert_visible": {"selector", "expected_text"}, "click": {"selector"},
    "check": {"selector"}, "uncheck": {"selector"},
    "fill": {"selector", "value"}, "assert_hidden": {"selector"},
    "assert_url": {"expected"}, "select_option": {"selector", "option_label"},
    "assert_text_contains": {"selector", "expected_text"}, "assert_checked": {"selector"},
    "assert_value": {"selector", "expected"},
    "assert_unchecked": {"selector"},
    "assert_selected": {"selector", "expected"}, "assert_enabled": {"selector"},
    "assert_disabled": {"selector"},
}


def _validate_action_parameters(exportable: ExportableTestCase) -> None:
    _validate_export_safety(exportable)
    for saved in exportable.plans:
        try:
            validate_executable_plan(saved.version.qa_test_plan)
        except PlanValidationError as error:
            raise TestPlanExportError("The saved automation plan is invalid or contains an unsupported action.") from error
        for action in saved.version.qa_test_plan.steps:
            if action.action not in ACTION_EXPORT_HANDLERS:
                raise TestPlanExportError(
                    f"Cannot export {exportable.test_case.public_id or exportable.test_case.name}: "
                    f"step {saved.step_order + 1}, action {action.action}, is unsupported."
                )
            extra = set(action.parameters) - _ALLOWED_EXPORT_PARAMETERS[action.action]
            if extra:
                raise TestPlanExportError(
                    f"Cannot export {exportable.test_case.public_id or exportable.test_case.name}: "
                    f"step {saved.step_order + 1}, action {action.action}, has unsupported parameters."
                )


def _python_function_name(name: str) -> str:
    slug = _slug(name).replace("-", "_")
    if not slug or slug[0].isdigit():
        slug = "test_" + (slug or "case")
    return "test_" + slug.removeprefix("test_")


def _ts_test_name(name: str) -> str:
    return name.strip() or "Exported TestCase"


def _csharp_identifier(name: str) -> str:
    parts = [part for part in _slug(name).split("-") if part]
    value = "".join(part[:1].upper() + part[1:] for part in parts) or "ExportedTestCase"
    if value[0].isdigit():
        value = "Test" + value
    return value


def _code_for(exportable: ExportableTestCase, language: str, *, identifier: str | None = None) -> str:
    _validate_action_parameters(exportable)
    handlers_index = {"python": 0, "typescript": 1, "csharp": 2}.get(language)
    if handlers_index is None:
        raise TestPlanExportError("Choose Python, TypeScript, or C# for source export.")
    lines: list[str]
    case = exportable.test_case
    if language == "python":
        lines = ["# Saved source for review; consult the project manifest for validation status.", "import re", "import pytest", "from playwright.sync_api import Page, expect", PYTHON_RUNTIME, "", f"def {_python_function_name(case.name)}(page: Page) -> None:", "    errors = []"]
    elif language == "typescript":
        lines = ["// Saved source for review; consult the project manifest for validation status.", "import { test, expect } from '@playwright/test';", TYPESCRIPT_RUNTIME, "", f"test({_js_string(_ts_test_name(case.name))}, async ({{ page }}) => {{", "    const errors: string[] = [];"]
    else:
        class_name = (identifier or _csharp_identifier(case.name)) + "Tests"
        method_name = _csharp_identifier(case.name) + "Test"
        lines = ["// Saved source for review; consult the project manifest for validation status.", "using System.Text.RegularExpressions;", "using System.Threading.Tasks;", "using Microsoft.Playwright;", "using Microsoft.Playwright.NUnit;", "using NUnit.Framework;", "", "[TestFixture]", f"public class {class_name} : PageTest", "{"]
        lines.append(CSHARP_RUNTIME)
        lines.extend(["    [Test]", f"    public async Task {method_name}()", "    {", "        var errors = new System.Collections.Generic.List<string>();"])
    indent = "    "
    if language == "csharp":
        indent = "        "
    action_index = 0
    active_segment = None
    source_base = case.base_url or exportable.plans[0].version.qa_test_plan.url
    for saved in exportable.plans:
        new_segment = saved.segment_order != active_segment
        if saved.segment_order != active_segment:
            active_segment = saved.segment_order
            segment_label = f"Execution segment {active_segment + 1}"
            if language == "python":
                lines.extend([f"    # {segment_label}", "    page = page.context.new_page()"])
            elif language == "typescript":
                lines.extend([f"  // {segment_label}", "  page = await page.context().newPage();"])
            elif action_index == 0:
                lines.extend([f"        // {segment_label}", "        var page = await Page.Context.NewPageAsync();"])
            else:
                lines.extend([f"        // {segment_label}", "        page = await page.Context.NewPageAsync();"])
        lines.append(indent + ("try:" if language == "python" else "try {"))
        if new_segment and saved.version.qa_test_plan.steps[0].action != "navigate":
            target = saved.segment_base_url or case.base_url
            if not target:
                raise TestPlanExportError("State-dependent execution requires a configured segment URL or an explicit navigation action.")
            emitted = ACTION_EXPORT_HANDLERS["navigate"][handlers_index]("navigate", {"url": target}, action_index)
            lines.extend(indent + "    " + line for line in _configured_lines(emitted, "navigate", {"url": target}, language, source_base))
        for step in saved.version.qa_test_plan.steps:
            action_index += 1
            handler = ACTION_EXPORT_HANDLERS[step.action][handlers_index]
            emitted = handler(step.action, step.parameters, action_index)
            lines.extend(indent + "    " + line for line in _configured_lines(emitted, step.action, step.parameters, language, source_base))
        label = f"Step {saved.step_order + 1} (plan v{saved.version.version})"
        if language == "python":
            lines.extend([indent + "except Exception as error:", indent + f"    errors.append({_python_string(label)} + ': ' + str(error))"])
            if saved.failure_policy == "BLOCK_REST":
                lines.append(indent + "    raise")
        elif language == "typescript":
            lines.extend([indent + "} catch (error) {", indent + f"    errors.push({_js_string(label)} + ': ' + String(error));"])
            if saved.failure_policy == "BLOCK_REST":
                lines.append(indent + "    throw error;")
            lines.append(indent + "}")
        else:
            lines.extend([indent + "} catch (System.Exception error) {", indent + f"    errors.Add({_csharp_string(label)} + \": \" + error.ToString());"])
            if saved.failure_policy == "BLOCK_REST":
                lines.append(indent + "    throw;")
            lines.append(indent + "}")
    if language == "typescript":
        lines.append("    if (errors.length) throw new Error(errors.join('\\n'));")
        lines.extend(["});", ""])
    elif language == "csharp":
        lines.append('        if (errors.Count > 0) Assert.Fail(string.Join("\\n", errors));')
        lines.extend(["    }", "}", ""])
    else:
        lines.extend(["    if errors:", "        pytest.fail('\\n'.join(errors), pytrace=False)"])
        lines.append("")
    if language == "python":
        source = "\n".join(lines)
        try:
            compile(source, "<playwright-export>", "exec")
        except SyntaxError as error:
            raise TestPlanExportError("The generated Python source is invalid.") from error
        return source
    return "\n".join(lines)


def _configured_lines(lines: list[str], action: str, p: dict, language: str, source_base: str) -> list[str]:
    quote = {"python": _python_string, "typescript": _js_string, "csharp": _csharp_string}[language]
    url_helper = {"python": "export_url", "typescript": "exportUrl", "csharp": "ExportUrl"}[language]
    timeout_helper = {"python": "export_timeout", "typescript": "exportTimeout", "csharp": "ExportTimeout"}[language]
    result = []
    for line in lines:
        if action in {"navigate", "assert_url"}:
            value = p["url" if action == "navigate" else "expected"]
            line = line.replace(quote(value), f"{url_helper}({quote(value)}, {quote(source_base)})", 1)
        # Canonical handlers retain shared timing defaults; generated runtimes
        # permit explicit overrides without silently dropping plan parameters.
        timeout_name = "NAVIGATION_TIMEOUT_MS" if action in {"navigate", "assert_page_loaded"} else ("ASSERTION_TIMEOUT_MS" if action.startswith("assert_") else "ACTION_TIMEOUT_MS")
        default = NAVIGATION_TIMEOUT_MS if timeout_name == "NAVIGATION_TIMEOUT_MS" else ASSERTION_TIMEOUT_MS if timeout_name == "ASSERTION_TIMEOUT_MS" else ACTION_TIMEOUT_MS
        pattern = rf"(timeout=|timeout: |Timeout = ){default}\b"
        # Only rewrite generated argument tokens, never selector/input/expected
        # string literals that happen to contain text such as timeout=5000.
        fragments = re.split(r'''("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''', line)
        line = "".join(
            fragment if index % 2 else re.sub(pattern, lambda m: m[1] + f"{timeout_helper}({quote(timeout_name)}, {default})", fragment)
            for index, fragment in enumerate(fragments)
        )
        result.append(line)
    return result


def python_source(exportable: ExportableTestCase) -> str:
    return _code_for(exportable, "python")


def typescript_source(exportable: ExportableTestCase) -> str:
    return _code_for(exportable, "typescript")


def csharp_source(exportable: ExportableTestCase) -> str:
    return _code_for(exportable, "csharp")


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-._ ")
    if slug in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}:
        slug = "item-" + slug
    return slug[:80].rstrip("-._ ") or "testcase"


def safe_basename(test_case: TestCase, used: set[str] | None = None) -> str:
    prefix = _slug(test_case.public_id or "TC")
    candidate = f"{prefix}_{_slug(test_case.name)}"
    if used is not None:
        base = candidate
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        used.add(candidate.casefold())
    return candidate


def _zip_info(path: str) -> zipfile.ZipInfo:
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts or "\\" in path or ":" in path or "\x00" in path:
        raise TestPlanExportError("Generated archive path is not safe.")
    info = zipfile.ZipInfo(str(normalized), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def _project_readme(language: str, exports: list[ExportableTestCase]) -> str:
    commands = {
        "python": "python -m pip install -r requirements.txt\npython -m playwright install chromium\npython -m pytest -q",
        "typescript": "npm install\nnpx playwright install chromium\nnpx playwright test",
        "csharp": "dotnet restore\ndotnet build\npwsh csharp/bin/Debug/net8.0/playwright.ps1 install chromium\ndotnet test",
    }
    return (
        "# AI QA Agent Playwright export\n\n"
        "These tests were generated deterministically from saved, versioned TestPlans. "
        "They do not call AI QA Agent or an LLM at runtime.\n\n"
        + ("Export eligibility: APPROVED_AND_BROWSER_VALIDATED for the exact saved versions. "
           "This records prior Browser Validation, not a standalone execution result.\n\n"
           if all(item.verification == "APPROVED_AND_BROWSER_VALIDATED" for item in exports)
           else "REVIEW_ONLY: this source project is not verified automation. Review, approve and Browser Validate its saved versions before using it as verified automation.\n\n")
        +
        f"## {language.title()}\n\n"
        "Install the listed dependencies and browser on your own machine before running:\n\n"
        "```text\n" + commands[language] + "\n```\n\n"
        "Selectors are preserved as Playwright locator strings from the saved plan. "
        "Review test data and target URLs before execution.\n\n"
        "## Configuration and CI\n\n"
        "By default URLs are the exact saved targets. Set BASE_URL to an http(s) origin "
        "(no credentials, path, query or fragment) to replace the TestCase base origin; "
        "paths, queries, fragments and navigation to other origins are preserved. "
        "URL assertions use the same explicit substitution.\n\n"
        "TIMEOUT_MS overrides action/assertion/navigation timeouts. ACTION_TIMEOUT_MS, "
        "ASSERTION_TIMEOUT_MS and NAVIGATION_TIMEOUT_MS override each individually; "
        "all must be positive integers. Browser state is shared across TestSteps in a segment "
        "and browser context across segments; each TestCase has an isolated context. "
        "CONTINUE records failures and continues with the next TestStep; BLOCK_REST stops immediately. "
        "Any recorded failure fails the test.\n\n"
        + ("BROWSER=chromium|firefox|webkit; HEADLESS=true|false (default true). "
           "Install the selected browser first.\n\n" if language != "csharp" else
           "NUnit uses csharp/export.runsettings. For browser/headed execution use "
           "`dotnet test -- Playwright.BrowserName=firefox Playwright.LaunchOptions.Headless=false`. "
           "Install the selected browser first.\n\n")
        + ("The included GitHub Actions workflow installs Chromium and runs pytest. "
           "Set the repository BASE_URL variable to a reachable test target or start your local fixture in CI.\n"
           if language == "python" else "CI: run the installation/build/browser commands above, then "
           + ("`npx playwright test`; its exit code fails CI.\n" if language == "typescript" else "`dotnet test`; its exit code fails CI.\n"))
    )


def _project_files(language: str, exports: list[ExportableTestCase], *, suite_name: str | None = None) -> dict[str, bytes]:
    used: set[str] = set()
    portable_used: set[str] = set()
    for exportable in exports:
        _validate_action_parameters(exportable)
        _validate_export_safety(exportable, suite_name)
    files: dict[str, bytes] = {"README.md": _project_readme(language, exports).encode("utf-8")}
    manifest_cases = []
    extension = {"python": ".py", "typescript": ".spec.ts", "csharp": "Tests.cs"}[language]
    for exportable in exports:
        case = exportable.test_case
        base = safe_basename(case, used)
        file_name = ("test_" + base.replace("-", "_") if language == "python" else base) + extension if language != "csharp" else _csharp_identifier(base) + extension
        # C# class/file names can collide after normalization; make those deterministic too.
        if language == "csharp":
            class_base = file_name
            suffix = 2
            while file_name.casefold() in used:
                file_name = f"{class_base[:-len(extension)]}{suffix}{extension}"
                suffix += 1
            used.add(file_name.casefold())
        if language == "python":
            content = python_source(exportable)
            root = "python/tests/"
        elif language == "typescript":
            content = typescript_source(exportable)
            root = "typescript/tests/"
        else:
            content = _code_for(exportable, "csharp", identifier=file_name.removesuffix(extension))
            root = "csharp/"
        files[root + file_name] = content.encode("utf-8")
        portable_name = safe_basename(case, portable_used) + ".testplan.json"
        files["portable/" + portable_name] = portable_json(exportable).encode("utf-8")
        manifest_cases.append({
            "public_id": case.public_id,
            "name": case.name,
            "source_file": root + file_name,
            "portable_file": "portable/" + portable_name,
            "verification": exportable.verification,
            "plan_versions": [
                {
                    "step_order": item.step_order,
                    "version": item.version.version,
                    "version_id": str(item.version.id),
                    "provenance": item.version.origin.value if item.version.origin else None,
                }
                for item in exportable.plans
            ],
        })
    if language == "python":
        files["requirements.txt"] = f"pytest=={installed_version('pytest')}\nplaywright=={installed_version('playwright')}\n".encode()
        files["pytest.ini"] = b"[pytest]\ntestpaths = python/tests\npython_files = test_*.py\n"
        files["python/tests/conftest.py"] = PYTHON_CONFTEST.encode()
        files[".github/workflows/tests.yml"] = PYTHON_CI.encode()
    elif language == "typescript":
        files["package.json"] = (json.dumps({
            "private": True,
            "scripts": {"test": "playwright test"},
            "devDependencies": {"@playwright/test": installed_version('playwright')},
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        files["playwright.config.ts"] = (
            "import { defineConfig } from '@playwright/test';\n"
            "const browser = process.env.BROWSER ?? 'chromium';\n"
            "if (!['chromium', 'firefox', 'webkit'].includes(browser)) throw new Error('Invalid BROWSER');\n"
            "const headless = (process.env.HEADLESS ?? 'true').toLowerCase();\n"
            "if (!['true', 'false', '1', '0'].includes(headless)) throw new Error('Invalid HEADLESS');\n"
            "const testTimeout = Number(process.env.TEST_TIMEOUT_MS ?? 120000);\n"
            "if (!Number.isInteger(testTimeout) || testTimeout <= 0) throw new Error('Invalid TEST_TIMEOUT_MS');\n"
            "export default defineConfig({ testDir: './typescript/tests', timeout: testTimeout, retries: 0, "
            "use: { browserName: browser as 'chromium' | 'firefox' | 'webkit', headless: ['true', '1'].includes(headless) } });\n"
        ).encode()
        files["tsconfig.json"] = b'{\n  "compilerOptions": { "target": "ES2020", "module": "commonjs", "strict": true },\n  "include": ["typescript/**/*.ts", "playwright.config.ts"]\n}\n'
    else:
        files["csharp/AIQAAgent.Export.csproj"] = (
            '<Project Sdk="Microsoft.NET.Sdk">\n  <PropertyGroup>\n    <TargetFramework>net8.0</TargetFramework>\n    <IsTestProject>true</IsTestProject>\n    <Nullable>enable</Nullable>\n  </PropertyGroup>\n  <ItemGroup>\n'
            '    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.10.0" />\n'
            '    <PackageReference Include="Microsoft.Playwright.NUnit" Version="1.44.0" />\n'
            '    <PackageReference Include="NUnit" Version="3.14.0" />\n'
            '    <PackageReference Include="NUnit3TestAdapter" Version="4.5.0" />\n'
            '  </ItemGroup>\n</Project>\n'
        ).encode("utf-8")
        files["csharp/AIQAAgent.Export.csproj"] = files["csharp/AIQAAgent.Export.csproj"].replace(
            b"<Nullable>enable</Nullable>", b"<Nullable>enable</Nullable>\n    <RunSettingsFilePath>$(MSBuildProjectDirectory)/export.runsettings</RunSettingsFilePath>"
        )
        files["csharp/export.runsettings"] = b'<RunSettings><Playwright><BrowserName>chromium</BrowserName><LaunchOptions><Headless>true</Headless></LaunchOptions></Playwright></RunSettings>\n'
        files["AIQAAgent.Export.sln"] = (
            'Microsoft Visual Studio Solution File, Format Version 12.00\n'
            'Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "AIQAAgent.Export", "csharp/AIQAAgent.Export.csproj", "{641D16E0-4D79-4AD8-97FD-D49534A80C50}"\nEndProject\n'
            'Global\nGlobalSection(SolutionConfigurationPlatforms) = preSolution\nDebug|Any CPU = Debug|Any CPU\nRelease|Any CPU = Release|Any CPU\nEndGlobalSection\n'
            'GlobalSection(ProjectConfigurationPlatforms) = postSolution\n'
            '{641D16E0-4D79-4AD8-97FD-D49534A80C50}.Debug|Any CPU.ActiveCfg = Debug|Any CPU\n'
            '{641D16E0-4D79-4AD8-97FD-D49534A80C50}.Debug|Any CPU.Build.0 = Debug|Any CPU\n'
            '{641D16E0-4D79-4AD8-97FD-D49534A80C50}.Release|Any CPU.ActiveCfg = Release|Any CPU\n'
            '{641D16E0-4D79-4AD8-97FD-D49534A80C50}.Release|Any CPU.Build.0 = Release|Any CPU\n'
            'EndGlobalSection\nEndGlobal\n'
        ).encode()
    manifest = {
        "format_version": EXPORT_FORMAT_VERSION,
        "generated_at": _snapshot_timestamp(exports),
        "standalone_execution": "NOT_TESTED",
        "language": language,
        "framework": {"python": "pytest + Playwright", "typescript": "Playwright Test", "csharp": "NUnit + Microsoft.Playwright"}[language],
        "suite": suite_name,
        "test_cases": manifest_cases,
    }
    files["export-manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return files


def project_zip(language: str, exports: list[ExportableTestCase], *, suite_name: str | None = None) -> bytes:
    if language not in {"python", "typescript", "csharp"}:
        raise TestPlanExportError("Choose Python, TypeScript, or C# for project export.")
    if not exports:
        raise TestPlanExportError("Select at least one TestCase to export.")
    files = _project_files(language, exports, suite_name=suite_name)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for name in sorted(files):
            archive.writestr(_zip_info(name), files[name])
    return output.getvalue()


def portable_zip(exports: list[ExportableTestCase], *, suite_name: str | None = None) -> bytes:
    if not exports:
        raise TestPlanExportError("Select at least one TestCase to export.")
    used: set[str] = set()
    files = {"README.md": (
        "# Portable TestPlans\n\n"
        "These versioned plans were exported from saved canonical automation. "
        "Exporting does not make an AI request. Portable JSON is versioned and specific to AI QA Agent.\n"
    ).encode("utf-8")}
    cases = []
    for exportable in exports:
        _validate_export_safety(exportable, suite_name)
        case = exportable.test_case
        filename = safe_basename(case, used) + ".testplan.json"
        files["portable/" + filename] = portable_json(exportable).encode("utf-8")
        cases.append({"public_id": case.public_id, "name": case.name, "file": "portable/" + filename})
    files["export-manifest.json"] = (json.dumps({
        "format_version": EXPORT_FORMAT_VERSION,
        "generated_at": _snapshot_timestamp(exports),
        "language": "portable-json",
        "suite": suite_name,
        "test_cases": cases,
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for name in sorted(files):
            archive.writestr(_zip_info(name), files[name])
    return output.getvalue()


def _snapshot_timestamp(exports: list[ExportableTestCase]) -> str:
    """Stable snapshot time, rather than a wall clock that changes ZIP bytes."""
    return max(item.version.created_at for export in exports for item in export.plans).isoformat()


class TestPlanExportService:
    """Load exact current saved versions and render offline exports."""

    def __init__(self, test_cases: TestCaseRepository, plan_store: PlanStore, *,
                 lifecycle: AutomationLifecycleService | None = None,
                 review: TestCaseReviewService | None = None) -> None:
        self._test_cases = test_cases
        self._plan_store = plan_store
        self._lifecycle = lifecycle
        self._review = review

    def get(self, test_case_id) -> ExportableTestCase:
        test_case = self._test_cases.get(test_case_id)
        if test_case is None:
            raise TestPlanExportError("TestCase not found.")
        return _plans_for_case(test_case, self._plan_store)

    def get_verified(self, test_case_id) -> ExportableTestCase:
        exportable = self.get(test_case_id)
        case = exportable.test_case
        if self._lifecycle is None or self._review is None:
            raise TestPlanExportError("Verified project export requires approval and Browser Validation metadata.")
        selected = plan_fingerprint_for_versions(case, {item.test_step_id: item.version.id for item in exportable.plans})
        def eligible():
            current = self._test_cases.get(case.id)
            return (
                current is not None
                and definition_fingerprint(current) == definition_fingerprint(case)
                and plan_fingerprint(case, self._plan_store) == selected
                and self._review.status(case.id) == TestCaseReviewStatus.APPROVED
                and self._review.validation_approved_for(case)
                and self._lifecycle.status(case) == AutomationStatus.AUTOMATION_READY
            )
        if not eligible():
            raise TestPlanExportError("Verified project export requires current approved, Browser Validated Automation Ready plans.")
        _validate_action_parameters(exportable)
        # Readiness helpers use current versions. Check again after validation
        # so a version/definition change during selection cannot bless a draft.
        if not eligible():
            raise TestPlanExportError("Saved automation changed during export; review and validate its current versions.")
        return replace(exportable, verification="APPROVED_AND_BROWSER_VALIDATED")

    def get_many(self, test_case_ids: list, *, preserve_order: bool = False) -> list[ExportableTestCase]:
        if not test_case_ids:
            raise TestPlanExportError("Select at least one TestCase to export.")
        if len(set(test_case_ids)) != len(test_case_ids):
            raise TestPlanExportError("A TestCase was selected more than once.")
        exports = [self.get(item) for item in test_case_ids]
        if not preserve_order:
            exports.sort(key=lambda item: (item.test_case.public_id or "", str(item.test_case.id)))
        return exports

    def portable_json(self, test_case_id) -> tuple[str, str]:
        exportable = self.get(test_case_id)
        return portable_json(exportable), safe_basename(exportable.test_case) + ".testplan.json"

    def source(self, test_case_id, language: str) -> tuple[str, str]:
        exportable = self.get(test_case_id)
        generator = {"python": python_source, "typescript": typescript_source, "csharp": csharp_source}.get(language)
        if generator is None:
            raise TestPlanExportError("Choose Python, TypeScript, or C# for source export.")
        extension = {"python": ".py", "typescript": ".spec.ts", "csharp": "Tests.cs"}[language]
        basename = safe_basename(exportable.test_case)
        if language == "python":
            basename = "test_" + basename.replace("-", "_")
        return generator(exportable), basename + extension

    def bulk_zip(self, test_case_ids: list, target: str) -> tuple[bytes, str]:
        exports = self.get_many(test_case_ids)
        if target == "portable":
            return portable_zip(exports), "ai-qa-portable-testplans.zip"
        language = {"python": "python", "typescript": "typescript", "csharp": "csharp"}.get(target)
        if language is None:
            raise TestPlanExportError("Choose Portable JSON, Python, TypeScript, or C# export.")
        exports = [self.get_verified(item.test_case.id) for item in exports]
        return project_zip(language, exports), f"ai-qa-{language}-project.zip"

