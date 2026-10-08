"""Deterministic, local-only export of saved canonical TestPlanVersions."""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Callable
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from qa_agent.models import QATestPlan, QATestStep, TestCase, TestPlanVersion
from qa_agent.plan_store import PlanStore
from qa_agent.test_case_repository import TestCaseRepository
from qa_agent.test_plan_validation import PlanValidationError, validate_executable_plan


PORTABLE_SCHEMA_VERSION = 1
EXPORT_FORMAT_VERSION = "1.0"


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
                    action_name = next(
                        (item.action for item in version.qa_test_plan.steps
                         if item.action not in QATestStep.ACTION_PARAMETER_FIELDS),
                        "unknown",
                    )
                    raise TestPlanExportError(
                        f"Cannot export {test_case.public_id or test_case.name}: "
                        f"step {step.order + 1} uses unsupported action {action_name!r}."
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


_WINDOWS_PATH = re.compile(r"(?i)(?:^|[\s=:'\"(])(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/])")
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


def _emit_python(action: str, p: dict, i: int) -> list[str]:
    if action == "navigate":
        return [f"page.goto({_python_string(p['url'])})"]
    if action == "click":
        return [f"page.locator({_python_string(p['selector'])}).click()", "page.wait_for_load_state('load', timeout=10000)"]
    if action == "fill":
        return [f"page.locator({_python_string(p['selector'])}).fill({_python_string(p['value'])})"]
    if action == "select_option":
        return [f"page.locator({_python_string(p['selector'])}).select_option(label={_python_string(p['option_label'])})"]
    if action == "assert_page_loaded":
        return ["page.wait_for_load_state('load')"]
    if action == "assert_title":
        return [f"expect(page).to_have_title({_python_string(p['expected'])})"]
    if action == "assert_url":
        return [f"expect(page).to_have_url({_python_string(p['expected'])})"]
    if action == "assert_visible":
        lines = [f"element_{i} = page.locator({_python_string(p['selector'])})", f"expect(element_{i}).to_be_visible()"]
        if "expected_text" in p:
            lines.append(f"expect(element_{i}).to_have_text({_python_string(_normalized_expected(p['expected_text']))}, use_inner_text=True)")
        return lines
    if action == "assert_hidden":
        return [f"expect(page.locator({_python_string(p['selector'])})).to_be_hidden()"]
    if action == "assert_text_contains":
        target = f"page.locator({_python_string(p['selector'])})" if p.get("selector") else "page.locator('body')"
        expected = _python_string(_normalized_expected(p["expected_text"]))
        return [f"target_{i} = {target}", f"expect(target_{i}.get_by_text({expected}, exact=False)).to_be_visible()", f"expect(target_{i}).to_contain_text({expected}, use_inner_text=True)"]
    if action == "assert_checked":
        return [f"expect(page.locator({_python_string(p['selector'])})).to_be_checked()"]
    if action == "assert_selected":
        lines = [f"element_{i} = page.locator({_python_string(p['selector'])})", f"element_{i}.wait_for(state='visible', timeout=5000)", f"kind_{i} = element_{i}.evaluate(\"element => ({{tag: element.tagName.toLowerCase(), type: element.type}})\")"]
        lines.extend(_python_selected_assertion(p, i))
        return lines
    if action in {"assert_enabled", "assert_disabled"}:
        method = "to_be_enabled" if action == "assert_enabled" else "to_be_disabled"
        return [f"expect(page.locator({_python_string(p['selector'])})).{method}()"]
    raise TestPlanExportError(f"The Python exporter does not support action {action}.")


def _python_selected_assertion(p: dict, i: int) -> list[str]:
    expected = p.get("expected")
    lines = [f"if kind_{i}['tag'] == 'input' and kind_{i}['type'] == 'radio':"]
    if expected is not None:
        lines.append("    raise AssertionError('Radio assert_selected does not accept an expected value.')")
    else:
        lines.append(f"    expect(element_{i}).to_be_checked()")
    lines.extend([f"elif kind_{i}['tag'] == 'select':"])
    if expected is None:
        lines.append("    raise AssertionError('Select assert_selected requires an expected label or value.')")
    else:
        lines.extend([
            f"    selected_{i} = element_{i}.locator('option:checked').first",
            f"    label_{i} = selected_{i}.inner_text()",
            f"    value_{i} = selected_{i}.get_attribute('value')",
            f"    assert {_python_string(expected)} in {{label_{i}, value_{i}}}",
        ])
    lines.append("else:")
    lines.append("    raise AssertionError('assert_selected supports radio inputs and select elements.')")
    return lines


def _emit_typescript(action: str, p: dict, i: int) -> list[str]:
    if action == "navigate":
        return [f"await page.goto({_js_string(p['url'])});"]
    if action == "click":
        return [f"await page.locator({_js_string(p['selector'])}).click();", "await page.waitForLoadState('load', { timeout: 10000 });"]
    if action == "fill":
        return [f"await page.locator({_js_string(p['selector'])}).fill({_js_string(p['value'])});"]
    if action == "select_option":
        return [f"await page.locator({_js_string(p['selector'])}).selectOption({{ label: {_js_string(p['option_label'])} }});"]
    if action == "assert_page_loaded":
        return ["await page.waitForLoadState('load');"]
    if action == "assert_title":
        return [f"await expect(page).toHaveTitle({_js_string(p['expected'])});"]
    if action == "assert_url":
        return [f"await expect(page).toHaveURL({_js_string(p['expected'])});"]
    if action == "assert_visible":
        lines = [f"const element_{i} = page.locator({_js_string(p['selector'])});", f"await expect(element_{i}).toBeVisible();"]
        if "expected_text" in p:
            lines.append(f"await expect(element_{i}).toHaveText({_js_string(_normalized_expected(p['expected_text']))}, {{ useInnerText: true }});")
        return lines
    if action == "assert_hidden":
        return [f"await expect(page.locator({_js_string(p['selector'])})).toBeHidden();"]
    if action == "assert_text_contains":
        target = f"page.locator({_js_string(p['selector'])})" if p.get("selector") else "page.locator('body')"
        expected = _js_string(_normalized_expected(p["expected_text"]))
        return [f"const target_{i} = {target};", f"await expect(target_{i}.getByText({expected}, {{ exact: false }})).toBeVisible();", f"await expect(target_{i}).toContainText({expected}, {{ useInnerText: true }});"]
    if action == "assert_checked":
        return [f"await expect(page.locator({_js_string(p['selector'])})).toBeChecked();"]
    if action == "assert_selected":
        lines = [f"const element_{i} = page.locator({_js_string(p['selector'])});", f"await element_{i}.waitFor({{ state: 'visible', timeout: 5000 }});", f"const kind_{i} = await element_{i}.evaluate(node => ({{ tag: node.tagName.toLowerCase(), type: (node as HTMLInputElement | HTMLSelectElement).type }}));"]
        lines.extend(_typescript_selected_assertion(p, i))
        return lines
    if action in {"assert_enabled", "assert_disabled"}:
        method = "toBeEnabled" if action == "assert_enabled" else "toBeDisabled"
        return [f"await expect(page.locator({_js_string(p['selector'])})).{method}();"]
    raise TestPlanExportError(f"The TypeScript exporter does not support action {action}.")


def _typescript_selected_assertion(p: dict, i: int) -> list[str]:
    expected = p.get("expected")
    lines = [f"if (kind_{i}.tag === 'input' && kind_{i}.type === 'radio') {{"]
    if expected is not None:
        lines.append("  throw new Error('Radio assert_selected does not accept an expected value.');")
    else:
        lines.append(f"  await expect(element_{i}).toBeChecked();")
    lines.append(f"}} else if (kind_{i}.tag === 'select') {{")
    if expected is None:
        lines.append("  throw new Error('Select assert_selected requires an expected label or value.');")
    else:
        lines.extend([
            f"  const selected_{i} = element_{i}.locator('option:checked').first();",
            f"  const label_{i} = await selected_{i}.innerText();",
            f"  const value_{i} = await selected_{i}.getAttribute('value');",
            f"  expect([{f'label_{i}, value_{i}'}]).toContain({_js_string(expected)});",
        ])
    lines.extend(["} else {", "  throw new Error('assert_selected supports radio inputs and select elements.');", "}"])
    return lines


def _emit_csharp(action: str, p: dict, i: int) -> list[str]:
    if action == "navigate":
        return [f"await page.GotoAsync({_csharp_string(p['url'])});"]
    if action == "click":
        return [f"await page.Locator({_csharp_string(p['selector'])}).ClickAsync();", "await page.WaitForLoadStateAsync(LoadState.Load, new() { Timeout = 10000 });"]
    if action == "fill":
        return [f"await page.Locator({_csharp_string(p['selector'])}).FillAsync({_csharp_string(p['value'])});"]
    if action == "select_option":
        return [f"await page.Locator({_csharp_string(p['selector'])}).SelectOptionAsync(new SelectOptionValue {{ Label = {_csharp_string(p['option_label'])} }});"]
    if action == "assert_page_loaded":
        return ["await page.WaitForLoadStateAsync(LoadState.Load);"]
    if action == "assert_title":
        return [f"Assert.That(await page.TitleAsync(), Is.EqualTo({_csharp_string(p['expected'])}));"]
    if action == "assert_url":
        return [f"Assert.That(page.Url, Is.EqualTo({_csharp_string(p['expected'])}));"]
    if action == "assert_visible":
        lines = [f"var element_{i} = page.Locator({_csharp_string(p['selector'])});", f"await element_{i}.WaitForAsync(new() {{ State = WaitForSelectorState.Visible, Timeout = 5000 }});", f"Assert.That(await element_{i}.IsVisibleAsync(), Is.True);"]
        if "expected_text" in p:
            lines.append(f"Assert.That(await element_{i}.InnerTextAsync(), Is.EqualTo({_csharp_string(_normalized_expected(p['expected_text']))}));")
        return lines
    if action == "assert_hidden":
        return [f"Assert.That(await page.Locator({_csharp_string(p['selector'])}).IsHiddenAsync(), Is.True);"]
    if action == "assert_text_contains":
        target = f"page.Locator({_csharp_string(p['selector'])})" if p.get("selector") else 'page.Locator("body")'
        expected = _csharp_string(_normalized_expected(p["expected_text"]))
        return [f"var target_{i} = {target};", f"await target_{i}.GetByText({expected}, new() {{ Exact = false }}).WaitForAsync(new() {{ State = WaitForSelectorState.Visible, Timeout = 1500 }});", f"Assert.That(await target_{i}.InnerTextAsync(), Does.Contain({expected}));"]
    if action == "assert_checked":
        return [f"Assert.That(await page.Locator({_csharp_string(p['selector'])}).IsCheckedAsync(), Is.True);"]
    if action == "assert_selected":
        lines = [f"var element_{i} = page.Locator({_csharp_string(p['selector'])});", f"await element_{i}.WaitForAsync(new() {{ State = WaitForSelectorState.Visible, Timeout = 5000 }});", f"var tag_{i} = await element_{i}.EvaluateAsync<string>(\"element => element.tagName.toLowerCase()\");", f"var type_{i} = await element_{i}.GetAttributeAsync(\"type\");", f"if (tag_{i} == \"input\" && type_{i} == \"radio\") {{"]
        expected = p.get("expected")
        if expected is not None:
            lines.append('    Assert.Fail("Radio assert_selected does not accept an expected value.");')
        else:
            lines.append(f"    Assert.That(await element_{i}.IsCheckedAsync(), Is.True);")
        lines.append(f"}} else if (tag_{i} == \"select\") {{")
        if expected is None:
            lines.append('    Assert.Fail("Select assert_selected requires an expected label or value.");')
        else:
            lines.extend([
                f"    var selected_{i} = element_{i}.Locator(\"option:checked\").First;",
                f"    var label_{i} = await selected_{i}.InnerTextAsync();",
                f"    var value_{i} = await selected_{i}.GetAttributeAsync(\"value\");",
                f"    Assert.That(new[] {{ label_{i}, value_{i} }}, Does.Contain({_csharp_string(expected)}));",
            ])
        lines.extend(["} else {", '    Assert.Fail("assert_selected supports radio inputs and select elements.");', "}"])
        return lines
    if action in {"assert_enabled", "assert_disabled"}:
        expected = "true" if action == "assert_enabled" else "false"
        return [f"Assert.That(await page.Locator({_csharp_string(p['selector'])}).IsEnabledAsync(), Is.EqualTo({expected}));"]
    raise TestPlanExportError(f"The C# exporter does not support action {action}.")


# This explicit table is the support contract for all three target languages.
ACTION_EXPORT_HANDLERS: dict[str, tuple[Callable, Callable, Callable]] = {
    "navigate": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_page_loaded": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_title": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_visible": (_emit_python, _emit_typescript, _emit_csharp),
    "click": (_emit_python, _emit_typescript, _emit_csharp),
    "fill": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_hidden": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_url": (_emit_python, _emit_typescript, _emit_csharp),
    "select_option": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_text_contains": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_checked": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_selected": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_enabled": (_emit_python, _emit_typescript, _emit_csharp),
    "assert_disabled": (_emit_python, _emit_typescript, _emit_csharp),
}

_ALLOWED_EXPORT_PARAMETERS = {
    "navigate": {"url"}, "assert_page_loaded": set(), "assert_title": {"expected"},
    "assert_visible": {"selector", "expected_text"}, "click": {"selector"},
    "fill": {"selector", "value"}, "assert_hidden": {"selector"},
    "assert_url": {"expected"}, "select_option": {"selector", "option_label"},
    "assert_text_contains": {"selector", "expected_text"}, "assert_checked": {"selector"},
    "assert_selected": {"selector", "expected"}, "assert_enabled": {"selector"},
    "assert_disabled": {"selector"},
}


def _validate_action_parameters(exportable: ExportableTestCase) -> None:
    for saved in exportable.plans:
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


def _code_for(exportable: ExportableTestCase, language: str) -> str:
    _validate_action_parameters(exportable)
    handlers_index = {"python": 0, "typescript": 1, "csharp": 2}.get(language)
    if handlers_index is None:
        raise TestPlanExportError("Choose Python, TypeScript, or C# for source export.")
    lines: list[str]
    case = exportable.test_case
    if language == "python":
        lines = ["import pytest", "from playwright.sync_api import Page, expect", "", "", f"def {_python_function_name(case.name)}(page: Page) -> None:"]
    elif language == "typescript":
        lines = ["import { test, expect } from '@playwright/test';", "", "", f"test({_js_string(_ts_test_name(case.name))}, async ({{ page }}) => {{"]
    else:
        class_name = _csharp_identifier(case.name) + "Tests"
        method_name = _csharp_identifier(case.name) + "Test"
        lines = ["using System.Threading.Tasks;", "using Microsoft.Playwright;", "using Microsoft.Playwright.NUnit;", "using NUnit.Framework;", "", "[TestFixture]", f"public class {class_name} : PageTest", "{"]
        lines.extend(["    [Test]", f"    public async Task {method_name}()", "    {"])
    indent = "    " if language != "python" else "    "
    if language == "csharp":
        indent = "        "
    action_index = 0
    active_segment = None
    for saved in exportable.plans:
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
        for step in saved.version.qa_test_plan.steps:
            action_index += 1
            handler = ACTION_EXPORT_HANDLERS[step.action][handlers_index]
            emitted = handler(step.action, step.parameters, action_index)
            lines.extend(indent + line for line in emitted)
    if language == "typescript":
        lines.extend(["});", ""])
    elif language == "csharp":
        lines.extend(["    }", "}", ""])
    else:
        lines.append("")
    if language == "python":
        source = "\n".join(lines)
        try:
            compile(source, "<playwright-export>", "exec")
        except SyntaxError as error:
            raise TestPlanExportError("The generated Python source is invalid.") from error
        return source
    return "\n".join(lines)


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
    prefix = (test_case.public_id or "TC").casefold()
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
    if normalized.is_absolute() or ".." in normalized.parts or "" in normalized.parts:
        raise TestPlanExportError("Generated archive path is not safe.")
    info = zipfile.ZipInfo(str(normalized), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def _project_readme(language: str) -> str:
    commands = {
        "python": "python -m pip install -r requirements.txt\npython -m playwright install chromium\npytest",
        "typescript": "npm install\nnpx playwright install chromium\nnpx playwright test",
        "csharp": "dotnet test csharp/AIQAAgent.Export.csproj",
    }
    return (
        "# AI QA Agent Playwright export\n\n"
        "These tests were generated deterministically from saved, versioned TestPlans. "
        "They do not call AI QA Agent or an LLM at runtime.\n\n"
        f"## {language.title()}\n\n"
        "Install the listed dependencies and browser on your own machine before running:\n\n"
        "```text\n" + commands[language] + "\n```\n\n"
        "Selectors are preserved as Playwright locator strings from the saved plan. "
        "Review test data and target URLs before execution.\n"
    )


def _project_files(language: str, exports: list[ExportableTestCase], *, suite_name: str | None = None) -> dict[str, bytes]:
    used: set[str] = set()
    portable_used: set[str] = set()
    files: dict[str, bytes] = {"README.md": _project_readme(language).encode("utf-8")}
    manifest_cases = []
    extension = {"python": ".py", "typescript": ".spec.ts", "csharp": "Tests.cs"}[language]
    for exportable in exports:
        case = exportable.test_case
        base = safe_basename(case, used)
        file_name = base + extension if language != "csharp" else _csharp_identifier(case.name) + extension
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
            root = "python/"
        elif language == "typescript":
            content = typescript_source(exportable)
            root = "typescript/"
        else:
            content = csharp_source(exportable)
            root = "csharp/"
        files[root + file_name] = content.encode("utf-8")
        portable_name = safe_basename(case, portable_used) + ".testplan.json"
        files["portable/" + portable_name] = portable_json(exportable).encode("utf-8")
        manifest_cases.append({
            "public_id": case.public_id,
            "name": case.name,
            "source_file": root + file_name,
            "portable_file": "portable/" + portable_name,
            "plan_versions": [
                {
                    "step_order": item.step_order,
                    "version": item.version.version,
                    "provenance": item.version.origin.value if item.version.origin else None,
                }
                for item in exportable.plans
            ],
        })
    if language == "python":
        files["requirements.txt"] = b"pytest>=8\nplaywright>=1.40\npytest-playwright>=0.4\n"
    elif language == "typescript":
        files["package.json"] = (json.dumps({
            "private": True,
            "scripts": {"test": "playwright test"},
            "devDependencies": {"@playwright/test": "^1.40.0", "typescript": "^5.0.0"},
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        files["playwright.config.ts"] = b"import { defineConfig } from '@playwright/test';\nexport default defineConfig({ testDir: './typescript' });\n"
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
    manifest = {
        "format_version": EXPORT_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
        case = exportable.test_case
        filename = safe_basename(case, used) + ".testplan.json"
        files["portable/" + filename] = portable_json(exportable).encode("utf-8")
        cases.append({"public_id": case.public_id, "name": case.name, "file": "portable/" + filename})
    files["export-manifest.json"] = (json.dumps({
        "format_version": EXPORT_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language": "portable-json",
        "suite": suite_name,
        "test_cases": cases,
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for name in sorted(files):
            archive.writestr(_zip_info(name), files[name])
    return output.getvalue()


class TestPlanExportService:
    """Load exact current saved versions and render offline exports."""

    def __init__(self, test_cases: TestCaseRepository, plan_store: PlanStore) -> None:
        self._test_cases = test_cases
        self._plan_store = plan_store

    def get(self, test_case_id) -> ExportableTestCase:
        test_case = self._test_cases.get(test_case_id)
        if test_case is None:
            raise TestPlanExportError("TestCase not found.")
        return _plans_for_case(test_case, self._plan_store)

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
        return generator(exportable), basename + extension

    def bulk_zip(self, test_case_ids: list, target: str) -> tuple[bytes, str]:
        exports = self.get_many(test_case_ids)
        if target == "portable":
            return portable_zip(exports), "ai-qa-portable-testplans.zip"
        language = {"python": "python", "typescript": "typescript", "csharp": "csharp"}.get(target)
        if language is None:
            raise TestPlanExportError("Choose Portable JSON, Python, TypeScript, or C# export.")
        return project_zip(language, exports), f"ai-qa-{language}-project.zip"

