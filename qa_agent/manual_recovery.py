"""Private, process-local recovery drafts; never audit/export payloads or plans."""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import re
from threading import RLock

from qa_agent.automation_editor import AutomationEditorError, plan_from_form
from qa_agent.generation_context import StepGenerationContext, observed_controls
from qa_agent.models import DiscoveryStatus, QATestPlan
from qa_agent.redaction import redact_secrets
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.test_plan_validation import PlanValidationError
from qa_agent.testplan_export import _is_unsafe_export_value

MAX_RECOVERY_BYTES = 64 * 1024
MAX_RECOVERY_ACTIONS = 64
MAX_RECOVERY_ENTRIES = 64
RECOVERY_TTL = timedelta(hours=1)
_CREDENTIAL_TEXT = re.compile(r'(?i)\b(?:password|api[_-]?key|access[_-]?token|token|secret|authorization)\s*[=:]|\bBearer\s+\S+|\b(?:sk|gsk|or)[-_][A-Za-z0-9_-]{8,}|\bAIza[A-Za-z0-9_-]{8,}')


def credential_bearing(value):
    return redact_secrets(value) != value or bool(_CREDENTIAL_TEXT.search(value))


class RecoveryUnavailable(ValueError):
    pass


def _private_copy(value):
    if isinstance(value, str):
        return '' if credential_bearing(value) or _is_unsafe_export_value(value) else value
    if isinstance(value, dict):
        return {key: _private_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_private_copy(item) for item in value]
    return value


@dataclass(repr=False)
class RecoveryEntry:
    operation_id: object
    test_case_id: object
    step: object = field(repr=False)
    requirement_context: str = field(repr=False)
    discovery: object = field(default=None, repr=False)
    step_context: object = field(default=None, repr=False)
    candidate: dict | None = field(default=None, repr=False)
    draft: dict | None = field(default=None, repr=False)
    revision: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ManualRecoveryStore:
    """Opt-in local storage, at most 4 MiB, with no files or database writes."""
    def __init__(self, *, clock=None):
        self._entries = {}
        self._lock = RLock()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _prune(self):
        now = self._clock()
        self._entries = {key: item for key, item in self._entries.items()
                         if now - item.created_at < RECOVERY_TTL}
        while len(self._entries) > MAX_RECOVERY_ENTRIES:
            del self._entries[min(self._entries, key=lambda key: self._entries[key].created_at)]

    def _put(self, entry):
        if any(isinstance(value, dict) and isinstance(value.get('steps'), list) and len(value['steps']) > MAX_RECOVERY_ACTIONS
               for value in (entry.candidate, entry.draft)):
            raise RecoveryUnavailable('Recovery drafts support at most 64 actions.')
        data = dict(step=entry.step.model_dump(mode='json'), requirement=entry.requirement_context,
                    discovery=entry.discovery.model_dump(mode='json') if entry.discovery else None,
                    candidate=entry.candidate, draft=entry.draft,
                    context=str(entry.step_context))
        if len(json.dumps(data, ensure_ascii=True).encode()) > MAX_RECOVERY_BYTES:
            raise RecoveryUnavailable('Recovery data exceeds the safe temporary storage limit.')
        self._entries[entry.operation_id] = deepcopy(entry)
        self._prune()

    def prepare(self, record, step, discovery, requirement_context, context):
        if record.test_case_id is None:
            return
        # Only local Web (or an explicitly supplied store) authorizes retention.
        entry = RecoveryEntry(record.id, record.test_case_id, deepcopy(step), requirement_context or '',
                              discovery.model_copy(deep=True), deepcopy(context), created_at=self._clock())
        # Discovery never retains configured secrets or credential-bearing URLs.
        entry.discovery = type(discovery).model_validate(_private_copy(discovery.model_dump(mode='json')))
        with self._lock:
            self._prune()
            self._put(entry)

    def capture(self, record, value):
        with self._lock:
            self._prune()
            entry = self._entries.get(record.id)
            if not entry or entry.test_case_id != record.test_case_id or entry.step.id != record.test_step_id:
                return
            data = value.model_dump(mode='json') if isinstance(value, QATestPlan) else value
            if not isinstance(data, dict) or not isinstance(data.get('steps'), list):
                return
            if len(data['steps']) > MAX_RECOVERY_ACTIONS:
                return
            data = _private_copy(deepcopy(data))
            controls = observed_controls(entry.discovery)
            for action in data['steps']:
                if not isinstance(action, dict) or not isinstance(action.get('parameters'), dict):
                    continue
                params = action['parameters']
                selector = params.get('selector')
                control = controls.get(selector) if isinstance(selector, str) else None
                if control and (control.input_type.casefold() == 'password' or
                                re.search(r'password|secret|token|api.?key', ' '.join((control.name, control.label, control.accessible_name)), re.I)):
                    for key in ('value', 'expected', 'expected_text'):
                        if key in params:
                            params[key] = ''
            updated = deepcopy(entry)
            updated.candidate = data
            updated.draft = None
            self._put(updated)

    def get(self, operation_id, case_id, step_id):
        with self._lock:
            self._prune()
            entry = self._entries.get(operation_id)
            if not entry or entry.test_case_id != case_id or entry.step.id != step_id:
                return None
            return deepcopy(entry)

    def empty(self, operation_id, case, step):
        context = recovery_step_context(case, step)
        entry = RecoveryEntry(operation_id, case.id, deepcopy(step), case.description,
                              step_context=context, created_at=self._clock())
        with self._lock:
            self._put(entry)
        return deepcopy(entry)

    def update(self, entry, revision, *, draft=None, discovery=None):
        with self._lock:
            current = self.get(entry.operation_id, entry.test_case_id, entry.step.id)
            if current is None or current.revision != revision:
                raise RecoveryUnavailable('This recovery draft changed or expired. Reload before editing.')
            if not current.requirement_context:
                current.requirement_context = entry.requirement_context
            if current.step_context is None:
                current.step_context = deepcopy(entry.step_context)
            if draft is not None:
                current.draft = _private_copy(draft)
            if discovery is not None:
                current.discovery = type(discovery).model_validate(_private_copy(discovery.model_dump(mode='json')))
                current.step_context = StepGenerationContext(current.step_context.segment_order,
                    previous_steps=current.step_context.previous_steps, remaining_steps=current.step_context.remaining_steps) if current.step_context else None
            current.revision += 1
            self._put(current)
            return deepcopy(current)

    def discard(self, operation_id):
        with self._lock:
            self._entries.pop(operation_id, None)

    def complete(self, operation_id):
        """Remove candidate/edit payloads; retain bounded gate context for later edits."""
        with self._lock:
            entry = self._entries.get(operation_id)
            if entry:
                entry = deepcopy(entry)
                entry.candidate = entry.draft = None
                entry.revision += 1
                self._put(entry)

    def commit(self, entry, save):
        """Serialize draft revision checks and version creation without executing it."""
        with self._lock:
            current = self.get(entry.operation_id, entry.test_case_id, entry.step.id)
            if current is None or current.revision != entry.revision:
                raise RecoveryUnavailable('This recovery draft changed during validation. Reload before saving.')
            save()
            self.complete(entry.operation_id)


def recovery_step_context(case, step):
    segment = next(item for item in case.segments if any(s.id == step.id for s in item.steps))
    return StepGenerationContext(segment.order,
        previous_steps=tuple(s for s in segment.steps if s.order < step.order),
        remaining_steps=tuple(s for s in segment.steps if s.order > step.order))


def recovery_validation(entry, values):
    """Reuse the form parser and the exact generation gate sequence, locally."""
    plan = plan_from_form(values)
    if not entry.step.description.strip() or not entry.step.expected.strip():
        raise AutomationEditorError({'actions': 'Needs Attention: original TestStep requirements are incomplete.'})
    if entry.discovery is None or entry.discovery.status == DiscoveryStatus.FAILED:
        raise AutomationEditorError({'actions': 'Needs Attention: deterministic Browser Discovery is required before validation.'})
    if plan.url != entry.discovery.url:
        raise AutomationEditorError({'plan_url': 'The plan URL must match the captured Browser Discovery URL.'})
    controls = observed_controls(entry.discovery)
    for index, action in enumerate(plan.steps):
        control = controls.get(action.parameters.get('selector'))
        sensitive = control and (control.input_type.casefold() == 'password' or re.search(r'password|secret|token|api.?key', ' '.join((control.name, control.label, control.accessible_name)), re.I))
        for name, value in action.parameters.items():
            if isinstance(value, str) and (credential_bearing(value) or sensitive and name in {'value', 'expected', 'expected_text'}):
                raise AutomationEditorError({f'action.{index}.param.{name}': 'Credential-bearing values cannot be retained in a recovery plan.'})
    try:
        return LLMTestPlanGenerator._validate_generated_plan(plan, entry.discovery, entry.step,
            requirement_context=entry.requirement_context, step_context=entry.step_context)
    except PlanValidationError as error:
        errors = {}
        for issue in error.issues:
            match = re.fullmatch(r'steps\[(\d+)\](?:\.parameters(?:\.([a-z_]+))?)?', issue.path)
            key = f'action.{match[1]}.param.{match[2]}' if match and match[2] else f'action.{match[1]}.type' if match else 'actions'
            errors[key] = f'{issue.code}' + (f' / {issue.reason_code}' if issue.reason_code else '') + ': ' + issue.message
        raise AutomationEditorError(errors) from error


def safe_recovery_form(entry, values):
    safe = _private_copy(values)
    controls = observed_controls(entry.discovery) if entry.discovery else {}
    for key, value in values.items():
        match = re.fullmatch(r'action\.(\d+)\.param\.selector', key)
        if not match:
            continue
        control = controls.get(value)
        if control and (control.input_type.casefold() == 'password' or re.search(r'password|secret|token|api.?key', ' '.join((control.name, control.label, control.accessible_name)), re.I)):
            for field in ('value', 'expected', 'expected_text'):
                name = f'action.{match[1]}.param.{field}'
                if name in safe:
                    safe[name] = ''
    return safe
