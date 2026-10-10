"""Policies and bounded metadata for the existing reliability/progress records."""
from datetime import datetime, timedelta, timezone
from contextvars import ContextVar
from contextlib import contextmanager
from enum import Enum
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qa_agent.redaction import redact_diagnostic


class DiagnosticLevel(str, Enum):
    OFF = 'OFF'
    NORMAL = 'NORMAL'
    DEBUG = 'DEBUG'
    TRACE = 'TRACE'


LEVEL_ORDER = {level: index for index, level in enumerate(DiagnosticLevel)}
LEVEL_DESCRIPTIONS = {
    DiagnosticLevel.OFF: 'Essential errors, audit events and outcomes only; optional collection is disabled.',
    DiagnosticLevel.NORMAL: 'Operation lifecycle, providers, step status, durations, failure codes and Quality Gates.',
    DiagnosticLevel.DEBUG: 'NORMAL plus routing, safe rejected actions, grounding, coverage and Discovery metadata.',
    DiagnosticLevel.TRACE: 'DEBUG plus internal stage timings and request/response structure; no payload contents.',
}
REASON_CODES = frozenset({
    'VALUE_NOT_DECLARED_IN_REQUIREMENT', 'FILL_ASSERT_VALUE_MISMATCH', 'MISSING_PRECEDING_FILL',
    'UNSUPPORTED_INPUT_CONTROL', 'AMBIGUOUS_INPUT_BINDING', 'CONFLICTING_REQUIREMENT_VALUES',
    'PROHIBITED_INPUT_VALUE', 'ASSERTION_TARGET_MISMATCH', 'EXPECTED_RESULT_NOT_COVERED',
    'UNSUPPORTED_EXPECTED_RESULT', 'INVALID_STRUCTURED_OUTPUT', 'OUTPUT_TRUNCATED',
    'PROVIDER_UNAVAILABLE', 'PROVIDER_RATE_LIMITED', 'BROWSER_PRECONDITION_FAILED',
    'SELECTOR_NOT_FOUND', 'BROWSER_ACTION_FAILED', 'UNKNOWN',
})
MAX_EVENTS = 128
MAX_OPTIONAL_BYTES = 32768
MAX_RETAINED_OPERATIONS = 100
RETENTION_DAYS = 7
_LEVEL = ContextVar('qa_diagnostic_level', default=None)


def current_diagnostic_level():
    return _LEVEL.get()


@contextmanager
def diagnostic_scope(level):
    try:
        value = DiagnosticLevel(level) if level is not None else _LEVEL.get()
    except (TypeError, ValueError):
        value = DiagnosticLevel.NORMAL
    token = _LEVEL.set(value)
    try:
        yield _LEVEL.get()
    finally:
        _LEVEL.reset(token)


class DiagnosticEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    timestamp: datetime
    kind: Literal['OPERATION_STARTED', 'PROVIDER_REQUEST_STARTED', 'PROVIDER_RESPONSE', 'PROVIDER_SKIPPED',
                  'REQUEST_STRUCTURE', 'QUALITY_GATE', 'DISCOVERY', 'COVERAGE', 'CANDIDATE_REJECTED']
    attempt_index: int | None = Field(default=None, ge=1, le=3)
    metadata: dict = Field(default_factory=dict)

    @field_validator('metadata')
    @classmethod
    def safe_metadata(cls, value):
        return structural_metadata(value)


def enabled(level, minimum=DiagnosticLevel.NORMAL):
    return LEVEL_ORDER[DiagnosticLevel(level)] >= LEVEL_ORDER[minimum]


def safe_label(value):
    return redact_diagnostic(value)[:180] if isinstance(value, str) else None


def record_diagnostic(kind, *, minimum=DiagnosticLevel.NORMAL, **metadata):
    try:
        from qa_agent.reliability import current_reliability_operation
        operation = current_reliability_operation()
        if operation is not None:
            operation.diagnostic(kind, minimum=minimum, **metadata)
    except Exception:
        pass


def structural_metadata(data):
    """Allowlist types and values; never accept prompts, responses, DOM or messages."""
    result = {}
    for key in ('request_sent', 'available', 'truncated', 'page_open', 'at_plan_url'):
        if type(data.get(key)) is bool:
            result[key] = data[key]
    for key in ('duration_ms', 'input_tokens', 'output_tokens', 'action_count', 'control_count',
                'form_count', 'warning_count', 'request_chars', 'snapshot_chars', 'response_chars'):
        if type(data.get(key)) is int and 0 <= data[key] <= 10**12:
            result[key] = data[key]
    if type(data.get('http_status')) is int and 100 <= data['http_status'] <= 599:
        result['http_status'] = data['http_status']
    for key in ('provider', 'model'):
        if isinstance(data.get(key), str):
            result[key] = safe_label(data[key])
    choices = {
        'finish_reason': {'stop', 'length', 'tool_calls', 'content_filter', 'STOP', 'MAX_TOKENS', 'completed', 'incomplete'},
        'provider_error_code': {'rate_limit_exceeded', 'rate_limit', 'model_not_found', 'invalid_request',
                                'invalid_json', 'invalid_schema_response', 'missing_structured_response', 'output_token_limit'},
        'status': {'SUCCESS', 'FAILED', 'SKIPPED', 'PASSED', 'NOT_RUN', 'ACCEPTED', 'REJECTED'},
        'stage': {'schema_and_actions', 'locator_identity', 'assertion_grounding', 'expected_result_coverage',
                  'step_boundaries', 'discovery', 'generation', 'validation', 'execution'},
        'discovery_status': {'SUCCESS', 'PARTIAL', 'FAILED'},
        'coverage': {'COVERED', 'PARTIALLY_COVERED', 'NOT_COVERED', 'UNKNOWN', 'NO_VERIFICATION_REQUIRED'},
        'reason_code': REASON_CODES,
    }
    for key, allowed in choices.items():
        if isinstance(data.get(key), str) and data[key] in allowed:
            result[key] = data[key]
    return result


def failure_reasons(error):
    try:
        return _failure_reasons(error)
    except Exception:
        return [dict(reason_code='UNKNOWN', root_cause_known=False)]


def _failure_reasons(error):
    """Specific codes require typed/local evidence; exception strings are never parsed."""
    from qa_agent.test_plan_validation import PlanValidationError
    from qa_agent.llm.errors import RetryableLLMError, provider_http_failure
    if isinstance(error, PlanValidationError):
        from qa_agent.candidate_diagnostics import safe_validation_issues
        fallback = {'EXPECTED_RESULT_NOT_COVERED': 'EXPECTED_RESULT_NOT_COVERED',
                    'INVALID_PLAN_STRUCTURE': 'INVALID_STRUCTURED_OUTPUT'}
        return [dict(**safe_validation_issues([issue])[0],
                     reason_code=issue.reason_code or fallback.get(issue.code, 'UNKNOWN'),
                     root_cause_known=bool(issue.reason_code or issue.code in fallback))
                for issue in error.issues[:8]]
    if isinstance(error, RetryableLLMError):
        category = error.category
        if category is None and type(error.http_status) is int and 100 <= error.http_status <= 599:
            category = provider_http_failure('Provider', error.http_status).category
        reason = ('OUTPUT_TRUNCATED' if error.provider_error_code == 'output_token_limit' else
                  {'INVALID_RESPONSE': 'INVALID_STRUCTURED_OUTPUT', 'RATE_LIMIT': 'PROVIDER_RATE_LIMITED',
                   'PROVIDER_UNAVAILABLE': 'PROVIDER_UNAVAILABLE'}.get(category, 'UNKNOWN'))
        return [dict(reason_code=reason, root_cause_known=reason != 'UNKNOWN')]
    return [dict(reason_code='UNKNOWN', root_cause_known=False)]


def bound_optional(record):
    try:
        return _bound_optional(record)
    except Exception:
        record = record.model_copy(deep=True)
        clear_optional(record)
        return record


def _bound_optional(record):
    """Leave mandatory audit fields intact; cap only new optional diagnostics."""
    if record.diagnostic_level is None:  # Historical records retain their original shape/data.
        return record
    record = record.model_copy(deep=True)
    record.diagnostic_events_omitted += max(0, len(record.diagnostic_events) - MAX_EVENTS)
    record.diagnostic_events = record.diagnostic_events[-MAX_EVENTS:]
    def size():
        return len(json.dumps({
            'events': [event.model_dump(mode='json') for event in record.diagnostic_events],
            'candidates': [attempt.candidate_diagnostics for attempt in record.attempts],
            'providers': [attempt.provider_metadata for attempt in record.attempts],
            'routing': record.effective_provider_order,
        }, ensure_ascii=True).encode())
    while size() > MAX_OPTIONAL_BYTES and record.diagnostic_events:
        record.diagnostic_events.pop(0)
        record.diagnostic_events_omitted += 1
    if size() > MAX_OPTIONAL_BYTES:
        clear_optional(record)
    return record


def expired(record, position, now):
    started = record.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (record.diagnostic_level is not None and record.outcome != 'RUNNING'
            and (position >= MAX_RETAINED_OPERATIONS or started < now - timedelta(days=RETENTION_DAYS)))


def clear_optional(record):
    record.diagnostic_events = []
    record.effective_provider_order = []
    record.diagnostics_expired = True
    for attempt in record.attempts:
        attempt.candidate_diagnostics = None
        attempt.provider_metadata = {}


def discovery_metadata(discovery):
    try:
        from qa_agent.generation_context import observed_controls
        metadata = dict(discovery_status=discovery.status.value,
                        control_count=len(observed_controls(discovery)), warning_count=len(discovery.warnings))
        forms = discovery.snapshot.get('forms')
        if isinstance(forms, list):
            metadata['form_count'] = len(forms)
        return metadata
    except Exception:
        return {}
