"""Bounded value-free summaries of rejected candidates, never executable plans."""
import hashlib
import json
import re

from qa_agent.models import QATestPlan, QATestStep
from qa_agent.test_plan_validation import PlanValidationError, validate_executable_plan

MAX_ACTIONS = 64
MAX_CLAUSES = 32


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


def candidate_action_summary(value):
    """Safe even when a provider JSON object cannot become a domain plan."""
    data = value.model_dump(mode='python') if isinstance(value, QATestPlan) else value
    raw = data.get('steps', []) if isinstance(data, dict) else []
    raw = raw if isinstance(raw, list) else []
    actions = []
    for index, item in enumerate(raw[:MAX_ACTIONS]):
        item = item if isinstance(item, dict) else {}
        action = item.get('action')
        action = action if isinstance(action, str) and action in QATestStep.ACTION_PARAMETER_FIELDS else 'UNSUPPORTED'
        row = {'index': index, 'action': action, 'assertion': action.startswith('assert_')}
        params = item.get('parameters')
        selector = params.get('selector') if isinstance(params, dict) else None
        if isinstance(selector, str):
            row['selector_hash'] = _hash(selector)
        actions.append(row)
    return {'actions': actions, 'actions_omitted': max(0, len(raw) - MAX_ACTIONS)}


def safe_action_summary(summary):
    """Revalidate even adapter-supplied summaries before persistence."""
    rows = summary.get('actions') if isinstance(summary, dict) else None
    rows = rows if isinstance(rows, list) else []
    safe = []
    for index, row in enumerate(rows[:MAX_ACTIONS]):
        row = row if isinstance(row, dict) else {}
        action = row.get('action')
        action = action if isinstance(action, str) and action in QATestStep.ACTION_PARAMETER_FIELDS else 'UNSUPPORTED'
        item = {'index': index, 'action': action, 'assertion': action.startswith('assert_')}
        identity = row.get('selector_hash')
        if isinstance(identity, str) and re.fullmatch(r'[a-f0-9]{16}', identity):
            item['selector_hash'] = identity
        safe.append(item)
    omitted = summary.get('actions_omitted', 0) if isinstance(summary, dict) else 0
    return {'actions': safe, 'actions_omitted': min(1000000, max(0, omitted)) if type(omitted) is int else 0}


def safe_validation_issues(issues):
    safe = []
    from qa_agent.test_plan_validation import PlanValidationIssue
    for issue in issues[:8] if isinstance(issues, (list, tuple)) else []:
        if isinstance(issue, PlanValidationIssue):
            code, path = issue.code, issue.path
        elif isinstance(issue, dict):
            try:
                parsed = PlanValidationIssue(code=issue.get('code'), path='plan', message='')
            except ValueError:
                continue
            code, path = parsed.code, issue.get('path')
        else:
            continue
        path = path if isinstance(path, str) and re.fullmatch(r'(?:plan|url|steps(?:\[\d{1,6}\])?(?:\.action|\.parameters(?:\.(?:selector|value|expected|expected_text|url|option_label))?)?)', path) else 'plan'
        safe.append({'code': code, 'path': path})
    return safe


def provider_schema_summary(value):
    summary = candidate_action_summary(value)
    try:
        validate_executable_plan(value)
    except PlanValidationError as error:
        summary['validation_issues'] = safe_validation_issues(error.issues)
    return summary


def attach_provider_candidate_summary(failure, raw):
    """Shared parser diagnostics for all adapters, without keeping raw output."""
    if not isinstance(raw, str) or not raw.strip():
        failure.provider_error_code = 'missing_structured_response'
        return
    try:
        data = json.loads(raw)
    except ValueError:
        failure.provider_error_code = 'invalid_json'
        return
    failure.provider_error_code = 'invalid_schema_response'
    from qa_agent.reliability import current_reliability_operation
    from qa_agent.diagnostic_mode import DiagnosticLevel, enabled
    operation = current_reliability_operation()
    if operation:
        operation.capture_recovery(data)
    if operation and not enabled(operation.record.settings.diagnostic_level, DiagnosticLevel.DEBUG):
        return
    failure.rejected_actions = provider_schema_summary(data)


def rejected_candidate_diagnostics(value, step, discovery, error):
    from qa_agent.expected_result_coverage import _expectations, _expectation_kind, _result_clauses, assertion_subject_matches, expected_result_coverage
    from qa_agent.reliability import classify_failure, gate_for_category
    summary = candidate_action_summary(value)
    issues = safe_validation_issues(error.issues) if isinstance(error, PlanValidationError) else []
    try:
        candidate = validate_executable_plan(value)
    except PlanValidationError:
        candidate = None
    coverage = expected_result_coverage(step, candidate, discovery=discovery)
    matches = (assertion_subject_matches(step, candidate, discovery=discovery)
               if candidate is not None and coverage.verification_required else {})
    covered = {i for indexes in matches.values() for i in indexes}
    clauses = _expectations(step) if coverage.verification_required else ()
    clause_rows = ([{'index': i, 'kind': clause.kind, 'clause_hash': _hash(clause.clause), 'covered': i in covered}
                    for i, clause in enumerate(clauses[:MAX_CLAUSES])] if clauses is not None else
                   [{'index': i, 'kind': _expectation_kind(clause) or 'UNKNOWN',
                     'clause_hash': _hash(clause), 'covered': False}
                    for i, clause in enumerate(_result_clauses(step.expected)[:MAX_CLAUSES])])
    clause_count = len(clauses) if clauses is not None else len(_result_clauses(step.expected))
    return {
        **summary,
        'failed_gate': gate_for_category(classify_failure(error)),
        'validation_issues': issues,
        'coverage': coverage.status.value,
        'requirement_clauses': clause_rows,
        'clauses_omitted': max(0, clause_count - MAX_CLAUSES),
        'uninterpreted_requirement': clauses is None,
        'requirement_hash': _hash(step.expected),
    }
