"""Allowlisted projections of existing records; no payloads or evidence files."""
from datetime import datetime, timezone
import hashlib
import io
import json
import re
from uuid import UUID
import zipfile

from qa_agent.candidate_diagnostics import safe_action_summary, safe_validation_issues
from qa_agent.diagnostic_mode import REASON_CODES, structural_metadata, safe_label
from qa_agent.reliability import QUALITY_GATES, safe_reason

MAX_REPORT_BYTES = 256 * 1024
MAX_EXPORT_OPERATIONS = 100
MAX_EXECUTIONS = 256
MAX_CHRONOLOGY = 512
CATEGORIES = frozenset({
    'UNKNOWN_ERROR', 'CANCELLED', 'TIMEOUT', 'RATE_LIMIT', 'PROVIDER_UNAVAILABLE', 'AUTH_ERROR',
    'MODEL_NOT_FOUND', 'INVALID_REQUEST', 'SCHEMA_ERROR', 'INVALID_RESPONSE', 'PLAN_SCHEMA_INVALID',
    'UNSUPPORTED_ACTION', 'EXPECTED_RESULT_NOT_COVERED', 'ASSERTION_NOT_GROUNDED',
    'LOCATOR_IDENTITY_UNCERTAIN', 'INSUFFICIENT_TESTCASE_REQUIREMENTS', 'ATTEMPT_LIMIT',
    'TIME_LIMIT', 'NO_PROVIDER', 'NO_FALLBACK', 'INFRASTRUCTURE_ERROR', 'PRODUCT_FAILURE',
    'UNSAFE_REPAIR', 'STEP_BOUNDARY_VIOLATION', 'AUTOMATION_DRIFT', 'AUTOMATION_EXECUTION_ERROR',
    'AUTOMATION_GENERATION_ERROR', 'PASSED', 'FAILED', 'BLOCKED', 'INCONCLUSIVE', 'SETUP_FAILURE',
})


def category(value):
    return value if value in CATEGORIES else 'UNKNOWN_ERROR' if value else None


def timestamp(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def reason_rows(rows):
    result = []
    for row in rows[:8] if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        reason = row.get('reason_code')
        reason = reason if isinstance(reason, str) and reason in REASON_CODES else 'UNKNOWN'
        item = {'reason_code': reason, 'root_cause_known': reason != 'UNKNOWN' and row.get('root_cause_known') is True}
        safe = safe_validation_issues([row])
        if safe:
            item.update(safe[0])
        result.append(item)
    return result or [{'reason_code': 'UNKNOWN', 'root_cause_known': False}]


def candidate_metadata(data):
    if not isinstance(data, dict):
        return None
    result = safe_action_summary(data)
    result['validation_issues'] = safe_validation_issues(data.get('validation_issues'))
    for key in ('actions_omitted', 'clauses_omitted'):
        if type(data.get(key)) is int:
            result[key] = max(0, min(10**6, data[key]))
    for key in ('candidate_unavailable', 'coverage_evaluated', 'uninterpreted_requirement'):
        if type(data.get(key)) is bool:
            result[key] = data[key]
    result.update(structural_metadata({'coverage': data.get('coverage')}))
    if data.get('failed_gate') in {*QUALITY_GATES, 'repair_integrity', 'provider_response_schema'}:
        result['failed_gate'] = data['failed_gate']
    result['requirement_clauses'] = []
    for clause in data.get('requirement_clauses', [])[:32] if isinstance(data.get('requirement_clauses'), list) else []:
        if not isinstance(clause, dict):
            continue
        identity = clause.get('clause_hash')
        if isinstance(identity, str) and re.fullmatch('[a-f0-9]{16}', identity):
            # Kind is a local classifier token, but do not trust arbitrary old rows.
            known_kinds = {'UNKNOWN', 'input_value', 'form_visible', 'page_loaded', 'page_visible',
                           'text', 'error_visible', 'error_hidden', 'success_visible', 'success_hidden',
                           'no_error', 'required_field_errors', 'submission_blocked', 'checked',
                           'unchecked', 'selected', 'enabled', 'disabled', 'url', 'title'}
            known_kinds.update({'visible', 'hidden', 'loaded', 'redirect', 'result'})
            result['requirement_clauses'].append({'index': len(result['requirement_clauses']),
                'kind': clause.get('kind') if clause.get('kind') in known_kinds else 'UNKNOWN',
                'clause_hash': identity, 'covered': clause.get('covered') is True})
    return result


def operation_data(record):
    attempts = []
    for attempt in record.attempts[:3]:
        gates = {key: value for key, value in attempt.quality_gates.items()
                 if key in {*QUALITY_GATES, 'repair_integrity'} and value in {'PASSED', 'FAILED', 'NOT_RUN'}}
        attempts.append({
            'attempt_index': attempt.index, 'action': attempt.action,
            'provider': safe_label(attempt.provider), 'model': safe_label(attempt.model),
            'provider_invoked': True, 'request_sent': attempt.request_sent, 'provider_status': attempt.provider_status,
            'plan_status': attempt.status, 'started_at': timestamp(attempt.started_at),
            'finished_at': timestamp(attempt.finished_at), 'duration_ms': attempt.duration_ms,
            'input_tokens': attempt.input_tokens, 'output_tokens': attempt.output_tokens,
            'error_category': category(attempt.error_category), 'quality_gates': gates,
            'failure_reasons': reason_rows(attempt.failure_reasons) if attempt.status == 'REJECTED' else [],
            'provider_metadata': structural_metadata(attempt.provider_metadata),
            'candidate_metadata': candidate_metadata(attempt.candidate_diagnostics),
            'candidate_metadata_availability': ('AVAILABLE' if attempt.candidate_diagnostics else
                'UNAVAILABLE_OR_EXPIRED' if record.diagnostics_expired or record.diagnostic_level is None else
                'NOT_COLLECTED_AT_THIS_LEVEL' if record.diagnostic_level in {'OFF', 'NORMAL'} else 'UNAVAILABLE'),
        })
    decisions = [{'timestamp': timestamp(item.timestamp), 'attempt_index': item.attempt_index,
                  'action': item.action if item.action in {'PROVIDER_RETRY', 'PROVIDER_FALLBACK', 'STOP', 'TARGETED_REPAIR', 'HUMAN_REVIEW_REQUIRED'} else 'UNKNOWN',
                  'category': category(item.category), 'reason': safe_reason(category(item.category)) if item.category else 'Generation acceptance does not establish Browser Validation or product PASS.'}
                 for item in record.decisions[:32]]
    kinds = {'OPERATION_STARTED', 'PROVIDER_REQUEST_STARTED', 'PROVIDER_RESPONSE', 'PROVIDER_SKIPPED',
             'REQUEST_STRUCTURE', 'QUALITY_GATE', 'DISCOVERY', 'COVERAGE', 'CANDIDATE_REJECTED'}
    events = [{'timestamp': timestamp(event.timestamp), 'kind': event.kind,
               'attempt_index': event.attempt_index, 'metadata': structural_metadata(event.metadata)}
              for event in record.diagnostic_events[-128:] if event.kind in kinds]
    return {
        'operation_id': str(record.id), 'test_case_id': str(record.test_case_id) if record.test_case_id else None,
        'test_step_id': str(record.test_step_id), 'candidate_version_id': str(record.candidate_version_id) if record.candidate_version_id else None,
        'settings': record.settings.model_dump(mode='json'), 'effective_level': record.diagnostic_level or 'UNKNOWN_HISTORICAL',
        'outcome': record.outcome, 'final_category': category(record.final_category),
        'started_at': timestamp(record.started_at), 'finished_at': timestamp(record.finished_at),
        'elapsed_ms': record.elapsed_ms, 'attempts': attempts, 'decisions': decisions,
        'effective_provider_order': [dict(priority=index + 1, **structural_metadata(row))
                                     for index, row in enumerate(record.effective_provider_order[:32]) if isinstance(row, dict)],
        'events': events, 'events_omitted': record.diagnostic_events_omitted,
        'optional_metadata_unavailable': record.diagnostics_expired or not bool(events or record.effective_provider_order or any(a.candidate_diagnostics for a in record.attempts)),
        'storage_fallback': record.diagnostic_storage_fallback,
    }


def execution_data(reference):
    diagnostic = reference.action_failure
    action_failure = None
    if diagnostic is not None:
        identity = diagnostic.selector_identity
        action_failure = {
            'action_index': diagnostic.action_index, 'action': diagnostic.action,
            'selector_hash': (identity.removeprefix('selector-sha256:') if identity and identity.startswith('selector-sha256:')
                              else hashlib.sha256(identity.encode()).hexdigest()[:16] if identity else None),
            'exception_category': diagnostic.exception_category, 'timed_out': diagnostic.timed_out,
            'page_open': diagnostic.page_open, 'at_plan_url': diagnostic.at_plan_url,
            'target_count': diagnostic.target_count, 'target_visible': diagnostic.target_visible,
            'target_enabled': diagnostic.target_enabled, 'reason_code': diagnostic.reason_code,
            'root_cause_known': diagnostic.root_cause_known,
        }
    return dict(execution_id=str(reference.execution_id), test_step_id=str(reference.test_step_id),
                plan_version_id=str(reference.test_plan_version_id), status=reference.status.value,
                classification=category(reference.classification), started_at=timestamp(reference.started_at),
                finished_at=timestamp(reference.finished_at), action_failure=action_failure,
                duration_ms=max(0, int((reference.finished_at - reference.started_at).total_seconds() * 1000)) if reference.finished_at else None,
                evidence_count=len(reference.evidence),
                evidence_availability='REFERENCED_NOT_EXPORTED' if reference.evidence else 'UNAVAILABLE')


def diagnostic_report(records, runs=(), *, run_id=None):
    """Runs are supplied as (record, relevant references), already scoped by caller."""
    records = list(records)
    data = {'schema_version': 'M2.3', 'exported_at': timestamp(datetime.now(timezone.utc)),
            'run_id': str(run_id) if run_id else None, 'operations': [operation_data(r) for r in records[:MAX_EXPORT_OPERATIONS]],
            'operations_omitted': max(0, len(records) - MAX_EXPORT_OPERATIONS), 'runs': [], 'runs_omitted': 0, 'chronology': [],
            'limitations': ['Raw prompts, responses, DOM, form values and evidence files are intentionally unavailable.',
                            'Missing metadata or evidence is not successful verification.',
                            'Provider SUCCESS, Plan ACCEPTED, Browser Validation PASSED and product PASS are separate outcomes.',
                            'Historical records may lack exact failure reasons, routing or stage timings. Unknown causes remain unknown.',
                            'Cross-run correlation uses saved plan-version and step IDs; unmatched or older Runs are not guessed.',
                            'Provider invocation alone does not prove a network request; request_sent is unknown unless an adapter marked it.'],
            'report_truncated': False}
    for run, references in list(runs)[:50]:
        references = list(references)
        data['runs'].append({
            'run_id': str(run.run_id), 'test_case_id': str(run.test_case_id), 'workflow': run.workflow_type.value,
            'trace_id': str(run.trace_id) if run.trace_id else None,
            'effective_diagnostic_level': run.diagnostic_level or 'UNKNOWN_HISTORICAL',
            'product_outcome': category(run.outcome), 'status': run.status.value,
            'browser_validation_status': (run.status.value if run.workflow_type.value == 'VALIDATION' else 'NOT_RECORDED_FOR_THIS_WORKFLOW'),
            'started_at': timestamp(run.started_at), 'finished_at': timestamp(run.finished_at), 'duration_ms': run.duration_ms,
            'setup_status': run.setup_status if run.setup_status in {'PASSED', 'FAILED', 'BLOCKED'} else 'UNKNOWN',
            'precondition_reason': 'BROWSER_PRECONDITION_FAILED' if run.setup_status in {'FAILED', 'BLOCKED'} else None,
            'executions': [execution_data(reference) for reference in references[:MAX_EXECUTIONS]],
            'executions_omitted': max(0, len(references) - MAX_EXECUTIONS),
        })
        for reference in data['runs'][-1]['executions']:
            reference['reliability_operation_ids'] = [op['operation_id'] for op in data['operations']
                if op['test_case_id'] == str(run.test_case_id) and op['test_step_id'] == reference['test_step_id']
                and op['candidate_version_id'] == reference['plan_version_id']]
    if not any(run['executions'] for run in data['runs']):
        data['limitations'].append('No correlated Browser Execution was recorded; Browser Validation and product PASS are unavailable.')
    if any(op['storage_fallback'] for op in data['operations']):
        data['limitations'].append('Diagnostics storage failed; some audit data used an in-memory fallback and may be unavailable after restart.')
    chronology = []
    for op in data['operations']:
        chronology.append(dict(timestamp=op['started_at'], kind='OPERATION_STARTED', operation_id=op['operation_id'], test_step_id=op['test_step_id']))
        for row in [*(event for event in op['events'] if event['kind'] != 'OPERATION_STARTED'), *op['decisions']]:
            chronology.append(dict(row, operation_id=op['operation_id'], test_step_id=op['test_step_id']))
        if op['finished_at']:
            chronology.append(dict(timestamp=op['finished_at'], kind='OPERATION_FINISHED', operation_id=op['operation_id'], outcome=op['outcome']))
    for run in data['runs']:
        for reference in run['executions']:
            chronology.append(dict(timestamp=reference['started_at'], kind='BROWSER_EXECUTION_STARTED', run_id=run['run_id'], execution_id=reference['execution_id'], test_step_id=reference['test_step_id']))
            if reference['finished_at']:
                chronology.append(dict(timestamp=reference['finished_at'], kind='BROWSER_EXECUTION_FINISHED', run_id=run['run_id'], execution_id=reference['execution_id'], status=reference['status'], classification=reference['classification']))
    data['chronology'] = sorted(chronology, key=lambda row: row['timestamp'])[-MAX_CHRONOLOGY:]
    data['chronology_omitted'] = max(0, len(chronology) - MAX_CHRONOLOGY)
    return data


def encode_report(report):
    report = json.loads(json.dumps(report, ensure_ascii=True))
    def encode():
        return json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False).encode('utf-8')
    while len(encode()) > MAX_REPORT_BYTES:
        report['report_truncated'] = True
        if report['chronology']:
            removed = max(1, len(report['chronology']) // 2)
            del report['chronology'][:removed]
            report['chronology_omitted'] += removed
        elif any(op['events'] or any(a['candidate_metadata'] for a in op['attempts']) for op in report['operations']):
            for op in report['operations']:
                op['events_omitted'] += len(op['events'])
                op['events'] = []
                for attempt in op['attempts']:
                    if attempt['candidate_metadata']:
                        attempt['candidate_metadata_availability'] = 'OMITTED_REPORT_SIZE_LIMIT'
                    attempt['candidate_metadata'] = None
        elif report['runs']:
            report['runs'].pop()
            report['runs_omitted'] += 1
        elif report['operations']:
            report['operations'].pop()
            report['operations_omitted'] += 1
        else:
            raise ValueError('Diagnostic report exceeds its bounded envelope.')
    return encode()


def diagnostic_download(report, identifier, format):
    identifier = str(UUID(str(identifier)))  # Caller-supplied names/paths are never used.
    data = encode_report(report)
    if format == 'json':
        return f'diagnostics-{identifier}.json', 'application/json', data
    if format != 'zip':
        raise ValueError('Unsupported diagnostic export format.')
    bounded = json.loads(data)
    summary = ['AI QA Agent diagnostics (M2.3)', 'No product PASS is inferred from generation.', '']
    for op in bounded['operations']:
        summary.append(f"Operation {op['operation_id']} / TestStep {op['test_step_id']}: {op['outcome']}")
        for attempt in op['attempts']:
            summary.append(f"  Attempt {attempt['attempt_index']}: {attempt['provider']} / {attempt['model']}; Provider {attempt['provider_status']}; Plan {attempt['plan_status']}; {attempt['duration_ms']} ms")
            summary.append('    Quality Gates: ' + ', '.join(f'{gate}={status}' for gate, status in attempt['quality_gates'].items()))
            summary.extend(f"    {reason['reason_code']} (exact cause known: {reason['root_cause_known']})" for reason in attempt['failure_reasons'])
    for run in bounded['runs']:
        summary.append(f"Run {run['run_id']} / {run['workflow']}: {run['product_outcome']}; Browser Validation {run['browser_validation_status']}")
        summary.extend(f"  Execution {reference['execution_id']} / TestStep {reference['test_step_id']}: {reference['status']}; {reference['classification']}; evidence {reference['evidence_availability']}" for reference in run['executions'])
    summary.extend(['', *bounded['limitations'], f"Report truncated: {bounded['report_truncated']}"])
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('diagnostics.json', data)
        summary_bytes = '\n'.join(summary).encode('utf-8')
        if len(summary_bytes) > 65536:
            summary_bytes = summary_bytes[:65000].decode('utf-8', errors='ignore').encode('utf-8') + b'\nSummary truncated; consult diagnostics.json for the bounded structured report.\n'
        archive.writestr('summary.txt', summary_bytes)
    return f'diagnostics-{identifier}.zip', 'application/zip', output.getvalue()
