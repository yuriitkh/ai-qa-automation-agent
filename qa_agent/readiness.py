"""Bounded, diagnostic readiness checks for local TestCase and Suite runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import errno
import http.client
import importlib.util
import ipaddress
import math
from functools import lru_cache
import os
from pathlib import Path
import queue
import socket
import sqlite3
import ssl
import tempfile
from threading import BoundedSemaphore, Event, Thread
from time import monotonic
from typing import Callable, TypeVar
from urllib.parse import parse_qsl, quote, urlsplit
from uuid import UUID

from qa_agent.cookie_consent import CookieConsentPolicy
from qa_agent.automation_lifecycle import AutomationStatus, AutomationLifecycleService
from qa_agent.evidence_policy import EvidencePolicy
from qa_agent.models import TestCase
from qa_agent.pinned_execution import PlanVersionSet, StepPlanSelection
from qa_agent.run_history import WorkflowType
from qa_agent.test_case_review import TestCaseReviewService, TestCaseReviewStatus


class ReadinessStatus(str, Enum):
    READY = "READY"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class ReadinessCheck:
    id: str
    name: str
    status: ReadinessStatus
    safe_message: str
    remediation: str | None = None
    blocking: bool = False
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_ms: int | None = None
    group: str | None = None


@dataclass(frozen=True)
class ReadinessReport:
    checks: tuple[ReadinessCheck, ...]

    @property
    def status(self) -> ReadinessStatus:
        if any(check.blocking or check.status == ReadinessStatus.BLOCKED for check in self.checks):
            return ReadinessStatus.BLOCKED
        if any(check.status == ReadinessStatus.WARNING for check in self.checks):
            return ReadinessStatus.WARNING
        if self.checks and all(check.status == ReadinessStatus.NOT_APPLICABLE for check in self.checks):
            return ReadinessStatus.NOT_APPLICABLE
        return ReadinessStatus.READY

    @property
    def can_continue(self) -> bool:
        return self.status != ReadinessStatus.BLOCKED

    @property
    def blocked(self) -> tuple[ReadinessCheck, ...]:
        return tuple(check for check in self.checks if check.blocking or check.status == ReadinessStatus.BLOCKED)


_REQUIRED_TABLES = frozenset({
    "test_cases", "cached_test_plans", "test_plan_versions", "executions",
    "run_history", "test_case_review", "test_case_automation_lifecycle",
    "drafts", "test_suites", "test_suite_members", "suite_runs",
    "provider_settings", "custom_provider_settings", "llm_usage",
})
_SECRET_QUERY_KEYS = frozenset({
    "api_key", "api-key", "apikey", "access_token", "access-token",
    "token", "password", "secret", "authorization", "credential",
    "key", "auth", "client_secret", "id_token", "refresh_token",
    "signature", "sig", "x-amz-credential", "x-amz-security-token", "x-amz-signature",
})
_DNS_SLOTS = BoundedSemaphore(4)
_TARGET_SLOTS = BoundedSemaphore(8)
_REPORT_SLOTS = BoundedSemaphore(4)
_T = TypeVar("_T")


def _bounded_call(
    operation: Callable[[], _T], timeout_seconds: float, slots: BoundedSemaphore,
    *, cancel: Callable[[], None] | None = None,
) -> _T:
    """Bound caller latency and outstanding work, including uninterruptible DNS."""
    deadline = monotonic() + timeout_seconds
    if timeout_seconds <= 0 or not slots.acquire(blocking=False):
        raise TimeoutError("Readiness check capacity or deadline exceeded.")
    results: queue.Queue = queue.Queue(maxsize=1)

    def execute() -> None:
        try:
            results.put((True, operation()))
        except Exception as error:
            results.put((False, error))
        finally:
            slots.release()

    try:
        Thread(target=execute, name="qa-readiness", daemon=True).start()
    except Exception:
        slots.release()
        raise
    try:
        succeeded, value = results.get(timeout=max(0, deadline - monotonic()))
    except queue.Empty:
        if cancel is not None:
            cancel()
        raise TimeoutError("Readiness deadline exceeded.") from None
    if not succeeded:
        raise value
    return value


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, address: str, port: int, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, port: int, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def check_target_url(value: str, *, timeout_seconds: float = 2.0) -> ReadinessCheck:
    """Bound DNS, address attempts, TLS, and HEAD response by one deadline."""
    started = monotonic()
    stopped = Event()
    connections: list[http.client.HTTPConnection] = []

    def cancel() -> None:
        stopped.set()
        for connection in connections:
            connection.close()

    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise TimeoutError
        return _bounded_call(
            lambda: _check_target_url(value, started + timeout_seconds, stopped, connections),
            timeout_seconds, _TARGET_SLOTS, cancel=cancel,
        )
    except Exception as error:
        timed_out = isinstance(error, TimeoutError)
        return ReadinessCheck(
            "target", "Target", ReadinessStatus.BLOCKED,
            "The target did not respond before the readiness timeout."
            if timed_out else "Cannot safely check the configured target website.",
            "Check the target URL and availability, then retry readiness.", True,
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )


def _check_target_url(
    value: str, deadline: float, stopped: Event,
    connections: list[http.client.HTTPConnection],
) -> ReadinessCheck:
    started = monotonic()

    def result(
        status: ReadinessStatus,
        message: str,
        remediation: str | None = None,
        *,
        blocking: bool | None = None,
    ) -> ReadinessCheck:
        is_blocking = status == ReadinessStatus.BLOCKED if blocking is None else blocking
        return ReadinessCheck(
            "target", "Target", status, message, remediation, is_blocking,
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )

    if (
        not isinstance(value, str) or not value.strip() or len(value) > 4096
        or any(ord(character) <= 32 or ord(character) == 127 for character in value)
        or "\\" in value
    ):
        return result(ReadinessStatus.BLOCKED, "The configured target URL is invalid.", "Enter a valid HTTP or HTTPS target URL.")
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return result(ReadinessStatus.BLOCKED, "The configured target URL is invalid.", "Correct the target URL and try again.")
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not hostname
        or "%" in hostname
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or any(key.casefold() in _SECRET_QUERY_KEYS for key, _ in parse_qsl(parsed.query, keep_blank_values=True))
    ):
        return result(
            ReadinessStatus.BLOCKED,
            "The configured target URL is invalid or contains credentials.",
            "Use an HTTP or HTTPS URL without credentials or secret query parameters.",
        )
    port = port or (443 if parsed.scheme.casefold() == "https" else 80)
    hostname = hostname.encode("idna").decode("ascii")
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
        addresses = [str(address)]
    except ValueError:
        resolved = _resolve_bounded(hostname, port, timeout_seconds=min(max(0, deadline - monotonic()), 1.0))
        if resolved is None:
            return result(
                ReadinessStatus.BLOCKED,
                "Cannot resolve the target website within the readiness timeout.",
                "Check the target hostname and local network connection.",
            )
        if not resolved:
            return result(
                ReadinessStatus.BLOCKED,
                "Cannot connect to the target website.",
                "Check the target hostname and local network connection.",
            )
        addresses = resolved
    loopback_name = hostname.casefold().rstrip(".") == "localhost"
    literal_loopback = False
    try:
        literal_loopback = ipaddress.ip_address(hostname.strip("[]")).is_loopback
    except ValueError:
        pass
    if any(
        not _allowed_target_address(address, allow_loopback=loopback_name or literal_loopback)
        for address in addresses
    ):
        return result(
            ReadinessStatus.BLOCKED,
            "The target resolves to a restricted local or private address.",
            "Use a public target or a loopback development target such as localhost.",
        )

    request_path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    if parsed.query:
        request_path += "?" + quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
    host_header = f"[{hostname}]" if ":" in hostname else hostname
    if port != (443 if parsed.scheme.casefold() == "https" else 80):
        host_header += f":{port}"
    last_error: OSError | TimeoutError | ssl.SSLError | None = None
    for address in addresses[:4]:
        remaining = deadline - monotonic()
        if stopped.is_set() or remaining <= 0:
            raise TimeoutError
        connection: http.client.HTTPConnection
        if parsed.scheme.casefold() == "https":
            connection = _PinnedHTTPSConnection(hostname, address, port, remaining)
        else:
            connection = _PinnedHTTPConnection(hostname, address, port, remaining)
        connections.append(connection)
        try:
            if stopped.is_set() or monotonic() >= deadline:
                raise TimeoutError
            connection.request(
                "HEAD", request_path,
                headers={"Host": host_header, "User-Agent": "AI-QA-Agent-readiness/1.0", "Connection": "close"},
            )
            if stopped.is_set() or monotonic() >= deadline:
                raise TimeoutError
            if connection.sock is not None:
                connection.sock.settimeout(max(0.001, deadline - monotonic()))
            response = connection.getresponse()
            response_status = int(response.status)
            response.close()
            if response_status >= 400:
                return result(
                    ReadinessStatus.WARNING,
                    f"The target responded with HTTP {response_status}; this check does not judge application behavior.",
                    "Review the target response if the status is unexpected; you may continue safely.",
                    blocking=False,
                )
            return result(ReadinessStatus.READY, f"The target responded with HTTP {response_status}.")
        except (OSError, TimeoutError, ssl.SSLError) as error:
            last_error = error
        finally:
            connection.close()
    if (
        isinstance(last_error, socket.timeout)
        or getattr(last_error, "winerror", None) == 10060
        or getattr(last_error, "errno", None) in {errno.ETIMEDOUT, 10060}
    ):
        return result(
            ReadinessStatus.BLOCKED,
            "The target did not respond before the readiness timeout.",
            "Check target availability and retry the readiness check.",
        )
    return result(
        ReadinessStatus.BLOCKED,
        "Cannot connect to the target website.",
        "Check the target URL, server availability, and local network connection.",
    )


def _allowed_target_address(value: str, *, allow_loopback: bool) -> bool:
    address = ipaddress.ip_address(value)
    if allow_loopback and address.is_loopback:
        return True
    if address.is_multicast or address.is_unspecified or address.is_reserved or address.is_link_local:
        return False
    if isinstance(address, ipaddress.IPv6Address) and (address.sixtofour is not None or address.teredo is not None):
        return False
    return address.is_global


def _resolve_bounded(hostname: str, port: int, *, timeout_seconds: float) -> list[str] | None:
    if not _DNS_SLOTS.acquire(blocking=False):
        return None
    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            values = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            addresses = list(dict.fromkeys(str(item[4][0]).split("%", 1)[0] for item in values))
            result_queue.put((True, addresses))
        except Exception:
            result_queue.put((False, None))
        finally:
            _DNS_SLOTS.release()

    try:
        Thread(target=resolve, name="qa-readiness-dns", daemon=True).start()
    except Exception:
        _DNS_SLOTS.release()
        return None
    try:
        succeeded, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return None
    return value if succeeded and isinstance(value, list) else []


class SystemReadinessService:
    """Compose local platform, workflow, target, and policy readiness checks."""

    def __init__(
        self,
        *,
        database_path: str | Path | None,
        evidence_directory: str | Path | None,
        application_ready: bool,
        run_service=None,
        background_runs=None,
        suite_run_service=None,
        provider_settings=None,
        test_cases=None,
        automation_lifecycle: AutomationLifecycleService | None = None,
        test_case_review: TestCaseReviewService | None = None,
        database_probe: Callable[[str | Path | None], ReadinessCheck] | None = None,
        browser_probe: Callable[[bool], ReadinessCheck] | None = None,
        evidence_probe: Callable[[str | Path | None, bool], ReadinessCheck] | None = None,
        target_probe: Callable[[str], ReadinessCheck] = check_target_url,
        aggregate_timeout_seconds: float = 10.0,
    ) -> None:
        self.database_path = database_path
        self.evidence_directory = evidence_directory
        self.application_ready = application_ready
        self.run_service = run_service
        self.background_runs = background_runs
        self.suite_run_service = suite_run_service
        self.provider_settings = provider_settings
        self.test_cases = test_cases
        self.automation_lifecycle = automation_lifecycle
        self.test_case_review = test_case_review
        self._database_probe = database_probe or _check_database
        self._browser_probe = browser_probe or _check_browser
        self._evidence_probe = evidence_probe or _check_evidence
        self._target_probe = target_probe
        if not math.isfinite(aggregate_timeout_seconds) or aggregate_timeout_seconds <= 0:
            raise ValueError("Readiness aggregate timeout must be finite and positive.")
        self.aggregate_timeout_seconds = aggregate_timeout_seconds

    def _bounded_report(self, operation: Callable[[float], ReadinessReport]) -> ReadinessReport:
        started = monotonic()
        deadline = started + self.aggregate_timeout_seconds
        try:
            return _bounded_call(lambda: operation(deadline), self.aggregate_timeout_seconds, _REPORT_SLOTS)
        except Exception as error:
            return ReadinessReport((ReadinessCheck(
                "readiness", "Readiness checks", ReadinessStatus.BLOCKED,
                "Readiness checks exceeded the aggregate timeout or all diagnostic slots are busy."
                if isinstance(error, TimeoutError) else "Readiness checks could not complete safely.",
                "Review local System Health and retry readiness before starting execution.", True,
                elapsed_ms=max(0, int((monotonic() - started) * 1000)),
            ),))

    def _target_check(self, target: str, deadline: float) -> ReadinessCheck:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError
        if self._target_probe is check_target_url:
            return check_target_url(target, timeout_seconds=min(2.0, remaining))
        return _bounded_call(lambda: self._target_probe(target), remaining, _TARGET_SLOTS)

    def system_report(
        self,
        *,
        deep_browser: bool = False,
        probe_evidence: bool = False,
        worker: str | None = None,
        include_provider: bool = True,
    ) -> ReadinessReport:
        if deep_browser:
            _cached_browser_snapshot.cache_clear()
        checks = [
            self._database_probe(self.database_path),
            self._browser_probe(deep_browser)
            if deep_browser else _cached_browser_snapshot(self._browser_probe),
            self._evidence_probe(self.evidence_directory, probe_evidence),
            self._worker_check(worker),
            ReadinessCheck(
                "application", "Application",
                ReadinessStatus.READY if self.application_ready else ReadinessStatus.BLOCKED,
                "Essential application services are initialized." if self.application_ready else "Essential application services are unavailable.",
                None if self.application_ready else "Restart the application and review its local startup diagnostics.",
                not self.application_ready,
            ),
            self._provider_summary() if include_provider else ReadinessCheck(
                "provider", "AI Provider", ReadinessStatus.NOT_APPLICABLE,
                "Saved automation does not require provider availability; no provider check was made.",
            ),
        ]
        return ReadinessReport(tuple(checks))

    def testcase_report(
        self,
        test_case: TestCase | None,
        workflow: WorkflowType,
        *,
        evidence_policy: EvidencePolicy,
        cookie_policy: CookieConsentPolicy,
    ) -> ReadinessReport:
        return self._bounded_report(lambda deadline: self._testcase_report(
            test_case, workflow, evidence_policy=evidence_policy,
            cookie_policy=cookie_policy, deadline=deadline,
        ))

    def _testcase_report(
        self, test_case: TestCase | None, workflow: WorkflowType, *,
        evidence_policy: EvidencePolicy, cookie_policy: CookieConsentPolicy, deadline: float,
    ) -> ReadinessReport:
        checks = list(self.system_report(
            deep_browser=True, probe_evidence=True, worker="case",
            include_provider=workflow == WorkflowType.AUTOMATION,
        ).checks)
        if test_case is None:
            checks.append(ReadinessCheck(
                "test_case", "TestCase", ReadinessStatus.BLOCKED,
                "TestCase not found.", "Return to the TestCase list and choose a saved TestCase.", True,
            ))
            return ReadinessReport(tuple(checks))
        checks.append(ReadinessCheck(
            "test_case", "TestCase",
            ReadinessStatus.READY if test_case.steps else ReadinessStatus.BLOCKED,
            "Executable TestCase steps are available." if test_case.steps else "This TestCase has no executable steps.",
            None if test_case.steps else "Edit the TestCase and add at least one step.",
            not bool(test_case.steps),
        ))
        self._append_review_check(checks, test_case)
        self._append_workflow_check(checks, test_case, workflow)
        if workflow == WorkflowType.AUTOMATION:
            checks.append(self._required_provider_check())
        targets, missing_targets = _test_case_targets(
            test_case, self.run_service, include_saved_plans=workflow != WorkflowType.AUTOMATION,
        )
        checks.extend(self._target_check(target, deadline) for target in targets)
        if missing_targets or not targets:
            checks.append(ReadinessCheck(
                "target", "Target", ReadinessStatus.BLOCKED,
                "One or more TestCase segments have no valid target URL configured.",
                "Edit the TestCase and set an HTTP or HTTPS URL without credentials for each segment.", True,
            ))
        checks.append(_policy_check(evidence_policy, cookie_policy))
        return ReadinessReport(tuple(checks))

    def suite_report(
        self,
        preview,
        *,
        evidence_policy: EvidencePolicy,
        cookie_policy: CookieConsentPolicy,
    ) -> ReadinessReport:
        return self._bounded_report(lambda deadline: self._suite_report(
            preview, evidence_policy=evidence_policy,
            cookie_policy=cookie_policy, deadline=deadline,
        ))

    def _suite_report(
        self, preview, *, evidence_policy: EvidencePolicy,
        cookie_policy: CookieConsentPolicy, deadline: float,
    ) -> ReadinessReport:
        # Resolve the preview inside the same deadline as the platform and target checks.
        if callable(preview):
            preview = preview()
        if monotonic() >= deadline:
            raise TimeoutError
        checks = list(self.system_report(
            deep_browser=True, probe_evidence=True, worker="suite", include_provider=False,
        ).checks)
        if preview is None or preview.suite is None:
            checks.append(ReadinessCheck(
                "suite", "Test Suite", ReadinessStatus.BLOCKED,
                "Test Suite not found.", "Return to Test Suites and choose a saved suite.", True,
            ))
            return ReadinessReport(tuple(checks))
        if preview.suite_error:
            checks.append(ReadinessCheck(
                "suite", "Test Suite", ReadinessStatus.BLOCKED,
                preview.suite_error, "Resolve the suite setup error before starting.", True,
            ))
        if not preview.eligibility:
            checks.append(ReadinessCheck(
                "suite_members", "Suite members", ReadinessStatus.BLOCKED,
                "This suite has no TestCases.", "Add at least one TestCase before starting a Suite Run.", True,
            ))
        target_checks: dict[str, ReadinessCheck] = {}
        for item in preview.eligibility:
            if monotonic() >= deadline:
                raise TimeoutError
            prefix = f"member_{item.order_index + 1}"
            test_case = None
            getter = getattr(self.test_cases, "get", None)
            if callable(getter):
                test_case = getter(item.test_case_id)
            reasons = list(item.reasons)
            if not item.eligible and not reasons:
                reasons.append("This suite member is not eligible for saved Regression execution.")
            if test_case is None:
                reasons.append("This suite member no longer exists.")
            else:
                if item.test_case_snapshot:
                    test_case = TestCase.model_validate(item.test_case_snapshot)
                selected = {pin.test_step_id: pin.test_plan_version_id for pin in item.pinned_plans}
                if self.test_case_review is not None:
                    if self.test_case_review.status(test_case.id) != TestCaseReviewStatus.APPROVED:
                        reasons.append("Approve this TestCase before running the suite.")
                    else:
                        approval = getattr(self.run_service, "pinned_regression_approval_error", None)
                        if callable(approval):
                            error = approval(test_case, PlanVersionSet(tuple(
                                StepPlanSelection(step_id, version_id) for step_id, version_id in selected.items()
                            )))
                            if error:
                                reasons.append(error)
                        elif not self.test_case_review.validation_approved_for(test_case):
                            reasons.append("Approve the current saved automation for Validation before running the suite.")
                targets, missing_targets = _test_case_targets(
                    test_case, self.run_service, selected_versions=selected,
                )
                if missing_targets or not targets:
                    reasons.append("One or more segments have no valid target URL configured.")
                for target in targets:
                    if target not in target_checks:
                        target_checks[target] = self._target_check(target, deadline)
                    target_check = target_checks[target]
                    checks.append(ReadinessCheck(
                        f"{prefix}_target", f"Suite member {item.order_index + 1} target",
                        target_check.status, target_check.safe_message, target_check.remediation,
                        target_check.blocking, target_check.checked_at, target_check.elapsed_ms,
                        group=prefix,
                    ))
            blocked = bool(reasons)
            checks.append(ReadinessCheck(
                f"{prefix}_automation", f"Suite member {item.order_index + 1} automation",
                ReadinessStatus.BLOCKED if blocked else ReadinessStatus.READY,
                "; ".join(reasons) if blocked else "Saved Regression automation and review approval are ready.",
                "Resolve this TestCase blocker before starting the Suite Run." if blocked else None,
                blocked,
                group=prefix,
            ))
        checks.append(_policy_check(evidence_policy, cookie_policy))
        return ReadinessReport(tuple(checks))

    def _append_review_check(self, checks: list[ReadinessCheck], test_case: TestCase) -> None:
        status = (
            self.test_case_review.status(test_case.id)
            if self.test_case_review is not None else TestCaseReviewStatus.APPROVED
        )
        approved = status == TestCaseReviewStatus.APPROVED
        checks.append(ReadinessCheck(
            "test_case_review", "TestCase review",
            ReadinessStatus.READY if approved else ReadinessStatus.BLOCKED,
            "TestCase is approved." if approved else "This TestCase is waiting for review approval.",
            None if approved else "Review and approve the TestCase before execution.",
            not approved,
        ))

    def _append_workflow_check(
        self, checks: list[ReadinessCheck], test_case: TestCase, workflow: WorkflowType
    ) -> None:
        if workflow == WorkflowType.AUTOMATION:
            ready = bool(getattr(self.run_service, "_automation_workflow", None))
            message = "Automation generation is available." if ready else "Automation generation is unavailable."
            remediation = None if ready else "Restart the application with its execution service configured."
        else:
            availability = None
            method = getattr(self.run_service, "workflow_availability_for_test_case", None)
            if callable(method):
                try:
                    availability = method(test_case)
                except Exception:
                    availability = None
            ready = bool(
                availability is not None
                and (
                    availability.validation_available
                    if workflow == WorkflowType.VALIDATION
                    else availability.regression_available
                )
            )
            message = (
                "Saved automation is ready for this workflow."
                if ready else (getattr(availability, "reason", None) or "This TestCase requires a usable saved TestPlan.")
            )
            remediation = None if ready else "Generate or repair automation, then review the saved plan versions."
            lifecycle = self.automation_lifecycle
            if lifecycle is not None:
                lifecycle_status = lifecycle.status(test_case)
                if lifecycle_status in {
                    AutomationStatus.NOT_AUTOMATED,
                    AutomationStatus.NEEDS_UPDATE,
                    AutomationStatus.AUTOMATION_FAILED,
                }:
                    ready = False
                    message = "The automation lifecycle does not allow this workflow with the current TestCase definition."
                    remediation = "Generate or repair automation for the current TestCase, then review and approve it."
                elif (
                    workflow == WorkflowType.REGRESSION
                    and availability is not None
                    and availability.regression_available
                    and not availability.validation_available
                ):
                    checks.append(ReadinessCheck(
                        "expected_result_coverage", "Expected-result coverage",
                        ReadinessStatus.WARNING,
                        "Regression can continue, but incomplete expected-result coverage may be reported as an automation error.",
                        "Review the saved assertions before relying on this Regression result.",
                        blocking=False,
                    ))
            if ready and self.test_case_review is not None and not self.test_case_review.validation_approved_for(test_case):
                ready = False
                message = "The current saved automation has not been approved for Validation."
                remediation = "Review and approve the current saved plan versions before running."
        checks.append(ReadinessCheck(
            "automation", "Automation",
            ReadinessStatus.READY if ready else ReadinessStatus.BLOCKED,
            message, remediation, not ready,
        ))

    def _provider_summary(self) -> ReadinessCheck:
        if self.provider_settings is None:
            return ReadinessCheck(
                "provider", "AI Provider", ReadinessStatus.NOT_APPLICABLE,
                "Provider settings are not configured; saved automation does not require a provider.",
            )
        try:
            providers = self.provider_settings.provider_views()
            eligible = [item for item in providers if item.enabled and item.status == "CONFIGURED"]
        except Exception:
            return ReadinessCheck(
                "provider", "AI Provider", ReadinessStatus.WARNING,
                "Provider configuration could not be read; saved automation remains available.",
                "Review AI Providers settings only if AI generation is required.",
            )
        if eligible:
            return ReadinessCheck(
                "provider", "AI Provider", ReadinessStatus.READY,
                f"{len(eligible)} enabled provider configuration(s) are available; no connection test was made.",
            )
        return ReadinessCheck(
            "provider", "AI Provider", ReadinessStatus.WARNING,
            "No enabled provider is configured; saved automation does not require one.",
            "Configure an AI provider only when AI automation generation is required.",
        )

    def _required_provider_check(self) -> ReadinessCheck:
        summary = self._provider_summary()
        configured = summary.status == ReadinessStatus.READY
        return ReadinessCheck(
            "provider_required", "AI Provider",
            ReadinessStatus.READY if configured else ReadinessStatus.BLOCKED,
            "An enabled AI provider is configured; no live connection test was made."
            if configured else "AI automation generation requires an enabled configured provider.",
            None if configured else "Configure an enabled provider in AI Providers settings.",
            not configured,
        )

    def _worker_check(self, worker: str | None) -> ReadinessCheck:
        services = []
        if worker in {None, "case"} and self.background_runs is not None:
            services.append(("TestCase", self.background_runs))
        if worker in {None, "suite"} and self.suite_run_service is not None:
            services.append(("Suite", self.suite_run_service))
        if not services:
            return ReadinessCheck(
                "background", "Background execution", ReadinessStatus.BLOCKED,
                "Background execution is unavailable.", "Restart the application execution services.", True,
            )
        states = [(name, _worker_state(service)) for name, service in services]
        if any(available for _name, (_closed, available, _active, _capacity) in states):
            if any(closed for _name, (closed, _available, _active, _capacity) in states):
                return ReadinessCheck(
                    "background", "Background execution", ReadinessStatus.WARNING,
                    "At least one execution worker is available; another worker is stopped.",
                    blocking=False,
                )
            active = sum(item[1][2] for item in states)
            capacity = sum(item[1][3] for item in states)
            return ReadinessCheck(
                "background", "Background execution", ReadinessStatus.READY,
                f"Execution workers are available ({active} of {capacity} slots in use).",
            )
        if all(closed for _name, (closed, _available, _active, _capacity) in states):
            return ReadinessCheck(
                "background", "Background execution", ReadinessStatus.BLOCKED,
                "Background execution workers are stopped.", "Restart the application.", True,
            )
        return ReadinessCheck(
            "background", "Background execution", ReadinessStatus.BLOCKED,
            "All background execution slots are busy.", "Wait for a running job to finish, then retry.", True,
        )


def _worker_state(service) -> tuple[bool, bool, int, int]:
    method = getattr(service, "readiness_state", None)
    if callable(method):
        try:
            value = method()
            return (
                bool(value.get("closed")), bool(value.get("available")),
                int(value.get("active", 0)), int(value.get("capacity", 0)),
            )
        except Exception:
            return False, False, 0, 0
    return False, True, 0, 1


@lru_cache(maxsize=4)
def _cached_browser_snapshot(probe: Callable[[bool], ReadinessCheck]) -> ReadinessCheck:
    return probe(False)


def _check_database(database_path: str | Path | None) -> ReadinessCheck:
    if database_path is None or str(database_path) == ":memory:":
        return ReadinessCheck(
            "database", "Database", ReadinessStatus.NOT_APPLICABLE,
            "No shared file-backed SQLite database is configured.",
        )
    started = monotonic()
    try:
        path = Path(database_path).expanduser().resolve()
        if not path.is_file():
            raise OSError
        uri = path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=0.5)
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("SELECT 1").fetchone()
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            connection.close()
        missing = _REQUIRED_TABLES - tables
        if missing:
            return ReadinessCheck(
                "database", "Database", ReadinessStatus.BLOCKED,
                "Cannot access the local database because required application tables are missing.",
                "Restart the application to complete its local storage initialization.", True,
                elapsed_ms=max(0, int((monotonic() - started) * 1000)),
            )
        return ReadinessCheck(
            "database", "Database", ReadinessStatus.READY,
            "Local database connection and required tables are available.",
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )
    except Exception:
        return ReadinessCheck(
            "database", "Database", ReadinessStatus.BLOCKED,
            "Cannot access the local database.", "Check the configured database file and restart the application.", True,
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )


def _check_browser(deep: bool) -> ReadinessCheck:
    started = monotonic()
    try:
        if importlib.util.find_spec("playwright") is None:
            raise ImportError
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                return ReadinessCheck(
                    "browser", "Browser", ReadinessStatus.BLOCKED,
                    "Playwright Chromium is not installed.",
                    "Install the Playwright Chromium browser with the project setup command.", True,
                    elapsed_ms=max(0, int((monotonic() - started) * 1000)),
                )
            if deep:
                browser = playwright.chromium.launch(headless=True, timeout=3500)
                browser.close()
        return ReadinessCheck(
            "browser", "Browser", ReadinessStatus.READY,
            "Playwright Chromium is installed and available." if deep else "Playwright and its Chromium executable are available.",
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )
    except Exception as error:
        category = type(error).__name__
        message = (
            "Playwright Chromium is not installed."
            if category in {"ImportError", "ModuleNotFoundError"}
            else "Playwright Chromium could not start."
            if deep else "Playwright Chromium is unavailable."
        )
        return ReadinessCheck(
            "browser", "Browser", ReadinessStatus.BLOCKED, message,
            "Install or repair the Playwright Chromium browser, then refresh System Health.", True,
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )


def _check_evidence(directory: str | Path | None, probe: bool) -> ReadinessCheck:
    started = monotonic()
    if directory is None:
        return ReadinessCheck(
            "evidence", "Evidence storage", ReadinessStatus.BLOCKED,
            "Evidence directory is not configured.", "Configure a local evidence directory before running tests.", True,
        )
    try:
        root = Path(directory).expanduser().resolve()
        if probe:
            root.mkdir(parents=True, exist_ok=True)
            descriptor, probe_name = tempfile.mkstemp(prefix=".qa-readiness-", dir=root)
            os.close(descriptor)
            Path(probe_name).unlink()
        elif root.is_dir():
            if not os.access(root, os.W_OK):
                raise OSError
        else:
            parent = root.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK):
                raise OSError
        return ReadinessCheck(
            "evidence", "Evidence storage", ReadinessStatus.READY,
            "Evidence directory is available and writable." if probe else "Evidence directory exists or can be prepared.",
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )
    except Exception:
        return ReadinessCheck(
            "evidence", "Evidence storage", ReadinessStatus.BLOCKED,
            "Evidence directory is not writable.", "Choose a writable local evidence directory and retry.", True,
            elapsed_ms=max(0, int((monotonic() - started) * 1000)),
        )


def _test_case_targets(
    test_case: TestCase, run_service, *, selected_versions: dict[UUID, UUID] | None = None,
    include_saved_plans: bool = True,
) -> tuple[tuple[str, ...], int]:
    values = []
    missing = 0
    store = getattr(run_service, "_plan_store", None)
    for segment in test_case.segments:
        value = segment.base_url or test_case.base_url
        segment_targets = [value] if value else []
        if include_saved_plans and store is not None:
            for step in segment.steps:
                if selected_versions is None:
                    version = store.find(step.id)
                else:
                    version_id = selected_versions.get(step.id)
                    version = store.get_version(version_id) if version_id is not None else None
                if version is not None:
                    segment_targets.append(version.qa_test_plan.url)
                    segment_targets.extend(
                        action.parameters.get("url") for action in version.qa_test_plan.steps
                        if action.action == "navigate"
                    )
        segment_targets = [target for target in segment_targets if target]
        for target in segment_targets:
            if target not in values:
                values.append(target)
        if not segment_targets:
            missing += 1
    return tuple(values), missing


def _policy_check(evidence_policy: EvidencePolicy, cookie_policy: CookieConsentPolicy) -> ReadinessCheck:
    evidence_ok = isinstance(evidence_policy, EvidencePolicy)
    cookie_ok = isinstance(cookie_policy, CookieConsentPolicy)
    valid = evidence_ok and cookie_ok
    if cookie_ok and cookie_policy == CookieConsentPolicy.AUTO_HANDLE:
        detail = "Automatic cookie handling is configured for execution; no consent state was read or changed."
    elif cookie_ok:
        detail = "Cookie consent will remain unchanged during execution."
    else:
        detail = "The selected evidence or cookie policy is invalid."
    return ReadinessCheck(
        "execution_policy", "Evidence and cookie policy",
        ReadinessStatus.READY if valid else ReadinessStatus.BLOCKED,
        detail,
        None if valid else "Choose a supported evidence mode and cookie policy.",
        not valid,
    )
