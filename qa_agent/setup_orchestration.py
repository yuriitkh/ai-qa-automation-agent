"""Reusable TestCase precondition setup and cleanup orchestration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol
from uuid import UUID

from qa_agent.models import Precondition, TestCase
from qa_agent.redaction import redact_secrets, safe_failure_reason
from qa_agent.run_context import RunContext


class SetupStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    PRECONDITION_NOT_ESTABLISHED = "PRECONDITION_NOT_ESTABLISHED"
    SETUP_INFRASTRUCTURE_ERROR = "SETUP_INFRASTRUCTURE_ERROR"


@dataclass(frozen=True)
class SetupOperationResult:
    """Result returned by one strategy; values are written to RunContext."""

    status: SetupStatus
    produced_data_keys: tuple[str, ...] = ()
    error: str | None = field(default=None, repr=False)
    error_type: str | None = None


CleanupAction = Callable[[RunContext], None]
CleanupRegistrar = Callable[[CleanupAction, str | None], None]


class SetupOperation(Protocol):
    """Infrastructure adapter for establishing one declarative precondition."""

    def execute(
        self,
        precondition: Precondition,
        run_context: RunContext,
        register_cleanup: CleanupRegistrar,
    ) -> SetupOperationResult: ...


@dataclass(frozen=True)
class PreconditionSetupOutcome:
    precondition_id: UUID
    status: SetupStatus
    produced_data_keys: tuple[str, ...] = ()
    cleanup_registered: bool = False
    error: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class SetupRunOutcome:
    status: SetupStatus
    preconditions: tuple[PreconditionSetupOutcome, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status == SetupStatus.SUCCEEDED


@dataclass(frozen=True)
class CleanupFailure:
    label: str
    error_type: str
    message: str


@dataclass(frozen=True)
class CleanupOutcome:
    failures: tuple[CleanupFailure, ...] = ()

    @property
    def succeeded(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class TestCaseRunOutcome:
    """Keep setup, product execution, and cleanup outcomes independent."""

    setup: SetupRunOutcome
    run_context: RunContext = field(repr=False)
    product_started: bool
    product_result: Any = field(default=None, repr=False)
    product_error_type: str | None = None
    product_error: str | None = None
    cleanup: CleanupOutcome = field(default_factory=CleanupOutcome)

    def __repr__(self) -> str:
        return (
            "TestCaseRunOutcome("
            f"setup={self.setup!r}, product_started={self.product_started!r}, "
            f"product_error_type={self.product_error_type!r}, "
            f"product_error={self.product_error!r}, cleanup={self.cleanup!r})"
        )


@dataclass(frozen=True)
class _RegisteredCleanup:
    action: CleanupAction = field(repr=False)
    label: str


class CleanupManager:
    """In-process LIFO cleanup stack with idempotent execution."""

    def __init__(self, run_context: RunContext) -> None:
        self._run_context = run_context
        self._actions: list[_RegisteredCleanup] = []
        self._running = False
        self._outcome: CleanupOutcome | None = None

    @property
    def registered_count(self) -> int:
        return len(self._actions)

    def register(self, action: CleanupAction, label: str | None = None) -> None:
        if self._running or self._outcome is not None:
            raise RuntimeError("Cannot register cleanup after cleanup has started.")
        if not callable(action):
            raise TypeError("Cleanup action must be callable.")
        if any(item.action is action for item in self._actions):
            raise ValueError("The same cleanup action cannot be registered twice.")
        self._actions.append(_RegisteredCleanup(action, label or "cleanup"))

    def run(self) -> CleanupOutcome:
        if self._outcome is not None:
            return self._outcome
        if self._running:
            raise RuntimeError("Cleanup is already running.")

        self._running = True
        failures: list[CleanupFailure] = []
        try:
            for registered in reversed(self._actions):
                try:
                    registered.action(self._run_context)
                except Exception as error:
                    failures.append(CleanupFailure(
                        label=_safe_message(registered.label, self._run_context),
                        error_type=type(error).__name__,
                        message=_safe_error(error, self._run_context),
                    ))
        finally:
            self._outcome = CleanupOutcome(failures=tuple(failures))
            self._running = False
        return self._outcome


class SetupCoordinator:
    """Run registered setup operations in declared TestCase precondition order."""

    def __init__(self, operations: Mapping[UUID, SetupOperation]) -> None:
        # An instance-local registry avoids a global service locator.
        self._operations = dict(operations)

    def run(
        self,
        test_case: TestCase,
        run_context: RunContext,
        cleanup: CleanupManager,
    ) -> SetupRunOutcome:
        outcomes: list[PreconditionSetupOutcome] = []
        for precondition in sorted(test_case.preconditions, key=lambda item: item.order):
            operation = self._operations.get(precondition.id)
            if operation is None:
                outcome = PreconditionSetupOutcome(
                    precondition_id=precondition.id,
                    status=SetupStatus.SETUP_INFRASTRUCTURE_ERROR,
                    error="No setup operation is registered for this precondition.",
                    error_type="MissingSetupOperation",
                )
                outcomes.append(outcome)
                return SetupRunOutcome(outcome.status, tuple(outcomes))

            registered_before = cleanup.registered_count
            try:
                result = operation.execute(
                    precondition,
                    run_context,
                    cleanup.register,
                )
                if not isinstance(result, SetupOperationResult):
                    raise TypeError("Setup operation must return SetupOperationResult.")
                if not isinstance(result.status, SetupStatus):
                    raise TypeError("Setup operation returned an unsupported status.")
                if len(set(result.produced_data_keys)) != len(result.produced_data_keys):
                    raise ValueError("Setup result produced_data_keys must be unique.")
                for key in result.produced_data_keys:
                    run_context.has_value(key)
            except Exception as error:
                outcome = PreconditionSetupOutcome(
                    precondition_id=precondition.id,
                    status=SetupStatus.SETUP_INFRASTRUCTURE_ERROR,
                    cleanup_registered=cleanup.registered_count > registered_before,
                    error=_safe_error(error, run_context),
                    error_type=type(error).__name__,
                )
                outcomes.append(outcome)
                return SetupRunOutcome(outcome.status, tuple(outcomes))

            status = result.status
            error = _safe_message(result.error, run_context) if result.error else None
            error_type = result.error_type
            if status == SetupStatus.SUCCEEDED:
                missing_produced = [
                    key for key in result.produced_data_keys
                    if not run_context.has_value(key)
                ]
                missing_declared = [
                    key for key in precondition.provided_data_keys
                    if not run_context.has_value(key)
                ]
                missing = list(dict.fromkeys(missing_produced + missing_declared))
                if missing:
                    status = SetupStatus.PRECONDITION_NOT_ESTABLISHED
                    error = (
                        "Setup reported success but required RunContext keys are "
                        f"missing: {', '.join(missing)}."
                    )
                    error_type = "SetupContractError"
            elif error is None:
                error = (
                    "Setup operation could not establish the precondition."
                    if status == SetupStatus.PRECONDITION_NOT_ESTABLISHED
                    else "Setup operation reported an infrastructure error."
                )

            outcome = PreconditionSetupOutcome(
                precondition_id=precondition.id,
                status=status,
                produced_data_keys=tuple(result.produced_data_keys),
                cleanup_registered=cleanup.registered_count > registered_before,
                error=error,
                error_type=error_type,
            )
            outcomes.append(outcome)
            if status != SetupStatus.SUCCEEDED:
                return SetupRunOutcome(status, tuple(outcomes))

        return SetupRunOutcome(SetupStatus.SUCCEEDED, tuple(outcomes))


class SetupCleanupCoordinator:
    """Wrap a product operation in setup and finally-style cleanup."""

    def __init__(self, operations: Mapping[UUID, SetupOperation]) -> None:
        self._setup = SetupCoordinator(operations)

    def run(
        self,
        test_case: TestCase,
        run_context: RunContext,
        product_execution: Callable[[RunContext], Any],
    ) -> TestCaseRunOutcome:
        cleanup = CleanupManager(run_context)
        setup_outcome: SetupRunOutcome
        product_started = False
        product_result: Any = None
        product_error_type: str | None = None
        product_error: str | None = None

        try:
            setup_outcome = self._setup.run(test_case, run_context, cleanup)
            if setup_outcome.succeeded:
                product_started = True
                try:
                    product_result = product_execution(run_context)
                except Exception as error:
                    product_error_type = type(error).__name__
                    product_error = _safe_error(error, run_context)
        finally:
            cleanup_outcome = cleanup.run()

        return TestCaseRunOutcome(
            setup=setup_outcome,
            run_context=run_context,
            product_started=product_started,
            product_result=product_result,
            product_error_type=product_error_type,
            product_error=product_error,
            cleanup=cleanup_outcome,
        )


def _safe_error(error: Exception, run_context: RunContext) -> str:
    return _redact_context_values(safe_failure_reason(error), run_context)


def _safe_message(message: str, run_context: RunContext) -> str:
    return _redact_context_values(redact_secrets(message), run_context)[:300]


def _redact_context_values(message: str, run_context: RunContext) -> str:
    for sensitive_value in _sensitive_strings(run_context):
        if sensitive_value:
            message = message.replace(sensitive_value, "[REDACTED]")
    return message[:300]


def _sensitive_strings(run_context: RunContext) -> list[str]:
    found: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value:
                found.add(value)
        elif isinstance(value, bytes):
            if value:
                found.add(value.decode("utf-8", errors="replace"))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                collect(key)
                collect(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                collect(item)
        elif value is not None:
            found.add(str(value))

    for item in run_context.values.values():
        if item.sensitive:
            collect(item.value)
    return sorted(found, key=len, reverse=True)
