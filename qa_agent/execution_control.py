"""Cooperative job cancellation and an atomic completion boundary.

Browser work stays on its owning thread. Isolated SDK work may outlive a
cancelled caller, but retains its bounded worker slot and cannot publish results.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import BoundedSemaphore, Event, RLock, Thread


class OperationCancelled(RuntimeError):
    category = "CANCELLED"

    def __init__(self):
        super().__init__("Stopped by user.")


class CancellationToken:
    def __init__(self, parent=None):
        self.event = Event()
        self.parent = parent
        # Children share the parent's lock: a Suite Stop and a child commit
        # have one ordering, while completing a child never seals its parent.
        self.lock = parent.lock if parent is not None else RLock()
        self.sealed = False

    @property
    def requested(self):
        return self.event.is_set() or (self.parent is not None and self.parent.requested)

    def request(self, acknowledge=lambda: None):
        with self.lock:
            if self.sealed:
                return False
            self.event.set()
            acknowledge()
            return True

    def check(self):
        if self.requested:
            raise OperationCancelled()


_active = ContextVar("execution_cancellation", default=None)
_isolated = ContextVar("isolated_cancellable_call", default=False)
_sdk_slots = BoundedSemaphore(4)


def current_cancellation():
    return _active.get()


def check_cancelled():
    token = current_cancellation()
    if token is not None:
        token.check()


def is_cancelled(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, OperationCancelled) or getattr(error, "category", None) == "CANCELLED":
            return True
        error = error.__cause__
    return False


@contextmanager
def cancellation_scope(token):
    scope = _active.set(token)
    try:
        yield token
    finally:
        _active.reset(scope)


@contextmanager
def completion_boundary():
    """Hold cancellation ordering across a complete, atomic persistence step."""
    token = current_cancellation()
    if token is None:
        yield None
    else:
        with token.lock:
            yield token


def cancellable_call(operation):
    """Bound cancellation acknowledgement to a 50 ms poll for isolated SDK work.

    No thread is killed. The worker slot is released only when the actual call
    returns. The caller owns persistence; a detached response is discarded.
    """
    token = current_cancellation()
    if token is None or _isolated.get():
        check_cancelled()
        return operation()
    token.check()
    while not _sdk_slots.acquire(timeout=0.05):
        token.check()
    done = Event()
    values = []
    context = copy_context()

    def isolated_operation():
        scope = _isolated.set(True)
        try:
            return operation()
        finally:
            _isolated.reset(scope)

    def invoke():
        try:
            token.check()
            values.append((True, context.run(isolated_operation)))
        except BaseException as error:
            values.append((False, error))
        finally:
            _sdk_slots.release()
            done.set()

    try:
        Thread(target=invoke, name="qa-agent-cancellable-sdk", daemon=True).start()
    except BaseException:
        _sdk_slots.release()
        raise
    while not done.wait(0.05):
        token.check()
    token.check()
    ok, value = values[0]
    if not ok:
        raise value
    return value
