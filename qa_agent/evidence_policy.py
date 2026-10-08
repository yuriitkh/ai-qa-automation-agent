"""Per-run screenshot policy and execution identity contexts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Iterator
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EvidenceMode(str, Enum):
    FAILURES_ONLY = "FAILURES_ONLY"
    EVERY_VERIFICATION = "EVERY_VERIFICATION"
    EVERY_STEP = "EVERY_STEP"


class ScreenshotMode(str, Enum):
    ELEMENT = "ELEMENT"
    PAGE = "PAGE"
    ELEMENT_AND_PAGE = "ELEMENT_AND_PAGE"


class EvidenceScope(str, Enum):
    ELEMENT = "ELEMENT"
    PAGE = "PAGE"


class EvidencePolicy(BaseModel):
    """Evidence capture policy persisted with a run.

    The conservative default preserves the existing failure-only page
    screenshot behavior.
    """

    model_config = ConfigDict(frozen=True)

    mode: EvidenceMode = EvidenceMode.FAILURES_ONLY
    screenshot_mode: ScreenshotMode = ScreenshotMode.PAGE


DEFAULT_EVIDENCE_POLICY = EvidencePolicy()


@dataclass(frozen=True)
class EvidenceExecutionIdentity:
    execution_id: UUID
    test_step_id: UUID


_active_policy: ContextVar[EvidencePolicy] = ContextVar(
    "qa_agent_evidence_policy", default=DEFAULT_EVIDENCE_POLICY
)
_active_execution: ContextVar[EvidenceExecutionIdentity | None] = ContextVar(
    "qa_agent_evidence_execution", default=None
)


def current_evidence_policy() -> EvidencePolicy:
    return _active_policy.get()


def current_evidence_execution() -> EvidenceExecutionIdentity | None:
    return _active_execution.get()


@contextmanager
def evidence_policy_scope(policy: EvidencePolicy | None) -> Iterator[None]:
    token: Token[EvidencePolicy] = _active_policy.set(policy or DEFAULT_EVIDENCE_POLICY)
    try:
        yield
    finally:
        _active_policy.reset(token)


@contextmanager
def evidence_execution_scope(identity: EvidenceExecutionIdentity) -> Iterator[None]:
    token: Token[EvidenceExecutionIdentity | None] = _active_execution.set(identity)
    try:
        yield
    finally:
        _active_execution.reset(token)


def evidence_mode_label(mode: EvidenceMode) -> str:
    return {
        EvidenceMode.FAILURES_ONLY: "Failures only",
        EvidenceMode.EVERY_VERIFICATION: "Every verification",
        EvidenceMode.EVERY_STEP: "Every step",
    }[mode]


def screenshot_mode_label(mode: ScreenshotMode) -> str:
    return {
        ScreenshotMode.ELEMENT: "Element",
        ScreenshotMode.PAGE: "Page",
        ScreenshotMode.ELEMENT_AND_PAGE: "Element + Page",
    }[mode]
