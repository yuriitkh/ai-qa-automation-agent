"""Validated manual creation and segment-safe TestCase editing helpers."""

from __future__ import annotations

from typing import Mapping
from uuid import uuid4

from pydantic import ValidationError

from qa_agent.models import ExecutionSegment, Precondition, TestCase, TestStep


class TestCaseEditError(ValueError):
    pass


def create_manual_test_case(fields: Mapping[str, str]) -> TestCase:
    name = _required(fields.get("name", ""), "TestCase name", 200)
    description = _required(fields.get("description", ""), "Scenario", 6000)
    base_url = _optional(fields.get("base_url", ""), "Base URL", 2048)
    preconditions = _preconditions(fields.get("preconditions", ""), ())
    steps = []
    for index in range(8):
        title = fields.get(f"step_name_{index}", "").strip()
        action = fields.get(f"step_action_{index}", "").strip()
        expected = fields.get(f"step_expected_{index}", "").strip()
        if not (title or action or expected):
            continue
        if not (title and action and expected):
            raise TestCaseEditError(f"Complete the title, action, and expected result for Step {index + 1}.")
        steps.append(TestStep(
            name=_required(title, f"Step {index + 1} title", 200),
            description=_required(action, f"Step {index + 1} action", 6000),
            expected=_required(expected, f"Step {index + 1} expected result", 6000),
            order=len(steps),
        ))
    if not steps:
        raise TestCaseEditError("Add at least one complete TestStep.")
    return TestCase(
        name=name,
        description=description,
        base_url=base_url or None,
        preconditions=preconditions,
        segments=[ExecutionSegment(
            order=0,
            base_url=base_url or None,
            is_implicit=True,
            steps=steps,
        )],
    )


def edit_test_case(
    test_case: TestCase,
    fields: Mapping[str, str],
    operation: str = "save",
) -> TestCase:
    name = _required(fields.get("name", test_case.name), "TestCase name", 200)
    description = _required(fields.get("description", test_case.description), "Scenario", 6000)
    base_url = _optional(fields.get("base_url", test_case.base_url or ""), "Base URL", 2048)
    preconditions = _preconditions(fields.get("preconditions", "\n".join(
        item.description for item in test_case.preconditions
    )), test_case.preconditions)
    segments = []
    for segment_index, segment in enumerate(test_case.segments):
        steps = []
        for step_index, step in enumerate(segment.steps):
            prefix = f"step_{segment_index}_{step_index}_"
            steps.append(step.model_copy(update={
                "name": _required(fields.get(prefix + "name", step.name), "Step title", 200),
                "description": _required(fields.get(prefix + "action", step.description), "Step action", 6000),
                "expected": _required(fields.get(prefix + "expected", step.expected), "Expected result", 6000),
            }))
        segments.append(segment.model_copy(update={"steps": steps}))

    if segments and any(segment.is_implicit for segment in segments):
        only = segments[0]
        if only.is_implicit and (only.base_url is None or only.base_url == test_case.base_url):
            segments[0] = only.model_copy(update={"base_url": base_url or None})

    if operation != "save":
        parts = operation.split(":")
        if len(parts) == 2 and parts[0] == "add":
            segment_index = _index(parts[1], len(segments), "segment")
            segment = segments[segment_index]
            new_step = TestStep(
                name=f"Step {sum(len(item.steps) for item in segments) + 1}",
                description="Describe the action for this step.",
                expected="Describe the expected result.",
                order=len(segment.steps),
            )
            segments[segment_index] = segment.model_copy(update={"steps": [*segment.steps, new_step]})
        elif len(parts) == 3 and parts[0] in {"duplicate", "delete", "up", "down"}:
            segment_index = _index(parts[1], len(segments), "segment")
            segment = segments[segment_index]
            step_index = _index(parts[2], len(segment.steps), "step")
            steps = list(segment.steps)
            if parts[0] == "duplicate":
                source = steps[step_index]
                copied = source.model_copy(update={"id": uuid4(), "name": f"{source.name} (copy)"})
                steps.insert(step_index + 1, copied)
            elif parts[0] == "delete":
                if len(steps) == 1:
                    raise TestCaseEditError("A segment must keep at least one step. Add another step before deleting this one.")
                steps.pop(step_index)
            else:
                target = step_index - 1 if parts[0] == "up" else step_index + 1
                if target < 0 or target >= len(steps):
                    raise TestCaseEditError("That step cannot move any farther in this segment.")
                steps[step_index], steps[target] = steps[target], steps[step_index]
            segments[segment_index] = segment.model_copy(update={"steps": steps})
        else:
            raise TestCaseEditError("Choose a valid TestStep operation.")

    next_order = 0
    ordered_segments = []
    for segment in segments:
        ordered_steps = []
        for step in segment.steps:
            ordered_steps.append(step.model_copy(update={"order": next_order}))
            next_order += 1
        ordered_segments.append(segment.model_copy(update={"steps": ordered_steps}))
    data = test_case.model_dump(exclude={"steps"})
    data.update({
        "name": name,
        "description": description,
        "base_url": base_url or None,
        "preconditions": preconditions,
        "segments": ordered_segments,
    })
    try:
        return TestCase.model_validate(data)
    except (ValidationError, TypeError, ValueError) as error:
        raise TestCaseEditError("The edited TestCase could not be validated.") from error


def _preconditions(text: str, existing: tuple[Precondition, ...] | list[Precondition]) -> list[Precondition]:
    descriptions = [line.strip() for line in text.splitlines() if line.strip()]
    if len(descriptions) > 30:
        raise TestCaseEditError("Use 30 or fewer preconditions.")
    if any(len(value) > 6000 for value in descriptions):
        raise TestCaseEditError("Each precondition must be 6,000 characters or fewer.")
    return [Precondition(
        id=existing[index].id if index < len(existing) else uuid4(),
        description=value,
        order=index,
        provided_data_keys=existing[index].provided_data_keys if index < len(existing) else [],
    ) for index, value in enumerate(descriptions)]


def _required(value: str, label: str, maximum: int) -> str:
    clean = value.strip()
    if not clean:
        raise TestCaseEditError(f"{label} is required.")
    if len(clean) > maximum:
        raise TestCaseEditError(f"{label} must be {maximum:,} characters or fewer.")
    return clean


def _optional(value: str, label: str, maximum: int) -> str:
    clean = value.strip()
    if len(clean) > maximum:
        raise TestCaseEditError(f"{label} must be {maximum:,} characters or fewer.")
    return clean


def _index(value: str, size: int, kind: str) -> int:
    try:
        index = int(value)
    except ValueError as error:
        raise TestCaseEditError(f"Invalid {kind} selection.") from error
    if index < 0 or index >= size:
        raise TestCaseEditError(f"Invalid {kind} selection.")
    return index
