"""Validated manual creation and segment-safe TestCase editing helpers."""

from __future__ import annotations

from typing import Mapping
from uuid import uuid4
import json

from pydantic import ValidationError

from qa_agent.models import ExecutionSegment, Precondition, TestCase, TestStep
from qa_agent.test_case_naming import fallback_summary, step_display_name


class TestCaseEditError(ValueError):
    pass


def create_manual_test_case(fields: Mapping[str, str]) -> TestCase:
    description = _required(fields.get("description", ""), "Scenario", 6000)
    name = fields.get("name", "")
    base_url = _optional(fields.get("base_url", ""), "Base URL", 2048)
    preconditions = _preconditions(fields.get("preconditions", ""), ())
    steps = []
    for index in range(8):
        title = fields.get(f"step_name_{index}", "").strip()
        action = fields.get(f"step_action_{index}", "").strip()
        expected = fields.get(f"step_expected_{index}", "").strip()
        if not (title or action or expected):
            continue
        steps.append(TestStep(
            name=_required(title or step_display_name(action), f"Step {index + 1} name", 200),
            description=_optional(action, f"Step {index + 1} Description", 6000) or " ",
            expected=_optional(expected, f"Step {index + 1} Expected Result", 6000) or " ",
            order=len(steps),
        ))
    if not steps:
        raise TestCaseEditError("Add at least one TestStep. Unfinished fields can be completed before approval.")
    name = _summary(name, description, [step.description for step in steps])
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
    name = fields.get("name", test_case.name)
    description = _required(fields.get("description", test_case.description), "Scenario", 6000)
    base_url = _optional(fields.get("base_url", test_case.base_url or ""), "Base URL", 2048)
    preconditions = _preconditions(fields.get("preconditions", "\n".join(
        item.description for item in test_case.preconditions
    )), test_case.preconditions)
    if "steps_json" in fields:
        segments = structured_steps(test_case, fields["steps_json"])
        if len(segments) == 1 and segments[0].is_implicit and segments[0].base_url == test_case.base_url:
            segments[0] = segments[0].model_copy(update={"base_url": base_url or None})
        name = _summary(name, description, [step.description for segment in segments for step in segment.steps])
        data = test_case.model_dump(exclude={"steps"})
        data.update(name=name, description=description, base_url=base_url or None,
                    preconditions=preconditions, segments=segments)
        return TestCase.model_validate(data)
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
    name = _summary(name, description, [step.description for segment in ordered_segments for step in segment.steps])
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
    if any(item.provided_data_keys for item in existing) and descriptions != [item.description for item in existing]:
        raise TestCaseEditError("These preconditions carry setup data contracts. Keep them unchanged in this editor.")
    matched = []
    used = set()
    for value in descriptions:
        item = next((item for item in existing if item.description == value and item.id not in used), None)
        matched.append(item)
        if item: used.add(item.id)
    for index, item in enumerate(matched):
        if item is None and index < len(existing) and existing[index].id not in used:
            matched[index] = existing[index]
            used.add(existing[index].id)
    return [Precondition(
        id=matched[index].id if matched[index] is not None else uuid4(),
        description=value,
        order=index,
        provided_data_keys=matched[index].provided_data_keys if matched[index] is not None else [],
    ) for index, value in enumerate(descriptions)]


def structured_steps(test_case: TestCase, raw: str) -> list[ExecutionSegment]:
    """Merge UI ordering into trusted segments; retain metadata on existing steps."""
    try:
        submitted = json.loads(raw)
    except (TypeError, ValueError, RecursionError) as error:
        raise TestCaseEditError("The step list is invalid. Review it before saving.") from error
    if not isinstance(submitted, list) or len(submitted) != len(test_case.segments):
        raise TestCaseEditError("Keep every execution segment; segments cannot be removed or flattened.")
    seen = set()
    result = []
    order = 0
    for segment_index, (segment, incoming) in enumerate(zip(test_case.segments, submitted)):
        if not isinstance(incoming, dict) or set(incoming) != {"id", "steps"} or incoming["id"] != str(segment.id):
            raise TestCaseEditError("Segment membership changed. Reload the editor and retry.")
        rows = incoming["steps"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 200:
            raise TestCaseEditError(f"Segment {segment_index + 1}: keep between 1 and 200 steps.")
        known = {str(step.id): step for step in segment.steps}
        steps = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "description", "expected"}:
                raise TestCaseEditError(f"Step {order + 1}: unexpected step fields.")
            step_id = row["id"]
            if not isinstance(step_id, str) or step_id in seen:
                raise TestCaseEditError(f"Step {order + 1}: invalid or duplicate step ID.")
            seen.add(step_id)
            old = known.get(step_id)
            if old is None and not re_new_step_id(step_id):
                raise TestCaseEditError(f"Step {order + 1}: the step does not belong to this segment.")
            if not all(isinstance(row[key], str) for key in ("description", "expected")):
                raise TestCaseEditError(f"Step {order + 1}: use text for Description and Expected Result.")
            action = _optional(row["description"], f"Step {order + 1} Description", 6000)
            expected = _optional(row["expected"], f"Step {order + 1} Expected Result", 6000)
            name = old.name if old is not None and " ".join(action.split()) == " ".join(old.description.split()) else step_display_name(action)
            updates = {"name": name, "description": action or " ", "expected": expected or " ", "order": order}
            steps.append(old.model_copy(update=updates) if old is not None else TestStep(**updates))
            order += 1
        result.append(segment.model_copy(update={"order": segment_index, "steps": steps}))
    return result


def re_new_step_id(value):
    from uuid import UUID
    try:
        return value.startswith("new:") and str(UUID(value[4:])) == value[4:]
    except ValueError:
        return False


def _summary(value: str, scenario: str, actions: list[str]) -> str:
    try:
        return _required(value.strip() or fallback_summary(scenario, actions), "Summary", 200)
    except ValueError as error:
        raise TestCaseEditError(str(error)) from error


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
