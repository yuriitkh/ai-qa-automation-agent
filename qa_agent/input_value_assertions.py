"""Bind exact input assertions to declared test data, never generated output text."""
import re


def _control(discovery, selector):
    # Local import avoids the coverage/context dependency cycle.
    from qa_agent.generation_context import observed_controls
    return observed_controls(discovery).get(selector) if discovery is not None else None


def _target_named(text, control):
    labels = (control.accessible_name, control.name, control.id, control.input_type)
    return any(label and len(label) > 2 and re.search(
        rf'(?<!\w){re.escape(label)}(?!\w)', text, re.I) for label in labels)


def input_value_subject_matches(clause, plan, index, discovery):
    action = plan.steps[index]
    control = _control(discovery, action.parameters.get('selector'))
    if control is None or not _target_named(clause, control):
        return False
    quoted = re.findall(r'["\']([^"\']*)["\']', clause)
    return not quoted or action.parameters.get('expected') in quoted


def input_value_is_grounded(plan, index, discovery, requirements):
    assertion = plan.steps[index]
    selector, value = assertion.parameters.get('selector'), assertion.parameters.get('expected')
    control = _control(discovery, selector)
    if control is None or not isinstance(value, str):
        return False
    # A generated fill is not evidence on its own. It must use a literal in
    # an explicit input instruction naming this observed control.
    prior = [action for action in plan.steps[:index]
             if action.parameters.get('selector') == selector and not action.action.startswith('assert_')]
    if not prior or prior[-1].action != 'fill' or prior[-1].parameters.get('value') != value:
        return False
    from qa_agent.generation_context import observed_controls
    other_inputs = [item for item in observed_controls(discovery).values()
                    if item.selector != selector and item.tag in {'input', 'textarea'}]
    for text in requirements:
        for clause in re.split(r'[;\n]', text):
            if not re.search(r'\b(?:enter|fill|type|input|set)\b', clause, re.I) or not _target_named(clause, control):
                continue
            literals = re.findall(r'["\']([^"\']*)["\']', clause)
            if re.search(r'\b(?:not|never|avoid|exclude|instead\s+of)\b', clause, re.I):
                if value in literals:
                    return False
                continue
            if any(_target_named(clause, item) for item in other_inputs):
                continue  # Multiple named inputs make the literal binding ambiguous.
            # Examples may select test input. They cannot require a generated
            # message/status or upgrade any other exact assertion type.
            if literals:
                # Original context precedes step elaboration. A different
                # declared literal cannot be replaced by a generated example.
                return value in literals
    return False
