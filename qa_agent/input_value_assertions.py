"""Bind exact input assertions to declared test data, never generated output text."""
import re


_QUOTED_INPUT = re.compile(r'''(?<!\w)(?:"([^"]*)"|'([^']*)')''')
_INPUT_ACTION = re.compile(
    r"^\s*(?:(?:please|do\s+not|never|avoid|exclude|don't)\s+)*(?:enter|fill|type|input|set)\b", re.I,
)
_INPUT_MENTION = re.compile(r'\b(?:enter(?:s|ed|ing)?|fill(?:s|ed|ing)?|typ(?:e|es|ed|ing)|input|set(?:s|ting)?|use|provide)\b', re.I)
_NEGATIVE = re.compile(r"\b(?:not|never|avoid|exclude|except|instead\s+of|rather\s+than)\b|\bdon't\b", re.I)
_EXAMPLE = re.compile(r'\b(?:such\s+as|for\s+example|e\s*\.\s*g\s*\.)(?=\s)', re.I)
_BARE_INPUT = re.compile(r'[a-z0-9_@+-](?:[a-z0-9_.@+-]*[a-z0-9_@+-])?(?=$|[\s,;)])', re.I)
_INSTRUCTION_BREAK = re.compile(
    r'[;\n]+|[.!?](?=\s|$)|\s+(?:and|but|then|also)\s+'
    r'(?=(?:(?:do\s+not|never|avoid|exclude|don\'t)\s+)?'
    r'(?:enter|fill|type|input|set|use|verify|check|confirm|ensure|assert|submit)\b)', re.I,
)


def _mask_quotes(text):
    # Non-whitespace keeps boundary regexes from consuming a protected literal.
    return _QUOTED_INPUT.sub(lambda match: '\x00' * len(match.group()), text)


def _input_instructions(text):
    """Split instruction boundaries outside literals, preserving addresses and e.g."""
    masked = _mask_quotes(text)
    masked = re.sub(r'\be\s*\.\s*g\s*\.', lambda match: match.group().replace('.', ' '), masked, flags=re.I)
    start = 0
    for match in _INSTRUCTION_BREAK.finditer(masked):
        yield text[start:match.start()].strip()
        start = match.end()
    yield text[start:].strip()


def _input_literals(instruction):
    """Return exact literals and value-free subject text; bare examples are one token."""
    matches = list(_QUOTED_INPUT.finditer(instruction))
    values = {match.group(1) if match.group(1) is not None else match.group(2) for match in matches}
    masked = _mask_quotes(instruction)
    invalid = bool(re.search(r'''(?<!\w)['"]|['"](?!\w)|\bor\b''', masked, re.I))
    for example in _EXAMPLE.finditer(masked):
        start = example.end() + len(instruction[example.end():]) - len(instruction[example.end():].lstrip())
        # A quoted example was already extracted and masked.
        if any(match.start() == start for match in matches):
            continue
        literal = _BARE_INPUT.match(instruction, start)
        if literal is None:
            invalid = True
            continue
        tail = instruction[literal.end():].strip()
        if tail and not (
            re.match(r'(?:into|in|to|for)\b', tail, re.I)
            or re.fullmatch(r'and\s+valid\s+(?:data|information|values?)\b.*\b(?:other|required|remaining)\b.*\bfields?\b', tail, re.I)
        ):
            invalid = True
        values.add(literal.group())
        masked = masked[:literal.start()] + ' ' * len(literal.group()) + masked[literal.end():]
    return values, masked, invalid


def _input_declarations(text, control, other_inputs, *, authoritative=False):
    """Collect all bindings and prohibitions before deciding whether a value is supported."""
    allowed, forbidden = set(), set()
    has_instruction = ambiguous = unbound = False
    for instruction in _input_instructions(text):
        values, subject, invalid = _input_literals(instruction)
        action = _INPUT_ACTION.match(subject)
        negative = bool(_NEGATIVE.search(subject))
        # Also recognize restrictions such as "Email must not contain 'x'".
        restriction = negative and bool(re.search(
            r'\b(?:use|contains?|equals?|enter(?:ing)?|fill(?:ing)?|type|typing|input|set)\b', subject, re.I,
        ))
        value_state = bool(re.search(r'\b(?:field|input)\b.*\b(?:contains?|matches?|equals?)\b', subject, re.I))
        value_state &= not bool(re.search(r'\b(?:error|message|confirmation|notification|alert|warning|notice)\b', subject, re.I))
        if action is None and not restriction:
            # Unsupported input phrasing still prevents atomic elaboration
            # from independently declaring values absent from the original.
            if (not value_state and _INPUT_MENTION.search(subject)) or (
                authoritative and values and _target_named(subject, control)
                and re.search(r'\b(?:contains?|values?|matches?|equals?)\b', subject, re.I)
            ):
                has_instruction = True
                if _target_named(subject, control):
                    unbound = True
                    ambiguous |= bool(values)
            continue
        has_instruction = True
        if not _target_named(subject, control):
            continue
        if invalid or (values and any(_target_named(subject, item) for item in other_inputs)):
            ambiguous = True
        elif not values:
            unbound = True
            ambiguous |= negative  # An uninterpreted prohibition must fail closed.
        elif negative:
            forbidden.update(values)
        elif len(values) != 1:
            ambiguous = True
        else:
            allowed.update(values)
    return allowed, forbidden, ambiguous, unbound, has_instruction


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
    if control is None or not _target_named(_mask_quotes(clause), control):
        return False
    quoted = {match.group(1) if match.group(1) is not None else match.group(2)
              for match in _QUOTED_INPUT.finditer(clause)}
    return not quoted or action.parameters.get('expected') in quoted


def input_value_is_grounded(plan, index, discovery, requirements, *, requirement_context=None,
                            allow_atomic_requirements=False):
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
    allowed, forbidden, ambiguous = set(), set(), False
    for text in requirements:
        declared, prohibited, invalid, _, _ = _input_declarations(text, control, other_inputs)
        allowed.update(declared)
        forbidden.update(prohibited)
        ambiguous |= invalid
    if isinstance(requirement_context, str) and requirement_context.strip():
        original, restrictions, invalid, unspecified, has_instruction = _input_declarations(
            requirement_context, control, other_inputs, authoritative=True,
        )
        if has_instruction:
            # Even an unresolved original input instruction is authoritative.
            # Atomic elaboration cannot supply a missing or conflicting value.
            if invalid or unspecified or original != {value} or value in restrictions:
                return False
            if ambiguous or allowed - original or value in forbidden:
                return False
            return True
        if not allow_atomic_requirements:
            # Related original scenarios cannot gain specificity from generated
            # step wording, even when they contain no parsed input instruction.
            return False
    return not ambiguous and allowed == {value} and value not in forbidden
