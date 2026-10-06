"""Formatting helpers for stable, human-readable public identifiers."""

import re


_FORMATS = {
    "TC-": re.compile(r"TC-(\d{4,})\Z"),
    "RUN-": re.compile(r"RUN-(\d{6,})\Z"),
}


def format_test_case_public_id(sequence: int) -> str:
    return _format("TC-", sequence, 4)


def format_run_public_id(sequence: int) -> str:
    return _format("RUN-", sequence, 6)


def parse_test_case_public_id(value: str | None) -> int | None:
    return _parse("TC-", value)


def parse_run_public_id(value: str | None) -> int | None:
    return _parse("RUN-", value)


def _format(prefix: str, sequence: int, minimum_width: int) -> str:
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("Public ID sequence must be a positive integer.")
    return f"{prefix}{sequence:0{minimum_width}d}"


def _parse(prefix: str, value: str | None) -> int | None:
    if not isinstance(value, str):
        return None
    match = _FORMATS[prefix].fullmatch(value)
    if match is None:
        return None
    sequence = int(match.group(1))
    return sequence if sequence > 0 else None
