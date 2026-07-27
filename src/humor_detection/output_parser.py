from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParseResult:
    """Parsed label and audit information."""

    label: int
    parser_failure: bool
    normalized: str


def parse_binary_output(raw_response: str | None) -> ParseResult:
    """Accept only an exact trimmed 0 or 1; default every other output to 0."""
    normalized = "" if raw_response is None else raw_response.strip()
    if normalized == "0":
        return ParseResult(label=0, parser_failure=False, normalized=normalized)
    if normalized == "1":
        return ParseResult(label=1, parser_failure=False, normalized=normalized)
    return ParseResult(label=0, parser_failure=True, normalized=normalized)
