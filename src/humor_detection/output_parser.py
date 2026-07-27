from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParseResult:

    label: int
    parser_failure: bool
    normalized: str


def parse_binary_output(raw_response: str | None) -> ParseResult:
    normalized = "" if raw_response is None else raw_response.strip()
    if normalized == "0":
        return ParseResult(label=0, parser_failure=False, normalized=normalized)
    if normalized == "1":
        return ParseResult(label=1, parser_failure=False, normalized=normalized)
    return ParseResult(label=0, parser_failure=True, normalized=normalized)
