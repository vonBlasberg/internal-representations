"""Strict parsing for the declared one-step negotiation output envelope."""
from __future__ import annotations

import json
from typing import Any


class OutputError(ValueError):
    """A completed response does not satisfy the declared output contract."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OutputError("DUPLICATE_JSON_KEY", key)
        result[key] = value
    return result


def parse_negotiation_output(raw: str, output_contract_version: int = 1) -> dict[str, Any]:
    if output_contract_version not in {1, 2}:
        raise ValueError("Unsupported negotiation output contract version")
    if not isinstance(raw, str):
        raise OutputError("NON_STRING_OUTPUT", "The completed response must be text.")
    if not raw.strip():
        raise OutputError("EMPTY_OUTPUT", "No JSON object was returned.")
    try:
        decoder = json.JSONDecoder(object_pairs_hook=_object_without_duplicates)
        value, end = decoder.raw_decode(raw.lstrip())
    except OutputError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputError("MALFORMED_JSON", str(exc)) from exc
    if raw.lstrip()[end:].strip():
        raise OutputError("TEXT_OUTSIDE_ENVELOPE", "Only one JSON object is permitted.")
    if not isinstance(value, dict):
        raise OutputError("NON_OBJECT_OUTPUT", "The response must be a JSON object.")
    required = ({"decision", "package", "claimed_self_payoff", "rationale"}
                if output_contract_version == 1 else {"decision", "package"})
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise OutputError("OUTPUT_FIELDS", f"missing={missing}; extra={extra}")
    if value["decision"] not in {"offer", "accept"}:
        raise OutputError("DECISION", "Expected 'offer' or 'accept'.")
    if not isinstance(value["package"], dict):
        raise OutputError("PACKAGE", "Expected one complete package object.")
    if output_contract_version == 1:
        if isinstance(value["claimed_self_payoff"], bool) or not isinstance(value["claimed_self_payoff"], int):
            raise OutputError("PAYOFF_CLAIM", "claimed_self_payoff must be an integer.")
        if not isinstance(value["rationale"], str) or not value["rationale"].strip():
            raise OutputError("RATIONALE", "A short nonempty rationale is required.")
        if len(value["rationale"]) > 600:
            raise OutputError("RATIONALE_LENGTH", "The rationale exceeds 600 characters.")
    return value
