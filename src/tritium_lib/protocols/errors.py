# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared error types for protocol parsers."""


class ParseError(Exception):
    """Raised when a protocol message cannot be parsed.

    Attributes:
        protocol: Which protocol parser raised the error.
        raw_data: The raw input that failed to parse.
        reason:   Human-readable explanation.
    """

    def __init__(self, protocol: str, reason: str, raw_data: object = None):
        self.protocol = protocol
        self.reason = reason
        self.raw_data = raw_data
        super().__init__(f"[{protocol}] {reason}")
