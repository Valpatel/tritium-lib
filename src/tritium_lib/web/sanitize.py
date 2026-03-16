# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sanitization utilities for web output and file handling.

Provides safe HTML escaping, JSON-safe string conversion, and filename
sanitization. Used by any Python code generating HTML or handling user input.

Usage::

    from tritium_lib.web.sanitize import html_escape, json_safe, sanitize_filename

    safe = html_escape('<script>alert("xss")</script>')
    # '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'

    filename = sanitize_filename('../../etc/passwd')
    # 'etcpasswd'
"""

from __future__ import annotations

import html
import os
import re
import unicodedata


def html_escape(text: str | None) -> str:
    """Escape a string for safe insertion into HTML.

    Handles ``<``, ``>``, ``&``, ``"``, and ``'``.
    Returns empty string for None or empty input.

    Args:
        text: Raw text to escape.

    Returns:
        HTML-safe string.
    """
    if not text:
        return ""
    return html.escape(str(text), quote=True)


def json_safe(value: str | None, max_length: int = 10000) -> str:
    """Make a string safe for embedding in JSON values.

    Escapes backslashes, quotes, and control characters.
    Truncates to *max_length* to prevent oversized payloads.
    Returns empty string for None.

    Args:
        value:      Raw string value.
        max_length: Maximum output length (default 10000).

    Returns:
        JSON-safe string (without surrounding quotes).
    """
    if value is None:
        return ""
    s = str(value)
    if len(s) > max_length:
        s = s[:max_length]
    # Replace control characters with their escape sequences
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    s = s.replace("\b", "\\b")
    s = s.replace("\f", "\\f")
    # Remove remaining control chars (U+0000 to U+001F except already handled)
    s = re.sub(r"[\x00-\x08\x0e-\x1f]", "", s)
    return s


def sanitize_filename(name: str | None, max_length: int = 200) -> str:
    """Sanitize a string for use as a safe filename.

    Removes path separators, null bytes, and other dangerous characters.
    Normalizes Unicode, strips leading/trailing dots and spaces.
    Returns ``'unnamed'`` for empty input.

    Args:
        name:       Raw filename string.
        max_length: Maximum filename length (default 200).

    Returns:
        Safe filename string.
    """
    if not name:
        return "unnamed"
    s = str(name)
    # Normalize Unicode
    s = unicodedata.normalize("NFKD", s)
    # Remove path separators and null bytes
    s = s.replace("/", "").replace("\\", "").replace("\x00", "")
    # Remove other dangerous characters
    s = re.sub(r'[<>:"|?*]', "", s)
    # Strip leading dots (hidden files) and whitespace
    s = s.lstrip(". ").rstrip(". ")
    # Collapse multiple spaces/underscores
    s = re.sub(r"[\s_]+", "_", s)
    # Truncate
    if len(s) > max_length:
        # Preserve extension if present
        base, ext = os.path.splitext(s)
        if ext and len(ext) <= 10:
            s = base[: max_length - len(ext)] + ext
        else:
            s = s[:max_length]
    return s or "unnamed"
