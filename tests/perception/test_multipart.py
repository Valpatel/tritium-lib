# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the multipart/x-mixed-replace stream reader.

Why this exists: every MJPEG-ish consumer in the tree used to scan for JPEG
SOI/EOI markers (``\\xff\\xd8`` .. ``\\xff\\xd9``).  That is a JPEG-only trick —
it cannot carry a PNG, and a 16-bit depth PNG is exactly what Isaac's
``/depth16`` channel serves.  Worse, the marker scan can also cut a JPEG early
because those byte pairs occur inside entropy-coded scan data.  The fix is to
parse the transport the way the transport is actually defined: by boundary.
"""

from __future__ import annotations

import io

import pytest

from tritium_lib.perception.multipart import (
    boundary_from_content_type,
    iter_multipart,
)


def _part(payload: bytes, mime: bytes = b"image/jpeg", boundary: bytes = b"frame") -> bytes:
    return (
        b"--" + boundary + b"\r\n"
        b"Content-Type: " + mime + b"\r\n"
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
        b"\r\n" + payload + b"\r\n"
    )


class TestBoundaryFromContentType:
    def test_parses_boundary(self):
        assert boundary_from_content_type(
            "multipart/x-mixed-replace; boundary=frame") == b"frame"

    def test_quoted_boundary(self):
        assert boundary_from_content_type(
            'multipart/x-mixed-replace; boundary="--myBoundary"') == b"--myBoundary"

    def test_missing_boundary_returns_none(self):
        assert boundary_from_content_type("image/jpeg") is None


class TestIterMultipart:
    def test_yields_payloads_in_order(self):
        stream = io.BytesIO(_part(b"AAA") + _part(b"BB") + _part(b"C"))
        got = [p.payload for p in iter_multipart(stream, b"frame")]
        assert got == [b"AAA", b"BB", b"C"]

    def test_exposes_content_type(self):
        stream = io.BytesIO(_part(b"x", mime=b"image/png"))
        parts = list(iter_multipart(stream, b"frame"))
        assert parts[0].content_type == "image/png"

    def test_payload_containing_boundary_like_bytes_survives(self):
        """A Content-Length part must be read by LENGTH, not by scanning.

        This is the case a naive scanner corrupts: binary payload bytes that
        happen to spell the boundary. PNG/JPEG entropy data does this.
        """
        evil = b"\x89PNG--frame\r\nnot a real boundary\r\n\xff\xd9tail"
        stream = io.BytesIO(_part(evil) + _part(b"next"))
        got = [p.payload for p in iter_multipart(stream, b"frame")]
        assert got == [evil, b"next"]

    def test_png_payload_round_trips(self):
        """The whole point: a PNG must survive, JPEG markers notwithstanding."""
        png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
        stream = io.BytesIO(_part(png, mime=b"image/png"))
        assert list(iter_multipart(stream, b"frame"))[0].payload == png

    def test_handles_chunk_split_across_reads(self):
        """Parts arriving in tiny TCP-sized chunks must reassemble."""
        raw = _part(b"HELLO") + _part(b"WORLD")

        class Trickle(io.RawIOBase):
            def __init__(self, data):
                self._d, self._i = data, 0

            def read(self, n=-1):
                chunk = self._d[self._i:self._i + 3]  # 3 bytes at a time
                self._i += len(chunk)
                return chunk

        got = [p.payload for p in iter_multipart(Trickle(raw), b"frame")]
        assert got == [b"HELLO", b"WORLD"]

    def test_no_content_length_falls_back_to_boundary_scan(self):
        """Some servers omit Content-Length; scan to the next boundary."""
        raw = (b"--frame\r\nContent-Type: image/jpeg\r\n\r\nBODY\r\n"
               b"--frame\r\nContent-Type: image/jpeg\r\n\r\nBODY2\r\n--frame--\r\n")
        got = [p.payload for p in iter_multipart(io.BytesIO(raw), b"frame")]
        assert got == [b"BODY", b"BODY2"]

    def test_stops_cleanly_at_end_of_stream(self):
        assert list(iter_multipart(io.BytesIO(b""), b"frame")) == []

    def test_ignores_preamble_before_first_boundary(self):
        stream = io.BytesIO(b"junk preamble\r\n" + _part(b"A"))
        assert [p.payload for p in iter_multipart(stream, b"frame")] == [b"A"]

    def test_max_part_bytes_guards_runaway(self):
        """A server that never closes a part must not exhaust memory."""
        raw = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"x" * 10_000
        with pytest.raises(ValueError, match="exceeds"):
            list(iter_multipart(io.BytesIO(raw), b"frame", max_part_bytes=1024))
