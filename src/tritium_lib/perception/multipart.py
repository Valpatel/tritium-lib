# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Reader for ``multipart/x-mixed-replace`` streams — the MJPEG transport.

Every camera stream in Tritium is served as ``multipart/x-mixed-replace``, and
until now every *consumer* read it by scanning for JPEG start/end markers
(``\\xff\\xd8`` .. ``\\xff\\xd9``).  That shortcut has two defects:

1. **It is JPEG-only.**  Isaac's metric-depth channel (``/depth16``) carries
   lossless 16-bit *PNG*, because a colormapped JPEG destroys range.  A marker
   scanner cannot see a PNG at all.
2. **It can truncate a JPEG.**  Those two byte pairs occur inside
   entropy-coded scan data, so the "end" it finds may be mid-frame.

So parse the transport as defined: split on the boundary, honour
``Content-Length`` when the server sends one, and fall back to a boundary scan
when it does not.  Payload bytes are then read by *length*, never by content
inspection, so a frame whose bytes happen to spell the boundary survives.

Format handled (RFC 2046 in the shape browsers and every MJPEG server use)::

    --boundary\\r\\n
    Content-Type: image/png\\r\\n
    Content-Length: 1234\\r\\n
    \\r\\n
    <1234 bytes>\\r\\n
    --boundary\\r\\n
    ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional

__all__ = ["MultipartPart", "boundary_from_content_type", "iter_multipart"]

#: Read granularity off the socket. Large enough that a 640x480 JPEG lands in a
#: handful of reads, small enough not to stall on a slow feed.
_CHUNK = 65536

#: Default ceiling for a single part. A camera frame is ~10-200 KB; 64 MB is far
#: above any real frame but still bounds a server that opens a part and never
#: closes it, which would otherwise grow the buffer without limit.
_MAX_PART_BYTES = 64 * 1024 * 1024

_BOUNDARY_RE = re.compile(r"boundary=(?:\"([^\"]+)\"|([^;\s]+))", re.IGNORECASE)


@dataclass(frozen=True)
class MultipartPart:
    """One frame lifted off the stream."""

    payload: bytes
    content_type: str = ""
    headers: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.headers is None:
            object.__setattr__(self, "headers", {})


def boundary_from_content_type(content_type: str) -> Optional[bytes]:
    """Extract the boundary token from a Content-Type header.

    Returns None when the header carries no boundary (i.e. it is not a
    multipart response), so a caller can fail loudly rather than guess a
    boundary and silently read garbage.
    """
    if not content_type:
        return None
    m = _BOUNDARY_RE.search(content_type)
    if not m:
        return None
    return (m.group(1) or m.group(2)).encode("latin-1")


def _read_chunk(stream) -> bytes:
    try:
        return stream.read(_CHUNK) or b""
    except (OSError, ValueError):
        # Socket closed mid-read is a normal end-of-stream for a live feed.
        return b""


def iter_multipart(
    stream,
    boundary: bytes,
    *,
    max_part_bytes: int = _MAX_PART_BYTES,
) -> Iterator[MultipartPart]:
    """Yield each part of a multipart stream until the stream ends.

    Args:
        stream: any file-like with ``.read(n)`` — an ``http.client.HTTPResponse``,
            a socket file, or a ``BytesIO`` in tests.
        boundary: the boundary token WITHOUT the leading ``--``.
        max_part_bytes: raise ``ValueError`` rather than buffer beyond this.

    Yields:
        MultipartPart, in stream order.

    Raises:
        ValueError: a single part exceeded ``max_part_bytes``.
    """
    delim = b"--" + boundary
    buf = b""
    # Skip any preamble: position at the first boundary.
    while True:
        idx = buf.find(delim)
        if idx != -1:
            buf = buf[idx:]
            break
        if len(buf) > max_part_bytes:
            raise ValueError(f"preamble exceeds {max_part_bytes} bytes")
        chunk = _read_chunk(stream)
        if not chunk:
            return
        # Keep a tail so a boundary split across reads is still found.
        buf = buf[-len(delim):] + chunk if len(buf) > len(delim) else buf + chunk

    while True:
        # buf starts at a boundary. Consume it and any terminator/CRLF.
        if not buf.startswith(delim):
            idx = buf.find(delim)
            if idx == -1:
                chunk = _read_chunk(stream)
                if not chunk:
                    return
                buf += chunk
                continue
            buf = buf[idx:]
        rest = buf[len(delim):]
        if rest.startswith(b"--"):
            return  # closing boundary "--boundary--"
        if not rest:
            chunk = _read_chunk(stream)
            if not chunk:
                return
            buf += chunk
            continue
        buf = rest.lstrip(b"\r\n") if rest[:1] in (b"\r", b"\n") else rest

        # Read headers up to the blank line.
        while b"\r\n\r\n" not in buf:
            if len(buf) > max_part_bytes:
                raise ValueError(f"headers exceed {max_part_bytes} bytes")
            chunk = _read_chunk(stream)
            if not chunk:
                return
            buf += chunk
        head, buf = buf.split(b"\r\n\r\n", 1)
        headers: dict[str, str] = {}
        for line in head.split(b"\r\n"):
            if b":" in line:
                k, _, v = line.partition(b":")
                headers[k.strip().decode("latin-1").lower()] = v.strip().decode("latin-1")

        length = headers.get("content-length")
        if length is not None and length.isdigit():
            # Length-delimited: read exactly N bytes. Payload bytes that spell
            # the boundary are therefore harmless.
            n = int(length)
            if n > max_part_bytes:
                raise ValueError(f"part of {n} bytes exceeds {max_part_bytes}")
            while len(buf) < n:
                chunk = _read_chunk(stream)
                if not chunk:
                    return
                buf += chunk
            payload, buf = buf[:n], buf[n:]
        else:
            # No Content-Length: scan forward to the next boundary.
            while True:
                idx = buf.find(delim)
                if idx != -1:
                    break
                if len(buf) > max_part_bytes:
                    raise ValueError(
                        f"unterminated part exceeds {max_part_bytes} bytes")
                chunk = _read_chunk(stream)
                if not chunk:
                    return
                buf += chunk
            payload, buf = buf[:idx], buf[idx:]
            payload = payload[:-2] if payload.endswith(b"\r\n") else payload

        yield MultipartPart(
            payload=payload,
            content_type=headers.get("content-type", ""),
            headers=headers,
        )

        # Position at the next boundary.
        while True:
            idx = buf.find(delim)
            if idx != -1:
                buf = buf[idx:]
                break
            if len(buf) > max_part_bytes:
                raise ValueError(f"inter-part gap exceeds {max_part_bytes} bytes")
            chunk = _read_chunk(stream)
            if not chunk:
                return
            buf += chunk
