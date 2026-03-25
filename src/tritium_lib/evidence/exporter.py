# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Evidence exporter — produce export packages from evidence collections.

Generates structured export packages containing a JSON manifest,
serialized evidence items, and custody chain records.  The actual
ZIP file creation is delegated to callers; this module produces the
in-memory data structures and file entries needed for packaging.

No direct file I/O — returns dicts and byte buffers that callers
can write to disk, send over HTTP, or buffer in memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .chain import CustodyAction, EvidenceChain
from .collection import EvidenceCollection
from .integrity import compute_sha256
from .models import Evidence


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use as a filename component.

    Prevents path traversal by stripping directory separators and
    parent-directory references. Only allows alphanumeric characters,
    hyphens, underscores, and dots. Raises ValueError if the result
    is empty or resolves to a reserved name.
    """
    import re
    # Strip any directory components — only keep the basename
    name = name.replace("\\", "/")
    name = name.split("/")[-1]
    # Remove parent-directory references
    name = name.replace("..", "")
    # Allow only safe characters: alphanumeric, hyphen, underscore, dot
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_.")
    if not name:
        raise ValueError("Sanitized filename is empty — invalid evidence ID")
    return name


class ExportEntry(object):
    """A single file entry in an export package.

    Attributes:
        filename: Relative path within the package.
        content: Raw bytes of the entry content.
        sha256: SHA-256 hash of the content.
    """

    __slots__ = ("filename", "content", "sha256")

    def __init__(self, filename: str, content: bytes, sha256: str = "") -> None:
        self.filename = filename
        self.content = content
        self.sha256 = sha256


class EvidenceExporter:
    """Export evidence collections as structured packages.

    Produces a list of ExportEntry objects representing the files
    that would go into a ZIP archive.  Callers handle actual I/O.

    Usage::

        exporter = EvidenceExporter()
        entries = exporter.export_collection(collection, actor="analyst")
        # entries is a list of ExportEntry with .filename and .content

        # To create a ZIP:
        import zipfile, io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for entry in entries:
                zf.writestr(entry.filename, entry.content)
    """

    def export_collection(
        self,
        collection: EvidenceCollection,
        actor: str = "system",
        include_chains: bool = True,
    ) -> list[ExportEntry]:
        """Export an entire evidence collection.

        Produces export entries for:
        - manifest.json — collection overview and evidence index
        - evidence/{id}.json — each evidence item
        - chains/{id}.json — each custody chain (if include_chains)
        - package_hash.json — integrity hash of the entire manifest

        Args:
            collection: Evidence collection to export.
            actor: Who is performing the export.
            include_chains: Whether to include custody chain records.

        Returns:
            List of ExportEntry objects.
        """
        entries: list[ExportEntry] = []
        exported_at = datetime.now(timezone.utc).isoformat()

        # Export individual evidence items
        evidence_hashes: dict[str, str] = {}
        for eid, ev in collection.evidence.items():
            safe_eid = _sanitize_filename(eid)
            ev_data = ev.model_dump(mode="json")
            ev_json = json.dumps(ev_data, indent=2, sort_keys=True, default=str)
            ev_bytes = ev_json.encode("utf-8")
            ev_hash = compute_sha256(ev_data)
            evidence_hashes[eid] = ev_hash
            entries.append(ExportEntry(
                filename=f"evidence/{safe_eid}.json",
                content=ev_bytes,
                sha256=ev_hash,
            ))

            # Record export in chain if present
            chain = collection.chains.get(eid)
            if chain:
                chain.record_export(actor=actor, details=f"Exported at {exported_at}")

        # Export custody chains
        if include_chains:
            for eid, chain in collection.chains.items():
                safe_eid = _sanitize_filename(eid)
                chain_data = chain.model_dump(mode="json")
                chain_json = json.dumps(chain_data, indent=2, sort_keys=True, default=str)
                chain_bytes = chain_json.encode("utf-8")
                entries.append(ExportEntry(
                    filename=f"chains/{safe_eid}.json",
                    content=chain_bytes,
                    sha256=compute_sha256(chain_data),
                ))

        # Build manifest
        manifest = collection.to_manifest()
        manifest["exported_at"] = exported_at
        manifest["exported_by"] = actor
        manifest["evidence_hashes"] = evidence_hashes
        manifest_json = json.dumps(manifest, indent=2, sort_keys=True, default=str)
        manifest_bytes = manifest_json.encode("utf-8")
        manifest_hash = compute_sha256(manifest)

        entries.append(ExportEntry(
            filename="manifest.json",
            content=manifest_bytes,
            sha256=manifest_hash,
        ))

        # Package hash — hash of the manifest hash for top-level verification
        package_meta = {
            "manifest_sha256": manifest_hash,
            "evidence_count": len(evidence_hashes),
            "exported_at": exported_at,
            "exported_by": actor,
        }
        package_json = json.dumps(package_meta, indent=2, sort_keys=True)
        entries.append(ExportEntry(
            filename="package_hash.json",
            content=package_json.encode("utf-8"),
            sha256=compute_sha256(package_meta),
        ))

        return entries

    def export_single(
        self,
        evidence: Evidence,
        chain: EvidenceChain | None = None,
        actor: str = "system",
    ) -> list[ExportEntry]:
        """Export a single evidence item with optional custody chain.

        Args:
            evidence: Evidence item to export.
            chain: Optional custody chain.
            actor: Who is exporting.

        Returns:
            List of ExportEntry objects.
        """
        entries: list[ExportEntry] = []
        safe_eid = _sanitize_filename(evidence.evidence_id)

        ev_data = evidence.model_dump(mode="json")
        ev_json = json.dumps(ev_data, indent=2, sort_keys=True, default=str)
        ev_bytes = ev_json.encode("utf-8")
        ev_hash = compute_sha256(ev_data)

        entries.append(ExportEntry(
            filename=f"evidence/{safe_eid}.json",
            content=ev_bytes,
            sha256=ev_hash,
        ))

        if chain:
            chain.record_export(actor=actor, details="Single evidence export")
            chain_data = chain.model_dump(mode="json")
            chain_json = json.dumps(chain_data, indent=2, sort_keys=True, default=str)
            entries.append(ExportEntry(
                filename=f"chains/{safe_eid}.json",
                content=chain_json.encode("utf-8"),
                sha256=compute_sha256(chain_data),
            ))

        return entries
