# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BackupManager — create and restore backups of Tritium tracking data.

All operations are local filesystem only. No network or SSH calls.
Backups are stored as directories with a JSON manifest describing contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackupManifest:
    """Metadata about a backup archive.

    Attributes
    ----------
    backup_id:
        Unique identifier for this backup.
    name:
        Human-readable backup name.
    source_dir:
        Directory that was backed up.
    backup_path:
        Where the backup is stored.
    created_at:
        Timestamp when the backup was created.
    file_count:
        Number of files in the backup.
    total_bytes:
        Total size of backed-up files in bytes.
    checksum:
        SHA-256 checksum of the manifest for integrity.
    tags:
        Arbitrary metadata tags.
    files:
        List of relative file paths included in the backup.
    """

    backup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_dir: str = ""
    backup_path: str = ""
    created_at: float = field(default_factory=time.time)
    file_count: int = 0
    total_bytes: int = 0
    checksum: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        """Total backup size in megabytes."""
        return self.total_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "backup_id": self.backup_id,
            "name": self.name,
            "source_dir": self.source_dir,
            "backup_path": self.backup_path,
            "created_at": self.created_at,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "checksum": self.checksum,
            "tags": dict(self.tags),
            "files": list(self.files),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupManifest:
        """Deserialize from a dictionary."""
        return cls(
            backup_id=data.get("backup_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            source_dir=data.get("source_dir", ""),
            backup_path=data.get("backup_path", ""),
            created_at=data.get("created_at", time.time()),
            file_count=data.get("file_count", 0),
            total_bytes=data.get("total_bytes", 0),
            checksum=data.get("checksum", ""),
            tags=data.get("tags", {}),
            files=data.get("files", []),
        )

    def compute_checksum(self) -> str:
        """Compute a SHA-256 checksum of the manifest data (excluding checksum)."""
        data = self.to_dict()
        data.pop("checksum", None)
        raw = json.dumps(data, sort_keys=True).encode()
        self.checksum = hashlib.sha256(raw).hexdigest()
        return self.checksum

    def verify_checksum(self) -> bool:
        """Verify the manifest checksum matches current data."""
        stored = self.checksum
        data = self.to_dict()
        data.pop("checksum", None)
        raw = json.dumps(data, sort_keys=True).encode()
        computed = hashlib.sha256(raw).hexdigest()
        return stored == computed


def _validate_safe_path(base_dir: str, child: str, label: str = "path") -> str:
    """Validate that a child path resolves under base_dir.

    Prevents path traversal attacks where a crafted child path
    (e.g., containing '..') could escape the base directory.

    Returns the resolved absolute path.

    Raises
    ------
    ValueError:
        If the resolved path escapes base_dir.
    """
    base = os.path.realpath(base_dir)
    resolved = os.path.realpath(os.path.join(base_dir, child))
    # Ensure resolved path is under base_dir (exact match or subdir)
    if not (resolved == base or resolved.startswith(base + os.sep)):
        raise ValueError(
            f"Path traversal detected in {label}: "
            f"'{child}' resolves outside '{base_dir}'"
        )
    return resolved


def _validate_backup_id(backup_id: str) -> str:
    """Validate a backup ID contains only safe characters.

    Backup IDs should be UUIDs or similar safe identifiers.

    Raises
    ------
    ValueError:
        If the backup ID contains unsafe characters.
    """
    import re
    if not backup_id:
        raise ValueError("Backup ID must not be empty")
    if "\x00" in backup_id:
        raise ValueError("Backup ID contains null bytes")
    if not re.match(r"^[a-zA-Z0-9_\-]+$", backup_id):
        raise ValueError(
            f"Backup ID contains unsafe characters: '{backup_id}'. "
            "Only alphanumeric, hyphens, and underscores are allowed."
        )
    if ".." in backup_id:
        raise ValueError(f"Backup ID contains path traversal: '{backup_id}'")
    return backup_id


class BackupManager:
    """Create and restore backups of Tritium data directories.

    All operations are local filesystem only.

    Parameters
    ----------
    backup_dir:
        Root directory where backups are stored.
    max_backups:
        Maximum number of backups to keep. Oldest are pruned first.
        0 = unlimited.
    """

    def __init__(
        self,
        backup_dir: str = "/var/backups/tritium",
        max_backups: int = 0,
    ) -> None:
        self.backup_dir = backup_dir
        self.max_backups = max_backups

    def create_backup(
        self,
        source_dir: str,
        name: str = "",
        tags: dict[str, str] | None = None,
    ) -> BackupManifest:
        """Create a backup of a source directory.

        Copies all files from source_dir into a timestamped subdirectory
        of backup_dir and writes a manifest.json file.

        Parameters
        ----------
        source_dir:
            Directory to back up.
        name:
            Human-readable name for the backup.
        tags:
            Optional metadata tags.

        Returns
        -------
        BackupManifest with details of the created backup.

        Raises
        ------
        FileNotFoundError:
            If source_dir does not exist.
        """
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        manifest = BackupManifest(
            name=name or f"backup-{time.strftime('%Y%m%d-%H%M%S')}",
            source_dir=os.path.abspath(source_dir),
            tags=tags or {},
        )

        # Create backup subdirectory
        backup_path = os.path.join(self.backup_dir, manifest.backup_id)
        os.makedirs(backup_path, exist_ok=True)
        data_path = os.path.join(backup_path, "data")

        # Copy files
        total_bytes = 0
        files: list[str] = []
        for root, _dirs, filenames in os.walk(source_dir):
            for fname in filenames:
                src_file = os.path.join(root, fname)
                rel_path = os.path.relpath(src_file, source_dir)
                dst_file = os.path.join(data_path, rel_path)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
                total_bytes += os.path.getsize(src_file)
                files.append(rel_path)

        manifest.backup_path = backup_path
        manifest.file_count = len(files)
        manifest.total_bytes = total_bytes
        manifest.files = files
        manifest.compute_checksum()

        # Write manifest
        manifest_path = os.path.join(backup_path, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Prune old backups if needed
        if self.max_backups > 0:
            self._prune_old_backups()

        return manifest

    def restore_backup(
        self,
        backup_id: str,
        target_dir: str,
    ) -> BackupManifest:
        """Restore a backup to a target directory.

        Parameters
        ----------
        backup_id:
            The backup_id from the BackupManifest.
        target_dir:
            Where to restore files.

        Returns
        -------
        The BackupManifest of the restored backup.

        Raises
        ------
        FileNotFoundError:
            If the backup directory or manifest is not found.
        ValueError:
            If the manifest checksum verification fails, or if the
            backup_id or file paths contain path traversal attempts.
        """
        _validate_backup_id(backup_id)
        backup_path = _validate_safe_path(self.backup_dir, backup_id, "backup_id")
        manifest_path = os.path.join(backup_path, "manifest.json")

        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Backup manifest not found: {manifest_path}"
            )

        with open(manifest_path) as f:
            data = json.load(f)
        manifest = BackupManifest.from_dict(data)

        if not manifest.verify_checksum():
            raise ValueError(
                f"Backup {backup_id} checksum verification failed — "
                "data may be corrupted"
            )

        # Copy files from backup to target
        data_path = os.path.join(backup_path, "data")
        os.makedirs(target_dir, exist_ok=True)

        for rel_path in manifest.files:
            # Validate each rel_path to prevent path traversal on restore
            _validate_safe_path(data_path, rel_path, "backup file source")
            _validate_safe_path(target_dir, rel_path, "restore file target")

            src_file = os.path.join(data_path, rel_path)
            dst_file = os.path.join(target_dir, rel_path)
            if os.path.isfile(src_file):
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)

        return manifest

    def list_backups(self) -> list[BackupManifest]:
        """List all available backups.

        Returns a list of BackupManifest objects sorted by creation time
        (newest first).
        """
        manifests: list[BackupManifest] = []
        if not os.path.isdir(self.backup_dir):
            return manifests

        for entry in os.listdir(self.backup_dir):
            # Skip entries with unsafe names
            try:
                _validate_backup_id(entry)
            except ValueError:
                continue
            manifest_path = os.path.join(
                self.backup_dir, entry, "manifest.json"
            )
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path) as f:
                        data = json.load(f)
                    manifests.append(BackupManifest.from_dict(data))
                except (json.JSONDecodeError, KeyError, OSError):
                    continue

        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup by ID.

        Returns True if the backup was found and deleted.

        Raises
        ------
        ValueError:
            If the backup_id contains path traversal characters.
        """
        _validate_backup_id(backup_id)
        backup_path = _validate_safe_path(self.backup_dir, backup_id, "backup_id")
        if os.path.isdir(backup_path):
            shutil.rmtree(backup_path)
            return True
        return False

    def _prune_old_backups(self) -> int:
        """Remove oldest backups exceeding max_backups limit.

        Returns the number of backups pruned.
        """
        if self.max_backups <= 0:
            return 0

        backups = self.list_backups()
        pruned = 0
        while len(backups) > self.max_backups:
            oldest = backups.pop()  # Already sorted newest-first
            self.delete_backup(oldest.backup_id)
            pruned += 1

        return pruned
