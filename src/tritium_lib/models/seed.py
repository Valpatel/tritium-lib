# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Seed/replication models — for self-replicating firmware distribution.

A SeedPackage is a self-contained bundle that an edge device can use to
provision other devices.  It contains firmware images, manifests, and
checksums so that firmware can propagate across a mesh network or via
SD card sneakernet without a central server.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SeedFile(BaseModel):
    """A single file within a seed package."""
    path: str  # relative path inside the package
    size_bytes: int = 0
    sha256: str = ""
    board: str = "any"  # target board or "any" for universal files
    description: str = ""


class SeedManifest(BaseModel):
    """Manifest describing the contents and compatibility of a seed package.

    The manifest is the source of truth for what a seed package contains.
    Edge devices check the manifest before applying updates.
    """
    package_id: str
    firmware_version: str
    build_timestamp: Optional[datetime] = None
    boards: list[str] = Field(default_factory=list)  # compatible board IDs
    family: str = "esp32"
    files: list[SeedFile] = Field(default_factory=list)
    total_size_bytes: int = 0
    manifest_sha256: str = ""  # checksum of the manifest itself
    notes: str = ""

    @property
    def file_count(self) -> int:
        return len(self.files)

    def files_for_board(self, board: str) -> list[SeedFile]:
        """Return files compatible with a specific board."""
        return [f for f in self.files if f.board in (board, "any")]

    def is_compatible(self, board: str) -> bool:
        """Check if this seed package supports a given board."""
        return board in self.boards or "any" in self.boards


class SeedStatus(str, Enum):
    """Status of a seed package in the distribution pipeline."""
    CREATED = "created"
    DISTRIBUTING = "distributing"
    COMPLETE = "complete"
    FAILED = "failed"


class SeedTransferStatus(str, Enum):
    """Status of a seed transfer between nodes."""
    PENDING = "pending"
    TRANSFERRING = "transferring"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


class SeedTransfer(BaseModel):
    """Tracks a seed package transfer between two nodes.

    Represents an active or completed transfer of a seed package from
    one device to another, with progress tracking.
    """
    id: str
    source_node: str  # device_id of sender
    dest_node: str  # device_id of receiver
    manifest: SeedManifest
    progress: float = 0.0  # 0.0 to 1.0
    status: SeedTransferStatus = SeedTransferStatus.PENDING
    bytes_transferred: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status in (
            SeedTransferStatus.TRANSFERRING,
            SeedTransferStatus.VERIFYING,
        )

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.manifest.total_size_bytes - self.bytes_transferred)


class SeedPackage(BaseModel):
    """A complete seed package for firmware distribution.

    Tracks the lifecycle of a seed from creation through distribution.
    """
    id: str
    manifest: SeedManifest
    status: SeedStatus = SeedStatus.CREATED
    source_device: str = ""  # device_id that created/sourced this seed
    created_at: Optional[datetime] = None
    distributed_to: list[str] = Field(default_factory=list)  # device_ids
    distribution_count: int = 0
    error: Optional[str] = None

    @property
    def total_size(self) -> int:
        return self.manifest.total_size_bytes
