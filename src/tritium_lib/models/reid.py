# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Re-identification models — track the same person or vehicle across cameras.

ReID works by extracting appearance feature vectors (embeddings) from detected
objects in camera frames.  When a new detection arrives, its embedding is
compared against stored embeddings via cosine similarity.  High-similarity
matches indicate the same individual seen from a different camera — no facial
recognition required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ReIDEmbedding(BaseModel):
    """A stored appearance embedding for a detected target.

    Each embedding captures the visual signature of a person, vehicle, or
    other tracked object as seen by a specific camera at a specific time.
    """
    embedding_id: str = ""
    target_id: str = ""
    source_camera: str = ""
    timestamp: Optional[datetime] = None
    dimensions: int = 0  # length of the feature vector

    model_config = {"frozen": False}


class ReIDMatch(BaseModel):
    """A cross-camera re-identification match.

    Represents a high-similarity pairing between two embeddings, indicating
    the same target was observed by different cameras.
    """
    query_id: str = ""
    matched_id: str = ""
    similarity: float = 0.0  # cosine similarity, 0.0 to 1.0
    source_cameras: list[str] = Field(default_factory=list)

    @property
    def is_strong_match(self) -> bool:
        """True if similarity exceeds 0.85 (strong re-identification)."""
        return self.similarity > 0.85
