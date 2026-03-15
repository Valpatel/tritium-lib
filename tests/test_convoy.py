# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for convoy model."""

import pytest
from tritium_lib.models.convoy import (
    Convoy,
    ConvoyFormation,
    ConvoyStatus,
    ConvoySummary,
)


class TestConvoy:
    """Tests for the Convoy model."""

    def test_create_default(self):
        c = Convoy()
        assert c.convoy_id == ""
        assert c.member_target_ids == []
        assert c.speed_avg_mps == 0.0
        assert c.heading_avg_deg == 0.0
        assert c.formation == ConvoyFormation.UNKNOWN
        assert c.status == ConvoyStatus.ACTIVE
        assert c.suspicious_score == 0.0
        assert c.first_seen is not None
        assert c.last_seen is not None

    def test_create_with_members(self):
        c = Convoy(
            convoy_id="convoy_test1",
            member_target_ids=["ble_aa", "ble_bb", "ble_cc"],
            speed_avg_mps=5.0,
            heading_avg_deg=90.0,
            formation=ConvoyFormation.LINE,
        )
        assert c.member_count == 3
        assert c.is_valid is True

    def test_invalid_convoy_too_few(self):
        c = Convoy(member_target_ids=["a", "b"])
        assert c.is_valid is False

    def test_add_member(self):
        c = Convoy(member_target_ids=["a", "b", "c"])
        assert c.add_member("d") is True
        assert c.member_count == 4
        assert c.add_member("d") is False  # duplicate

    def test_remove_member_disperses(self):
        c = Convoy(member_target_ids=["a", "b", "c"])
        assert c.remove_member("c") is True
        assert c.status == ConvoyStatus.DISPERSED
        assert c.is_valid is False

    def test_remove_nonexistent(self):
        c = Convoy(member_target_ids=["a", "b", "c"])
        assert c.remove_member("z") is False

    def test_compute_suspicious_score(self):
        c = Convoy(
            member_target_ids=["a", "b", "c", "d", "e"],
            heading_variance_deg=5.0,
            speed_variance_mps=0.2,
            duration_s=600.0,
        )
        score = c.compute_suspicious_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Tight coordination + long duration + 5 members

    def test_suspicious_score_low_for_loose(self):
        c = Convoy(
            member_target_ids=["a", "b", "c"],
            heading_variance_deg=40.0,
            speed_variance_mps=1.8,
            duration_s=30.0,
        )
        score = c.compute_suspicious_score()
        assert score < 0.3

    def test_formation_enum(self):
        assert ConvoyFormation.LINE == "line"
        assert ConvoyFormation.CLUSTER == "cluster"
        assert ConvoyFormation.SPREAD == "spread"

    def test_status_enum(self):
        assert ConvoyStatus.ACTIVE == "active"
        assert ConvoyStatus.DISPERSED == "dispersed"
        assert ConvoyStatus.STOPPED == "stopped"
        assert ConvoyStatus.MERGED == "merged"


class TestConvoySummary:
    """Tests for ConvoySummary."""

    def test_create_default(self):
        s = ConvoySummary()
        assert s.total_convoys == 0
        assert s.active_convoys == 0
        assert s.avg_suspicious_score == 0.0

    def test_create_with_values(self):
        s = ConvoySummary(
            total_convoys=3,
            active_convoys=2,
            total_members=9,
            avg_suspicious_score=0.65,
            highest_suspicious_score=0.9,
            largest_convoy_size=5,
        )
        assert s.total_convoys == 3
        assert s.largest_convoy_size == 5
