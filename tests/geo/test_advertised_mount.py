# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""A camera server that says what it is bolted to, and is believed correctly.

A body-mounted camera renders from a lens that USD (or a URDF) composes from
a body-frame offset.  The consumer that draws the FOV cone on the map has to
reconstruct that SAME offset, and until now it did so by an operator typing
the numbers in a second time.  Two hand-entered copies of one geometry is a
drift bug with a delay fuse: the picture and the cone disagree, and nothing
in the system can tell.

These tests pin the parse of the advertisement that removes the second copy.
They are deliberately hostile about malformed input, because the document
comes off the network from a process that may be a different version.
"""

import pytest

from tritium_lib.geo.camera_mount import (
    AdvertisedMount,
    CameraMount,
    parse_advertised_mount,
)


def _status(**mount):
    """A camera server /status document advertising a mount."""
    base = {
        "camera_id": "isaac_rgb",
        "width": 1280,
        "height": 720,
        "fov_angle": 70.0,
        "fov_range": 80.0,
        "status": "online",
    }
    if mount:
        base["mount"] = mount
    return base


class TestNoMount:
    """A wall camera must stay a wall camera."""

    def test_status_without_mount_block_parses_to_none(self):
        assert parse_advertised_mount(_status()) is None

    def test_empty_mount_block_is_not_a_mount(self):
        # A server that emits "mount": {} is telling us nothing, and binding
        # a feed to target "" would silently null the cone forever.
        assert parse_advertised_mount({"mount": {}}) is None

    def test_mount_naming_neither_target_nor_prim_is_not_a_mount(self):
        assert parse_advertised_mount(
            {"mount": {"forward_m": 0.3, "up_m": 0.25}}
        ) is None

    def test_junk_documents_do_not_raise(self):
        # Straight off a socket: a 404 body, a list, None.
        for junk in (None, [], "not json", {"mount": "yes"}, {"mount": 3}):
            assert parse_advertised_mount(junk) is None


class TestParsedGeometry:
    def test_offsets_and_tilt_reach_the_mount(self):
        adv = parse_advertised_mount(_status(
            attach_to="robot_go2", prim="/World/Go2",
            forward_m=0.30, left_m=-0.05, up_m=0.25, tilt_deg=-10.0,
        ))
        assert isinstance(adv, AdvertisedMount)
        assert adv.attach_to == "robot_go2"
        assert adv.prim == "/World/Go2"
        assert adv.mount.forward_m == pytest.approx(0.30)
        assert adv.mount.left_m == pytest.approx(-0.05)
        assert adv.mount.up_m == pytest.approx(0.25)
        assert adv.mount.tilt_deg == pytest.approx(-10.0)

    def test_fov_comes_from_the_top_level_status_not_the_mount_block(self):
        # The server advertises FOV alongside width/height because it is a
        # property of the render, not of the bracket holding the camera.
        adv = parse_advertised_mount(_status(attach_to="robot_go2"))
        assert adv.mount.hfov_deg == pytest.approx(70.0)
        assert adv.mount.range_m == pytest.approx(80.0)

    def test_absent_offsets_default_to_a_lens_at_the_body_origin(self):
        adv = parse_advertised_mount({"mount": {"attach_to": "r1"}})
        assert adv.mount.forward_m == 0.0
        assert adv.mount.left_m == 0.0
        assert adv.mount.up_m == 0.0
        assert adv.mount.tilt_deg == 0.0

    def test_null_offsets_are_treated_as_absent_not_as_a_crash(self):
        # json.dumps of an unset argparse default emits null, not omission.
        adv = parse_advertised_mount(
            {"mount": {"attach_to": "r1", "forward_m": None, "prim": None}}
        )
        assert adv.prim is None
        assert adv.mount.forward_m == 0.0

    def test_unparseable_numbers_fall_back_rather_than_reject_the_mount(self):
        adv = parse_advertised_mount(
            {"mount": {"attach_to": "r1", "forward_m": "0.3", "up_m": "junk"}}
        )
        assert adv.mount.forward_m == pytest.approx(0.3)
        assert adv.mount.up_m == 0.0

    def test_out_of_range_fov_does_not_take_the_whole_mount_down(self):
        # CameraMount rejects hfov 0; the mount itself is still valid, and
        # dropping the bind over a bad FOV would lose the body attachment.
        adv = parse_advertised_mount(
            {"mount": {"attach_to": "r1"}, "fov_angle": 0.0}
        )
        assert adv is not None
        assert adv.mount.hfov_deg > 0.0

    def test_prim_alone_is_enough_to_be_a_mount(self):
        # Isaac's --mount-prim without --attach-to: we know it rides a body,
        # we just do not know that body's tracked-target id yet.
        adv = parse_advertised_mount({"mount": {"prim": "/World/Go2"}})
        assert adv is not None
        assert adv.attach_to is None
        assert adv.prim == "/World/Go2"


class TestFeedExtra:
    """The parsed mount has to land on the config keys SC already stores."""

    def test_emits_the_canonical_mount_keys(self):
        adv = parse_advertised_mount(_status(
            attach_to="robot_go2", forward_m=0.30, left_m=-0.05,
            up_m=0.25, tilt_deg=-10.0,
        ))
        extra = adv.to_feed_extra()
        assert extra["attach_to"] == "robot_go2"
        assert extra["mount_forward_m"] == pytest.approx(0.30)
        assert extra["mount_left_m"] == pytest.approx(-0.05)
        assert extra["mount_up_m"] == pytest.approx(0.25)
        assert extra["mount_tilt_deg"] == pytest.approx(-10.0)
        assert extra["fov_angle"] == pytest.approx(70.0)
        assert extra["fov_range"] == pytest.approx(80.0)

    def test_omits_attach_to_when_the_server_named_only_a_prim(self):
        # Writing attach_to=None would make the follower treat the feed as
        # attached-but-broken instead of as an ordinary posed camera.
        extra = parse_advertised_mount({"mount": {"prim": "/World/Go2"}}).to_feed_extra()
        assert "attach_to" not in extra

    def test_round_trips_through_the_same_geometry_the_render_used(self):
        # The drift gate: what SC stores must rebuild the identical mount.
        # If these ever disagree, the picture and the cone disagree.
        adv = parse_advertised_mount(_status(
            attach_to="robot_go2", forward_m=0.30, left_m=-0.05,
            up_m=0.25, tilt_deg=-10.0,
        ))
        extra = adv.to_feed_extra()
        rebuilt = CameraMount(
            forward_m=extra["mount_forward_m"],
            left_m=extra["mount_left_m"],
            up_m=extra["mount_up_m"],
            tilt_deg=extra["mount_tilt_deg"],
            hfov_deg=extra["fov_angle"],
            vfov_deg=adv.mount.vfov_deg,
            range_m=extra["fov_range"],
        )
        assert rebuilt == adv.mount
        assert rebuilt.stage_offset() == adv.mount.stage_offset()
