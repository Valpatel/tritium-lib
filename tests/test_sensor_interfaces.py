# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sensor plugin interfaces.

Verifies that the abstract interface hierarchy works correctly:
- SensorPlugin cannot be instantiated directly
- SDRPlugin, RadarPlugin, CameraPlugin cannot be instantiated directly
- Concrete implementations that fulfill all abstract methods can be instantiated
- Inheritance chain is correct
"""

import pytest

from tritium_lib.interfaces import (
    CameraPlugin,
    RadarPlugin,
    SDRMonitorConfig,
    SDRPlugin,
    SensorPlugin,
)
from tritium_lib.models.radar import RadarConfig, RadarScan, RadarTrack
from tritium_lib.models.sdr import RFSignal, SpectrumScan


# --- Mock implementations ---


class MockSensor(SensorPlugin):
    """Minimal concrete SensorPlugin for testing."""

    def get_name(self) -> str:
        return "mock-sensor"

    def get_sensor_type(self) -> str:
        return "mock"

    def get_capabilities(self) -> list[str]:
        return ["test"]

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_status(self) -> dict:
        return {"running": False, "error": None, "uptime_s": 0.0}

    def get_mqtt_topics(self) -> list[str]:
        return ["tritium/test/mock/status"]


class MockSDR(SDRPlugin):
    """Minimal concrete SDRPlugin for testing."""

    def get_name(self) -> str:
        return "mock-rtlsdr"

    def get_sensor_type(self) -> str:
        return "sdr"

    def get_capabilities(self) -> list[str]:
        return ["spectrum_scan", "signal_decode"]

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_status(self) -> dict:
        return {"running": True, "error": None, "uptime_s": 42.0}

    def get_mqtt_topics(self) -> list[str]:
        return ["tritium/site1/sdr/rtlsdr0/signal"]

    def get_frequency_range(self) -> tuple[float, float]:
        return (24e6, 1766e6)

    def get_sample_rate(self) -> int:
        return 2_048_000

    def tune(self, frequency_hz: float) -> None:
        pass

    def set_gain(self, gain_db: float) -> None:
        pass

    def get_spectrum(self, center_freq_mhz: float, span_mhz: float) -> SpectrumScan:
        return SpectrumScan(
            center_freq_mhz=center_freq_mhz,
            span_mhz=span_mhz,
            bins=[-80.0, -75.0, -60.0, -75.0, -80.0],
            source_id="mock-rtlsdr",
        )

    def start_monitoring(self, config: SDRMonitorConfig) -> None:
        pass

    def get_detected_signals(self) -> list[RFSignal]:
        return [RFSignal(frequency_mhz=433.92, power_dbm=-45.0, source_id="mock-rtlsdr")]


class MockRadar(RadarPlugin):
    """Minimal concrete RadarPlugin for testing."""

    def get_name(self) -> str:
        return "mock-radar"

    def get_sensor_type(self) -> str:
        return "radar"

    def get_capabilities(self) -> list[str]:
        return ["track_detection", "doppler"]

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_status(self) -> dict:
        return {"running": True, "error": None, "uptime_s": 100.0}

    def get_mqtt_topics(self) -> list[str]:
        return ["tritium/site1/radar/radar0/scan"]

    def get_max_range(self) -> float:
        return 5000.0

    def get_tracks(self) -> list[RadarTrack]:
        return [
            RadarTrack(
                track_id="t001",
                range_m=1200.0,
                azimuth_deg=45.0,
                velocity_mps=15.0,
                source_id="mock-radar",
            )
        ]

    def start_scanning(self, config: RadarConfig) -> None:
        pass

    def get_scan(self) -> RadarScan:
        return RadarScan(scan_id="s001", tracks=self.get_tracks())


class MockCamera(CameraPlugin):
    """Minimal concrete CameraPlugin for testing."""

    def get_name(self) -> str:
        return "mock-camera"

    def get_sensor_type(self) -> str:
        return "camera"

    def get_capabilities(self) -> list[str]:
        return ["streaming", "object_detection"]

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_status(self) -> dict:
        return {"running": True, "error": None, "uptime_s": 60.0}

    def get_mqtt_topics(self) -> list[str]:
        return ["tritium/site1/cameras/cam0/frame"]

    def get_frame(self) -> bytes:
        return b"\xff\xd8\xff\xe0"  # JPEG magic bytes

    def get_detections(self) -> list[dict]:
        return [{"class": "person", "confidence": 0.92, "bbox": [100, 200, 300, 400]}]

    def set_resolution(self, width: int, height: int) -> None:
        pass


# --- Tests ---


class TestSensorPluginABC:
    """Test that abstract classes cannot be instantiated."""

    def test_sensor_plugin_is_abstract(self):
        with pytest.raises(TypeError):
            SensorPlugin()  # type: ignore

    def test_sdr_plugin_is_abstract(self):
        with pytest.raises(TypeError):
            SDRPlugin()  # type: ignore

    def test_radar_plugin_is_abstract(self):
        with pytest.raises(TypeError):
            RadarPlugin()  # type: ignore

    def test_camera_plugin_is_abstract(self):
        with pytest.raises(TypeError):
            CameraPlugin()  # type: ignore


class TestInheritanceChain:
    """Test that the inheritance hierarchy is correct."""

    def test_sdr_inherits_sensor(self):
        assert issubclass(SDRPlugin, SensorPlugin)

    def test_radar_inherits_sensor(self):
        assert issubclass(RadarPlugin, SensorPlugin)

    def test_camera_inherits_sensor(self):
        assert issubclass(CameraPlugin, SensorPlugin)

    def test_mock_sdr_is_sensor(self):
        sdr = MockSDR()
        assert isinstance(sdr, SensorPlugin)
        assert isinstance(sdr, SDRPlugin)

    def test_mock_radar_is_sensor(self):
        radar = MockRadar()
        assert isinstance(radar, SensorPlugin)
        assert isinstance(radar, RadarPlugin)

    def test_mock_camera_is_sensor(self):
        cam = MockCamera()
        assert isinstance(cam, SensorPlugin)
        assert isinstance(cam, CameraPlugin)


class TestMockSensor:
    """Test basic SensorPlugin mock."""

    def test_identity(self):
        s = MockSensor()
        assert s.get_name() == "mock-sensor"
        assert s.get_sensor_type() == "mock"

    def test_capabilities(self):
        s = MockSensor()
        assert "test" in s.get_capabilities()

    def test_status(self):
        s = MockSensor()
        status = s.get_status()
        assert "running" in status
        assert "error" in status

    def test_mqtt_topics(self):
        s = MockSensor()
        topics = s.get_mqtt_topics()
        assert len(topics) > 0


class TestMockSDR:
    """Test SDRPlugin mock implementation."""

    def test_frequency_range(self):
        sdr = MockSDR()
        lo, hi = sdr.get_frequency_range()
        assert lo < hi
        assert lo == 24e6
        assert hi == 1766e6

    def test_sample_rate(self):
        sdr = MockSDR()
        assert sdr.get_sample_rate() == 2_048_000

    def test_spectrum(self):
        sdr = MockSDR()
        scan = sdr.get_spectrum(433.92, 2.0)
        assert isinstance(scan, SpectrumScan)
        assert scan.center_freq_mhz == 433.92
        assert len(scan.bins) == 5

    def test_detected_signals(self):
        sdr = MockSDR()
        signals = sdr.get_detected_signals()
        assert len(signals) == 1
        assert isinstance(signals[0], RFSignal)
        assert signals[0].frequency_mhz == 433.92

    def test_sensor_type(self):
        sdr = MockSDR()
        assert sdr.get_sensor_type() == "sdr"


class TestMockRadar:
    """Test RadarPlugin mock implementation."""

    def test_max_range(self):
        radar = MockRadar()
        assert radar.get_max_range() == 5000.0

    def test_tracks(self):
        radar = MockRadar()
        tracks = radar.get_tracks()
        assert len(tracks) == 1
        assert isinstance(tracks[0], RadarTrack)
        assert tracks[0].range_m == 1200.0

    def test_scan(self):
        radar = MockRadar()
        scan = radar.get_scan()
        assert isinstance(scan, RadarScan)
        assert scan.scan_id == "s001"
        assert len(scan.tracks) == 1

    def test_sensor_type(self):
        radar = MockRadar()
        assert radar.get_sensor_type() == "radar"


class TestMockCamera:
    """Test CameraPlugin mock implementation."""

    def test_frame(self):
        cam = MockCamera()
        frame = cam.get_frame()
        assert isinstance(frame, bytes)
        assert frame[:2] == b"\xff\xd8"  # JPEG magic

    def test_detections(self):
        cam = MockCamera()
        dets = cam.get_detections()
        assert len(dets) == 1
        assert dets[0]["class"] == "person"
        assert 0.0 <= dets[0]["confidence"] <= 1.0

    def test_sensor_type(self):
        cam = MockCamera()
        assert cam.get_sensor_type() == "camera"


class TestSDRMonitorConfig:
    """Test SDRMonitorConfig model."""

    def test_defaults(self):
        config = SDRMonitorConfig()
        assert config.center_freq_mhz == 433.92
        assert config.sample_rate_hz == 2_048_000
        assert config.gain_db == 40.0
        assert config.enabled is True

    def test_custom(self):
        config = SDRMonitorConfig(
            frequencies_mhz=[433.92, 868.0, 915.0],
            center_freq_mhz=868.0,
            span_mhz=5.0,
            sample_rate_hz=1_024_000,
            gain_db=30.0,
            squelch_db=-40.0,
            decode_protocols=["rtl_433", "lora"],
            scan_interval_s=0.5,
        )
        assert len(config.frequencies_mhz) == 3
        assert config.decode_protocols == ["rtl_433", "lora"]

    def test_serialization(self):
        config = SDRMonitorConfig(frequencies_mhz=[433.92])
        d = config.model_dump()
        assert "frequencies_mhz" in d
        assert "center_freq_mhz" in d
        restored = SDRMonitorConfig.model_validate(d)
        assert restored.frequencies_mhz == [433.92]


class TestPartialImplementationFails:
    """Test that incomplete implementations cannot be instantiated."""

    def test_partial_sdr_fails(self):
        """An SDR plugin missing some abstract methods should fail."""

        class IncompletSDR(SDRPlugin):
            def get_name(self) -> str:
                return "bad"

            # Missing everything else

        with pytest.raises(TypeError):
            IncompletSDR()  # type: ignore

    def test_partial_radar_fails(self):
        class IncompleteRadar(RadarPlugin):
            def get_name(self) -> str:
                return "bad"

        with pytest.raises(TypeError):
            IncompleteRadar()  # type: ignore

    def test_partial_camera_fails(self):
        class IncompleteCamera(CameraPlugin):
            def get_name(self) -> str:
                return "bad"

        with pytest.raises(TypeError):
            IncompleteCamera()  # type: ignore
