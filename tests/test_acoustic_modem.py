"""Tests for tritium_lib.models.acoustic_modem."""

from tritium_lib.models.acoustic_modem import (
    AcousticFrame,
    AcousticConfig,
    AcousticChannelStats,
    ModulationType,
)


class TestAcousticFrame:
    def test_create(self):
        frame = AcousticFrame(
            preamble=b"\xaa\xaa",
            payload_bytes=b"hello",
            crc=0x1234,
            modulation=ModulationType.FSK,
        )
        assert frame.payload_size == 5
        assert frame.total_size == 7
        assert frame.crc == 0x1234

    def test_defaults(self):
        frame = AcousticFrame()
        assert frame.modulation == ModulationType.FSK
        assert frame.payload_size == 0
        assert frame.total_size == 0
        assert frame.frame_id is None

    def test_psk_modulation(self):
        frame = AcousticFrame(
            payload_bytes=b"\x01\x02\x03",
            modulation=ModulationType.PSK,
        )
        assert frame.modulation == ModulationType.PSK
        assert frame.payload_size == 3

    def test_json_roundtrip(self):
        frame = AcousticFrame(
            payload_bytes=b"test",
            crc=42,
            modulation=ModulationType.OFDM,
            frame_id=7,
        )
        frame2 = AcousticFrame.model_validate_json(frame.model_dump_json())
        assert frame2.crc == 42
        assert frame2.modulation == ModulationType.OFDM
        assert frame2.frame_id == 7


class TestAcousticConfig:
    def test_create(self):
        cfg = AcousticConfig(
            frequency_hz=2000,
            baud_rate=1200,
            modulation=ModulationType.PSK,
        )
        assert cfg.frequency_hz == 2000
        assert cfg.baud_rate == 1200
        assert cfg.modulation == ModulationType.PSK

    def test_defaults(self):
        cfg = AcousticConfig()
        assert cfg.frequency_hz == 1000
        assert cfg.baud_rate == 300
        assert cfg.modulation == ModulationType.FSK
        assert cfg.sample_rate_hz == 44100

    def test_json_roundtrip(self):
        cfg = AcousticConfig(frequency_hz=5000, baud_rate=9600)
        cfg2 = AcousticConfig.model_validate_json(cfg.model_dump_json())
        assert cfg2.frequency_hz == 5000
        assert cfg2.baud_rate == 9600


class TestAcousticChannelStats:
    def test_create(self):
        stats = AcousticChannelStats(
            snr_db=15.5,
            bit_error_rate=0.001,
            throughput_bps=2400.0,
            frames_sent=100,
            frames_received=95,
            frames_dropped=5,
        )
        assert stats.snr_db == 15.5
        assert stats.throughput_bps == 2400.0
        assert stats.frames_dropped == 5

    def test_frame_loss_rate(self):
        stats = AcousticChannelStats(
            frames_received=80,
            frames_dropped=20,
        )
        assert abs(stats.frame_loss_rate - 0.2) < 0.001

    def test_frame_loss_rate_zero(self):
        stats = AcousticChannelStats()
        assert stats.frame_loss_rate == 0.0

    def test_perfect_channel(self):
        stats = AcousticChannelStats(
            snr_db=30.0,
            bit_error_rate=0.0,
            frames_received=1000,
            frames_dropped=0,
        )
        assert stats.frame_loss_rate == 0.0
        assert stats.bit_error_rate == 0.0

    def test_json_roundtrip(self):
        stats = AcousticChannelStats(snr_db=10.0, throughput_bps=1200.0)
        stats2 = AcousticChannelStats.model_validate_json(stats.model_dump_json())
        assert stats2.snr_db == 10.0
        assert stats2.throughput_bps == 1200.0
