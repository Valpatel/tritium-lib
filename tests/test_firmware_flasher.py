# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the firmware flasher abstraction."""

import asyncio
import pytest

from tritium_lib.firmware.base import (
    FirmwareFlasher,
    FlashResult,
    FlashStatus,
    DeviceDetection,
)
from tritium_lib.firmware.esp32 import (
    ESP32Flasher,
    ALL_KNOWN_VIDS,
    CHIP_FLASH_OFFSETS,
)
from tritium_lib.firmware.meshtastic_flasher import (
    MeshtasticFlasher,
    BOARD_FIRMWARE_MAP,
    MESHTASTIC_FLASH_OFFSETS,
    FIRMWARE_CACHE_DIR,
)


class TestDeviceDetection:
    def test_empty(self):
        d = DeviceDetection()
        assert d.detected is False
        assert d.port == ""

    def test_to_dict_omits_empty(self):
        d = DeviceDetection(detected=True, port="/dev/ttyACM0", chip="ESP32-S3")
        result = d.to_dict()
        assert result["detected"] is True
        assert result["chip"] == "ESP32-S3"
        assert "chip_id" not in result

    def test_to_dict_full(self):
        d = DeviceDetection(
            detected=True, port="/dev/ttyACM0", chip="ESP32-S3",
            chip_id="aa:bb:cc:dd", flash_size="16MB",
            firmware_version="2.5.19", board="tlora-pager",
        )
        assert len(d.to_dict()) == 7


class TestFlashResult:
    def test_default_pending(self):
        r = FlashResult()
        assert r.success is False
        assert r.status == FlashStatus.PENDING

    def test_to_dict(self):
        r = FlashResult(success=True, status=FlashStatus.COMPLETED,
                        firmware_version="2.5.19", port="/dev/ttyACM0",
                        duration_s=12.345)
        d = r.to_dict()
        assert d["success"] is True
        assert d["status"] == "completed"
        assert d["duration_s"] == 12.3


class TestFlashStatus:
    def test_all_statuses(self):
        statuses = [s.value for s in FlashStatus]
        for expected in ["pending", "writing", "completed", "failed",
                         "detecting", "downloading", "erasing", "verifying"]:
            assert expected in statuses


class TestFirmwareFlasherBase:
    def test_abstract(self):
        with pytest.raises(TypeError):
            FirmwareFlasher()

    def test_find_serial_ports(self):
        ports = FirmwareFlasher.find_serial_ports()
        assert isinstance(ports, list)


class TestESP32Flasher:
    def test_create(self):
        f = ESP32Flasher(port="/dev/ttyACM0")
        assert f.port == "/dev/ttyACM0"
        assert f.baud == 921600

    def test_custom_baud(self):
        f = ESP32Flasher(baud=115200)
        assert f.baud == 115200

    def test_known_vids(self):
        assert "303a" in ALL_KNOWN_VIDS
        assert "10c4" in ALL_KNOWN_VIDS
        assert "1a86" in ALL_KNOWN_VIDS

    def test_chip_offsets(self):
        assert CHIP_FLASH_OFFSETS["esp32s3"] == "0x0"
        assert CHIP_FLASH_OFFSETS["esp32"] == "0x1000"

    def test_progress_callback(self):
        f = ESP32Flasher()
        calls = []
        f.on_progress(lambda s, p, m: calls.append((s, p, m)))
        f._emit_progress(FlashStatus.WRITING, 50.0, "test")
        assert len(calls) == 1
        assert calls[0] == (FlashStatus.WRITING, 50.0, "test")

    def test_detect_no_esptool(self):
        f = ESP32Flasher()
        f._esptool_path = None
        result = asyncio.run(f.detect())
        assert result.detected is False
        assert "esptool" in result.error.lower()

    def test_flash_no_esptool(self):
        f = ESP32Flasher()
        f._esptool_path = None
        result = asyncio.run(f.flash("/tmp/test.bin"))
        assert result.success is False
        assert "esptool" in result.error.lower()

    def test_flash_missing_file(self):
        f = ESP32Flasher(port="/dev/ttyACM0")
        f._esptool_path = "/usr/bin/esptool.py"
        result = asyncio.run(f.flash("/nonexistent/firmware.bin"))
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_erase_no_esptool(self):
        f = ESP32Flasher()
        f._esptool_path = None
        result = asyncio.run(f.erase_flash())
        assert result.success is False


class TestMeshtasticFlasher:
    def test_create(self):
        f = MeshtasticFlasher(port="/dev/ttyACM0")
        assert f.port == "/dev/ttyACM0"
        assert isinstance(f, ESP32Flasher)

    def test_hierarchy(self):
        assert issubclass(MeshtasticFlasher, ESP32Flasher)
        assert issubclass(MeshtasticFlasher, FirmwareFlasher)

    def test_board_map_tlora_pager(self):
        assert BOARD_FIRMWARE_MAP["TLORA_PAGER"] == "tlora-pager"

    def test_board_map_common(self):
        assert "TBEAM" in BOARD_FIRMWARE_MAP
        assert "T_DECK" in BOARD_FIRMWARE_MAP
        assert "HELTEC_V3" in BOARD_FIRMWARE_MAP

    def test_board_map_count(self):
        assert len(BOARD_FIRMWARE_MAP) >= 15

    def test_flash_offsets(self):
        assert MESHTASTIC_FLASH_OFFSETS["ESP32-S3"] == "0x0"
        assert MESHTASTIC_FLASH_OFFSETS["ESP32"] == "0x1000"

    def test_find_firmware_asset_bin(self):
        assets = [
            {"name": "firmware-tlora-pager-2.5.19.bin", "browser_download_url": "http://x"},
            {"name": "firmware-tbeam-2.5.19.bin", "browser_download_url": "http://y"},
        ]
        result = MeshtasticFlasher._find_firmware_asset(assets, "tlora-pager")
        assert result is not None
        assert "tlora-pager" in result["name"]

    def test_find_firmware_asset_not_found(self):
        assets = [{"name": "firmware-tbeam-2.5.19.bin", "browser_download_url": "http://x"}]
        assert MeshtasticFlasher._find_firmware_asset(assets, "tlora-pager") is None

    def test_find_firmware_asset_zip(self):
        assets = [{"name": "firmware-tlora-pager-2.5.19.zip", "browser_download_url": "http://x"}]
        assert MeshtasticFlasher._find_firmware_asset(assets, "tlora-pager") is not None

    def test_find_firmware_case_insensitive(self):
        assets = [{"name": "firmware-TLORA-PAGER-2.5.19.bin", "browser_download_url": "http://x"}]
        assert MeshtasticFlasher._find_firmware_asset(assets, "tlora-pager") is not None

    def test_cache_dir(self):
        assert "meshtastic" in str(FIRMWARE_CACHE_DIR)

    def test_flash_latest_no_device(self):
        f = MeshtasticFlasher(port="/dev/nonexistent")
        f._esptool_path = None
        result = asyncio.run(f.flash_latest())
        assert result.success is False

    def test_flash_with_cli_no_cli(self):
        f = MeshtasticFlasher()
        f._meshtastic_cli = None
        result = asyncio.run(f.flash_with_meshtastic_cli())
        assert result.success is False
        assert "meshtastic CLI" in result.error


class TestImports:
    def test_import_all(self):
        from tritium_lib.firmware import (
            FirmwareFlasher, FlashResult, DeviceDetection,
            ESP32Flasher, MeshtasticFlasher,
        )
        assert all([FirmwareFlasher, ESP32Flasher, MeshtasticFlasher,
                     FlashResult, DeviceDetection])
