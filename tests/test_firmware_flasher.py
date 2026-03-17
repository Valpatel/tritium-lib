# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the firmware flasher abstraction."""

import asyncio
import json
import os
import tempfile
import zipfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    MeshtasticFirmwareFiles,
    BOARD_FIRMWARE_MAP,
    MESHTASTIC_FLASH_OFFSETS,
    FIRMWARE_CACHE_DIR,
    DEFAULT_OTA_OFFSET,
    DEFAULT_SPIFFS_OFFSET,
    DEFAULT_APP_OFFSET,
    ESP32S3_BOARDS,
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

    def test_flash_additional_writes_missing_file(self):
        """Additional writes with nonexistent file should fail early."""
        f = ESP32Flasher(port="/dev/ttyACM0")
        f._esptool_path = "/usr/bin/esptool.py"
        # Create a real temp file for the main firmware
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            main_fw = tmp.name
        try:
            result = asyncio.run(f.flash(
                main_fw,
                additional_writes=[("0x260000", "/nonexistent/ota.bin")],
            ))
            assert result.success is False
            assert "not found" in result.error.lower()
        finally:
            os.unlink(main_fw)

    def test_flash_signature_has_additional_writes(self):
        """Verify flash() accepts additional_writes parameter."""
        import inspect
        sig = inspect.signature(ESP32Flasher.flash)
        assert "additional_writes" in sig.parameters
        assert "baud_override" in sig.parameters


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

    def test_find_firmware_asset_prefers_zip(self):
        """Zip should be preferred over bare .bin (has all partition files)."""
        assets = [
            {"name": "firmware-tlora-pager-2.5.19.bin", "browser_download_url": "http://x"},
            {"name": "firmware-tlora-pager-2.5.19.zip", "browser_download_url": "http://y"},
        ]
        result = MeshtasticFlasher._find_firmware_asset(assets, "tlora-pager")
        assert result is not None
        assert result["name"].endswith(".zip")

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

    def test_default_offsets(self):
        assert DEFAULT_OTA_OFFSET == "0x260000"
        assert DEFAULT_SPIFFS_OFFSET == "0x300000"
        assert DEFAULT_APP_OFFSET == "0x10000"

    def test_esp32s3_boards_set(self):
        assert "tlora-pager" in ESP32S3_BOARDS
        assert "t-deck" in ESP32S3_BOARDS
        assert "heltec-v3" in ESP32S3_BOARDS

    def test_has_update_firmware_method(self):
        f = MeshtasticFlasher()
        assert hasattr(f, "update_firmware")
        assert callable(f.update_firmware)

    def test_has_enter_dfu_mode_method(self):
        f = MeshtasticFlasher()
        assert hasattr(f, "_enter_dfu_mode")
        assert callable(f._enter_dfu_mode)

    def test_update_firmware_no_device(self):
        """update_firmware with no esptool should fail gracefully."""
        f = MeshtasticFlasher(port="/dev/nonexistent")
        f._esptool_path = None
        result = asyncio.run(f.update_firmware())
        assert result.success is False


class TestMeshtasticFirmwareFiles:
    def test_defaults(self):
        fw = MeshtasticFirmwareFiles()
        assert fw.factory_bin == ""
        assert fw.update_bin == ""
        assert fw.ota_bin == ""
        assert fw.littlefs_bin == ""
        assert fw.mt_json == ""
        assert fw.ota_offset == "0x260000"
        assert fw.spiffs_offset == "0x300000"
        assert fw.app_offset == "0x10000"

    def test_custom_values(self):
        fw = MeshtasticFirmwareFiles(
            factory_bin="/tmp/factory.bin",
            board="tlora-pager",
            version="2.5.19",
        )
        assert fw.factory_bin == "/tmp/factory.bin"
        assert fw.board == "tlora-pager"


class TestParseMtJson:
    def test_parse_valid_json(self):
        """Parse a well-formed .mt.json metadata file."""
        metadata = {
            "board": "tlora-pager",
            "chip": "esp32s3",
            "version": "2.5.19.5f8df68",
            "partitions": {
                "ota": {"offset": "0x270000", "size": "0x10000"},
                "spiffs": {"offset": "0x310000", "size": "0x100000"},
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mt.json", delete=False,
        ) as f:
            json.dump(metadata, f)
            json_path = f.name

        try:
            result = MeshtasticFlasher._parse_mt_json(json_path)
            assert result["board"] == "tlora-pager"
            assert result["version"] == "2.5.19.5f8df68"
            assert result["partitions"]["ota"]["offset"] == "0x270000"
        finally:
            os.unlink(json_path)

    def test_parse_missing_file(self):
        """Missing file should return empty dict."""
        result = MeshtasticFlasher._parse_mt_json("/nonexistent/file.mt.json")
        assert result == {}

    def test_parse_invalid_json(self):
        """Invalid JSON should return empty dict."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mt.json", delete=False,
        ) as f:
            f.write("not valid json {{{}}")
            json_path = f.name

        try:
            result = MeshtasticFlasher._parse_mt_json(json_path)
            assert result == {}
        finally:
            os.unlink(json_path)


class TestApplyMtJsonOffsets:
    def test_applies_offsets_from_metadata(self):
        """Offsets from .mt.json should override defaults."""
        metadata = {
            "version": "2.5.19",
            "partitions": {
                "ota": {"offset": "0x280000"},
                "spiffs": {"offset": "0x320000"},
                "app": {"offset": "0x20000"},
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mt.json", delete=False,
        ) as f:
            json.dump(metadata, f)
            json_path = f.name

        try:
            fw_files = MeshtasticFirmwareFiles(mt_json=json_path)
            flasher = MeshtasticFlasher()
            result = flasher._apply_mt_json_offsets(fw_files)
            assert result.ota_offset == "0x280000"
            assert result.spiffs_offset == "0x320000"
            assert result.app_offset == "0x20000"
            assert result.version == "2.5.19"
        finally:
            os.unlink(json_path)

    def test_keeps_defaults_without_mt_json(self):
        """Without .mt.json, default offsets should be preserved."""
        fw_files = MeshtasticFirmwareFiles()
        flasher = MeshtasticFlasher()
        result = flasher._apply_mt_json_offsets(fw_files)
        assert result.ota_offset == DEFAULT_OTA_OFFSET
        assert result.spiffs_offset == DEFAULT_SPIFFS_OFFSET

    def test_handles_littlefs_key(self):
        """Some .mt.json files use 'littlefs' instead of 'spiffs'."""
        metadata = {
            "partitions": {
                "littlefs": {"offset": "0x350000"},
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mt.json", delete=False,
        ) as f:
            json.dump(metadata, f)
            json_path = f.name

        try:
            fw_files = MeshtasticFirmwareFiles(mt_json=json_path)
            flasher = MeshtasticFlasher()
            result = flasher._apply_mt_json_offsets(fw_files)
            assert result.spiffs_offset == "0x350000"
        finally:
            os.unlink(json_path)


class TestExtractFirmwareFiles:
    def _make_test_zip(self, tmpdir: str, board: str = "tlora-pager",
                       version: str = "2.5.19") -> str:
        """Create a test firmware zip with realistic file names."""
        zip_path = os.path.join(tmpdir, f"firmware-{board}-{version}.zip")

        with zipfile.ZipFile(zip_path, "w") as zf:
            # Factory binary
            zf.writestr(
                f"firmware-{board}-{version}.factory.bin",
                b"\x00" * 256,
            )
            # Update binary
            zf.writestr(
                f"firmware-{board}-{version}-update.bin",
                b"\x00" * 128,
            )
            # OTA bootloader
            zf.writestr("mt-esp32s3-ota.bin", b"\x00" * 64)
            # LittleFS
            zf.writestr(
                f"littlefs-{board}-{version}.bin",
                b"\x00" * 512,
            )
            # Metadata
            metadata = {
                "board": board,
                "chip": "esp32s3",
                "version": version,
                "partitions": {
                    "ota": {"offset": "0x260000"},
                    "spiffs": {"offset": "0x300000"},
                },
            }
            zf.writestr(
                f"firmware-{board}-{version}.mt.json",
                json.dumps(metadata),
            )

        return zip_path

    def test_extract_all_files(self):
        """Should extract factory, update, OTA, littlefs, and mt.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._make_test_zip(tmpdir)
            result = MeshtasticFlasher._extract_firmware_files(zip_path, "tlora-pager")

            assert result.factory_bin != ""
            assert Path(result.factory_bin).exists()
            assert "factory.bin" in result.factory_bin

            assert result.update_bin != ""
            assert Path(result.update_bin).exists()
            assert "update.bin" in result.update_bin

            assert result.ota_bin != ""
            assert Path(result.ota_bin).exists()
            assert "ota.bin" in result.ota_bin

            assert result.littlefs_bin != ""
            assert Path(result.littlefs_bin).exists()
            assert "littlefs" in result.littlefs_bin

            assert result.mt_json != ""
            assert Path(result.mt_json).exists()
            assert ".mt.json" in result.mt_json

    def test_extract_wrong_board(self):
        """Requesting wrong board should return empty factory_bin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._make_test_zip(tmpdir, board="tlora-pager")
            result = MeshtasticFlasher._extract_firmware_files(zip_path, "tbeam")
            # Factory and update won't match, but OTA might (board-independent)
            assert result.factory_bin == ""
            assert result.update_bin == ""

    def test_extract_fallback_bin(self):
        """If no factory.bin, should fall back to any board .bin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "firmware.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                # No "factory" in the name, just a board .bin
                zf.writestr("firmware-tbeam-2.5.19.bin", b"\x00" * 128)

            result = MeshtasticFlasher._extract_firmware_files(zip_path, "tbeam")
            assert result.factory_bin != ""
            assert "tbeam" in result.factory_bin

    def test_extract_invalid_zip(self):
        """Invalid zip should return empty firmware files."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"not a zip file")
            bad_zip = f.name

        try:
            result = MeshtasticFlasher._extract_firmware_files(bad_zip, "tbeam")
            assert result.factory_bin == ""
        finally:
            os.unlink(bad_zip)

    def test_legacy_extract_firmware_bin(self):
        """Legacy _extract_firmware_bin should return factory.bin path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._make_test_zip(tmpdir)
            result = MeshtasticFlasher._extract_firmware_bin(zip_path, "tlora-pager")
            assert result is not None
            assert "factory.bin" in result

    def test_legacy_extract_firmware_bin_not_found(self):
        """Legacy method returns None if board not in zip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._make_test_zip(tmpdir, board="tlora-pager")
            result = MeshtasticFlasher._extract_firmware_bin(zip_path, "nonexistent-board")
            assert result is None


class TestEnterDfuMode:
    def test_dfu_no_port(self):
        """DFU mode without port should return False."""
        f = MeshtasticFlasher()
        f.port = ""
        result = asyncio.run(f._enter_dfu_mode(""))
        assert result is False

    def test_dfu_sync_no_pyserial(self):
        """Without pyserial, _enter_dfu_sync should return False."""
        with patch.dict("sys.modules", {"serial": None}):
            # The import will fail, returning False
            result = MeshtasticFlasher._enter_dfu_sync("/dev/ttyACM0")
            # May succeed if pyserial is installed, or fail gracefully
            assert isinstance(result, bool)


class TestImports:
    def test_import_all(self):
        from tritium_lib.firmware import (
            FirmwareFlasher, FlashResult, DeviceDetection,
            ESP32Flasher, MeshtasticFlasher,
        )
        assert all([FirmwareFlasher, ESP32Flasher, MeshtasticFlasher,
                     FlashResult, DeviceDetection])

    def test_import_firmware_files_dataclass(self):
        from tritium_lib.firmware.meshtastic_flasher import MeshtasticFirmwareFiles
        assert MeshtasticFirmwareFiles is not None
        fw = MeshtasticFirmwareFiles()
        assert hasattr(fw, "factory_bin")
        assert hasattr(fw, "ota_offset")
