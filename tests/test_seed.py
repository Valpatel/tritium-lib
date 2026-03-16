"""Tests for tritium_lib.models.seed."""

from datetime import datetime, timezone

from tritium_lib.models.seed import (
    SeedFile,
    SeedManifest,
    SeedPackage,
    SeedStatus,
    SeedTransfer,
    SeedTransferStatus,
)


class TestSeedFile:
    def test_create(self):
        f = SeedFile(
            path="firmware/touch-lcd-35bc.bin",
            size_bytes=1_200_000,
            sha256="abcdef1234567890",
            board="touch-lcd-35bc",
        )
        assert f.path == "firmware/touch-lcd-35bc.bin"
        assert f.size_bytes == 1_200_000
        assert f.board == "touch-lcd-35bc"

    def test_universal_file(self):
        f = SeedFile(path="config/defaults.json", size_bytes=512)
        assert f.board == "any"

    def test_json_roundtrip(self):
        f = SeedFile(path="fw.bin", size_bytes=100, sha256="abc")
        f2 = SeedFile.model_validate_json(f.model_dump_json())
        assert f2.path == f.path
        assert f2.sha256 == f.sha256


class TestSeedManifest:
    def test_create(self):
        m = SeedManifest(
            package_id="seed-001",
            firmware_version="2.1.0",
            boards=["touch-lcd-35bc", "touch-amoled-241b"],
            files=[
                SeedFile(path="fw-35bc.bin", size_bytes=1000, board="touch-lcd-35bc"),
                SeedFile(path="fw-241b.bin", size_bytes=2000, board="touch-amoled-241b"),
                SeedFile(path="config.json", size_bytes=100, board="any"),
            ],
            total_size_bytes=3100,
        )
        assert m.file_count == 3
        assert m.firmware_version == "2.1.0"

    def test_files_for_board(self):
        m = SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["board-a", "board-b"],
            files=[
                SeedFile(path="a.bin", size_bytes=100, board="board-a"),
                SeedFile(path="b.bin", size_bytes=200, board="board-b"),
                SeedFile(path="shared.json", size_bytes=50, board="any"),
            ],
        )
        a_files = m.files_for_board("board-a")
        assert len(a_files) == 2  # a.bin + shared.json
        assert all(f.board in ("board-a", "any") for f in a_files)

    def test_is_compatible(self):
        m = SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["touch-lcd-35bc"],
        )
        assert m.is_compatible("touch-lcd-35bc") is True
        assert m.is_compatible("amoled-191m") is False

    def test_is_compatible_any(self):
        m = SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["any"],
        )
        assert m.is_compatible("anything") is True

    def test_json_roundtrip(self):
        m = SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["b1"],
            files=[SeedFile(path="f.bin", size_bytes=100)],
            total_size_bytes=100,
        )
        m2 = SeedManifest.model_validate_json(m.model_dump_json())
        assert m2.package_id == m.package_id
        assert m2.file_count == 1


class TestSeedPackage:
    def test_create(self):
        manifest = SeedManifest(
            package_id="seed-001",
            firmware_version="2.0.0",
            boards=["touch-lcd-35bc"],
            total_size_bytes=5000,
        )
        pkg = SeedPackage(
            id="pkg-001",
            manifest=manifest,
            source_device="esp32-001",
        )
        assert pkg.status == SeedStatus.CREATED
        assert pkg.total_size == 5000
        assert pkg.distribution_count == 0

    def test_distribution(self):
        manifest = SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["b1"],
            total_size_bytes=1000,
        )
        pkg = SeedPackage(
            id="pkg-001",
            manifest=manifest,
            status=SeedStatus.DISTRIBUTING,
            distributed_to=["esp32-002", "esp32-003"],
            distribution_count=2,
        )
        assert pkg.status == SeedStatus.DISTRIBUTING
        assert len(pkg.distributed_to) == 2

    def test_failed_status(self):
        manifest = SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["b1"],
        )
        pkg = SeedPackage(
            id="pkg-001",
            manifest=manifest,
            status=SeedStatus.FAILED,
            error="Checksum mismatch on target device",
        )
        assert pkg.status == SeedStatus.FAILED
        assert "Checksum" in pkg.error

    def test_json_roundtrip(self):
        manifest = SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["b1"],
            total_size_bytes=500,
        )
        pkg = SeedPackage(id="pkg-001", manifest=manifest)
        pkg2 = SeedPackage.model_validate_json(pkg.model_dump_json())
        assert pkg2.id == pkg.id
        assert pkg2.manifest.firmware_version == "1.0.0"
        assert pkg2.total_size == 500


class TestSeedTransfer:
    def _manifest(self, size: int = 1000) -> SeedManifest:
        return SeedManifest(
            package_id="seed-001",
            firmware_version="1.0.0",
            boards=["b1"],
            total_size_bytes=size,
        )

    def test_create(self):
        transfer = SeedTransfer(
            id="xfer-001",
            source_node="esp32-001",
            dest_node="esp32-002",
            manifest=self._manifest(5000),
        )
        assert transfer.status == SeedTransferStatus.PENDING
        assert transfer.progress == 0.0
        assert transfer.remaining_bytes == 5000

    def test_in_progress(self):
        transfer = SeedTransfer(
            id="xfer-001",
            source_node="esp32-001",
            dest_node="esp32-002",
            manifest=self._manifest(10000),
            status=SeedTransferStatus.TRANSFERRING,
            progress=0.5,
            bytes_transferred=5000,
        )
        assert transfer.is_active is True
        assert transfer.remaining_bytes == 5000

    def test_complete(self):
        transfer = SeedTransfer(
            id="xfer-001",
            source_node="esp32-001",
            dest_node="esp32-002",
            manifest=self._manifest(1000),
            status=SeedTransferStatus.COMPLETE,
            progress=1.0,
            bytes_transferred=1000,
        )
        assert transfer.is_active is False
        assert transfer.remaining_bytes == 0

    def test_failed(self):
        transfer = SeedTransfer(
            id="xfer-001",
            source_node="esp32-001",
            dest_node="esp32-002",
            manifest=self._manifest(),
            status=SeedTransferStatus.FAILED,
            error="CRC mismatch",
        )
        assert transfer.is_active is False
        assert "CRC" in transfer.error

    def test_json_roundtrip(self):
        transfer = SeedTransfer(
            id="xfer-001",
            source_node="esp32-001",
            dest_node="esp32-002",
            manifest=self._manifest(2000),
            bytes_transferred=500,
        )
        transfer2 = SeedTransfer.model_validate_json(transfer.model_dump_json())
        assert transfer2.id == "xfer-001"
        assert transfer2.remaining_bytes == 1500
