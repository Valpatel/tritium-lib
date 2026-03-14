# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ExportManifest, ExportPackage, and related models."""

import pytest

from tritium_lib.models.export import (
    ExportFormat,
    ExportManifest,
    ExportPackage,
    ExportScope,
    ExportSection,
    ExportSectionType,
    ImportResult,
    create_export_manifest,
    validate_import_compatibility,
)


class TestExportSection:
    def test_create_default(self):
        s = ExportSection(section_type=ExportSectionType.AUTOMATION_RULES)
        assert s.section_type == ExportSectionType.AUTOMATION_RULES
        assert s.item_count == 0
        assert s.data == []

    def test_with_data(self):
        items = [{"name": "rule1"}, {"name": "rule2"}]
        s = ExportSection(
            section_type=ExportSectionType.AUTOMATION_RULES,
            item_count=2,
            data=items,
        )
        assert s.item_count == 2
        assert len(s.data) == 2


class TestExportManifest:
    def test_create_default(self):
        m = ExportManifest()
        assert m.export_id  # auto-generated UUID
        assert m.created_at > 0
        assert m.scope == ExportScope.CUSTOM
        assert m.format == ExportFormat.JSON

    def test_create_with_metadata(self):
        m = create_export_manifest(
            scope=ExportScope.RULES,
            site_id="site-1",
            site_name="HQ",
            description="Weekly rules backup",
            tags=["backup", "rules"],
        )
        assert m.scope == ExportScope.RULES
        assert m.source_site_id == "site-1"
        assert m.source_site_name == "HQ"
        assert "backup" in m.tags


class TestExportPackage:
    def test_add_section(self):
        pkg = ExportPackage()
        rules = [
            {"rule_id": "r1", "name": "Alert on unknown"},
            {"rule_id": "r2", "name": "Escalate signal"},
        ]
        pkg.add_section(ExportSectionType.AUTOMATION_RULES, rules)

        assert len(pkg.sections) == 1
        assert pkg.manifest.total_items == 2
        assert ExportSectionType.AUTOMATION_RULES in pkg.manifest.sections

    def test_get_section(self):
        pkg = ExportPackage()
        pkg.add_section(ExportSectionType.GEOFENCES, [{"id": "z1"}])
        pkg.add_section(ExportSectionType.DOSSIERS, [{"id": "d1"}, {"id": "d2"}])

        s = pkg.get_section(ExportSectionType.DOSSIERS)
        assert s is not None
        assert s.item_count == 2

        assert pkg.get_section(ExportSectionType.THREAT_FEEDS) is None

    def test_section_types(self):
        pkg = ExportPackage()
        pkg.add_section(ExportSectionType.AUTOMATION_RULES, [])
        pkg.add_section(ExportSectionType.KNOWN_DEVICES, [{"mac": "AA:BB:CC"}])
        types = pkg.section_types()
        assert ExportSectionType.AUTOMATION_RULES in types
        assert ExportSectionType.KNOWN_DEVICES in types

    def test_roundtrip_json(self):
        pkg = ExportPackage(
            manifest=create_export_manifest(ExportScope.FULL, site_id="s1")
        )
        pkg.add_section(
            ExportSectionType.AUTOMATION_RULES,
            [{"rule_id": "r1", "name": "test"}],
        )
        json_str = pkg.model_dump_json()
        restored = ExportPackage.model_validate_json(json_str)
        assert restored.manifest.source_site_id == "s1"
        assert len(restored.sections) == 1
        assert restored.sections[0].data[0]["rule_id"] == "r1"


class TestImportResult:
    def test_default_success(self):
        r = ImportResult()
        assert r.success is True
        assert r.items_imported == 0


class TestValidateImportCompatibility:
    def test_all_supported(self):
        m = ExportManifest(sections=[
            ExportSectionType.AUTOMATION_RULES,
            ExportSectionType.GEOFENCES,
        ])
        result = validate_import_compatibility(m)
        assert result.success is True
        assert len(result.sections_imported) == 2
        assert len(result.warnings) == 0

    def test_unsupported_sections(self):
        m = ExportManifest(sections=[
            ExportSectionType.AUTOMATION_RULES,
            ExportSectionType.DOSSIERS,
        ])
        supported = {ExportSectionType.AUTOMATION_RULES}
        result = validate_import_compatibility(m, supported)
        assert result.success is True
        assert "dossiers" in result.sections_skipped
        assert len(result.warnings) == 1

    def test_encrypted_not_supported(self):
        m = ExportManifest(encrypted=True)
        result = validate_import_compatibility(m)
        assert result.success is False
        assert any("ncrypt" in e for e in result.errors)
