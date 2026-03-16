# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ReportTemplate models."""

import pytest
from tritium_lib.models.template import (
    BRIEFING_TEMPLATE,
    BUILTIN_TEMPLATES,
    INVESTIGATION_TEMPLATE,
    ReportFormat,
    ReportTemplate,
    SITREP_TEMPLATE,
    TemplateSection,
    TemplateSectionType,
    TemplateVariable,
)


class TestTemplateVariable:
    def test_create_default(self):
        v = TemplateVariable()
        assert v.name == ""
        assert v.var_type == "text"
        assert v.required is False
        assert v.source == ""

    def test_create_with_values(self):
        v = TemplateVariable(
            name="dtg",
            label="Date-Time Group",
            var_type="date",
            required=True,
            source="auto",
        )
        assert v.name == "dtg"
        assert v.label == "Date-Time Group"
        assert v.var_type == "date"
        assert v.required is True


class TestTemplateSection:
    def test_create_default(self):
        s = TemplateSection()
        assert s.section_id == ""
        assert s.section_type == TemplateSectionType.CUSTOM
        assert s.order == 0
        assert s.optional is False

    def test_create_with_type(self):
        s = TemplateSection(
            section_id="findings",
            title="FINDINGS",
            section_type=TemplateSectionType.FINDINGS,
            body_template="{{findings_text}}",
            order=3,
        )
        assert s.section_type == TemplateSectionType.FINDINGS
        assert s.order == 3


class TestReportTemplate:
    def test_create_empty(self):
        t = ReportTemplate()
        assert t.template_id == ""
        assert t.format == ReportFormat.MARKDOWN
        assert t.sections == []
        assert t.variables == []
        assert t.version == 1

    def test_add_section(self):
        t = ReportTemplate(template_id="test_1", name="Test")
        s = TemplateSection(section_id="s1", title="Section 1")
        t.add_section(s)
        assert len(t.sections) == 1
        assert t.sections[0].order == 1
        assert t.updated_at is not None

    def test_add_variable(self):
        t = ReportTemplate(template_id="test_2")
        v = TemplateVariable(name="target_count", required=True)
        t.add_variable(v)
        assert len(t.variables) == 1
        required = t.get_required_variables()
        assert len(required) == 1
        assert required[0].name == "target_count"

    def test_section_ordering(self):
        t = ReportTemplate()
        t.add_section(TemplateSection(section_id="c", order=3))
        t.add_section(TemplateSection(section_id="a", order=1))
        t.add_section(TemplateSection(section_id="b", order=2))
        ordered = t.get_section_order()
        assert [s.section_id for s in ordered] == ["a", "b", "c"]

    def test_render_preview(self):
        t = ReportTemplate()
        t.add_variable(TemplateVariable(name="name", default_value="TEST"))
        t.add_section(TemplateSection(
            section_id="intro",
            title="Intro",
            body_template="Hello {{name}}!",
            order=1,
        ))

        # With default values
        output = t.render_preview()
        assert "Hello TEST!" in output
        assert "## Intro" in output

        # With provided values
        output = t.render_preview({"name": "World"})
        assert "Hello World!" in output

    def test_render_preview_missing_var(self):
        t = ReportTemplate()
        t.add_variable(TemplateVariable(name="missing_var"))
        t.add_section(TemplateSection(
            section_id="s1",
            body_template="Count: {{missing_var}}",
            order=1,
        ))
        output = t.render_preview()
        assert "[missing_var]" in output

    def test_report_format_enum(self):
        assert ReportFormat.MARKDOWN == "markdown"
        assert ReportFormat.HTML == "html"
        assert ReportFormat.PDF == "pdf"
        assert ReportFormat.COT_XML == "cot_xml"

    def test_section_type_enum(self):
        assert TemplateSectionType.HEADER == "header"
        assert TemplateSectionType.TARGETS == "targets"
        assert TemplateSectionType.MAP_SNAPSHOT == "map_snapshot"


class TestBuiltinTemplates:
    def test_sitrep_template(self):
        t = SITREP_TEMPLATE
        assert t.template_id == "builtin_sitrep"
        assert t.name == "SITREP"
        assert len(t.sections) == 4
        assert len(t.variables) >= 5
        assert "sitrep" in t.tags

    def test_briefing_template(self):
        t = BRIEFING_TEMPLATE
        assert t.template_id == "builtin_briefing"
        assert t.name == "Mission Briefing"
        assert len(t.sections) == 4
        required = t.get_required_variables()
        assert len(required) >= 2

    def test_investigation_template(self):
        t = INVESTIGATION_TEMPLATE
        assert t.template_id == "builtin_investigation"
        assert len(t.sections) == 4
        required = t.get_required_variables()
        assert any(v.name == "incident_id" for v in required)

    def test_builtin_list(self):
        assert len(BUILTIN_TEMPLATES) == 3
        ids = {t.template_id for t in BUILTIN_TEMPLATES}
        assert "builtin_sitrep" in ids
        assert "builtin_briefing" in ids
        assert "builtin_investigation" in ids

    def test_sitrep_render(self):
        output = SITREP_TEMPLATE.render_preview({
            "dtg": "141200ZMAR2026",
            "location": "Base Alpha",
            "situation_summary": "All quiet.",
            "hostile_count": "0",
        })
        assert "141200ZMAR2026" in output
        assert "All quiet." in output

    def test_serialization_roundtrip(self):
        """Verify templates survive Pydantic serialization."""
        for t in BUILTIN_TEMPLATES:
            data = t.model_dump()
            restored = ReportTemplate(**data)
            assert restored.template_id == t.template_id
            assert len(restored.sections) == len(t.sections)
            assert len(restored.variables) == len(t.variables)
