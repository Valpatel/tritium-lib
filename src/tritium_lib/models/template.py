# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Report template models for customizable report generation.

A ReportTemplate defines the structure, sections, and variable placeholders
for generating standardized reports (SITREP, briefing, investigation).
Templates are reusable across missions and can be shared between operators.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReportFormat(str, Enum):
    """Output format for generated reports."""
    PLAINTEXT = "plaintext"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    COT_XML = "cot_xml"  # Cursor on Target XML for TAK interop


class TemplateSectionType(str, Enum):
    """Type of section in a report template."""
    HEADER = "header"
    SUMMARY = "summary"
    FINDINGS = "findings"
    TIMELINE = "timeline"
    TARGETS = "targets"
    RECOMMENDATIONS = "recommendations"
    APPENDIX = "appendix"
    MAP_SNAPSHOT = "map_snapshot"
    SENSOR_DATA = "sensor_data"
    CUSTOM = "custom"


class TemplateVariable(BaseModel):
    """A variable placeholder within a report template.

    Variables are resolved at render time from live system state,
    operator input, or data queries.
    """
    name: str = ""
    label: str = ""
    description: str = ""
    var_type: str = "text"  # text, number, date, target_id, list, boolean
    default_value: str = ""
    required: bool = False
    source: str = ""  # auto-fill source: "tracker", "fleet", "dossier", "operator", ""

    model_config = {"frozen": False}


class TemplateSection(BaseModel):
    """A section within a report template.

    Each section has a type, title, and body template with variable
    placeholders using {{variable_name}} syntax.
    """
    section_id: str = ""
    title: str = ""
    section_type: TemplateSectionType = TemplateSectionType.CUSTOM
    body_template: str = ""  # Template text with {{variable}} placeholders
    order: int = 0
    optional: bool = False
    variables: list[str] = Field(
        default_factory=list,
        description="Variable names used in this section",
    )

    model_config = {"frozen": False}


class ReportTemplate(BaseModel):
    """A reusable report template for structured report generation.

    Templates define the sections, format, and variable placeholders
    for generating reports like SITREPs, briefings, and investigation
    summaries.  They can be saved, shared, and versioned across missions.
    """
    template_id: str = ""
    name: str = ""
    description: str = ""
    format: ReportFormat = ReportFormat.MARKDOWN
    sections: list[TemplateSection] = Field(default_factory=list)
    variables: list[TemplateVariable] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    version: int = 1
    created_by: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"frozen": False}

    def add_section(self, section: TemplateSection) -> None:
        """Append a section to the template."""
        if not section.order:
            section.order = len(self.sections) + 1
        self.sections.append(section)
        self.updated_at = datetime.now(timezone.utc)

    def add_variable(self, variable: TemplateVariable) -> None:
        """Register a variable for this template."""
        self.variables.append(variable)
        self.updated_at = datetime.now(timezone.utc)

    def get_required_variables(self) -> list[TemplateVariable]:
        """Return all required variables."""
        return [v for v in self.variables if v.required]

    def get_section_order(self) -> list[TemplateSection]:
        """Return sections sorted by order."""
        return sorted(self.sections, key=lambda s: s.order)

    def render_preview(self, values: dict[str, str] | None = None) -> str:
        """Render a preview of the template with provided or default values.

        Substitutes {{variable}} placeholders with values from the dict,
        falling back to default_value or the placeholder name itself.
        """
        if values is None:
            values = {}

        # Build value lookup with defaults
        lookup: dict[str, str] = {}
        for var in self.variables:
            lookup[var.name] = values.get(var.name, var.default_value or f"[{var.name}]")

        lines: list[str] = []
        for section in self.get_section_order():
            if section.title:
                lines.append(f"## {section.title}")
            body = section.body_template
            for var_name, var_val in lookup.items():
                body = body.replace(f"{{{{{var_name}}}}}", var_val)
            lines.append(body)
            lines.append("")

        return "\n".join(lines)


# -- Built-in templates -------------------------------------------------------

SITREP_TEMPLATE = ReportTemplate(
    template_id="builtin_sitrep",
    name="SITREP",
    description="Situation Report — periodic operational status update",
    format=ReportFormat.MARKDOWN,
    sections=[
        TemplateSection(
            section_id="situation",
            title="SITUATION",
            section_type=TemplateSectionType.SUMMARY,
            body_template="DTG: {{dtg}}\nLocation: {{location}}\n\n{{situation_summary}}",
            order=1,
            variables=["dtg", "location", "situation_summary"],
        ),
        TemplateSection(
            section_id="targets",
            title="ENEMY / THREAT FORCES",
            section_type=TemplateSectionType.TARGETS,
            body_template="Active hostile targets: {{hostile_count}}\nNew contacts: {{new_contacts}}\n\n{{threat_assessment}}",
            order=2,
            variables=["hostile_count", "new_contacts", "threat_assessment"],
        ),
        TemplateSection(
            section_id="friendly",
            title="FRIENDLY FORCES",
            section_type=TemplateSectionType.SENSOR_DATA,
            body_template="Nodes online: {{nodes_online}}\nSensors active: {{sensors_active}}\n\n{{friendly_status}}",
            order=3,
            variables=["nodes_online", "sensors_active", "friendly_status"],
        ),
        TemplateSection(
            section_id="recommendations",
            title="RECOMMENDATIONS",
            section_type=TemplateSectionType.RECOMMENDATIONS,
            body_template="{{recommendations}}",
            order=4,
            variables=["recommendations"],
        ),
    ],
    variables=[
        TemplateVariable(name="dtg", label="Date-Time Group", var_type="date", source="auto", required=True),
        TemplateVariable(name="location", label="Location", var_type="text", source="auto"),
        TemplateVariable(name="situation_summary", label="Situation Summary", var_type="text", required=True),
        TemplateVariable(name="hostile_count", label="Hostile Count", var_type="number", source="tracker"),
        TemplateVariable(name="new_contacts", label="New Contacts", var_type="number", source="tracker"),
        TemplateVariable(name="threat_assessment", label="Threat Assessment", var_type="text"),
        TemplateVariable(name="nodes_online", label="Nodes Online", var_type="number", source="fleet"),
        TemplateVariable(name="sensors_active", label="Sensors Active", var_type="number", source="fleet"),
        TemplateVariable(name="friendly_status", label="Friendly Status", var_type="text"),
        TemplateVariable(name="recommendations", label="Recommendations", var_type="text"),
    ],
    tags=["sitrep", "operational", "builtin"],
    created_by="system",
)

BRIEFING_TEMPLATE = ReportTemplate(
    template_id="builtin_briefing",
    name="Mission Briefing",
    description="Pre-mission briefing with objectives, threats, and assets",
    format=ReportFormat.MARKDOWN,
    sections=[
        TemplateSection(
            section_id="mission",
            title="MISSION",
            section_type=TemplateSectionType.HEADER,
            body_template="Mission: {{mission_name}}\nObjective: {{objective}}\nStart: {{start_time}}",
            order=1,
            variables=["mission_name", "objective", "start_time"],
        ),
        TemplateSection(
            section_id="area_of_ops",
            title="AREA OF OPERATIONS",
            section_type=TemplateSectionType.MAP_SNAPSHOT,
            body_template="Center: {{map_center}}\nRadius: {{ops_radius}}\n\n{{terrain_notes}}",
            order=2,
            variables=["map_center", "ops_radius", "terrain_notes"],
        ),
        TemplateSection(
            section_id="known_threats",
            title="KNOWN THREATS",
            section_type=TemplateSectionType.TARGETS,
            body_template="{{threat_summary}}",
            order=3,
            variables=["threat_summary"],
        ),
        TemplateSection(
            section_id="assets",
            title="AVAILABLE ASSETS",
            section_type=TemplateSectionType.SENSOR_DATA,
            body_template="{{asset_inventory}}",
            order=4,
            variables=["asset_inventory"],
        ),
    ],
    variables=[
        TemplateVariable(name="mission_name", label="Mission Name", var_type="text", required=True),
        TemplateVariable(name="objective", label="Objective", var_type="text", required=True),
        TemplateVariable(name="start_time", label="Start Time", var_type="date"),
        TemplateVariable(name="map_center", label="Map Center", var_type="text", source="auto"),
        TemplateVariable(name="ops_radius", label="Ops Radius (m)", var_type="number", default_value="500"),
        TemplateVariable(name="terrain_notes", label="Terrain Notes", var_type="text"),
        TemplateVariable(name="threat_summary", label="Threat Summary", var_type="text", source="tracker"),
        TemplateVariable(name="asset_inventory", label="Asset Inventory", var_type="text", source="fleet"),
    ],
    tags=["briefing", "mission", "builtin"],
    created_by="system",
)

INVESTIGATION_TEMPLATE = ReportTemplate(
    template_id="builtin_investigation",
    name="Investigation Report",
    description="Post-incident investigation with evidence and conclusions",
    format=ReportFormat.MARKDOWN,
    sections=[
        TemplateSection(
            section_id="incident",
            title="INCIDENT OVERVIEW",
            section_type=TemplateSectionType.SUMMARY,
            body_template="Incident ID: {{incident_id}}\nDate: {{incident_date}}\nType: {{incident_type}}\n\n{{incident_description}}",
            order=1,
            variables=["incident_id", "incident_date", "incident_type", "incident_description"],
        ),
        TemplateSection(
            section_id="timeline",
            title="TIMELINE OF EVENTS",
            section_type=TemplateSectionType.TIMELINE,
            body_template="{{event_timeline}}",
            order=2,
            variables=["event_timeline"],
        ),
        TemplateSection(
            section_id="evidence",
            title="EVIDENCE",
            section_type=TemplateSectionType.FINDINGS,
            body_template="{{evidence_list}}",
            order=3,
            variables=["evidence_list"],
        ),
        TemplateSection(
            section_id="conclusions",
            title="CONCLUSIONS",
            section_type=TemplateSectionType.RECOMMENDATIONS,
            body_template="{{conclusions}}\n\nRecommended actions:\n{{recommended_actions}}",
            order=4,
            variables=["conclusions", "recommended_actions"],
        ),
    ],
    variables=[
        TemplateVariable(name="incident_id", label="Incident ID", var_type="text", required=True),
        TemplateVariable(name="incident_date", label="Incident Date", var_type="date", required=True),
        TemplateVariable(name="incident_type", label="Incident Type", var_type="text"),
        TemplateVariable(name="incident_description", label="Description", var_type="text", required=True),
        TemplateVariable(name="event_timeline", label="Event Timeline", var_type="text", source="dossier"),
        TemplateVariable(name="evidence_list", label="Evidence", var_type="text"),
        TemplateVariable(name="conclusions", label="Conclusions", var_type="text", required=True),
        TemplateVariable(name="recommended_actions", label="Recommended Actions", var_type="text"),
    ],
    tags=["investigation", "post-incident", "builtin"],
    created_by="system",
)

BUILTIN_TEMPLATES = [SITREP_TEMPLATE, BRIEFING_TEMPLATE, INVESTIGATION_TEMPLATE]
