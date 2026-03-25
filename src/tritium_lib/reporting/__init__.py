# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.reporting — situation report generation from tracking data.

Generates structured situation reports, daily summaries, and incident reports
from the current tracking state and event history. Supports plain text, HTML,
and JSON output formats.

Usage
-----
    from tritium_lib.reporting import SitRepGenerator, DailySummary, IncidentReport
    from tritium_lib.tracking import TargetTracker
    from tritium_lib.store.event_store import EventStore

    tracker = TargetTracker()
    events = EventStore(":memory:")

    # Generate a situation report
    gen = SitRepGenerator(tracker=tracker, event_store=events)
    report = gen.generate()
    print(report.to_text())
    print(report.to_html())
    print(report.to_json())

    # Daily summary
    summary = DailySummary.from_tracker_and_events(tracker, events)
    print(summary.to_text())

    # Incident report
    incident = IncidentReport(
        title="Hostile detected in restricted zone",
        target_ids=["ble_aa11bb22"],
        events=[...],
    )
    print(incident.to_text())
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.store.event_store import EventStore, TacticalEvent, SEVERITY_LEVELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: float) -> str:
    """Format a unix/monotonic timestamp as ISO-like string."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))
    except (OSError, OverflowError, ValueError):
        return f"t={ts:.1f}"


def _pct(part: int, total: int) -> str:
    """Format a percentage string."""
    if total == 0:
        return "0.0%"
    return f"{100.0 * part / total:.1f}%"


def _source_from_target_id(target_id: str) -> str:
    """Infer source type from target ID prefix convention."""
    if target_id.startswith("ble_"):
        return "ble"
    elif target_id.startswith("wifi_"):
        return "wifi"
    elif target_id.startswith("det_"):
        return "camera"
    elif target_id.startswith("mesh_"):
        return "mesh"
    elif target_id.startswith("adsb_"):
        return "adsb"
    return "other"


# ---------------------------------------------------------------------------
# TargetBreakdown — counts targets by type / source / alliance
# ---------------------------------------------------------------------------

@dataclass
class TargetBreakdown:
    """Breakdown of tracked targets for reporting."""

    total: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_alliance: dict[str, int] = field(default_factory=dict)
    by_asset_type: dict[str, int] = field(default_factory=dict)
    fused_count: int = 0  # targets with multiple confirming sources

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_source": dict(self.by_source),
            "by_alliance": dict(self.by_alliance),
            "by_asset_type": dict(self.by_asset_type),
            "fused_count": self.fused_count,
        }

    @staticmethod
    def from_targets(targets: list[TrackedTarget]) -> TargetBreakdown:
        """Build a breakdown from a list of TrackedTarget objects."""
        by_source: dict[str, int] = {}
        by_alliance: dict[str, int] = {}
        by_asset_type: dict[str, int] = {}
        fused = 0

        for t in targets:
            src = t.source or "unknown"
            by_source[src] = by_source.get(src, 0) + 1

            alliance = t.alliance or "unknown"
            by_alliance[alliance] = by_alliance.get(alliance, 0) + 1

            atype = t.asset_type or "unknown"
            by_asset_type[atype] = by_asset_type.get(atype, 0) + 1

            if len(t.confirming_sources) > 1:
                fused += 1

        return TargetBreakdown(
            total=len(targets),
            by_source=by_source,
            by_alliance=by_alliance,
            by_asset_type=by_asset_type,
            fused_count=fused,
        )


# ---------------------------------------------------------------------------
# ThreatSummary — threat level distribution
# ---------------------------------------------------------------------------

@dataclass
class ThreatSummary:
    """Distribution of threat scores across targets."""

    total_assessed: int = 0
    high_threat: int = 0       # threat_score >= 0.7
    medium_threat: int = 0     # 0.3 <= threat_score < 0.7
    low_threat: int = 0        # 0.0 < threat_score < 0.3
    no_threat: int = 0         # threat_score == 0.0
    hostile_count: int = 0
    suspicious_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_assessed": self.total_assessed,
            "high_threat": self.high_threat,
            "medium_threat": self.medium_threat,
            "low_threat": self.low_threat,
            "no_threat": self.no_threat,
            "hostile_count": self.hostile_count,
            "suspicious_targets": list(self.suspicious_targets),
        }

    @staticmethod
    def from_targets(targets: list[TrackedTarget]) -> ThreatSummary:
        """Build a threat summary from tracked targets."""
        high = 0
        medium = 0
        low = 0
        none_ = 0
        hostile = 0
        suspicious: list[str] = []

        for t in targets:
            score = t.threat_score
            if score >= 0.7:
                high += 1
                suspicious.append(t.target_id)
            elif score >= 0.3:
                medium += 1
                suspicious.append(t.target_id)
            elif score > 0.0:
                low += 1
            else:
                none_ += 1

            if t.alliance == "hostile":
                hostile += 1

        return ThreatSummary(
            total_assessed=len(targets),
            high_threat=high,
            medium_threat=medium,
            low_threat=low,
            no_threat=none_,
            hostile_count=hostile,
            suspicious_targets=suspicious[:20],  # cap at 20 for readability
        )


# ---------------------------------------------------------------------------
# ZoneActivity — zone entry/exit summary from events
# ---------------------------------------------------------------------------

@dataclass
class ZoneActivity:
    """Summary of geofence zone activity."""

    total_events: int = 0
    entries: int = 0
    exits: int = 0
    zones_active: list[str] = field(default_factory=list)
    most_active_zone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "entries": self.entries,
            "exits": self.exits,
            "zones_active": list(self.zones_active),
            "most_active_zone": self.most_active_zone,
        }

    @staticmethod
    def from_events(events: list[TacticalEvent]) -> ZoneActivity:
        """Build zone activity from geofence-related events."""
        entries = 0
        exits = 0
        zone_counts: dict[str, int] = {}

        for ev in events:
            etype = ev.event_type.lower()
            if "geofence" in etype or "zone" in etype:
                if "enter" in etype:
                    entries += 1
                elif "exit" in etype:
                    exits += 1

                zone_name = ev.data.get("zone_name", "") or ev.data.get("zone_id", "")
                if zone_name:
                    zone_counts[zone_name] = zone_counts.get(zone_name, 0) + 1

        total = entries + exits
        zones = sorted(zone_counts.keys())
        most_active = max(zone_counts, key=zone_counts.get, default="") if zone_counts else ""

        return ZoneActivity(
            total_events=total,
            entries=entries,
            exits=exits,
            zones_active=zones,
            most_active_zone=most_active,
        )


# ---------------------------------------------------------------------------
# AnomalySummary — top anomalies from events
# ---------------------------------------------------------------------------

@dataclass
class AnomalySummary:
    """Summary of anomalies detected in the reporting period."""

    total_anomalies: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    top_anomalies: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_anomalies": self.total_anomalies,
            "by_severity": dict(self.by_severity),
            "top_anomalies": list(self.top_anomalies),
        }

    @staticmethod
    def from_events(events: list[TacticalEvent], max_top: int = 10) -> AnomalySummary:
        """Extract anomaly summary from events."""
        anomaly_events: list[TacticalEvent] = []
        for ev in events:
            etype = ev.event_type.lower()
            if "anomaly" in etype or ev.severity in ("warning", "error", "critical"):
                anomaly_events.append(ev)

        by_severity: dict[str, int] = {}
        for ev in anomaly_events:
            sev = ev.severity or "info"
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Top anomalies sorted by severity then timestamp
        severity_rank = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
        sorted_anomalies = sorted(
            anomaly_events,
            key=lambda e: (-severity_rank.get(e.severity, 0), -e.timestamp),
        )

        top: list[dict[str, Any]] = []
        for ev in sorted_anomalies[:max_top]:
            top.append({
                "event_id": ev.event_id,
                "event_type": ev.event_type,
                "severity": ev.severity,
                "summary": ev.summary,
                "timestamp": ev.timestamp,
                "target_id": ev.target_id,
            })

        return AnomalySummary(
            total_anomalies=len(anomaly_events),
            by_severity=by_severity,
            top_anomalies=top,
        )


# ---------------------------------------------------------------------------
# EventTimeline — significant events in chronological order
# ---------------------------------------------------------------------------

@dataclass
class EventTimeline:
    """Timeline of significant events for reporting."""

    events: list[dict[str, Any]] = field(default_factory=list)
    total_events: int = 0
    period_start: float = 0.0
    period_end: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": list(self.events),
            "total_events": self.total_events,
            "period_start": self.period_start,
            "period_end": self.period_end,
        }

    @staticmethod
    def from_events(
        events: list[TacticalEvent],
        min_severity: str = "info",
        max_entries: int = 50,
    ) -> EventTimeline:
        """Build a timeline from events, filtering by minimum severity."""
        try:
            min_idx = SEVERITY_LEVELS.index(min_severity)
        except ValueError:
            min_idx = 0

        filtered: list[TacticalEvent] = []
        for ev in events:
            try:
                sev_idx = SEVERITY_LEVELS.index(ev.severity)
            except ValueError:
                sev_idx = 0
            if sev_idx >= min_idx:
                filtered.append(ev)

        # Sort chronologically (oldest first)
        filtered.sort(key=lambda e: e.timestamp)

        entries: list[dict[str, Any]] = []
        for ev in filtered[-max_entries:]:
            entries.append({
                "timestamp": ev.timestamp,
                "event_type": ev.event_type,
                "severity": ev.severity,
                "summary": ev.summary,
                "source": ev.source,
                "target_id": ev.target_id,
            })

        start = filtered[0].timestamp if filtered else 0.0
        end = filtered[-1].timestamp if filtered else 0.0

        return EventTimeline(
            events=entries,
            total_events=len(filtered),
            period_start=start,
            period_end=end,
        )


# ---------------------------------------------------------------------------
# SitRep — situation report
# ---------------------------------------------------------------------------

@dataclass
class SitRep:
    """A generated situation report."""

    generated_at: float = field(default_factory=time.time)
    title: str = "Situation Report"
    targets: TargetBreakdown = field(default_factory=TargetBreakdown)
    threats: ThreatSummary = field(default_factory=ThreatSummary)
    zones: ZoneActivity = field(default_factory=ZoneActivity)
    anomalies: AnomalySummary = field(default_factory=AnomalySummary)
    timeline: EventTimeline = field(default_factory=EventTimeline)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "generated_at": self.generated_at,
            "title": self.title,
            "targets": self.targets.to_dict(),
            "threats": self.threats.to_dict(),
            "zones": self.zones.to_dict(),
            "anomalies": self.anomalies.to_dict(),
            "timeline": self.timeline.to_dict(),
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Generate a plain-text situation report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  {self.title}")
        lines.append(f"  Generated: {_fmt_ts(self.generated_at)}")
        lines.append("=" * 60)
        lines.append("")

        # Target summary
        lines.append("--- TARGET SUMMARY ---")
        lines.append(f"Total targets tracked: {self.targets.total}")
        lines.append(f"Fused (multi-source):  {self.targets.fused_count}")
        if self.targets.by_source:
            lines.append("By source:")
            for src, cnt in sorted(self.targets.by_source.items()):
                lines.append(f"  {src:20s} {cnt:5d}  ({_pct(cnt, self.targets.total)})")
        if self.targets.by_alliance:
            lines.append("By alliance:")
            for alliance, cnt in sorted(self.targets.by_alliance.items()):
                lines.append(f"  {alliance:20s} {cnt:5d}")
        if self.targets.by_asset_type:
            lines.append("By asset type:")
            for atype, cnt in sorted(self.targets.by_asset_type.items()):
                lines.append(f"  {atype:20s} {cnt:5d}")
        lines.append("")

        # Threat assessment
        lines.append("--- THREAT ASSESSMENT ---")
        lines.append(f"Targets assessed: {self.threats.total_assessed}")
        lines.append(f"Hostile:          {self.threats.hostile_count}")
        lines.append(f"High threat:      {self.threats.high_threat}")
        lines.append(f"Medium threat:    {self.threats.medium_threat}")
        lines.append(f"Low threat:       {self.threats.low_threat}")
        lines.append(f"No threat:        {self.threats.no_threat}")
        if self.threats.suspicious_targets:
            lines.append(f"Suspicious IDs:   {', '.join(self.threats.suspicious_targets[:5])}")
        lines.append("")

        # Zone activity
        lines.append("--- ZONE ACTIVITY ---")
        lines.append(f"Zone events: {self.zones.total_events} (entries={self.zones.entries}, exits={self.zones.exits})")
        if self.zones.zones_active:
            lines.append(f"Active zones: {', '.join(self.zones.zones_active)}")
        if self.zones.most_active_zone:
            lines.append(f"Most active: {self.zones.most_active_zone}")
        lines.append("")

        # Anomalies
        lines.append("--- ANOMALIES ---")
        lines.append(f"Total anomalies: {self.anomalies.total_anomalies}")
        if self.anomalies.by_severity:
            for sev, cnt in sorted(self.anomalies.by_severity.items()):
                lines.append(f"  {sev:12s} {cnt:5d}")
        if self.anomalies.top_anomalies:
            lines.append("Top anomalies:")
            for a in self.anomalies.top_anomalies[:5]:
                lines.append(f"  [{a['severity']}] {a['event_type']}: {a['summary']}")
        lines.append("")

        # Timeline
        if self.timeline.events:
            lines.append("--- EVENT TIMELINE ---")
            lines.append(f"Period: {_fmt_ts(self.timeline.period_start)} to {_fmt_ts(self.timeline.period_end)}")
            lines.append(f"Significant events: {self.timeline.total_events}")
            for ev in self.timeline.events[-10:]:
                ts_str = _fmt_ts(ev["timestamp"])
                lines.append(f"  {ts_str}  [{ev['severity']}] {ev['event_type']}: {ev['summary']}")
            lines.append("")

        if self.notes:
            lines.append("--- NOTES ---")
            lines.append(self.notes)
            lines.append("")

        lines.append("=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate an HTML situation report with cyberpunk styling."""
        esc = html.escape

        parts: list[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html><head><meta charset='utf-8'>")
        parts.append(f"<title>{esc(self.title)}</title>")
        parts.append("<style>")
        parts.append("body { background: #0a0a0f; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }")
        parts.append("h1 { color: #00f0ff; border-bottom: 2px solid #00f0ff; padding-bottom: 8px; }")
        parts.append("h2 { color: #ff2a6d; margin-top: 24px; }")
        parts.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
        parts.append("th { background: #1a1a2e; color: #00f0ff; text-align: left; padding: 8px; border: 1px solid #333; }")
        parts.append("td { padding: 8px; border: 1px solid #333; }")
        parts.append("tr:nth-child(even) { background: #12121e; }")
        parts.append(".stat { color: #05ffa1; font-weight: bold; }")
        parts.append(".severity-critical { color: #ff2a6d; font-weight: bold; }")
        parts.append(".severity-error { color: #ff6b35; }")
        parts.append(".severity-warning { color: #fcee0a; }")
        parts.append(".severity-info { color: #05ffa1; }")
        parts.append(".timestamp { color: #888; font-size: 0.85em; }")
        parts.append("</style></head><body>")

        parts.append(f"<h1>{esc(self.title)}</h1>")
        parts.append(f"<p class='timestamp'>Generated: {esc(_fmt_ts(self.generated_at))}</p>")

        # Target summary
        parts.append("<h2>Target Summary</h2>")
        parts.append(f"<p>Total tracked: <span class='stat'>{self.targets.total}</span> | ")
        parts.append(f"Fused: <span class='stat'>{self.targets.fused_count}</span></p>")

        if self.targets.by_source:
            parts.append("<table><tr><th>Source</th><th>Count</th><th>%</th></tr>")
            for src, cnt in sorted(self.targets.by_source.items()):
                parts.append(f"<tr><td>{esc(src)}</td><td>{cnt}</td><td>{_pct(cnt, self.targets.total)}</td></tr>")
            parts.append("</table>")

        if self.targets.by_alliance:
            parts.append("<table><tr><th>Alliance</th><th>Count</th></tr>")
            for alliance, cnt in sorted(self.targets.by_alliance.items()):
                parts.append(f"<tr><td>{esc(alliance)}</td><td>{cnt}</td></tr>")
            parts.append("</table>")

        # Threats
        parts.append("<h2>Threat Assessment</h2>")
        parts.append("<table><tr><th>Level</th><th>Count</th></tr>")
        parts.append(f"<tr><td class='severity-critical'>High Threat</td><td>{self.threats.high_threat}</td></tr>")
        parts.append(f"<tr><td class='severity-warning'>Medium Threat</td><td>{self.threats.medium_threat}</td></tr>")
        parts.append(f"<tr><td class='severity-info'>Low Threat</td><td>{self.threats.low_threat}</td></tr>")
        parts.append(f"<tr><td>No Threat</td><td>{self.threats.no_threat}</td></tr>")
        parts.append(f"<tr><td class='severity-critical'>Hostile</td><td>{self.threats.hostile_count}</td></tr>")
        parts.append("</table>")

        # Zone activity
        parts.append("<h2>Zone Activity</h2>")
        parts.append(f"<p>Events: <span class='stat'>{self.zones.total_events}</span> | ")
        parts.append(f"Entries: {self.zones.entries} | Exits: {self.zones.exits}</p>")
        if self.zones.zones_active:
            parts.append(f"<p>Active zones: {esc(', '.join(self.zones.zones_active))}</p>")

        # Anomalies
        parts.append("<h2>Anomalies</h2>")
        parts.append(f"<p>Total: <span class='stat'>{self.anomalies.total_anomalies}</span></p>")
        if self.anomalies.top_anomalies:
            parts.append("<table><tr><th>Severity</th><th>Type</th><th>Summary</th></tr>")
            for a in self.anomalies.top_anomalies[:10]:
                sev_class = f"severity-{a['severity']}"
                parts.append(f"<tr><td class='{sev_class}'>{esc(a['severity'])}</td>")
                parts.append(f"<td>{esc(a['event_type'])}</td><td>{esc(a['summary'])}</td></tr>")
            parts.append("</table>")

        # Timeline
        if self.timeline.events:
            parts.append("<h2>Event Timeline</h2>")
            parts.append(f"<p class='timestamp'>Period: {esc(_fmt_ts(self.timeline.period_start))} to {esc(_fmt_ts(self.timeline.period_end))}</p>")
            parts.append("<table><tr><th>Time</th><th>Severity</th><th>Type</th><th>Summary</th></tr>")
            for ev in self.timeline.events[-15:]:
                sev_class = f"severity-{ev['severity']}"
                parts.append(f"<tr><td class='timestamp'>{esc(_fmt_ts(ev['timestamp']))}</td>")
                parts.append(f"<td class='{sev_class}'>{esc(ev['severity'])}</td>")
                parts.append(f"<td>{esc(ev['event_type'])}</td><td>{esc(ev['summary'])}</td></tr>")
            parts.append("</table>")

        if self.notes:
            parts.append("<h2>Notes</h2>")
            parts.append(f"<p>{esc(self.notes)}</p>")

        parts.append("</body></html>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# SitRepGenerator — builds SitRep from live tracker + event store
# ---------------------------------------------------------------------------

class SitRepGenerator:
    """Generates situation reports from the current tracking state and event history.

    Args:
        tracker: The TargetTracker holding current tracked targets.
        event_store: Optional EventStore for historical event data.
        title: Report title.
    """

    def __init__(
        self,
        tracker: TargetTracker,
        event_store: Optional[EventStore] = None,
        title: str = "Situation Report",
    ) -> None:
        self._tracker = tracker
        self._event_store = event_store
        self._title = title

    def generate(
        self,
        event_time_range: Optional[tuple[float, float]] = None,
        notes: str = "",
    ) -> SitRep:
        """Generate a full situation report.

        Args:
            event_time_range: Optional (start, end) unix timestamps to scope events.
                If None, queries the last hour of events.
            notes: Optional freeform notes to include.

        Returns:
            A SitRep object with all sections populated.
        """
        targets = self._tracker.get_all()
        target_breakdown = TargetBreakdown.from_targets(targets)
        threat_summary = ThreatSummary.from_targets(targets)

        # Fetch events
        events: list[TacticalEvent] = []
        if self._event_store is not None:
            if event_time_range is not None:
                start, end = event_time_range
            else:
                end = time.time()
                start = end - 3600.0
            events = self._event_store.query_time_range(start=start, end=end, limit=1000)

        zone_activity = ZoneActivity.from_events(events)
        anomaly_summary = AnomalySummary.from_events(events)
        timeline = EventTimeline.from_events(events)

        return SitRep(
            generated_at=time.time(),
            title=self._title,
            targets=target_breakdown,
            threats=threat_summary,
            zones=zone_activity,
            anomalies=anomaly_summary,
            timeline=timeline,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# DailySummary — aggregated daily activity report
# ---------------------------------------------------------------------------

@dataclass
class DailySummary:
    """Aggregated daily activity summary.

    Provides a high-level overview of a day's tracking activity including
    targets seen, source diversity, threat distribution, and event counts.
    """

    date: str = ""
    generated_at: float = field(default_factory=time.time)
    total_targets_seen: int = 0
    new_targets: int = 0
    targets_by_source: dict[str, int] = field(default_factory=dict)
    targets_by_alliance: dict[str, int] = field(default_factory=dict)
    fused_targets: int = 0
    threats_detected: int = 0
    threat_breakdown: dict[str, int] = field(default_factory=dict)
    zone_entries: int = 0
    zone_exits: int = 0
    total_events: int = 0
    events_by_severity: dict[str, int] = field(default_factory=dict)
    anomaly_count: int = 0
    top_event_types: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "date": self.date,
            "generated_at": self.generated_at,
            "total_targets_seen": self.total_targets_seen,
            "new_targets": self.new_targets,
            "targets_by_source": dict(self.targets_by_source),
            "targets_by_alliance": dict(self.targets_by_alliance),
            "fused_targets": self.fused_targets,
            "threats_detected": self.threats_detected,
            "threat_breakdown": dict(self.threat_breakdown),
            "zone_entries": self.zone_entries,
            "zone_exits": self.zone_exits,
            "total_events": self.total_events,
            "events_by_severity": dict(self.events_by_severity),
            "anomaly_count": self.anomaly_count,
            "top_event_types": [{"type": t, "count": c} for t, c in self.top_event_types],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Generate plain-text daily summary."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  DAILY SUMMARY: {self.date}")
        lines.append(f"  Generated: {_fmt_ts(self.generated_at)}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Targets seen:    {self.total_targets_seen}")
        lines.append(f"New targets:     {self.new_targets}")
        lines.append(f"Fused targets:   {self.fused_targets}")
        lines.append(f"Threats found:   {self.threats_detected}")
        lines.append(f"Zone entries:    {self.zone_entries}")
        lines.append(f"Zone exits:      {self.zone_exits}")
        lines.append(f"Total events:    {self.total_events}")
        lines.append(f"Anomalies:       {self.anomaly_count}")
        lines.append("")

        if self.targets_by_source:
            lines.append("Targets by source:")
            for src, cnt in sorted(self.targets_by_source.items()):
                lines.append(f"  {src:20s} {cnt:5d}")
            lines.append("")

        if self.threat_breakdown:
            lines.append("Threat breakdown:")
            for level, cnt in sorted(self.threat_breakdown.items()):
                lines.append(f"  {level:20s} {cnt:5d}")
            lines.append("")

        if self.events_by_severity:
            lines.append("Events by severity:")
            for sev, cnt in sorted(self.events_by_severity.items()):
                lines.append(f"  {sev:12s} {cnt:5d}")
            lines.append("")

        if self.top_event_types:
            lines.append("Top event types:")
            for etype, cnt in self.top_event_types[:10]:
                lines.append(f"  {etype:30s} {cnt:5d}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate an HTML daily summary with cyberpunk styling."""
        esc = html.escape
        parts: list[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        parts.append(f"<title>Daily Summary: {esc(self.date)}</title>")
        parts.append("<style>")
        parts.append("body { background: #0a0a0f; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }")
        parts.append("h1 { color: #00f0ff; } h2 { color: #ff2a6d; }")
        parts.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
        parts.append("th { background: #1a1a2e; color: #00f0ff; text-align: left; padding: 8px; border: 1px solid #333; }")
        parts.append("td { padding: 8px; border: 1px solid #333; }")
        parts.append(".stat { color: #05ffa1; font-weight: bold; }")
        parts.append("</style></head><body>")
        parts.append(f"<h1>Daily Summary: {esc(self.date)}</h1>")
        parts.append(f"<p>Targets: <span class='stat'>{self.total_targets_seen}</span> | ")
        parts.append(f"New: <span class='stat'>{self.new_targets}</span> | ")
        parts.append(f"Threats: <span class='stat'>{self.threats_detected}</span> | ")
        parts.append(f"Events: <span class='stat'>{self.total_events}</span></p>")

        if self.targets_by_source:
            parts.append("<h2>Sources</h2><table><tr><th>Source</th><th>Count</th></tr>")
            for src, cnt in sorted(self.targets_by_source.items()):
                parts.append(f"<tr><td>{esc(src)}</td><td>{cnt}</td></tr>")
            parts.append("</table>")

        parts.append("</body></html>")
        return "\n".join(parts)

    @staticmethod
    def from_tracker_and_events(
        tracker: TargetTracker,
        event_store: Optional[EventStore] = None,
        date: Optional[str] = None,
        event_start: Optional[float] = None,
        event_end: Optional[float] = None,
    ) -> DailySummary:
        """Build a daily summary from live tracker and event store.

        Args:
            tracker: TargetTracker with current targets.
            event_store: Optional EventStore for historical data.
            date: Date string (defaults to today's date).
            event_start: Start of period (unix timestamp).
            event_end: End of period (unix timestamp).
        """
        if date is None:
            date = time.strftime("%Y-%m-%d")

        targets = tracker.get_all()
        breakdown = TargetBreakdown.from_targets(targets)
        threat_summary = ThreatSummary.from_targets(targets)

        # Count new targets (first_seen within the last 24h using monotonic offset)
        now_mono = time.monotonic()
        new_count = sum(1 for t in targets if (now_mono - t.first_seen) < 86400.0)

        # Fetch events
        events: list[TacticalEvent] = []
        events_by_severity: dict[str, int] = {}
        event_type_counts: dict[str, int] = {}
        zone_entries = 0
        zone_exits = 0
        anomaly_count = 0

        if event_store is not None:
            if event_start is None:
                event_end_ts = event_end or time.time()
                event_start = event_end_ts - 86400.0
            events = event_store.query_time_range(
                start=event_start, end=event_end, limit=5000,
            )

            for ev in events:
                sev = ev.severity or "info"
                events_by_severity[sev] = events_by_severity.get(sev, 0) + 1

                etype = ev.event_type
                event_type_counts[etype] = event_type_counts.get(etype, 0) + 1

                etype_lower = etype.lower()
                if "geofence" in etype_lower or "zone" in etype_lower:
                    if "enter" in etype_lower:
                        zone_entries += 1
                    elif "exit" in etype_lower:
                        zone_exits += 1

                if "anomaly" in etype_lower or ev.severity in ("warning", "error", "critical"):
                    anomaly_count += 1

        # Top event types
        top_types = sorted(event_type_counts.items(), key=lambda x: -x[1])[:10]

        return DailySummary(
            date=date,
            generated_at=time.time(),
            total_targets_seen=breakdown.total,
            new_targets=new_count,
            targets_by_source=breakdown.by_source,
            targets_by_alliance=breakdown.by_alliance,
            fused_targets=breakdown.fused_count,
            threats_detected=threat_summary.high_threat + threat_summary.medium_threat,
            threat_breakdown={
                "high": threat_summary.high_threat,
                "medium": threat_summary.medium_threat,
                "low": threat_summary.low_threat,
                "none": threat_summary.no_threat,
            },
            zone_entries=zone_entries,
            zone_exits=zone_exits,
            total_events=len(events),
            events_by_severity=events_by_severity,
            anomaly_count=anomaly_count,
            top_event_types=top_types,
        )


# ---------------------------------------------------------------------------
# IncidentReport — detailed report for a specific incident
# ---------------------------------------------------------------------------

@dataclass
class IncidentReport:
    """Detailed report for a specific incident or alert.

    Can be constructed manually or via the from_event class method
    to populate from an EventStore event.
    """

    title: str = ""
    incident_id: str = ""
    generated_at: float = field(default_factory=time.time)
    severity: str = "info"
    description: str = ""
    target_ids: list[str] = field(default_factory=list)
    target_details: list[dict[str, Any]] = field(default_factory=list)
    related_events: list[dict[str, Any]] = field(default_factory=list)
    location: Optional[tuple[float, float]] = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "title": self.title,
            "incident_id": self.incident_id,
            "generated_at": self.generated_at,
            "severity": self.severity,
            "description": self.description,
            "target_ids": list(self.target_ids),
            "target_details": list(self.target_details),
            "related_events": list(self.related_events),
            "location": list(self.location) if self.location else None,
            "recommendations": list(self.recommendations),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Generate a plain-text incident report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  INCIDENT REPORT")
        if self.incident_id:
            lines.append(f"  ID: {self.incident_id}")
        lines.append(f"  {self.title}")
        lines.append(f"  Severity: {self.severity.upper()}")
        lines.append(f"  Generated: {_fmt_ts(self.generated_at)}")
        lines.append("=" * 60)
        lines.append("")

        if self.description:
            lines.append("DESCRIPTION:")
            lines.append(self.description)
            lines.append("")

        if self.location:
            lines.append(f"LOCATION: ({self.location[0]:.4f}, {self.location[1]:.4f})")
            lines.append("")

        if self.target_ids:
            lines.append(f"INVOLVED TARGETS ({len(self.target_ids)}):")
            for tid in self.target_ids:
                lines.append(f"  - {tid}")
            lines.append("")

        if self.target_details:
            lines.append("TARGET DETAILS:")
            for td in self.target_details:
                tid = td.get("target_id", "unknown")
                alliance = td.get("alliance", "unknown")
                source = td.get("source", "unknown")
                atype = td.get("asset_type", "unknown")
                lines.append(f"  {tid}: alliance={alliance}, source={source}, type={atype}")
            lines.append("")

        if self.related_events:
            lines.append(f"RELATED EVENTS ({len(self.related_events)}):")
            for ev in self.related_events[:20]:
                ts_str = _fmt_ts(ev.get("timestamp", 0))
                lines.append(f"  {ts_str}  [{ev.get('severity', 'info')}] {ev.get('event_type', '')}: {ev.get('summary', '')}")
            lines.append("")

        if self.recommendations:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("END OF INCIDENT REPORT")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate an HTML incident report with cyberpunk styling."""
        esc = html.escape
        parts: list[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        parts.append(f"<title>Incident: {esc(self.title)}</title>")
        parts.append("<style>")
        parts.append("body { background: #0a0a0f; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }")
        parts.append("h1 { color: #ff2a6d; } h2 { color: #00f0ff; }")
        parts.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
        parts.append("th { background: #1a1a2e; color: #00f0ff; text-align: left; padding: 8px; border: 1px solid #333; }")
        parts.append("td { padding: 8px; border: 1px solid #333; }")
        parts.append(".severity-critical { color: #ff2a6d; font-weight: bold; }")
        parts.append(".severity-error { color: #ff6b35; }")
        parts.append(".severity-warning { color: #fcee0a; }")
        parts.append(".severity-info { color: #05ffa1; }")
        parts.append(".stat { color: #05ffa1; font-weight: bold; }")
        parts.append(".timestamp { color: #888; font-size: 0.85em; }")
        parts.append("</style></head><body>")

        sev_class = f"severity-{self.severity}"
        parts.append(f"<h1>Incident Report</h1>")
        if self.incident_id:
            parts.append(f"<p class='timestamp'>ID: {esc(self.incident_id)}</p>")
        parts.append(f"<p><strong>{esc(self.title)}</strong></p>")
        parts.append(f"<p>Severity: <span class='{sev_class}'>{esc(self.severity.upper())}</span></p>")

        if self.description:
            parts.append(f"<h2>Description</h2><p>{esc(self.description)}</p>")

        if self.target_details:
            parts.append("<h2>Involved Targets</h2>")
            parts.append("<table><tr><th>Target ID</th><th>Alliance</th><th>Source</th><th>Type</th></tr>")
            for td in self.target_details:
                parts.append(f"<tr><td>{esc(td.get('target_id', ''))}</td>")
                parts.append(f"<td>{esc(td.get('alliance', ''))}</td>")
                parts.append(f"<td>{esc(td.get('source', ''))}</td>")
                parts.append(f"<td>{esc(td.get('asset_type', ''))}</td></tr>")
            parts.append("</table>")

        if self.related_events:
            parts.append("<h2>Related Events</h2>")
            parts.append("<table><tr><th>Time</th><th>Severity</th><th>Type</th><th>Summary</th></tr>")
            for ev in self.related_events[:20]:
                ev_sev = ev.get("severity", "info")
                parts.append(f"<tr><td class='timestamp'>{esc(_fmt_ts(ev.get('timestamp', 0)))}</td>")
                parts.append(f"<td class='severity-{ev_sev}'>{esc(ev_sev)}</td>")
                parts.append(f"<td>{esc(ev.get('event_type', ''))}</td>")
                parts.append(f"<td>{esc(ev.get('summary', ''))}</td></tr>")
            parts.append("</table>")

        if self.recommendations:
            parts.append("<h2>Recommendations</h2><ol>")
            for rec in self.recommendations:
                parts.append(f"<li>{esc(rec)}</li>")
            parts.append("</ol>")

        parts.append("</body></html>")
        return "\n".join(parts)

    @staticmethod
    def from_event(
        event: TacticalEvent,
        tracker: Optional[TargetTracker] = None,
        event_store: Optional[EventStore] = None,
        related_window: float = 300.0,
    ) -> IncidentReport:
        """Build an IncidentReport from a triggering event.

        Args:
            event: The primary event that triggered this incident.
            tracker: Optional tracker to pull target details.
            event_store: Optional store to pull related events.
            related_window: Seconds before/after the event to search
                for related events (default 5 minutes).
        """
        target_ids: list[str] = []
        target_details: list[dict[str, Any]] = []
        if event.target_id:
            target_ids.append(event.target_id)

        # Enrich with tracker data
        if tracker is not None:
            for tid in target_ids:
                t = tracker.get_target(tid)
                if t is not None:
                    target_details.append({
                        "target_id": t.target_id,
                        "name": t.name,
                        "alliance": t.alliance,
                        "asset_type": t.asset_type,
                        "source": t.source,
                        "position": list(t.position),
                        "threat_score": t.threat_score,
                    })

        # Pull related events
        related_events: list[dict[str, Any]] = []
        if event_store is not None:
            start = event.timestamp - related_window
            end = event.timestamp + related_window
            related = event_store.query_time_range(start=start, end=end, limit=50)
            for rev in related:
                if rev.event_id == event.event_id:
                    continue
                related_events.append({
                    "event_id": rev.event_id,
                    "timestamp": rev.timestamp,
                    "event_type": rev.event_type,
                    "severity": rev.severity,
                    "summary": rev.summary,
                    "target_id": rev.target_id,
                })

        location = None
        if event.position_lat is not None and event.position_lng is not None:
            location = (event.position_lat, event.position_lng)

        return IncidentReport(
            title=event.summary or f"{event.event_type} incident",
            incident_id=event.event_id,
            generated_at=time.time(),
            severity=event.severity,
            description=event.summary,
            target_ids=target_ids,
            target_details=target_details,
            related_events=related_events,
            location=location,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SitRepGenerator",
    "SitRep",
    "DailySummary",
    "IncidentReport",
    "TargetBreakdown",
    "ThreatSummary",
    "ZoneActivity",
    "AnomalySummary",
    "EventTimeline",
]
