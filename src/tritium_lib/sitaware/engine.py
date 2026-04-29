# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SitAwareEngine — the unified situational awareness orchestrator.

Composes FusionEngine, AlertEngine, AnomalyEngine, AnalyticsEngine,
HealthMonitor, IncidentManager, and MissionPlanner into a single
coherent operating picture.

This is the capstone module: pure orchestration over existing subsystems.
It creates nothing new — it just ties everything together into one view.

Thread-safe. All public methods acquire appropriate locks.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from tritium_lib.alerting import AlertEngine, AlertRecord
from tritium_lib.analytics import AnalyticsEngine
from tritium_lib.events.bus import EventBus, QueueEventBus
from tritium_lib.fusion import FusionEngine, FusedTarget, FusionSnapshot
from tritium_lib.intelligence.anomaly_engine import AnomalyEngine, AnomalyAlert
from tritium_lib.incident import IncidentManager
from tritium_lib.mission import MissionPlanner
from tritium_lib.monitoring import HealthMonitor, ComponentHealth, ComponentStatus

logger = logging.getLogger("tritium.sitaware")


# ---------------------------------------------------------------------------
# UpdateType — categories of picture updates
# ---------------------------------------------------------------------------

class UpdateType(str, Enum):
    """Categories of changes to the operating picture."""

    TARGET_NEW = "target_new"
    TARGET_UPDATED = "target_updated"
    TARGET_LOST = "target_lost"
    TARGET_CORRELATED = "target_correlated"
    ALERT_FIRED = "alert_fired"
    ANOMALY_DETECTED = "anomaly_detected"
    INCIDENT_CREATED = "incident_created"
    INCIDENT_UPDATED = "incident_updated"
    INCIDENT_RESOLVED = "incident_resolved"
    MISSION_CREATED = "mission_created"
    MISSION_UPDATED = "mission_updated"
    MISSION_COMPLETED = "mission_completed"
    HEALTH_CHANGED = "health_changed"
    ZONE_BREACH = "zone_breach"
    FULL_REFRESH = "full_refresh"


# ---------------------------------------------------------------------------
# PictureUpdate — a single delta update to the operating picture
# ---------------------------------------------------------------------------

@dataclass
class PictureUpdate:
    """A delta update to the operating picture.

    Each update represents a single change: a new target appeared, an alert
    fired, an incident was created, etc.  Subscribers receive these in
    real-time via callbacks registered with ``SitAwareEngine.subscribe()``.

    Attributes
    ----------
    update_id:
        Unique ID for this update.
    update_type:
        Category of the change (see UpdateType).
    timestamp:
        When the update occurred (epoch seconds).
    data:
        Payload specific to the update type. Always JSON-serializable.
    source:
        Which subsystem generated this update (e.g., "fusion", "alerting").
    target_id:
        Target ID if the update relates to a specific target.
    zone_id:
        Zone ID if the update relates to a specific zone.
    severity:
        Severity hint for UI prioritization ("info", "warning", "critical").
    """

    update_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    update_type: UpdateType = UpdateType.FULL_REFRESH
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    target_id: str = ""
    zone_id: str = ""
    severity: str = "info"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "update_id": self.update_id,
            "update_type": self.update_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "source": self.source,
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# OperatingPicture — the full state of everything
# ---------------------------------------------------------------------------

@dataclass
class OperatingPicture:
    """Complete situational awareness snapshot — the unified operating picture.

    Contains the current state of every target, zone, threat, alert,
    incident, mission, and system component at a point in time.

    This is what gets rendered on the dashboard, fed to Amy AI, and
    served via the REST API.

    Attributes
    ----------
    timestamp:
        When this picture was generated (epoch seconds).
    targets:
        All fused targets with full sensor context.
    target_count:
        Total number of tracked targets.
    multi_source_targets:
        Targets confirmed by 2+ sensor sources.
    alerts:
        Recent alert records (newest first).
    active_alert_count:
        Number of alerts in the current time window.
    anomalies:
        Recent anomaly alerts (newest first).
    active_anomaly_count:
        Number of active anomaly detections.
    incidents:
        Active (non-resolved, non-closed) incidents.
    incident_count:
        Total number of active incidents.
    missions:
        Active missions.
    mission_count:
        Total number of active missions.
    health:
        System health status.
    analytics:
        Real-time analytics snapshot.
    zones:
        All geofence zones.
    zone_count:
        Total number of zones.
    summary:
        Human-readable one-line summary of the current situation.
    threat_level:
        Overall threat level: "green", "yellow", "orange", "red".
    """

    timestamp: float = field(default_factory=time.time)

    # Targets
    targets: list[dict[str, Any]] = field(default_factory=list)
    target_count: int = 0
    multi_source_targets: int = 0

    # Alerts
    alerts: list[dict[str, Any]] = field(default_factory=list)
    active_alert_count: int = 0

    # Anomalies
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    active_anomaly_count: int = 0

    # Incidents
    incidents: list[dict[str, Any]] = field(default_factory=list)
    incident_count: int = 0

    # Missions
    missions: list[dict[str, Any]] = field(default_factory=list)
    mission_count: int = 0

    # System health
    health: dict[str, Any] = field(default_factory=dict)

    # Analytics
    analytics: dict[str, Any] = field(default_factory=dict)

    # Zones
    zones: list[dict[str, Any]] = field(default_factory=list)
    zone_count: int = 0

    # Situational summary
    summary: str = ""
    threat_level: str = "green"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation of the full picture."""
        return {
            "timestamp": self.timestamp,
            "targets": self.targets,
            "target_count": self.target_count,
            "multi_source_targets": self.multi_source_targets,
            "alerts": self.alerts,
            "active_alert_count": self.active_alert_count,
            "anomalies": self.anomalies,
            "active_anomaly_count": self.active_anomaly_count,
            "incidents": self.incidents,
            "incident_count": self.incident_count,
            "missions": self.missions,
            "mission_count": self.mission_count,
            "health": self.health,
            "analytics": self.analytics,
            "zones": self.zones,
            "zone_count": self.zone_count,
            "summary": self.summary,
            "threat_level": self.threat_level,
        }


# ---------------------------------------------------------------------------
# Subscriber type
# ---------------------------------------------------------------------------

PictureSubscriber = Callable[[PictureUpdate], None]


# ---------------------------------------------------------------------------
# SitAwareEngine — the master orchestrator
# ---------------------------------------------------------------------------

class SitAwareEngine:
    """The unified situational awareness engine.

    Composes all Tritium subsystems into a single coherent operating picture.
    This is the top-level entry point for querying the state of the world.

    Parameters
    ----------
    event_bus:
        Shared EventBus for inter-component communication. If None, a
        new one is created.
    fusion:
        Existing FusionEngine instance. If None, creates one.
    alerting:
        Existing AlertEngine instance. If None, creates one.
    anomaly:
        Existing AnomalyEngine instance. If None, creates one.
    analytics:
        Existing AnalyticsEngine instance. If None, creates one.
    health:
        Existing HealthMonitor instance. If None, creates one.
    incidents:
        Existing IncidentManager instance. If None, creates one.
    missions:
        Existing MissionPlanner instance. If None, creates one.
    alert_window:
        Time window in seconds for "active" alerts (default 300 = 5 min).
    anomaly_window:
        Time window in seconds for "active" anomalies (default 300 = 5 min).
    max_updates:
        Maximum number of PictureUpdates to retain for delta queries.
    auto_wire:
        If True (default), automatically wire subsystem events to the
        operating picture update stream.
    """

    def __init__(
        self,
        event_bus: EventBus | QueueEventBus | None = None,
        *,
        fusion: FusionEngine | None = None,
        alerting: AlertEngine | None = None,
        anomaly: AnomalyEngine | None = None,
        analytics: AnalyticsEngine | None = None,
        health: HealthMonitor | None = None,
        incidents: IncidentManager | None = None,
        missions: MissionPlanner | None = None,
        alert_window: float = 300.0,
        anomaly_window: float = 300.0,
        max_updates: int = 10000,
        auto_wire: bool = True,
    ) -> None:
        self._lock = threading.Lock()

        # Bridge state — only used when caller passes a QueueEventBus.
        # SitAwareEngine is built on the topic-based EventBus API
        # (subscribe(topic, callback)).  When integrating with subsystems
        # that use the queue-based QueueEventBus (e.g. tritium-sc), we
        # transparently bridge: SitAware uses its own internal EventBus
        # for wiring, and a background thread forwards events from the
        # queue bus into the topic bus.
        self._external_queue_bus: QueueEventBus | None = None
        self._bridge_queue: Any = None
        self._bridge_thread: threading.Thread | None = None
        self._bridge_running = False

        # EventBus — the nervous system
        if event_bus is None:
            self._event_bus = EventBus()
        elif isinstance(event_bus, QueueEventBus):
            # Caller passed a queue-based bus.  SitAware needs the
            # topic-based EventBus API, so create one and bridge events
            # from the queue bus into it.
            self._external_queue_bus = event_bus
            self._event_bus = EventBus()
            logger.info(
                "SitAwareEngine: bridging from QueueEventBus to internal "
                "topic-based EventBus (subscribers receive topic events)"
            )
        elif isinstance(event_bus, EventBus):
            self._event_bus = event_bus
        else:
            # Unknown bus type — refuse to silently misbehave.
            raise TypeError(
                f"SitAwareEngine: unsupported event_bus type "
                f"{type(event_bus).__name__!r}; expected EventBus, "
                f"QueueEventBus, or None"
            )

        # Subsystem engines — accept externally-managed instances or create new
        self._fusion = fusion if fusion is not None else FusionEngine(
            event_bus=self._event_bus,
        )
        self._alerting = alerting if alerting is not None else AlertEngine(
            event_bus=self._event_bus,
        )
        self._anomaly = anomaly if anomaly is not None else AnomalyEngine(
            event_bus=self._event_bus,
        )
        self._analytics = analytics if analytics is not None else AnalyticsEngine()
        self._health = health if health is not None else HealthMonitor()
        self._incidents = incidents if incidents is not None else IncidentManager(
            event_bus=self._event_bus,
        )
        self._missions = missions if missions is not None else MissionPlanner(
            event_bus=self._event_bus,
        )

        # Configuration
        self._alert_window = alert_window
        self._anomaly_window = anomaly_window
        self._max_updates = max_updates

        # Update stream
        self._updates: list[PictureUpdate] = []
        self._subscribers: list[PictureSubscriber] = []
        self._subscriber_lock = threading.Lock()

        # Track target IDs for new/lost detection
        self._known_target_ids: set[str] = set()

        # Last picture timestamp for change detection
        self._last_picture_time: float = 0.0

        # Auto-wire event subscriptions
        if auto_wire:
            self._wire_events()

        # Register built-in health checks
        self._register_health_checks()

        logger.info("SitAwareEngine initialized with %d subsystems", 7)

    # ------------------------------------------------------------------
    # Component accessors — expose subsystems for direct use
    # ------------------------------------------------------------------

    @property
    def event_bus(self) -> EventBus:
        """The shared EventBus."""
        return self._event_bus

    @property
    def fusion(self) -> FusionEngine:
        """The FusionEngine — multi-sensor target fusion."""
        return self._fusion

    @property
    def alerting(self) -> AlertEngine:
        """The AlertEngine — rule-based alert evaluation."""
        return self._alerting

    @property
    def anomaly(self) -> AnomalyEngine:
        """The AnomalyEngine — behavioral anomaly detection."""
        return self._anomaly

    @property
    def analytics(self) -> AnalyticsEngine:
        """The AnalyticsEngine — real-time statistics."""
        return self._analytics

    @property
    def health(self) -> HealthMonitor:
        """The HealthMonitor — system health checks."""
        return self._health

    @property
    def incidents(self) -> IncidentManager:
        """The IncidentManager — incident lifecycle."""
        return self._incidents

    @property
    def missions(self) -> MissionPlanner:
        """The MissionPlanner — mission coordination."""
        return self._missions

    # ------------------------------------------------------------------
    # Event wiring — connect subsystem events to update stream
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        """Subscribe to subsystem events and translate them to PictureUpdates."""
        bus = self._event_bus

        # Fusion events
        bus.subscribe("fusion.sensor.ingested", self._on_sensor_ingested)
        bus.subscribe("fusion.target.correlated", self._on_target_correlated)

        # Geofence events
        bus.subscribe("geofence:enter", self._on_zone_breach)
        bus.subscribe("geofence:exit", self._on_zone_breach)

        # Alert escalation events
        bus.subscribe("alert.escalation", self._on_alert_escalation)

        # Anomaly events
        bus.subscribe("anomaly.alert", self._on_anomaly_alert)

        # Incident lifecycle events
        bus.subscribe("incident.created", self._on_incident_event)
        bus.subscribe("incident.state_changed", self._on_incident_event)
        bus.subscribe("incident.resolved", self._on_incident_event)

        # Mission lifecycle events
        bus.subscribe("mission.created", self._on_mission_event)
        bus.subscribe("mission.state_changed", self._on_mission_event)
        bus.subscribe("mission.completed", self._on_mission_event)

        logger.debug("Event wiring complete: subscribed to subsystem topics")

    def _on_sensor_ingested(self, event) -> None:
        """Handle a fusion sensor ingestion event."""
        data = event.data if hasattr(event, "data") else (event if isinstance(event, dict) else {})
        if isinstance(data, dict):
            target_id = data.get("target_id", "")
            source = data.get("source", "")
        else:
            target_id = ""
            source = ""

        # Determine if this is a new target
        is_new = False
        with self._lock:
            if target_id and target_id not in self._known_target_ids:
                self._known_target_ids.add(target_id)
                is_new = True

        update_type = UpdateType.TARGET_NEW if is_new else UpdateType.TARGET_UPDATED
        update = PictureUpdate(
            update_type=update_type,
            data={"target_id": target_id, "source": source},
            source="fusion",
            target_id=target_id,
        )
        self._emit_update(update)

        # Feed analytics
        self._analytics.record_detection(target_id, source=source)

    def _on_target_correlated(self, event) -> None:
        """Handle a target correlation event."""
        data = event.data if hasattr(event, "data") else (event if isinstance(event, dict) else {})
        if isinstance(data, dict):
            primary_id = data.get("primary_id", "")
            secondary_id = data.get("secondary_id", "")
            confidence = data.get("confidence", 0.0)
        else:
            primary_id = ""
            secondary_id = ""
            confidence = 0.0

        update = PictureUpdate(
            update_type=UpdateType.TARGET_CORRELATED,
            data={
                "primary_id": primary_id,
                "secondary_id": secondary_id,
                "confidence": confidence,
            },
            source="fusion",
            target_id=primary_id,
            severity="info",
        )
        self._emit_update(update)

        # Feed analytics
        self._analytics.record_correlation(primary_id, secondary_id, success=True)

    def _on_zone_breach(self, event) -> None:
        """Handle geofence entry/exit events."""
        data = event.data if hasattr(event, "data") else (event if isinstance(event, dict) else {})
        if isinstance(data, dict):
            target_id = data.get("target_id", "")
            zone_id = data.get("zone_id", "")
            zone_type = data.get("zone_type", "")
        else:
            target_id = ""
            zone_id = ""
            zone_type = ""

        severity = "warning" if zone_type == "restricted" else "info"

        update = PictureUpdate(
            update_type=UpdateType.ZONE_BREACH,
            data=dict(data) if isinstance(data, dict) else {},
            source="geofence",
            target_id=target_id,
            zone_id=zone_id,
            severity=severity,
        )
        self._emit_update(update)

    def _on_alert_escalation(self, event) -> None:
        """Handle alert escalation events."""
        data = event.data if hasattr(event, "data") else (event if isinstance(event, dict) else {})
        if isinstance(data, dict):
            target_id = data.get("target_id", "")
            sev = data.get("severity", "info")
        else:
            target_id = ""
            sev = "info"

        update = PictureUpdate(
            update_type=UpdateType.ALERT_FIRED,
            data=dict(data) if isinstance(data, dict) else {},
            source="alerting",
            target_id=target_id,
            severity=sev,
        )
        self._emit_update(update)

        # Feed analytics
        self._analytics.record_alert("escalation", severity=sev)

    def _on_anomaly_alert(self, event) -> None:
        """Handle anomaly detection events."""
        data = event.data if hasattr(event, "data") else (event if isinstance(event, dict) else {})
        if isinstance(data, dict):
            target_id = data.get("target_id", "")
            zone_id = data.get("zone_id", "")
            sev = data.get("severity", "low")
            alert_type = data.get("alert_type", "")
        else:
            target_id = ""
            zone_id = ""
            sev = "low"
            alert_type = ""

        update = PictureUpdate(
            update_type=UpdateType.ANOMALY_DETECTED,
            data=dict(data) if isinstance(data, dict) else {},
            source="anomaly",
            target_id=target_id,
            zone_id=zone_id,
            severity=sev,
        )
        self._emit_update(update)

        # Feed analytics
        self._analytics.record_alert(alert_type, severity=sev)

    def _on_incident_event(self, event) -> None:
        """Handle incident lifecycle events."""
        data = event.data if hasattr(event, "data") else (event if isinstance(event, dict) else {})
        topic = event.topic if hasattr(event, "topic") else ""

        if isinstance(data, dict):
            incident_id = data.get("incident_id", "")
            sev = data.get("severity", "info")
        else:
            incident_id = ""
            sev = "info"

        if "created" in topic:
            utype = UpdateType.INCIDENT_CREATED
        elif "resolved" in topic:
            utype = UpdateType.INCIDENT_RESOLVED
        else:
            utype = UpdateType.INCIDENT_UPDATED

        update = PictureUpdate(
            update_type=utype,
            data=dict(data) if isinstance(data, dict) else {},
            source="incidents",
            severity=sev,
        )
        self._emit_update(update)

    def _on_mission_event(self, event) -> None:
        """Handle mission lifecycle events."""
        data = event.data if hasattr(event, "data") else (event if isinstance(event, dict) else {})
        topic = event.topic if hasattr(event, "topic") else ""

        if isinstance(data, dict):
            pass
        else:
            data = {}

        if "created" in topic:
            utype = UpdateType.MISSION_CREATED
        elif "completed" in topic:
            utype = UpdateType.MISSION_COMPLETED
        else:
            utype = UpdateType.MISSION_UPDATED

        update = PictureUpdate(
            update_type=utype,
            data=dict(data) if isinstance(data, dict) else {},
            source="missions",
        )
        self._emit_update(update)

    # ------------------------------------------------------------------
    # Health check registration
    # ------------------------------------------------------------------

    def _register_health_checks(self) -> None:
        """Register built-in health checks for all subsystems."""
        self._health.register("fusion", self._check_fusion_health)
        self._health.register("alerting", self._check_alerting_health)
        self._health.register("anomaly", self._check_anomaly_health)
        self._health.register("analytics", self._check_analytics_health)
        self._health.register("incidents", self._check_incidents_health)
        self._health.register("missions", self._check_missions_health)

    def _check_fusion_health(self) -> ComponentHealth:
        """Health check for the FusionEngine."""
        try:
            targets = self._fusion.get_fused_targets()
            return ComponentHealth(
                name="fusion",
                status=ComponentStatus.UP,
                message=f"{len(targets)} fused targets",
                details={"target_count": len(targets)},
            )
        except Exception as exc:
            return ComponentHealth(
                name="fusion",
                status=ComponentStatus.DOWN,
                error=str(exc),
            )

    def _check_alerting_health(self) -> ComponentHealth:
        """Health check for the AlertEngine."""
        try:
            stats = self._alerting.get_stats()
            return ComponentHealth(
                name="alerting",
                status=ComponentStatus.UP,
                message=f"{stats['total_rules']} rules, "
                        f"{stats['total_alerts_fired']} alerts fired",
                details=stats,
            )
        except Exception as exc:
            return ComponentHealth(
                name="alerting",
                status=ComponentStatus.DOWN,
                error=str(exc),
            )

    def _check_anomaly_health(self) -> ComponentHealth:
        """Health check for the AnomalyEngine."""
        try:
            stats = self._anomaly.get_stats()
            return ComponentHealth(
                name="anomaly",
                status=ComponentStatus.UP,
                message=f"{stats['zone_count']} zones monitored, "
                        f"{stats['total_alerts']} alerts",
                details=stats,
            )
        except Exception as exc:
            return ComponentHealth(
                name="anomaly",
                status=ComponentStatus.DOWN,
                error=str(exc),
            )

    def _check_analytics_health(self) -> ComponentHealth:
        """Health check for the AnalyticsEngine."""
        try:
            snap = self._analytics.snapshot()
            return ComponentHealth(
                name="analytics",
                status=ComponentStatus.UP,
                message="Analytics engine operational",
                details={"timestamp": snap.get("timestamp", 0)},
            )
        except Exception as exc:
            return ComponentHealth(
                name="analytics",
                status=ComponentStatus.DOWN,
                error=str(exc),
            )

    def _check_incidents_health(self) -> ComponentHealth:
        """Health check for the IncidentManager."""
        try:
            stats = self._incidents.get_stats()
            return ComponentHealth(
                name="incidents",
                status=ComponentStatus.UP,
                message=f"{stats.get('total', 0)} incidents tracked",
                details=stats,
            )
        except Exception as exc:
            return ComponentHealth(
                name="incidents",
                status=ComponentStatus.DOWN,
                error=str(exc),
            )

    def _check_missions_health(self) -> ComponentHealth:
        """Health check for the MissionPlanner."""
        try:
            stats = self._missions.get_stats()
            return ComponentHealth(
                name="missions",
                status=ComponentStatus.UP,
                message=f"{stats.get('total', 0)} missions tracked",
                details=stats,
            )
        except Exception as exc:
            return ComponentHealth(
                name="missions",
                status=ComponentStatus.DOWN,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Update stream management
    # ------------------------------------------------------------------

    def _emit_update(self, update: PictureUpdate) -> None:
        """Store an update and notify all subscribers."""
        with self._lock:
            self._updates.append(update)
            # Trim to max_updates
            if len(self._updates) > self._max_updates:
                self._updates = self._updates[-self._max_updates:]

        # Notify subscribers (outside the main lock to avoid deadlocks)
        with self._subscriber_lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(update)
            except Exception:
                logger.debug(
                    "Subscriber callback failed for %s",
                    update.update_type.value,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Subscription API
    # ------------------------------------------------------------------

    def subscribe(self, callback: PictureSubscriber) -> None:
        """Subscribe to real-time operating picture updates.

        The callback is invoked for every PictureUpdate generated by
        any subsystem. Callbacks should be fast and non-blocking.

        Parameters
        ----------
        callback:
            Function that accepts a PictureUpdate.
        """
        with self._subscriber_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
        logger.debug("Subscriber added (total: %d)", len(self._subscribers))

    def unsubscribe(self, callback: PictureSubscriber) -> bool:
        """Remove a subscriber. Returns True if found and removed."""
        with self._subscriber_lock:
            try:
                self._subscribers.remove(callback)
                return True
            except ValueError:
                return False

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        with self._subscriber_lock:
            return len(self._subscribers)

    # ------------------------------------------------------------------
    # Core query API — the main interface
    # ------------------------------------------------------------------

    def get_picture(self) -> OperatingPicture:
        """Get the full operating picture — current state of everything.

        This is the primary query method. It assembles a complete snapshot
        from all subsystems into a single OperatingPicture.

        Returns
        -------
        OperatingPicture
            Complete situational awareness snapshot.
        """
        now = time.time()

        # Fused targets
        fused_targets = self._fusion.get_fused_targets()
        target_dicts = [ft.to_dict() for ft in fused_targets]
        multi_source = sum(1 for ft in fused_targets if ft.source_count >= 2)

        # Alerts (recent within window)
        alert_cutoff = now - self._alert_window
        recent_alerts = self._alerting.get_history(limit=50, since=alert_cutoff)
        alert_dicts = [a.to_dict() for a in recent_alerts]

        # Anomalies (recent within window)
        anomaly_cutoff = now - self._anomaly_window
        recent_anomalies = self._anomaly.get_alert_history(limit=50)
        active_anomalies = [
            a for a in recent_anomalies
            if a.timestamp >= anomaly_cutoff and not a.suppressed
        ]
        anomaly_dicts = [a.to_dict() for a in active_anomalies]

        # Incidents (active = not resolved or closed)
        active_incidents = self._incidents.get_all(state=None)
        active_incidents = [
            inc for inc in active_incidents
            if inc.state.value not in ("resolved", "closed")
        ]
        incident_dicts = [inc.to_dict() for inc in active_incidents]

        # Missions (active = not completed or aborted)
        all_missions = self._missions.get_missions()
        active_missions = [
            m for m in all_missions
            if m.state.value not in ("completed", "aborted")
        ]
        mission_dicts = [m.to_dict() for m in active_missions]

        # System health
        health_status = self._health.check_all()
        health_dict = health_status.to_dict()

        # Analytics
        analytics_snap = self._analytics.snapshot()

        # Zones
        zones = self._fusion.get_zones()
        zone_dicts = [z.to_dict() for z in zones]

        # Compute threat level
        threat_level = self._compute_threat_level(
            alert_count=len(recent_alerts),
            anomaly_count=len(active_anomalies),
            incident_count=len(active_incidents),
            recent_alerts=recent_alerts,
            health_overall=health_status.overall.value,
        )

        # Build summary
        summary = self._build_summary(
            target_count=len(fused_targets),
            multi_source=multi_source,
            alert_count=len(recent_alerts),
            anomaly_count=len(active_anomalies),
            incident_count=len(active_incidents),
            mission_count=len(active_missions),
            health_overall=health_status.overall.value,
            threat_level=threat_level,
        )

        with self._lock:
            self._last_picture_time = now

        return OperatingPicture(
            timestamp=now,
            targets=target_dicts,
            target_count=len(fused_targets),
            multi_source_targets=multi_source,
            alerts=alert_dicts,
            active_alert_count=len(recent_alerts),
            anomalies=anomaly_dicts,
            active_anomaly_count=len(active_anomalies),
            incidents=incident_dicts,
            incident_count=len(active_incidents),
            missions=mission_dicts,
            mission_count=len(active_missions),
            health=health_dict,
            analytics=analytics_snap,
            zones=zone_dicts,
            zone_count=len(zones),
            summary=summary,
            threat_level=threat_level,
        )

    def get_updates_since(self, timestamp: float) -> list[PictureUpdate]:
        """Get all operating picture updates since a given timestamp.

        Used for delta synchronization: a consumer calls ``get_picture()``
        once, then polls ``get_updates_since()`` with the picture's
        timestamp to receive only the changes.

        Parameters
        ----------
        timestamp:
            Epoch seconds. Returns updates with timestamp > this value.

        Returns
        -------
        list[PictureUpdate]
            Updates sorted oldest first.
        """
        with self._lock:
            updates = [u for u in self._updates if u.timestamp > timestamp]
        updates.sort(key=lambda u: u.timestamp)
        return updates

    def get_updates_by_type(
        self,
        update_type: UpdateType,
        *,
        limit: int = 50,
        since: float = 0.0,
    ) -> list[PictureUpdate]:
        """Get updates filtered by type.

        Parameters
        ----------
        update_type:
            Only return updates of this type.
        limit:
            Maximum number of updates to return.
        since:
            Only return updates after this timestamp.

        Returns
        -------
        list[PictureUpdate]
            Updates sorted newest first.
        """
        with self._lock:
            updates = [
                u for u in self._updates
                if u.update_type == update_type
                and u.timestamp > since
            ]
        updates.sort(key=lambda u: u.timestamp, reverse=True)
        return updates[:limit]

    # ------------------------------------------------------------------
    # Convenience methods — common queries
    # ------------------------------------------------------------------

    def get_target_picture(self, target_id: str) -> dict[str, Any] | None:
        """Get the full picture for a single target.

        Combines fusion data, recent alerts, anomalies, and incident
        associations into one comprehensive target view.

        Parameters
        ----------
        target_id:
            The target ID to look up.

        Returns
        -------
        dict or None
            Comprehensive target information, or None if not found.
        """
        fused = self._fusion.get_fused_target(target_id)
        if fused is None:
            return None

        # Get dossier if available
        dossier = self._fusion.get_target_dossier(target_id)

        # Get alerts involving this target
        alerts = self._alerting.get_history(limit=20, target_id=target_id)

        # Get anomalies involving this target
        anomalies = self._anomaly.get_alert_history(
            limit=20, target_id=target_id,
        )

        # Get incidents involving this target
        all_incidents = self._incidents.get_all()
        target_incidents = [
            inc for inc in all_incidents
            if target_id in getattr(inc, "target_ids", [])
        ]

        return {
            "target": fused.to_dict(),
            "dossier": dossier,
            "alerts": [a.to_dict() for a in alerts],
            "anomalies": [a.to_dict() for a in anomalies],
            "incidents": [inc.to_dict() for inc in target_incidents],
            "timestamp": time.time(),
        }

    def get_zone_picture(self, zone_id: str) -> dict[str, Any]:
        """Get the full picture for a specific zone.

        Returns
        -------
        dict
            Zone activity, targets, alerts, and anomalies.
        """
        # Zone activity from fusion
        zone_activity = self._fusion.get_zone_activity(zone_id)

        # Zone targets
        zone_targets = self._fusion.get_targets_in_zone(zone_id)

        # Zone alerts
        zone_alerts = self._alerting.get_history(limit=20, zone_id=zone_id)

        # Zone anomalies
        zone_anomalies = self._anomaly.get_alert_history(
            limit=20, zone_id=zone_id,
        )

        # Zone baseline
        baseline = self._anomaly.get_zone_stats(zone_id)

        return {
            "zone_activity": zone_activity,
            "targets": [ft.to_dict() for ft in zone_targets],
            "target_count": len(zone_targets),
            "alerts": [a.to_dict() for a in zone_alerts],
            "anomalies": [a.to_dict() for a in zone_anomalies],
            "baseline": baseline,
            "timestamp": time.time(),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get engine-wide statistics from all subsystems.

        Returns
        -------
        dict
            Aggregated statistics from every subsystem.
        """
        with self._lock:
            update_count = len(self._updates)

        return {
            "fusion": {
                "target_count": len(self._fusion.get_fused_targets()),
                "zone_count": len(self._fusion.get_zones()),
            },
            "alerting": self._alerting.get_stats(),
            "anomaly": self._anomaly.get_stats(),
            "analytics": {
                "detection_rate": self._analytics.detection_rate,
                "alert_rate": self._analytics.alert_rate,
                "correlation_success_rate": self._analytics.correlation_success_rate,
            },
            "incidents": self._incidents.get_stats(),
            "missions": self._missions.get_stats(),
            "health": self._health.check_all().to_dict(),
            "sitaware": {
                "update_count": update_count,
                "subscriber_count": self.subscriber_count,
                "max_updates": self._max_updates,
                "known_target_ids": len(self._known_target_ids),
            },
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Threat level computation
    # ------------------------------------------------------------------

    def _compute_threat_level(
        self,
        alert_count: int,
        anomaly_count: int,
        incident_count: int,
        recent_alerts: list[AlertRecord],
        health_overall: str,
    ) -> str:
        """Compute the overall threat level.

        Rules:
          - RED:    Any critical incident or critical alert, or system is down
          - ORANGE: Multiple high-severity alerts or active incidents
          - YELLOW: Some alerts or anomalies, or system is degraded
          - GREEN:  No significant activity, system healthy
        """
        # Check for critical alerts
        has_critical = any(
            a.severity in ("critical", "error") for a in recent_alerts
        )

        if has_critical or health_overall == "down":
            return "red"

        # Check for high-severity alerts
        high_alerts = sum(
            1 for a in recent_alerts if a.severity in ("warning", "high")
        )

        if incident_count >= 2 or high_alerts >= 3:
            return "orange"

        if alert_count > 0 or anomaly_count > 0 or incident_count > 0 or health_overall == "degraded":
            return "yellow"

        return "green"

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        target_count: int,
        multi_source: int,
        alert_count: int,
        anomaly_count: int,
        incident_count: int,
        mission_count: int,
        health_overall: str,
        threat_level: str,
    ) -> str:
        """Build a human-readable one-line summary of the current situation."""
        parts: list[str] = []

        # Threat preamble
        threat_labels = {
            "green": "ALL CLEAR",
            "yellow": "ELEVATED",
            "orange": "HIGH ALERT",
            "red": "CRITICAL",
        }
        parts.append(f"[{threat_labels.get(threat_level, 'UNKNOWN')}]")

        # Targets
        if target_count == 0:
            parts.append("No targets tracked.")
        else:
            t_str = f"{target_count} target{'s' if target_count != 1 else ''}"
            if multi_source > 0:
                t_str += f" ({multi_source} multi-sensor)"
            parts.append(f"{t_str}.")

        # Active issues
        issues: list[str] = []
        if alert_count > 0:
            issues.append(f"{alert_count} alert{'s' if alert_count != 1 else ''}")
        if anomaly_count > 0:
            issues.append(f"{anomaly_count} anomal{'ies' if anomaly_count != 1 else 'y'}")
        if incident_count > 0:
            issues.append(f"{incident_count} incident{'s' if incident_count != 1 else ''}")
        if issues:
            parts.append(f"Active: {', '.join(issues)}.")

        # Missions
        if mission_count > 0:
            parts.append(f"{mission_count} mission{'s' if mission_count != 1 else ''} active.")

        # Health
        if health_overall != "up":
            parts.append(f"System health: {health_overall}.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # QueueEventBus bridge (used when SC-style queue bus is supplied)
    # ------------------------------------------------------------------

    def _start_queue_bridge(self) -> None:
        """Start the queue->topic bridge thread (if a QueueEventBus was given).

        Subscribes to the external QueueEventBus, drains messages of the
        form ``{"type": topic, "data": ...}`` from the returned queue,
        and republishes them onto SitAware's internal topic EventBus so
        subscribe(topic, callback) wiring fires.
        """
        if self._external_queue_bus is None:
            return
        if self._bridge_running:
            return

        self._bridge_queue = self._external_queue_bus.subscribe()
        self._bridge_running = True
        self._bridge_thread = threading.Thread(
            target=self._queue_bridge_loop,
            daemon=True,
            name="sitaware-queue-bridge",
        )
        self._bridge_thread.start()
        logger.info(
            "SitAwareEngine: queue->topic bridge thread started"
        )

    def _stop_queue_bridge(self) -> None:
        """Stop the queue->topic bridge thread (if running)."""
        if not self._bridge_running:
            return
        self._bridge_running = False
        if (
            self._bridge_thread is not None
            and self._bridge_thread.is_alive()
        ):
            self._bridge_thread.join(timeout=2.0)
        if (
            self._external_queue_bus is not None
            and self._bridge_queue is not None
        ):
            try:
                self._external_queue_bus.unsubscribe(self._bridge_queue)
            except Exception:
                pass

    def _queue_bridge_loop(self) -> None:
        """Drain the external queue bus and republish on internal topic bus."""
        import queue as _qmod
        while self._bridge_running:
            try:
                msg = self._bridge_queue.get(timeout=0.5)
            except _qmod.Empty:
                continue
            except Exception:
                continue
            try:
                if not isinstance(msg, dict):
                    continue
                topic = msg.get("type", "")
                data = msg.get("data")
                if topic:
                    self._event_bus.publish(topic, data)
            except Exception:
                logger.debug("queue->topic bridge error", exc_info=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return True if the engine has been started and is operational.

        An engine is "running" when its alerting subsystem is started
        and (if a queue bus was supplied) the bridge thread is alive.
        """
        # AlertEngine uses _started; some other subsystems use _running.
        # Accept either to be robust across versions.
        alerting_running = False
        try:
            for attr in ("_started", "_running", "is_running"):
                v = getattr(self._alerting, attr, None)
                if v is None:
                    continue
                alerting_running = bool(v() if callable(v) else v)
                if alerting_running:
                    break
        except Exception:
            alerting_running = False
        if self._external_queue_bus is not None:
            bridge_alive = bool(
                self._bridge_running
                and self._bridge_thread is not None
                and self._bridge_thread.is_alive()
            )
            return alerting_running and bridge_alive
        return alerting_running

    def start(self) -> None:
        """Start all subsystem engines that have background loops.

        Starts the AlertEngine event subscriptions and the FusionEngine
        correlator. Call this after all setup is complete.
        """
        self._alerting.start()
        # If we were given a QueueEventBus, start the bridge so that
        # subsystem events reach our topic-based wiring.
        self._start_queue_bridge()
        logger.info("SitAwareEngine started — all subsystems online")

    def stop(self) -> None:
        """Stop all subsystem background loops."""
        self._stop_queue_bridge()
        self._alerting.stop()
        self._fusion.shutdown()
        logger.info("SitAwareEngine stopped")

    def reset(self) -> None:
        """Reset all state across all subsystems. Useful for testing."""
        self.stop()
        self._fusion.clear()
        self._alerting.reset()
        self._anomaly.reset()
        self._analytics.clear()
        self._incidents.reset() if hasattr(self._incidents, "reset") else None
        with self._lock:
            self._updates.clear()
            self._known_target_ids.clear()
            self._last_picture_time = 0.0
        with self._subscriber_lock:
            self._subscribers.clear()
        logger.info("SitAwareEngine reset complete")

    def shutdown(self) -> None:
        """Full shutdown — stop all engines and clear subscribers."""
        self.stop()
        with self._subscriber_lock:
            self._subscribers.clear()
        logger.info("SitAwareEngine shutdown complete")
