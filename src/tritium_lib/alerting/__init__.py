# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.alerting — rules engine for generating alerts and dispatching responses.

High-level orchestrator that subscribes to FusionEngine, GeofenceEngine, and
AnomalyEngine events, evaluates AlertRules against incoming data, and dispatches
responses (notify, log, escalate, dispatch) via NotificationManager.

Built-in rules:
  - geofence_entry: fires when a target enters any monitored geofence zone
  - threat_level_change: fires when a target's threat level escalates
  - sensor_offline: fires when a sensor stops reporting
  - target_loitering: fires when a target dwells too long in a zone

Quick start::

    from tritium_lib.alerting import AlertEngine, DispatchAction
    from tritium_lib.events.bus import EventBus
    from tritium_lib.notifications import NotificationManager

    bus = EventBus()
    notifications = NotificationManager()
    engine = AlertEngine(event_bus=bus, notification_manager=notifications)

    # Built-in rules are loaded automatically
    engine.start()

    # Or evaluate a single event manually
    alerts = engine.evaluate_event("geofence:enter", {
        "target_id": "ble_aabbccdd",
        "zone_id": "perimeter",
        "zone_type": "restricted",
    })

Architecture
------------
- **AlertRule** — condition + severity + action (from models.alert_rules)
- **AlertEngine** — evaluates rules against incoming events
- **DispatchAction** — what to do when a rule triggers (notify, log, escalate, dispatch)
- **AlertRecord** — an immutable record of a fired alert with full context
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from tritium_lib.models.alert_rules import (
    AlertCondition,
    AlertRule,
    AlertTrigger,
    ConditionOperator,
    DEFAULT_ALERT_RULES,
)
from tritium_lib.models.notification_rules import (
    NotificationChannel,
    NotificationSeverity,
)

logger = logging.getLogger("alerting")


# ---------------------------------------------------------------------------
# DispatchAction — what to do when a rule fires
# ---------------------------------------------------------------------------

class DispatchAction(str, Enum):
    """Actions the alert engine can take when a rule fires."""
    NOTIFY = "notify"        # Push a notification via NotificationManager
    LOG = "log"              # Write to alert log only
    ESCALATE = "escalate"    # Publish escalation event to EventBus
    DISPATCH = "dispatch"    # Trigger a dispatch command (e.g., send drone)
    SUPPRESS = "suppress"    # Acknowledge but suppress (for known patterns)


# ---------------------------------------------------------------------------
# AlertRecord — immutable record of a fired alert
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlertRecord:
    """An immutable record of a fired alert with full context.

    Created each time an AlertRule fires. Stored in the alert history
    for audit trail and dashboard display.
    """
    record_id: str
    rule_id: str
    rule_name: str
    trigger: str
    severity: str
    action: str
    message: str
    event_data: dict
    target_id: str = ""
    zone_id: str = ""
    device_id: str = ""
    timestamp: float = 0.0
    notification_id: str = ""  # ID from NotificationManager, if generated

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "trigger": self.trigger,
            "severity": self.severity,
            "action": self.action,
            "message": self.message,
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "notification_id": self.notification_id,
        }


# ---------------------------------------------------------------------------
# Event-to-trigger mapping
# ---------------------------------------------------------------------------

# Maps EventBus topic patterns to AlertTrigger enum values
_EVENT_TRIGGER_MAP: dict[str, AlertTrigger] = {
    "geofence:enter": AlertTrigger.TARGET_ENTER_ZONE,
    "geofence:exit": AlertTrigger.TARGET_EXIT_ZONE,
    "fusion.target.correlated": AlertTrigger.CORRELATION_EVENT,
    "fusion.sensor.ingested": AlertTrigger.TARGET_NEW,
    "anomaly.alert": AlertTrigger.THREAT_DETECTED,
    "anomaly.alert.speed": AlertTrigger.TARGET_SPEED,
    "anomaly.alert.dwell": AlertTrigger.TARGET_LOITER,
    "anomaly.alert.route": AlertTrigger.THREAT_DETECTED,
    "anomaly.alert.count": AlertTrigger.THREAT_DETECTED,
    "anomaly.alert.reappearance": AlertTrigger.THREAT_DETECTED,
    "sensor.offline": AlertTrigger.DEVICE_OFFLINE,
    "sensor.health.degraded": AlertTrigger.DEVICE_ERROR,
    "device.heartbeat.missed": AlertTrigger.DEVICE_OFFLINE,
    "target.threat_level.changed": AlertTrigger.THREAT_DETECTED,
    "target.new": AlertTrigger.TARGET_NEW,
    "target.lost": AlertTrigger.TARGET_LOST,
    "dwell.event": AlertTrigger.TARGET_LOITER,
    "sensor.motion": AlertTrigger.SENSOR_MOTION,
    "sensor.acoustic": AlertTrigger.SENSOR_ACOUSTIC,
    "geofence.breach": AlertTrigger.GEOFENCE_BREACH,
    "device.battery_low": AlertTrigger.DEVICE_BATTERY_LOW,
}


# ---------------------------------------------------------------------------
# Built-in alert rules (supplements DEFAULT_ALERT_RULES from models)
# ---------------------------------------------------------------------------

def _builtin_rules() -> list[AlertRule]:
    """Create the four built-in domain rules.

    These complement the DEFAULT_ALERT_RULES from the models layer with
    higher-level behavioral rules specific to the alerting engine.
    """
    return [
        AlertRule(
            rule_id="builtin-geofence-entry",
            name="Geofence entry alert",
            trigger=AlertTrigger.TARGET_ENTER_ZONE,
            severity=NotificationSeverity.WARNING,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.LOG],
            message_template="TARGET ENTERED ZONE: {target_id} entered {zone_name} ({zone_type})",
            cooldown_seconds=30,
            tags=["builtin", "geofence"],
        ),
        AlertRule(
            rule_id="builtin-threat-level-change",
            name="Threat level escalation",
            trigger=AlertTrigger.THREAT_DETECTED,
            conditions=[
                AlertCondition(
                    field="severity",
                    operator=ConditionOperator.IN_LIST,
                    value=["high", "critical"],
                ),
            ],
            severity=NotificationSeverity.CRITICAL,
            channels=[
                NotificationChannel.WEBSOCKET,
                NotificationChannel.MQTT,
                NotificationChannel.LOG,
            ],
            message_template="THREAT ESCALATION: {target_id} — {detail}",
            cooldown_seconds=15,
            tags=["builtin", "threat"],
        ),
        AlertRule(
            rule_id="builtin-sensor-offline",
            name="Sensor offline alert",
            trigger=AlertTrigger.DEVICE_OFFLINE,
            severity=NotificationSeverity.WARNING,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.LOG],
            message_template="SENSOR OFFLINE: {device_id} — last seen {last_seen}",
            cooldown_seconds=300,
            tags=["builtin", "sensor"],
        ),
        AlertRule(
            rule_id="builtin-target-loitering",
            name="Target loitering alert",
            trigger=AlertTrigger.TARGET_LOITER,
            conditions=[
                AlertCondition(
                    field="duration_seconds",
                    operator=ConditionOperator.GREATER_THAN,
                    value=300,
                ),
            ],
            severity=NotificationSeverity.INFO,
            channels=[NotificationChannel.WEBSOCKET],
            message_template="LOITERING: {target_id} stationary for {duration_seconds}s in {zone_id}",
            cooldown_seconds=120,
            tags=["builtin", "dwell"],
        ),
    ]


# ---------------------------------------------------------------------------
# AlertEngine — the central rules engine
# ---------------------------------------------------------------------------

class AlertEngine:
    """Rules engine for generating alerts and dispatching responses.

    Subscribes to FusionEngine, GeofenceEngine, and AnomalyEngine events
    via the EventBus, evaluates AlertRules against each event, and
    dispatches actions (notify, log, escalate, dispatch) when rules fire.

    Thread-safe. All public methods acquire the internal lock.

    Parameters
    ----------
    event_bus:
        EventBus instance to subscribe to for incoming events.
    notification_manager:
        NotificationManager for creating user-facing notifications.
    load_defaults:
        If True (default), loads built-in rules and DEFAULT_ALERT_RULES.
    default_action:
        Default DispatchAction for rules without an explicit action mapping.
    max_history:
        Maximum number of AlertRecords to retain.
    action_handlers:
        Optional dict mapping DispatchAction -> callable(AlertRecord).
        Allows custom dispatch logic (e.g., send drone, trigger alarm).
    """

    def __init__(
        self,
        event_bus=None,
        notification_manager=None,
        *,
        load_defaults: bool = True,
        default_action: DispatchAction = DispatchAction.NOTIFY,
        max_history: int = 5000,
        action_handlers: dict[DispatchAction, Callable[[AlertRecord], None]] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._notification_manager = notification_manager
        self._lock = threading.Lock()

        # Configuration
        self._default_action = default_action
        self._max_history = max_history

        # Rule storage: rule_id -> AlertRule
        self._rules: dict[str, AlertRule] = {}

        # Per-rule action override: rule_id -> DispatchAction
        self._rule_actions: dict[str, DispatchAction] = {}

        # Custom action handlers
        self._action_handlers: dict[DispatchAction, Callable[[AlertRecord], None]] = (
            dict(action_handlers) if action_handlers else {}
        )

        # Alert history (bounded ring buffer)
        self._history: list[AlertRecord] = []

        # Counters
        self._total_events_processed = 0
        self._total_alerts_fired = 0
        self._total_suppressed = 0

        # Subscription tracking for cleanup
        self._subscribed_topics: list[str] = []
        self._started = False

        # Load default rules
        if load_defaults:
            for rule in _builtin_rules():
                self._rules[rule.rule_id] = rule
            for rule in DEFAULT_ALERT_RULES:
                self._rules[rule.rule_id] = rule

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        rule: AlertRule,
        action: DispatchAction | None = None,
    ) -> AlertRule:
        """Add or update an alert rule.

        Parameters
        ----------
        rule:
            The AlertRule to add.
        action:
            Optional DispatchAction override for this rule.

        Returns the rule.
        """
        with self._lock:
            self._rules[rule.rule_id] = rule
            if action is not None:
                self._rule_actions[rule.rule_id] = action
        logger.info("Alert rule added: %s (%s)", rule.name, rule.rule_id)
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if found and removed."""
        with self._lock:
            if rule_id not in self._rules:
                return False
            del self._rules[rule_id]
            self._rule_actions.pop(rule_id, None)
        logger.info("Alert rule removed: %s", rule_id)
        return True

    def get_rule(self, rule_id: str) -> AlertRule | None:
        """Get a rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def get_rules(self) -> list[AlertRule]:
        """Return all rules."""
        with self._lock:
            return list(self._rules.values())

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule. Returns True if found."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None:
                return False
            rule.enabled = True
        return True

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule. Returns True if found."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None:
                return False
            rule.enabled = False
        return True

    def set_rule_action(self, rule_id: str, action: DispatchAction) -> bool:
        """Set the dispatch action for a specific rule. Returns True if found."""
        with self._lock:
            if rule_id not in self._rules:
                return False
            self._rule_actions[rule_id] = action
        return True

    def register_action_handler(
        self,
        action: DispatchAction,
        handler: Callable[[AlertRecord], None],
    ) -> None:
        """Register a custom handler for a dispatch action type.

        When a rule fires with this action, the handler is called with
        the AlertRecord. Useful for custom dispatch logic (send drone,
        trigger alarm, call API, etc.).
        """
        with self._lock:
            self._action_handlers[action] = handler

    # ------------------------------------------------------------------
    # Event subscription (EventBus integration)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to all relevant EventBus topics.

        Subscribes to geofence, fusion, anomaly, sensor health, dwell,
        and target events. Call stop() to unsubscribe.
        """
        if self._event_bus is None:
            logger.warning("AlertEngine.start() called without event_bus")
            return

        if self._started:
            return

        topics = list(_EVENT_TRIGGER_MAP.keys())
        for topic in topics:
            self._event_bus.subscribe(topic, self._on_event)
            self._subscribed_topics.append(topic)

        self._started = True
        logger.info(
            "AlertEngine started, subscribed to %d topics with %d rules",
            len(topics),
            len(self._rules),
        )

    def stop(self) -> None:
        """Unsubscribe from all EventBus topics."""
        if self._event_bus is None or not self._started:
            return

        for topic in self._subscribed_topics:
            try:
                self._event_bus.unsubscribe(topic, self._on_event)
            except Exception:
                pass

        self._subscribed_topics.clear()
        self._started = False
        logger.info("AlertEngine stopped")

    def _on_event(self, event) -> None:
        """EventBus callback — route event to evaluate_event.

        Handles both Event objects (from EventBus) and plain dicts.
        """
        if hasattr(event, "topic"):
            topic = event.topic
            data = event.data if hasattr(event, "data") else {}
        elif isinstance(event, dict):
            topic = event.get("type", event.get("topic", ""))
            data = event.get("data", event)
        else:
            return

        if isinstance(data, dict):
            event_data = dict(data)
        else:
            event_data = {"data": data}

        try:
            self.evaluate_event(topic, event_data)
        except Exception:
            logger.debug("Error evaluating event %s", topic, exc_info=True)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_event(
        self,
        event_topic: str,
        event_data: dict,
    ) -> list[AlertRecord]:
        """Evaluate an event against all rules and dispatch actions.

        This is the main entry point for processing events. It can be
        called directly for manual event injection or is called
        automatically by the EventBus subscription.

        Parameters
        ----------
        event_topic:
            The event topic/type string (e.g., "geofence:enter").
        event_data:
            Dictionary of event data.

        Returns
        -------
        list[AlertRecord]:
            All alerts that fired for this event.
        """
        # Map the event topic to an AlertTrigger
        trigger = _EVENT_TRIGGER_MAP.get(event_topic)
        if trigger is None:
            # Try to match the raw topic string to a trigger value
            for t in AlertTrigger:
                if t.value == event_topic:
                    trigger = t
                    break
            if trigger is None:
                # Unknown event — still increment counter
                with self._lock:
                    self._total_events_processed += 1
                return []

        trigger_value = trigger.value
        fired: list[AlertRecord] = []

        with self._lock:
            self._total_events_processed += 1

            for rule in self._rules.values():
                if not rule.matches(trigger_value, event_data):
                    continue

                if not rule.is_cooled_down():
                    self._total_suppressed += 1
                    continue

                # Rule fires
                action = self._rule_actions.get(rule.rule_id, self._default_action)
                message = rule.render_message(
                    trigger=trigger_value,
                    severity=rule.severity.value,
                    target_id=event_data.get("target_id", ""),
                    device_id=event_data.get("device_id", ""),
                    zone_id=event_data.get("zone_id", ""),
                    zone_name=event_data.get("zone_name", ""),
                    zone_type=event_data.get("zone_type", ""),
                    message=event_data.get("detail", event_data.get("message", "")),
                    detail=event_data.get("detail", ""),
                    duration_seconds=str(event_data.get("duration_seconds", "")),
                    last_seen=str(event_data.get("last_seen", "")),
                    battery_level=str(event_data.get("battery_level", "")),
                    timestamp=str(time.time()),
                )

                record = AlertRecord(
                    record_id=uuid.uuid4().hex[:12],
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    trigger=trigger_value,
                    severity=rule.severity.value,
                    action=action.value,
                    message=message,
                    event_data=dict(event_data),
                    target_id=event_data.get("target_id", ""),
                    zone_id=event_data.get("zone_id", ""),
                    device_id=event_data.get("device_id", ""),
                    timestamp=time.time(),
                )

                rule.record_firing()
                self._total_alerts_fired += 1
                self._history.append(record)

                # Trim history
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

                fired.append(record)

        # Dispatch actions outside the lock
        for record in fired:
            self._dispatch(record)

        return fired

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, record: AlertRecord) -> None:
        """Execute the dispatch action for a fired alert."""
        action = DispatchAction(record.action)

        # Always log
        logger.info(
            "ALERT [%s] %s: %s (rule=%s)",
            record.severity,
            record.trigger,
            record.message,
            record.rule_id,
        )

        # Custom handler takes priority
        handler = self._action_handlers.get(action)
        if handler is not None:
            try:
                handler(record)
            except Exception:
                logger.debug(
                    "Action handler failed for %s", action, exc_info=True
                )
            return

        if action == DispatchAction.NOTIFY:
            self._dispatch_notify(record)
        elif action == DispatchAction.ESCALATE:
            self._dispatch_escalate(record)
        elif action == DispatchAction.DISPATCH:
            self._dispatch_command(record)
        elif action == DispatchAction.LOG:
            pass  # Already logged above
        elif action == DispatchAction.SUPPRESS:
            pass  # Acknowledged but suppressed

    def _dispatch_notify(self, record: AlertRecord) -> None:
        """Create a notification via NotificationManager."""
        if self._notification_manager is None:
            return

        severity_map = {
            "critical": "critical",
            "error": "critical",
            "warning": "warning",
            "info": "info",
            "debug": "info",
        }
        notif_severity = severity_map.get(record.severity, "info")

        try:
            self._notification_manager.add(
                title=record.rule_name,
                message=record.message,
                severity=notif_severity,
                source="alerting",
                entity_id=record.target_id or record.device_id or None,
            )
        except Exception:
            logger.debug("Failed to create notification", exc_info=True)

    def _dispatch_escalate(self, record: AlertRecord) -> None:
        """Publish an escalation event to the EventBus."""
        if self._event_bus is None:
            return

        try:
            self._event_bus.publish(
                "alert.escalation",
                data=record.to_dict(),
                source="alerting",
            )
        except Exception:
            logger.debug("Failed to publish escalation", exc_info=True)

    def _dispatch_command(self, record: AlertRecord) -> None:
        """Publish a dispatch command event to the EventBus."""
        if self._event_bus is None:
            return

        try:
            self._event_bus.publish(
                "alert.dispatch",
                data={
                    "record": record.to_dict(),
                    "target_id": record.target_id,
                    "zone_id": record.zone_id,
                    "severity": record.severity,
                },
                source="alerting",
            )
        except Exception:
            logger.debug("Failed to publish dispatch command", exc_info=True)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_history(
        self,
        limit: int = 100,
        *,
        rule_id: str = "",
        severity: str = "",
        target_id: str = "",
        zone_id: str = "",
        since: float = 0.0,
    ) -> list[AlertRecord]:
        """Retrieve alert history with optional filtering.

        Parameters
        ----------
        limit:
            Maximum records to return.
        rule_id:
            Filter by rule ID.
        severity:
            Filter by severity level.
        target_id:
            Filter by target ID.
        zone_id:
            Filter by zone ID.
        since:
            Only return records after this timestamp.

        Returns records newest-first.
        """
        with self._lock:
            records = list(self._history)

        if rule_id:
            records = [r for r in records if r.rule_id == rule_id]
        if severity:
            records = [r for r in records if r.severity == severity]
        if target_id:
            records = [r for r in records if r.target_id == target_id]
        if zone_id:
            records = [r for r in records if r.zone_id == zone_id]
        if since > 0:
            records = [r for r in records if r.timestamp >= since]

        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Return engine-wide statistics."""
        with self._lock:
            rules_by_trigger: dict[str, int] = {}
            for rule in self._rules.values():
                key = rule.trigger.value
                rules_by_trigger[key] = rules_by_trigger.get(key, 0) + 1

            return {
                "total_rules": len(self._rules),
                "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
                "disabled_rules": sum(1 for r in self._rules.values() if not r.enabled),
                "total_events_processed": self._total_events_processed,
                "total_alerts_fired": self._total_alerts_fired,
                "total_suppressed": self._total_suppressed,
                "history_size": len(self._history),
                "max_history": self._max_history,
                "subscribed_topics": len(self._subscribed_topics),
                "started": self._started,
                "rules_by_trigger": rules_by_trigger,
                "action_handlers": list(self._action_handlers.keys()),
            }

    def get_rule_stats(self) -> list[dict[str, Any]]:
        """Return per-rule statistics."""
        with self._lock:
            result = []
            for rule in self._rules.values():
                result.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "trigger": rule.trigger.value,
                    "severity": rule.severity.value,
                    "enabled": rule.enabled,
                    "fire_count": rule.fire_count,
                    "last_fired_at": (
                        rule.last_fired_at.isoformat()
                        if rule.last_fired_at
                        else None
                    ),
                    "cooldown_seconds": rule.cooldown_seconds,
                    "action": self._rule_actions.get(
                        rule.rule_id, self._default_action
                    ).value,
                })
            return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear_history(self) -> int:
        """Clear alert history. Returns count removed."""
        with self._lock:
            count = len(self._history)
            self._history.clear()
            return count

    def reset(self) -> None:
        """Reset all state: stop subscriptions, clear history and rules."""
        self.stop()
        with self._lock:
            self._rules.clear()
            self._rule_actions.clear()
            self._history.clear()
            self._total_events_processed = 0
            self._total_alerts_fired = 0
            self._total_suppressed = 0

    def reset_counters(self) -> None:
        """Reset statistics counters without clearing rules or history."""
        with self._lock:
            self._total_events_processed = 0
            self._total_alerts_fired = 0
            self._total_suppressed = 0
            for rule in self._rules.values():
                rule.fire_count = 0
                rule.last_fired_at = None


__all__ = [
    "AlertEngine",
    "AlertRecord",
    "DispatchAction",
    # Re-exports from models for convenience
    "AlertRule",
    "AlertTrigger",
    "AlertCondition",
    "ConditionOperator",
    "NotificationChannel",
    "NotificationSeverity",
    "DEFAULT_ALERT_RULES",
]
