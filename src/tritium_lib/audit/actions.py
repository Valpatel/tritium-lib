# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Predefined audit action types for compliance tracking."""

from __future__ import annotations

from enum import Enum


class AuditAction(str, Enum):
    """Canonical action types recorded in the audit trail.

    Each action maps to a specific compliance-relevant operation.
    Custom actions are allowed via ``AuditTrail.record()``, but
    these predefined types ensure consistency across the system.
    """

    # --- Target operations ---------------------------------------------------
    TARGET_ACCESSED = "target_accessed"
    TARGET_CREATED = "target_created"
    TARGET_UPDATED = "target_updated"
    TARGET_DELETED = "target_deleted"
    TARGET_CLASSIFIED = "target_classified"
    TARGET_CORRELATED = "target_correlated"

    # --- Zone / geofence operations ------------------------------------------
    ZONE_CREATED = "zone_created"
    ZONE_MODIFIED = "zone_modified"
    ZONE_DELETED = "zone_deleted"
    ZONE_BREACH = "zone_breach"

    # --- Alert operations ----------------------------------------------------
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    ALERT_DISMISSED = "alert_dismissed"
    ALERT_ESCALATED = "alert_escalated"

    # --- Report operations ---------------------------------------------------
    REPORT_GENERATED = "report_generated"
    REPORT_EXPORTED = "report_exported"

    # --- Configuration -------------------------------------------------------
    CONFIG_CHANGED = "config_changed"
    CONFIG_RESET = "config_reset"

    # --- Authentication / Authorization --------------------------------------
    AUTH_LOGIN = "auth_login"
    AUTH_LOGOUT = "auth_logout"
    AUTH_FAILED = "auth_failed"
    AUTH_TOKEN_ISSUED = "auth_token_issued"
    AUTH_PERMISSION_DENIED = "auth_permission_denied"

    # --- System operations ---------------------------------------------------
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"

    # --- Sensor operations ---------------------------------------------------
    SENSOR_ADDED = "sensor_added"
    SENSOR_REMOVED = "sensor_removed"
    SENSOR_CALIBRATED = "sensor_calibrated"

    # --- Plugin operations ---------------------------------------------------
    PLUGIN_ENABLED = "plugin_enabled"
    PLUGIN_DISABLED = "plugin_disabled"

    # --- Data operations -----------------------------------------------------
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    DATA_PURGED = "data_purged"
