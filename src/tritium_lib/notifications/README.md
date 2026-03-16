# Notifications

**Where you are:** `tritium-lib/src/tritium_lib/notifications/`

**Parent:** [../](../) | [../../../CLAUDE.md](../../../CLAUDE.md)

## What This Is

Shared notification system used by tritium-sc (WebSocket broadcast to UI) and tritium-edge (fleet server alerts). Provides a Notification dataclass and a thread-safe NotificationManager that collects, stores, queries, marks read, and optionally broadcasts notifications via a callback.

## Key Files

| File | Purpose |
|------|---------|
| `__init__.py` | All code lives here — Notification dataclass and NotificationManager class |

## API

| Method | Purpose |
|--------|---------|
| `NotificationManager.add(title, message, severity, source, entity_id)` | Create and store a notification (info/warning/critical) |
| `NotificationManager.get_unread()` | Return all unread notifications |
| `NotificationManager.get_all(limit, since)` | Return notifications, newest first |
| `NotificationManager.mark_read(id)` | Mark one notification as read |
| `NotificationManager.mark_all_read()` | Mark all notifications as read |
| `NotificationManager.count_unread()` | Count of unread notifications |
| `NotificationManager.clear()` | Remove all notifications |

Broadcast callback receives `{"type": "notification:new", "data": {...}}` for each new notification. Used by tritium-sc to push to WebSocket clients.

## Related

- [../../../../tritium-sc/src/engine/comms/notifications.py](../../../../tritium-sc/src/engine/comms/notifications.py) — SC-side notification dispatch
- [../../../../tritium-sc/src/app/routers/ws.py](../../../../tritium-sc/src/app/routers/ws.py) — WebSocket endpoint that broadcasts notifications
- [../models/alert.py](../models/alert.py) — Alert model (related but separate from notifications)
