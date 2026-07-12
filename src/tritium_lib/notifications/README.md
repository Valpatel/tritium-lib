# tritium_lib.notifications

A tiny, thread-safe notification store: a `Notification` dataclass and a
`NotificationManager` that collects, queries, marks-read, and optionally
broadcasts. Deliberately minimal — the shared *shape* of a notification and
a reference in-memory manager, ~140 lines total.

**Where you are:** `tritium-lib/src/tritium_lib/notifications/`

## What it's for

Anything in the system may want to say "a target entered the perimeter."
This package defines what that notification *is* (`Notification`:
id/title/message/severity/source/timestamp/read/entity_id) and gives a
bounded, lock-guarded manager to hold a stream of them with an optional
broadcast callback. It is the lightweight lib-side model; richer,
subscription-driven dispatch lives elsewhere (see below).

## Files

| File | What's in it |
|------|--------------|
| `__init__.py` | Everything: the `Notification` dataclass and `NotificationManager` (thread-safe, `max_notifications`-bounded ring, optional `broadcast` callback). |
| `demos/notification_demo.py` | Runnable demo. |

## API

| Method | Purpose |
|--------|---------|
| `add(title, message, severity, source, entity_id)` | Store a notification (`severity` coerced to `info`/`warning`/`critical`), fire the broadcast callback, return its id |
| `get_unread()` | Unread notifications as dicts |
| `get_all(limit, since)` | Notifications, newest first |
| `mark_read(id)` / `mark_all_read()` | Mark read |
| `count_unread()` | Unread count |
| `clear()` | Drop all, return count removed |

The optional `broadcast` callback receives
`{"type": "notification:new", "data": {...}}` for each `add()` — best-effort
(exceptions swallowed). `add()` is never throttled; any anti-spam is the
caller's job.

## How it's consumed (verified 2026-07-11)

Honest status: **lib-internal only.** No `from tritium_lib.notifications`
import exists in non-test `tritium-sc`, `tritium-edge`, or `tritium-addons`
code.

- Inside lib, `alerting/__init__.py:20` type-imports `NotificationManager`
  as the optional `notification_manager` target for `DispatchAction.NOTIFY`.
  That's the one real consumer.
- **tritium-sc does not use this class.** SC has its **own, separate,
  richer** `NotificationManager` at
  `tritium-sc/src/engine/comms/notifications.py` — EventBus-subscribed,
  with per-`(event_type, entity_key)` cooldown dedup, a global rate cap,
  and WebSocket broadcast. It does not import or subclass this package.
  (Earlier revisions of this README claimed SC and the edge fleet-server
  consume it directly — corrected here after a source grep; they do not.)

4 test files cover this package.

## Related

- [../alerting/](../alerting/) — the one real consumer (`NOTIFY` dispatch)
- [../../../../tritium-sc/src/engine/comms/notifications.py](../../../../tritium-sc/src/engine/comms/notifications.py) — SC's independent, richer manager (the one actually driving the UI)
- [../models/notification_rules.py](../models/notification_rules.py) — `NotificationChannel`, `NotificationSeverity`
- [../models/alert.py](../models/alert.py) — the Alert model (related but separate)
