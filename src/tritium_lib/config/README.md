# tritium_lib.config — layered settings base

**Where you are:** `tritium-lib/src/tritium_lib/config/` — a
`pydantic-settings` base class and TOML/env helpers a Tritium service can
subclass instead of writing its own settings loader.

**Parent:** [`../README.md`](../README.md) ·
[`../../../CLAUDE.md`](../../../CLAUDE.md)

## What this package is

One file — `__init__.py` — providing `TritiumBaseSettings` (and its alias
`TritiumSettings`) plus standalone helpers.

**`TritiumBaseSettings`** carries the fields every service shares —
`app_name`, `debug`, `host`, `port`, `site_id`, `log_level`, and the MQTT
block (`mqtt_enabled`, `mqtt_host`, `mqtt_port`, `mqtt_site_id`,
`mqtt_username`, `mqtt_password`) — with real validators: `log_level` must be
one of the standard five, `site_id` is charset-restricted, and a
model-validator enforces "if `mqtt_enabled`, `mqtt_host` must be set"
(`__init__.py:203-235`). `to_dict(mask_secrets=True)` dumps config for
debugging with password/secret/token/key fields masked.

Standalone helpers: `load_toml()`, `get_addon_config(name)` (reads the
`[addons.<name>]` table), and `validate_settings()` (turns a Pydantic
`ValidationError` into a human-readable `ConfigError`).

## The source-priority chain

`settings_customise_sources` inserts a TOML source below dotenv
(`__init__.py:237-265`). Highest wins:

```
init kwargs  >  TRITIUM_* env  >  .env file  >  ~/.tritium/config.toml  >  field defaults
```

The TOML file is optional — `load_toml` returns `{}` when it's absent, so a
service with no config file still boots on defaults.

## Read this before you wire against it

**As of 2026-07-11 nothing in tritium-sc or tritium-edge subclasses
`TritiumBaseSettings`** (DATED grep of `TritiumBaseSettings` /
`TritiumSettings` / `from tritium_lib.config` across `tritium-sc/src`,
`tritium-sc/plugins`, `tritium-edge` — zero code hits). tritium-sc ships its
own loader — `src/app/config.py` defines `class Settings(BaseSettings)`
extending `pydantic_settings.BaseSettings` **directly**, not this base. So,
like [`../auth/`](../auth/) and [`../mqtt/`](../mqtt/), this is a **shared
primitive that is ready but not yet adopted** — exercised only by the lib
test suite. The docstring's "Both tritium-sc and tritium-edge extend these
base settings" describes the intent, not today's wiring.

> Note the name collision: `tritium_lib.config` (this settings package) is
> **not** `tritium_lib.models.config` (`SystemConfigModel` round-trip
> helpers) nor `tritium_lib.store.config_store` (`ConfigStore`). Three
> different "config" things; only this one is settings-loading.

## Ontology lens

Settings are the object *"service instance"*'s configured properties, resolved
from a defined precedence of sources — the same "explicit beats inherited
beats default" resolution shape the [`../auth/`](../auth/) ACL uses for
permissions, here applied to configuration values.

## Tests

`tests/test_config.py`, `tests/test_config_management.py` — source priority,
TOML loading, addon-section extraction, validators, and secret masking.

## Related

- The MQTT fields configured here feed: [`../mqtt/`](../mqtt/) topic grammar
- SC's own loader that could adopt this: `tritium-sc/src/app/config.py`
