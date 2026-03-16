# Tritium-Lib Stores

SQLite-backed persistent data stores. All stores inherit from BaseStore for consistent API.

## Key Files

- `base.py` — BaseStore: common init, table creation, connection management
- `ble.py` — BleStore: BLE sighting persistence
- `target.py` — TargetStore: tracked target state
- `reid.py` — ReIDStore: re-identification embeddings
- `audit.py` — AuditStore: API request audit log
- `config.py` — ConfigStore: namespaced key-value configuration
- `dossier.py` — DossierStore: persistent entity intelligence

## Related

- Models: `src/tritium_lib/models/`
- Graph store: `src/tritium_lib/graph/store.py`
