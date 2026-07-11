# Tritium-Lib Status — snapshot 2026-07-11

Dated snapshot of the shared library, re-measured against `dev`
(`f77dbf2`). The wave cadence that produced earlier snapshots ended;
counts below were measured on the date above, commands given so the
next sweep reproduces rather than re-litigates.

## Baselines (measured 2026-07-11)

| Metric | Count | How measured |
|--------|------:|--------------|
| Test files | 497 | `find tests -name 'test_*.py' \| wc -l` |
| Collectible tests | 17,111 | `pytest tests/ --collect-only -q` (29.7 s, 0 errors) |
| Python source files (`src/`) | 569 | `find src -name '*.py' \| wc -l` |
| Sim engine Python files | 188 | `find src/tritium_lib/sim_engine -name '*.py' \| wc -l` |
| Model files / Pydantic classes | 107 / 297 | `ls src/tritium_lib/models/*.py`; `grep -rE "^class .*BaseModel"` |
| Store package | 12 files | 8 SQLite stores + `BaseStore` + in-memory `EmbodimentRegistry` + `sweep_dir` — see [`store/README.md`](../src/tritium_lib/store/README.md) |
| Data JSON databases | 10 | `ls src/tritium_lib/data/*.json` (933 BLE company IDs in `ble_fingerprints.json`) |

## Key packages

| Package | Purpose |
|---------|---------|
| `models/` | 297 Pydantic classes across 107 files — canonical data contracts (incl. the 2026-07 robotics set: `quadruped.py`, `fire_control.py`, `hits.py`) |
| `sim_engine/` | Full tactical simulation, 188 files — `ai/`, `unit_types/`, `world/`, `game/`, `combat/`, `behavior/`, `physics/`, plus 46 demo scripts |
| `store/` | The persistence layer — [own README](../src/tritium_lib/store/README.md) |
| `tracking/` | Live read model: `TargetTracker` (now with health/max_health), `DossierManager` (extracted from SC 2026-07) |
| `intelligence/` | RL metrics, fusion metrics, position estimator |
| `classifier/` | Multi-signal BLE/WiFi device classifier |
| `sdk/` | Addon SDK — AddonBase, BaseRunner, GeoJSON layers |
| `graph/` | KuzuDB graph — **SHELFWARE, do not build against** (`graph/store.py:4-6`); not wired to any live API |

For the full package map see [`src/tritium_lib/README.md`](../src/tritium_lib/README.md)
(owed a refresh — it lists 9 of ~55 packages as of this snapshot).

## Standing notes (carried forward, still true)

- `ai/steering.py` vs `ai/steering_np.py`: intentional split — pure Python
  for composable forces, NumPy variant for 500+ agents at 10 Hz. Not
  redundant.
