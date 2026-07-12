# tritium_lib.data_exchange

**Export/import Tritium data as JSON, CSV, or GeoJSON.** A `TritiumExporter` /
`TritiumImporter` pair that reads targets, target history, dossiers, events,
and zones out of the stores into portable documents — and reads them back with
size/field safety limits.

**Where you are:** `tritium-lib/src/tritium_lib/data_exchange/`
**Parent:** [`../`](../) — the tritium-lib package map

> **Status: DEPRECATED (its own docstring, `__init__.py:7-16`).** No production
> consumers. The live export path is the SC `/api/dossier/*` endpoints + the
> heatmap GeoJSON endpoint, which do not use this package. The docstring's own
> TODO: *"delete this package and its dedicated tests once the integration
> tests can be rewritten without the Exporter/Importer stage."* Documented here
> for completeness, not for adoption — do not build against it.

## What it was for

Sharing operational data between Tritium instances (a pre-`federation`
approach), archiving for post-action review, exporting to analysis tools (CSV
for pandas/Excel, GeoJSON for QGIS), and incremental sync (`export_json(since=…)`
emits only records updated after a timestamp).

## Files

Single-module package (`__init__.py`, ~950 lines):

| Object | Where | What it does |
|--------|-------|--------------|
| `TritiumExporter` | `__init__.py:143` | Reads the three stores → documents. `export_json` (`:252`, full or `since=` incremental, with a `_MAGIC`/version envelope), `export_targets_csv` / `export_dossiers_csv` / `export_events_csv` (`:357`–`:399`), `export_geojson` (`:421`, positions as features), `get_export_stats` (`:576`). |
| `TritiumImporter` | `__init__.py:598` | The inverse, with hardening: `import_json` (`:625`), `import_csv` (`:877`); private `_import_*` per record type; `_sanitize_str` + `_MAX_FIELD_LENGTH` / `_MAX_JSON_DOC_SIZE` (50 MB) / `_MAX_CSV_ROWS` (1 M) caps against hostile input. |
| `ImportResult` | `__init__.py:104` | Outcome record; `total_imported` / `total_skipped`. |

Depends on `store.TargetStore` / `DossierStore` / `EventStore` (`__init__.py:52-54`).

## How it's consumed (verified 2026-07-11)

**No production consumer.** Dated grep for `from tritium_lib.data_exchange`
across sc/edge/addons: **0 hits.** Only three lib integration tests import it
(`test_data_exchange.py`, `test_end_to_end_pipeline.py`,
`test_full_integration.py`) — exactly as the docstring records. When those
tests are rewritten without the exporter/importer stage, the package is a
delete candidate (a code decision, not a docs one — flagged, not actioned).

## Related

- [../federation/](../federation/) — the successor for cross-site sharing (trust levels + share policies, transport-agnostic)
- [../recording/](../recording/) — the live archival path (JSONL battle recordings + retention)
- [../store/](../store/) — the `TargetStore`/`DossierStore`/`EventStore` this reads from and writes to
