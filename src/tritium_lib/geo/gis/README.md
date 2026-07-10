# GIS Layers — real public government data

**Where you are:** `tritium-lib/src/tritium_lib/geo/gis/`

**Parent:** [../](../) | [../../../../CLAUDE.md](../../../../CLAUDE.md)

## What This Is

Turns four free U.S. government GIS sources into the two normalized shapes the
rest of Tritium consumes — a **GeoJSON `FeatureCollection` dict** for vector
layers and an **`ElevationGrid` raster** for terrain — plus a disk cache and
packaged demo-AO fixtures so every layer renders **fully offline**.

Stdlib only (`urllib.request`, `dataclasses`, `json`, `importlib.resources`).
**Zero new hard dependencies. No pydantic.**

Demo AO (used everywhere): center `37.7159, -121.8960` (Dublin, CA),
bbox `west=-121.912 south=37.704 east=-121.880 north=37.728`.

| Source | Fetcher | Output |
|--------|---------|--------|
| USGS 3DEP elevation | `UsgsElevationFetcher.fetch_grid(bbox, ncols, nrows)` | `ElevationGrid` |
| US Census TIGERweb roads (layer 8) | `TigerRoadsFetcher.fetch(bbox)` | FeatureCollection |
| FEMA National Flood Hazard Layer (layer 28) | `FemaFloodFetcher.fetch(bbox)` | FeatureCollection |
| NOAA / NWS active weather alerts | `NoaaAlertsFetcher.fetch(bbox)` | FeatureCollection |
| OpenStreetMap building footprints (Overpass) | `OverpassBuildingsFetcher.fetch(bbox)` | FeatureCollection |

Plus a derived layer computed from the DEM (no network of its own):

| Derived | Function | Output |
|---------|----------|--------|
| Iso-elevation contours | `contour_lines(grid, levels)` / `auto_levels(grid, n)` | FeatureCollection of `LineString`s |

Plus the constant `USGS_HILLSHADE_TILE_URL` (a `{z}/{y}/{x}` shaded-relief tile
template — no fetcher needed, hand it straight to the map client).

## ⚠️ Raster convention — `ElevationGrid` (READ THIS)

The costmap lane consumes `ElevationGrid` directly. Two things are **load-bearing
and must not be silently changed**:

1. **`values` is row-major and ROW 0 IS THE NORTH EDGE.** Index
   `iy * ncols + ix`. As `iy` grows, latitude **decreases** (you move south).
   `values[0]` is the north-west corner; `values[-1]` is the south-east corner.
   `None` marks a NoData cell.
2. **Cells sit on the inclusive edge grid.** Column `0` is exactly on `west`,
   column `ncols-1` exactly on `east`; row `0` exactly on `north`, row `nrows-1`
   exactly on `south`:

   ```
   cell_lon(ix) = west  + (east  - west ) * ix / (ncols - 1)
   cell_lat(iy) = north - (north - south) * iy / (nrows - 1)   # row 0 = north
   ```

Helpers: `value_at(ix, iy)`, `min_max()`, `to_dict()` / `from_dict()` (exact
JSON round-trip; `from_dict` tolerates an extra `"fixture"` key), and
`slope_deg()` — per-cell max-gradient terrain slope in degrees via central
differences. Slope metres come from the *sampled cell spacing* (bbox / (n-1)),
longitude scaled by `cos(lat)` using `METERS_PER_DEG_LAT` from `tritium_lib.geo`;
a cell whose stencil touches NoData yields `None`.

## ⚠️ Vector convention — feature `properties` (READ THIS)

Every normalized feature is WGS-84 lon/lat GeoJSON, and **every feature's
`properties` carries `source` and `kind`** plus a small set of per-layer fields:

| Layer | `source` | `kind` | Extra properties |
|-------|----------|--------|------------------|
| roads | `"tiger"` | MTFCC code, e.g. `"S1400"` | `name` (street name, may be `""`) |
| flood | `"fema"` | FLD_ZONE, e.g. `"A"`,`"AE"`,`"X"` | `subtype` (ZONE_SUBTY, may be `""`), `sfha` (bool — Special Flood Hazard Area) |
| alerts | `"noaa"` | NWS event, e.g. `"Heat Advisory"` | `severity`, `headline`, `expires` |
| buildings | `"osm"` | `building` tag, e.g. `"house"`,`"retail"` (`"yes"` when untyped) | `name` (may be `""`), `height_m` (float), `levels` (int) |
| contours | `"usgs"` | `"contour"` | `elevation_m` (level, 1 dp), `level_index` (int) |

Style props (`fill_color`, `stroke_width`, …) are added by the **SC provider**,
never in this lib. NOAA features **without geometry are dropped** (the layer must
be renderable); an empty FeatureCollection is a **valid** result (a genuine
"no active alerts", distinct from a fetch failure).

## Elevation contours — `contours.py` (marching squares)

`contour_lines(grid, levels)` traces iso-elevation lines over an `ElevationGrid`
using standard 16-case **marching squares** on the cell-corner lattice (the grid
samples *are* the lattice nodes). Crossings are linearly interpolated along cell
edges; the two ambiguous saddle cases (5 and 10) are resolved by the sign of the
cell-centre average; and **any cell that touches a NoData corner is skipped**, so
a contour never crosses a hole. Segments are joined into polylines wherever their
endpoints coincide (tolerance `1e-9` deg — crossings shared by adjacent cells are
computed identically, so they match exactly). Row 0 stays north — coordinates
come straight from `grid.cell_lon` / `grid.cell_lat`.

Output is a `FeatureCollection` of `LineString` features; each feature's
`properties` are `{"source": "usgs", "kind": "contour", "elevation_m": <level 1
dp>, "level_index": <i>}`.

`auto_levels(grid, n=8)` returns `n` levels at fractions `i/(n+1)` of the value
range — every level is **strictly inside** `(min, max)` (never on the flat outer
boundary). Returns `[]` when the grid has fewer than two distinct present values.

```python
from tritium_lib.geo.gis import UsgsElevationFetcher, auto_levels, contour_lines
grid = UsgsElevationFetcher().fetch_grid(bbox, ncols=32, nrows=32)
fc = contour_lines(grid, auto_levels(grid, 10))   # GeoJSON LineStrings
```

## Building footprints — `OverpassBuildingsFetcher`

`OverpassBuildingsFetcher.fetch(bbox)` returns OSM building footprints as closed
`Polygon` features via the Overpass API (`out geom`). It uses the same
degradation chain as the other vector fetchers, but **POSTs** an Overpass QL body
(note Overpass bbox order is `south, west, north, east`, not `w,s,e,n`). Ways with
fewer than three geometry points are dropped; `height_m` comes from the `height`
tag (strip `"m"`), else `building:levels * 3 + 1`, else `8.0`; `levels` is
`max(1, int(height_m / 3))`.

## Fixture bbox-clipping — `filter_features_bbox`

Packaged fixtures cover the whole demo AO, so a query for a *distant* window
would otherwise get the entire AO back. `filter_features_bbox(fc, bbox)` keeps
only features whose geometry bounding box intersects `bbox` (computed from all
coordinates, any geometry type; features with no coordinates are dropped; the
top-level `"fixture": true` marker is preserved). It is applied **only on the
packaged-fixture branch** of `fetch()` — live results are already bbox-scoped and
cache keys are bbox-rounded, so those two branches are left untouched. Net effect:
an offline `fetch()` for a far-away bbox now returns an empty (but valid,
`fixture`-marked) collection instead of the whole AO.

## Cache retention — `GISCache.sweep()`

`GISCache.sweep(*, retention_days, max_total_bytes, now=None)` bounds the on-disk
cache and returns `{"removed": [relpaths], "freed_bytes": int, "remaining_bytes":
int}`. It mirrors `tritium_lib.recording.retention.sweep_recordings` (on `dev`,
commit `ffbfd6a`) — two oldest-first passes (age, then size), caller-supplied
bounds, safe unlink — generalized for the GIS cache: it walks `cache_dir`
**recursively** (covering the `tiles/` XYZ tree), only ever deletes files whose
suffix is in `{.json, .bin}` **and** which resolve inside `cache_dir` (an
out-of-tree symlink is never followed for deletion), and prunes now-empty
subdirectories best-effort. A missing dir is a safe no-op; it never raises. A
later unification with `sweep_recordings` should be mechanical.

## Degradation chain

Every fetcher degrades so the demo AO always works:

```
live HTTP  --success-->  parse, cache.put, return
           --failure-->  cache.get (no age limit)
                            --miss-->  packaged demo fixture (fixtures/)
                                          --miss-->  empty result
                                                     ( {FeatureCollection:[]} or
                                                       an all-NoData grid )
```

`GISCache(cache_dir=None)` is a filename-keyed JSON cache. Default dir: env
`TRITIUM_GIS_CACHE` else `data/gis_cache` (relative cwd). `key(source, bbox,
**params)` rounds the bbox to 4 dp so near-identical viewports share an entry.
All cache IO is best-effort — a corrupt or unreadable entry never raises, it
just misses. User-Agent on all requests:
`Tritium/1.0 (+https://github.com/Valpatel/tritium)`.

## Fixtures (`fixtures/`, package data, checked in)

Generated by running the `parse_*` functions over real captured government
payloads, then trimmed + coordinate-rounded. Loaded via `importlib.resources`.
Each carries a top-level `"fixture": true` marker; total < 150 KB.

| File | Contents |
|------|----------|
| `usgs_dem_ao.json` | full 16×16 `ElevationGrid.to_dict()`, `source="usgs-fixture"` (elev 101–191 m) |
| `tiger_roads_ao.json` | 60 road features (coords 6 dp; keeps S1630/S1730 for style variety) |
| `fema_flood_ao.json` | 17 flood-zone polygons (coords 6 dp; A/AE/X, 11 SFHA) |
| `noaa_alerts_ao.json` | 4 alert polygons (coords 4 dp) |
| `osm_buildings_ao.json` | 120 building `Polygon`s (coords 6 dp; 71 named, 17 `building` kinds, 5–40 vertices; ~45 KB) |

> **NOAA fixture note.** The captured `alerts/active?area=CA` response is
> entirely zone/UGC-based alerts with `geometry=null` (the common NWS case), so
> `parse_alerts` over it yields an *empty* collection — the documented
> geometry-drop / valid-empty behavior, covered by tests. To give the demo AO a
> renderable NOAA layer, the fixture attaches synthetic AO-local polygons to 4
> **real** alerts (real `event`/`severity`/`headline`/`expires`), passed through
> `parse_alerts` exactly like a live geometry-bearing response.

## Public API

```python
from tritium_lib.geo.gis import (
    GeoBBox, ElevationGrid, GISCache,
    UsgsElevationFetcher, TigerRoadsFetcher, FemaFloodFetcher, NoaaAlertsFetcher,
    OverpassBuildingsFetcher, auto_levels, contour_lines, filter_features_bbox,
    USGS_HILLSHADE_TILE_URL,
)

bbox = GeoBBox.from_string("-121.912,37.704,-121.880,37.728")
grid = UsgsElevationFetcher().fetch_grid(bbox, ncols=16, nrows=16)  # ElevationGrid
roads = TigerRoadsFetcher().fetch(bbox)                             # FeatureCollection
buildings = OverpassBuildingsFetcher().fetch(bbox)                 # FeatureCollection
contours = contour_lines(grid, auto_levels(grid, 8))              # LineString features
```

`GeoBBox`: `from_string("w,s,e,n")` (raises `ValueError` on bad input),
`to_string()`, `center()` → `(lon, lat)`, `contains(lon, lat)`.

## Related

- [../__init__.py](../__init__.py) — `METERS_PER_DEG_LAT` and coordinate transforms
- [../../models/gis.py](../../models/gis.py) — `TileCoord` / `MapLayer` map models
- SC providers `plugins/gis_layers/` — add style props + expose HTTP routes,
  including `GET /api/gis/elevation/grid` (the costmap lane's endpoint)
