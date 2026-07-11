# tritium_lib.planning — costmaps + open A* global planner

The **open-source baseline** for fleet route planning (UX Loop 3: dispatch a
robot, route it around obstacles). Pure stdlib, no third-party dependencies.
An advanced flow-field planner exists privately elsewhere; this package is the
clean, deterministic baseline any unit can run against with zero extra deps.

> **This lane consumes the parallel GIS lane's output.** The GIS pipeline
> produces WGS-84 elevation grids and GeoJSON layers; the costmap builder
> ingests them via `local_grid_from_gis` + `add_gis_features` (see
> [Consuming real GIS layers](#consuming-real-gis-layers)) without knowing
> where the data came from.

Coordinate frame everywhere: **local meters, +X = East, +Y = North** — the
same frame as `tritium_lib.geo`.

## LocalElevationGrid — planning's local DEM convention

A regular grid of elevation samples (meters) in **local meters**.

```python
LocalElevationGrid(origin_x, origin_y, resolution, data)
```

- `origin_x`, `origin_y` — the **south-west corner** node, local meters.
- `resolution` — node spacing in meters.
- `data` — row-major nested list; `data[row][col]` is the elevation at
  `x = origin_x + col*resolution`, `y = origin_y + row*resolution`.
- **`row 0` is the southernmost row; `col 0` is the westernmost column.**
- Samples are node values at cell corners → `elevation_at(x, y)` is bilinear
  (blends the four surrounding nodes), returning `None` out of bounds.
- `slope_at(x, y)` is the gradient magnitude (rise/run, unitless) via central
  differences; `0.0` if a sample falls out of bounds.
- `LocalElevationGrid.from_callable((min_x, min_y, max_x, max_y), resolution, fn)`
  builds a synthetic DEM from `fn(x, y) -> elevation` (for tests / demos).

> **Renamed from `ElevationGrid` (GIS-lane merge).** Planning's raster is
> **local meters with `row 0` = south**. The GIS lane owns the name
> `ElevationGrid` for its **WGS-84 wire model**
> (`tritium_lib.geo.gis.models.ElevationGrid`): a flat row-major `values`
> list over a `west/south/east/north` lat-lng bbox with the **opposite**
> convention — **`row 0` = north**. Two different rasters, two names.
> `tritium_lib.planning` does **not** export `ElevationGrid` at all; convert
> the wire grid with `local_grid_from_gis` (below).

## GeoJSON layer-input contract

Obstacle and road layers are plain GeoJSON `FeatureCollection` dicts.

- **Coordinates are LOCAL METERS `[x, y]` by default.** For WGS-84 layers,
  pass raw `[lng, lat]` coordinates and supply a `to_local(lng, lat) -> (x, y)`
  callable — `wgs84_to_local()` returns one built on the `tritium_lib.geo`
  reference singleton (raises if the singleton is uninitialised).
- Geometry types handled: `Polygon`, `MultiPolygon` (exterior rings only —
  holes are ignored for now), `LineString`, `MultiLineString`. Other types
  (Point, GeometryCollection, …) are skipped gracefully.
- **Roads**: a line feature's stamp half-width is `properties.width_m / 2`;
  when `width_m` is absent the builder's `weights.road_width_m` is used.
  Polygon road features mark all covered cells.
- **Obstacles**: polygon features become lethal cells. Use for buildings,
  water, and flood masks — the `kind` tag is recorded for introspection but
  the lethal semantics are identical.

## Costmap grid convention

`CostmapBuilder(bounds, resolution, weights).build() -> Costmap`

- `origin_x`, `origin_y` — **south-west corner** of the grid.
- `grid[row][col]` float costs; **`row 0` is the southernmost row**.
- `grid_to_world(col, row)` returns the **cell center**.
- `Costmap.LETHAL = inf` marks impassable cells; out-of-bounds reads as lethal.
- Per-cell cost is applied in a **fixed order regardless of layer call order**:
  1. `base = base_cost + slope_weight * slope`
  2. `* road_discount` if the cell is a road cell
  3. `* cost_zone_multiplier` if the cell falls in one or more soft-cost zones
     (the **MAX** multiplier over covering zones — zones never compound and
     never make a cell lethal)
  4. `slope > max_slope` → `LETHAL`
  5. obstacle / water cell → `LETHAL`
  6. optional inflation: non-lethal cells within `obstacle_inflation_m` of a
     lethal cell are raised to at least `inflation_cost`.
- `costmap_from_terrain_map(terrain_map)` adapts the existing sim `TerrainMap`
  (building/water → lethal, road → discount, else → base).

### Telemetry

`Costmap.to_telemetry(max_cells=40000)` returns a JSON-friendly dict:

```json
{"grid": [[...]], "cell_size": 5.0, "bounds": [min_x, min_y, max_x, max_y], "max_cost": 3.2}
```

`grid` is row-major (`row 0` = south) with **`LETHAL` encoded as `-1.0`**. If
`width * height > max_cells` the grid is downsampled by an integer stride
(`cell_size` scales up accordingly; within a block, lethal wins, else the block
takes the max cost).

## Planning a route

`plan_route(costmap, start, goal, *, smooth=True, max_expansions=None, snap_radius_m=None, clearance_m=0.0, strategy="auto")`

Deterministic 8-connected A* (octile heuristic × `min_traversable_cost`, so it
stays admissible with road discounts). No diagonal corner-cutting. A lethal
start/goal snaps to the nearest free cell within `snap_radius_m` (default
`3 * resolution`). Returns a `RouteResult(success, path, cost, expansions,
reason)`; on success `path[0]` is the exact start and `path[-1]` the exact goal.
Smoothing greedily shortcuts while preserving road preference (a shortcut can
never cross costlier cells than the subpath it replaces, and never crosses a
lethal cell). `clearance_m > 0` treats any cell within that distance of a lethal
cell as blocked (a wide unit keeps more wall standoff), relaxing to 0 rather
than failing a routable dispatch.

## Scaling to large AOs — hierarchical planning

Flat A* is optimal and cheap on simulator-scale maps but does not scale: on a
city-scale AO (600²+ cells, >9 km² at 5 m) with a continuous soft-cost field
(slope/weather), the octile heuristic is a weak lower bound and a single flat
solve blows the 200k-expansion cap and returns no path. `strategy` picks the
planner:

- `"flat"` — force the flat baseline above.
- `"hierarchical"` — force `plan_route_hierarchical`: coarsen the costmap by
  `DEFAULT_COARSE_FACTOR` (8×; a coarse cell is lethal only when *every* fine
  cell in its block is, so narrow corridors survive, and its soft cost is the
  block mean plus a mild congestion term), plan a tiny coarse A* first, then run
  the SAME flat A* restricted to a dilated corridor around the coarse route.
  Fine expansions are bounded by the corridor size, not the whole grid.
  Obstacle avoidance, clearance and road precedence are byte-for-byte the flat
  planner's (the fine solve runs the real costs + real obstacle-distance
  clearance inside the band). Completeness backstop: widen the corridor, then
  fall back to full flat A* — never `no_path` where flat would succeed.
- `"auto"` (default) — flat below `_AUTO_HIERARCHICAL_MIN_CELLS` (250k cells ≈
  500²; every simulator map, so `auto` is byte-for-byte flat there) and
  hierarchical at or above it. Existing callers (`engine.route_path`,
  `/api/route/plan`) upgrade automatically with no code change.

Measured (soft-cost field): flat **fails** at 600² (`max_expansions`, ~200k exp,
~2 s) where hierarchical **succeeds** (~61k exp, ~0.6 s) at an **identical route
cost** (corridor contains the optimum).

### Example

```python
from tritium_lib.planning import CostmapBuilder, CostmapWeights, LocalElevationGrid, plan_route

bounds = (0.0, 0.0, 500.0, 500.0)  # local meters

# Synthetic layers (a real GIS lane would supply these).
dem = LocalElevationGrid.from_callable(bounds, 10.0, lambda x, y: 0.02 * x)
buildings = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {},
     "geometry": {"type": "Polygon",
                  "coordinates": [[[200, 100], [260, 100], [260, 400], [200, 400], [200, 100]]]}}]}
roads = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"width_m": 8.0},
     "geometry": {"type": "LineString", "coordinates": [[0, 50], [500, 50]]}}]}

cm = (CostmapBuilder(bounds, resolution=5.0, weights=CostmapWeights())
      .add_dem(dem)
      .add_obstacles(buildings, kind="building")
      .add_roads(roads)
      .build())

route = plan_route(cm, start=(20.0, 20.0), goal=(480.0, 480.0))
if route.success:
    print(route.path)          # world-coordinate waypoints
    telemetry = cm.to_telemetry()  # for the tactical map
```

## Consuming real GIS layers

The parallel GIS lane fetches WGS-84 elevation and tagged vector features.
Planning ingests them through two entry points that translate the wire model
into the local-meter cost world.

### `local_grid_from_gis(gis_grid, *, to_local=None, resolution=None, nodata_fill=None)`

Adapts the GIS lane's WGS-84 elevation grid
(`tritium_lib.geo.gis.models.ElevationGrid`) into a `LocalElevationGrid`.
`gis_grid` is **duck-typed** — pass either that object or the plain dict from
`GET /api/gis/elevation/grid`, both carrying:

- `west, south, east, north` — lat/lng bbox (degrees).
- `ncols, nrows` — grid dimensions.
- `values` — **flat row-major** list of length `ncols*nrows`, **`row 0` =
  north**, values increasing eastward, `None` = NoData. Inclusive-edge
  sampling (col 0 on `west`, col `ncols-1` on `east`; row 0 on `north`, row
  `nrows-1` on `south`). Extra keys (`source`, `resolution_m`, `fixture`, …)
  are tolerated.

How it maps:

- `to_local(lng, lat) -> (x, y)` defaults to `wgs84_to_local()` (the geo
  singleton). The four bbox corners are projected; their min/max define the
  axis-aligned local extent.
- **Affine assumption:** the tritium geo projection is equirectangular over an
  AO-sized bbox, so local `(x, y)` → fractional source index is affine:
  `ix_frac = (x - x_west)/(x_east - x_west) * (ncols - 1)`,
  `iy_frac = (y_north - y)/(y_north - y_south) * (nrows - 1)` — note the **row
  FLIP** (source `row 0` = north, output `row 0` = south).
- Each output node is **bilinearly** interpolated over its four surrounding
  source cells. If any of the four is `None`, it falls back to the nearest
  non-`None` of the four (by fractional distance); if all four are `None` it
  uses `nodata_fill` (default: mean of all non-`None` values).
- `resolution` defaults to the mean source cell spacing in meters
  (floored at `1e-6`).
- Degenerate input (`ncols` or `nrows` < 2, `values` length mismatch, an
  all-`None` grid, or a bbox that projects to zero area) raises `ValueError`.

```python
from tritium_lib.planning import local_grid_from_gis
dem = local_grid_from_gis(gis_elevation_dict)   # -> LocalElevationGrid
cm_builder.add_dem(dem)
```

### `CostmapBuilder.add_gis_features(feature_collection, to_local=None) -> summary`

Routes a mixed GeoJSON `FeatureCollection` by `properties.source` and returns
a `{"roads", "flood", "zones", "ignored"}` count (for logging). Every feature
carries `source` and `kind`:

- **`"tiger"`** (roads) → road stamp. Width is `properties.width_m` if present,
  else the MTFCC table `MTFCC_WIDTHS_M` keyed by `properties.kind`, else
  `weights.road_width_m`. The width table is the **consumer-side** mapping —
  the GIS lane tags the MTFCC class but does not define a physical width:

  | MTFCC | meaning | width (m) |
  |-------|---------|-----------|
  | S1100 | primary road | 18.0 |
  | S1200 | secondary road | 12.0 |
  | S1400 | local street | 8.0 |
  | S1630 | ramp | 8.0 |
  | S1640 | service drive | 8.0 |
  | S1710 | walkway | 3.0 |
  | S1720 | stairway | 3.0 |
  | S1730 | alley | 4.0 |

- **`"fema"`** (flood) → **rasterize from `sfha`, never from style props.** A
  truthy `properties.sfha` (Special Flood Hazard Area) stamps a **lethal**
  obstacle tagged `"flood"`. `sfha` falsey (e.g. zone X minimal hazard) is
  traversable and **ignored**.
- **`"noaa"`** (weather alerts) → `properties.severity` of `"Severe"` or
  `"Extreme"` stamps a soft-cost zone with multiplier `2.0`; other severities
  are ignored. `expires` is ignored here — **expiry filtering is the fetch
  layer's job.**
- unknown / missing `source` → ignored gracefully.

### `CostmapBuilder.add_cost_zones(feature_collection, multiplier, to_local=None)`

Generic soft-cost polygons: covered cells have their final cost **multiplied**
by `multiplier` (`> 1` = avoid). Stacking keeps the **MAX** multiplier over
covering zones (never compounds). Applied in `build()` **after** the road
discount and **before** any lethal override, so a zone can raise cost but
never make a cell lethal. The full deterministic per-cell order is therefore:

> `base + slope` → `road discount` → `cost zones (max multiplier)` →
> `slope lethal` → `obstacle lethal` → `inflation`.

```python
b = CostmapBuilder(bounds, resolution=5.0)
b.add_dem(local_grid_from_gis(gis_elevation_dict))
summary = b.add_gis_features(gis_feature_collection)   # roads + flood + alerts
b.add_cost_zones(soft_avoid_fc, multiplier=1.5)        # optional extra zones
cm = b.build()
# summary -> {"roads": 12, "flood": 3, "zones": 1, "ignored": 4}
```
