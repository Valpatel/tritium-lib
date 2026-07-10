# tritium_lib.planning — costmaps + open A* global planner

The **open-source baseline** for fleet route planning (UX Loop 3: dispatch a
robot, route it around obstacles). Pure stdlib, no third-party dependencies.
An advanced flow-field planner exists privately elsewhere; this package is the
clean, deterministic baseline any unit can run against with zero extra deps.

> **This file is the contract for the parallel GIS lane.** The GIS pipeline
> produces `ElevationGrid` DEMs and GeoJSON layers in exactly the shapes below;
> the costmap builder consumes them without knowing where the data came from.

Coordinate frame everywhere: **local meters, +X = East, +Y = North** — the
same frame as `tritium_lib.geo`.

## ElevationGrid — the DEM convention

A regular grid of elevation samples (meters). A GIS lane that downloads USGS
3DEP tiles rasterizes them into this structure.

```python
ElevationGrid(origin_x, origin_y, resolution, data)
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
- `ElevationGrid.from_callable((min_x, min_y, max_x, max_y), resolution, fn)`
  builds a synthetic DEM from `fn(x, y) -> elevation` (for tests / demos).

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
  3. `slope > max_slope` → `LETHAL`
  4. obstacle / water cell → `LETHAL`
  5. optional inflation: non-lethal cells within `obstacle_inflation_m` of a
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

`plan_route(costmap, start, goal, *, smooth=True, max_expansions=None, snap_radius_m=None)`

Deterministic 8-connected A* (octile heuristic × `min_traversable_cost`, so it
stays admissible with road discounts). No diagonal corner-cutting. A lethal
start/goal snaps to the nearest free cell within `snap_radius_m` (default
`3 * resolution`). Returns a `RouteResult(success, path, cost, expansions,
reason)`; on success `path[0]` is the exact start and `path[-1]` the exact goal.
Smoothing greedily shortcuts while preserving road preference (a shortcut can
never cross costlier cells than the subpath it replaces, and never crosses a
lethal cell).

### Example

```python
from tritium_lib.planning import CostmapBuilder, CostmapWeights, ElevationGrid, plan_route

bounds = (0.0, 0.0, 500.0, 500.0)  # local meters

# Synthetic layers (a real GIS lane would supply these).
dem = ElevationGrid.from_callable(bounds, 10.0, lambda x, y: 0.02 * x)
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
