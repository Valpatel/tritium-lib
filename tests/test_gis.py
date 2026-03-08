"""Tests for tritium_lib.models.gis."""

from tritium_lib.models.gis import (
    TileCoord,
    TileBounds,
    MapLayer,
    MapLayerType,
    MapRegion,
    TilePackage,
    OfflineRegion,
    lat_lon_to_tile,
    tile_to_lat_lon,
    tiles_in_bounds,
)


class TestTileCoord:
    def test_create(self):
        t = TileCoord(x=1234, y=5678, zoom=15)
        assert t.x == 1234
        assert t.y == 5678
        assert t.zoom == 15

    def test_url_path(self):
        t = TileCoord(x=10, y=20, zoom=5)
        assert t.url_path == "5/10/20"

    def test_quadkey_zoom1(self):
        # At zoom 1: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
        assert TileCoord(x=0, y=0, zoom=1).quadkey == "0"
        assert TileCoord(x=1, y=0, zoom=1).quadkey == "1"
        assert TileCoord(x=0, y=1, zoom=1).quadkey == "2"
        assert TileCoord(x=1, y=1, zoom=1).quadkey == "3"

    def test_quadkey_zoom0(self):
        assert TileCoord(x=0, y=0, zoom=0).quadkey == ""

    def test_json_roundtrip(self):
        t = TileCoord(x=100, y=200, zoom=12)
        t2 = TileCoord.model_validate_json(t.model_dump_json())
        assert t2 == t


class TestLatLonToTile:
    def test_sf_zoom10(self):
        # San Francisco (37.7749, -122.4194) at zoom 10
        t = lat_lon_to_tile(37.7749, -122.4194, 10)
        assert t.zoom == 10
        assert t.x == 163
        assert t.y == 395

    def test_origin_zoom0(self):
        t = lat_lon_to_tile(0.0, 0.0, 0)
        assert t.x == 0
        assert t.y == 0

    def test_negative_coords(self):
        t = lat_lon_to_tile(-33.8688, 151.2093, 10)  # Sydney
        assert t.zoom == 10
        assert 0 <= t.x < 1024
        assert 0 <= t.y < 1024

    def test_clamped_to_valid_range(self):
        t = lat_lon_to_tile(85.0, 179.99, 1)
        assert 0 <= t.x <= 1
        assert 0 <= t.y <= 1


class TestTileToLatLon:
    def test_origin(self):
        lat, lon = tile_to_lat_lon(0, 0, 0)
        assert abs(lat - 85.05) < 0.1  # NW corner of world
        assert abs(lon - (-180.0)) < 0.01

    def test_roundtrip(self):
        # Convert a known point to tile and back — should be close
        orig_lat, orig_lon = 37.7749, -122.4194
        t = lat_lon_to_tile(orig_lat, orig_lon, 15)
        lat, lon = tile_to_lat_lon(t.x, t.y, 15)
        # At zoom 15, tile is ~1km, so should be within ~0.01 degrees
        assert abs(lat - orig_lat) < 0.02
        assert abs(lon - orig_lon) < 0.02


class TestTilesInBounds:
    def test_single_tile(self):
        # At zoom 0, the whole world is one tile
        tiles = tiles_in_bounds((-85, -180, 85, 180), 0)
        assert len(tiles) == 1
        assert tiles[0].x == 0
        assert tiles[0].y == 0

    def test_small_region_zoom10(self):
        # Small region around SF at zoom 10
        bounds = (37.7, -122.5, 37.8, -122.4)
        tiles = tiles_in_bounds(bounds, 10)
        assert len(tiles) >= 1
        for t in tiles:
            assert t.zoom == 10

    def test_returns_correct_type(self):
        tiles = tiles_in_bounds((0, 0, 1, 1), 5)
        assert all(isinstance(t, TileCoord) for t in tiles)


class TestMapLayer:
    def test_create(self):
        layer = MapLayer(
            id="osm-standard",
            name="OpenStreetMap",
            url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attribution="OpenStreetMap contributors",
        )
        assert layer.id == "osm-standard"
        assert layer.layer_type == MapLayerType.RASTER
        assert layer.max_zoom == 19

    def test_vector_layer(self):
        layer = MapLayer(
            id="vector-1",
            name="Vector Tiles",
            layer_type=MapLayerType.VECTOR,
            format="pbf",
        )
        assert layer.layer_type == MapLayerType.VECTOR
        assert layer.format == "pbf"


class TestMapRegion:
    def test_create(self):
        r = MapRegion(
            id="sf-bay",
            name="SF Bay Area",
            south=37.2,
            west=-122.6,
            north=37.9,
            east=-121.8,
        )
        assert r.name == "SF Bay Area"

    def test_center(self):
        r = MapRegion(id="r1", name="Test", south=10.0, west=20.0, north=30.0, east=40.0)
        assert r.center_lat == 20.0
        assert r.center_lon == 30.0

    def test_bounds_tuple(self):
        r = MapRegion(id="r1", name="Test", south=1.0, west=2.0, north=3.0, east=4.0)
        assert r.bounds == (1.0, 2.0, 3.0, 4.0)

    def test_json_roundtrip(self):
        r = MapRegion(id="r1", name="Test", south=1, west=2, north=3, east=4)
        r2 = MapRegion.model_validate_json(r.model_dump_json())
        assert r2.id == r.id
        assert r2.bounds == r.bounds


class TestTilePackage:
    def test_create(self):
        region = MapRegion(id="r1", name="Test", south=1, west=2, north=3, east=4)
        pkg = TilePackage(
            id="pkg-1",
            name="SF Offline",
            region=region,
            layers=["osm-standard"],
            min_zoom=10,
            max_zoom=14,
            tile_count=5000,
            size_bytes=50_000_000,
            sha256="abc123",
        )
        assert pkg.tile_count == 5000
        assert pkg.format == "mbtiles"
        assert pkg.mbtiles_version == "1.3"

    def test_json_roundtrip(self):
        region = MapRegion(id="r1", name="Test", south=1, west=2, north=3, east=4)
        pkg = TilePackage(id="pkg-1", name="Test", region=region)
        pkg2 = TilePackage.model_validate_json(pkg.model_dump_json())
        assert pkg2.id == pkg.id
        assert pkg2.region.bounds == region.bounds


class TestTileBounds:
    def test_create(self):
        b = TileBounds(min_lat=37.0, min_lon=-122.5, max_lat=38.0, max_lon=-121.5)
        assert b.min_lat == 37.0
        assert b.max_lon == -121.5

    def test_center(self):
        b = TileBounds(min_lat=10.0, min_lon=20.0, max_lat=30.0, max_lon=40.0)
        assert b.center_lat == 20.0
        assert b.center_lon == 30.0

    def test_contains(self):
        b = TileBounds(min_lat=10.0, min_lon=20.0, max_lat=30.0, max_lon=40.0)
        assert b.contains(20.0, 30.0) is True
        assert b.contains(10.0, 20.0) is True  # edge
        assert b.contains(5.0, 30.0) is False
        assert b.contains(20.0, 50.0) is False

    def test_json_roundtrip(self):
        b = TileBounds(min_lat=1.0, min_lon=2.0, max_lat=3.0, max_lon=4.0)
        b2 = TileBounds.model_validate_json(b.model_dump_json())
        assert b2.min_lat == 1.0
        assert b2.max_lon == 4.0


class TestOfflineRegion:
    def test_create(self):
        bounds = TileBounds(min_lat=37.0, min_lon=-122.5, max_lat=38.0, max_lon=-121.5)
        region = OfflineRegion(
            id="sf-offline",
            name="SF Bay Area",
            bounds=bounds,
            zoom_levels=[10, 11, 12, 13, 14],
            tile_count=5000,
            size_bytes=50_000_000,
        )
        assert region.min_zoom == 10
        assert region.max_zoom == 14
        assert region.tile_count == 5000

    def test_empty_zoom_levels(self):
        bounds = TileBounds(min_lat=0, min_lon=0, max_lat=1, max_lon=1)
        region = OfflineRegion(id="r1", name="Test", bounds=bounds)
        assert region.min_zoom == 0
        assert region.max_zoom == 0

    def test_json_roundtrip(self):
        bounds = TileBounds(min_lat=1, min_lon=2, max_lat=3, max_lon=4)
        region = OfflineRegion(
            id="r1",
            name="Test",
            bounds=bounds,
            zoom_levels=[5, 6, 7],
        )
        region2 = OfflineRegion.model_validate_json(region.model_dump_json())
        assert region2.id == "r1"
        assert region2.bounds.min_lat == 1.0
        assert region2.max_zoom == 7
