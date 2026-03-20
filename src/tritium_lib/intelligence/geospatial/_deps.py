# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Optional dependency guards for geospatial segmentation.

Heavy deps (torch, rasterio, shapely, etc.) are optional. Classes check
these flags at construction time, not import time, so:

    from tritium_lib.intelligence.geospatial import TerrainLayer

always works. An error fires only when you call a method that needs
the missing dependency.
"""

HAS_NUMPY = False
HAS_PILLOW = False
HAS_RASTERIO = False
HAS_SHAPELY = False
HAS_GEOPANDAS = False
HAS_TORCH = False
HAS_SAM = False
HAS_CV2 = False

try:
    import numpy as np  # noqa: F401
    HAS_NUMPY = True
except ImportError:
    pass

try:
    from PIL import Image  # noqa: F401
    HAS_PILLOW = True
except ImportError:
    pass

try:
    import rasterio  # noqa: F401
    HAS_RASTERIO = True
except ImportError:
    pass

try:
    import shapely  # noqa: F401
    HAS_SHAPELY = True
except ImportError:
    pass

try:
    import geopandas  # noqa: F401
    HAS_GEOPANDAS = True
except ImportError:
    pass

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    pass

try:
    from segment_anything import SamPredictor  # noqa: F401
    HAS_SAM = True
except ImportError:
    try:
        from sam2.build_sam import build_sam2  # noqa: F401
        HAS_SAM = True
    except ImportError:
        pass

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    pass


def require(flag: bool, name: str, install_extra: str = "geospatial") -> None:
    """Raise ImportError if a required dependency is missing."""
    if not flag:
        raise ImportError(
            f"{name} is required for this operation. "
            f"Install it with: pip install 'tritium-lib[{install_extra}]'"
        )
