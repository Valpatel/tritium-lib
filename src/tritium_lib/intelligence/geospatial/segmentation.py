# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SAM-based image segmentation engine.

Wraps Segment Anything Model (SAM2/SAM3) for segmenting satellite
and aerial imagery into distinct regions. Falls back gracefully
when torch is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from tritium_lib.intelligence.geospatial._deps import (
    HAS_NUMPY,
    HAS_PILLOW,
    HAS_TORCH,
    HAS_SAM,
)
from tritium_lib.intelligence.geospatial.models import SegmentationConfig

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """Segments images into distinct regions using SAM or fallback methods.

    When SAM/torch are available, uses the Segment Anything Model for
    high-quality automatic segmentation. When unavailable, falls back
    to OpenCV contour detection or grid-based segmentation.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self.config = config or SegmentationConfig()
        self._model: Any = None
        self._device: str = ""

    def segment_image(self, image_path: Path) -> list[dict]:
        """Segment an image into distinct regions.

        Returns list of dicts with keys:
            mask: binary mask as numpy array (H, W)
            area: pixel area of the segment
            bbox: (x, y, w, h) bounding box
            stability_score: confidence score (0-1)

        Falls back to simpler methods when SAM is unavailable.
        """
        if HAS_SAM and HAS_TORCH:
            return self._segment_with_sam(image_path)
        elif HAS_NUMPY and HAS_PILLOW:
            return self._segment_with_color_regions(image_path)
        else:
            logger.warning(
                "No segmentation backend available. Install torch+SAM "
                "for best results, or numpy+Pillow for color-based fallback."
            )
            return []

    def _segment_with_sam(self, image_path: Path) -> list[dict]:
        """Segment using SAM automatic mask generator."""
        import numpy as np
        from PIL import Image

        self._load_model()

        img = np.array(Image.open(image_path).convert("RGB"))

        # Tile large images to avoid OOM
        if img.shape[0] > 1024 or img.shape[1] > 1024:
            return self._segment_tiled(img)

        return self._run_sam(img)

    def _run_sam(self, image: Any) -> list[dict]:
        """Run SAM on a single image array."""
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            import torch

            if self._model is None:
                self._load_model()

            generator = SamAutomaticMaskGenerator(
                self._model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=int(self.config.min_area_m2),
            )
            masks = generator.generate(image)

            results = []
            for m in masks:
                results.append({
                    "mask": m["segmentation"],
                    "area": m["area"],
                    "bbox": m["bbox"],
                    "stability_score": m.get("stability_score", 0.0),
                })
            return results

        except Exception as e:
            logger.error("SAM segmentation failed: %s", e)
            return self._segment_with_color_regions_array(image)

    def _segment_tiled(self, image: Any) -> list[dict]:
        """Segment a large image by tiling with overlap."""
        import numpy as np

        tile_size = 1024
        overlap = 128
        h, w = image.shape[:2]
        all_masks: list[dict] = []

        for y0 in range(0, h, tile_size - overlap):
            for x0 in range(0, w, tile_size - overlap):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                tile = image[y0:y1, x0:x1]

                tile_masks = self._run_sam(tile)
                for m in tile_masks:
                    # Offset bbox to global coordinates
                    bx, by, bw, bh = m["bbox"]
                    m["bbox"] = (bx + x0, by + y0, bw, bh)
                    # Offset mask
                    full_mask = np.zeros((h, w), dtype=bool)
                    full_mask[y0:y1, x0:x1] = m["mask"]
                    m["mask"] = full_mask
                    all_masks.append(m)

        # Deduplicate overlapping masks via IoU
        return self._deduplicate_masks(all_masks)

    def _deduplicate_masks(self, masks: list[dict]) -> list[dict]:
        """Remove duplicate masks from tiled segmentation using IoU."""
        import numpy as np

        if len(masks) <= 1:
            return masks

        keep = [True] * len(masks)
        for i in range(len(masks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(masks)):
                if not keep[j]:
                    continue
                mi = masks[i]["mask"]
                mj = masks[j]["mask"]
                intersection = np.logical_and(mi, mj).sum()
                union = np.logical_or(mi, mj).sum()
                if union > 0 and intersection / union > 0.5:
                    # Keep the one with higher stability score
                    if masks[i].get("stability_score", 0) >= masks[j].get("stability_score", 0):
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return [m for m, k in zip(masks, keep) if k]

    def _segment_with_color_regions(self, image_path: Path) -> list[dict]:
        """Fallback: segment by color similarity using flood-fill regions."""
        import numpy as np
        from PIL import Image

        img = np.array(Image.open(image_path).convert("RGB"))
        return self._segment_with_color_regions_array(img)

    def _segment_with_color_regions_array(self, img: Any) -> list[dict]:
        """Color-based region segmentation using connected components.

        Strategy:
        1. Quantize to reduce color space (16 levels per channel)
        2. For each dominant quantized color, find all matching pixels
        3. Run connected component labeling to split into contiguous regions
        4. Each connected region becomes one segment

        Uses cv2.connectedComponents when available, falls back to
        scipy.ndimage.label, then to a pure-numpy flood approach.
        """
        import numpy as np

        h, w = img.shape[:2]

        # Tile-based approach: divide image into blocks, classify each block
        # as one region. This produces rectangular segments that align with
        # the terrain grid and avoid the quantization-merge problem.
        #
        # Block size 32px at zoom 16 ≈ 40m × 40m — balanced between
        # too fine (thousands of blocks) and too coarse (mixed pixels).
        block_size = 32
        masks: list[dict] = []

        for by in range(0, h, block_size):
            for bx in range(0, w, block_size):
                by1 = min(by + block_size, h)
                bx1 = min(bx + block_size, w)

                # Create block mask
                block_mask = np.zeros((h, w), dtype=bool)
                block_mask[by:by1, bx:bx1] = True
                area = (by1 - by) * (bx1 - bx)

                if area < 100:
                    continue

                masks.append({
                    "mask": block_mask,
                    "area": area,
                    "bbox": (bx, by, bx1 - bx, by1 - by),
                    "stability_score": 0.6,
                })

                # Memory safety
                if len(masks) >= 1000:
                    return masks

        # Return individual blocks — each one is a ~40m terrain cell.
        # The classifier handles each block independently, so water
        # blocks stay water even if adjacent to similar-colored vegetation.
        # Merging is done post-classification by the terrain_layer if needed.
        return masks

    def _merge_similar_blocks(
        self,
        img: Any,
        blocks: list[dict],
        block_size: int,
        color_threshold: float = 15.0,
    ) -> list[dict]:
        """Merge adjacent blocks with similar mean color into larger regions.

        This produces connected terrain regions from the tile grid while
        respecting color boundaries — rivers stay separate from parks.
        """
        import numpy as np

        if not blocks:
            return blocks

        h, w = img.shape[:2]

        # Compute mean color for each block
        block_colors = []
        for blk in blocks:
            pixels = img[blk["mask"]]
            if len(pixels) == 0:
                block_colors.append(np.array([0, 0, 0], dtype=np.float32))
            else:
                block_colors.append(pixels.mean(axis=0).astype(np.float32))

        # Build adjacency and merge similar neighbors
        merged = [False] * len(blocks)
        result: list[dict] = []

        for i in range(len(blocks)):
            if merged[i]:
                continue

            # Start a new merged region from this block
            region_mask = blocks[i]["mask"].copy()
            region_color = block_colors[i]
            merged[i] = True

            # BFS: find all adjacent similar blocks
            queue = [i]
            while queue:
                current = queue.pop(0)
                bx, by = blocks[current]["bbox"][:2]

                # Check 4-connected neighbors
                for ni in range(len(blocks)):
                    if merged[ni]:
                        continue
                    nx, ny = blocks[ni]["bbox"][:2]
                    # Adjacent? (within one block_size in either direction)
                    if (abs(nx - bx) <= block_size and abs(ny - by) == 0) or \
                       (abs(ny - by) <= block_size and abs(nx - bx) == 0):
                        # Similar color?
                        color_dist = np.linalg.norm(block_colors[ni] - region_color)
                        if color_dist < color_threshold:
                            region_mask |= blocks[ni]["mask"]
                            merged[ni] = True
                            queue.append(ni)

            area = int(region_mask.sum())
            if area < 100:
                continue

            ys_nz, xs_nz = np.nonzero(region_mask)
            result.append({
                "mask": region_mask,
                "area": area,
                "bbox": (
                    int(xs_nz.min()),
                    int(ys_nz.min()),
                    int(xs_nz.max() - xs_nz.min()),
                    int(ys_nz.max() - ys_nz.min()),
                ),
                "stability_score": 0.6,
            })

            if len(result) >= 500:
                break

        return result

    @staticmethod
    def _connected_components(binary_mask: Any) -> list[Any]:
        """Split a binary mask into connected regions.

        Uses cv2 when available, falls back to scipy, then to
        a pure-numpy approach.
        """
        import numpy as np

        from tritium_lib.intelligence.geospatial._deps import HAS_CV2

        if HAS_CV2:
            import cv2
            mask_u8 = binary_mask.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
            components = []
            for label_id in range(1, min(num_labels, 200)):  # skip background (0), cap at 200
                comp = labels == label_id
                if comp.sum() >= 100:
                    components.append(comp)
            return components

        # Scipy fallback
        try:
            from scipy.ndimage import label as ndimage_label
            labeled, num_features = ndimage_label(binary_mask)
            components = []
            for i in range(1, min(num_features + 1, 200)):
                comp = labeled == i
                if comp.sum() >= 100:
                    components.append(comp)
            return components
        except ImportError:
            pass

        # Pure numpy fallback — return the whole mask as one component
        # This is less accurate but avoids dependency issues
        return [binary_mask]

    def _load_model(self) -> None:
        """Load the SAM model."""
        if self._model is not None:
            return

        if not HAS_SAM or not HAS_TORCH:
            return

        import torch

        self._device = self._detect_device()

        try:
            from segment_anything import sam_model_registry

            model_type = {
                "sam2-tiny": "vit_t",
                "sam2-large": "vit_l",
                "sam-vit-h": "vit_h",
                "sam-vit-l": "vit_l",
                "sam-vit-b": "vit_b",
            }.get(self.config.model_name, "vit_b")

            # Look for model checkpoint in cache
            cache_dir = Path("data/cache/models/sam")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Try to find a checkpoint
            checkpoints = list(cache_dir.glob("*.pth"))
            if checkpoints:
                checkpoint = checkpoints[0]
                self._model = sam_model_registry[model_type](checkpoint=str(checkpoint))
                self._model.to(device=self._device)
                logger.info("Loaded SAM model %s on %s", model_type, self._device)
            else:
                logger.warning(
                    "No SAM checkpoint found in %s. "
                    "Download a model checkpoint to enable SAM segmentation.",
                    cache_dir,
                )

        except Exception as e:
            logger.error("Failed to load SAM model: %s", e)

    def _detect_device(self) -> str:
        """Detect the best available compute device."""
        if self.config.device != "auto":
            return self.config.device

        if not HAS_TORCH:
            return "cpu"

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
