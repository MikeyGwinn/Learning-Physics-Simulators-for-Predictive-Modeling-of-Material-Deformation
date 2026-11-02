"""The public, staged SEM image-to-mesh pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon

from .surface import MeshResult, MeshingOptions, mesh_polygons, polygons_from_mask


@dataclass(frozen=True)
class SegmentationOptions:
    """Threshold and cleanup controls; crop is (left, top, width, height).

    When threshold_percentile is set, the cutoff is that percentile of the
    cropped image intensities and the fixed threshold value is ignored.
    """

    threshold: float = 0.5
    threshold_percentile: float | None = None
    crop: tuple[int, int, int, int] | None = None
    min_solid_pixels: int = 16
    fill_holes_pixels: int = 8
    component: str = "all"


def _valid_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_segmentation(options: SegmentationOptions, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(options, SegmentationOptions):
        raise TypeError("segmentation must be SegmentationOptions")
    if isinstance(options.threshold, bool) or not isinstance(options.threshold, (int, float)) or not math.isfinite(options.threshold) or not 0 <= options.threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    percentile = options.threshold_percentile
    if percentile is not None and (isinstance(percentile, bool) or not isinstance(percentile, (int, float))
                                   or not math.isfinite(percentile) or not 0 < percentile < 100):
        raise ValueError("threshold_percentile must be between 0 and 100, exclusive")
    if options.component not in ("all", "largest"):
        raise ValueError("component must be 'all' or 'largest'")
    for name in ("min_solid_pixels", "fill_holes_pixels"):
        if not _valid_nonnegative_int(getattr(options, name)):
            raise ValueError(f"{name} must be a nonnegative integer")
    if options.crop is None:
        return 0, 0, width, height
    crop = options.crop
    if not isinstance(crop, tuple) or len(crop) != 4 or not all(_valid_nonnegative_int(v) for v in crop):
        raise ValueError("crop must be (left, top, width, height) in pixels")
    x, y, w, h = crop
    if w == 0 or h == 0 or x + w > width or y + h > height:
        raise ValueError("crop must have positive dimensions and fit inside the image")
    return crop


class SEMImage:
    """Load, segment, contour, and mesh one grayscale SEM image.

    Calling an earlier stage again invalidates every later stage. Geometry and
    mesh coordinates are in millimeters, with Y increasing upwards.
    """

    def __init__(self, path: str | Path, *, pixel_size_nm: float):
        if isinstance(pixel_size_nm, bool) or not isinstance(pixel_size_nm, (int, float)) or not math.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be finite and positive")
        self.path = Path(path)
        self.pixel_size_nm = float(pixel_size_nm)
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.geometry: tuple[Polygon, ...] | None = None
        self.mesh: MeshResult | None = None
        self.metadata: dict = {}
        self._crop: tuple[int, int, int, int] | None = None

    def load(self) -> np.ndarray:
        """Read an 8- or 16-bit, single-channel image without contrast changes."""
        self.image = None
        self.mask = None
        self.geometry = None
        self.mesh = None
        self._crop = None
        self.metadata = {}
        with Image.open(self.path) as source:
            if source.mode not in ("L", "I;16", "I;16L", "I;16B"):
                raise ValueError("input must be a single-channel 8- or 16-bit image")
            image = np.array(source)
        if image.ndim != 2 or image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError("input must be a single-channel 8- or 16-bit image")
        self.image = image
        self.metadata = {
            "source_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "image_width_px": image.shape[1],
            "image_height_px": image.shape[0],
            "pixel_size_nm": self.pixel_size_nm,
        }
        return image

    def segment(self, options: SegmentationOptions | None = None) -> np.ndarray:
        """Mark pixels at or above the normalized threshold as solid."""
        if self.image is None:
            self.load()
        self.mask = None
        self.geometry = None
        self.mesh = None
        self._crop = None
        self.metadata = {key: self.metadata[key] for key in ("source_sha256", "image_width_px", "image_height_px", "pixel_size_nm")}
        options = options if options is not None else SegmentationOptions()
        x, y, w, h = _validate_segmentation(options, self.image.shape[1], self.image.shape[0])
        roi = self.image[y:y+h, x:x+w]
        maximum = np.iinfo(roi.dtype).max
        if options.threshold_percentile is None:
            cutoff = float(options.threshold)
        else:
            if roi.min() == roi.max():
                raise ValueError("percentile threshold requires varying image intensities in the crop")
            cutoff = float(np.percentile(roi, options.threshold_percentile) / maximum)
        original = roi.astype(np.float64) / maximum >= cutoff
        solid = original.copy()
        if options.min_solid_pixels:
            labels, _ = ndimage.label(solid)
            sizes = np.bincount(labels.ravel())
            solid &= sizes[labels] >= options.min_solid_pixels
        if options.fill_holes_pixels:
            holes = ndimage.binary_fill_holes(solid) & ~solid
            labels, _ = ndimage.label(holes)
            sizes = np.bincount(labels.ravel())
            solid |= (labels > 0) & (sizes[labels] <= options.fill_holes_pixels)
        labels, count = ndimage.label(solid)
        if count == 0:
            raise ValueError("no solid material remains after segmentation")
        if options.component == "largest":
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            solid = labels == int(np.argmax(sizes))
        _, retained = ndimage.label(solid)
        self.mask = solid
        self.geometry = None
        self.mesh = None
        self._crop = (x, y, w, h)
        self.metadata = {
            **self.metadata,
            "crop": self._crop,
            "threshold": cutoff,
            "threshold_mode": "percentile" if options.threshold_percentile is not None else "fixed",
            "threshold_percentile": options.threshold_percentile,
            "threshold_raw": cutoff * maximum,
            "original_solid_pixels": int(original.sum()),
            "min_solid_pixels": options.min_solid_pixels,
            "fill_holes_pixels": options.fill_holes_pixels,
            "component": options.component,
            "solid_pixels": int(solid.sum()),
            "changed_pixels": int(np.count_nonzero(original != solid)),
            "components_retained": retained,
            "porosity": float(1.0 - solid.mean()),
            "mask_sha256": hashlib.sha256(solid.tobytes()).hexdigest(),
            "width_mm": w * self.pixel_size_nm * 1e-6,
            "height_mm": h * self.pixel_size_nm * 1e-6,
        }
        return solid

    def create_geometry(self, *, tolerance_px: float = 0.5) -> tuple[Polygon, ...]:
        """Extract topology-preserving solid polygons in ROI-local pixel units."""
        if self.mask is None:
            raise RuntimeError("call segment() before create_geometry()")
        self.geometry = None
        self.mesh = None
        self.metadata.pop("mesh", None)
        self.metadata.pop("tolerance_px", None)
        self.metadata.pop("contour_area_pixels", None)
        self.metadata.pop("contour_area_relative_error", None)
        if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, (int, float)) or not math.isfinite(tolerance_px) or not 0 <= tolerance_px <= 2:
            raise ValueError("tolerance_px must be between 0 and 2")
        geometry = tuple(polygons_from_mask(self.mask, float(tolerance_px)))
        self.geometry = geometry
        self.metadata["tolerance_px"] = float(tolerance_px)
        self.metadata["contour_area_pixels"] = float(sum(poly.area for poly in geometry))
        self.metadata["contour_area_relative_error"] = abs(self.metadata["contour_area_pixels"] - self.metadata["solid_pixels"]) / self.metadata["solid_pixels"]
        return geometry

    def generate_mesh(self, options: MeshingOptions | None = None) -> MeshResult:
        """Mesh current geometry; return arrays, metadata, and MSH export bytes."""
        if self.geometry is None:
            raise RuntimeError("call create_geometry() before generate_mesh()")
        options = options if options is not None else MeshingOptions()
        self.mesh = None
        result = mesh_polygons(self.geometry, self.mask, self.pixel_size_nm, options, self.metadata)
        self.mesh = result
        self.metadata["mesh"] = result.quality.copy()
        return result

    def run(
        self,
        *,
        segmentation: SegmentationOptions | None = None,
        tolerance_px: float = 0.5,
        meshing: MeshingOptions | None = None,
    ) -> MeshResult:
        """Run all stages from the source file and return the generated mesh."""
        self.load()
        self.segment(segmentation)
        self.create_geometry(tolerance_px=tolerance_px)
        return self.generate_mesh(meshing)
