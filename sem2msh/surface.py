"""Topology-preserving contours and Gmsh generation for the SEMImage API."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import threading

import numpy as np
from shapely.geometry import Polygon
from skimage.measure import find_contours


_GMSH_LOCK = threading.Lock()
_ALGORITHMS = {"delaunay": 5, "frontal delaunay": 6, "frontal delaunay quads": 8}


@dataclass(frozen=True)
class MeshingOptions:
    """Gmsh sizing controls in nanometers."""

    boundary_size_nm: float = 30.0
    interior_size_nm: float = 80.0
    refinement_distance_nm: float = 100.0
    algorithm: str = "frontal delaunay"
    smoothing_iterations: int = 5


@dataclass
class MeshResult:
    """In-memory mesh and its original Gmsh MSH export."""

    mask: np.ndarray
    contours: tuple[Polygon, ...]
    points: np.ndarray
    triangles: np.ndarray
    parameters: dict
    quality: dict
    metadata: dict
    _msh_bytes: bytes

    def save_msh(self, path: str | Path) -> Path:
        """Write the generated first-order triangle mesh to a .msh file."""
        destination = Path(path)
        destination.write_bytes(self._msh_bytes)
        return destination


def polygons_from_mask(solid: np.ndarray, tolerance: float) -> list[Polygon]:
    """Convert raster solid to polygons, preserving holes and nested islands."""
    loops = []
    for contour in find_contours(np.pad(solid.astype(float), 1), 0.5, fully_connected="low"):
        points = np.c_[contour[:, 1] - 0.5, solid.shape[0] + 0.5 - contour[:, 0]]
        polygon = Polygon(points)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("invalid raster boundary; inspect diagonal connections")
        loops.append(polygon)
    loops.sort(key=lambda polygon: polygon.area, reverse=True)
    parents = [next((j for j in range(i-1, -1, -1) if loops[j].contains(loop.representative_point())), None)
               for i, loop in enumerate(loops)]
    depths = []
    for parent in parents:
        depths.append(0 if parent is None else depths[parent] + 1)
    result = []
    for i, loop in enumerate(loops):
        if depths[i] % 2:
            continue
        holes = [list(loops[j].exterior.coords) for j in range(len(loops)) if parents[j] == i and depths[j] % 2]
        polygon = Polygon(loop.exterior.coords, holes)
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        if not simplified.is_valid or simplified.is_empty or len(simplified.interiors) != len(polygon.interiors):
            raise ValueError("contour simplification changed topology")
        result.append(simplified)
    if not result:
        raise ValueError("no valid solid contours")
    return result


def _validate_meshing(options: MeshingOptions) -> int:
    if not isinstance(options, MeshingOptions):
        raise TypeError("meshing must be MeshingOptions")
    for name in ("boundary_size_nm", "interior_size_nm", "refinement_distance_nm"):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if options.boundary_size_nm > options.interior_size_nm:
        raise ValueError("boundary_size_nm must be <= interior_size_nm")
    if isinstance(options.smoothing_iterations, bool) or not isinstance(options.smoothing_iterations, int) or options.smoothing_iterations < 0:
        raise ValueError("smoothing_iterations must be a nonnegative integer")
    if not isinstance(options.algorithm, str):
        raise ValueError("algorithm must be a supported Gmsh algorithm name")
    name = options.algorithm.strip().lower().replace("_", " ")
    if name not in _ALGORITHMS:
        raise ValueError(f"algorithm must be one of: {', '.join(_ALGORITHMS)}")
    return _ALGORITHMS[name]


def mesh_polygons(
    polygons: tuple[Polygon, ...],
    mask: np.ndarray,
    pixel_size_nm: float,
    options: MeshingOptions,
    metadata: dict,
) -> MeshResult:
    """Generate a 2D triangle mesh from ROI-local solid polygons."""
    import gmsh

    algorithm = _validate_meshing(options)
    scale = pixel_size_nm * 1e-6
    if sum(len(poly.exterior.coords) + sum(len(r.coords) for r in poly.interiors) for poly in polygons) > 100_000:
        raise ValueError("too many contour vertices; reduce crop or increase tolerance")
    with _GMSH_LOCK:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.option.setNumber("Geometry.Tolerance", scale * 1e-5)
            gmsh.model.add("sem_surface")
            curves, surfaces = [], []
            for poly in polygons:
                loops = []
                for ring in [poly.exterior, *poly.interiors]:
                    coords = list(ring.coords)[:-1]
                    vertices = [gmsh.model.geo.addPoint(float(x*scale), float(y*scale), 0) for x, y in coords]
                    edges = [gmsh.model.geo.addLine(vertices[i], vertices[(i+1) % len(vertices)]) for i in range(len(vertices))]
                    curves.extend(edges)
                    loops.append(gmsh.model.geo.addCurveLoop(edges))
                surfaces.append(gmsh.model.geo.addPlaneSurface(loops))
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(2, surfaces, 1)
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "CurvesList", curves)
            gmsh.model.mesh.field.setNumber(distance, "Sampling", 20)
            field = gmsh.model.mesh.field.add("Threshold")
            for name, value in {
                "InField": distance,
                "SizeMin": options.boundary_size_nm * 1e-6,
                "SizeMax": options.interior_size_nm * 1e-6,
                "DistMin": 0,
                "DistMax": options.refinement_distance_nm * 1e-6,
            }.items():
                gmsh.model.mesh.field.setNumber(field, name, value)
            gmsh.model.mesh.field.setAsBackgroundMesh(field)
            for name, value in {
                "Mesh.MeshSizeFromPoints": 0,
                "Mesh.MeshSizeFromCurvature": 0,
                "Mesh.MeshSizeExtendFromBoundary": 0,
                "Mesh.Algorithm": algorithm,
                "Mesh.Smoothing": options.smoothing_iterations,
                "Mesh.ElementOrder": 1,
            }.items():
                gmsh.option.setNumber(name, value)
            gmsh.model.mesh.generate(2)
            tags, coords, _ = gmsh.model.mesh.getNodes()
            points = np.asarray(coords).reshape(-1, 3)[:, :2].copy()
            node_index = {int(tag): i for i, tag in enumerate(tags)}
            element_tags, cells = gmsh.model.mesh.getElementsByType(2)
            triangles = np.array([node_index[int(tag)] for tag in cells], dtype=np.int64).reshape(-1, 3)
            if len(triangles) == 0:
                raise ValueError("Gmsh generated no triangles")
            qualities = gmsh.model.mesh.getElementQualities(element_tags, "minSICN")
            minimum_quality = float(np.min(qualities))
            if not math.isfinite(minimum_quality) or minimum_quality <= 0:
                raise ValueError("Gmsh generated nonpositive-quality triangles")
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                gmsh.write(str(temporary_path))
                msh_bytes = temporary_path.read_bytes()
            finally:
                temporary_path.unlink(missing_ok=True)
        finally:
            gmsh.finalize()
    parameters = {
        "pixel_size_nm": pixel_size_nm,
        "boundary_size_nm": options.boundary_size_nm,
        "interior_size_nm": options.interior_size_nm,
        "refinement_distance_nm": options.refinement_distance_nm,
        "algorithm": options.algorithm,
        "smoothing_iterations": options.smoothing_iterations,
    }
    quality = {
        "nodes": len(points), "triangles": len(triangles),
        "minimum_quality": minimum_quality,
        "mean_quality": float(np.mean(qualities)),
        "boundary_curves": len(curves), "gmsh_version": gmsh.__version__,
    }
    return MeshResult(mask.copy(), polygons, points, triangles, parameters, quality,
                      {**metadata, "mesh": quality.copy()}, msh_bytes)
