"""Calibrated SEM segmentation, topology-preserving contours, and Gmsh meshes."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.measure import find_contours

DEFAULTS = {"image_id": "HPC22_0100", "mode": "provided", "threshold": 0.45,
            "dark_pores": True, "pixel_size_nm": 10.0, "crop_x": 1400, "crop_y": 900,
            "crop_width": 384, "crop_height": 384, "min_solid_pixels": 16,
            "fill_holes_pixels": 8, "component": "all", "simplify_px": 0.5,
            "boundary_size_nm": 30.0, "bulk_size_nm": 80.0, "refinement_band_nm": 100.0,
            "algorithm": 6, "smoothing": 5, "ell_nm": 240.0}


def validate_params(params):
    p = {**DEFAULTS, **params}
    if p["mode"] not in ("provided", "threshold") or p["component"] not in ("largest", "all"):
        raise ValueError("Unknown segmentation/component mode")
    if not isinstance(p["dark_pores"], bool) or not isinstance(p["image_id"], str):
        raise ValueError("Invalid image ID or threshold polarity")
    for key in ("crop_x", "crop_y", "crop_width", "crop_height", "min_solid_pixels", "fill_holes_pixels", "smoothing"):
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < (1 if key in ("crop_width", "crop_height") else 0):
            raise ValueError(f"{key} must be a valid nonnegative integer")
    for key in ("pixel_size_nm", "boundary_size_nm", "bulk_size_nm", "refinement_band_nm", "ell_nm"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(p["simplify_px"]) or not 0 <= p["simplify_px"] <= 2:
        raise ValueError("Contour tolerance must be 0..2 pixels")
    if not 0 <= p["threshold"] <= 1 or p["algorithm"] not in (5, 6, 8):
        raise ValueError("Invalid threshold or Gmsh algorithm")
    if p["boundary_size_nm"] > p["bulk_size_nm"]:
        raise ValueError("Boundary mesh size must be <= bulk size")
    if p["crop_width"]*p["crop_height"] > 16_000_000:
        raise ValueError("ROI too large; use at most 16 million pixels")
    return p


def dataset_record(root, image_id):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    matches = [r for r in manifest["images"] if r["id"] == image_id]
    if len(matches) != 1:
        raise ValueError("Select an image from the dataset manifest")
    record = matches[0]
    for key in ("image", "mask"):
        path = (root / record[key]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Dataset path is missing or outside its root")
    return manifest, record


def prepare_image(root, params):
    p = validate_params(params)
    t = time.perf_counter()
    manifest, r = dataset_record(root, p["image_id"])
    image = np.asarray(Image.open(Path(root)/r["image"]))
    supplied = np.asarray(Image.open(Path(root)/r["mask"]))
    if image.ndim != 2 or supplied.shape != image.shape:
        raise ValueError("Raw/mask shapes must match as 2D grayscale arrays")
    x, y, w, h = (p[k] for k in ("crop_x", "crop_y", "crop_width", "crop_height"))
    if x+w > image.shape[1] or y+h > image.shape[0]:
        raise ValueError(f"ROI exceeds image dimensions {image.shape[1]} × {image.shape[0]}")
    roi = image[y:y+h, x:x+w]
    # Fixed dtype normalization avoids crop-dependent segmentation thresholds.
    maximum = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    normalized = roi.astype(float)/maximum
    if p["mode"] == "provided":
        # Upstream porosity is mean(~A); imwrite(A) saves solid=true.
        pore = supplied[y:y+h, x:x+w] == 0
    else:
        pore = normalized < p["threshold"] if p["dark_pores"] else normalized > p["threshold"]
    solid = ~pore
    original = solid.copy()
    labels, count = ndimage.label(solid)
    if p["min_solid_pixels"]:
        sizes = np.bincount(labels.ravel())
        solid &= sizes[labels] >= p["min_solid_pixels"]
    if p["fill_holes_pixels"]:
        holes = ndimage.binary_fill_holes(solid) & ~solid
        lab, _ = ndimage.label(holes)
        sizes = np.bincount(lab.ravel())
        solid |= (lab > 0) & (sizes[lab] <= p["fill_holes_pixels"])
    labels, count = ndimage.label(solid)
    if not count:
        raise ValueError("No solid material remains in this ROI")
    if p["component"] == "largest":
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        solid = labels == int(np.argmax(sizes))
    labels, retained = ndimage.label(solid)
    left, right = set(labels[:, 0])- {0}, set(labels[:, -1])- {0}
    spans = bool(left & right)
    warnings = []
    if p["pixel_size_nm"] != manifest.get("pixel_size_nm", 10.0):
        warnings.append("Pixel calibration differs from the published 10 nm scale")
    changed = int(np.count_nonzero(original != solid))
    if changed:
        warnings.append(f"{changed} pixels changed by cleanup/component selection; inspect the overlay")
    if retained != 1 or not spans:
        warnings.append("Fracture export needs one solid component spanning both horizontal grips")
    if p["mode"] == "provided":
        warnings.append("Provided masks are author CNN predictions, not manual ground truth")
    metadata = {"doi": manifest["doi"], "image_id": r["id"], "mask_kind": manifest["mask_kind"],
                "source_image_sha256": hashlib.sha256((Path(root)/r["image"]).read_bytes()).hexdigest(),
                "source_mask_sha256": hashlib.sha256((Path(root)/r["mask"]).read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(solid.tobytes()).hexdigest(),
                "width": w*p["pixel_size_nm"]*1e-6, "height": h*p["pixel_size_nm"]*1e-6,
                "porosity": float(1-solid.mean()), "original_porosity": float(1-original.mean()),
                "components_before": count, "components_retained": retained,
                "changed_pixels": changed, "ready_for_fracture": retained == 1 and spans,
                "warnings": warnings, "parameters": p, "image_segmentation_seconds": time.perf_counter()-t}
    return normalized, solid, metadata


def polygons_from_mask(solid, tolerance=0.5):
    loops = []
    for contour in find_contours(np.pad(solid.astype(float), 1), 0.5, fully_connected="low"):
        points = np.c_[contour[:, 1]-0.5, solid.shape[0]+0.5-contour[:, 0]]
        polygon = Polygon(points)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Invalid raster boundary; inspect diagonal connections")
        loops.append(polygon)
    loops.sort(key=lambda x: x.area, reverse=True)
    # Containment parity distinguishes solid rings, pore rings and nested islands.
    parents = [next((j for j in range(i-1, -1, -1) if loops[j].contains(loop.representative_point())), None)
               for i, loop in enumerate(loops)]
    depths = []
    for parent in parents:
        depths.append(0 if parent is None else depths[parent]+1)
    result = []
    for i, loop in enumerate(loops):
        if depths[i] % 2:
            continue
        holes = [list(loops[j].exterior.coords) for j in range(len(loops)) if parents[j] == i and depths[j] % 2]
        polygon = Polygon(loop.exterior.coords, holes)
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        if not simplified.is_valid or simplified.is_empty or len(simplified.interiors) != len(polygon.interiors):
            raise ValueError("Simplification changed topology")
        result.append(simplified)
    if not result:
        raise ValueError("No valid solid contours")
    return result


def build_sem_mesh(root, params, output):
    import gmsh
    p = validate_params(params)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    image, solid, meta = prepare_image(root, p)
    Image.fromarray(np.uint8(np.clip(image, 0, 1)*255)).save(output/"image.png")
    Image.fromarray(np.uint8(solid)*255).save(output/"solid_mask.png")
    start = time.perf_counter()
    polygons = polygons_from_mask(solid, p["simplify_px"])
    scale = p["pixel_size_nm"]*1e-6
    rings = [{"outer": np.asarray(poly.exterior.coords).tolist(),
              "holes": [np.asarray(r.coords).tolist() for r in poly.interiors]} for poly in polygons]
    meta["contour_area_pixels"] = float(sum(poly.area for poly in polygons))
    meta["contour_area_relative_error"] = abs(meta["contour_area_pixels"]-int(solid.sum()))/int(solid.sum())
    meta["geometry_hash"] = hashlib.sha256(json.dumps({"rings": rings, "scale": scale},sort_keys=True).encode()).hexdigest()
    (output/"contours.json").write_text(json.dumps({"units": "pixels", "rings": rings}))
    meta["contour_seconds"] = time.perf_counter()-start
    t = time.perf_counter()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Geometry.Tolerance", scale*1e-5)
        gmsh.model.add("sem_microstructure")
        curves, surfaces = [], []
        for poly in polygons:
            loops = []
            for ring in [poly.exterior, *poly.interiors]:
                coords = list(ring.coords)[:-1]
                vertices = [gmsh.model.geo.addPoint(x*scale, y*scale, 0) for x, y in coords]
                edges = [gmsh.model.geo.addLine(vertices[i], vertices[(i+1)%len(vertices)]) for i in range(len(vertices))]
                curves.extend(edges)
                loops.append(gmsh.model.geo.addCurveLoop(edges))
            surfaces.append(gmsh.model.geo.addPlaneSurface(loops))
        if len(curves) > 100000:
            raise ValueError("Too many contours; reduce the ROI or adjust contour tolerance")
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        groups = {k: [] for k in range(1, 6)}
        for edge in curves:
            bb = gmsh.model.getBoundingBox(1, edge)
            tol = scale*1e-4
            key = 1 if abs(bb[0])+abs(bb[3]) < tol else 2 if abs(bb[0]-meta["width"])+abs(bb[3]-meta["width"]) < tol else 3 if abs(bb[1])+abs(bb[4]) < tol else 4 if abs(bb[1]-meta["height"])+abs(bb[4]-meta["height"]) < tol else 5
            groups[key].append(edge)
        for key, edges in groups.items():
            if edges:
                gmsh.model.addPhysicalGroup(1, edges, key)
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", curves)
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 20)
        field = gmsh.model.mesh.field.add("Threshold")
        for key, value in {"InField": distance, "SizeMin": p["boundary_size_nm"]*1e-6,
                           "SizeMax": p["bulk_size_nm"]*1e-6, "DistMin": 0,
                           "DistMax": p["refinement_band_nm"]*1e-6}.items():
            gmsh.model.mesh.field.setNumber(field, key, value)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        for key, value in {"Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
                           "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.Algorithm": p["algorithm"],
                           "Mesh.Smoothing": p["smoothing"], "Mesh.ElementOrder": 1}.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output/"specimen.msh"))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coords).reshape(-1, 3)[:, :2]
        index = {int(tag): i for i, tag in enumerate(tags)}
        _, cells = gmsh.model.mesh.getElementsByType(2)
        triangles = np.array([index[int(tag)] for tag in cells]).reshape(-1, 3)
        qualities = gmsh.model.mesh.getElementQualities(gmsh.model.mesh.getElementsByType(2)[0], "minSICN")
        np.savez_compressed(output/"mesh.npz", points=points, triangles=triangles)
        meta.update(nodes=len(points), cells=len(triangles), total_dofs=3*len(points),
                    minimum_quality=float(np.min(qualities)), mean_quality=float(np.mean(qualities)),
                    boundary_curves=len(curves), gmsh_version=gmsh.__version__)
        meta["ready_for_fracture"] &= bool(groups[1] and groups[2])
        if meta["minimum_quality"] <= 0:
            meta["ready_for_fracture"] = False
            meta["warnings"].append("Mesh contains nonpositive-quality elements")
    finally:
        gmsh.finalize()
    meta["mesh_seconds"] = time.perf_counter()-t
    (output/"geometry.json").write_text(json.dumps(meta, indent=2)+"\n")
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/skarberg"))
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_sem_mesh(args.dataset, json.loads(args.parameters.read_text()), args.output), indent=2))
