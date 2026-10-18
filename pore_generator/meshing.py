"""Optional Gmsh meshing of material regions or a perforated specimen."""

import hashlib
import json
import math
from pathlib import Path

from .export import mesh_geometry, write_mesh_preview
from .generation import Config


def mesh_sample(sample_directory, mesh_format="material"):
    """Mesh saved pores as material regions or holes, preserving existing results."""
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise ValueError("Gmsh is unavailable. Install pore_generator/requirements-mesh.txt "
                         "in your active environment; on Linux its libraries may also require libGLU.") from exc
    if gmsh.__version__ != "4.15.2":
        raise ValueError(f"Expected Gmsh 4.15.2, found {gmsh.__version__}")
    sample_directory = Path(sample_directory)
    sample = json.loads((sample_directory / "sample.json").read_text(encoding="utf-8"))
    if sample["schema_version"] != 1 or sample["units"] != "mm":
        raise ValueError("Expected sample schema version 1 with millimeter units")
    Config(**sample["config"])
    geometry = mesh_geometry(sample, mesh_format)
    output = sample_directory / f"{mesh_format}_mesh"
    if any((output / name).exists() for name in ("specimen.msh", "mesh_summary.json", "mesh_preview.svg", "geometry.json")):
        raise ValueError(f"Mesh output already exists: {output}. Use a new sample directory.")
    geo_file = output / "specimen.geo"
    if geo_file.exists() and geo_file.read_text(encoding="utf-8") != geometry:
        raise ValueError("Saved geometry differs from sample.json; use a new sample directory")
    output.mkdir(parents=True, exist_ok=True)
    geo_file.write_text(geometry, encoding="utf-8")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.open(str(geo_file))
        gmsh.model.mesh.generate(2)
        inspect_mesh(gmsh, sample, mesh_format)
        gmsh.write(str(output / "specimen.msh"))
        # Check the serialized mesh as well as the in-memory geometry.
        gmsh.clear()
        gmsh.open(str(output / "specimen.msh"))
        summary, nodes, triangles = inspect_mesh(gmsh, sample, mesh_format)
        write_mesh_preview(nodes, triangles, sample["config"]["size_mm"], output / "mesh_preview.svg", mesh_format)
        (output / "mesh_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")
        if mesh_format == "perforated":
            size = sample["config"]["size_mm"]
            metadata = {
                "units": "mm", "representation": "perforated", "width": size, "height": size,
                "geometry_hash": hashlib.sha256(geo_file.read_bytes()).hexdigest(),
                "ready_for_fracture": True,
                "porosity": summary["hole_area_mm2"] / size ** 2,
                "explicit_porosity": summary["analytic_explicit_pore_area_mm2"] / size ** 2,
                "small_porosity": sum(math.pi * p["radius_mm"] ** 2 for p in sample["pores"]
                                      if p["kind"] == "small") / size ** 2,
                "total_sample_porosity": sample["statistics"]["achieved_porosity"],
                "small_pore_correction_applied": False,
                "mesh_size_mm": sample["config"]["mesh_size_um"] * 0.001,
            }
            (output / "geometry.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")
        return summary
    finally:
        gmsh.finalize()


def inspect_mesh(gmsh, sample, mesh_format):
    """Check regions, matrix connectivity, and shared interfaces or hole edges."""
    perforated = mesh_format == "perforated"
    size = sample["config"]["size_mm"]
    has_pores = any(p["kind"] == "explicit" for p in sample["pores"])
    has_pore_cells = has_pores and not perforated
    expected = {(1, tag) for tag in range(1, 5)} | {(2, 1)}
    if has_pores:
        expected.add((1, 5))
    if has_pore_cells:
        expected.add((2, 2))
    if set(map(tuple, gmsh.model.getPhysicalGroups())) != expected:
        raise ValueError("Mesh physical groups do not match the expected material/boundary tags")
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    nodes = {int(tag): tuple(coordinates[3*i:3*i+2]) for i, tag in enumerate(tags)}
    pore_edges = {}
    boundary_counts = {}
    pore_boundary_area = 0.0
    for tag in range(1, 6 if has_pores else 5):
        count = 0
        for curve in gmsh.model.getEntitiesForPhysicalGroup(1, tag):
            types, _, groups = gmsh.model.mesh.getElements(1, curve)
            if any(t != 1 for t in types):
                raise ValueError("Expected first-order line elements on mesh boundaries")
            for group in groups:
                for i in range(0, len(group), 2):
                    a, b = map(int, group[i:i+2])
                    edge = tuple(sorted((a, b)))
                    if tag == 5:
                        pore_edges[edge] = []
                        x1, y1 = nodes[a]
                        x2, y2 = nodes[b]
                        pore_boundary_area += (x1*y2 - x2*y1) / 2
                    else:
                        axis = 0 if tag in (1, 2) else 1
                        position = 0 if tag in (1, 3) else size
                        if any(abs(nodes[node][axis] - position) > size * 1e-10 for node in edge):
                            raise ValueError(f"Boundary tag {tag} is not on the expected square edge")
                    count += 1
        if not count:
            raise ValueError(f"Boundary/interface tag {tag} has no line elements")
        boundary_counts[tag] = count
    pore_boundary_area = abs(pore_boundary_area)
    parents = {}

    def root(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    triangles = []
    element_tags = []
    areas = {1: 0.0, 2: 0.0}
    counts = {1: 0, 2: 0}
    minimum_area = math.inf
    for region in (1, 2) if has_pore_cells else (1,):
        for surface in gmsh.model.getEntitiesForPhysicalGroup(2, region):
            types, tag_groups, groups = gmsh.model.mesh.getElements(2, surface)
            if any(t != 2 for t in types):
                raise ValueError("Expected only first-order triangles")
            for cell_tags, group in zip(tag_groups, groups):
                element_tags.extend(map(int, cell_tags))
                for i in range(0, len(group), 3):
                    a, b, c = map(int, group[i:i+3])
                    x1, y1 = nodes[a]
                    x2, y2 = nodes[b]
                    x3, y3 = nodes[c]
                    area = ((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)) / 2
                    if not math.isfinite(area) or area <= 0:
                        raise ValueError("Mesh contains nonpositive-area triangles")
                    areas[region] += area
                    counts[region] += 1
                    minimum_area = min(minimum_area, area)
                    triangles.append((a, b, c, region))
                    if region == 1:
                        anchor = root(a)
                        parents[root(b)] = anchor
                        parents[root(c)] = anchor
                    for edge in ((a, b), (b, c), (c, a)):
                        edge = tuple(sorted(edge))
                        if edge in pore_edges:
                            pore_edges[edge].append(region)
    components = len({root(node) for node in parents})
    if components != 1 or (has_pore_cells and not counts[2]):
        raise ValueError("Mesh must contain one connected matrix and the expected pore regions")
    expected_regions = [1] if perforated else [1, 2]
    if any(sorted(regions) != expected_regions for regions in pore_edges.values()):
        raise ValueError("Pore edges must border one matrix triangle and, for material meshes, one pore triangle")
    hole_area = pore_boundary_area if perforated else 0.0
    if not math.isclose(sum(areas.values()) + hole_area, size ** 2, rel_tol=1e-8):
        raise ValueError("Meshed regions and holes do not cover the sample square")
    if has_pore_cells and not math.isclose(areas[2], pore_boundary_area, rel_tol=1e-8):
        raise ValueError("Pore triangle area differs from the area enclosed by pore boundaries")
    quality = min(gmsh.model.mesh.getElementQualities(element_tags, "minSICN"))
    if not math.isfinite(quality) or quality <= 0:
        raise ValueError("Mesh contains nonpositive-quality elements")
    analytic_pore_area = sum(math.pi * p["radius_mm"] ** 2 for p in sample["pores"] if p["kind"] == "explicit")
    summary = {
        "representation": mesh_format,
        "gmsh_version": gmsh.__version__, "units": "mm", "msh_version": "4.1", "binary": False,
        "node_count": len(nodes), "triangle_count": len(triangles),
        "matrix_triangles": counts[1], "pore_triangles": counts[2],
        "matrix_area_mm2": areas[1], "pore_area_mm2": areas[2],
        "hole_area_mm2": hole_area,
        "analytic_explicit_pore_area_mm2": analytic_pore_area,
        "pore_area_relative_error": abs(pore_boundary_area - analytic_pore_area) / analytic_pore_area if analytic_pore_area else 0.0,
        "minimum_triangle_area_mm2": minimum_area, "minimum_quality": float(quality),
        "matrix_components": components, "interface_edges": 0 if perforated else len(pore_edges),
        "pore_boundary_edges": len(pore_edges),
        "boundary_line_counts": boundary_counts,
    }
    return summary, nodes, triangles
