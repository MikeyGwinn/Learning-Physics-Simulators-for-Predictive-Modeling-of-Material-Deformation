"""Write portable pore descriptions and a dependency-free SVG preview."""

import csv
import json
from pathlib import Path


def write_sample(sample, output):
    """Write a successful sample to a new directory; never overwrite existing work."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "sample.json").write_text(json.dumps(sample, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "pores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "x_mm", "y_mm", "radius_mm", "kind"])
        writer.writeheader()
        writer.writerows(sample["pores"])
    (output / "preview.svg").write_text(preview_svg(sample), encoding="utf-8")
    mesh_dir = output / "material_mesh"
    mesh_dir.mkdir()
    (mesh_dir / "specimen.geo").write_text(material_geometry(sample), encoding="utf-8")


def material_geometry(sample):
    """Describe matrix and pore surfaces using the same four arcs per circle."""
    size = sample["config"]["size_mm"]
    spacing = sample["config"]["mesh_size_um"] * 0.001
    lines = ["// Lengths in mm; small pores remain in sample.json.",
             f"h = {spacing!r};", "Mesh.ElementOrder = 1;", "Mesh.MshFileVersion = 4.1;",
             "Mesh.Binary = 0;", "Mesh.Algorithm = 6;", "Mesh.MinimumCirclePoints = 16;"]
    for tag, (x, y) in enumerate([(0, 0), (size, 0), (size, size), (0, size)], 1):
        lines.append(f"Point({tag}) = {{{x!r}, {y!r}, 0, h}};")
    for tag in range(1, 5):
        lines.append(f"Line({tag}) = {{{tag}, {tag % 4 + 1}}};")
    lines.append("Curve Loop(1) = {1, 2, 3, 4};")
    pore_loops = []
    interfaces = []
    for i, pore in enumerate(p for p in sample["pores"] if p["kind"] == "explicit"):
        x, y, r = pore["x_mm"], pore["y_mm"], pore["radius_mm"]
        center = 5 + 5 * i
        points = [(x, y), (x + r, y), (x, y + r), (x - r, y), (x, y - r)]
        for offset, (px, py) in enumerate(points):
            lines.append(f"Point({center + offset}) = {{{px!r}, {py!r}, 0, h}};")
        arcs = list(range(5 + 4 * i, 9 + 4 * i))
        for j, arc in enumerate(arcs):
            lines.append(f"Circle({arc}) = {{{center + j + 1}, {center}, {center + (j + 1) % 4 + 1}}};")
        loop = i + 2
        lines.append(f"Curve Loop({loop}) = {{{', '.join(map(str, arcs))}}};")
        pore_loops.append(loop)
        interfaces.extend(arcs)
    loops = ", ".join(map(str, [1, *pore_loops]))
    lines.extend([f"Plane Surface(1) = {{{loops}}};", 'Physical Surface("matrix", 1) = {1};'])
    for loop in pore_loops:
        lines.append(f"Plane Surface({loop}) = {{{loop}}};")
    if pore_loops:
        lines.append(f'Physical Surface("pores", 2) = {{{", ".join(map(str, pore_loops))}}};')
        lines.append(f'Physical Curve("interfaces", 5) = {{{", ".join(map(str, interfaces))}}};')
    for name, tag, curve in [("left", 1, 4), ("right", 2, 2), ("bottom", 3, 1), ("top", 4, 3)]:
        lines.append(f'Physical Curve("{name}", {tag}) = {{{curve}}};')
    return "\n".join(lines) + "\n"


def preview_svg(sample):
    """Flip y only for display; the saved pore coordinates remain Cartesian."""
    size = sample["config"]["size_mm"]
    scale = 600 / size
    statistics = sample["statistics"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 760" width="760" height="760" role="img">',
        '<title>Synthetic pore sample</title>',
        '<desc>Bottom-left origin, x right, y up. Blue pores are explicit; orange pores are small.</desc>',
        '<rect width="760" height="760" fill="white"/>',
        '<g font-family="sans-serif" fill="#172b4d">',
        f'<text x="80" y="32" font-size="22">Synthetic pores | seed {sample["seed"]}</text>',
        f'<text x="80" y="55" font-size="14">{size:g} mm square | total porosity {statistics["achieved_porosity"]:.2%}</text>',
        '<rect x="80" y="75" width="600" height="600" fill="#eef2f6" stroke="#172b4d"/>',
    ]
    for pore in sample["pores"]:
        color = "#2563a6" if pore["kind"] == "explicit" else "#c45b12"
        lines.append(f'<circle cx="{80 + pore["x_mm"] * scale:.9f}" '
                     f'cy="{75 + (size - pore["y_mm"]) * scale:.9f}" '
                     f'r="{pore["radius_mm"] * scale:.9f}" fill="{color}"/>')
    lines.extend([
        '<text x="80" y="696" font-size="13">(0, 0)</text>',
        f'<text x="680" y="696" text-anchor="end" font-size="13">x = {size:g} mm; y points up</text>',
        '<circle cx="86" cy="722" r="6" fill="#2563a6"/>',
        f'<text x="100" y="727" font-size="14">Explicit: {statistics["explicit_pore_count"]} pores, {statistics["explicit_porosity"]:.2%}</text>',
        '<circle cx="386" cy="722" r="6" fill="#c45b12"/>',
        f'<text x="400" y="727" font-size="14">Small: {statistics["small_pore_count"]} pores, {statistics["small_porosity"]:.2%}</text>',
        '</g></svg>\n',
    ])
    return "\n".join(lines)


def write_mesh_preview(nodes, triangles, size, output):
    """Render the actual mesh, distinguishing matrix and pore cell tags."""
    scale = 600 / size
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 720" width="720" height="720">',
             '<title>Tagged material mesh: matrix gray, pores blue</title>',
             '<rect width="720" height="720" fill="white"/>',
             '<text x="60" y="32" font-family="sans-serif" font-size="20">Tagged material mesh</text>',
             '<g stroke="#172b4d" stroke-width="0.15" stroke-linejoin="round">']
    for a, b, c, region in triangles:
        points = " ".join(f"{60 + nodes[n][0] * scale:.6f},{60 + (size - nodes[n][1]) * scale:.6f}"
                          for n in (a, b, c))
        color = "#e2e8f0" if region == 1 else "#80b6e6"
        lines.append(f'<polygon points="{points}" fill="{color}"/>')
    lines.extend(['</g>', f'<text x="60" y="692" font-family="sans-serif" font-size="14">'
                  f'{len(triangles)} triangles | {size:g} mm square | matrix gray, pores blue</text>', '</svg>\n'])
    Path(output).write_text("\n".join(lines), encoding="utf-8")
