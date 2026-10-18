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
