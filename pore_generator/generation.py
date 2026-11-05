"""Sample nonoverlapping circular pores; all stored lengths are millimeters."""

from dataclasses import asdict, dataclass
import math
import platform
import random


MAX_ATTEMPTS = 10_000


@dataclass(frozen=True)
class Config:
    # defaults taken from the paper
    size_mm: float = 1.0
    radius_mu: float = 2.941
    radius_sigma: float = 0.703
    min_radius_um: float = 1.0
    geometric_radius_um: float = 20.0
    target_porosity: float = 0.20
    clearance_um: float = 0.0
    mesh_size_um: float = 4.0
    effective_cell_um: float = 50.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("size_mm", "radius_sigma", "min_radius_um", "geometric_radius_um", "mesh_size_um",
                     "effective_cell_um"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        cells = self.size_mm * 1000 / self.effective_cell_um
        if round(cells) < 1 or abs(cells - round(cells)) > 1e-9 * cells:
            raise ValueError("size_mm must be a whole number of effective cells")
        if not 0 <= self.target_porosity < 1:
            raise ValueError("target_porosity must be in [0, 1)")
        if self.clearance_um < 0:
            raise ValueError("clearance_um must be nonnegative")
        if self.geometric_radius_um < self.min_radius_um:
            raise ValueError("geometric_radius_um must be >= min_radius_um")
        if 2 * (self.min_radius_um + self.clearance_um) * 0.001 >= self.size_mm:
            raise ValueError("Minimum pore radius and clearance cannot fit inside the sample")


def sample_radius_mm(rng, config):
    """Draw ln(radius / micrometer) ~ N(mu, sigma²), truncated below the minimum."""
    for _ in range(MAX_ATTEMPTS):
        try:
            radius_um = rng.lognormvariate(config.radius_mu, config.radius_sigma)
        except OverflowError as exc:
            raise ValueError("Sampled radius overflowed; check distribution parameters") from exc
        if not math.isfinite(radius_um):
            raise ValueError("Sampled radius is not finite; check distribution parameters")
        if radius_um >= config.min_radius_um:
            return radius_um * 0.001
    raise ValueError("Could not sample a radius above the minimum after 10000 draws")


def circle_rectangle_area(x, y, r, x0, x1, y0, y1):
    """Exact area (mm²) of the circle (x, y, r) inside the rectangle [x0, x1] × [y0, y1]."""
    def half_chord(u):
        return math.sqrt(max(r * r - u * u, 0.0))

    def half_chord_integral(u):
        u = min(max(u, -r), r)
        return (u * half_chord(u) + r * r * math.asin(u / r)) / 2

    left, right = max(x0, x - r), min(x1, x + r)
    if left >= right:
        return 0.0
    # Split x where the circle crosses y0 or y1; each piece then has a closed-form integral.
    breaks = {left, right}
    for edge in (y0, y1):
        if abs(edge - y) < r:
            offset = half_chord(edge - y)
            breaks.update(b for b in (x - offset, x + offset) if left < b < right)
    breaks = sorted(breaks)
    area = 0.0
    for a, b in zip(breaks, breaks[1:]):
        h = half_chord((a + b) / 2 - x)
        if min(y1, y + h) <= max(y0, y - h):
            continue
        # Top and bottom are each a rectangle edge or y ± half chord: (constant, half-chord sign).
        top = (y, 1) if y + h < y1 else (y1, 0)
        bottom = (y, -1) if y - h > y0 else (y0, 0)
        area += ((top[0] - bottom[0]) * (b - a)
                 + (top[1] - bottom[1]) * (half_chord_integral(b - x) - half_chord_integral(a - x)))
    return area


def effective_porosity(pores, config):
    """Small-pore area fraction per square cell, [n][n] with row 0 at the bottom.

    Chakraborty et al. (2016), Section 3: pores below the geometric cutoff become an
    effective porosity on a coarse grid, and the solver uses E_eff = (1 - f_eff) E.
    """
    cell = config.effective_cell_um * 0.001
    n = round(config.size_mm / cell)
    areas = [[0.0] * n for _ in range(n)]
    for pore in pores:
        if pore["kind"] != "small":
            continue
        x, y, r = pore["x_mm"], pore["y_mm"], pore["radius_mm"]
        for j in range(max(int((y - r) // cell), 0), min(int((y + r) // cell), n - 1) + 1):
            for i in range(max(int((x - r) // cell), 0), min(int((x + r) // cell), n - 1) + 1):
                areas[j][i] += circle_rectangle_area(x, y, r, i * cell, (i + 1) * cell, j * cell, (j + 1) * cell)
    return [[area / cell ** 2 for area in row] for row in areas]


def generate(config, seed):
    """Place one fixed radius at a time, stopping at the first porosity crossing.

    Centers are uniform within the radius- and clearance-dependent interior.
    Clearance applies both between pores and from pores to the domain boundary.
    Failed placement aborts the sample rather than replacing a difficult radius.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    pores = []
    area = 0.0
    explicit_area = 0.0
    small_area = 0.0
    domain_area = config.size_mm ** 2
    clearance = config.clearance_um * 0.001
    threshold = config.geometric_radius_um * 0.001
    while area < config.target_porosity * domain_area:
        radius = sample_radius_mm(rng, config)
        margin = radius + clearance
        if 2 * margin >= config.size_mm:
            raise ValueError(f"Pore {len(pores) + 1}: radius {radius:g} mm cannot fit inside the sample")
        for _ in range(MAX_ATTEMPTS):
            x = rng.uniform(margin, config.size_mm - margin)
            y = rng.uniform(margin, config.size_mm - margin)
            if min(x, y, config.size_mm - x, config.size_mm - y) <= margin:
                continue
            if all(math.hypot(x - p["x_mm"], y - p["y_mm"]) > radius + p["radius_mm"] + clearance
                   for p in pores):
                break
        else:
            raise ValueError(
                f"Pore {len(pores) + 1}: could not place radius {radius:g} mm after 10000 attempts; "
                f"porosity reached {area / domain_area:.6f}. No successful sample was produced."
            )
        kind = "explicit" if radius >= threshold else "small"
        pores.append({"id": len(pores) + 1, "x_mm": x, "y_mm": y,
                      "radius_mm": radius, "kind": kind})
        pore_area = math.pi * radius ** 2
        area += pore_area
        if kind == "explicit":
            explicit_area += pore_area
        else:
            small_area += pore_area
    return {
        "schema_version": 1,
        "generator": {"python_version": platform.python_version(),
                      "python_implementation": platform.python_implementation()},
        "seed": seed,
        "units": "mm",
        "coordinates": {"origin": "bottom-left", "x": "right", "y": "up"},
        "config": asdict(config),
        "pores": pores,
        "effective_porosity": {"cell_mm": config.effective_cell_um * 0.001,
                               "values": effective_porosity(pores, config)},
        "statistics": {
            "pore_count": len(pores),
            "explicit_pore_count": sum(p["kind"] == "explicit" for p in pores),
            "small_pore_count": sum(p["kind"] == "small" for p in pores),
            "achieved_porosity": area / domain_area,
            "explicit_porosity": explicit_area / domain_area,
            "small_porosity": small_area / domain_area,
            "porosity_overshoot": area / domain_area - config.target_porosity,
        },
    }
