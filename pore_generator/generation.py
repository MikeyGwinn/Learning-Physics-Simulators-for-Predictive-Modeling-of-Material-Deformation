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

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("size_mm", "radius_sigma", "min_radius_um", "geometric_radius_um", "mesh_size_um"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
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
        "statistics": {
            "pore_count": len(pores),
            "explicit_pore_count": sum(p["kind"] == "explicit" for p in pores),
            "small_pore_count": sum(p["kind"] == "small" for p in pores),
            "achieved_porosity": area / domain_area,
            "explicit_porosity": explicit_area / domain_area,
            "small_porosity": small_area / domain_area,
            "porosity_overshoot": area / domain_area - config.target_porosity,
        },
        "assumptions": [
            "Natural-log radius distribution uses micrometers; pore records use millimeters.",
            "Radius distribution is truncated below min_radius_um.",
            "Circular pores do not overlap or touch and remain strictly inside the square.",
            "clearance_um is the minimum pore-to-pore and pore-to-boundary gap.",
            "Stop at the first total pore-area crossing; never rescale radii to match porosity.",
            "Placement and stopping can change the empirical radius distribution.",
            "Placement rules are provisional choices, not a complete reproduction of the paper.",
            "Small pores are retained as data; no effective-stiffness correction is applied yet.",
        ],
    }
