"""Configuration validation for SEM fracture workflows (no FEM imports)."""
from __future__ import annotations

import json
import math
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).parent / "configs/sem_fracture.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(c):
    g, m, mesh, load, s = (c[k] for k in ("geometry", "material", "mesh", "loading", "solver"))
    for name, value in [("width", g["width"]), ("height", g["height"]),
                        ("E", m["E"]), ("Gc", m["Gc"]), ("ell", m["ell"]),
                        ("thickness", m["thickness"]), ("mesh size", mesh["size"]),
                        ("separation", load["maximum_separation"]),
                        ("minimum increment", load["minimum_increment"])]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not -1 < m["nu"] < 0.5:
        raise ValueError("Poisson ratio must lie in (-1, 0.5)")
    if not 0 < m["residual_stiffness"] < 1:
        raise ValueError("residual_stiffness must lie in (0, 1)")
    if m["kinematics"] not in ("plane_strain", "plane_stress"):
        raise ValueError("kinematics must be plane_strain or plane_stress")
    if g["preset"] not in ("ligament", "random", "solid", "explicit", "notched"):
        raise ValueError("Unknown geometry preset")
    for name, value in [("increments", load["increments"]),
                        ("maximum_stagger_iterations", s["maximum_stagger_iterations"]),
                        ("field_stride", c["output"]["field_stride"]),
                        ("ksp_max_it", s["ksp_max_it"]), ("snes_max_it", s["snes_max_it"])]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("ksp_rtol", "ksp_atol", "snes_rtol", "snes_atol", "stagger_tolerance"):
        if not math.isfinite(s[name]) or s[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if s["backend"] not in ("cpu", "gpu"):
        raise ValueError("backend must be cpu or gpu")
    if s["linear_solver"] not in ("gmres", "direct") or s["preconditioner"] not in ("jacobi", "gamg"):
        raise ValueError("Supported solvers: GMRES with Jacobi/GAMG or CPU direct/MUMPS")
    if s["backend"] == "gpu" and s["linear_solver"] == "direct":
        raise ValueError("Direct reference is CPU only")
    if "targets" in load:
        targets = load["targets"]
        if not isinstance(targets, list) or not targets or not any(v > 0 for v in targets) or not all(math.isfinite(v) and v >= 0 for v in targets):
            raise ValueError("Loading targets must be finite, nonnegative and include a positive separation")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
