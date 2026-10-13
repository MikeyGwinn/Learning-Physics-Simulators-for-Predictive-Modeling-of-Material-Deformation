"""Agreement validation for SEM benchmarks and scientific solver checks."""
from __future__ import annotations

import csv
import json

import numpy as np



def validate_pair(cpu_path, gpu_path, tolerance=2e-3):
    """Check physics agreement, allowing independently adapted load paths."""
    cpu, gpu = np.load(cpu_path / "final_state.npz"), np.load(gpu_path / "final_state.npz")
    cm = json.loads((cpu_path / "metadata.json").read_text())
    gm = json.loads((gpu_path / "metadata.json").read_text())
    if cm["geometry_hash"] != gm["geometry_hash"] or cm["dofs"] != gm["dofs"]:
        return False, "Geometry or DOFs differ"
    if not np.array_equal(cpu["coordinates"], gpu["coordinates"]):
        return False, "Mesh node ordering differs"
    u_error = np.linalg.norm(cpu["displacement"]-gpu["displacement"])/max(np.linalg.norm(cpu["displacement"]), 1e-12)
    d_error = np.max(np.abs(cpu["damage"]-gpu["damage"]))
    histories = []
    for path in (cpu_path, gpu_path):
        with (path / "history.csv").open() as f:
            histories.append(list(csv.DictReader(f)))
    # Compare complete force curves on common accepted load checkpoints.
    ch, gh = histories
    cx, gx = [float(r["separation"]) for r in ch], [float(r["separation"]) for r in gh]
    if np.any(np.diff(cx) < 0) or np.any(np.diff(gx) < 0):
        return False, "Scaling benchmark requires monotone loading"
    cf, gf = np.array([float(r["force"]) for r in ch]), np.array([float(r["force"]) for r in gh])
    force_error = np.max(np.abs(cf-np.interp(cx, gx, gf)))/max(np.max(np.abs(cf)), 1e-12)
    ec = float(ch[-1]["elastic_energy"])+float(ch[-1]["fracture_energy"])
    eg = float(gh[-1]["elastic_energy"])+float(gh[-1]["fracture_energy"])
    energy_error = abs(ec-eg)/max(abs(ec), 1e-12)
    valid = bool(max(u_error, d_error, force_error, energy_error) <= tolerance)
    return valid, f"u_relative={u_error:.3g}, damage_max_abs={d_error:.3g}, force_relative={force_error:.3g}, energy_relative={energy_error:.3g}"
