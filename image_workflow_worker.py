"""Internal benchmark worker: one independent image/mesh/fracture pipeline."""
import csv
import json
import sys
import time
from pathlib import Path


def run_task(task_path):
    start = time.perf_counter()
    task_path = Path(task_path)
    task = json.loads(task_path.read_text())
    output = task_path.parent
    from sem_geometry import build_sem_mesh
    from scientific_common import load_config, write_json
    from simulate_fracture import run_simulation

    meta = build_sem_mesh(task["dataset"], task["parameters"], output/"mesh")
    if not meta["ready_for_fracture"]:
        raise ValueError("ROI does not contain one connected solid spanning the horizontal grips")
    config = load_config(task["fracture_config"])
    config["geometry"].update(preset="solid", width=meta["width"], height=meta["height"])
    config["material"]["ell"] = task["parameters"]["ell_nm"]*1e-6
    config["mesh"]["size"] = task["parameters"]["bulk_size_nm"]*1e-6
    config["loading"]["maximum_separation"] = task["maximum_strain"]*meta["width"]
    config["loading"]["minimum_increment"] = config["loading"]["maximum_separation"]/config["loading"]["increments"]/128
    config["solver"].update(backend=task["backend"], linear_solver="gmres")
    config["output"].update(save_fields=False, save_linear_systems=False)
    summary = run_simulation(config, output/"fracture", output/"mesh")
    with (output/"fracture/history.csv").open() as f:
        last = list(csv.DictReader(f))[-1]
    report = {"total_dofs": summary["dofs"]["total"], "image_segmentation_s": meta["image_segmentation_seconds"],
              "contour_s": meta["contour_seconds"], "mesh_s": meta["mesh_seconds"],
              "fracture_setup_s": summary["setup_seconds"], "fracture_compute_s": summary["compute_seconds"],
              "workflow_seconds": time.perf_counter()-start, "geometry_hash": meta["geometry_hash"],
              "source_image_sha256": meta["source_image_sha256"], "accepted_steps": summary["accepted_steps"],
              "rejected_steps": summary["rejected_steps"], "final_damage_max": summary["final_damage_max"],
              "energy_imbalance_relative": abs(float(last["energy_balance_error"]))/max(abs(float(last["external_work"])), 1e-30)}
    write_json(output/"report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run_task(sys.argv[1]), indent=2))
