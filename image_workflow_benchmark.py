"""Plan or execute an image -> geometry -> mesh -> fracture CPU/CUDA study.

The default is a dry run. Execution requires the explicit --execute flag.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from scientific_common import load_config, write_json
from sem_geometry import dataset_record, validate_params

ROOT = Path(__file__).resolve().parent


def plan_tasks(plan, plan_path):
    parent = Path(plan_path).resolve().parent
    # Paths in the shipped plan are relative to the workspace, not the shell cwd.
    base = (parent / plan.get("workspace", "..")).resolve()
    dataset = (base / plan["dataset"]).resolve()
    load_config(base / plan["fracture_config"])
    if not plan["images"] or len(set(plan["images"])) != len(plan["images"]):
        raise ValueError("Select unique image IDs")
    if not plan["mesh_sizes_nm"] or len(set(plan["mesh_sizes_nm"])) != len(plan["mesh_sizes_nm"]):
        raise ValueError("Select unique mesh sizes")
    if not plan["backends"] or any(b not in ("cpu", "gpu") for b in plan["backends"]) or len(set(plan["backends"])) != len(plan["backends"]):
        raise ValueError("Backends must be unique cpu/gpu selections")
    for key in ("repeats", "warmups", "cpu_threads"):
        if not isinstance(plan[key], int) or isinstance(plan[key], bool) or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < plan["maximum_strain"] <= 0.1 or plan["timeout_seconds"] <= 0:
        raise ValueError("Invalid loading strain or timeout")
    tasks = []
    backends = sorted(plan["backends"], key=lambda x: x != "cpu")
    for image in plan["images"]:
        dataset_record(dataset, image)
        for size in plan["mesh_sizes_nm"]:
            params = validate_params({**plan["meshing"], "image_id": image,
                         "bulk_size_nm": size, "boundary_size_nm": size*plan["boundary_size_ratio"]})
            for backend in backends:
                for repetition in range(-plan["warmups"], plan["repeats"]):
                    tasks.append({"dataset": str(dataset), "parameters": params, "backend": backend,
                                  "repeat": repetition, "fracture_config": str(base/plan["fracture_config"]),
                                  "maximum_strain": plan["maximum_strain"], "mesh_size_nm": size,
                                  "cpu_threads": plan["cpu_threads"]})
    return tasks


def execute(plan, tasks, output):
    from benchmark import validate_pair
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"plan.json", plan)
    write_json(output/"tasks.json", tasks)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gpu = "unavailable"
    write_json(output/"hardware.json", {"host": platform.node(), "platform": platform.platform(),
                "cpu_threads": plan["cpu_threads"], "gpu": gpu, "mpi_ranks": 1,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")})
    references, failed = {}, False
    columns = ["image_id", "mesh_size_nm", "backend", "repeat", "status", "total_dofs",
               "image_segmentation_s", "contour_s", "mesh_s", "fracture_setup_s", "fracture_compute_s",
               "workflow_seconds", "process_wall_seconds", "geometry_hash", "source_image_sha256",
               "accepted_steps", "rejected_steps", "final_damage_max", "energy_imbalance_relative",
               "agreement", "reason", "folder"]
    with (output/"measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, task in enumerate(tasks):
            folder = output/f"task_{index:04d}"
            folder.mkdir()
            write_json(folder/"task.json", task)
            row = dict.fromkeys(columns, "")
            row.update(image_id=task["parameters"]["image_id"], mesh_size_nm=task["mesh_size_nm"],
                       backend=task["backend"], repeat=task["repeat"], status="failed", folder=str(folder))
            print(f"{index+1}/{len(tasks)} {row['image_id']} h={row['mesh_size_nm']} nm {row['backend']} repeat={row['repeat']}", flush=True)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS=str(task["cpu_threads"]), OPENBLAS_NUM_THREADS=str(task["cpu_threads"]), MKL_NUM_THREADS=str(task["cpu_threads"]))
            t = time.perf_counter()
            try:
                with (folder/"worker.log").open("w") as log:
                    result = subprocess.run([sys.executable, str(ROOT/"image_workflow_worker.py"), str(folder/"task.json")],
                                            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=plan["timeout_seconds"])
                wall = time.perf_counter()-t
                if result.returncode:
                    raise RuntimeError((folder/"worker.log").read_text()[-2000:])
                report = json.loads((folder/"report.json").read_text())
                row.update(report)
                row["process_wall_seconds"] = wall
                case = (row["image_id"], row["mesh_size_nm"])
                if task["backend"] == "cpu" and case not in references:
                    references[case] = folder/"fracture"
                if case in references:
                    valid, agreement = validate_pair(references[case], folder/"fracture")
                    row["agreement"] = agreement
                    if not valid:
                        raise RuntimeError("CPU/backend fracture results differ beyond tolerance: "+agreement)
                else:
                    row["agreement"] = "CPU reference not selected; no speedup validation"
                row["status"] = "complete"
            except Exception as exc:
                failed = True
                row["status"], row["reason"] = "failed", str(exc)
                write_json(folder/"failure.json", {"status": "failed", "reason": str(exc)})
            writer.writerow(row)
            stream.flush()
    write_json(output/"status.json", {"status": "failed" if failed else "complete", "tasks": len(tasks),
               "timing_scope": "process_wall includes process start, image decode/segmentation, contours, meshing, FEM import/JIT/setup, all load steps and final output. Network download excluded. Warmups have negative repeat."})
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT/"configs/sem_workflow_benchmark.json")
    parser.add_argument("--output", type=Path, default=Path("results/image_workflow_benchmark"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true", help="Explicitly run the complete benchmark")
    group.add_argument("--dry-run", action="store_true", help="Validate and print the plan only (default)")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    tasks = plan_tasks(plan, args.plan)
    if not args.execute:
        print(json.dumps({"status": "dry_run", "task_count": len(tasks),
                          "measured_tasks": sum(t["repeat"] >= 0 for t in tasks),
                          "images": plan["images"], "mesh_sizes_nm": plan["mesh_sizes_nm"],
                          "backends": plan["backends"], "repeats": plan["repeats"],
                          "output": str(args.output), "note": "No image processing, meshing, fracture solves or result-directory writes performed"}, indent=2))
        return
    if not execute(plan, tasks, args.output):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
