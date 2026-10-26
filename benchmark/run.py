"""Stage 300 SEM slices and submit four persistent Slurm conversion workers.

Examples:
  python -m benchmark.run --crop 1400 900 384 384
  python -m benchmark.run --full-image --dry-run
  python -m benchmark.run --resume benchmark/results/RUN_ID
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import uuid

# Direct execution puts benchmark/ (rather than the checkout) on sys.path.
# On this cluster the shell's default Python is 3.9 without the meshing stack;
# use the existing shared environment before importing the package.
_CHECKOUT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(_CHECKOUT))
    __package__ = "benchmark"
_MISSING = [name for name in ("numpy", "PIL", "scipy", "skimage", "shapely", "gmsh")
            if importlib.util.find_spec(name) is None]
if sys.version_info < (3, 10) or _MISSING:
    _FALLBACK = _CHECKOUT.parent / "PhaseFieldTest" / ".conda" / "bin" / "python"
    if (_FALLBACK.is_file() and os.access(_FALLBACK, os.X_OK)
            and Path(sys.executable).resolve() != _FALLBACK.resolve()):
        print(f"Using {_FALLBACK} (current Python lacks the benchmark requirements)", file=sys.stderr)
        os.execv(str(_FALLBACK), [str(_FALLBACK), str(Path(__file__).resolve()), *sys.argv[1:]])
    missing = ", ".join(_MISSING) if _MISSING else "none"
    raise SystemExit(f"The benchmark requires Python >=3.10 and its meshing dependencies (missing: {missing}). "
                     "See benchmark/README.md for environment setup.")

from sem2msh import MeshingOptions, SegmentationOptions
from sem2msh.image import _validate_segmentation
from sem2msh.surface import _validate_meshing

from .staging import ARCHIVE_URL, DATASET_DOI, selected_images, stage_images
from .worker import complete_report, convert_one, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "benchmark" / "cache"
DEFAULT_RESULTS = ROOT / "benchmark" / "results"
GUI_SEED = ROOT.parent / "PhaseFieldTest" / "datasets" / "skarberg" / "raw"
PARTITIONS = ("share", "dgxh", "dgx2", "eecs", "gpu", "ampere")
DEFAULT_PARTITIONS = "share,dgx2"
TIMING_FIELDS = ("load_seconds", "segment_seconds", "contour_seconds", "mesh_seconds",
                 "msh_write_seconds", "png_render_seconds", "total_seconds")


def resolved_config(crop: tuple[int, int, int, int] | None, override: Path | None = None,
                    threshold_percentile: float | None = None) -> dict:
    """Merge a small optional JSON override with the GUI's mesh-only defaults."""
    config = {
        "pixel_size_nm": 10.0,
        "segmentation": {
            "threshold": 0.45, "threshold_percentile": None, "crop": crop,
            "min_solid_pixels": 16, "fill_holes_pixels": 8, "component": "all",
        },
        "tolerance_px": 0.5,
        "meshing": asdict(MeshingOptions()),
    }
    if override is not None:
        values = json.loads(Path(override).read_text())
        if not isinstance(values, dict) or set(values) - set(config):
            raise ValueError("configuration has unknown top-level keys")
        for section in ("segmentation", "meshing"):
            if section in values:
                if not isinstance(values[section], dict) or set(values[section]) - set(config[section]):
                    raise ValueError(f"configuration has unknown {section} keys")
                if section == "segmentation" and "crop" in values[section]:
                    raise ValueError("choose crop with --crop or --full-image, not the configuration file")
                config[section].update(values.pop(section))
        config.update(values)
    if threshold_percentile is not None:
        config["segmentation"]["threshold_percentile"] = threshold_percentile
    scale = config["pixel_size_nm"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("pixel_size_nm must be finite and positive")
    tolerance = config["tolerance_px"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 <= tolerance <= 2:
        raise ValueError("tolerance_px must be between 0 and 2")
    _validate_segmentation(SegmentationOptions(**config["segmentation"]), 3372, 2247)
    _validate_meshing(MeshingOptions(**config["meshing"]))
    return config


def array_spec(indices: list[int]) -> str:
    """Start up to four workers, each drawing images from the shared queue."""
    if not indices or len(indices) != len(set(indices)) or any(i < 0 for i in indices):
        raise ValueError("array indices must be unique nonnegative integers")
    return f"0-{min(4, len(indices)) - 1}%4"


def _job_id(output: str) -> str:
    value = output.strip().split(";", 1)[0]
    if not value.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    return value


def _submit(command: list[str], label: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{label} submission failed (sbatch exit {result.returncode}): {detail}")
    return _job_id(result.stdout)


def parse_partitions(value: str) -> str:
    """Return Slurm's comma-separated partition argument without shell quotes."""
    names = [name.strip() for name in value.split(",")]
    if not names or any(name not in PARTITIONS for name in names) or len(names) != len(set(names)):
        raise ValueError(f"partitions must be unique names from: {', '.join(PARTITIONS)}")
    return ",".join(names)


def submit_array(run_dir: Path, indices: list[int], *, wait_for_all_nodes: bool = False,
                 partitions: str = DEFAULT_PARTITIONS) -> dict:
    """Submit persistent workers and an afterany aggregation job."""
    run_dir = Path(run_dir).resolve()
    specification = array_spec(indices)
    partitions = parse_partitions(partitions)
    worker_count = min(4, len(indices))
    (run_dir / "logs").mkdir(exist_ok=True)
    queue_path = run_dir / f"queue-{uuid.uuid4().hex}.json"
    write_json(queue_path, {"indices": sorted(indices), "next": 0})
    worker_script = run_dir / "worker.sbatch"
    worker_command = [sys.executable, "-m", "benchmark.run", "work-queue", str(run_dir), str(queue_path)]
    if wait_for_all_nodes:
        worker_command = ["srun", f"--nodes={worker_count}", f"--ntasks={worker_count}",
                          "--ntasks-per-node=1", "--cpus-per-task=24", *worker_command]
    worker_script.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(worker_command) + "\n")
    array_command = [
        "sbatch", "--parsable", f"--partition={partitions}",
    ]
    if wait_for_all_nodes:
        array_command += [f"--nodes={worker_count}", f"--ntasks={worker_count}",
                          "--ntasks-per-node=1", "--wait-all-nodes=1"]
    else:
        array_command += [f"--array={specification}", "--nodes=1", "--ntasks=1"]
    array_command += [
        "--cpus-per-task=24", "--exclusive", "--time=12:00:00", f"--chdir={ROOT}",
        f"--output={run_dir}/logs/{'%j' if wait_for_all_nodes else '%A_%a'}.out",
        f"--error={run_dir}/logs/{'%j' if wait_for_all_nodes else '%A_%a'}.err",
        str(worker_script)]
    array_id = _submit(array_command, "conversion array")
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    submission = {"array_job_id": array_id, "task_indices": indices,
                  "queue_path": str(queue_path), "worker_count": worker_count,
                  "wait_for_all_nodes": wait_for_all_nodes, "partitions": partitions,
                  "submitted_utc": datetime.now(timezone.utc).isoformat()}
    run.setdefault("submissions", []).append(submission)
    write_json(run_path, run)
    aggregate_script = run_dir / "aggregate.sbatch"
    aggregate_script.write_text("#!/bin/bash\nset -euo pipefail\nexec "
                                + shlex.join([sys.executable, "-m", "benchmark.run", "aggregate", str(run_dir)])
                                + "\n")
    aggregate_command = [
        "sbatch", "--parsable", "--partition=share", f"--dependency=afterany:{array_id}",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--time=00:10:00",
        f"--chdir={ROOT}", f"--output={run_dir}/logs/aggregate_%j.out",
        f"--error={run_dir}/logs/aggregate_%j.err", str(aggregate_script),
    ]
    submission["aggregate_job_id"] = _submit(aggregate_command, "aggregation job")
    write_json(run_path, run)
    return submission


def pending_indices(run_dir: Path) -> list[int]:
    """Find missing or failed outputs, including reports left running by a kill."""
    run_dir = Path(run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    pending = []
    for index, task in enumerate(run["tasks"]):
        folder = run_dir / "tasks" / task["id"]
        try:
            report = json.loads((folder / "report.json").read_text())
            if report.get("id") == task["id"] and complete_report(report, folder):
                continue
        except (OSError, ValueError):
            pass
        pending.append(index)
    return pending


def check_previous_jobs_finished(run_dir: Path) -> None:
    run = json.loads((Path(run_dir) / "run.json").read_text())
    result = subprocess.run(["squeue", "--noheader", f"--user={getpass.getuser()}", "--format=%i"],
                            check=True, capture_output=True, text=True)
    active = result.stdout.splitlines()
    for submission in run.get("submissions", []):
        for key in ("array_job_id", "aggregate_job_id"):
            job_id = submission.get(key)
            if job_id and any(entry == job_id or entry.startswith(job_id + "_") for entry in active):
                raise RuntimeError(f"Slurm job {job_id} is still active")


def aggregate(run_dir: Path) -> dict:
    """Collect per-image reports after the array finishes; retain failures."""
    run_dir = Path(run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    columns = ["index", "id", "status", "node", "partition", "slurm_job_id",
               "started_utc", "finished_utc", "source_sha256",
               *TIMING_FIELDS, "nodes", "triangles", "minimum_quality", "mean_quality", "error"]
    rows = []
    for index, task in enumerate(run["tasks"]):
        folder = run_dir / "tasks" / task["id"]
        try:
            report = json.loads((folder / "report.json").read_text())
        except (OSError, ValueError):
            report = {"status": "missing", "error": "worker report is missing"}
        status = report.get("status", "missing")
        if status == "complete" and report.get("id") != task["id"]:
            status = "failed"
            report["error"] = "worker report image ID does not match the run manifest"
        if status == "complete" and not complete_report(report, folder):
            status = "failed"
            report["error"] = "mesh or PNG output is missing"
        row = {"index": index, "id": task["id"], "status": status,
               "node": report.get("node", ""), "partition": report.get("partition", ""),
               "slurm_job_id": report.get("slurm_job_id", ""),
               "started_utc": report.get("started_utc", ""),
               "finished_utc": report.get("finished_utc", ""),
               "source_sha256": task["sha256"], "error": report.get("error", "")}
        row.update({name: report.get("timing", {}).get(name, "") for name in TIMING_FIELDS})
        row["total_seconds"] = report.get("total_seconds", "")
        row.update({name: report.get("quality", {}).get(name, "")
                    for name in ("nodes", "triangles", "minimum_quality", "mean_quality")})
        rows.append(row)
    csv_path = run_dir / "measurements.csv"
    temporary_csv = run_dir / f"measurements.{uuid.uuid4().hex}.part"
    try:
        with temporary_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary_csv.replace(csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
    complete = [row for row in rows if row["status"] == "complete"]
    terminal = [row for row in rows if row["status"] in ("complete", "failed")]
    timed = [row for row in terminal if row["started_utc"] and row["finished_utc"]]
    wall_seconds = None
    if len(timed) == len(rows):
        start = min(datetime.fromisoformat(row["started_utc"]) for row in timed)
        finish = max(datetime.fromisoformat(row["finished_utc"]) for row in timed)
        wall_seconds = (finish - start).total_seconds()
    summary = {
        "status": "complete" if len(complete) == len(rows) else "incomplete",
        "task_count": len(rows), "completed": len(complete),
        "failed": sum(row["status"] == "failed" for row in rows),
        "missing_or_running": sum(row["status"] not in ("complete", "failed") for row in rows),
        "timing_scope": "Worker configuration and conversion through MSH and mesh-only PNG write; download, queue wait, and Python startup excluded.",
        "conversion_baseline": {
            "wall_seconds": wall_seconds,
            "images_per_second": len(rows) / wall_seconds if wall_seconds and wall_seconds > 0 else None,
            "attempted": len(terminal),
            "nodes_used": len({row["node"] for row in terminal if row["node"]}),
            "all_nodes_allocated_before_start": bool(run.get("submissions", [])
                                                     and run["submissions"][-1].get("wait_for_all_nodes")),
        },
        "total_seconds": {
            "median": statistics.median(row["total_seconds"] for row in complete) if complete else None,
            "mean": statistics.mean(row["total_seconds"] for row in complete) if complete else None,
            "maximum": max((row["total_seconds"] for row in complete), default=None),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    area = parser.add_mutually_exclusive_group()
    area.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    area.add_argument("--full-image", action="store_true")
    parser.add_argument("--config", type=Path, help="JSON override of GUI threshold/mesh defaults")
    parser.add_argument("--threshold-percentile", type=float, metavar="P",
                        help="Choose each image's cutoff from percentile P of its cropped intensities (0 < P < 100)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the published ZIP")
    parser.add_argument("--seed-dir", type=Path, default=GUI_SEED if GUI_SEED.is_dir() else None,
                        help="Optional read-only directory of already cached GUI TIFFs")
    parser.add_argument("--partitions", default=None,
                        help="Comma-separated Slurm partitions: " + ", ".join(PARTITIONS)
                             + f" (default: {DEFAULT_PARTITIONS})")
    parser.add_argument("--wait-for-all-nodes", action=argparse.BooleanOptionalAction, default=None,
                        help="Request all worker nodes in one Slurm allocation before converting images")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print queue without writes or submission")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="Submit only missing/failed tasks from an existing run")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "worker":
        if len(argv) != 3:
            raise SystemExit("usage: python -m benchmark.run worker RUN_DIR TASK_INDEX")
        report = convert_one(Path(argv[1]), int(argv[2]))
        print(json.dumps({key: report.get(key) for key in ("id", "status", "total_seconds", "error")}))
        return 0 if report["status"] == "complete" else 1
    if argv and argv[0] == "work-queue":
        if len(argv) != 3:
            raise SystemExit("usage: python -m benchmark.run work-queue RUN_DIR QUEUE_PATH")
        from .worker import process_queue
        result = process_queue(Path(argv[1]), Path(argv[2]))
        print(json.dumps(result))
        return 0 if result["failed"] == 0 else 1
    if argv and argv[0] == "aggregate":
        if len(argv) != 2:
            raise SystemExit("usage: python -m benchmark.run aggregate RUN_DIR")
        summary = aggregate(Path(argv[1]))
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "complete" else 1
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        partitions = parse_partitions(args.partitions or DEFAULT_PARTITIONS)
    except ValueError as error:
        parser.error(str(error))
    if args.resume is not None:
        if (args.crop is not None or args.full_image or args.config is not None
                or args.archive is not None or args.threshold_percentile is not None):
            parser.error("--resume uses the original run configuration and cannot change crop, threshold, config or archive")
        run_dir = args.resume.resolve()
        run = json.loads((run_dir / "run.json").read_text())
        previous = run.get("submissions", [])
        prior = previous[-1] if previous else {}
        if args.partitions is None:
            partitions = prior.get("partitions", DEFAULT_PARTITIONS)
        wait_for_all_nodes = (prior.get("wait_for_all_nodes", False) if args.wait_for_all_nodes is None
                              else args.wait_for_all_nodes)
        indices = pending_indices(run_dir)
        if args.dry_run:
            print(json.dumps({"run_dir": str(run_dir), "pending_indices": indices,
                              "pending_count": len(indices), "partitions": partitions,
                              "wait_for_all_nodes": wait_for_all_nodes}, indent=2))
            return 0
        check_previous_jobs_finished(run_dir)
        if not indices:
            print(json.dumps({"run_dir": str(run_dir), "pending_count": 0, "status": "complete"}))
            return 0
        print(json.dumps({"run_dir": str(run_dir), "resubmitted": len(indices),
                          "slurm": submit_array(run_dir, indices,
                                                wait_for_all_nodes=wait_for_all_nodes,
                                                partitions=partitions)}, indent=2))
        return 0
    if args.crop is None and not args.full_image:
        parser.error("choose --crop LEFT TOP WIDTH HEIGHT or --full-image")
    crop = tuple(args.crop) if args.crop is not None else None
    config = resolved_config(crop, args.config, args.threshold_percentile)
    records = selected_images()
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "task_count": len(records), "image_ids": [r["id"] for r in records],
                          "config": config, "partition": partitions, "exclusive_nodes": True,
                          "partitions": partitions, "wait_for_all_nodes": bool(args.wait_for_all_nodes),
                          "maximum_concurrent_nodes": 4, "worker_count": 4,
                          "note": "No images downloaded, results written, or jobs submitted"}, indent=2))
        return 0
    # Check that the submitting interpreter also has the lazy Gmsh dependency.
    import gmsh  # noqa: F401

    staged = stage_images(records, args.cache, archive=args.archive, seed_dir=args.seed_dir)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    tasks = [{**record, "index": index} for index, record in enumerate(staged)]
    write_json(run_dir / "run.json", {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_doi": DATASET_DOI, "source_archive": str(args.archive.resolve()) if args.archive else ARCHIVE_URL,
        "config": config, "tasks": tasks, "submissions": [],
    })
    submission = submit_array(run_dir, list(range(len(tasks))),
                              wait_for_all_nodes=bool(args.wait_for_all_nodes), partitions=partitions)
    print(json.dumps({"run_dir": str(run_dir), "task_count": len(tasks), "slurm": submission}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
