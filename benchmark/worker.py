"""Convert staged TIFFs to meshes, PNGs, and per-image reports."""

from __future__ import annotations

import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from sem2msh import MeshingOptions, SEMImage, SegmentationOptions


def write_json(path: Path, value: dict) -> None:
    """Write a small state file atomically so resume never reads half a report."""
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_mesh_png(points: np.ndarray, triangles: np.ndarray, mask_shape: tuple[int, int],
                    pixel_size_nm: float, path: Path, longest_side: int = 2048) -> None:
    """Draw all triangle edges in black on white, in physical aspect ratio."""
    height_px, width_px = mask_shape
    ratio = width_px / height_px
    width = longest_side if ratio >= 1 else max(2, round(longest_side * ratio))
    height = longest_side if ratio <= 1 else max(2, round(longest_side / ratio))
    scale = pixel_size_nm * 1e-6
    x = np.rint(points[:, 0] / (width_px * scale) * (width - 1)).astype(np.int64)
    y = np.rint((1 - points[:, 1] / (height_px * scale)) * (height - 1)).astype(np.int64)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for a, b, c in triangles:
        draw.line([(int(x[a]), int(y[a])), (int(x[b]), int(y[b])),
                   (int(x[c]), int(y[c])), (int(x[a]), int(y[a]))], fill="black", width=1)
    canvas.save(path, format="PNG")


def complete_report(report: dict, folder: Path) -> bool:
    return (report.get("status") == "complete"
            and (folder / "mesh.msh").is_file() and (folder / "mesh.msh").stat().st_size > 0
            and (folder / "mesh.png").is_file() and (folder / "mesh.png").stat().st_size > 0)


def claim_next(queue_path: Path) -> int | None:
    """Claim one manifest index under a filesystem lock shared by Slurm nodes."""
    queue_path = Path(queue_path)
    lock_path = queue_path.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            queue = json.loads(queue_path.read_text())
            position = queue["next"]
            if position >= len(queue["indices"]):
                return None
            index = queue["indices"][position]
            queue["next"] = position + 1
            write_json(queue_path, queue)
            return index
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def process_queue(run_dir: Path, queue_path: Path) -> dict:
    """Keep a node occupied until it has converted every image it can claim."""
    completed = failed = 0
    while (index := claim_next(queue_path)) is not None:
        report = convert_one(run_dir, index)
        if report["status"] == "complete":
            completed += 1
        else:
            failed += 1
        print(json.dumps({key: report.get(key) for key in
                          ("id", "status", "total_seconds", "error")}), flush=True)
    return {"completed": completed, "failed": failed}


def convert_one(run_dir: Path, index: int) -> dict:
    """Execute one conversion, recording a failed report instead of losing it."""
    run_dir = Path(run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    if not 0 <= index < len(run["tasks"]):
        raise ValueError(f"task index {index} is outside the run manifest")
    task = run["tasks"][index]
    folder = run_dir / "tasks" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / "report.json"
    report = {
        "id": task["id"], "index": index, "status": "running",
        "node": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "source_sha256": task["sha256"],
    }
    write_json(report_path, report)
    started = time.perf_counter()
    times = {}
    mesh_temp = folder / "mesh.msh.part"
    png_temp = folder / "mesh.png.part"
    try:
        config = run["config"]
        segmentation_values = config["segmentation"].copy()
        if segmentation_values["crop"] is not None:
            segmentation_values["crop"] = tuple(segmentation_values["crop"])
        segment = SegmentationOptions(**segmentation_values)
        meshing = MeshingOptions(**config["meshing"])
        image = SEMImage(task["path"], pixel_size_nm=config["pixel_size_nm"])
        t = time.perf_counter()
        image.load()
        times["load_seconds"] = time.perf_counter() - t
        if image.metadata["source_sha256"] != task["sha256"]:
            raise ValueError("staged source TIFF differs from the run manifest")
        t = time.perf_counter()
        image.segment(segment)
        times["segment_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        image.create_geometry(tolerance_px=config["tolerance_px"])
        times["contour_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        result = image.generate_mesh(meshing)
        times["mesh_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        result.save_msh(mesh_temp)
        os.replace(mesh_temp, folder / "mesh.msh")
        times["msh_write_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        render_mesh_png(result.points, result.triangles, result.mask.shape,
                        config["pixel_size_nm"], png_temp)
        os.replace(png_temp, folder / "mesh.png")
        times["png_render_seconds"] = time.perf_counter() - t
        report.update(status="complete", timing=times,
                      total_seconds=time.perf_counter() - started,
                      quality=result.quality, metadata=result.metadata,
                      mesh_path=str(folder / "mesh.msh"), png_path=str(folder / "mesh.png"))
    except Exception as error:
        report.update(status="failed", timing=times,
                      total_seconds=time.perf_counter() - started,
                      error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        mesh_temp.unlink(missing_ok=True)
        png_temp.unlink(missing_ok=True)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
    return report
